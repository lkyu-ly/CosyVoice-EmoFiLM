"""Ticket 10 — low/medium/high 强度三元评测的 focused 测试。

覆盖（brief 10 §A-C / issues/10 checklist / MAP §3 评测不变量）：
- 严格单调（low<med<high 通过）；
- 平坦（单调性失败但组有效）；
- 反向（low>med>high，单调性与 emotion 保持均失败）；
- emotion 漂移（三档 emotion 预测不一致 → preservation 下降）；
- WER 退化（分档 WER 随强度升高而升高）；
- 缺失成员（整组 invalid，不进单调率分母，failure_reason 分布计数）；
- 非 EOS（整组 invalid，不进声学）；
- aggregate 确定性重算；
- failure-reason 分布；
- 每条逐样本 row 通过 ``validate_eval_row``；aggregate 通过 ``validate_aggregate``；
- 消费 08 ``FakeAcousticEvaluator``（arousal + emotion）；不加载真实模型。

CPU 合同 / 行为测试：不加载真实模型、不调用真实 ASR/MFA；合成 frame 轨迹
直接驱动核心逻辑，集成路径用 FakeAcousticEvaluator + FakeWerEvaluator。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval.acoustic_evaluators import (
    EMOTION_LABEL_SPACE,
    FakeAcousticEvaluator,
    SyntheticReferenceClip,
)
from eval.triplet_eval import (
    INTENSITY_TIERS,
    INTENSITY_TIER_TO_ID,
    METRIC_CONTRACT_VERSION,
    _check_group_membership,
    build_triplet_aggregate,
    build_triplet_member_eval_row,
    compute_arousal_score,
    compute_emotion_prediction,
    compute_wer,
    evaluate_triplet_dataset,
    evaluate_triplet_group,
    extract_span_range,
)
from tests._emofilm_fakes import (
    ClipMappedEvaluator,
    FakeWerEvaluator,
)
from tools.build_emofilm_contract import (
    validate_aggregate,
    validate_eval_row,
)


ROOT = Path(__file__).resolve().parent.parent

FRAME_RATE = 50.0
LABELS = list(EMOTION_LABEL_SPACE)  # ("ang","hap","neu","sad","sur")


# ============================================================
# helpers —— 合成帧 + manifest 构造器
# ============================================================


def _arousal_frames(
    mean_arousal: float,
    duration_sec: float = 2.0,
    frame_rate: float = FRAME_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic arousal trajectory (T,) with a given mean.

    Deterministic small perturbation around ``mean_arousal`` so the mean is
    preserved but the trajectory is not constant.
    """
    n = max(2, int(round(duration_sec * frame_rate)))
    times = np.arange(n, dtype=np.float64) / frame_rate
    perturbation = 0.02 * np.sin(np.arange(n) * 0.1)
    frames = np.clip(np.full(n, float(mean_arousal)) + perturbation, 0.0, 1.0)
    return frames, times


def _emotion_frames(
    emotion: str,
    duration_sec: float = 2.0,
    frame_rate: float = FRAME_RATE,
    prob: float = 0.82,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic (T, 5) emotion distribution dominated by ``emotion``."""
    n = max(2, int(round(duration_sec * frame_rate)))
    times = np.arange(n, dtype=np.float64) / frame_rate
    frames = np.full((n, len(LABELS)), (1.0 - prob) / (len(LABELS) - 1))
    idx = LABELS.index(emotion)
    frames[:, idx] = prob
    return frames, times


def _make_base_case(
    *,
    text: str = "the quick brown fox jumps over the lazy dog",
    emotion: str = "ang",
    speaker: str = "speaker_0011",
    prompt_ref: str = "prompt/speaker_0011",
    checkpoint_sha256: str = "c" * 64,
    source_revision: str = "abc123",
    decode_config: dict | None = None,
) -> dict[str, Any]:
    """Build the shared base-case metadata for a triplet group."""
    return {
        "text": text,
        "emotion": emotion,
        "speaker": speaker,
        "prompt_ref": prompt_ref,
        "checkpoint_sha256": checkpoint_sha256,
        "source_revision": source_revision,
        "decode_config": decode_config or {
            "min_token_text_ratio": 3.0,
            "max_token_text_ratio": 12.0,
            "max_len_hard_cap": 200,
        },
    }


def _make_control_span(
    utt_id: str,
    base_case: dict[str, Any],
    intensity_tier: str,
) -> dict[str, Any]:
    """Build a SupervisionSpan-shaped control for one triplet member.

    The control differs from its siblings ONLY in ``control_intensity_id``.
    """
    from tools.build_emofilm_contract import CONTROL_EMOTION_ID_RANGE

    emotion_to_id = {"ang": 1, "hap": 2, "neu": 3, "sad": 4, "sur": 5}
    return {
        "utt_id": utt_id,
        "label_source": "triplet_intensity_sweep",
        "supervision_granularity": "utterance",
        "start_sec": 0.0,
        "end_sec": 2.0,
        "control_emotion_id": emotion_to_id[base_case["emotion"]],
        "control_intensity_id": INTENSITY_TIER_TO_ID[intensity_tier],
        "calibrated": False,
        "emotion_mask": False,
        "intensity_mask": False,
        "supervision_weight": 1.0,
        "provenance": (
            f"triplet-intensity-sweep/emotion={base_case['emotion']}/"
            f"intensity={intensity_tier}"
        ),
        "intensity_policy": f"fixed_{intensity_tier}",
        "text": base_case["text"],
        "emotion": base_case["emotion"],
        "speaker": base_case["speaker"],
    }


def _make_gen_row(
    utt_id: str,
    base_case: dict[str, Any],
    intensity_tier: str,
    *,
    finish_reason: str = "eos",
    checkpoint_sha256: str | None = None,
    decode_config: dict | None = None,
    source_revision: str | None = None,
    seed: int = 1986,
) -> dict[str, Any]:
    """Build a v2 GenerationRow for one triplet member."""
    if finish_reason == "eos":
        row: dict[str, Any] = {"wav_path": f"wav/{utt_id}.wav"}
    else:
        row = {}
    row.update({
        "utt_id": utt_id,
        "finish_reason": finish_reason,
        "source_revision": source_revision or base_case["source_revision"],
        "checkpoint_sha256": checkpoint_sha256 or base_case["checkpoint_sha256"],
        "control_row_ref": f"control/{utt_id}",
        "prompt_row_ref": base_case["prompt_ref"],
        "decode_config": decode_config or dict(base_case["decode_config"]),
        # 非合同元数据，供 triplet orchestrator 消费。
        "intensity_tier": intensity_tier,
        "group_id": f"{base_case['emotion']}_base",
        # seed（T4 加入 schema §2；validate_generation_row 强制非负 int）.
        "seed": seed,
    })
    return row


# ============================================================
# 1. Pure helpers
# ============================================================


class TestComputeArousalScore:
    def test_mean_of_trajectory(self):
        frames, _ = _arousal_frames(mean_arousal=0.6, duration_sec=2.0)
        out = {"frames": frames}
        score = compute_arousal_score(out)
        assert score == pytest.approx(0.6, abs=0.01)

    def test_empty_returns_none_not_zero(self):
        # Task #11: 空轨迹 → None（无效样本，不污染分母）。
        assert compute_arousal_score({"frames": np.array([], dtype=np.float64)}) is None

    def test_all_nan_returns_none(self):
        # Task #11: 全 NaN 轨迹 → None（NaN 不进 mean）。
        frames = np.full(8, np.nan)
        assert compute_arousal_score({"frames": frames}) is None

    def test_missing_frames_key_returns_none(self):
        # 无 frames key → None。
        assert compute_arousal_score({}) is None


class TestComputeEmotionPrediction:
    def test_argmax_of_frame_mean(self):
        frames, _ = _emotion_frames("sad", duration_sec=2.0)
        out = {"frames": frames, "label_space": LABELS}
        assert compute_emotion_prediction(out, LABELS) == "sad"

    def test_uniform_distribution_returns_first_label(self):
        n = 10
        frames = np.full((n, len(LABELS)), 1.0 / len(LABELS))
        out = {"frames": frames, "label_space": LABELS}
        # 均匀分布 → argmax 平局取第一个 label。
        assert compute_emotion_prediction(out, LABELS) == LABELS[0]

    def test_empty_returns_none_not_first_label(self):
        # Task #11: 空分布 → None（不再返回 label_space[0]）。
        assert compute_emotion_prediction(
            {"frames": np.array([], dtype=np.float64)}, ["ang", "hap"]
        ) is None

    def test_single_column_returns_none(self):
        # Task #11: 1D 单列（无标签维度）→ None。
        frames = np.array([0.4, 0.5, 0.6])
        assert compute_emotion_prediction({"frames": frames}, ["ang", "hap"]) is None

    def test_all_nan_returns_none(self):
        # Task #11: 全 NaN 分布 → None。
        frames = np.full((4, 5), np.nan)
        assert compute_emotion_prediction({"frames": frames}, LABELS) is None


class TestComputeWer:
    """compute_wer 的同形参数化用例（原先 6 个微测试合并为 1 个参数化测试）。

    每个 case 通过 ``expected`` dict 指定要断言的字段（覆盖等价：
    perfect case 断 4 字段；substitution/deletion/insertion 各断其操作计数
    + wer；两个空输入 case 只断 wer）。原 6 个独立微测试的断言完整保留。
    """

    @pytest.mark.parametrize(
        "ref,hyp,expected",
        [
            pytest.param(
                "hello world", "hello world",
                {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0},
                id="perfect_transcription_zero_wer",
            ),
            pytest.param(
                "hello world", "hello there",
                {"wer": 0.5, "substitutions": 1},
                id="substitution",
            ),
            pytest.param(
                "hello world there", "hello world",
                {"wer": 1.0 / 3.0, "deletions": 1},
                id="deletion",
            ),
            pytest.param(
                "hello world", "hello brave new world",
                {"wer": 1.0, "insertions": 2},
                id="insertion",
            ),
            pytest.param(
                "", "",
                {"wer": 0.0},
                id="empty_reference_and_hypothesis",
            ),
            pytest.param(
                "", "hello",
                {"wer": 1.0},
                id="empty_reference_nonempty_hypothesis",
            ),
        ],
    )
    def test_compute_wer(self, ref, hyp, expected):
        result = compute_wer(ref, hyp)
        for key, value in expected.items():
            if key == "wer":
                assert result[key] == pytest.approx(value)
            else:
                assert result[key] == value


class TestExtractSpanRange:
    def test_start_end_from_times(self):
        times = np.array([0.0, 0.02, 0.04, 0.06])
        span = extract_span_range(times)
        assert span["start_sec"] == pytest.approx(0.0)
        assert span["end_sec"] == pytest.approx(0.06)

    def test_empty_times(self):
        span = extract_span_range(np.array([], dtype=np.float64))
        assert span["start_sec"] == pytest.approx(0.0)
        assert span["end_sec"] == pytest.approx(0.0)


# ============================================================
# 2. Per-member eval row
# ============================================================


class TestBuildTripletMemberEvalRow:
    def _make_inputs(self, utt_id="t_low", intensity_tier="low",
                     arousal_mean=0.2, emotion="ang",
                     hypothesis="the quick brown fox jumps over the lazy dog"):
        base = _make_base_case()
        ctrl = _make_control_span(utt_id, base, intensity_tier)
        gen = _make_gen_row(utt_id, base, intensity_tier)
        a_frames, a_times = _arousal_frames(arousal_mean)
        e_frames, e_times = _emotion_frames(emotion)
        arousal_out = {
            "frames": a_frames, "times_sec": a_times,
            "frame_rate_hz": FRAME_RATE,
        }
        emotion_out = {
            "frames": e_frames, "times_sec": e_times,
            "frame_rate_hz": FRAME_RATE, "label_space": LABELS,
        }
        return base, ctrl, gen, arousal_out, emotion_out, hypothesis

    def test_row_passes_validate_eval_row(self):
        base, ctrl, gen, arousal_out, emotion_out, hyp = self._make_inputs()
        fake_a = FakeAcousticEvaluator(kind="arousal")
        fake_e = FakeAcousticEvaluator(kind="emotion")
        fake_w = FakeWerEvaluator()
        row = build_triplet_member_eval_row(
            "t_low", base, "low", gen, ctrl,
            arousal_out, emotion_out, hyp,
            fake_a.identity(), fake_e.identity(), fake_w.identity(),
        )
        validated = validate_eval_row(row)
        assert validated["utt_id"] == "t_low"
        assert validated["boundary_evidence_tier"] == "exact"
        assert validated["evaluator"]["name"]
        assert validated["evaluator"]["version"]

    def test_row_contains_required_metrics(self):
        base, ctrl, gen, arousal_out, emotion_out, hyp = self._make_inputs()
        fake_a = FakeAcousticEvaluator(kind="arousal")
        fake_e = FakeAcousticEvaluator(kind="emotion")
        fake_w = FakeWerEvaluator()
        row = build_triplet_member_eval_row(
            "t_low", base, "low", gen, ctrl,
            arousal_out, emotion_out, hyp,
            fake_a.identity(), fake_e.identity(), fake_w.identity(),
        )
        m = row["metrics"]
        assert "arousal_score" in m
        assert "emotion_prediction" in m
        assert "wer" in m
        assert "span_range" in m
        assert m["arousal_score"] == pytest.approx(0.2, abs=0.01)
        assert m["emotion_prediction"] == "ang"
        assert m["wer"] == pytest.approx(0.0)
        assert m["span_range"]["start_sec"] == pytest.approx(0.0)

    def test_row_records_generation_ref_and_control_span(self):
        base, ctrl, gen, arousal_out, emotion_out, hyp = self._make_inputs()
        fake_a = FakeAcousticEvaluator(kind="arousal")
        fake_e = FakeAcousticEvaluator(kind="emotion")
        fake_w = FakeWerEvaluator()
        row = build_triplet_member_eval_row(
            "t_low", base, "low", gen, ctrl,
            arousal_out, emotion_out, hyp,
            fake_a.identity(), fake_e.identity(), fake_w.identity(),
        )
        # generation_row 和 control_span 内嵌（合同允许 dict）。
        assert row["generation_row"]["utt_id"] == "t_low"
        assert row["control_span"]["control_intensity_id"] == 1


# ============================================================
# 3. Per-group metrics
# ============================================================


def _build_group_inputs(
    group_id: str,
    *,
    arousal_by_tier: dict[str, float] | None = None,
    emotion_by_tier: dict[str, float] | None = None,
    hypothesis_by_tier: dict[str, str] | None = None,
    base: dict[str, Any] | None = None,
):
    """Construct (group_id, base_case, member_rows) for one group.

    Returns a 3-tuple so callers can spread directly into
    ``evaluate_triplet_group(*_build_group_inputs("g"))``.

    Default: strict-monotonic arousal (low=0.2, med=0.5, high=0.8),
    consistent emotion, perfect transcription.
    """
    base = base or _make_base_case()
    arousal_by_tier = arousal_by_tier or {"low": 0.2, "medium": 0.5, "high": 0.8}
    emotion_by_tier = emotion_by_tier or {
        "low": base["emotion"], "medium": base["emotion"], "high": base["emotion"],
    }
    hypothesis_by_tier = hypothesis_by_tier or {
        tier: base["text"] for tier in INTENSITY_TIERS
    }
    fake_a = FakeAcousticEvaluator(kind="arousal")
    fake_e = FakeAcousticEvaluator(kind="emotion")
    fake_w = FakeWerEvaluator()
    member_rows: dict[str, dict[str, Any]] = {}
    for tier in INTENSITY_TIERS:
        utt_id = f"{group_id}_{tier}"
        ctrl = _make_control_span(utt_id, base, tier)
        gen = _make_gen_row(utt_id, base, tier)
        a_frames, a_times = _arousal_frames(arousal_by_tier[tier])
        e_frames, e_times = _emotion_frames(emotion_by_tier[tier])
        arousal_out = {
            "frames": a_frames, "times_sec": a_times,
            "frame_rate_hz": FRAME_RATE,
        }
        emotion_out = {
            "frames": e_frames, "times_sec": e_times,
            "frame_rate_hz": FRAME_RATE, "label_space": LABELS,
        }
        row = build_triplet_member_eval_row(
            utt_id, base, tier, gen, ctrl,
            arousal_out, emotion_out, hypothesis_by_tier[tier],
            fake_a.identity(), fake_e.identity(), fake_w.identity(),
        )
        member_rows[tier] = row
    return group_id, base, member_rows


class TestEvaluateTripletGroup:
    def test_strict_monotonic_passes(self):
        _, base, members = _build_group_inputs("g1")
        group = evaluate_triplet_group("g1", base, members)
        assert group["valid"] is True
        assert group["failure_reason"] is None
        assert group["metrics"]["arousal_monotonic"] is True
        assert group["metrics"]["arousal_strict_monotonic"] is True
        assert group["metrics"]["arousal_span_gap"] == pytest.approx(0.6, abs=0.02)
        assert group["metrics"]["arousal_low"] < group["metrics"]["arousal_medium"]
        assert group["metrics"]["arousal_medium"] < group["metrics"]["arousal_high"]

    def test_flat_not_monotonic_but_valid(self):
        _, base, members = _build_group_inputs(
            "g2", arousal_by_tier={"low": 0.5, "medium": 0.5, "high": 0.5},
        )
        group = evaluate_triplet_group("g2", base, members)
        assert group["valid"] is True
        assert group["metrics"]["arousal_monotonic"] is False
        assert group["metrics"]["arousal_strict_monotonic"] is False
        assert group["metrics"]["arousal_span_gap"] == pytest.approx(0.0, abs=0.02)

    def test_reverse_monotonic_fails(self):
        _, base, members = _build_group_inputs(
            "g3", arousal_by_tier={"low": 0.8, "medium": 0.5, "high": 0.2},
        )
        group = evaluate_triplet_group("g3", base, members)
        assert group["valid"] is True
        assert group["metrics"]["arousal_monotonic"] is False
        assert group["metrics"]["arousal_span_gap"] < 0

    def test_emotion_drift_breaks_preservation(self):
        _, base, members = _build_group_inputs(
            "g4",
            emotion_by_tier={"low": "ang", "medium": "hap", "high": "sad"},
        )
        group = evaluate_triplet_group("g4", base, members)
        assert group["valid"] is True
        assert group["metrics"]["emotion_preserved"] is False
        assert set(group["metrics"]["emotion_predictions"].values()) == {
            "ang", "hap", "sad",
        }

    def test_emotion_preserved_when_all_match(self):
        _, base, members = _build_group_inputs("g5")
        group = evaluate_triplet_group("g5", base, members)
        assert group["metrics"]["emotion_preserved"] is True

    def test_wer_degrade_detected(self):
        base = _make_base_case()
        # high 档有替换 → WER > 0；low/medium 完美。
        hyp_by_tier = {
            "low": base["text"],
            "medium": base["text"],
            "high": base["text"].replace("lazy", "crazy"),
        }
        _, _, members = _build_group_inputs("g6", hypothesis_by_tier=hyp_by_tier,
                                          base=base)
        group = evaluate_triplet_group("g6", base, members)
        m = group["metrics"]
        assert m["wer_low"] == pytest.approx(0.0)
        assert m["wer_medium"] == pytest.approx(0.0)
        assert m["wer_high"] > 0

    def test_per_tier_wer_all_recorded(self):
        _, base, members = _build_group_inputs("g7")
        group = evaluate_triplet_group("g7", base, members)
        m = group["metrics"]
        for tier in INTENSITY_TIERS:
            assert f"wer_{tier}" in m

    def test_group_metadata_records_shared_context(self):
        _, base, members = _build_group_inputs("g8")
        group = evaluate_triplet_group("g8", base, members)
        assert group["group_id"] == "g8"
        assert group["base_case"]["text"] == base["text"]
        assert group["base_case"]["emotion"] == base["emotion"]
        assert group["base_case"]["checkpoint_sha256"] == base["checkpoint_sha256"]
        # seed_policy 已从 fixture 删除（生产代码不消费该字段；
        # remediation T10 删除了 §2 的 seed_policy 比对，eval/triplet_eval.py:432
        # 注释说明 "从无 seed_policy 字段"）。原先的 echo assert 恒真，已删。
        # 三个成员引用均存在。
        for tier in INTENSITY_TIERS:
            assert tier in group["members"]

    def test_evaluator_identity_embedded(self):
        _, base, members = _build_group_inputs("g9")
        group = evaluate_triplet_group("g9", base, members)
        assert group["evaluator"]["name"]
        assert group["evaluator"]["version"]

    def test_missing_member_invalid(self):
        _, base, members = _build_group_inputs("g10")
        del members["medium"]
        group = evaluate_triplet_group("g10", base, members)
        assert group["valid"] is False
        assert "missing" in group["failure_reason"] or "medium" in group["failure_reason"]

    def test_duplicate_intensity_invalid(self):
        _, base, members = _build_group_inputs("g11")
        # 用第二个 low 替换 medium。
        members["medium"] = dict(members["low"])
        members["medium"]["utt_id"] = members["low"]["utt_id"]
        group = evaluate_triplet_group("g11", base, members)
        assert group["valid"] is False
        assert group["failure_reason"]

    def test_checkpoint_mismatch_invalid(self):
        _, base, members = _build_group_inputs("g12")
        members["high"]["generation_row"]["checkpoint_sha256"] = "d" * 64
        group = evaluate_triplet_group("g12", base, members)
        assert group["valid"] is False
        assert "checkpoint" in group["failure_reason"]

    def test_decode_config_mismatch_invalid(self):
        _, base, members = _build_group_inputs("g13")
        members["high"]["generation_row"]["decode_config"] = {
            "min_token_text_ratio": 5.0,
            "max_token_text_ratio": 15.0,
            "max_len_hard_cap": 250,
        }
        group = evaluate_triplet_group("g13", base, members)
        assert group["valid"] is False
        assert "decode_config" in group["failure_reason"]

    def test_text_mismatch_invalid(self):
        _, base, members = _build_group_inputs("g14")
        members["high"]["control_span"]["text"] = "completely different text"
        group = evaluate_triplet_group("g14", base, members)
        assert group["valid"] is False
        assert "text" in group["failure_reason"].lower()

    def test_non_eos_member_invalid(self):
        _, base, members = _build_group_inputs("g15")
        members["high"]["generation_row"]["finish_reason"] = "max_len_reached"
        # 非 eos 行不得带 wav_path（合同要求）。
        members["high"]["generation_row"].pop("wav_path", None)
        group = evaluate_triplet_group("g15", base, members)
        assert group["valid"] is False
        assert "eos" in group["failure_reason"].lower() or "finish_reason" in group["failure_reason"].lower()

    def test_intensity_id_must_match_tier(self):
        _, base, members = _build_group_inputs("g16")
        # 交换 low 和 high 的 control_intensity_id —— 现在 low 档携带 id 3。
        members["low"]["control_span"]["control_intensity_id"] = 3
        group = evaluate_triplet_group("g16", base, members)
        assert group["valid"] is False
        assert "intensity" in group["failure_reason"].lower()

    def test_prompt_row_ref_match_valid(self):
        # 三档 prompt_row_ref 相同 → 组仍 valid（默认 _build_group_inputs 即此场景）.
        # 显式核验三档共享同一条 prompt，确认 ticket 08 修复后该不变量被守卫。
        _, base, members = _build_group_inputs("g17")
        refs_before = {
            members[t]["generation_row"]["prompt_row_ref"]
            for t in INTENSITY_TIERS
        }
        assert len(refs_before) == 1  # 前置：构造确实三档同 prompt_row_ref
        group = evaluate_triplet_group("g17", base, members)
        assert group["valid"] is True
        assert group["failure_reason"] is None

    def test_prompt_row_ref_mismatch_invalid(self):
        # high 档换不同 prompt_row_ref → 三档 prompt 不一致 → 组 invalid
        # （否则 prompt 差异会被误归因于 intensity，见 ticket 08 / 核查 #6）.
        _, base, members = _build_group_inputs("g18")
        members["high"]["generation_row"]["prompt_row_ref"] = "prompt/speaker_0099"
        group = evaluate_triplet_group("g18", base, members)
        assert group["valid"] is False
        assert "prompt_row_ref" in group["failure_reason"]
        assert "mismatch" in group["failure_reason"]


# ============================================================
# 4. Aggregate
# ============================================================


class TestBuildTripletAggregate:
    def test_passes_validate_aggregate(self):
        _, base, members = _build_group_inputs("a1")
        group = evaluate_triplet_group("a1", base, members)
        agg = build_triplet_aggregate([group])
        validate_aggregate(agg)

    def test_valid_invalid_counts(self):
        g_valid = evaluate_triplet_group(*_build_group_inputs("v1"))
        g_invalid = evaluate_triplet_group(*_build_group_inputs("v2"))
        g_invalid["valid"] = False
        g_invalid["failure_reason"] = "missing_member"
        agg = build_triplet_aggregate([g_valid, g_invalid])
        assert agg["metrics"]["n_valid_groups"] == 1
        assert agg["metrics"]["n_invalid_groups"] == 1
        assert agg["n_samples"] == 2  # n_samples at aggregate level = n_groups

    def test_monotonicity_rate(self):
        # 2 valid groups, 1 monotonic, 1 flat.
        g_mono = evaluate_triplet_group(*_build_group_inputs("m1"))
        g_flat = evaluate_triplet_group(*_build_group_inputs(
            "m2", arousal_by_tier={"low": 0.5, "medium": 0.5, "high": 0.5},
        ))
        agg = build_triplet_aggregate([g_mono, g_flat])
        assert agg["metrics"]["monotonicity_rate"] == pytest.approx(0.5)
        assert agg["metrics"]["strict_monotonicity_rate"] == pytest.approx(0.5)

    def test_span_stats(self):
        g = evaluate_triplet_group(*_build_group_inputs("s1"))
        agg = build_triplet_aggregate([g])
        assert "mean_span_gap" in agg["metrics"]
        assert "min_span_gap" in agg["metrics"]
        assert "max_span_gap" in agg["metrics"]

    def test_emotion_preservation_rate(self):
        g_keep = evaluate_triplet_group(*_build_group_inputs("e1"))
        g_drift = evaluate_triplet_group(*_build_group_inputs(
            "e2", emotion_by_tier={"low": "ang", "medium": "hap", "high": "sad"},
        ))
        agg = build_triplet_aggregate([g_keep, g_drift])
        assert agg["metrics"]["emotion_preservation_rate"] == pytest.approx(0.5)

    def test_per_tier_wer(self):
        g = evaluate_triplet_group(*_build_group_inputs("w1"))
        agg = build_triplet_aggregate([g])
        for tier in INTENSITY_TIERS:
            assert f"mean_wer_{tier}" in agg["metrics"]

    def test_failure_reason_distribution(self):
        g1 = evaluate_triplet_group(*_build_group_inputs("f1"))
        g1["valid"] = False
        g1["failure_reason"] = "missing_member"
        g2 = evaluate_triplet_group(*_build_group_inputs("f2"))
        g2["valid"] = False
        g2["failure_reason"] = "missing_member"
        g3 = evaluate_triplet_group(*_build_group_inputs("f3"))
        g3["valid"] = False
        g3["failure_reason"] = "non_eos"
        agg = build_triplet_aggregate([g1, g2, g3])
        dist = agg["metrics"]["failure_reason_distribution"]
        assert dist["missing_member"] == 2
        assert dist["non_eos"] == 1

    def test_determinism_recompute(self):
        groups = [
            evaluate_triplet_group(*_build_group_inputs(f"d{i}"))
            for i in range(3)
        ]
        agg1 = build_triplet_aggregate(groups)
        agg2 = build_triplet_aggregate(groups)
        # JSON 等价。
        assert json.dumps(agg1, sort_keys=True, default=str) == \
               json.dumps(agg2, sort_keys=True, default=str)

    def test_empty_groups(self):
        agg = build_triplet_aggregate([])
        validate_aggregate(agg)
        assert agg["n_samples"] == 0
        assert agg["metrics"]["n_valid_groups"] == 0
        assert agg["metrics"]["n_invalid_groups"] == 0
        assert agg["metrics"]["monotonicity_rate"] == 0.0

    def test_invalid_groups_excluded_from_monotonicity_denominator(self):
        # 1 valid monotonic + 1 invalid → monotonicity rate over valid only = 1.0.
        g_valid = evaluate_triplet_group(*_build_group_inputs("im1"))
        g_invalid = evaluate_triplet_group(*_build_group_inputs("im2"))
        g_invalid["valid"] = False
        g_invalid["failure_reason"] = "missing_member"
        agg = build_triplet_aggregate([g_valid, g_invalid])
        # 分母 = n_valid_groups = 1。
        assert agg["metrics"]["monotonicity_rate"] == pytest.approx(1.0)

    def test_metric_contract_version(self):
        g = evaluate_triplet_group(*_build_group_inputs("cv1"))
        agg = build_triplet_aggregate([g])
        assert agg["metric_contract_version"] == METRIC_CONTRACT_VERSION

    def test_evidence_tier_is_exact(self):
        # Triplet control span = 整句（不做 FEDD 式近似）。
        g = evaluate_triplet_group(*_build_group_inputs("et1"))
        agg = build_triplet_aggregate([g])
        assert agg["evidence_tier"] == "exact"


# ============================================================
# 5. Full pipeline (evaluate_triplet_dataset)
# ============================================================


class TestEvaluateTripletDataset:
    def _make_dataset(
        self,
        group_id: str,
        base: dict[str, Any] | None = None,
        *,
        arousal_by_tier: dict[str, float] | None = None,
        emotion_by_tier: dict[str, float] | None = None,
        hypothesis_by_tier: dict[str, str] | None = None,
        finish_reasons: dict[str, str] | None = None,
    ):
        """Build (triplet_specs, generation_rows, clip_map) for one group."""
        base = base or _make_base_case()
        arousal_by_tier = arousal_by_tier or {"low": 0.2, "medium": 0.5, "high": 0.8}
        emotion_by_tier = emotion_by_tier or {
            "low": base["emotion"], "medium": base["emotion"], "high": base["emotion"],
        }
        hypothesis_by_tier = hypothesis_by_tier or {
            tier: base["text"] for tier in INTENSITY_TIERS
        }
        finish_reasons = finish_reasons or {}

        # Triplet 规格：group_id + base_case + control spans。
        ctrl_spans = {
            tier: _make_control_span(f"{group_id}_{tier}", base, tier)
            for tier in INTENSITY_TIERS
        }
        spec = {"group_id": group_id, "base_case": base, "control_spans": ctrl_spans}

        # 构造 generation rows。
        gen_rows = []
        clip_map: dict[str, SyntheticReferenceClip] = {}
        wer_hypotheses: dict[str, str] = {}
        for tier in INTENSITY_TIERS:
            utt_id = f"{group_id}_{tier}"
            fr = finish_reasons.get(tier, "eos")
            gen = _make_gen_row(utt_id, base, tier, finish_reason=fr)
            gen_rows.append(gen)
            if fr == "eos":
                # 已知 arousal rank 来自均值：0.2 → 0, 0.5 → 1, 0.8 → 2。
                rank = {0.2: 0, 0.5: 1, 0.8: 2}.get(arousal_by_tier[tier])
                # 近似：在 {0,1,2} 中取最接近的。
                if rank is None:
                    rank = min([0, 1, 2], key=lambda r: abs((0.2 + 0.3 * r) - arousal_by_tier[tier]))
                clip_map[utt_id] = SyntheticReferenceClip(
                    wav_path=gen["wav_path"], duration_sec=2.0,
                    known_arousal_rank=rank,
                    known_emotion=emotion_by_tier[tier],
                )
                wer_hypotheses[utt_id] = hypothesis_by_tier[tier]
        return spec, gen_rows, clip_map, wer_hypotheses

    def test_strict_monotonic_pipeline(self):
        spec, gen_rows, clip_map, wer_hyps = self._make_dataset("p1")
        arousal_eval = FakeAcousticEvaluator(kind="arousal")
        emotion_eval = FakeAcousticEvaluator(kind="emotion")
        wer_eval = FakeWerEvaluator(hypotheses=wer_hyps)
        mapped_a = ClipMappedEvaluator(arousal_eval, clip_map)
        mapped_e = ClipMappedEvaluator(emotion_eval, clip_map)
        result = evaluate_triplet_dataset(
            [spec], gen_rows,
            arousal_eval=mapped_a, emotion_eval=mapped_e, wer_eval=wer_eval,
        )
        assert len(result["group_rows"]) == 1
        g = result["group_rows"][0]
        assert g["valid"] is True
        assert g["metrics"]["arousal_monotonic"] is True
        validate_aggregate(result["aggregate"])

    def test_three_groups_mixed_validity(self):
        spec1, gen1, clip1, wer1 = self._make_dataset("mix1")  # monotonic
        spec2, gen2, clip2, wer2 = self._make_dataset(
            "mix2", arousal_by_tier={"low": 0.8, "medium": 0.5, "high": 0.2},
        )  # reverse
        spec3, gen3, clip3, wer3 = self._make_dataset(
            "mix3", finish_reasons={"high": "max_len_reached"},
        )  # non-eos high → invalid
        clip_map = {**clip1, **clip2, **clip3}
        wer_hyps = {**wer1, **wer2, **wer3}
        mapped_a = ClipMappedEvaluator(FakeAcousticEvaluator(kind="arousal"), clip_map)
        mapped_e = ClipMappedEvaluator(FakeAcousticEvaluator(kind="emotion"), clip_map)
        result = evaluate_triplet_dataset(
            [spec1, spec2, spec3], gen1 + gen2 + gen3,
            arousal_eval=mapped_a, emotion_eval=mapped_e,
            wer_eval=FakeWerEvaluator(hypotheses=wer_hyps),
        )
        groups = {g["group_id"]: g for g in result["group_rows"]}
        assert groups["mix1"]["valid"] is True
        assert groups["mix1"]["metrics"]["arousal_monotonic"] is True
        assert groups["mix2"]["valid"] is True
        assert groups["mix2"]["metrics"]["arousal_monotonic"] is False
        assert groups["mix3"]["valid"] is False
        # 无效组从单调性分母中排除。
        agg = result["aggregate"]
        assert agg["metrics"]["n_valid_groups"] == 2
        assert agg["metrics"]["n_invalid_groups"] == 1
        assert agg["metrics"]["monotonicity_rate"] == pytest.approx(0.5)

    def test_missing_member_in_generation(self):
        """One intensity tier missing from generation manifest → group invalid."""
        spec, gen_rows, clip_map, wer_hyps = self._make_dataset("mm1")
        # 丢弃 high generation row。
        gen_rows = [g for g in gen_rows if not g["utt_id"].endswith("_high")]
        mapped_a = ClipMappedEvaluator(FakeAcousticEvaluator(kind="arousal"), clip_map)
        mapped_e = ClipMappedEvaluator(FakeAcousticEvaluator(kind="emotion"), clip_map)
        result = evaluate_triplet_dataset(
            [spec], gen_rows,
            arousal_eval=mapped_a, emotion_eval=mapped_e,
            wer_eval=FakeWerEvaluator(hypotheses=wer_hyps),
        )
        g = result["group_rows"][0]
        assert g["valid"] is False
        assert "missing" in g["failure_reason"].lower() or "member" in g["failure_reason"].lower()

    def test_per_sample_rows_all_pass_validate_eval_row(self):
        spec, gen_rows, clip_map, wer_hyps = self._make_dataset("ps1")
        mapped_a = ClipMappedEvaluator(FakeAcousticEvaluator(kind="arousal"), clip_map)
        mapped_e = ClipMappedEvaluator(FakeAcousticEvaluator(kind="emotion"), clip_map)
        result = evaluate_triplet_dataset(
            [spec], gen_rows,
            arousal_eval=mapped_a, emotion_eval=mapped_e,
            wer_eval=FakeWerEvaluator(hypotheses=wer_hyps),
        )
        for row in result["member_rows"]:
            validate_eval_row(row)

    def test_self_evidence_risk_inherited(self):
        """Aggregate records IEMOCAP weak-arousal self-evidence risk (08)."""
        spec, gen_rows, clip_map, wer_hyps = self._make_dataset("se1")
        mapped_a = ClipMappedEvaluator(FakeAcousticEvaluator(kind="arousal"), clip_map)
        mapped_e = ClipMappedEvaluator(FakeAcousticEvaluator(kind="emotion"), clip_map)
        result = evaluate_triplet_dataset(
            [spec], gen_rows,
            arousal_eval=mapped_a, emotion_eval=mapped_e,
            wer_eval=FakeWerEvaluator(hypotheses=wer_hyps),
        )
        agg = result["aggregate"]
        # intensity conclusion = 独立 evaluator 下的相对控制响应；
        # IEMOCAP 弱 arousal 不是词级真值。
        assert "intensity_conclusion" in agg["metrics"]
        assert "self_evidence_risk" in agg["metrics"]["intensity_conclusion"]

    def test_unknown_utt_in_generation_skipped_from_other_groups(self):
        """Extra generation rows not matching any group are ignored (not hard-fail)."""
        spec, gen_rows, clip_map, wer_hyps = self._make_dataset("ig1")
        gen_rows.append(_make_gen_row("stray_orphan", _make_base_case(), "low"))
        mapped_a = ClipMappedEvaluator(FakeAcousticEvaluator(kind="arousal"), clip_map)
        mapped_e = ClipMappedEvaluator(FakeAcousticEvaluator(kind="emotion"), clip_map)
        result = evaluate_triplet_dataset(
            [spec], gen_rows,
            arousal_eval=mapped_a, emotion_eval=mapped_e,
            wer_eval=FakeWerEvaluator(hypotheses=wer_hyps),
        )
        # 组仍正常评测；游离行被忽略。
        assert len(result["group_rows"]) == 1
        assert result["group_rows"][0]["valid"] is True


# ============================================================
# 4. Ticket #10: 三档 seed 一致 + 身份不可全缺
# ============================================================


def _membership_member_rows(
    *,
    seed_by_tier: dict[str, int] | None = None,
    checkpoint_sha256: str | None = "a" * 64,
    source_revision: str | None = "9c6d84b",
) -> dict[str, dict[str, Any]]:
    """Build minimal member_rows for _check_group_membership tests.

    全档共享一致 control_span（含 control_intensity_id 匹配 tier），避免触发
    其他与 #10 无关的 mismatch；checkpoint/source/seed 可被测试覆写。
    """
    seed_by_tier = seed_by_tier or {t: 1986 for t in INTENSITY_TIERS}
    rows: dict[str, dict[str, Any]] = {}
    for tier in INTENSITY_TIERS:
        rows[tier] = {
            "generation_row": {
                "utt_id": f"g10_{tier}",
                "finish_reason": "eos",
                "wav_path": f"wav/g10_{tier}.wav",
                "checkpoint_sha256": checkpoint_sha256,
                "source_revision": source_revision,
                "prompt_row_ref": "prompt/speaker_0011",
                "control_row_ref": f"control/g10_{tier}",
                "decode_config": {"max_len_hard_cap": 200},
                "seed": seed_by_tier[tier],
            },
            "control_span": {
                "utt_id": f"g10_{tier}",
                "text": "the quick brown fox",
                "emotion": "ang",
                "speaker": "speaker_0011",
                "control_intensity_id": INTENSITY_TIER_TO_ID[tier],
            },
            "metrics": {"arousal_score": 0.5, "emotion_prediction": "ang",
                        "wer": 0.0, "span_range": {"start_sec": 0.0, "end_sec": 1.0}},
        }
    return rows


class TestGroupMembershipSeedAndIdentity:
    """核验 #10：三档 seed 必须一致 + 身份族不可全缺。"""

    def test_different_seed_invalid(self):
        """三档 seed 不同 → invalid（当前不比 seed，恒通过）。"""
        member_rows = _membership_member_rows(
            seed_by_tier={"low": 1, "medium": 2, "high": 3},
        )
        ok, msg = _check_group_membership("g10_seed", _make_base_case(), member_rows)
        assert ok is False
        assert "seed" in msg

    def test_all_missing_identity_invalid(self):
        """checkpoint_sha256 + source_revision 三档全 None → invalid（missing_identity）。

        防御性 double-check：入口 ``validate_generation_row`` 保证身份族≥1，
        若 row 绕过入口直接进 _check_group_membership，此处兜底拒绝。
        """
        member_rows = _membership_member_rows(
            checkpoint_sha256=None, source_revision=None,
        )
        ok, msg = _check_group_membership("g10_ident", _make_base_case(), member_rows)
        assert ok is False
        assert "identity" in msg

    def test_consistent_seed_and_identity_passes(self):
        """三档 seed 一致 + 身份齐 → 通过（绿色回归）。"""
        member_rows = _membership_member_rows()  # 默认 seed=1986 / cp+src 齐全
        ok, msg = _check_group_membership("g10_ok", _make_base_case(), member_rows)
        assert ok is True
        assert msg is None


# ============================================================
# 5. Ticket #11: 空 / NaN evaluator 输出标记 invalid，不污染分母
# ============================================================


class TestMemberEvalRowValidityFlag:
    """Task #11: build_triplet_member_eval_row metrics.valid 反映输出是否可用。"""

    def _make_eval_inputs(self, arousal_out, emotion_out):
        base = _make_base_case()
        ctrl = _make_control_span("v_low", base, "low")
        gen = _make_gen_row("v_low", base, "low")
        fake_a = FakeAcousticEvaluator(kind="arousal")
        fake_e = FakeAcousticEvaluator(kind="emotion")
        fake_w = FakeWerEvaluator()
        return build_triplet_member_eval_row(
            "v_low", base, "low", gen, ctrl,
            arousal_out, emotion_out, "",
            fake_a.identity(), fake_e.identity(), fake_w.identity(),
        )

    def test_empty_arousal_marks_invalid(self):
        """空 arousal 输出 → metrics.valid=False（非占位 0.0 进统计）。"""
        arousal_out = {"frames": np.array([], dtype=np.float64),
                       "times_sec": np.array([], dtype=np.float64)}
        e_frames, e_times = _emotion_frames("ang")
        emotion_out = {"frames": e_frames, "times_sec": e_times,
                       "frame_rate_hz": FRAME_RATE, "label_space": LABELS}
        row = self._make_eval_inputs(arousal_out, emotion_out)
        assert row["metrics"].get("valid") is False
        assert row["metrics"]["arousal_score"] is None

    def test_all_nan_emotion_marks_invalid(self):
        """全 NaN emotion 分布 → metrics.valid=False。"""
        a_frames, a_times = _arousal_frames(0.4)
        arousal_out = {"frames": a_frames, "times_sec": a_times,
                       "frame_rate_hz": FRAME_RATE}
        emotion_out = {"frames": np.full((4, 5), np.nan),
                       "times_sec": np.arange(4) / 50.0,
                       "frame_rate_hz": FRAME_RATE, "label_space": LABELS}
        row = self._make_eval_inputs(arousal_out, emotion_out)
        assert row["metrics"].get("valid") is False
        assert row["metrics"]["emotion_prediction"] is None

    def test_normal_outputs_marked_valid(self):
        """正常 evaluator 输出 → metrics.valid=True（回归）。"""
        a_frames, a_times = _arousal_frames(0.4)
        e_frames, e_times = _emotion_frames("ang")
        arousal_out = {"frames": a_frames, "times_sec": a_times,
                       "frame_rate_hz": FRAME_RATE}
        emotion_out = {"frames": e_frames, "times_sec": e_times,
                       "frame_rate_hz": FRAME_RATE, "label_space": LABELS}
        row = self._make_eval_inputs(arousal_out, emotion_out)
        assert row["metrics"].get("valid") is True
        assert row["metrics"]["arousal_score"] is not None
        assert row["metrics"]["emotion_prediction"] == "ang"


class TestInvalidMemberExcludedFromAggregate:
    """Task #11: 含 invalid member 的 group → group invalid，不进 aggregate 分母。"""

    def test_group_with_invalid_member_marked_invalid(self):
        """某 member metrics.valid=False → _check_group_membership 判定 group invalid。"""
        base = _make_base_case()
        fake_a = FakeAcousticEvaluator(kind="arousal")
        fake_e = FakeAcousticEvaluator(kind="emotion")
        fake_w = FakeWerEvaluator()
        member_rows: dict[str, dict[str, Any]] = {}
        for tier in INTENSITY_TIERS:
            utt_id = f"inv_{tier}"
            ctrl = _make_control_span(utt_id, base, tier)
            gen = _make_gen_row(utt_id, base, tier)
            if tier == "high":
                # high tier 的 evaluator 输出为空（模拟异常）。
                arousal_out = {"frames": np.array([], dtype=np.float64),
                               "times_sec": np.array([], dtype=np.float64)}
                emotion_out = {"frames": np.full((1, 5), np.nan),
                               "times_sec": np.zeros(1),
                               "frame_rate_hz": FRAME_RATE,
                               "label_space": LABELS}
            else:
                a_frames, a_times = _arousal_frames(0.4)
                e_frames, e_times = _emotion_frames("ang")
                arousal_out = {"frames": a_frames, "times_sec": a_times,
                               "frame_rate_hz": FRAME_RATE}
                emotion_out = {"frames": e_frames, "times_sec": e_times,
                               "frame_rate_hz": FRAME_RATE, "label_space": LABELS}
            member_rows[tier] = build_triplet_member_eval_row(
                utt_id, base, tier, gen, ctrl,
                arousal_out, emotion_out, "",
                fake_a.identity(), fake_e.identity(), fake_w.identity(),
            )
        ok, msg = _check_group_membership("inv_g", base, member_rows)
        assert ok is False
        assert "invalid_member_output" in msg

    def test_aggregate_excludes_group_with_invalid_member(self):
        """group invalid（因 member valid=False）→ 不进 monotonicity/emotion 分母。"""
        base = _make_base_case()
        fake_a = FakeAcousticEvaluator(kind="arousal")
        fake_e = FakeAcousticEvaluator(kind="emotion")
        fake_w = FakeWerEvaluator()
        # 构造一个 valid 组（严格单调 + 一致 emotion）。
        good_members: dict[str, dict[str, Any]] = {}
        for tier in INTENSITY_TIERS:
            utt_id = f"good_{tier}"
            ctrl = _make_control_span(utt_id, base, tier)
            gen = _make_gen_row(utt_id, base, tier)
            mean = {"low": 0.2, "medium": 0.5, "high": 0.8}[tier]
            a_frames, a_times = _arousal_frames(mean)
            e_frames, e_times = _emotion_frames("ang")
            arousal_out = {"frames": a_frames, "times_sec": a_times,
                           "frame_rate_hz": FRAME_RATE}
            emotion_out = {"frames": e_frames, "times_sec": e_times,
                           "frame_rate_hz": FRAME_RATE, "label_space": LABELS}
            good_members[tier] = build_triplet_member_eval_row(
                utt_id, base, tier, gen, ctrl,
                arousal_out, emotion_out, base["text"],
                fake_a.identity(), fake_e.identity(), fake_w.identity(),
            )
        # 构造一个含 invalid member 的组。
        bad_members: dict[str, dict[str, Any]] = {}
        for tier in INTENSITY_TIERS:
            utt_id = f"bad_{tier}"
            ctrl = _make_control_span(utt_id, base, tier)
            gen = _make_gen_row(utt_id, base, tier)
            if tier == "low":
                # 空 arousal 触发 valid=False。
                arousal_out = {"frames": np.array([], dtype=np.float64),
                               "times_sec": np.array([], dtype=np.float64)}
            else:
                a_frames, a_times = _arousal_frames(0.5)
                arousal_out = {"frames": a_frames, "times_sec": a_times,
                               "frame_rate_hz": FRAME_RATE}
            e_frames, e_times = _emotion_frames("ang")
            emotion_out = {"frames": e_frames, "times_sec": e_times,
                           "frame_rate_hz": FRAME_RATE, "label_space": LABELS}
            bad_members[tier] = build_triplet_member_eval_row(
                utt_id, base, tier, gen, ctrl,
                arousal_out, emotion_out, "",
                fake_a.identity(), fake_e.identity(), fake_w.identity(),
            )
        g_good = evaluate_triplet_group("good_g", base, good_members)
        g_bad = evaluate_triplet_group("bad_g", base, bad_members)
        assert g_good["valid"] is True
        assert g_bad["valid"] is False
        agg = build_triplet_aggregate([g_good, g_bad])
        # 仅 good_g 计入分母。
        assert agg["metrics"]["n_valid_groups"] == 1
        assert agg["metrics"]["n_invalid_groups"] == 1
        assert agg["metrics"]["monotonicity_rate"] == pytest.approx(1.0)

