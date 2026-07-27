"""Ticket 09 — FEDD 局部转场逐 span 评测的 focused 测试。

覆盖（brief 09 §B / issues/09 checklist / MAP §3 评测不变量）：
- 正确转场（前 emo_from / 后 emo_to 命中 + direction correct）；
- 无转场（保持单一，direction maintained）；
- 反向转场（direction reverse）；
- 边界偏早 / 偏晚（时间误差符号与量级）；
- FEDD-A approximate 不进 exact aggregate；
- 逐样本异常传播（非 EOS / 缺失 / 重复 / 身份不一致 → utt_id hard-fail）；
- 无有效 MFA 对齐时不伪造边界误差（boundary_error null + reason）；
- aggregate 可从 rows 重算（determinism）；
- 消费 08 的 FakeAcousticEvaluator。

CPU 合同 / 行为测试：不加载真实模型；合成 frame 轨迹直接驱动核心逻辑，
集成路径用 FakeAcousticEvaluator + FakeForcedAligner（无 GPU / MFA 依赖）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval.eval_local_control import (
    APPROX_METHOD,
    EXACT_METHOD,
    METRIC_CONTRACT_VERSION,
    AlignmentResult,
    WordBoundary,
    build_aggregate_from_rows,
    build_eval_row,
    compute_boundary_time_error,
    detect_transition_from_frames,
    derive_evidence_tier,
    evaluate_fedd_dataset,
    evaluate_spans_from_frames,
    resolve_aligned_boundary_sec,
)
from tests._emofilm_fakes import (
    ClipMappedEvaluator,
    FakeForcedAligner,
)
from eval.acoustic_evaluators import (
    EMOTION_LABEL_SPACE,
    FakeAcousticEvaluator,
    SyntheticReferenceClip,
)
from tools.build_emofilm_contract import (
    BOUNDARY_EVIDENCE_TIERS,
    validate_aggregate,
    validate_eval_row,
)


ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# helpers —— 合成帧轨迹 + manifest 构造器
# ============================================================

FRAME_RATE = 50.0
LABELS = list(EMOTION_LABEL_SPACE)  # ("ang","hap","neu","sad","sur")
LABEL_IDX = {lab: i for i, lab in enumerate(LABELS)}


def _make_frames(
    duration_sec: float = 3.0,
    *,
    front_emotion: str | None = None,
    back_emotion: str | None = None,
    boundary_sec: float | None = None,
    front_prob: float = 0.82,
    back_prob: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic (T, 5) emotion distribution trajectory.

    If front/back emotion given, first half (up to boundary_sec) is dominated
    by front_emotion at ``front_prob`` and the rest by back_emotion at
    ``back_prob``.  Remaining mass distributed uniformly across other labels.
    """
    n = max(2, int(round(duration_sec * FRAME_RATE)))
    times = np.arange(n, dtype=np.float64) / FRAME_RATE
    frames = np.full((n, 5), 0.2, dtype=np.float64)

    if front_emotion is not None and back_emotion is not None:
        if boundary_sec is None:
            boundary_sec = duration_sec / 2.0
        boundary_frame = max(1, min(int(round(boundary_sec * FRAME_RATE)), n - 1))
        fi = LABEL_IDX[front_emotion]
        bi = LABEL_IDX[back_emotion]
        # front 段（边界之前）
        rest_f = (1.0 - front_prob) / 4.0
        frames[:boundary_frame, :] = rest_f
        frames[:boundary_frame, fi] = front_prob
        # back 段（边界及之后）
        rest_b = (1.0 - back_prob) / 4.0
        frames[boundary_frame:, :] = rest_b
        frames[boundary_frame:, bi] = back_prob
    elif front_emotion is not None:
        fi = LABEL_IDX[front_emotion]
        rest = (1.0 - front_prob) / 4.0
        frames[:, :] = rest
        frames[:, fi] = front_prob

    return frames, times


def _make_control_record(
    utt_id: str,
    *,
    emo_from: str = "ang",
    emo_to: str = "hap",
    boundary_word_index: int | None = 3,
    text: str = "the quick brown fox jumps over",
    part: str = "B",
    extra: dict | None = None,
) -> dict:
    """Build a FEDD control record (construction-known transition)."""
    method = EXACT_METHOD if boundary_word_index is not None else APPROX_METHOD
    rec: dict[str, Any] = {
        "utt_id": utt_id,
        "text": text,
        "emo_from": emo_from,
        "emo_to": emo_to,
        "boundary_word_index": boundary_word_index,
        "method": method,
        "part": part,
        "label_source": "construction_known_transition",
        "intensity_policy": "fixed_medium",
    }
    if extra:
        rec.update(extra)
    return rec


def _make_gen_row(
    utt_id: str,
    *,
    wav_path: str | None = None,
    finish_reason: str = "eos",
    checkpoint_sha256: str = "a" * 64,
    source_revision: str = "abc123",
    decode_config: dict | None = None,
) -> dict:
    """Build a v2 GenerationRow dict."""
    if wav_path is None:
        wav_path = f"wav/{utt_id}.wav"
    if finish_reason == "eos" and wav_path is not None:
        row: dict[str, Any] = {"wav_path": wav_path}
    else:
        row = {}
    row.update({
        "utt_id": utt_id,
        "finish_reason": finish_reason,
        "source_revision": source_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "control_row_ref": f"control/{utt_id}",
        "prompt_row_ref": "prompt/speaker_0011",
        "decode_config": decode_config or {
            "min_token_text_ratio": 3.0,
            "max_token_text_ratio": 12.0,
            "max_len_hard_cap": 200,
        },
    })
    return row


# ============================================================
# 1. evaluate_spans_from_frames — core span logic
# ============================================================


class TestEvaluateSpansFromFrames:
    """Pure-function span evaluation on synthetic frame arrays."""

    def test_correct_transition(self):
        """Front=emo_from hit, back=emo_to hit, direction=correct."""
        frames, times = _make_frames(
            duration_sec=3.0, front_emotion="ang", back_emotion="hap",
            boundary_sec=1.5,
        )
        result = evaluate_spans_from_frames(
            frames, times, boundary_sec=1.5,
            emo_from="ang", emo_to="hap", label_space=LABELS,
        )
        assert result["front"]["predicted_emotion"] == "ang"
        assert result["back"]["predicted_emotion"] == "hap"
        assert result["front"]["hit"] is True
        assert result["back"]["hit"] is True
        assert result["front_back_both_hit"] is True
        assert result["transition_direction"] == "correct"

    def test_no_transition_maintained(self):
        """Single emotion maintained across boundary → direction=maintained."""
        frames, times = _make_frames(duration_sec=3.0, front_emotion="ang")
        result = evaluate_spans_from_frames(
            frames, times, boundary_sec=1.5,
            emo_from="ang", emo_to="hap", label_space=LABELS,
        )
        assert result["front"]["predicted_emotion"] == "ang"
        assert result["back"]["predicted_emotion"] == "ang"
        assert result["transition_direction"] == "maintained"
        assert result["front_back_both_hit"] is False

    def test_reverse_transition(self):
        """Front=emo_to, back=emo_from → direction=reverse."""
        frames, times = _make_frames(
            duration_sec=3.0, front_emotion="hap", back_emotion="ang",
            boundary_sec=1.5,
        )
        result = evaluate_spans_from_frames(
            frames, times, boundary_sec=1.5,
            emo_from="ang", emo_to="hap", label_space=LABELS,
        )
        assert result["front"]["predicted_emotion"] == "hap"
        assert result["back"]["predicted_emotion"] == "ang"
        assert result["transition_direction"] == "reverse"
        assert result["front_back_both_hit"] is False

    def test_span_time_ranges(self):
        """Span time ranges are correctly recorded."""
        frames, times = _make_frames(duration_sec=4.0, front_emotion="ang",
                                     back_emotion="sad", boundary_sec=2.0)
        result = evaluate_spans_from_frames(
            frames, times, boundary_sec=2.0,
            emo_from="ang", emo_to="sad", label_space=LABELS,
        )
        assert result["front"]["start_sec"] == pytest.approx(0.0)
        assert result["front"]["end_sec"] == pytest.approx(2.0)
        assert result["back"]["start_sec"] == pytest.approx(2.0)
        assert result["back"]["end_sec"] == pytest.approx(times[-1])

    def test_front_score_is_mean_target_prob(self):
        """Front score = mean probability of target emotion in front span."""
        frames, times = _make_frames(
            duration_sec=2.0, front_emotion="neu", back_emotion="sad",
            boundary_sec=1.0, front_prob=0.70,
        )
        result = evaluate_spans_from_frames(
            frames, times, boundary_sec=1.0,
            emo_from="neu", emo_to="sad", label_space=LABELS,
        )
        assert result["front"]["score"] == pytest.approx(0.70, abs=0.01)


# ============================================================
# 2. detect_transition_from_frames — transition detection
# ============================================================


class TestDetectTransition:
    """Transition detection from per-frame argmax trajectory."""

    def test_correct_detection(self):
        frames, times = _make_frames(
            duration_sec=3.0, front_emotion="ang", back_emotion="hap",
            boundary_sec=1.5,
        )
        result = detect_transition_from_frames(
            frames, times, "ang", "hap", LABELS, FRAME_RATE,
        )
        assert result["detected"] is True
        assert result["detected_sec"] is not None
        assert abs(result["detected_sec"] - 1.5) < 0.05

    def test_no_transition_detected(self):
        frames, times = _make_frames(duration_sec=3.0, front_emotion="ang")
        result = detect_transition_from_frames(
            frames, times, "ang", "hap", LABELS, FRAME_RATE,
        )
        assert result["detected"] is False
        assert result["detected_sec"] is None


# ============================================================
# 3. compute_boundary_time_error — boundary error
# ============================================================


class TestBoundaryTimeError:
    """Boundary time error = detected - aligned."""

    def test_early_transition_negative_error(self):
        """Transition before boundary → negative error."""
        result = compute_boundary_time_error(
            detected_sec=1.0, aligned_sec=1.5,
        )
        assert result["boundary_error_sec"] == pytest.approx(-0.5)
        assert result["detected_sec"] == pytest.approx(1.0)
        assert result["aligned_sec"] == pytest.approx(1.5)

    def test_late_transition_positive_error(self):
        """Transition after boundary → positive error."""
        result = compute_boundary_time_error(
            detected_sec=2.0, aligned_sec=1.5,
        )
        assert result["boundary_error_sec"] == pytest.approx(0.5)

    def test_no_detection_null_error(self):
        """No detected transition → null error."""
        result = compute_boundary_time_error(
            detected_sec=None, aligned_sec=1.5,
        )
        assert result["boundary_error_sec"] is None
        assert "reason" in result

    def test_no_alignment_null_error(self):
        """No aligned boundary → null error."""
        result = compute_boundary_time_error(
            detected_sec=1.5, aligned_sec=None,
        )
        assert result["boundary_error_sec"] is None
        assert "reason" in result


# ============================================================
# 4. derive_evidence_tier
# ============================================================


class TestDeriveEvidenceTier:
    def test_exact_from_method(self):
        assert derive_evidence_tier(method=EXACT_METHOD) == "exact"

    def test_approx_from_method(self):
        assert derive_evidence_tier(method=APPROX_METHOD) == "approximate"

    def test_exact_from_boundary_word_index(self):
        assert derive_evidence_tier(boundary_word_index=3) == "exact"

    def test_approximate_no_boundary(self):
        assert derive_evidence_tier(boundary_word_index=None) == "approximate"


# ============================================================
# 5. resolve_aligned_boundary_sec
# ============================================================


class TestResolveAlignedBoundary:
    def test_valid_alignment_word_end(self):
        """Boundary at word_index k = end_sec of k-th word (1-indexed)."""
        words = [
            WordBoundary(0.0, 0.5, "the"),
            WordBoundary(0.5, 1.0, "quick"),
            WordBoundary(1.0, 1.5, "brown"),
            WordBoundary(1.5, 2.0, "fox"),
        ]
        result = AlignmentResult(status="aligned", words=words)
        boundary_sec, status = resolve_aligned_boundary_sec(result, 3)
        assert boundary_sec == pytest.approx(1.5)
        assert status == "aligned"

    def test_failed_alignment_no_boundary(self):
        result = AlignmentResult(status="failed", words=[], reason="low_score")
        boundary_sec, status = resolve_aligned_boundary_sec(result, 3)
        assert boundary_sec is None
        assert status == "failed"

    def test_word_index_out_of_range(self):
        """Word index beyond aligned words → no valid boundary."""
        words = [WordBoundary(0.0, 0.5, "the")]
        result = AlignmentResult(status="aligned", words=words)
        boundary_sec, status = resolve_aligned_boundary_sec(result, 3)
        assert boundary_sec is None
        assert status == "failed"


# ============================================================
# 6. build_eval_row — EvaluationRow construction
# ============================================================


class TestBuildEvalRow:
    def _frames_output(self, front_emo="ang", back_emo="hap", duration=3.0):
        frames, times = _make_frames(
            duration, front_emotion=front_emo, back_emotion=back_emo,
            boundary_sec=duration / 2,
        )
        return {
            "frames": frames, "times_sec": times,
            "frame_rate_hz": FRAME_RATE, "label_space": LABELS,
        }

    def test_row_passes_validate_eval_row(self):
        control = _make_control_record("fedd_b_001")
        gen = _make_gen_row("fedd_b_001")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        aligner = FakeForcedAligner()
        row = build_eval_row(
            "fedd_b_001", control, gen,
            evaluator.identity(), self._frames_output(),
            aligner.align(gen["wav_path"], control["text"]),
            evidence_tier="exact",
        )
        validated = validate_eval_row(row)
        assert validated["utt_id"] == "fedd_b_001"
        assert validated["boundary_evidence_tier"] == "exact"
        assert validated["evaluator"]["name"]
        assert validated["evaluator"]["version"]

    def test_row_contains_span_metrics(self):
        control = _make_control_record("fedd_b_002")
        gen = _make_gen_row("fedd_b_002")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        aligner = FakeForcedAligner()
        row = build_eval_row(
            "fedd_b_002", control, gen,
            evaluator.identity(), self._frames_output("ang", "hap"),
            aligner.align(gen["wav_path"], control["text"]),
            evidence_tier="exact",
        )
        m = row["metrics"]
        assert "front_span" in m
        assert "back_span" in m
        assert "transition_direction" in m
        assert m["front_span"]["target_emotion"] == "ang"
        assert m["back_span"]["target_emotion"] == "hap"

    def test_approximate_row_has_null_boundary_error(self):
        """Part A (approximate) → boundary_error_sec is None."""
        control = _make_control_record(
            "fedd_a_001", boundary_word_index=None, part="A",
        )
        gen = _make_gen_row("fedd_a_001")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        row = build_eval_row(
            "fedd_a_001", control, gen,
            evaluator.identity(), self._frames_output("sad", "sur"),
            AlignmentResult(status="not_attempted", words=[]),
            evidence_tier="approximate",
        )
        assert row["boundary_evidence_tier"] == "approximate"
        assert row["metrics"]["boundary_error_sec"] is None

    def test_no_alignment_null_boundary_error_exact(self):
        """Exact tier + alignment failed → boundary_error_sec null + reason."""
        control = _make_control_record("fedd_b_003")
        gen = _make_gen_row("fedd_b_003")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        failed = AlignmentResult(status="failed", words=[], reason="low_score")
        row = build_eval_row(
            "fedd_b_003", control, gen,
            evaluator.identity(), self._frames_output("ang", "hap"),
            failed, evidence_tier="exact",
        )
        assert row["metrics"]["boundary_error_sec"] is None
        assert row["metrics"]["alignment_status"] == "failed"
        assert row["metrics"]["alignment_reason"] == "low_score"


# ============================================================
# 7. evaluate_fedd_dataset — full pipeline
# ============================================================


class TestEvaluateFeddDataset:
    """Integration: control + generation manifests → rows + aggregates."""

    def test_correct_transition_pipeline(self):
        """Correct transition: front emo_from hit + back emo_to hit."""
        control = _make_control_record("u1", emo_from="ang", emo_to="hap",
                                        boundary_word_index=3)
        gen = _make_gen_row("u1")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        # 为 fake evaluator 注册一个带已知 transition 的 clip
        clip_map = {
            "u1": SyntheticReferenceClip(
                wav_path=gen["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from="ang", known_transition_to="hap",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        aligner = FakeForcedAligner(boundaries_by_utt={"u1": [
            (0.0, 0.5, "the"), (0.5, 1.0, "quick"), (1.0, 1.5, "brown"),
            (1.5, 2.0, "fox"), (2.0, 2.5, "jumps"), (2.5, 3.0, "over"),
        ]})
        result = evaluate_fedd_dataset(
            [control], [gen], mapped_eval, aligner=aligner,
        )
        rows = result["rows"]
        assert len(rows) == 1
        m = rows[0]["metrics"]
        assert m["front_span"]["hit"] is True
        assert m["back_span"]["hit"] is True
        assert m["transition_direction"] == "correct"
        assert m["boundary_error_sec"] is not None

    def test_reverse_transition_pipeline(self):
        """Reverse: front=emo_to, back=emo_from."""
        control = _make_control_record("u2", emo_from="ang", emo_to="hap")
        gen = _make_gen_row("u2")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        clip_map = {
            "u2": SyntheticReferenceClip(
                wav_path=gen["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from="hap", known_transition_to="ang",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        aligner = FakeForcedAligner()
        result = evaluate_fedd_dataset(
            [control], [gen], mapped_eval, aligner=aligner,
        )
        m = result["rows"][0]["metrics"]
        assert m["transition_direction"] == "reverse"

    def test_maintained_pipeline(self):
        """Maintained: single emotion throughout."""
        control = _make_control_record("u3", emo_from="neu", emo_to="sad")
        gen = _make_gen_row("u3")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        clip_map = {
            "u3": SyntheticReferenceClip(
                wav_path=gen["wav_path"], duration_sec=3.0,
                known_emotion="neu",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        aligner = FakeForcedAligner()
        result = evaluate_fedd_dataset(
            [control], [gen], mapped_eval, aligner=aligner,
        )
        m = result["rows"][0]["metrics"]
        assert m["transition_direction"] == "maintained"

    def test_boundary_early_negative_error(self):
        """Transition before aligned boundary → negative error."""
        control = _make_control_record("u4", boundary_word_index=4)
        gen = _make_gen_row("u4")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        # transition 在 1.0s；对齐边界在 2.0s（第 4 个词的结尾）
        clip_map = {
            "u4": SyntheticReferenceClip(
                wav_path=gen["wav_path"], duration_sec=4.0,
                known_transition_sec=1.0,
                known_transition_from="ang", known_transition_to="hap",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        aligner = FakeForcedAligner(boundaries_by_utt={"u4": [
            (0.0, 0.5, "the"), (0.5, 1.0, "quick"), (1.0, 1.5, "brown"),
            (1.5, 2.0, "fox"), (2.0, 2.5, "jumps"), (2.5, 3.0, "over"),
        ]})
        result = evaluate_fedd_dataset(
            [control], [gen], mapped_eval, aligner=aligner,
        )
        m = result["rows"][0]["metrics"]
        assert m["boundary_error_sec"] is not None
        assert m["boundary_error_sec"] < 0  # early = negative

    def test_boundary_late_positive_error(self):
        """Transition after aligned boundary → positive error."""
        control = _make_control_record("u5", boundary_word_index=2)
        gen = _make_gen_row("u5")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        # transition 在 2.0s；对齐边界在 1.0s（第 2 个词的结尾）
        clip_map = {
            "u5": SyntheticReferenceClip(
                wav_path=gen["wav_path"], duration_sec=3.0,
                known_transition_sec=2.0,
                known_transition_from="ang", known_transition_to="hap",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        aligner = FakeForcedAligner(boundaries_by_utt={"u5": [
            (0.0, 0.5, "the"), (0.5, 1.0, "quick"), (1.0, 1.5, "brown"),
        ]})
        result = evaluate_fedd_dataset(
            [control], [gen], mapped_eval, aligner=aligner,
        )
        m = result["rows"][0]["metrics"]
        assert m["boundary_error_sec"] is not None
        assert m["boundary_error_sec"] > 0  # late = positive

    def test_fedda_not_in_exact_aggregate(self):
        """FEDD-A approximate rows do not enter exact aggregate."""
        ctrl_b = _make_control_record("b1", boundary_word_index=3, part="B")
        ctrl_a = _make_control_record("a1", boundary_word_index=None, part="A")
        gen_b = _make_gen_row("b1")
        gen_a = _make_gen_row("a1")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        clip_map = {
            "b1": SyntheticReferenceClip(
                wav_path=gen_b["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from="ang", known_transition_to="hap",
            ),
            "a1": SyntheticReferenceClip(
                wav_path=gen_a["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from="sad", known_transition_to="sur",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        aligner = FakeForcedAligner()
        result = evaluate_fedd_dataset(
            [ctrl_b, ctrl_a], [gen_b, gen_a], mapped_eval, aligner=aligner,
        )
        exact_agg = result["aggregate_exact"]
        approx_agg = result["aggregate_approximate"]
        assert exact_agg["evidence_tier"] == "exact"
        assert exact_agg["n_samples"] == 1
        assert approx_agg["evidence_tier"] == "approximate"
        assert approx_agg["n_samples"] == 1

    def test_no_alignment_null_boundary(self):
        """Alignment failure → boundary_error_sec null, not fabricated."""
        control = _make_control_record("u6")
        gen = _make_gen_row("u6")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        clip_map = {
            "u6": SyntheticReferenceClip(
                wav_path=gen["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from="ang", known_transition_to="hap",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        # 始终失败的 aligner
        aligner = FakeForcedAligner(always_fail=True)
        result = evaluate_fedd_dataset(
            [control], [gen], mapped_eval, aligner=aligner,
        )
        m = result["rows"][0]["metrics"]
        assert m["boundary_error_sec"] is None
        assert m["alignment_status"] == "failed"

    def test_all_rows_pass_validate_eval_row(self):
        """Every row in the output passes validate_eval_row."""
        controls = [
            _make_control_record(f"u{i}", emo_from=e1, emo_to=e2,
                                  boundary_word_index=3 if i % 2 == 0 else None)
            for i, (e1, e2) in enumerate(
                [("ang", "hap"), ("neu", "sad"), ("sad", "sur")])
        ]
        gens = [_make_gen_row(c["utt_id"]) for c in controls]
        evaluator = FakeAcousticEvaluator(kind="emotion")
        clip_map = {
            c["utt_id"]: SyntheticReferenceClip(
                wav_path=g["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from=c["emo_from"],
                known_transition_to=c["emo_to"],
            )
            for c, g in zip(controls, gens)
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        result = evaluate_fedd_dataset(
            controls, gens, mapped_eval, aligner=FakeForcedAligner(),
        )
        for row in result["rows"]:
            validate_eval_row(row)

    def test_aggregates_pass_validate_aggregate(self):
        controls = [_make_control_record("u7", boundary_word_index=3)]
        gens = [_make_gen_row("u7")]
        evaluator = FakeAcousticEvaluator(kind="emotion")
        clip_map = {
            "u7": SyntheticReferenceClip(
                wav_path=gens[0]["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from="ang", known_transition_to="hap",
            ),
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        result = evaluate_fedd_dataset(
            controls, gens, mapped_eval, aligner=FakeForcedAligner(),
        )
        validate_aggregate(result["aggregate_exact"])


# ============================================================
# 8. Per-sample anomaly propagation (hard-fail)
# ============================================================


class TestAnomalyPropagation:
    """Anomalies carry utt_id and hard-fail (no partial means)."""

    def test_non_eos_hard_fail(self):
        """Non-EOS finish_reason → RuntimeError carrying utt_id."""
        control = _make_control_record("ne1")
        gen = _make_gen_row("ne1", finish_reason="max_len_reached")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises(RuntimeError, match="ne1"):
            evaluate_fedd_dataset([control], [gen], evaluator)

    def test_missing_utt_in_generation(self):
        """utt_id in control but not in generation → hard-fail."""
        control = _make_control_record("m1")
        gen = _make_gen_row("m2")  # different utt_id
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises((ValueError, RuntimeError), match="m1|m2"):
            evaluate_fedd_dataset([control], [gen], evaluator)

    def test_duplicate_utt_in_generation(self):
        """Duplicate utt_id in generation manifest → hard-fail."""
        control = _make_control_record("d1")
        gen1 = _make_gen_row("d1")
        gen2 = _make_gen_row("d1")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises((ValueError, RuntimeError), match="d1"):
            evaluate_fedd_dataset([control], [gen1, gen2], evaluator)

    def test_duplicate_utt_in_control(self):
        """Duplicate utt_id in control manifest → hard-fail."""
        control1 = _make_control_record("d2")
        control2 = _make_control_record("d2")
        gen = _make_gen_row("d2")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises((ValueError, RuntimeError), match="d2"):
            evaluate_fedd_dataset([control1, control2], [gen], evaluator)

    def test_eos_without_wav_path_hard_fail(self):
        """EOS without wav_path → hard-fail."""
        control = _make_control_record("nw1")
        gen = _make_gen_row("nw1")
        del gen["wav_path"]  # explicitly remove wav_path from eos row
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises((ValueError, RuntimeError), match="nw1"):
            evaluate_fedd_dataset([control], [gen], evaluator)

    def test_checkpoint_mismatch_hard_fail(self):
        """Different checkpoint_sha256 across rows → identity mismatch."""
        control1 = _make_control_record("cm1")
        control2 = _make_control_record("cm2")
        gen1 = _make_gen_row("cm1", checkpoint_sha256="a" * 64)
        gen2 = _make_gen_row("cm2", checkpoint_sha256="b" * 64)
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises((ValueError, RuntimeError), match="checkpoint"):
            evaluate_fedd_dataset(
                [control1, control2], [gen1, gen2], evaluator,
            )


# ============================================================
# 9. Aggregate determinism
# ============================================================


class TestAggregateDeterminism:
    """Aggregate can be deterministically recomputed from rows."""

    def test_recompute_gives_same_result(self):
        controls = [
            _make_control_record(f"det{i}", boundary_word_index=3)
            for i in range(3)
        ]
        gens = [_make_gen_row(c["utt_id"]) for c in controls]
        evaluator = FakeAcousticEvaluator(kind="emotion")
        clip_map = {
            c["utt_id"]: SyntheticReferenceClip(
                wav_path=g["wav_path"], duration_sec=3.0,
                known_transition_sec=1.5,
                known_transition_from=c["emo_from"],
                known_transition_to=c["emo_to"],
            )
            for c, g in zip(controls, gens)
        }
        from tests._emofilm_fakes import ClipMappedEvaluator
        mapped_eval = ClipMappedEvaluator(evaluator, clip_map)
        result = evaluate_fedd_dataset(
            controls, gens, mapped_eval, aligner=FakeForcedAligner(),
        )
        exact_rows = [r for r in result["rows"] if r["boundary_evidence_tier"] == "exact"]
        recomputed = build_aggregate_from_rows(exact_rows, "exact")
        original = result["aggregate_exact"]
        assert recomputed["n_samples"] == original["n_samples"]
        for key in original["metrics"]:
            if isinstance(original["metrics"][key], (int, float)):
                assert recomputed["metrics"][key] == pytest.approx(
                    original["metrics"][key]
                )

    def test_empty_rows_aggregate_n_zero(self):
        """No rows of a tier → aggregate with n_samples=0."""
        agg = build_aggregate_from_rows([], "approximate")
        validate_aggregate(agg)
        assert agg["n_samples"] == 0


# ============================================================
# 10. Embedded identity check (ticket 07 / 核查 #5)
# ============================================================


class TestEmbeddedIdentityCheck:
    """gen 内嵌 control_row_ref / prompt_row_ref 必须与配对实际使用的一致。

    防止 generation row 的 WAV 实际来自另一控制条件却被按外层 utt_id 配对
    打分（核查 #5：_strict_pair 此前仅按 utt_id 配对，不校验内嵌身份）。
    """

    def test_identity_consistent_passes(self):
        """(a) gen 内嵌 control/prompt 身份与配对一致 → 通过。"""
        # 控制记录显式声明 control_row_ref / prompt_row_ref
        ctrl = _make_control_record("id1", extra={
            "control_row_ref": "control/id1",
            "prompt_row_ref": "prompt/speaker_0011",
        })
        # _make_gen_row 默认 control_row_ref=f"control/{utt_id}",
        # prompt_row_ref="prompt/speaker_0011" —— 与控制记录一致
        gen = _make_gen_row("id1")
        evaluator = FakeAcousticEvaluator(kind="emotion")
        result = evaluate_fedd_dataset([ctrl], [gen], evaluator)
        assert result["n_samples"] == 1

    def test_utt_id_fallback_identity_consistent_passes(self):
        """控制记录无 control_row_ref 时，用 utt_id 作为身份仍能与 gen 一致。"""
        # _make_control_record 默认不带 control_row_ref → 回退到 utt_id
        ctrl = _make_control_record("id1b")
        gen = _make_gen_row("id1b")  # control_row_ref=control/id1b → 核心id1b
        evaluator = FakeAcousticEvaluator(kind="emotion")
        result = evaluate_fedd_dataset([ctrl], [gen], evaluator)
        assert result["n_samples"] == 1

    def test_control_row_ref_pointing_elsewhere_hard_fails(self):
        """(b) gen 内嵌 control_row_ref 指向另一请求 → hard-fail。"""
        ctrl = _make_control_record("id2")  # utt_id=id2 → 期望 control 核心 "id2"
        gen = _make_gen_row("id2")
        gen["control_row_ref"] = "control/other_request"  # 指向另一请求
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises(ValueError, match=r"control_row_ref mismatch"):
            evaluate_fedd_dataset([ctrl], [gen], evaluator)

    @pytest.mark.parametrize(
        "ctrl_extra",
        [
            pytest.param(
                {"prompt_row_ref": "prompt/speaker_0011"},
                id="ctrl_with_prompt_row_ref",
            ),
            pytest.param({}, id="ctrl_without_prompt_row_ref"),
        ],
    )
    def test_prompt_mismatch_does_not_hard_fail(self, ctrl_extra):
        """(ticket 09) per-pair prompt 校验已删 → 行为回归。

        参数化 ctrl 是否带 ``prompt_row_ref``（前者 schema §1 非法字段、后者
        合法 SupervisionSpan）：两种情形下 gen 内嵌不一致的 prompt_row_ref
        均不触发 per-pair 校验（旧 ``_extract_ctrl_prompt_core`` 恒 "" → 死代码
        已删）。gen 的 prompt 族存在性由 ``validate_generation_row`` 保证
        （schema §2 四族各≥1），核心"prompt 死代码已删"由
        ``tests/test_eval_local_control.py::test_prompt_pair_check_removed``
        守护；本测试仅做行为回归。合并自原
        ``test_prompt_row_ref_mismatch_no_longer_hard_fails`` 与
        ``test_prompt_no_longer_checked_in_per_pair``（两测试断言一致，差异
        仅 ctrl 是否带 prompt_row_ref）。
        """
        ctrl = _make_control_record("id3", extra=ctrl_extra)
        gen = _make_gen_row("id3")
        gen["prompt_row_ref"] = "prompt/mismatch"  # 与 ctrl 不一致（若 ctrl 有）
        evaluator = FakeAcousticEvaluator(kind="emotion")
        # 不 raise —— per-pair prompt 校验已删（ticket 09）
        result = evaluate_fedd_dataset([ctrl], [gen], evaluator)
        assert result["n_samples"] == 1

    def test_mismatch_error_contains_utt_id_and_field(self):
        """hard-fail 错误信息含 utt_id 与冲突字段名（control_row_ref mismatch）。"""
        ctrl = _make_control_record("id6")
        gen = _make_gen_row("id6")
        gen["control_row_ref"] = "control/wrong"
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises(ValueError, match=r"id6") as exc_info:
            evaluate_fedd_dataset([ctrl], [gen], evaluator)
        assert "control_row_ref mismatch" in str(exc_info.value)

    def test_control_row_dict_consistent_passes(self):
        """gen 用 control_row（dict）内嵌身份且 utt_id 一致 → 通过。"""
        ctrl = _make_control_record("id4")
        gen = _make_gen_row("id4")
        # 用 control_row dict 替代 control_row_ref 字符串
        del gen["control_row_ref"]
        gen["control_row"] = {
            "utt_id": "id4", "emo_from": "ang", "emo_to": "hap",
        }
        evaluator = FakeAcousticEvaluator(kind="emotion")
        result = evaluate_fedd_dataset([ctrl], [gen], evaluator)
        assert result["n_samples"] == 1

    def test_control_row_dict_mismatch_hard_fails(self):
        """gen 用 control_row（dict）但 utt_id 指向另一请求 → hard-fail。"""
        ctrl = _make_control_record("id5")  # utt_id=id5
        gen = _make_gen_row("id5")
        del gen["control_row_ref"]
        gen["control_row"] = {"utt_id": "other_request"}
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises(ValueError, match=r"control_row_ref mismatch"):
            evaluate_fedd_dataset([ctrl], [gen], evaluator)

    def test_multi_row_one_mismatch_hard_fails(self):
        """多行中仅一行 control 身份错位 → 该行 hard-fail（携 utt_id）。"""
        ctrl_a = _make_control_record("m1")
        ctrl_b = _make_control_record("m2")
        gen_a = _make_gen_row("m1")
        gen_b = _make_gen_row("m2")
        gen_b["control_row_ref"] = "control/m1"  # m2 的 gen 指向 m1 的控制
        evaluator = FakeAcousticEvaluator(kind="emotion")
        with pytest.raises(ValueError, match=r"control_row_ref mismatch") as exc:
            evaluate_fedd_dataset(
                [ctrl_a, ctrl_b], [gen_a, gen_b], evaluator,
            )
        # m2 在 sorted 顺序中后于 m1，错误应在 m2 行触发
        assert "m2" in str(exc.value)
