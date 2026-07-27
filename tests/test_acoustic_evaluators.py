"""Ticket 08 — 声学 evaluator 接口 + Fake + 方向性校验逻辑 focused 测试。

覆盖（见 brief 08 §D）：
- ``identity()`` 字段齐全（model_id / revision / label_mapping / sample_rate /
  frame_rate / window / semantics / limitations / calibration /
  shares_source_with_iemocap_weak_supervision）；
- ``is_frozen=True``，包装对象不持有 trainable 任务头；
- Fake 在合成输入上行为正确且确定性；
- 类别映射 / 方向性 / transition 定位 / arousal 方向 校验逻辑行为正确；
- 未校准时 ``calibration=None`` 且 score 不命名 confidence；
- ``shares_source_with_iemocap_weak_supervision`` 字段存在，emotion2vec 实现下为 True，
  Fake 下为 False。

CPU，用 FakeAcousticEvaluator（不加载真实 emotion2vec）。MAP §4 合同/行为测试。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from eval.acoustic_evaluators import (
    EMOTION_LABEL_SPACE,
    EMOTION_LABEL_TO_IDX,
    FAKE_MODEL_ID,
    FakeAcousticEvaluator,
    SyntheticReferenceClip,
    validate_arousal_direction,
    validate_emotion_label_mapping,
    validate_transition_localization,
)


# ============================================================
# identity() 完整性
# ============================================================

_IDENTITY_REQUIRED_KEYS = frozenset({
    "model_id", "revision", "label_mapping", "sample_rate_hz",
    "frame_rate_hz", "window_strategy", "output_semantics",
    "known_limitations", "calibration",
    "shares_source_with_iemocap_weak_supervision",
    # 合同 Evaluator TypedDict 字段（build_emofilm_contract.Evaluator）
    "name", "version", "label_space", "self_evidence_risk",
})


def test_fake_emotion_identity_has_all_required_fields():
    fake = FakeAcousticEvaluator(kind="emotion")
    ident = fake.identity()
    missing = _IDENTITY_REQUIRED_KEYS - ident.keys()
    assert not missing, f"identity() missing required keys: {sorted(missing)}"


def test_fake_arousal_identity_has_all_required_fields():
    fake = FakeAcousticEvaluator(kind="arousal")
    ident = fake.identity()
    missing = _IDENTITY_REQUIRED_KEYS - ident.keys()
    assert not missing, f"identity() missing required keys: {sorted(missing)}"


def test_fake_label_space_is_five_emotions():
    fake = FakeAcousticEvaluator(kind="emotion")
    ident = fake.identity()
    assert ident["label_space"] == list(EMOTION_LABEL_SPACE)
    assert len(ident["label_space"]) == 5


def test_fake_arousal_label_mapping_is_none():
    fake = FakeAcousticEvaluator(kind="arousal")
    ident = fake.identity()
    assert ident["label_mapping"] is None


# ============================================================
# is_frozen
# ============================================================

def test_fake_is_frozen_true():
    fake = FakeAcousticEvaluator()
    assert fake.is_frozen is True


# ============================================================
# 确定性
# ============================================================

def _make_clip(path_str, **kwargs):
    return SyntheticReferenceClip(wav_path=path_str, duration_sec=2.0, **kwargs)


def test_fake_emotion_predict_frames_is_deterministic():
    clip = _make_clip("utt_hap.wav", known_emotion="hap")
    fake = FakeAcousticEvaluator(kind="emotion")
    out1 = fake.predict_frames(clip)
    out2 = fake.predict_frames(clip)
    np.testing.assert_array_equal(out1["frames"], out2["frames"])


def test_fake_arousal_predict_frames_is_deterministic():
    clip = _make_clip("utt_low.wav", known_arousal_rank=0)
    fake = FakeAcousticEvaluator(kind="arousal")
    out1 = fake.predict_frames(clip)
    out2 = fake.predict_frames(clip)
    np.testing.assert_array_equal(out1["frames"], out2["frames"])


# ============================================================
# 输出格式
# ============================================================

def test_fake_emotion_output_shape_and_normalization():
    clip = _make_clip("utt_neu.wav", known_emotion="neu")
    fake = FakeAcousticEvaluator(kind="emotion")
    out = fake.predict_frames(clip)
    assert out["frames"].ndim == 2
    assert out["frames"].shape[1] == 5  # 5 emotions
    assert out["frames"].shape[0] > 0
    # 每帧分布和为 1
    row_sums = out["frames"].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    assert "label_space" in out
    assert out["label_space"] == list(EMOTION_LABEL_SPACE)
    assert "frame_rate_hz" in out
    assert "times_sec" in out
    assert len(out["times_sec"]) == out["frames"].shape[0]


def test_fake_arousal_output_shape():
    clip = _make_clip("utt_ramp.wav", known_arousal_rank=1)
    fake = FakeAcousticEvaluator(kind="arousal")
    out = fake.predict_frames(clip)
    assert out["frames"].ndim == 1
    assert out["frames"].shape[0] > 0
    assert "frame_rate_hz" in out
    assert "times_sec" in out
    assert len(out["times_sec"]) == out["frames"].shape[0]


def test_fake_output_never_has_confidence_key():
    """未校准时 score 不命名 confidence（MAP §3）。"""
    clip = _make_clip("utt_neu.wav", known_emotion="neu")
    fake = FakeAcousticEvaluator(kind="emotion")
    out = fake.predict_frames(clip)
    assert "confidence" not in out, (
        "predict_frames output must never contain 'confidence' "
        "(MAP §3: use raw_score + calibration)"
    )


def test_fake_uncalibrated_identity_calibration_is_none():
    fake = FakeAcousticEvaluator()
    assert fake.identity()["calibration"] is None


# ============================================================
# shares_source_with_iemocap_weak_supervision
# ============================================================

def test_fake_shares_source_is_false():
    """Fake 不共享 emotion2vec 上游，无自证风险。"""
    fake = FakeAcousticEvaluator()
    assert fake.identity()["shares_source_with_iemocap_weak_supervision"] is False
    assert fake.identity()["self_evidence_risk"] is False


def test_fake_model_id_is_fake():
    fake = FakeAcousticEvaluator()
    assert fake.identity()["model_id"] == FAKE_MODEL_ID


# ============================================================
# validate_emotion_label_mapping
# ============================================================

def test_validate_emotion_label_mapping_passes_on_correct_clip():
    """已知 emotion 参考片段上 argmax 与已知标签一致 → pass。"""
    clips = [
        _make_clip("a.wav", known_emotion="hap"),
        _make_clip("b.wav", known_emotion="ang"),
        _make_clip("c.wav", known_emotion="sad"),
    ]
    fake = FakeAcousticEvaluator(kind="emotion")
    result = validate_emotion_label_mapping(fake, clips)
    assert result["passed"] is True
    assert result["n_total"] == 3
    assert result["n_passed"] == 3


def test_validate_emotion_label_mapping_fails_on_wrong_clip():
    """当 evaluator 预测与已知标签不一致 → fail（不止凭模型名）。"""
    clips = [_make_clip("a.wav", known_emotion="hap")]
    # 构造一个「故意反」的 Fake：把 hap 映射到 ang
    fake = FakeAcousticEvaluator(
        kind="emotion",
        emotion_override={"hap": "ang"},
    )
    result = validate_emotion_label_mapping(fake, clips)
    assert result["passed"] is False
    assert result["n_passed"] == 0


def test_validate_emotion_label_mapping_skips_clips_without_known_emotion():
    clips = [
        _make_clip("known.wav", known_emotion="hap"),
        _make_clip("unknown.wav"),  # no known_emotion
    ]
    fake = FakeAcousticEvaluator(kind="emotion")
    result = validate_emotion_label_mapping(fake, clips)
    assert result["n_total"] == 1  # only the known one counted


# ============================================================
# validate_emotion_label_mapping —— 非有限/空分布门禁（Ticket #12）
# ============================================================
# 若 predict_frames 返全 NaN / inf / 空，不能让 argmax 退化为 label_space[0]
# 而误判为 PASS。必须在 argmax 之前拦截。

def test_all_nan_distribution_fails_calibration(monkeypatch):
    """全 NaN 分布 → mean_dist 全 NaN → argmax 返 0（=label_space[0]）。
    若 known_emotion 恰为 label_space[0]（"ang"），未修复时会误通过。
    """
    fake = FakeAcousticEvaluator(kind="emotion")
    monkeypatch.setattr(
        fake, "predict_frames",
        lambda _clip: {"frames": np.full((4, 5), np.nan)},
    )
    clip = _make_clip("x", known_emotion="ang")  # ang==label_space[0]
    result = validate_emotion_label_mapping(fake, [clip])
    assert result["passed"] is False
    assert result["n_passed"] == 0
    err = str(result["details"][0].get("error", "")).lower()
    assert "non-finite" in err, f"error should mention non-finite, got: {err!r}"


def test_inf_distribution_fails_calibration(monkeypatch):
    """含 inf 的分布同样非有限 → fail。"""
    fake = FakeAcousticEvaluator(kind="emotion")
    frames = np.full((4, 5), 0.2)
    frames[1, 2] = np.inf
    monkeypatch.setattr(fake, "predict_frames", lambda _clip: {"frames": frames})
    clip = _make_clip("x", known_emotion="neu")  # neu==label_space[2]
    result = validate_emotion_label_mapping(fake, [clip])
    assert result["passed"] is False
    err = str(result["details"][0].get("error", "")).lower()
    assert "non-finite" in err


def test_empty_frames_distribution_fails_calibration(monkeypatch):
    """空 frames（shape[0]==0）→ mean_dist 为空 → fail。"""
    fake = FakeAcousticEvaluator(kind="emotion")
    monkeypatch.setattr(
        fake, "predict_frames",
        lambda _clip: {"frames": np.zeros((0, 5))},
    )
    clip = _make_clip("x", known_emotion="ang")
    result = validate_emotion_label_mapping(fake, [clip])
    assert result["passed"] is False
    err = str(result["details"][0].get("error", "")).lower()
    assert "non-finite" in err or "empty" in err, (
        f"error should mention non-finite/empty, got: {err!r}"
    )


def test_nan_and_valid_mix_overall_fails(monkeypatch):
    """一个 clip 全 NaN + 一个 clip 正常 → 整体 fail（n_passed=1 但 n_total=2）。"""
    fake = FakeAcousticEvaluator(kind="emotion")

    def predict(clip):
        if clip.wav_path == "bad":
            return {"frames": np.full((4, 5), np.nan)}
        # 正常 hap 分布
        frames = np.full((4, 5), 0.05)
        frames[:, EMOTION_LABEL_TO_IDX["hap"]] = 0.8
        return {"frames": frames}

    monkeypatch.setattr(fake, "predict_frames", predict)
    clips = [
        _make_clip("bad", known_emotion="ang"),
        _make_clip("good", known_emotion="hap"),
    ]
    result = validate_emotion_label_mapping(fake, clips)
    assert result["passed"] is False
    assert result["n_total"] == 2
    assert result["n_passed"] == 1  # 只有 good 通过
    # bad clip 的 detail 应含 error
    bad_detail = next(d for d in result["details"] if d["wav_path"] == "bad")
    assert "non-finite" in str(bad_detail.get("error", "")).lower()


# ============================================================
# validate_transition_localization
# ============================================================

def test_validate_transition_localization_locates_known_transition():
    """已知 transition 时间上检测到的转换点在容差内。"""
    clip = SyntheticReferenceClip(
        wav_path="trans.wav",
        duration_sec=4.0,
        known_transition_sec=2.0,
        known_transition_from="neu",
        known_transition_to="ang",
    )
    fake = FakeAcousticEvaluator(kind="emotion")
    result = validate_transition_localization(fake, [clip], tolerance_sec=0.5)
    assert result["passed"] is True
    assert abs(result["details"][0]["detected_transition_sec"] - 2.0) < 0.5


def test_validate_transition_localization_fails_when_no_transition_found():
    """给了一个已知 transition 但 evaluator 输出无转换 → fail。"""
    clip = SyntheticReferenceClip(
        wav_path="trans.wav",
        duration_sec=4.0,
        known_transition_sec=2.0,
        known_transition_from="neu",
        known_transition_to="ang",
    )
    # Fake 无 transition 信息 → 输出恒定分布 → 检测不到转换
    fake = FakeAcousticEvaluator(
        kind="emotion",
        ignore_transition=True,
    )
    result = validate_transition_localization(fake, [clip], tolerance_sec=0.5)
    assert result["passed"] is False


# ============================================================
# validate_arousal_direction
# ============================================================

def test_validate_arousal_direction_monotonic_on_ordered_clips():
    """arousal 在可排序参考上单调递增 → pass。"""
    clips = [
        _make_clip("low.wav", known_arousal_rank=0),
        _make_clip("mid.wav", known_arousal_rank=1),
        _make_clip("high.wav", known_arousal_rank=2),
    ]
    fake = FakeAcousticEvaluator(kind="arousal")
    result = validate_arousal_direction(fake, clips)
    assert result["passed"] is True
    means = result["details"]["mean_arousal_by_rank"]
    assert means[0] < means[1] < means[2]


def test_validate_arousal_direction_sorts_before_checking():
    # 输入 clips 的 rank 顺序非单调（0, 2, 1）；validate 应先按 rank 排序
    # 再检查 mean 单调，因此实际 pass（Fake 按 rank 线性 → 排序后 mean 单调）。
    # 此测试覆盖"排序正路径"；fail 分支由 test_validate_arousal_direction_fails_when_flat 覆盖。
    clips = [
        _make_clip("low.wav", known_arousal_rank=0),
        _make_clip("high.wav", known_arousal_rank=2),
        _make_clip("mid.wav", known_arousal_rank=1),
    ]
    fake = FakeAcousticEvaluator(kind="arousal")
    result = validate_arousal_direction(fake, clips)
    assert result["passed"] is True


def test_validate_arousal_direction_fails_when_flat():
    """arousal 无区分度（恒定）→ fail。"""
    clips = [
        _make_clip("low.wav", known_arousal_rank=0),
        _make_clip("high.wav", known_arousal_rank=2),
    ]
    fake = FakeAcousticEvaluator(
        kind="arousal",
        flat_arousal=True,
    )
    result = validate_arousal_direction(fake, clips)
    assert result["passed"] is False


# ============================================================
# identity 兼容合同 Evaluator TypedDict
# ============================================================

def test_fake_identity_evaluator_dict_is_valid_contract_evaluator():
    """identity() 的 name/version/label_space 等字段满足 v2 合同 Evaluator。"""
    from tools.build_emofilm_contract import Evaluator, validate_eval_row

    fake = FakeAcousticEvaluator(kind="emotion")
    ident = fake.identity()
    # 合同 Evaluator 要求 name/version 非空
    assert ident["name"]
    assert ident["version"]
    # 构造一个合法 eval row 并校验
    eval_row = {
        "utt_id": "test_utt",
        "generation_row_ref": "gen_001",
        "control_span_ref": "span_001",
        "evaluator": {
            "name": ident["name"],
            "version": ident["version"],
            "label_space": ident["label_space"],
            "frame_rate_hz": ident["frame_rate_hz"],
            "self_evidence_risk": ident["self_evidence_risk"],
        },
        "boundary_evidence_tier": "exact",
        "metrics": {"emo_sim": 0.95},
    }
    validated = validate_eval_row(eval_row)
    assert validated["evaluator"]["name"] == ident["name"]


# ============================================================
# EmotionEvaluator / ArousalEvaluator 协议合规
# ============================================================

def test_fake_emotion_satisfies_emotion_evaluator_protocol():
    from eval.acoustic_evaluators import EmotionEvaluator

    fake = FakeAcousticEvaluator(kind="emotion")
    assert isinstance(fake, EmotionEvaluator)


def test_fake_arousal_satisfies_arousal_evaluator_protocol():
    from eval.acoustic_evaluators import ArousalEvaluator

    fake = FakeAcousticEvaluator(kind="arousal")
    assert isinstance(fake, ArousalEvaluator)
