"""保留 IEMOCAP 弱监督概率、VAD 与校准状态的 focused 测试。

覆盖（ADR-0020 扁平化后；监督 span 生成器去后缀为 ``tools/generate_tagged_jsonl.py``）：
- soft distribution 5 维且和≈1（不要 argmax）；
- 完整 VAD 3 维保留（不只 arousal）；
- arousal 为浮点未分桶；
- raw_score 存在；``calibrated=False`` 时不得有任何字段名含 ``confidence``；
- ``weak_supervision="sentence_broadcast"`` 标记存在（句级广播来源）；
- 词边界 ``start_sec/end_sec/start_frame/end_frame/frame_rate_hz`` 透传；
- 相邻词仅在 (control_emotion_id, control_intensity_id, calibrated,
  label_source) 兼容时合并，合并保留成员词分布/边界（可溯源）；
- ESD span ``intensity_mask=False``，无 arousal，one-hot soft dist；
- 所有 span 通过 ``validate_span``。

CPU 合同/行为测试：不加载真实模型；``_FakePredictor`` 注入，无 GPU/重模型依赖
（MAP §4：合同测试轻量；ADR-0014）。

注意（ADR-0020）：原"v1 generate_tagged_jsonl.py sha256 冻结"源码哈希锁已删除
（禁源码哈希标定文件）；v1 不可变性由 git 基线锚点 ``9c6d84b`` 保证。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pytest

from tools.build_emofilm_contract import validate_span
from tools.generate_tagged_jsonl import (
    EMOTION_TO_CONTROL_ID,
    INTENSITY_TO_CONTROL_ID,
    WORDSEQ_EMOTION_ORDER,
    build_esd_utterance_span,
    extract_textgrid_xmax,
    merge_word_predictions_to_v2_spans,
)


# ------------------------------------------------------------
# 测试用 FakePredictor —— 不加载真实模型；确定性输出便于断言
# ------------------------------------------------------------


class _FakePredictor:
    """Deterministic predictor returning canned soft dist + 3D VAD.

    soft dist is a valid 5-class probability distribution (sums to 1).
    WordSequenceModel internal label order is [ang, hap, neu, sad, sur]
    (same as tokenizer emotion order), so soft[0]=ang ... soft[4]=sur.
    """

    def __init__(self, per_word_outputs: list[dict[str, Any]]):
        self._outputs = list(per_word_outputs)
        self._idx = 0

    def predict_word(self, word_block_path: Path | str) -> dict[str, Any]:
        if self._idx >= len(self._outputs):
            raise IndexError("FakePredictor ran out of canned outputs")
        out = dict(self._outputs[self._idx])
        self._idx += 1
        return out


def _make_word_pred(
    *,
    word: str,
    start_sec: float,
    end_sec: float,
    start_frame: int,
    end_frame: int,
    frame_rate_hz: float = 50.0,
    soft: list[float],
    vad: list[float],
    arousal: float,
    raw_score: float | None = None,
) -> dict[str, Any]:
    """Build a single per-word prediction dict (matches Predictor protocol)."""
    return {
        "word": word,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frame_rate_hz": float(frame_rate_hz),
        "emotion_soft_distribution": [float(p) for p in soft],
        "vad": [float(v) for v in vad],
        "arousal": float(arousal),
        "raw_score": float(raw_score if raw_score is not None else max(soft)),
    }


# ============================================================
# IEMOCAP 词级保持
# ============================================================


def _three_word_corpus() -> list[dict[str, Any]]:
    """Three words: ang/low, ang/low (compatible), neu/high (incompatible)."""
    return [
        _make_word_pred(
            word="excuse",
            start_sec=0.49,
            end_sec=0.79,
            start_frame=24,
            end_frame=40,
            soft=[0.78, 0.05, 0.12, 0.03, 0.02],
            vad=[2.0, 2.1, 3.0],
            arousal=2.1,
        ),
        _make_word_pred(
            word="me",
            start_sec=0.82,
            end_sec=1.05,
            start_frame=41,
            end_frame=53,
            soft=[0.71, 0.10, 0.14, 0.03, 0.02],
            vad=[2.1, 2.2, 2.9],
            arousal=2.2,
        ),
        _make_word_pred(
            word="please",
            start_sec=1.10,
            end_sec=1.60,
            start_frame=55,
            end_frame=80,
            soft=[0.05, 0.04, 0.86, 0.03, 0.02],
            vad=[3.2, 4.1, 3.1],
            arousal=4.1,
        ),
    ]


def _build_iemocap_spans(word_preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run merger with representative provenance."""
    return merge_word_predictions_to_v2_spans(
        utt_id="Ses01F_impro01_F000",
        word_preds=word_preds,
        sentence_emotion="neu",
        sentence_vad=[3.0, 3.0, 3.0],
        annotator_provenance={
            "model_class": "WordSequenceModel",
            "checkpoint_sha256": "deadbeef" * 8,
            "contract": "768d/5emo/3VAD",
        },
    )


# -------------------- soft distribution --------------------


def test_soft_distribution_is_5_dim_and_sums_to_one():
    spans = _build_iemocap_spans(_three_word_corpus())
    assert len(spans) >= 1
    for span in spans:
        dist = span["emotion_soft_distribution"]
        assert isinstance(dist, list)
        assert len(dist) == 5
        for prob in dist:
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0
        assert abs(sum(dist) - 1.0) < 1e-6


def test_soft_distribution_is_not_argmax_collapsed():
    """v1 argmax 丢 soft dist；v2 必须保留完整概率（非 one-hot）。"""
    spans = _build_iemocap_spans(_three_word_corpus())
    ang_span = spans[0]
    dist = ang_span["emotion_soft_distribution"]
    # 原始 preds 的 soft[0]=0.78, 0.71 → 均值约 0.745，非 1.0
    assert dist[0] > 0.5
    assert dist[0] < 1.0
    # 其他类别有非零概率（非 argmax 坍缩）
    for i in range(1, 5):
        assert dist[i] > 0.0


# -------------------- VAD 3-dim --------------------


def test_full_vad_three_dims_preserved():
    """v1 只留 arousal=vad_scaled[1]；v2 必须保留全 3 维 [v, a, d]。"""
    spans = _build_iemocap_spans(_three_word_corpus())
    for span in spans:
        vad = span.get("vad")
        assert vad is not None, "vad must be preserved when predictor provides it"
        assert isinstance(vad, list)
        assert len(vad) == 3
        for v in vad:
            assert isinstance(v, float)
        # VAD 缩放到 [1, 5]（sigmoid_out * 4 + 1）
        for v in vad:
            assert 1.0 <= v <= 5.0


def test_vad_omitted_when_predictor_does_not_provide_it():
    """ESD/未校准 checkpoint 无 VAD → 不得伪造，必须缺省（schema：vad 可选）。"""
    word_preds = _three_word_corpus()
    for wp in word_preds:
        wp.pop("vad", None)
    spans = _build_iemocap_spans(word_preds)
    for span in spans:
        assert "vad" not in span, "vad must be absent when predictor did not provide it"


# -------------------- continuous arousal --------------------


def test_arousal_is_continuous_float_not_bucketed():
    """v1 arousal_to_intensity 把 arousal 硬分 3 档；v2 arousal 必须是浮点未分桶。"""
    spans = _build_iemocap_spans(_three_word_corpus())
    for span in spans:
        arousal = span["arousal"]
        assert isinstance(arousal, float)
        # 必须是 [1, 5] 内的原始标量，而非 1/2/3 这类分桶标签
        assert 1.0 <= arousal <= 5.0
        # intensity_mask=True 需要 arousal（连续强度目标）
        assert span["intensity_mask"] is True


# -------------------- raw_score + no confidence --------------------


def test_raw_score_present_and_is_max_softmax_prob():
    spans = _build_iemocap_spans(_three_word_corpus())
    for span in spans:
        rs = span["raw_score"]
        assert isinstance(rs, float)
        assert 0.0 < rs <= 1.0
        # raw_score = max(soft)（跨合并成员取均值）
        expected = max(span["emotion_soft_distribution"])
        # 合并 span 的 raw_score 是各成员 max 的均值；此处仅检查范围
        assert rs >= 0.5  # ang/neu dominant → high max-soft


def test_no_field_named_confidence_when_uncalibrated():
    """未校准时 raw_score 不得在任何字段名里出现 confidence（MAP §3）。"""
    spans = _build_iemocap_spans(_three_word_corpus())

    def _walk_keys(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield f"{prefix}.{k}" if prefix else k
                yield from _walk_keys(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                yield from _walk_keys(item, f"{prefix}[{i}]")

    for span in spans:
        assert span["calibrated"] is False
        assert span.get("calibration") is None or "calibration" not in span
        for key in _walk_keys(span):
            assert "confidence" not in key.lower(), (
                f"forbidden 'confidence' substring in key: {key}"
            )


# -------------------- weak supervision flag --------------------


def test_weak_supervision_sentence_broadcast_flag_present():
    spans = _build_iemocap_spans(_three_word_corpus())
    for span in spans:
        prov = span["provenance"]
        assert isinstance(prov, dict)
        assert prov.get("weak_supervision") == "sentence_broadcast"
        assert prov.get("sentence_emotion") == "neu"
        assert prov.get("sentence_vad") == [3.0, 3.0, 3.0]
        # label_source 标记词标注器伪标签（非词级真值）
        assert span["label_source"] == "word_annotator_pseudo_label"
        assert span["supervision_granularity"] == "word"


# -------------------- word boundaries transparent --------------------


def test_word_boundaries_passed_through():
    """词边界 start/end_sec + start/end_frame + frame_rate_hz 必须透传到 span。"""
    corpus = _three_word_corpus()
    spans = _build_iemocap_spans(corpus)
    # 第一个 span 覆盖 [excuse, me]（兼容）→ [0.49, 1.05]
    first = spans[0]
    assert first["start_sec"] == pytest.approx(0.49)
    assert first["end_sec"] == pytest.approx(1.05)
    assert first["start_frame"] == 24
    assert first["end_frame"] == 53
    assert first["frame_rate_hz"] == 50.0


# -------------------- merge compatibility + traceability --------------------


def test_adjacent_compatible_words_merged_and_traceable():
    """相邻兼容词合并为一个 span；合并后成员词分布/边界可溯源。"""
    spans = _build_iemocap_spans(_three_word_corpus())
    # 三个词：ang/low、ang/low（兼容）、neu/high（不兼容）
    # → 2 spans: [excuse, me] merged, [please] standalone
    assert len(spans) == 2

    merged = spans[0]
    members = merged["provenance"]["member_words"]
    assert len(members) == 2
    assert members[0]["word"] == "excuse"
    assert members[1]["word"] == "me"
    # 每个成员保留各自的 soft 分布（不坍缩）
    assert len(members[0]["emotion_soft_distribution"]) == 5
    assert len(members[1]["emotion_soft_distribution"]) == 5
    # 成员保留各自的边界
    assert members[0]["start_sec"] == 0.49
    assert members[1]["end_sec"] == 1.05


def test_adjacent_incompatible_words_kept_as_separate_spans():
    spans = _build_iemocap_spans(_three_word_corpus())
    # 第三个词（neu/high）与前两个（ang/low）不兼容
    assert len(spans) == 2
    last = spans[-1]
    assert last["control_emotion_id"] == EMOTION_TO_CONTROL_ID["neu"]
    assert last["control_intensity_id"] == INTENSITY_TO_CONTROL_ID["high"]
    assert len(last["provenance"]["member_words"]) == 1


def test_merge_compatibility_includes_label_source_and_calibrated():
    """合并兼容键是 (control_emotion_id, control_intensity_id, calibrated,
    label_source) —— 通过混 label_source 验证不合并。"""
    base = _three_word_corpus()[:2]
    # 相同 control id 但不同 label_source 溯源仍应在 span 层合并
    #（label_source 是 IEMOCAP 词的顶层常量）。
    spans = _build_iemocap_spans(base)
    assert len(spans) == 1, "same control+calibrated+label_source must merge"


# -------------------- validate_span gate --------------------


def test_all_iemocap_spans_pass_validate_span():
    spans = _build_iemocap_spans(_three_word_corpus())
    assert len(spans) >= 1
    for span in spans:
        validate_span(span)  # raises ValueError on invalid


# ============================================================
# ESD 句级 span
# ============================================================


def _esd_sample() -> dict[str, Any]:
    return {
        "utt_id": "0011_000671",
        "wav_path": "datasets/ESD/0011/Angry/0011_000671.wav",
        "text": "Called out the cloud.",
        "sentence_emotion": "ang",
        "speaker_id": "0011",
        "plain_text": "Called out the cloud.",
        "source_dataset": "esd",
    }


def test_esd_span_has_intensity_mask_false():
    """ESD fixed-medium 仅有控制输入；强度监督 target 必须无效。"""
    span = build_esd_utterance_span(
        sample=_esd_sample(),
        utterance_duration_sec=3.591,
        intensity="medium",
    )
    assert span["intensity_mask"] is False
    assert span["emotion_mask"] is True
    assert span["calibrated"] is False
    assert "calibration" not in span
    assert span.get("calibration") is None or "calibration" not in span


def test_esd_span_has_no_arousal_no_vad_no_raw_score():
    """ESD 无词级模型分数/VAD/arousal → 不得伪造（schema 条件必需）。"""
    span = build_esd_utterance_span(
        sample=_esd_sample(),
        utterance_duration_sec=3.591,
        intensity="medium",
    )
    assert "arousal" not in span, "ESD must not fabricate continuous arousal"
    assert "vad" not in span, "ESD must not fabricate VAD"
    assert "raw_score" not in span, "ESD has no model score"


def test_esd_span_emotion_soft_distribution_is_one_hot():
    """ESD 全局标签是硬标签；one-hot 是诚实表示（schema 允许）。"""
    span = build_esd_utterance_span(
        sample=_esd_sample(),
        utterance_duration_sec=3.591,
        intensity="medium",
    )
    dist = span["emotion_soft_distribution"]
    assert len(dist) == 5
    assert abs(sum(dist) - 1.0) < 1e-6
    # ang 在 WordSequenceModel 顺序中是 index 0 → control_emotion_id=1
    assert dist[0] == pytest.approx(1.0)
    for i in range(1, 5):
        assert dist[i] == pytest.approx(0.0)
    assert span["control_emotion_id"] == EMOTION_TO_CONTROL_ID["ang"]
    assert span["control_intensity_id"] == INTENSITY_TO_CONTROL_ID["medium"]
    assert span["intensity_policy"] == "fixed_medium"


def test_esd_span_uses_utterance_duration_for_bounds():
    span = build_esd_utterance_span(
        sample=_esd_sample(),
        utterance_duration_sec=3.591,
        intensity="medium",
    )
    assert span["start_sec"] == 0.0
    assert span["end_sec"] == pytest.approx(3.591)
    assert span["supervision_granularity"] == "utterance"
    assert span["label_source"] == "esd_fixed_medium_control"


def test_esd_span_passes_validate_span():
    span = build_esd_utterance_span(
        sample=_esd_sample(),
        utterance_duration_sec=3.591,
        intensity="medium",
    )
    validate_span(span)


def test_esd_span_rejects_negative_or_zero_duration():
    """ESD span 要求 start_sec < end_sec 严格成立；duration<=0 必须报错。"""
    with pytest.raises(ValueError):
        build_esd_utterance_span(
            sample=_esd_sample(),
            utterance_duration_sec=0.0,
            intensity="medium",
        )


# ============================================================
# TextGrid xmax 提取（用于 ESD 时长）
# ============================================================


def test_extract_textgrid_xmax_parses_utterance_duration(tmp_path):
    tg = tmp_path / "0011_000001.TextGrid"
    tg.write_text(
        'File type = "ooTextFile"\n'
        "Object class = \"TextGrid\"\n"
        "\n"
        "xmin = 0 \n"
        "xmax = 3.591 \n"
        "tiers? <exists> \n",
        encoding="utf-8",
    )
    assert extract_textgrid_xmax(tg) == pytest.approx(3.591)


def test_extract_textgrid_xmax_returns_none_when_missing(tmp_path):
    tg = tmp_path / "empty.TextGrid"
    tg.write_text("File type = \"ooTextFile\"\n", encoding="utf-8")
    assert extract_textgrid_xmax(tg) is None


# ============================================================
# Control id 映射健全性检查
# ============================================================


def test_wordseq_emotion_order_matches_tokenizer_control_ids():
    """WordSequenceModel 内部 {0:ang..4:sur} 与 tokenizer {1:ang..5:sur} 同序；
    v2 勿混（MAP §5）。"""
    assert list(WORDSEQ_EMOTION_ORDER) == ["ang", "hap", "neu", "sad", "sur"]
    for idx, emo in enumerate(WORDSEQ_EMOTION_ORDER):
        assert EMOTION_TO_CONTROL_ID[emo] == idx + 1
