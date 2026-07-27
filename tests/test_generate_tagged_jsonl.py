"""EmoFiLM 数据流水线辅助函数测试（ADR-0020 扁平化后）。

``tools/generate_tagged_jsonl.py`` 已合并为监督 span 生成器（原 v1 argmax 词级
标注器仅存 git 基线 ``9c6d84b``）。本文件保留对**仍存活的数据流水线辅助函数**
的测试：
- ``merge_word_predictions``（相邻词 emotion+intensity 双键合并 → ``<emotion>`` 标签）；
- ``classify_text_coverage``（精确词覆盖 / 撇号切分等价；其他差异拒绝）；
- ``intensity_from_arousal``（arousal → 离散强度控制输入阈值；与原 v1
  ``arousal_to_intensity`` 同语义，仅作控制接口，不作强度真值）。

v1 生成器专属测试（v1 CLI 子进程产 ``<emotion>`` 标签、v1 ``generate_tagged_dataset``
拒绝过滤）随 v1 代码移除——其覆盖的 v1 argmax 路径已被监督 span 生成器取代，
对应行为在 ``test_emofilm_weak_supervision.py`` 覆盖。
"""

import pytest

from tools.build_emofilm_contract import (
    classify_text_coverage,
    merge_word_predictions,
    validate_span,
)
from tools.generate_tagged_jsonl import (
    intensity_from_arousal,
    merge_word_predictions_to_v2_spans,
)


def test_merge_requires_matching_emotion_and_intensity():
    tagged = merge_word_predictions(
        [
            {"word": "I", "predicted_emotion": "ang", "predicted_intensity": "high"},
            {"word": "am", "predicted_emotion": "ang", "predicted_intensity": "high"},
            {"word": "happy", "predicted_emotion": "hap", "predicted_intensity": "medium"},
            {"word": "today", "predicted_emotion": "hap", "predicted_intensity": "low"},
        ]
    )
    assert tagged.count("<emotion") == 3
    assert "I am" in tagged
    assert "happy</emotion>" in tagged
    assert "today</emotion>" in tagged


def test_text_coverage_accepts_only_exact_or_apostrophe_equivalent_pairs():
    assert classify_text_coverage(
        "Everybody's told the story.",
        "<emotion type='neu' intensity='low'>everybody</emotion> "
        "<emotion type='neu' intensity='low'>'s told the story</emotion>",
    )["decision"] == "keep"
    mismatch = classify_text_coverage(
        "Mmhmm. Yeah.",
        "<emotion type='neu' intensity='low'>yeah</emotion>",
    )
    assert mismatch["decision"] == "reject"
    assert mismatch["category"] == "audio_text_mismatch"
    assert mismatch["missing_from_tagged"] == ["mmhmm"]


def test_intensity_from_arousal_bucketing():
    """arousal → 离散强度控制输入阈值（仅控制接口；连续 arousal 仍是监督 target）。"""
    assert intensity_from_arousal(4.0) == "high"
    assert intensity_from_arousal(3.0) == "medium"
    assert intensity_from_arousal(2.0) == "low"
    assert intensity_from_arousal(1.5) == "low"


# ============================================================
# Task 7：calibrated/calibration 信息贯穿（brief 07）
# ============================================================


def _base_pred(**overrides: object) -> dict[str, object]:
    """构造一条合法的 predictor 输出（默认未校准），可被 overrides 覆盖。"""
    pred: dict[str, object] = {
        "word": "hi",
        "start_sec": 0.0,
        "end_sec": 0.5,
        "start_frame": 0,
        "end_frame": 12,
        "frame_rate_hz": 50.0,
        "emotion_soft_distribution": [0.1, 0.8, 0.05, 0.03, 0.02],
        "arousal": 3.0,
        "raw_score": 0.8,
    }
    pred.update(overrides)
    return pred


def test_calibrated_predictor_propagates_to_span():
    """predictor 返 calibrated=True + calibration → span 透传（不死写 False）。"""
    pred = _base_pred(
        calibrated=True,
        calibration={"method": "temperature", "version": "v1"},
    )
    spans = merge_word_predictions_to_v2_spans(
        utt_id="u1",
        word_preds=[pred],
        sentence_emotion="hap",
        sentence_vad=None,
        annotator_provenance={"model": "x"},
    )
    assert spans[0]["calibrated"] is True
    assert spans[0]["calibration"] == {"method": "temperature", "version": "v1"}
    # 成员词也应保留 calibrated/calibration 以便溯源（brief §1）。
    member = spans[0]["provenance"]["member_words"][0]
    assert member["calibrated"] is True
    assert member["calibration"] == {"method": "temperature", "version": "v1"}


def test_mixed_calibrated_members_raise():
    """相邻成员 calibrated 不一致 → raise ValueError 携 utt_id（合并兼容键含 calibrated）。

    场景：两词同 control_emotion_id/control_intensity_id（默认 medium），故仅
    calibrated 差异触发合并路径 → 不一致必须 raise，否则 span 会写首个成员的值
    而丢失另一成员的校准状态。
    """
    pred_a = _base_pred(
        word="hello",
        calibrated=True,
        calibration={"method": "temperature", "version": "v1"},
    )
    pred_b = _base_pred(
        word="world",
        calibrated=False,
    )
    # 两词 emotion/intensity 兼容（同 neu/medium）；calibrated 不一致 → raise。
    with pytest.raises(ValueError, match="u1") as exc_info:
        merge_word_predictions_to_v2_spans(
            utt_id="u1",
            word_preds=[pred_a, pred_b],
            sentence_emotion="hap",
            sentence_vad=None,
            annotator_provenance={"model": "x"},
        )
    assert "calibrat" in str(exc_info.value).lower()


def test_default_calibrated_false_when_predictor_omits():
    """predictor 不返 calibrated → 默认 False + calibration None（语义正确）。"""
    pred = _base_pred()  # 无 calibrated/calibration
    spans = merge_word_predictions_to_v2_spans(
        utt_id="u1",
        word_preds=[pred],
        sentence_emotion="hap",
        sentence_vad=None,
        annotator_provenance={"model": "x"},
    )
    assert spans[0]["calibrated"] is False
    assert spans[0].get("calibration") is None


def test_calibrated_span_passes_validator_end_to_end():
    """端到端：predictor 返完整 calibration → 生成的 span 通过 validate_span。

    覆盖票据 05 validator 的 calibrated=True 分支（此前死校验：生成器硬编码 False
    导致该分支无法被真实数据触发）。calibration 必须含样本集溯源字段
    （calibration_sample_set_ref / n_calibration_samples）。
    """
    pred = _base_pred(
        calibrated=True,
        calibration={
            "method": "isotonic",
            "version": "emotion2vec-v1",
            "calibration_sample_set_ref": "iemocap_calib_split:v1",
            "n_calibration_samples": 512,
        },
    )
    spans = merge_word_predictions_to_v2_spans(
        utt_id="iemocap_sess1_0001",
        word_preds=[pred],
        sentence_emotion="hap",
        sentence_vad=None,
        annotator_provenance={"model": "x"},
    )
    # 必须真正是 calibrated=True（修复前生成器硬编码 False 导致 validator True 分支死锁）。
    assert spans[0]["calibrated"] is True
    assert spans[0]["calibration"]["method"] == "isotonic"
    # 生成器产出 → validator 接受（此前 calibrated=True 分支不可达）。
    validate_span(spans[0])
