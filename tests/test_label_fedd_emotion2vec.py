"""FEDD emotion2vec 全局打标/一致性校验测试。

不加载真实模型；fake model 验证标签映射、批量推理契约、manifest 字段写回、
一致性统计、--skip_labeled 只推理未标条目。
"""
import os
import sys
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# emotion2vec_plus 9 类中文/英文标签顺序（前 5 类为 FEDD 关心的 5 类）
_LABELS = ["生气/angry", "开心/happy", "中立/neutral", "难过/sad", "吃惊/surprised"]


def _result_for(label, score):
    """构造单条 emotion2vec generate 结果 dict：top=label@score，其余均分低分。"""
    scores = [0.02] * len(_LABELS)
    scores[_LABELS.index(label)] = score
    return {"labels": list(_LABELS), "scores": scores}


def test_map_label():
    from tools.label_fedd_emotion2vec import map_label

    assert map_label("生气/angry") == "ang"
    assert map_label("开心/happy") == "hap"
    assert map_label("中立/neutral") == "neu"
    assert map_label("难过/sad") == "sad"
    assert map_label("吃惊/surprised") == "sur"
    assert map_label("恐惧/fearful") == "other"
    assert map_label("<unk>") == "other"


def test_batch_inference_single_call_and_report(tmp_path):
    """批量契约：一次 model.generate(wav 列表, batch_size) 覆盖全部；结果按序回填。"""
    from tools.label_fedd_emotion2vec import label_manifest

    entries = [
        {"utt_id": "b1", "wav_path": "b1.wav", "emo_from": "ang", "emo_to": "hap", "part": "B"},
        {"utt_id": "b2", "wav_path": "b2.wav", "emo_from": "sad", "emo_to": "sur", "part": "B"},
        {"utt_id": "a1", "wav_path": "a1.wav", "emo_from": "neu", "emo_to": "hap", "part": "A"},
    ]
    # fake：b1 判 angry（pass）、b2 判 neutral（fail）、a1 判 happy（pass）
    verdicts = {"b1.wav": ("生气/angry", 0.9),
                "b2.wav": ("中立/neutral", 0.6),
                "a1.wav": ("开心/happy", 0.8)}
    calls = {}

    def fake_generate(inp, **kw):
        # inp 必须是 wav 路径列表（批量），返回等长结果列表，顺序对齐
        assert isinstance(inp, list)
        calls["input"] = inp
        calls["kw"] = kw
        return [_result_for(*verdicts[w]) for w in inp]

    model = MagicMock()
    model.generate.side_effect = fake_generate

    new_entries, report = label_manifest(model, entries, batch_size=8)

    # 单次批量调用，透传 batch_size / granularity / extract_embedding
    assert model.generate.call_count == 1
    assert calls["input"] == ["b1.wav", "b2.wav", "a1.wav"]
    assert calls["kw"]["batch_size"] == 8
    assert calls["kw"]["granularity"] == "utterance"
    assert calls["kw"]["extract_embedding"] is False

    # 结果按序回填
    assert new_entries[0]["emo2vec_label"] == "ang"
    assert abs(new_entries[0]["emo2vec_score"] - 0.9) < 1e-6
    assert new_entries[1]["emo2vec_label"] == "neu"
    assert new_entries[2]["emo2vec_label"] == "hap"

    # 一致性统计（整体 + 分 part）
    assert report["total"] == 3 and report["passed"] == 2
    assert report["by_part"]["B"]["total"] == 2 and report["by_part"]["B"]["passed"] == 1
    assert report["by_part"]["A"]["passed"] == 1


def test_skip_labeled_only_infers_missing():
    """--skip_labeled：仅对无 emo2vec_label 的条目推理，已标条目保留旧标签。"""
    from tools.label_fedd_emotion2vec import label_manifest

    entries = [
        {"utt_id": "b1", "wav_path": "b1.wav", "emo_from": "ang", "emo_to": "hap",
         "part": "B", "emo2vec_label": "ang", "emo2vec_score": 0.7},   # 已标 → 跳过
        {"utt_id": "a1", "wav_path": "a1.wav", "emo_from": "neu", "emo_to": "hap",
         "part": "A"},                                                  # 未标 → 推理
    ]
    seen = {}

    def fake_generate(inp, **kw):
        seen["input"] = inp
        return [_result_for("开心/happy", 0.8) for _ in inp]

    model = MagicMock()
    model.generate.side_effect = fake_generate

    new_entries, report = label_manifest(model, entries, skip_labeled=True, batch_size=4)

    assert model.generate.call_count == 1
    assert seen["input"] == ["a1.wav"]                 # 只推理未标的
    assert new_entries[0]["emo2vec_label"] == "ang"    # 保留旧标签
    assert abs(new_entries[0]["emo2vec_score"] - 0.7) < 1e-6
    assert new_entries[1]["emo2vec_label"] == "hap"    # 新标
    assert report["total"] == 2 and report["passed"] == 2


def test_skip_labeled_all_present_no_inference():
    """全部已标 + --skip_labeled：不触发任何推理调用。"""
    from tools.label_fedd_emotion2vec import label_manifest

    entries = [{"utt_id": "x", "wav_path": "x.wav", "emo_from": "ang", "emo_to": "hap",
                "part": "B", "emo2vec_label": "ang", "emo2vec_score": 0.7}]
    model = MagicMock()
    new_entries, report = label_manifest(model, entries, skip_labeled=True)
    model.generate.assert_not_called()
    assert report["total"] == 1 and report["passed"] == 1
