"""ESD 句级 + FEDD 词级 tagged_text 生成器测试。

验证标签格式与 emo_tokenizer 的 EMOTION_TAG_PATTERN 兼容（论文 Global/Fine-grained Label 条件）。
"""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.build_esd_tagged_text import sentence_tagged_text, build as build_esd
from tools.build_fedd_tagged_text import word_level_tagged_text, build as build_fedd

# emo_tokenizer.py 的解析正则，确保产物可被真实前端解析
TAG_RE = re.compile(r"<emotion type='(\w+)' intensity='(\w+)'>(.*?)</emotion>")


def test_esd_sentence_tag_format():
    t = sentence_tagged_text("Hello world", "hap", "medium")
    m = list(TAG_RE.finditer(t))
    assert len(m) == 1
    assert m[0].groups() == ("hap", "medium", "Hello world")


def test_esd_invalid_emotion_raises():
    with pytest.raises(ValueError):
        sentence_tagged_text("x", "happy", "medium")  # 必须是 hap 不是 happy


def test_esd_build_adds_tagged_text():
    recs = [
        {"utt_id": "a", "text": "Hello", "sentence_emotion": "ang"},
        {"utt_id": "b", "text": "Bye", "sentence_emotion": "sad"},
    ]
    out = build_esd(recs, "medium")
    assert all("tagged_text" in r for r in out)
    assert TAG_RE.search(out[0]["tagged_text"]).groups() == ("ang", "medium", "Hello")
    assert recs[0].get("tagged_text") is None  # 不原地改输入


def test_fedd_word_level_split_at_boundary():
    """boundary_word_index=2：前2词 emo_from，其余 emo_to。"""
    t = word_level_tagged_text("one two three four five", "ang", "hap", boundary_word_index=2)
    segs = TAG_RE.findall(t)
    assert len(segs) == 2
    assert segs[0] == ("ang", "medium", "one two")
    assert segs[1] == ("hap", "medium", "three four five")


def test_fedd_boundary_clamped():
    """越界 k 被 clamp 到 [1, n-1]，仍产出两段。"""
    t = word_level_tagged_text("a b c", "sad", "sur", boundary_word_index=99)
    segs = TAG_RE.findall(t)
    assert len(segs) == 2
    assert segs[0][2] == "a b" and segs[1][2] == "c"


def test_fedd_single_word_no_split():
    t = word_level_tagged_text("word", "neu", "hap", boundary_word_index=1)
    segs = TAG_RE.findall(t)
    assert len(segs) == 1 and segs[0][0] == "neu"


def test_fedd_build_partb_and_missing_emo():
    import pytest
    valid = [{"utt_id": "b1", "text": "one two three", "emo_from": "ang", "emo_to": "hap", "boundary_word_index": 1, "part": "B"}]
    out = build_fedd(valid, "medium")
    assert TAG_RE.findall(out[0]["tagged_text"]) == [("ang", "medium", "one"), ("hap", "medium", "two three")]
    # Part B 有 boundary → method = exact_concatenation_boundary（ADR-0003 戳记）
    assert out[0]["method"] == "exact_concatenation_boundary"
    assert out[0]["label_source"] == "construction_known_transition"
    # 缺过渡信息 → hard-fail（ADR-0003：FEDD 控制标签必须来自构造已知转折，不再 neutral 兜底）
    with pytest.raises(ValueError, match="emo_from/emo_to"):
        build_fedd([{"utt_id": "x", "text": "no emo info", "part": "?"}], "medium")
