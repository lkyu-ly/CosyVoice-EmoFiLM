"""QwenTokenizer_Emotion 单测。覆盖 spec 7.3 所有验收条件。"""
import os

import pytest
import torch

# Check that QwenTokenizer base is importable before using emotion wrapper
TOKEN_PATH = "/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN"
from cosyvoice.tokenizer.emo_tokenizer import QwenTokenizer_Emotion


@pytest.fixture
def tokenizer():
    return QwenTokenizer_Emotion(token_path=TOKEN_PATH)


def test_triple_equal_length(tokenizer):
    """text_token/emotion_ids/intensity_ids 等长。"""
    result = tokenizer.encode_plus("<emotion type='hap' intensity='high'>hello world</emotion>")
    assert len(result["text_token"]) == len(result["emotion_ids"])
    assert len(result["text_token"]) == len(result["intensity_ids"])
    assert len(result["text_token"]) > 0


def test_unlabeled_text_defaults_to_neu_low(tokenizer):
    """无标签文本映射到默认 neu/low。"""
    result = tokenizer.encode_plus("hello world")
    neu_id = tokenizer.emotion_to_id["neu"]
    low_id = tokenizer.intensity_to_id["low"]
    for eid in result["emotion_ids"]:
        assert eid == neu_id
    for iid in result["intensity_ids"]:
        assert iid == low_id


def test_no_lowercase_by_default(tokenizer):
    """默认不 lower text。"""
    result = tokenizer.encode_plus("HELLO World")
    decoded = tokenizer.decode(result["text_token"])
    assert "HELLO" in decoded or "Hello" in decoded  # 不强制 lowercase


def test_mixed_tagged_and_plain(tokenizer):
    """标签内外混合文本对齐正确。"""
    result = tokenizer.encode_plus(
        "I am <emotion type='sad' intensity='low'>feeling down</emotion> today"
    )
    assert len(result["text_token"]) == len(result["emotion_ids"])
    sad_id = tokenizer.emotion_to_id["sad"]
    neu_id = tokenizer.emotion_to_id["neu"]
    # 存在 sad 段（feeling down）
    assert sad_id in result["emotion_ids"].tolist()
    # 存在 neu 段（I am / today）
    assert neu_id in result["emotion_ids"].tolist()


def test_missing_closing_tag_raises_strict(tokenizer):
    """缺失闭合标签在严格模式下报错。"""
    tokenizer.strict = True
    with pytest.raises(ValueError, match="closing"):
        tokenizer.encode_plus("<emotion type='hap' intensity='high'>hello")


def test_original_tokenize_regression(tokenizer):
    """原 CosyVoice2 tokenizer encode() 不可破坏。"""
    # encode() 应和 AutoTokenizer 一致
    tokens = tokenizer.encode("hello")
    assert len(tokens) > 0
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)


def test_emotion_id_range(tokenizer):
    """情感 ID 在 [1,5] 范围，强度 ID 在 [1,3] 范围。"""
    result = tokenizer.encode_plus(
        "<emotion type='ang' intensity='high'>angry</emotion>"
        "<emotion type='sur' intensity='medium'>surprise</emotion>"
    )
    eids = result["emotion_ids"].tolist()
    iids = result["intensity_ids"].tolist()
    assert all(1 <= e <= 5 for e in eids)
    assert all(1 <= i <= 3 for i in iids)


def test_multi_token_word_same_label(tokenizer):
    """多 token 单词同一情感标签。"""
    result = tokenizer.encode_plus("<emotion type='hap' intensity='medium'>absolutely</emotion>")
    emo_ids = result["emotion_ids"].tolist()
    # absolutely → 多个 token，所有 token 同 ID
    hap_id = tokenizer.emotion_to_id["hap"]
    for eid in emo_ids:
        assert eid == hap_id
