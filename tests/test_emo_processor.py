"""dataset processor 扩展 tokenize_emo + padding 单测。"""
import os
import sys
import torch

ROOT = "/home/lkyu/LLM-Audio/CosyVoice-EmoFiLM"
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))

TOKEN_PATH = "/home/lkyu/LLM-Audio/CosyVoice-EmoFiLM/pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN"


def test_tokenize_emo_output_has_emotion_fields():
    """tokenize_emo 输出含 emotion_ids/intensity_ids。"""
    from cosyvoice.tokenizer.emo_tokenizer import get_emo_tokenizer
    from cosyvoice.dataset.processor import tokenize_emo

    samples = [{"text": "<emotion type='hap' intensity='high'>hello world</emotion> test"}]

    def get_tok():
        return get_emo_tokenizer(token_path=TOKEN_PATH)

    result = list(tokenize_emo(iter(samples), get_tok, "all"))
    assert len(result) == 1
    sample = result[0]
    assert "text_token" in sample
    assert "emotion_ids" in sample
    assert "intensity_ids" in sample
    assert len(sample["emotion_ids"]) == len(sample["text_token"])


def test_tokenize_emo_does_not_break_tokenize():
    """原 tokenize 函数输出不受影响。"""
    from cosyvoice.tokenizer.emo_tokenizer import get_emo_tokenizer
    from cosyvoice.dataset.processor import tokenize, tokenize_emo

    samples = [{"text": "hello world"}]

    def get_tok():
        return get_emo_tokenizer(token_path=TOKEN_PATH)

    result_orig = list(tokenize(iter(samples), get_tok, "all"))
    result_emo = list(tokenize_emo(iter(samples), get_tok, "all"))

    assert result_orig[0]["text_token"] == result_emo[0]["text_token"]


def test_padding_includes_emotion_fields():
    """padding 扩展后 batch 含 emotion_ids/intensity_ids。"""
    from cosyvoice.tokenizer.emo_tokenizer import get_emo_tokenizer
    from cosyvoice.dataset.processor import tokenize_emo, padding

    samples = [
        {"text": "<emotion type='ang' intensity='low'>angry</emotion>", "utt": "u1"},
        {"text": "<emotion type='hap' intensity='high'>happy longer</emotion>", "utt": "u2"},
    ]

    def get_tok():
        return get_emo_tokenizer(token_path=TOKEN_PATH)

    processed = list(tokenize_emo(iter(samples), get_tok, "all"))
    # 手工补上 padding 需要的字段
    for i, s in enumerate(processed):
        s["speechFeat"] = torch.randn(50, 80)
        s["speech_feat"] = torch.randn(50, 80)
        s["speech"] = torch.randn(1, 1200)
        s["utt_embedding"] = torch.randn(192)
        s["spk_embedding"] = torch.randn(192)
    batched = list(padding(iter([processed]), use_spk_embedding=False))
    assert len(batched) == 1
    batch = batched[0]
    assert "emotion_ids" in batch
    assert "intensity_ids" in batch
    assert batch["emotion_ids"].dim() == 2
    assert batch["intensity_ids"].dim() == 2
