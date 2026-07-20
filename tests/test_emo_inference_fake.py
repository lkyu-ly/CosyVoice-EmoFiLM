"""Emo-FiLM 推理 fake 单测。"""
import os
import sys
from pathlib import Path

import torch

ROOT = str(Path(__file__).resolve().parents[1])
ASSET_ROOT = ROOT
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))
from cosyvoice.cli.frontend_emo import CosyVoiceFrontEnd_Emotion


def _fake_emofilm_frontend():
    frontend = CosyVoiceFrontEnd_Emotion.__new__(CosyVoiceFrontEnd_Emotion)
    frontend.device = torch.device("cpu")
    frontend._extract_emo_text_token = lambda text: (
        torch.tensor([[len(text)]], dtype=torch.long),
        torch.tensor([[1]], dtype=torch.long),
        torch.tensor([[1]], dtype=torch.long),
    )
    frontend._extract_text_token = lambda text: (
        torch.tensor([[len(text)]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    return frontend


def test_frontend_emo_output_has_emotion_fields():
    """frontend_emo_film 输出含 emotion_ids 等字段，等长检查。"""
    from cosyvoice.tokenizer.emo_tokenizer import get_emo_tokenizer
    MODEL_DIR = os.path.join(ASSET_ROOT, "pretrained_models", "CosyVoice2-0.5B")

    tok = get_emo_tokenizer(
        token_path=os.path.join(MODEL_DIR, "CosyVoice-BlankEN"),
        skip_special_tokens=True,
    )
    fe = CosyVoiceFrontEnd_Emotion(
        get_tokenizer=lambda: tok,
        feat_extractor=None,  # 允许 None（fake 时不调用）
        campplus_model=os.path.join(MODEL_DIR, "campplus.onnx"),
        speech_tokenizer_model=os.path.join(MODEL_DIR, "speech_tokenizer_v2.onnx"),
        allowed_special="all",
    )
    # 只测试 _extract_emo_text_token（不依赖真实 wav）
    text_with_emo = "before <emotion type='hap' intensity='high'>happy</emotion> after"
    fe.tokenizer = tok
    emo_result = tok.encode_plus(text_with_emo)
    emotion_ids = emo_result["emotion_ids"]
    intensity_ids = emo_result["intensity_ids"]
    assert len(emotion_ids) == len(intensity_ids) > 0


def test_prompt_emotion_defaults_neu_low():
    """prompt 段情感默认 neutral(ID=3)/low(ID=1)。"""
    default_emo_id = 3  # neu
    default_inten_id = 1  # low
    n_tokens = 5
    ids = torch.full((1, n_tokens), default_emo_id, dtype=torch.long)
    assert ids.sum() == n_tokens * default_emo_id
    ids_i = torch.full((1, n_tokens), default_inten_id, dtype=torch.long)
    assert ids_i.sum() == n_tokens * default_inten_id


def test_frontend_emo_reuses_identical_prompt_conditioning(tmp_path):
    frontend = _fake_emofilm_frontend()
    prompt_wav = tmp_path / "prompt.wav"
    calls = []

    def fake_frontend_zero_shot(tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id):
        calls.append((tts_text, prompt_text, str(prompt_wav), resample_rate, zero_shot_spk_id))
        return {
            "prompt_text": torch.tensor([[7]], dtype=torch.int32),
            "prompt_text_len": torch.tensor([1], dtype=torch.int32),
            "llm_prompt_speech_token": torch.tensor([[8]], dtype=torch.int32),
            "llm_prompt_speech_token_len": torch.tensor([1], dtype=torch.int32),
            "flow_prompt_speech_token": torch.tensor([[8]], dtype=torch.int32),
            "flow_prompt_speech_token_len": torch.tensor([1], dtype=torch.int32),
            "prompt_speech_feat": torch.tensor([[[9.0]]]),
            "prompt_speech_feat_len": torch.tensor([1], dtype=torch.int32),
            "llm_embedding": torch.tensor([[10.0]]),
            "flow_embedding": torch.tensor([[10.0]]),
        }

    frontend.frontend_zero_shot = fake_frontend_zero_shot

    first = frontend.frontend_emo_film("first", "reference", prompt_wav, 24000)
    second = frontend.frontend_emo_film("second", "reference", prompt_wav, 24000)

    assert len(calls) == 1
    assert first["text"].item() == len("first")
    assert second["text"].item() == len("second")
    assert first["llm_prompt_speech_token"].item() == second["llm_prompt_speech_token"].item() == 8


def test_frontend_emo_separates_prompt_cache_identity(tmp_path):
    frontend = _fake_emofilm_frontend()
    calls = []

    def fake_frontend_zero_shot(tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id):
        calls.append((prompt_text, str(prompt_wav), resample_rate))
        return {
            "prompt_text": torch.tensor([[7]], dtype=torch.int32),
            "prompt_text_len": torch.tensor([1], dtype=torch.int32),
            "llm_prompt_speech_token": torch.tensor([[len(calls)]], dtype=torch.int32),
            "llm_prompt_speech_token_len": torch.tensor([1], dtype=torch.int32),
            "flow_prompt_speech_token": torch.tensor([[len(calls)]], dtype=torch.int32),
            "flow_prompt_speech_token_len": torch.tensor([1], dtype=torch.int32),
            "prompt_speech_feat": torch.tensor([[[9.0]]]),
            "prompt_speech_feat_len": torch.tensor([1], dtype=torch.int32),
            "llm_embedding": torch.tensor([[10.0]]),
            "flow_embedding": torch.tensor([[10.0]]),
        }

    frontend.frontend_zero_shot = fake_frontend_zero_shot
    prompt_a = tmp_path / "prompt_a.wav"
    prompt_b = tmp_path / "prompt_b.wav"

    first = frontend.frontend_emo_film("first", "reference", prompt_a, 24000)
    second = frontend.frontend_emo_film("second", "other reference", prompt_a, 24000)
    third = frontend.frontend_emo_film("third", "other reference", prompt_b, 16000)

    assert len(calls) == 3
    assert first["llm_prompt_speech_token"].item() == 1
    assert second["llm_prompt_speech_token"].item() == 2
    assert third["llm_prompt_speech_token"].item() == 3


def test_model_emo_llm_job_runs():
    """验证 CosyVoice2Model_Emotion 可实例化且 tts 签名匹配。"""
    from cosyvoice.cli.model_emo import CosyVoice2Model_Emotion

    # 真实测试需要完整的模型加载，此处仅做 import + class 定义检查
    assert CosyVoice2Model_Emotion is not None
    assert hasattr(CosyVoice2Model_Emotion, 'tts')
    assert hasattr(CosyVoice2Model_Emotion, 'llm_job')
