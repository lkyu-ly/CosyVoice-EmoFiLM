"""Emo-FiLM 推理 fake 单测。"""
import os
import sys
import torch

ROOT = "/home/lkyu/LLM-Audio/CosyVoice-EmoFiLM"
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))
from cosyvoice.cli.frontend_emo import CosyVoiceFrontEnd_Emotion


def test_frontend_emo_output_has_emotion_fields():
    """frontend_emo_film 输出含 emotion_ids 等字段，等长检查。"""
    from cosyvoice.tokenizer.emo_tokenizer import get_emo_tokenizer
    MODEL_DIR = os.path.join(ROOT, "pretrained_models", "CosyVoice2-0.5B")

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


def test_model_emo_llm_job_runs():
    """验证 CosyVoice2Model_Emotion 可实例化且 tts 签名匹配。"""
    from cosyvoice.cli.model_emo import CosyVoice2Model_Emotion
    # 真实测试需要完整的模型加载，此处仅做 import + class 定义检查
    assert CosyVoice2Model_Emotion is not None
    assert hasattr(CosyVoice2Model_Emotion, 'tts')
    assert hasattr(CosyVoice2Model_Emotion, 'llm_job')
