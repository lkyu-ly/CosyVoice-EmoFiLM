"""Task 3 (#2): frontend_emo_film target-only 协议闭合 —— prompt_* 死字段回归。

背景：``frontend_emo_film`` 曾把 ``prompt_emotion_ids`` / ``prompt_intensity_ids``
塞进 ``model_input``，但 ``CosyVoice2Model_Emotion.tts`` 签名并不消费它们
（被 ``**kwargs`` 静默吞），属于 v1 残留死字段。``_PROMPT_CONDITIONING_KEYS``
同时保留了 ``prompt_text`` / ``prompt_text_len``，也是死字段（target-only 协议
下 LLM 只看 target 端 ``text`` + ``emotion_ids`` + ``intensity_ids``）。

本模块锁定：修复后输出只携带 ``tts`` 真正消费的键 —— target 端三件套
（``text`` / ``emotion_ids`` / ``intensity_ids``）+ 声学 prompt 键
（``flow_embedding`` / ``llm_embedding`` / ``llm_prompt_speech_token`` /
``flow_prompt_speech_token`` / ``prompt_speech_feat``）。
"""
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))

from cosyvoice.cli.frontend_emo import CosyVoiceFrontEnd_Emotion


@pytest.fixture
def fake_frontend(tmp_path):
    """最小 fake frontend：仅覆盖 ``frontend_emo_film`` 真正调用的方法。

    不加载真实模型 / tokenizer / wav；``frontend_zero_shot`` 返回固定声学 prompt
    张量。同时埋一个调用计数探针，以便测试断言死代码路径不再触发。
    """
    frontend = CosyVoiceFrontEnd_Emotion.__new__(CosyVoiceFrontEnd_Emotion)
    frontend.device = torch.device("cpu")

    # target 端 emo tokenizer stub —— ``frontend_emo_film`` 仍调用。
    frontend._extract_emo_text_token = lambda text: (
        torch.tensor([[max(1, len(text))]], dtype=torch.long),
        torch.tensor([[1]], dtype=torch.long),
        torch.tensor([[1]], dtype=torch.long),
    )
    # 纯文本 tokenizer stub —— 死代码删除后 ``frontend_emo_film`` 不应再调用。
    # 保留可调用 stub 避免误报 AttributeError；调用次数由测试显式断言。
    calls = {"count": 0}

    def _stub_extract_text_token(text):
        calls["count"] += 1
        return (
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        )

    frontend._extract_text_token = _stub_extract_text_token
    frontend._extract_text_token_calls = calls

    def fake_frontend_zero_shot(
        tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        # 返回包含 v1 prompt_text 字段的“脏”字典 —— 修复后应被
        # _PROMPT_CONDITIONING_KEYS 过滤掉，不进入 frontend_emo_film 输出。
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
    return frontend


def test_frontend_emo_film_no_dead_prompt_fields(fake_frontend, tmp_path):
    """target-only 协议：``frontend_emo_film`` 输出不得携带 v1 prompt_* 死字段。"""
    prompt_wav = tmp_path / "prompt.wav"
    out = fake_frontend.frontend_emo_film(
        "<emotion type='hap'>hi</emotion>", "ref text", prompt_wav
    )

    dead = {
        "prompt_text",
        "prompt_text_len",
        "prompt_emotion_ids",
        "prompt_intensity_ids",
        "prompt_text_token",
        "prompt_emo_ids",
        "prompt_inten_ids",
    }
    assert dead.isdisjoint(out.keys()), f"死字段仍在: {dead & set(out.keys())}"

    # 声学 prompt 键必须保留（tts 真正消费）
    for k in (
        "flow_prompt_speech_token",
        "prompt_speech_feat",
        "flow_embedding",
        "llm_prompt_speech_token",
        "llm_embedding",
    ):
        assert k in out, f"声学 prompt 键 {k} 被误删"

    # target 端三件套必须保留
    for k in ("text", "emotion_ids", "intensity_ids"):
        assert k in out, f"target 字段 {k} 缺失"

    # 死代码路径不应触发
    assert fake_frontend._extract_text_token_calls["count"] == 0, (
        "_extract_text_token 被调用 —— prompt 端纯文本路径未删除"
    )
