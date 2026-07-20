"""Emo-FiLM 推理前端: 扩展 CosyVoiceFrontEnd 支持情感标签解析。"""
from pathlib import Path

import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd


class CosyVoiceFrontEnd_Emotion(CosyVoiceFrontEnd):
    """在 zero-shot 前端基础上新增 emotion_ids/intensity_ids 输出。"""

    _PROMPT_CONDITIONING_KEYS = (
        "prompt_text",
        "prompt_text_len",
        "llm_prompt_speech_token",
        "llm_prompt_speech_token_len",
        "flow_prompt_speech_token",
        "flow_prompt_speech_token_len",
        "prompt_speech_feat",
        "prompt_speech_feat_len",
        "llm_embedding",
        "flow_embedding",
    )

    def _prompt_cache_key(self, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id):
        """Return the identity of the prompt-side zero-shot conditioning."""
        prompt_path = str(Path(prompt_wav).expanduser().resolve())
        return prompt_path, prompt_text, resample_rate, zero_shot_spk_id

    def _frontend_zero_shot_prompt(self, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id):
        """Prepare or reuse prompt-side conditioning for Emo-FiLM inference."""
        cache = getattr(self, "_prompt_conditioning_cache", None)
        if cache is None:
            cache = self._prompt_conditioning_cache = {}

        key = self._prompt_cache_key(prompt_text, prompt_wav, resample_rate, zero_shot_spk_id)
        prompt_conditioning = cache.get(key)
        if prompt_conditioning is None:
            model_input = self.frontend_zero_shot(
                "", prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
            )
            prompt_conditioning = {
                name: model_input[name]
                for name in self._PROMPT_CONDITIONING_KEYS
            }
            cache[key] = prompt_conditioning
        return dict(prompt_conditioning)

    def _extract_emo_text_token(self, text_with_emo):
        """使用 emo tokenizer 解析带标签文本，返回 (text_token, emotion_ids, intensity_ids)。"""
        result = self.tokenizer.encode_plus(text_with_emo)
        text_token = result["text_token"].unsqueeze(0).to(self.device)
        emotion_ids = result["emotion_ids"].unsqueeze(0).to(self.device)
        intensity_ids = result["intensity_ids"].unsqueeze(0).to(self.device)
        return text_token, emotion_ids, intensity_ids

    def frontend_emo_film(self, tts_text_with_emo, prompt_text, prompt_wav, resample_rate=24000, zero_shot_spk_id=''):
        """情感推理前端: 在 standard zero-shot 基础上增加 emotion 字段。

        prompt 段默认情感 neu(3)/low(1)（spec 11.4）。
        """
        # 1. 获取 prompt 端的 embedding 和 speech_token/feat；相同 prompt 只提取一次
        model_input = self._frontend_zero_shot_prompt(
            prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
        )

        # 2. 用 emo tokenizer 处理带标签的目标文本
        text_token, emotion_ids, intensity_ids = self._extract_emo_text_token(tts_text_with_emo)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)

        # 3. prompt 文本用纯文本 tokenizer + 默认情感标签
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_emo_ids = torch.full((1, prompt_text_token.shape[1]), 3, dtype=torch.long).to(self.device)  # neu=3
        prompt_inten_ids = torch.full((1, prompt_text_token.shape[1]), 1, dtype=torch.long).to(self.device)  # low=1

        model_input["text"] = text_token
        model_input["text_len"] = text_token_len
        model_input["emotion_ids"] = emotion_ids
        model_input["emotion_ids_len"] = torch.tensor([emotion_ids.shape[1]], dtype=torch.int32).to(self.device)
        model_input["intensity_ids"] = intensity_ids
        model_input["intensity_ids_len"] = torch.tensor([intensity_ids.shape[1]], dtype=torch.int32).to(self.device)
        model_input["prompt_emotion_ids"] = prompt_emo_ids
        model_input["prompt_intensity_ids"] = prompt_inten_ids

        return model_input
