"""QwenTokenizer_Emotion: 带情感标签解析的 Qwen tokenizer。

标签格式: <emotion type='hap' intensity='high'>text</emotion>
输出: text_token, emotion_ids, intensity_ids (等长三元组)
"""
import re
import torch
from transformers import AutoTokenizer


class QwenTokenizer_Emotion:
    EMOTIONS = ["ang", "hap", "neu", "sad", "sur"]
    INTENSITIES = ["low", "medium", "high"]
    DEFAULT_EMOTION = "neu"
    DEFAULT_INTENSITY = "low"
    EMOTION_TAG_PATTERN = r"<emotion type='(\w+)' intensity='(\w+)'>(.*?)</emotion>"

    def __init__(self, token_path: str, skip_special_tokens: bool = True,
                 lowercase_text: bool = True, strict: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

        # 添加情感控制 token 到 special tokens（不改 Qwen 词表大小）
        emotion_control_tokens = []
        for emo in self.EMOTIONS:
            for inten in self.INTENSITIES:
                emotion_control_tokens.append(f"<emotion type='{emo}' intensity='{inten}'>")
        emotion_control_tokens.append("</emotion>")

        original_additional = [
            "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
            "[breath]", "<strong>", "</strong>", "[noise]",
            "[laughter]", "[cough]", "[clucking]", "[accent]", "[quick_breath]",
            "<laughter>", "</laughter>", "[hissing]", "[sigh]",
            "[vocalized-noise]", "[lipsmack]", "[mn]",
        ]
        self.tokenizer.add_special_tokens({
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "additional_special_tokens": original_additional + emotion_control_tokens,
        })

        self.skip_special_tokens = skip_special_tokens
        self.lowercase_text = lowercase_text
        self.strict = strict
        self.emotion_to_id = {emo: i + 1 for i, emo in enumerate(self.EMOTIONS)}
        self.intensity_to_id = {intensity: i + 1 for i, intensity in enumerate(self.INTENSITIES)}
        self.emotion_pattern = re.compile(self.EMOTION_TAG_PATTERN)

    def encode_plus(self, text: str, **kwargs) -> dict:
        text_token_ids, emotion_ids, intensity_ids = [], [], []
        last_end = 0

        for match in self.emotion_pattern.finditer(text):
            unmatched = text[last_end:match.start()]
            if unmatched:
                self._process_segment(unmatched, self.DEFAULT_EMOTION, self.DEFAULT_INTENSITY,
                                      text_token_ids, emotion_ids, intensity_ids)
            emo, inten, content = match.groups()
            if content:
                self._process_segment(content, emo, inten,
                                      text_token_ids, emotion_ids, intensity_ids)
            last_end = match.end()

        remaining = text[last_end:]
        if remaining:
            self._process_segment(remaining, self.DEFAULT_EMOTION, self.DEFAULT_INTENSITY,
                                  text_token_ids, emotion_ids, intensity_ids)

        # 严格模式：检查是否有未闭合标签
        if self.strict and "<emotion" in text:
            opens = text.count("<emotion")
            closes = text.count("</emotion>")
            if opens != closes:
                raise ValueError(f"Mismatched emotion tags: {opens} open, {closes} closing in: {text[:100]}...")

        return {
            "text_token": torch.tensor(text_token_ids, dtype=torch.long),
            "emotion_ids": torch.tensor(emotion_ids, dtype=torch.long),
            "intensity_ids": torch.tensor(intensity_ids, dtype=torch.long),
        }

    def _process_segment(self, text_segment, emo, inten, text_token_ids, emotion_ids, intensity_ids):
        stripped = text_segment.strip()
        if not stripped:
            return
        text_to_tokenize = text_segment.lower() if self.lowercase_text else text_segment
        tokenized = self.tokenizer.encode(text_to_tokenize, add_special_tokens=False)
        n_tokens = len(tokenized)
        if n_tokens == 0:
            return
        emo_id = self.emotion_to_id.get(emo, self.emotion_to_id[self.DEFAULT_EMOTION])
        inten_id = self.intensity_to_id.get(inten, self.intensity_to_id[self.DEFAULT_INTENSITY])
        text_token_ids.extend(tokenized)
        emotion_ids.extend([emo_id] * n_tokens)
        intensity_ids.extend([inten_id] * n_tokens)

    def encode(self, text, **kwargs):
        """纯文本编码（不解析情感标签），兼容 CosyVoice 原有 pipeline。"""
        tokens = self.tokenizer([text], return_tensors="pt")
        return tokens["input_ids"][0].cpu().tolist()

    def decode(self, tokens, **kwargs):
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=self.skip_special_tokens)[0]


def get_emo_tokenizer(token_path: str, skip_special_tokens: bool = True) -> QwenTokenizer_Emotion:
    """factory function for YAML: !name:cosyvoice.tokenizer.emo_tokenizer.get_emo_tokenizer"""
    return QwenTokenizer_Emotion(token_path=token_path, skip_special_tokens=skip_special_tokens)
