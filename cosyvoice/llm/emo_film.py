"""Emo-FiLM 核心模块: EmotionEncoder + FiLMLayer。

来源: Emo_PA cosyvoice/llm_emo/llm_emo.py:25-101，修正 alpha 不生效问题。
"""
import torch
import torch.nn as nn


class EmotionEncoder(nn.Module):
    """双 embedding 相加的情感编码器。"""
    def __init__(self, emotion_vocab_size: int, intensity_vocab_size: int, model_dim: int):
        super().__init__()
        self.emotion_embedding = nn.Embedding(emotion_vocab_size, model_dim)
        self.intensity_embedding = nn.Embedding(intensity_vocab_size, model_dim)

    def forward(self, emotion_ids: torch.Tensor, intensity_ids: torch.Tensor) -> torch.Tensor:
        return self.emotion_embedding(emotion_ids) + self.intensity_embedding(intensity_ids)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h̃ = γ ⊙ x + β。alpha 为配置占位，不参与有效计算。"""
    def __init__(self, model_dim: int):
        super().__init__()
        self.projection = nn.Linear(model_dim, model_dim * 2)
        # 恒等初始化
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        with torch.no_grad():
            self.projection.bias[:model_dim].fill_(1.0)
            self.projection.bias[model_dim:].fill_(0.0)

    def forward(self, text_features: torch.Tensor, emotion_features: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.projection(emotion_features)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return gamma * text_features + beta
