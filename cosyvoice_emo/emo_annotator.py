"""EmoFiLM WordSequenceModel：768d 输入、5 类情感、3D VAD。"""

import torch
import torch.nn as nn


class WordSequenceModel(nn.Module):
    """实现 EmoFiLM 标注合同要求的有效结构。"""

    INPUT_DIM = 768
    NUM_CLASSES = 5
    NUM_HEADS = 8
    REG_DIM = 3

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
        num_heads: int = NUM_HEADS,
        dropout_rate: float = 0.3,
        reg_dim: int = REG_DIM,
    ):
        super().__init__()
        if (input_dim, num_classes, num_heads, reg_dim) != (
            self.INPUT_DIM,
            self.NUM_CLASSES,
            self.NUM_HEADS,
            self.REG_DIM,
        ):
            raise ValueError(
                "EmoFiLM WordSequenceModel contract requires input_dim=768, "
                "num_classes=5, num_heads=8, reg_dim=3"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.reg_dim = reg_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.classification_head = nn.Linear(input_dim, num_classes)
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, reg_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, padding_mask=None):
        """输入 ``(B,T,768)``，返回 ``(B,5)`` 与 ``(B,3)``。"""
        if x.ndim != 3 or x.shape[-1] != self.INPUT_DIM:
            raise ValueError(f"expected input shape (B,T,{self.INPUT_DIM}), got {tuple(x.shape)}")
        key_mask = padding_mask
        if key_mask is not None and key_mask.shape != x.shape[:2]:
            raise ValueError(
                f"padding_mask must have shape {tuple(x.shape[:2])}, got {tuple(key_mask.shape)}"
            )

        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        if padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = ~padding_mask
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
            pooled = x.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)

        return self.classification_head(pooled), self.regression_head(pooled)
