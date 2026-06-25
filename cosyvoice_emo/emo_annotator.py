"""WordSequenceModel: 轻量 Transformer 词级情感分类器。

复刻 Emo_PA annotate_data/model.py，适配 emotion2vec_plus_large (1024维)。
输入: per-word frame sequence (B,T,1024) + padding_mask (B,T)
输出: class_logits (B,5) + vad_pred (B,3) [sigmoid: 0-1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WordSequenceModel(nn.Module):
    def __init__(self, input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads,
            dropout=dropout_rate, batch_first=True,
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
            nn.Linear(input_dim, 3),
            nn.Sigmoid(),
        )

    def forward(self, x, padding_mask=None):
        """x (B,T,D), padding_mask (B,T) bool=True means ignore."""
        if padding_mask is not None and x.dim() == 2:
            key_mask = padding_mask.view(-1)
        elif padding_mask is not None:
            key_mask = padding_mask
        else:
            key_mask = None
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if padding_mask is not None:
            mask = ~padding_mask
            lengths = mask.sum(dim=1, keepdim=True)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
            pooled = x.sum(dim=1) / lengths.clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        class_logits = self.classification_head(pooled)
        vad_pred = self.regression_head(pooled)
        return class_logits, vad_pred
