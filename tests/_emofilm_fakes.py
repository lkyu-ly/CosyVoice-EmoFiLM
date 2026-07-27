"""EmoFiLM 共享测试替身（test-only）：训推模型 backbone fake。

集中存放训推链路供 CPU 合同 / 行为测试使用的确定性 backbone 替身，避免
test-only 实现耦合到生产模块（ADR-0014：只锁外部行为与 schema）。

- ``_FakeBackbone``：最小 backbone（仅 ``embed_tokens``）。
- ``_FakeHF``：最小 HF wrapper（``.model`` 指向 ``_FakeBackbone``）。
- ``_FakeQwen``：恒等 backbone（``forward`` / ``forward_one_step`` 透传），
  满足 ``Qwen2LM_Emotion`` 所需 backbone 接口。不加载真实权重，不需 GPU。

评测 fake（FakeForcedAligner / FakeWerEvaluator / ClipMappedEvaluator）已随
v2 评测模块（eval_local_control / triplet_eval / acoustic_evaluators）一并移除
——评测还原为 baseline ``eval_emo_film``（整体质量 WER/Emo-SIM/DTW）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _FakeBackbone(nn.Module):
    """最小 backbone：仅一个 ``embed_tokens``，满足模型所需的属性探测。"""

    def __init__(self, model_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, model_dim)


class _FakeHF(nn.Module):
    """最小 HF wrapper：``.model`` 指向 ``_FakeBackbone``。"""

    def __init__(self, model_dim):
        super().__init__()
        self.model = _FakeBackbone(model_dim)


class _FakeQwen(nn.Module):
    """恒等 backbone：``forward`` / ``forward_one_step`` 透传 ``xs``。

    用于 CPU 合同测试，满足 ``Qwen2LM_Emotion`` 所需的 backbone 接口
    （``.model.embed_tokens`` / ``forward`` / ``forward_one_step``）。
    """

    def __init__(self, model_dim=4):
        super().__init__()
        self.model = _FakeHF(model_dim)

    def forward_one_step(self, xs, masks=None, cache=None):
        return xs, cache

    def forward(self, xs, xs_lens):
        return xs, torch.ones(
            xs.shape[0], 1, xs.shape[1], dtype=torch.bool, device=xs.device
        )


__all__ = ["_FakeBackbone", "_FakeHF", "_FakeQwen"]
