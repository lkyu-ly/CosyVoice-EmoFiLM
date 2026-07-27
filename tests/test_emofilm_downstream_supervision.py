"""Ticket 02（experiment-readiness）— 下游监督开关 ``downstream_supervision`` 测试。

把 B1 的静默降级（``_batch_has_spans`` 恒 False → 静默 loss_tts only）改成由
``downstream_supervision`` 显式裁决：

  - ``'disabled'`` + 无 span → 正常 ``loss_tts`` only（FiLM-only 实验口径）
  - ``'enabled'``  + 无 span → ``RuntimeError``（防监督头未接线被静默吞）
  - 有 span（无论开关）→ 照常计算 head loss（span 在就算）

CPU fake-backbone，无需 GPU。复用 ``test_emofilm_downstream_heads`` 的 helper（DRY）。
"""
from __future__ import annotations

import pytest
import torch

from tests.test_emofilm_downstream_heads import (
    _add_one_span,
    _base_batch,
    _make_model,
)


def test_default_is_disabled():
    """构造默认 downstream_supervision='disabled'（本次实验口径）。"""
    model = _make_model()
    assert model.downstream_supervision == "disabled"


def test_invalid_value_rejected():
    """非法值在构造时即 ValueError（防止拼写错误静默成 disabled）。"""
    with pytest.raises(ValueError, match="downstream_supervision"):
        _make_model(downstream_supervision="yes")


def test_disabled_no_span_returns_tts_only():
    """disabled + 无 span → loss_tts only，不报错、无 head loss 键。"""
    model = _make_model(downstream_supervision="disabled")
    out = model.forward(_base_batch(), torch.device("cpu"))
    assert "loss_tts" in out and out["loss_tts"] is not None
    assert "loss_emotion" not in out
    assert "loss_intensity" not in out


def test_enabled_no_span_raises():
    """enabled + 无 span → RuntimeError（B1 防静默降级的核心断言）。"""
    model = _make_model(downstream_supervision="enabled")
    with pytest.raises(RuntimeError, match="downstream_supervision"):
        model.forward(_base_batch(), torch.device("cpu"))


def test_disabled_with_span_still_computes_heads():
    """disabled 不禁止有 span 时算 head loss（span 在就算，开关只控无 span 行为）。"""
    model = _make_model(downstream_supervision="disabled")
    batch = _add_one_span(_base_batch(), tok_start=1, tok_end=3)
    out = model.forward(batch, torch.device("cpu"))
    assert "loss_emotion" in out
    assert "loss_intensity" in out
