"""Emo-FiLM checkpoint 边界与参数身份（活跃主线权威）。

本模块是 Emo-FiLM 的单一活跃 checkpoint 加载器（ADR-0020 扁平化）。允许缺失的
前缀与活跃 ``Qwen2LM_Emotion`` 拓扑一致：FiLM（``emotion_encoder`` /
``emotion_adapter``）+ 下游监督任务头（``emotion_head`` / ``arousal_head``）
允许在 base CosyVoice2 ``llm.pt`` 上缺失（这些是随机初始化、base ckpt 不含的
新增模块）。历史 v1 的 ``emotion_classifier.`` 前缀已随输入端 classifier 一并
从活跃代码删除（反模式；仅存于 git 基线锚点 ``9c6d84b``）。
"""
import hashlib
from typing import Mapping

import torch


#: base checkpoint 加载时允许缺失的顶层模块前缀（活跃 ``Qwen2LM_Emotion`` 拓扑）。
#: FiLM + 下游 emotion/arousal 任务头允许缺失；backbone / decoder / embedding
#: 缺失或任何多余键 → 失败。
ALLOWED_MISSING_PREFIXES = (
    "emotion_encoder.",
    "emotion_adapter.",
    "emotion_head.",
    "arousal_head.",
)


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _state_keys(state):
    return set(state.keys())


def _raise_mismatch(kind, missing, unexpected):
    parts = []
    if missing:
        parts.append(f"missing keys: {sorted(missing)}")
    if unexpected:
        parts.append(f"unexpected keys: {sorted(unexpected)}")
    raise RuntimeError(f"{kind} checkpoint schema mismatch; " + "; ".join(parts))


def load_base_state(model, state: Mapping[str, torch.Tensor]):
    """加载基础 checkpoint，只允许活跃情感/任务头模块缺失。"""
    expected = set(model.state_dict().keys())
    actual = _state_keys(state)
    missing = expected - actual
    unexpected = actual - expected
    disallowed_missing = {
        key for key in missing
        if not key.startswith(ALLOWED_MISSING_PREFIXES)
    }
    if disallowed_missing or unexpected:
        _raise_mismatch("base", disallowed_missing, unexpected)
    result = model.load_state_dict(dict(state), strict=False)
    if result.unexpected_keys:
        _raise_mismatch("base", set(), set(result.unexpected_keys))
    return result


def load_trained_state(model, state: Mapping[str, torch.Tensor]):
    """严格加载训练后 checkpoint，缺失和多余键均失败。"""
    expected = set(model.state_dict().keys())
    actual = _state_keys(state)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        _raise_mismatch("trained", missing, unexpected)
    return model.load_state_dict(dict(state), strict=True)


def hash_model_state(model) -> str:
    """按 key、dtype、shape 和连续 tensor bytes 计算 state-dict SHA-256。"""
    digest = hashlib.sha256()
    for key, tensor in sorted(_unwrap_model(model).state_dict().items()):
        if not torch.is_tensor(tensor):
            raise TypeError(f"state entry {key!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        # 直接按连续存储的 uint8 视图读取，兼容 bfloat16 等无法直接
        # 转换为 NumPy 的 dtype，同时保留原始 dtype 的字节表示。
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()
