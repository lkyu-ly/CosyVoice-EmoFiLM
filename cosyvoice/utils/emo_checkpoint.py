"""Emo-FiLM checkpoint 边界与参数身份。"""
import hashlib
from typing import Mapping

import torch


_EMOTION_PREFIXES = (
    "emotion_encoder.",
    "emotion_adapter.",
    "emotion_classifier.",
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
    """加载基础 checkpoint，只允许新版情感模块缺失。"""
    expected = set(model.state_dict().keys())
    actual = _state_keys(state)
    missing = expected - actual
    unexpected = actual - expected
    disallowed_missing = {
        key for key in missing
        if not key.startswith(_EMOTION_PREFIXES)
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
