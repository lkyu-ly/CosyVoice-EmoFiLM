"""Emo-FiLM checkpoint 边界与参数身份（活跃主线权威）。

本模块是 Emo-FiLM 的单一活跃 checkpoint 加载器（ADR-0020 扁平化）。允许缺失的
前缀与活跃 ``Qwen2LM_Emotion`` 拓扑一致：FiLM（``emotion_encoder`` /
``emotion_adapter``）+ 下游监督任务头（``emotion_head`` / ``arousal_head``）+
input-end 句级监督探针（``emotion_classifier``）均为随机新增模块，base
CosyVoice2 ``llm.pt`` 不含，允许在 base 加载时缺失。

``emotion_classifier`` 是训练期专用模块（恒构造、冻结、推理不调用）：trained
加载时同样允许其缺失（旧 disabled 基线 ckpt 不含该键，随机初始化即可）；但
``emotion_head`` / ``arousal_head`` 在 trained 加载时**不允许**缺失（v1 旧制品
防冒充守卫，ADR-0019/0020）。
"""
import hashlib
from typing import Mapping

import torch


#: base checkpoint 加载时允许缺失的顶层模块前缀（活跃 ``Qwen2LM_Emotion`` 拓扑）。
#: FiLM + 下游 emotion/arousal 任务头 + input-end 探针允许缺失；backbone /
#: decoder / embedding 缺失或任何多余键 → 失败。
ALLOWED_MISSING_PREFIXES = (
    "emotion_encoder.",
    "emotion_adapter.",
    "emotion_head.",
    "arousal_head.",
    "emotion_classifier.",
)

#: trained checkpoint 加载时允许缺失的顶层模块前缀（模型有、旧 ckpt 无）。
#: 仅 ``emotion_classifier.``：27-epoch disabled 基线（film_only_longepoch）的
#: final.pt 在冻结探针恒构造重构之前训练，不含该键；加载时随机初始化即可
#: （冻结随机权重对推理零影响）。sentlvl 及未来 ckpt 均应含该键。
#: 刻意不含 ``emotion_head.`` / ``arousal_head.`` —— v1 旧制品缺任务头
#: 必须在 trained 加载时失败（防冒充当前训练产物，ADR-0019/0020）。
TRAINED_ALLOWED_MISSING_PREFIXES = (
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
    """加载基础 checkpoint，只允许新增情感模块缺失、不允许任何多余键。

    - missing（模型有、ckpt 无）：仅允许 ``ALLOWED_MISSING_PREFIXES`` 前缀
      （FiLM / 下游任务头 / input-end 探针，base ``llm.pt`` 不含的新增模块）。
    - 其余 missing 或**任何** unexpected → schema mismatch 失败（base 必须是
      CosyVoice2 ``llm.pt``，不应携带超出模型拓扑的键）。
    """
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
    return model.load_state_dict(dict(state), strict=False)


def load_trained_state(model, state: Mapping[str, torch.Tensor]):
    """严格加载训练后 checkpoint；仅容忍训练期专用模块缺失。

    - missing：仅允许 ``TRAINED_ALLOWED_MISSING_PREFIXES``
      （``emotion_classifier.``，旧 disabled ckpt 不含的冻结探针）。
    - unexpected：任何多余键失败。
    """
    expected = set(model.state_dict().keys())
    actual = _state_keys(state)
    missing = expected - actual
    unexpected = actual - expected
    disallowed_missing = {
        key for key in missing
        if not key.startswith(TRAINED_ALLOWED_MISSING_PREFIXES)
    }
    if disallowed_missing or unexpected:
        _raise_mismatch("trained", disallowed_missing, unexpected)
    return model.load_state_dict(dict(state), strict=False)


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
