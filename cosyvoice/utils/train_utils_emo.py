"""Emo-FiLM 训练工具: 冻结策略、optimizer 与最小 checkpoint 生命周期。

DDP 兼容（核实 train_utils.py:100 `DistributedDataParallel(model, find_unused_parameters=True)`）：
- freeze 应在 wrap_cuda_model **之前** 调用（裸 Module），named_parameters 不带 'module.' 前缀；
- optimizer 应在 wrap_cuda_model **之后** 调用，但用 `model.module.named_parameters()` 取参数名。

无论何时调用，本模块都会自动 unwrap DDP 包装，使用 'module.' 前缀剥离后的真实模块名。
"""
import logging
import os
import torch.optim as optim
import torch


EMOFILM_TRAINABLE_MODULES = (
    "emotion_encoder",
    "emotion_adapter",
    "llm_decoder",
)


def _unwrap_model(model):
    """剥离 DDP 包装，返回底层 Module。

    DDP 包装后 model.named_parameters() 名字带 'module.' 前缀；
    本 helper 统一返回裸 Module，使外部代码不感知 DDP。
    """
    while hasattr(model, "module"):
        model = model.module
    return model


def _matches_module_name(param_name, module_name):
    """精确模块前缀匹配，避免子串歧义。

    匹配规则：param_name == module_name 或 param_name.startswith(module_name + '.')
    例：module_name='llm_decoder' 匹配 'llm_decoder.weight' 但不匹配 'llm_decoder_backup.weight'。
    """
    return param_name == module_name or param_name.startswith(module_name + ".")


def freeze_all_except(model, modules_to_unfreeze):
    """冻结所有参数，只解冻指定模块（精确前缀匹配）。返回可训练参数总数。

    Args:
        model: 裸 Module 或 DDP 包装后的 Module（自动 unwrap）
        modules_to_unfreeze: list[str]，需要解冻的模块名（顶层属性名）

    Returns:
        n_trainable: int，解冻后可训练参数总数

    spec 9.4 要求：
    - 必须打印可训练参数数 / 总参数数 / 占比
    - 必须列出每个被解冻的 module name
    - 兼容 DDP 包装前后
    """
    bare_model = _unwrap_model(model)

    requested = set(modules_to_unfreeze)
    for name, p in bare_model.named_parameters():
        p.requires_grad = any(
            _matches_module_name(name, tm) for tm in EMOFILM_TRAINABLE_MODULES
        ) and any(
            _matches_module_name(name, tm) for tm in requested
        )
        if _matches_module_name(name, "emotion_classifier"):
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in bare_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in bare_model.parameters())
    pct = 100 * n_trainable / n_total if n_total > 0 else 0
    logging.info(f"[freeze] Trainable params: {n_trainable:,} / {n_total:,} ({pct:.2f}%)")
    logging.info(f"[freeze] Unfrozen modules: {modules_to_unfreeze}")

    # 列出实际被解冻的参数名（前 10 条 + 总数），便于 DDP find_unused_parameters 调试
    trainable_names = [n for n, p in bare_model.named_parameters() if p.requires_grad]
    logging.info(f"[freeze] Trainable parameter names (first 10 of {len(trainable_names)}):")
    for n in trainable_names[:10]:
        logging.info(f"  {n}")

    return n_trainable


def init_optimizer_emo(model, configs):
    """构建 optimizer，按 new_params vs base 分组 lr。

    configs['train_conf']['optim_conf']['new_params_lr'] 用于 emotion 模块，
    configs['train_conf']['optim_conf']['lr'] 用于其它 requires_grad 参数。

    自动 unwrap DDP 包装。emotion_module_names 用精确前缀匹配（同 freeze_all_except）。
    """
    conf = configs["train_conf"]
    new_params_lr = conf["optim_conf"].get("new_params_lr", conf["optim_conf"]["lr"])
    base_lr = conf["optim_conf"]["lr"]

    bare_model = _unwrap_model(model)
    emotion_module_names = set(EMOFILM_TRAINABLE_MODULES)

    new_params = []
    base_params = []
    for name, p in bare_model.named_parameters():
        if not p.requires_grad:
            continue
        if _matches_module_name(name, "emotion_classifier"):
            continue
        if any(_matches_module_name(name, em) for em in emotion_module_names):
            new_params.append(p)
        else:
            base_params.append(p)

    param_groups = []
    if new_params:
        param_groups.append({"params": new_params, "lr": new_params_lr, "name": "emotion_new"})
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr, "name": "base"})

    if conf["optim"] == "adam":
        optimizer = optim.Adam(param_groups)
    elif conf["optim"] == "adamw":
        optimizer = optim.AdamW(param_groups)
    else:
        raise ValueError(f"unknown optimizer: {conf['optim']}")

    logging.info(f"[optimizer] {len(param_groups)} param groups:")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        logging.info(f"  {pg['name']}: {len(pg['params'])} tensors, {n_params:,} params, lr={pg['lr']}")
    return optimizer


def _model_state_dict(model):
    bare_model = _unwrap_model(model)
    return {
        key: value.detach().cpu()
        for key, value in bare_model.state_dict().items()
    }


def save_latest_checkpoint(model, model_dir, epoch, step):
    """原子覆盖唯一训练中间 checkpoint。"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None
    os.makedirs(model_dir, exist_ok=True)
    latest_path = os.path.join(model_dir, "latest.pt")
    temp_path = os.path.join(
        model_dir, f".latest.{os.getpid()}.{id(model)}.tmp"
    )
    payload = _model_state_dict(model)
    payload.update({"epoch": int(epoch), "step": int(step)})
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, latest_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return latest_path


def finalize_latest_checkpoint(model_dir):
    """将 latest.pt 原子收口为 final.pt，并移除 latest.pt。"""
    latest_path = os.path.join(model_dir, "latest.pt")
    final_path = os.path.join(model_dir, "final.pt")
    if not os.path.isfile(latest_path):
        raise FileNotFoundError(latest_path)
    os.replace(latest_path, final_path)
    return final_path
