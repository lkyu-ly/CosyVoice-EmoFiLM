"""Emo-FiLM optimizer / scheduler / 冻结 / 最小 checkpoint 生命周期（活跃主线权威）。

本模块是 EmoFiLM 的单一活跃训练工具权威（ADR-0020 扁平化）。修复历史死配置：

1. **死分组**：历史 ``init_optimizer_emo`` 把全部 trainable（含预训练
   ``llm_decoder``）塞进单一 emotion_new 组；base 组恒空；新旧 LR 无差异。
   → 本模块落实 **3 个稳定命名、互斥、非空、完整覆盖 trainable** 的参数组：
   - ``random_new_condition``：FiLM 随机新增（``emotion_encoder`` / ``emotion_adapter``）
   - ``downstream_heads``：下游监督任务头随机新增（``emotion_head`` / ``arousal_head``）
   - ``pretrained_decoder``：预训练 decoder（``llm_decoder``）

2. **死 warmup**：历史 ``ConstantLR(optimizer)`` 不传 scheduler_conf；
   ``WarmupLR`` 存在但未用；``warmup_steps: 2500`` 是死配置。→ 本模块单一
   scheduler 工厂：warmup 实际改变前期 LR；constant 拒绝残留 warmup_steps；
   未知字段启动失败。

复用 ``cosyvoice/utils/scheduler.py`` 已存在的 ``WarmupLR`` / ``ConstantLR`` 类
（仅 import，不修改它们）。

**默认值声明**：本模块与 ``conf/emo_film.yaml`` 中的 LR / weight_decay 默认值均为
**工程占位默认**，**非静态审计最优**。正式实验需由调参实验确定。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Sequence

import torch
import torch.optim as optim

from cosyvoice.utils.scheduler import ConstantLR, WarmupLR

# ============================================================
# 三稳定命名参数组（角色 → 模块属性前缀）
# ============================================================

#: 角色 → 该组覆盖的顶层模块属性名（来自 ``Qwen2LM_Emotion``）。
OPTIMIZER_PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    # FiLM 随机新增条件模块（保留自 emo_film.py，随机初始化、可训练）
    "random_new_condition": ("emotion_encoder", "emotion_adapter"),
    # 下游监督任务头（随机初始化、可训练；不接收 control ID/loss target）
    "downstream_heads": ("emotion_head", "arousal_head"),
    # 预训练 speech-token decoder（base CosyVoice2 llm.pt 含；续训）
    "pretrained_decoder": ("llm_decoder",),
}

#: 所有 trainable 前缀（三组并集）；``freeze`` 保留这些、冻结其余。
_TRAINABLE_PREFIXES: tuple[str, ...] = tuple(
    prefix for group in OPTIMIZER_PARAM_GROUPS.values() for prefix in group
)

#: 冻结的顶层模块（主干预冻，显式列出便于审计；精确前缀匹配确保
#: ``llm.*`` 不吞掉 ``llm_decoder.*``）。
_FROZEN_PREFIXES: tuple[str, ...] = ("llm", "speech_embedding", "llm_embedding")


# ============================================================
# 已知 / 已消费字段（未知 → 启动失败）
# ============================================================

_KNOWN_OPTIMIZERS = ("adam", "adamw")
_KNOWN_SCHEDULERS = ("warmup", "constant")

#: optim_conf 顶层已消费字段。顶层 ``lr`` **不在**此列——它是死字段
#: （``init_optimizer_emo`` 只读 ``groups[role].lr``），由
#: ``validate_optim_scheduler_conf`` 显式 hard-fail（票据 06）。
_CONSUMED_OPTIM_TOP_FIELDS = frozenset({"groups"})
#: 每个组的已消费字段。
_CONSUMED_GROUP_FIELDS = frozenset({"lr", "weight_decay"})
#: scheduler_conf 按 scheduler 类型已消费字段。``constant`` 必须为空集
#: （残留 ``warmup_steps`` → fail）。
_CONSUMED_SCHEDULER_FIELDS: dict[str, frozenset[str]] = {
    "warmup": frozenset({"warmup_steps"}),
    "constant": frozenset(),
}


# ============================================================
# helpers
# ============================================================


def _unwrap_model(model):
    """剥离 DDP 包装，返回裸 Module。"""
    while hasattr(model, "module"):
        model = model.module
    return model


def _matches_module_name(param_name: str, module_name: str) -> bool:
    """精确模块前缀匹配，避免子串歧义。

    ``"llm_decoder.weight".startswith("llm.")`` 为 False，故 ``llm.*`` 不会吞掉
    ``llm_decoder.*``（关键边界 case）。
    """
    return param_name == module_name or param_name.startswith(module_name + ".")


# ============================================================
# 冻结策略
# ============================================================


def freeze(model) -> int:
    """冻结所有参数，只解冻三组 trainable 模块（精确前缀匹配）。

    Returns:
        n_trainable: 解冻后可训练参数总数。
    """
    bare = _unwrap_model(model)
    for _, p in bare.named_parameters():
        p.requires_grad = False
    for name, p in bare.named_parameters():
        if any(_matches_module_name(name, prefix) for prefix in _TRAINABLE_PREFIXES):
            p.requires_grad = True

    n_trainable = sum(p.numel() for p in bare.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in bare.parameters())
    pct = 100 * n_trainable / n_total if n_total > 0 else 0.0
    logging.info(
        "[freeze] trainable params: %d / %d (%.4f%%)",
        n_trainable,
        n_total,
        pct,
    )
    trainable_names = [n for n, p in bare.named_parameters() if p.requires_grad]
    logging.info("[freeze] trainable parameter names (%d):", len(trainable_names))
    for n in trainable_names[:10]:
        logging.info("  %s", n)
    return n_trainable


# ============================================================
# 配置校验（启动失败 on unknown / 未消费字段）
# ============================================================


def validate_optim_scheduler_conf(conf: Mapping[str, Any]) -> None:
    """启动时拒绝未知 / 未消费的 optim / scheduler 字段。

    检查项：
    - ``conf["optim"]`` ∈ {adam, adamw}；
    - ``conf["optim_conf"]`` 顶层**不得出现** ``lr``（死字段：训练只读
      ``groups[role].lr``，改顶层 lr 不生效也不报错 → 显式 hard-fail，
      票据 06）；
    - ``conf["optim_conf"]`` 其余顶层键 ⊆ ``_CONSUMED_OPTIM_TOP_FIELDS``；
    - ``conf["optim_conf"]["groups"]`` 角色 ⊆ ``OPTIMIZER_PARAM_GROUPS.keys()``；
    - 每组键 ⊆ ``_CONSUMED_GROUP_FIELDS``；
    - ``conf["scheduler"]`` ∈ {warmup, constant}；
    - ``conf["scheduler_conf"]`` 键 ⊆ 该 scheduler 的已消费字段
      （``constant`` 必须为空集）。
    """
    if not isinstance(conf, Mapping):
        raise ValueError("train_conf must be a mapping")

    # optim
    optim = conf.get("optim")
    if optim not in _KNOWN_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer {optim!r}; expected one of {list(_KNOWN_OPTIMIZERS)}"
        )

    optim_conf = conf.get("optim_conf", {})
    if not isinstance(optim_conf, Mapping):
        raise ValueError("optim_conf must be a mapping")
    # 顶层 lr 是死字段：init_optimizer_emo 只读 groups[role].lr，改顶层 lr
    # 不生效也不报错。显式 hard-fail 消除死配置（票据 06）。
    if "lr" in optim_conf:
        raise ValueError(
            "顶层 optim_conf.lr 是死字段（训练只读 groups[role].lr）；"
            "请移除顶层 lr，改用 optim_conf.groups[role].lr 配置每组学习率"
        )
    unknown_top = set(optim_conf.keys()) - _CONSUMED_OPTIM_TOP_FIELDS
    if unknown_top:
        raise ValueError(
            f"unknown optim_conf fields: {sorted(unknown_top)} "
            f"(consumed: {sorted(_CONSUMED_OPTIM_TOP_FIELDS)})"
        )

    groups = optim_conf.get("groups", {})
    if not isinstance(groups, Mapping):
        raise ValueError("optim_conf.groups must be a mapping")
    unknown_roles = set(groups.keys()) - set(OPTIMIZER_PARAM_GROUPS.keys())
    if unknown_roles:
        raise ValueError(
            f"unknown optimizer group roles: {sorted(unknown_roles)} "
            f"(expected: {sorted(OPTIMIZER_PARAM_GROUPS.keys())})"
        )
    # 必须三组全到齐（formal 拓扑）
    missing_roles = set(OPTIMIZER_PARAM_GROUPS.keys()) - set(groups.keys())
    if missing_roles:
        raise ValueError(
            f"missing optimizer group roles: {sorted(missing_roles)} "
            f"(all three required: {sorted(OPTIMIZER_PARAM_GROUPS.keys())})"
        )
    for role, gcfg in groups.items():
        if not isinstance(gcfg, Mapping):
            raise ValueError(f"group {role!r} config must be a mapping")
        unknown_g = set(gcfg.keys()) - _CONSUMED_GROUP_FIELDS
        if unknown_g:
            raise ValueError(
                f"unknown fields in group {role!r}: {sorted(unknown_g)} "
                f"(consumed: {sorted(_CONSUMED_GROUP_FIELDS)})"
            )
        for key in ("lr", "weight_decay"):
            if key not in gcfg:
                raise ValueError(f"group {role!r} missing required field {key!r}")

    # scheduler
    sched = conf.get("scheduler")
    if sched not in _KNOWN_SCHEDULERS:
        raise ValueError(
            f"unknown scheduler {sched!r}; expected one of {list(_KNOWN_SCHEDULERS)}"
        )
    sched_conf = conf.get("scheduler_conf", {}) or {}
    if not isinstance(sched_conf, Mapping):
        raise ValueError("scheduler_conf must be a mapping")
    consumed = _CONSUMED_SCHEDULER_FIELDS[sched]
    unknown_sched = set(sched_conf.keys()) - consumed
    if unknown_sched:
        raise ValueError(
            f"unknown {sched!r} scheduler_conf fields: {sorted(unknown_sched)} "
            f"(consumed by {sched!r}: {sorted(consumed) or '{}'})"
        )
    # warmup 必须有 warmup_steps
    if sched == "warmup":
        if "warmup_steps" not in sched_conf:
            raise ValueError("scheduler=warmup requires scheduler_conf.warmup_steps")
        ws = sched_conf["warmup_steps"]
        if not isinstance(ws, int) or isinstance(ws, bool) or ws <= 0:
            raise ValueError(f"warmup_steps must be a positive int, got {ws!r}")


# ============================================================
# 组覆盖不变量校验（启动时）
# ============================================================


def _validate_group_coverage(
    bare_model,
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> None:
    """启动时校验三组不变量。

    - 每组非空；
    - 组间参数 id 无交集（互斥）；
    - 并集 = 全体 requires_grad=True 参数（完整覆盖）；
    - 无 requires_grad=False 参数混入任何组。
    """
    # 非空
    for role, params in groups.items():
        if not params:
            raise ValueError(
                f"optimizer group {role!r} is empty "
                "(formal topology requires all three groups to have params)"
            )
    # 互斥（id 无交集）
    ids_per_group = {role: {id(p) for p in params} for role, params in groups.items()}
    roles = list(ids_per_group)
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            overlap = ids_per_group[roles[i]] & ids_per_group[roles[j]]
            if overlap:
                raise ValueError(
                    f"optimizer groups {roles[i]!r} and {roles[j]!r} share "
                    f"{len(overlap)} param id(s) (must be mutually exclusive)"
                )
    # 完整覆盖
    grouped_ids = set().union(*ids_per_group.values())
    trainable_ids = {id(p) for p in bare_model.parameters() if p.requires_grad}
    missing = trainable_ids - grouped_ids
    if missing:
        raise ValueError(
            f"{len(missing)} trainable param(s) not covered by any optimizer group"
        )
    extra = grouped_ids - trainable_ids
    if extra:
        raise ValueError(
            f"{len(extra)} param(s) in groups are not requires_grad=True "
            "(frozen params leaked into a group)"
        )


# ============================================================
# optimizer 构建
# ============================================================


def init_optimizer_emo(model, conf: Mapping[str, Any]):
    """构建 optimizer：3 个稳定命名组，每组独立 LR + weight_decay。

    Args:
        model: 裸 Module 或 DDP 包装后的 Module（自动 unwrap）。调用方应先
            ``freeze(model)``；本函数仅消费 ``requires_grad=True`` 参数。
        conf: ``train_conf`` 映射（**非**完整 configs），结构为::

            optim: adam | adamw
            optim_conf:                        # 顶层不得出现 lr（死字段，票据 06）
              groups:
                random_new_condition: {lr, weight_decay}
                downstream_heads:    {lr, weight_decay}
                pretrained_decoder:   {lr, weight_decay}
            scheduler: warmup | constant

    Returns:
        torch.optim.Optimizer（参数组 ``name`` 字段 = 角色名）。

    Raises:
        ValueError: 三组任一为空、组间参数 id 交集、trainable 参数未覆盖、
            冻结参数混入组。
    """
    validate_optim_scheduler_conf(conf)

    bare = _unwrap_model(model)

    # 按角色前缀分桶（仅 requires_grad 参数）
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for role, prefixes in OPTIMIZER_PARAM_GROUPS.items():
        bucket: list[torch.nn.Parameter] = []
        for name, p in bare.named_parameters():
            if not p.requires_grad:
                continue
            if any(_matches_module_name(name, prefix) for prefix in prefixes):
                bucket.append(p)
        groups[role] = bucket

    _validate_group_coverage(bare, groups)

    group_confs = conf["optim_conf"]["groups"]
    param_groups = []
    for role, params in groups.items():
        gcfg = group_confs[role]
        param_groups.append(
            {
                "params": params,
                "lr": float(gcfg["lr"]),
                "weight_decay": float(gcfg["weight_decay"]),
                "name": role,
            }
        )

    if conf["optim"] == "adam":
        optimizer = optim.Adam(param_groups)
    elif conf["optim"] == "adamw":
        optimizer = optim.AdamW(param_groups)
    else:  # pragma: no cover - validate_optim_scheduler_conf 已拦截
        raise ValueError(f"unknown optimizer: {conf['optim']}")

    logging.info("[optimizer] %d param groups:", len(param_groups))
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        logging.info(
            "  %s: %d tensors, %d params, lr=%g, weight_decay=%g",
            pg["name"],
            len(pg["params"]),
            n_params,
            pg["lr"],
            pg["weight_decay"],
        )
    return optimizer


# ============================================================
# scheduler 构建（单一工厂）
# ============================================================


def build_scheduler(optimizer, conf: Mapping[str, Any]):
    """单一 scheduler 工厂。

    - ``scheduler=warmup`` → ``WarmupLR(optimizer, warmup_steps=...)``，
      warmup **实际**改变前期 LR（``get_lr`` 在 step<warmup 时 LR<base_lr）。
    - ``scheduler=constant`` → ``ConstantLR(optimizer)``，且 ``scheduler_conf``
      **不得**残留 ``warmup_steps`` 字段（否则 fail）。
    - 未知 scheduler / 未消费字段 → ValueError。

    复用 ``cosyvoice/utils/scheduler.py`` 的 ``WarmupLR`` / ``ConstantLR`` 类，
    不修改它们。
    """
    validate_optim_scheduler_conf(conf)
    sched = conf["scheduler"]
    sched_conf = conf.get("scheduler_conf", {}) or {}

    if sched == "warmup":
        ws = int(sched_conf["warmup_steps"])
        return WarmupLR(optimizer, warmup_steps=ws)
    if sched == "constant":
        return ConstantLR(optimizer)
    # pragma: no cover - validate_optim_scheduler_conf 已拦截
    raise ValueError(f"unknown scheduler: {sched!r}")


# ============================================================
# CV 早停 + 容忍度（纯状态跟踪，无 IO）
# ============================================================


class EarlyStopTracker:
    """CV 指标早停 + 容忍度跟踪器（纯状态，无 IO）。

    每个 epoch 末用 CV 指标调用 :meth:`update`；当连续 ``patience`` 个 epoch
    无改善（改善 = ``cv_value < best_value - min_delta``）且已达 ``min_epoch``，
    判定应停止。``min_delta`` 是"容忍度"——小于该量的波动不算改善，避免在
    噪声平台上反复刷新 best 而无法触发早停。

    所有 rank 必须以**相同的 cv_value** 调用（单 GPU 训练或确定性 CV），保证
    break 决策跨 rank 一致（否则 DDP 会 hang）。``executor.cv`` 在每个 rank 上
    都对同一 CV 集前向计算 loss，单 GPU 下天然一致。
    """

    def __init__(
        self,
        metric: str = "loss_tts",
        min_delta: float = 0.0,
        patience: int = 0,
        min_epoch: int = 0,
    ):
        self.metric = str(metric)
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.min_epoch = int(min_epoch)
        self.best_value = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0

    def update(self, epoch: int, cv_value: float) -> tuple[bool, bool]:
        """记录该 epoch 的 CV 指标。

        Returns:
            (improved, should_stop): improved 表示相对 best 有突破容忍度的改善；
            should_stop 表示已连续 patience 个 epoch 无改善且达到 min_epoch。
        """
        value = float(cv_value)
        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        # min_epoch 语义=「至少训满多少 epoch」：epoch 为 0-indexed（已训 epoch+1
        # 个），故 epoch+1 >= min_epoch 才允许停（min_epoch=5 ⇒ 训满 0..4 共 5 个后）。
        should_stop = (
            int(epoch) + 1 >= self.min_epoch and self.bad_epochs >= self.patience
        )
        return improved, should_stop


def build_early_stop_tracker(conf: Mapping[str, Any]):
    """从 train_conf 构建 EarlyStopTracker；未启用时返回 None。

    返回 None 时调用方应走原生循环（与无早停基线**完全一致**：不产生 best.pt、
    不 restore-best），保证 canonical 5-epoch 基线行为零回归。

    Args:
        conf: ``train_conf`` 映射。识别字段（均可缺省）::

            early_stop: true | false        # 开关；缺省/false → 返回 None
            early_stop_metric: loss_tts     # 监控的 CV loss 键
            early_stop_min_delta: 0.001     # 容忍度
            early_stop_patience: 5          # 无改善耐心（epoch 数）
            early_stop_min_epoch: 3         # 至少训满多少 epoch 才允许早停
    """
    if not conf.get("early_stop"):
        return None
    return EarlyStopTracker(
        metric=conf.get("early_stop_metric", "loss_tts"),
        min_delta=conf.get("early_stop_min_delta", 0.0),
        # 缺省非零：避免「只写 early_stop: true 漏配耐心」时 patience/min_epoch=0
        # 导致首 epoch 即触发 should_stop（bad=0>=0）只训 1 个 epoch。
        patience=conf.get("early_stop_patience", 5),
        min_epoch=conf.get("early_stop_min_epoch", 1),
    )


# ============================================================
# optimizer / scheduler 身份摘要（绑定到训练 identity）
# ============================================================


def summarize_optimizer_identity(
    model,
    optimizer,
    scheduler,
    conf: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """记录每个 optimizer 参数组的张量数 / 参数数 / 初始 LR / weight_decay +
    scheduler 类型与关键参数。

    此摘要被 ``write_emofilm_train_identity`` 绑定到训练 identity
    （schema_version=2）。
    """
    group_stats = []
    for pg in optimizer.param_groups:
        params = pg["params"]
        group_stats.append(
            {
                "role": pg.get("name", "?"),
                "tensor_count": len(params),
                "param_count": int(sum(p.numel() for p in params)),
                "initial_lr": float(pg["lr"]),
                "weight_decay": float(pg.get("weight_decay", 0.0)),
            }
        )

    # scheduler 类型 + 关键参数（用 isinstance 而非类名字符串，避免 scheduler
    # 被子类化时类型描述失真）
    sched_type = type(scheduler).__name__
    key_params: dict[str, Any] = {}
    if isinstance(scheduler, WarmupLR):
        key_params["warmup_steps"] = int(getattr(scheduler, "warmup_steps", -1))
    # ConstantLR 无关键参数（key_params 保持空 dict）

    return {
        "param_groups": group_stats,
        "scheduler": {
            "type": sched_type,
            "key_params": key_params,
        },
    }


# ============================================================
# 最小 checkpoint 生命周期（latest → final 原子收口）
# ============================================================


def _model_state_dict(model):
    bare_model = _unwrap_model(model)
    return {key: value.detach().cpu() for key, value in bare_model.state_dict().items()}


def save_latest_checkpoint(model, model_dir, epoch, step):
    """原子覆盖唯一训练中间 checkpoint。"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None
    os.makedirs(model_dir, exist_ok=True)
    latest_path = os.path.join(model_dir, "latest.pt")
    temp_path = os.path.join(model_dir, f".latest.{os.getpid()}.{id(model)}.tmp")
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


__all__ = [
    "OPTIMIZER_PARAM_GROUPS",
    "freeze",
    "validate_optim_scheduler_conf",
    "init_optimizer_emo",
    "build_scheduler",
    "build_early_stop_tracker",
    "EarlyStopTracker",
    "summarize_optimizer_identity",
    "save_latest_checkpoint",
    "finalize_latest_checkpoint",
]
