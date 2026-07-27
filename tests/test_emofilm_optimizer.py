"""Ticket 07 — EmoFiLM v2 optimizer / scheduler / 训练身份合同 focused 测试（CPU fake）。

核心修复（MAP §2 + brief 07）：
  - v1 ``init_optimizer_emo`` 把全部 trainable（含预训练 ``llm_decoder``）塞进单一
    ``emotion_new`` 组；``base`` 组恒空；``new_params_lr == lr`` 无差异。
    → v2 必须有 3 个稳定命名、互斥、非空、完整覆盖 trainable 的组。
  - v1 ``ConstantLR(optimizer)`` 不传 scheduler_conf；``WarmupLR`` 存在但未用；
    ``warmup_steps:2500`` 是死配置。→ v2 scheduler 实际行为与 resolved config 一致。
  - v1 identity 无 optimizer/scheduler 统计绑定。→ v2 identity 绑定合同 + base ckpt +
    resolved config + param-group 统计 + scheduler 统计 + seed + 输出 ckpt + 源身份。

覆盖（brief 07 DoD A-D）：
  A. 三组非空 / 互斥 / 完整覆盖 trainable / 冻结参数不进组 / 组间 ID 无交集；
  B. 每组 LR/WD 可独立配；
  C. warmup 实际改变前期 LR（step<warmup 时 LR<base_lr）；constant 拒绝残留
     warmup_steps；未知 scheduler/optim 字段启动失败；
  D. ``summarize_optimizer_identity`` 与 optimizer 实际一致；
  E. 短周期 fake 训练验证各组获得非空梯度 + LR 轨迹；
  F. fake-ckpt 往返：scheduler step / 组角色 / identity 一致；
  G. v1 基线锚 git ``9c6d84b``（ADR-0020 扁平化后不再用源码 sha256 锁）：
     ``train_utils_emo.py`` / ``scheduler.py`` 是活跃主线代码可演化；
     测试改为反转语义（断言 v1 反模式已删）+ 签名兼容锁；
     ``write_emofilm_run_identity.py`` 允许新增 v2 函数，锁 v1 入口签名。

CPU fake-backbone 测试（仿 06 的 ``_FakeQwen``）。无需 GPU / 真实权重。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion
from cosyvoice.utils.train_utils_emo import (
    OPTIMIZER_PARAM_GROUPS,
    build_scheduler,
    freeze,
    init_optimizer_emo,
    summarize_optimizer_identity,
    validate_optim_scheduler_conf,
)
from tests._emofilm_fakes import _FakeBackbone, _FakeHF, _FakeQwen

ROOT = Path(__file__).resolve().parent.parent

# ADR-0020 扁平化：v1 源码哈希锁已删除（禁源码哈希标定文件）。train_utils_emo.py
# / scheduler.py 是活跃主线代码，可演化；v1 基线由 git 锚点 ``9c6d84b`` 保证。
# ``tools/write_emofilm_run_identity.py`` 允许新增 v2 函数（MAP §1 identity 行；
# brief 07 "add a v2 function or new module"），但其 v1 入口签名与行为必须不变。
# 因此不对该文件做 sha256 冻结，改为 signature + behavior 测试（见下）。


# ============================================================
# fake backbone：从 tests._emofilm_fakes 复用（_FakeBackbone / _FakeHF /
# _FakeQwen 原先在本文件本地定义，现已 DRY 整合到共享测试辅助模块）。
# ============================================================


class _MixingQwen(_FakeQwen):
    """跨位置混合的 fake backbone：``lm_output = xs + mean(xs, dim=1)``。

    修复身份 fake 的局限：identity backbone 不混合位置 → text 列的
    modulated_text_emb 不影响 speech 列 hidden → FiLM 参数（emotion_encoder /
    emotion_adapter）拿不到梯度。真实 Qwen2Encoder 经 self-attention 跨位置混合，
    故 FiLM 必有梯度。本类通过 mean-mixing 模拟跨位置依赖，使三组都能在短训练中
    获得非零梯度（brief 07 DoD D：fake 训练验证各组获得梯度）。
    """

    def forward(self, xs, xs_lens):
        mixed = xs + xs.mean(dim=1, keepdim=True)
        return mixed, torch.ones(
            xs.shape[0], 1, xs.shape[1], dtype=torch.bool, device=xs.device
        )


def _make_model(speech_token_size=10, llm=None):
    return Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=speech_token_size,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=llm or _FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emotion_head_weight=1.0,
        intensity_head_weight=1.0,
    )


def _base_batch(text_len=3, speech_len=6, speech_token_size=10, B=1):
    return {
        "text_token": torch.tensor([[10 + i for i in range(text_len)]] * B),
        "text_token_len": torch.tensor([text_len] * B, dtype=torch.int32),
        "speech_token": torch.tensor(
            [[i % speech_token_size for i in range(speech_len)]] * B
        ),
        "speech_token_len": torch.tensor([speech_len] * B, dtype=torch.int32),
        "emotion_ids": torch.ones(B, text_len, dtype=torch.long),
        "intensity_ids": torch.ones(B, text_len, dtype=torch.long),
    }


def _add_one_span(batch, tok_start, tok_end, **kwargs):
    """向 batch 注入单样本单 span（与 03 collate_aligned_spans 对齐）。"""
    soft_dist = kwargs.get("soft_dist", [0.0, 1.0, 0.0, 0.0, 0.0])
    arousal = kwargs.get("arousal", 0.5)
    batch.update(
        {
            "span_mask": torch.tensor([[True]]),
            "span_valid": torch.tensor([[kwargs.get("valid", True)]]),
            "span_tok_start": torch.tensor([[tok_start]]),
            "span_tok_end": torch.tensor([[tok_end]]),
            "span_emotion_mask": torch.tensor([[kwargs.get("emotion_mask", True)]]),
            "span_intensity_mask": torch.tensor([[kwargs.get("intensity_mask", True)]]),
            "span_emotion_soft_dist": torch.tensor(
                [[list(soft_dist)]], dtype=torch.float32
            ),
            "span_arousal": torch.tensor([[float(arousal)]], dtype=torch.float32),
            "span_supervision_weight": torch.tensor(
                [[float(kwargs.get("supervision_weight", 1.0))]], dtype=torch.float32
            ),
            "span_control_emotion_id": torch.tensor(
                [[kwargs.get("control_emotion_id", 2)]]
            ),
            "span_control_intensity_id": torch.tensor(
                [[kwargs.get("control_intensity_id", 2)]]
            ),
        }
    )
    return batch


def _three_group_conf(
    *,
    lr_rnc=1e-4,
    lr_dh=1e-4,
    lr_pd=1e-5,
    wd_rnc=0.0,
    wd_dh=0.0,
    wd_pd=0.0,
    scheduler="warmup",
    warmup_steps=10,
):
    """规范的三组 + warmup/constant 配置（非审计最优，工程占位默认）。"""
    conf = {
        "optim": "adam",
        "optim_conf": {
            # 顶层 lr 是死字段（票据 06）；训练只读 groups[role].lr
            "groups": {
                "random_new_condition": {"lr": lr_rnc, "weight_decay": wd_rnc},
                "downstream_heads": {"lr": lr_dh, "weight_decay": wd_dh},
                "pretrained_decoder": {"lr": lr_pd, "weight_decay": wd_pd},
            },
        },
        "scheduler": scheduler,
    }
    if scheduler == "warmup":
        conf["scheduler_conf"] = {"warmup_steps": warmup_steps}
    elif scheduler == "constant":
        pass  # constant 不得携带 warmup_steps
    return conf


# ============================================================
# A. 三组结构不变量（brief 07 DoD A 核心）
# ============================================================


def test_param_group_constants_match_06_attribute_names():
    """确认 06 实际的属性名作为组前缀（emotion_encoder/emotion_adapter
    =random_new_condition; emotion_head/arousal_head=downstream_heads;
    llm_decoder=pretrained_decoder）。"""
    assert OPTIMIZER_PARAM_GROUPS == {
        "random_new_condition": ("emotion_encoder", "emotion_adapter"),
        "downstream_heads": ("emotion_head", "arousal_head"),
        "pretrained_decoder": ("llm_decoder",),
    }


def test_three_groups_each_non_empty():
    """正式 v2 拓扑中三组都存在参数（非空）。"""
    model = _make_model()
    freeze(model)
    opt = init_optimizer_emo(model, _three_group_conf())
    names = {pg["name"] for pg in opt.param_groups}
    assert names == {"random_new_condition", "downstream_heads", "pretrained_decoder"}
    for pg in opt.param_groups:
        assert len(pg["params"]) > 0, f"group {pg['name']} must be non-empty"


def test_groups_are_mutually_exclusive_disjoint_param_ids():
    """组间参数 id 无交集（同一 tensor 不进两个组）。"""
    model = _make_model()
    freeze(model)
    opt = init_optimizer_emo(model, _three_group_conf())
    ids_per_group = {}
    for pg in opt.param_groups:
        ids_per_group[pg["name"]] = {id(p) for p in pg["params"]}
    roles = list(ids_per_group)
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            overlap = ids_per_group[roles[i]] & ids_per_group[roles[j]]
            assert not overlap, (
                f"groups {roles[i]} and {roles[j]} share param ids: {len(overlap)}"
            )


def test_groups_complete_coverage_of_trainable():
    """所有 requires_grad=True 参数恰好被一个组覆盖。"""
    model = _make_model()
    freeze(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = init_optimizer_emo(model, _three_group_conf())
    grouped = [p for pg in opt.param_groups for p in pg["params"]]
    # 恰好覆盖（id 集合相等 + 无重复）
    trainable_ids = [id(p) for p in trainable]
    grouped_ids = [id(p) for p in grouped]
    assert sorted(trainable_ids) == sorted(grouped_ids)
    assert len(grouped_ids) == len(set(grouped_ids)), "duplicate param across groups"


def test_frozen_params_excluded_from_all_groups():
    """冻结参数（llm backbone / speech_embedding / llm_embedding）不进任何组。"""
    model = _make_model()
    freeze(model)
    # 验证 llm / speech_embedding / llm_embedding 被冻结
    frozen_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_names.append(name)
    assert frozen_names, "expected some frozen params after freeze"
    top_frozen = {n.split(".")[0] for n in frozen_names}
    # 主干预冻模块（llm backbone / speech_embedding / llm_embedding）
    assert "llm" in top_frozen
    assert "speech_embedding" in top_frozen
    assert "llm_embedding" in top_frozen
    # 冻结参数 id 不出现在任何组
    opt = init_optimizer_emo(model, _three_group_conf())
    grouped_ids = {id(p) for pg in opt.param_groups for p in pg["params"]}
    for _, p in model.named_parameters():
        if not p.requires_grad:
            assert id(p) not in grouped_ids, "frozen param leaked into a group"


def test_llm_decoder_does_not_clash_with_llm_backbone_prefix():
    """精确前缀匹配：``llm_decoder.*`` 不被 ``llm.*`` 吞掉（关键边界 case）。"""
    model = _make_model()
    freeze(model)
    opt = init_optimizer_emo(model, _three_group_conf())
    # llm_decoder 参数应在 pretrained_decoder 组，不在其他组
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    for pg in opt.param_groups:
        for p in pg["params"]:
            name = id_to_name[id(p)]
            if name.startswith("llm_decoder"):
                assert pg["name"] == "pretrained_decoder"
            elif name.startswith("llm.") or name.startswith("llm_embedding"):
                pytest.fail(f"frozen {name!r} leaked into group {pg['name']!r}")


# ============================================================
# B. 每组独立 LR / weight_decay
# ============================================================


def test_each_group_has_independent_lr_and_weight_decay():
    model = _make_model()
    freeze(model)
    opt = init_optimizer_emo(
        model,
        _three_group_conf(
            lr_rnc=3e-4, lr_dh=5e-4, lr_pd=1e-5,
            wd_rnc=0.01, wd_dh=0.02, wd_pd=0.0,
        ),
    )
    by_role = {pg["name"]: pg for pg in opt.param_groups}
    assert by_role["random_new_condition"]["lr"] == 3e-4
    assert by_role["downstream_heads"]["lr"] == 5e-4
    assert by_role["pretrained_decoder"]["lr"] == 1e-5
    assert by_role["random_new_condition"]["weight_decay"] == 0.01
    assert by_role["downstream_heads"]["weight_decay"] == 0.02
    assert by_role["pretrained_decoder"]["weight_decay"] == 0.0


def test_defaults_are_not_claimed_as_audit_optimal():
    """源码注释声明默认值非审计最优（brief 07 DoD A / MAP §3 optimizer）。"""
    import inspect

    src = inspect.getsource(importlib.import_module("cosyvoice.utils.train_utils_emo"))
    # 至少一处声明非审计最优（中/英均可）
    assert (
        "audit" in src.lower()
        or "审计" in src
        or "not.*optimal" in src.lower()
        or "非.*最优" in src
    ), "defaults must be disclaimed as non-audit-optimal in module source"


# ============================================================
# C. scheduler 合同：warmup / constant / unknown
# ============================================================


def test_warmup_scheduler_actually_lowers_early_lr():
    """warmup 配置在 step<warmup_steps 时 LR 严格小于 base_lr（实际改变前期 LR）。

    WarmupLR.get_lr 用 step_num = last_epoch + 1；warmup_steps=10 时 step_num=1 →
    LR = base * warmup_steps^0.5 * min(1^-0.5, 1 * warmup_steps^-1.5) = base * 0.1。
    base_lrs = scheduler 初始记录的 optimizer param_groups lr（被 WarmupLR.__init__
    之前的值）；pg['lr'] 在 __init__ 后已被覆盖为 warmed-up 值，故用 base_lrs。
    """
    model = _make_model()
    freeze(model)
    conf = _three_group_conf(scheduler="warmup", warmup_steps=10)
    opt = init_optimizer_emo(model, conf)
    sched = build_scheduler(opt, conf)
    # 构造后 last_epoch=0 → step_num=1（warmup 早期）
    cur_lrs = sched.get_last_lr()
    base_lrs = sched.base_lrs  # scheduler 记录的 base LR（init 前的 pg['lr']）
    assert len(cur_lrs) == len(base_lrs)
    for base_lr, cur_lr in zip(base_lrs, cur_lrs):
        assert cur_lr < base_lr, (
            f"warmup must lower early LR: base={base_lr} cur={cur_lr} "
            f"(ratio={cur_lr / base_lr if base_lr else 'n/a'})"
        )
    # 额外：比值应远小于 1（warmup_steps=10 时 ~0.1）
    ratio = cur_lrs[0] / base_lrs[0]
    assert ratio < 0.5, f"warmup early LR ratio should be <0.5, got {ratio}"


def test_constant_scheduler_rejects_residual_warmup_steps():
    """constant 配置若残留 warmup_steps 字段 → 启动失败（validate + build 均拦截）。"""
    model = _make_model()
    freeze(model)
    bad_conf = {
        "optim": "adam",
        "optim_conf": {
            # 顶层 lr 是死字段（票据 06）；此处省略，聚焦测 warmup_steps 残留
            "groups": {
                "random_new_condition": {"lr": 1e-4, "weight_decay": 0.0},
                "downstream_heads": {"lr": 1e-4, "weight_decay": 0.0},
                "pretrained_decoder": {"lr": 1e-5, "weight_decay": 0.0},
            },
        },
        "scheduler": "constant",
        "scheduler_conf": {"warmup_steps": 250},  # 残留 → 必须 fail
    }
    # validate 在启动时拦截
    with pytest.raises(ValueError, match="warmup_steps"):
        validate_optim_scheduler_conf(bad_conf)
    # build_scheduler 内部也调用 validate，同样拦截
    opt = init_optimizer_emo(model, _three_group_conf())
    with pytest.raises(ValueError, match="warmup_steps"):
        build_scheduler(opt, bad_conf)


def test_unknown_scheduler_type_fails():
    model = _make_model()
    freeze(model)
    bad_conf = _three_group_conf()
    bad_conf["scheduler"] = "cosine_unknown"
    opt = init_optimizer_emo(model, _three_group_conf())
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_scheduler(opt, bad_conf)


def test_unknown_optim_field_fails_at_startup():
    """optim_conf 顶层或组级未知字段 → validate 启动失败。"""
    base_conf = _three_group_conf()
    # 顶层未知字段
    bad1 = json.loads(json.dumps(base_conf))
    bad1["optim_conf"]["mystery_field"] = 7
    with pytest.raises(ValueError, match="unknown.*optim_conf.*mystery_field|mystery_field"):
        validate_optim_scheduler_conf(bad1)
    # 组级未知字段
    bad2 = json.loads(json.dumps(base_conf))
    bad2["optim_conf"]["groups"]["random_new_condition"]["momentum"] = 0.9
    with pytest.raises(ValueError, match="unknown.*random_new_condition.*momentum|momentum"):
        validate_optim_scheduler_conf(bad2)
    # 未知组角色
    bad3 = json.loads(json.dumps(base_conf))
    bad3["optim_conf"]["groups"]["phantom_group"] = {"lr": 1e-3, "weight_decay": 0.0}
    with pytest.raises(ValueError, match="unknown.*group.*phantom|phantom_group"):
        validate_optim_scheduler_conf(bad3)


def test_unknown_scheduler_conf_field_fails_at_startup():
    """scheduler_conf 未知字段 → validate 启动失败（与 build_scheduler 一致）。"""
    base_conf = _three_group_conf(scheduler="warmup", warmup_steps=10)
    bad = json.loads(json.dumps(base_conf))
    bad["scheduler_conf"]["decay_rate"] = 0.5
    with pytest.raises(ValueError, match="unknown.*warmup.*decay_rate|decay_rate"):
        validate_optim_scheduler_conf(bad)


def test_validate_conf_accepts_canonical_config():
    """合法配置通过 validate（正向基线）。"""
    validate_optim_scheduler_conf(_three_group_conf())
    validate_optim_scheduler_conf(_three_group_conf(scheduler="constant"))


def test_top_level_optim_conf_lr_is_dead_field_rejected():
    """顶层 optim_conf.lr 是死字段（训练只读 groups[role].lr）→ 启动 hard-fail。

    票据 06：历史顶层 lr 被标记"已消费"但 ``init_optimizer_emo`` 从不读取，
    改它不生效也不报错。现显式拒绝，消除死配置。
    """
    base = _three_group_conf()
    bad = json.loads(json.dumps(base))
    bad["optim_conf"]["lr"] = 1e-5  # 注入顶层 lr → 死字段
    with pytest.raises(ValueError, match="死字段|dead field|optim_conf\\.lr"):
        validate_optim_scheduler_conf(bad)


def test_no_top_level_lr_only_groups_lr_passes():
    """无顶层 lr、仅 groups[role].lr 的合法配置通过校验（正向基线）。

    票据 06 清理后，``_three_group_conf`` 不再含顶层 lr；本测试锁死该不变量。
    """
    conf = _three_group_conf()
    assert "lr" not in conf["optim_conf"], "顶层 lr 不应出现在规范配置"
    validate_optim_scheduler_conf(conf)


# ============================================================
# D. summarize_optimizer_identity 与 optimizer 实际一致
# ============================================================


def test_summarize_matches_optimizer_param_groups():
    model = _make_model()
    freeze(model)
    conf = _three_group_conf(
        lr_rnc=3e-4, lr_dh=5e-4, lr_pd=1e-5,
        wd_rnc=0.01, wd_dh=0.02, wd_pd=0.0,
    )
    opt = init_optimizer_emo(model, conf)
    sched = build_scheduler(opt, conf)
    summary = summarize_optimizer_identity(model, opt, sched, conf)
    assert "param_groups" in summary and "scheduler" in summary
    by_role = {g["role"]: g for g in summary["param_groups"]}
    assert set(by_role) == {
        "random_new_condition", "downstream_heads", "pretrained_decoder",
    }
    # 与 optimizer 实际对照
    opt_by_role = {pg["name"]: pg for pg in opt.param_groups}
    for role, pg in opt_by_role.items():
        stat = by_role[role]
        assert stat["tensor_count"] == len(pg["params"])
        assert stat["param_count"] == sum(p.numel() for p in pg["params"])
        assert stat["initial_lr"] == pytest.approx(pg["lr"])
        assert stat["weight_decay"] == pytest.approx(pg.get("weight_decay", 0.0))
    # scheduler 统计
    assert summary["scheduler"]["type"] in {"WarmupLR", "ConstantLR"}
    if summary["scheduler"]["type"] == "WarmupLR":
        assert summary["scheduler"]["key_params"]["warmup_steps"] == 10


def test_summarize_records_scheduler_key_params():
    model = _make_model()
    freeze(model)
    conf_w = _three_group_conf(scheduler="warmup", warmup_steps=42)
    opt = init_optimizer_emo(model, conf_w)
    sched = build_scheduler(opt, conf_w)
    summary = summarize_optimizer_identity(model, opt, sched, conf_w)
    assert summary["scheduler"]["type"] == "WarmupLR"
    assert summary["scheduler"]["key_params"] == {"warmup_steps": 42}

    conf_c = _three_group_conf(scheduler="constant")
    opt2 = init_optimizer_emo(model, conf_c)
    sched2 = build_scheduler(opt2, conf_c)
    summary2 = summarize_optimizer_identity(model, opt2, sched2, conf_c)
    assert summary2["scheduler"]["type"] == "ConstantLR"
    assert summary2["scheduler"]["key_params"] == {}


# ============================================================
# E. 短周期 fake 训练：每组非空梯度 + LR 轨迹
# ============================================================


def _train_short(model, opt, sched, steps=5, collect_grads=False):
    """跑 N 步 fake 训练，记录每步每组 LR 轨迹；可选收集最后一步每组非零 grad 数。"""
    lr_traj = []  # list of list of lr per group per step
    final_non_zero = None
    for step in range(steps):
        batch = _base_batch(text_len=2, speech_len=4)
        _add_one_span(batch, tok_start=0, tok_end=2, soft_dist=[1, 0, 0, 0, 0], arousal=0.5)
        opt.zero_grad()
        out = model.forward(batch, torch.device("cpu"))
        out["loss"].backward()
        lr_traj.append([pg["lr"] for pg in opt.param_groups])
        # 在 step 前可选收集 grad（backward 后、step 前）
        if collect_grads and step == steps - 1:
            final_non_zero = {}
            for pg in opt.param_groups:
                final_non_zero[pg["name"]] = sum(
                    1 for p in pg["params"]
                    if p.grad is not None and p.grad.abs().sum().item() > 0
                )
        opt.step()
        if sched is not None:
            sched.step()
    return lr_traj, final_non_zero


def test_short_fake_training_produces_grads_in_each_group():
    """短周期 fake 训练验证每组获得非零梯度（brief 07 DoD D）。

    使用 ``_MixingQwen``（跨位置混合）使 FiLM 参数经 text→speech 混合路径获得
    梯度（identity backbone 不混合位置，无法测 FiLM 组梯度；真实 Qwen2Encoder 经
    self-attention 混合，故 FiLM 必有梯度）。
    """
    model = _make_model(llm=_MixingQwen(4))
    freeze(model)
    conf = _three_group_conf(scheduler="warmup", warmup_steps=20)
    opt = init_optimizer_emo(model, conf)
    sched = build_scheduler(opt, conf)
    _, non_zero = _train_short(model, opt, sched, steps=3, collect_grads=True)
    # 每个组至少一个 tensor 有非空 .grad
    assert non_zero is not None
    for role, count in non_zero.items():
        assert count > 0, (
            f"group {role} got no non-zero grads after short training"
        )


def test_lr_trajectory_changes_under_warmup():
    """warmup 下 LR 随 step 改变（轨迹非恒定）。"""
    model = _make_model(llm=_MixingQwen(4))
    freeze(model)
    conf = _three_group_conf(scheduler="warmup", warmup_steps=10)
    opt = init_optimizer_emo(model, conf)
    sched = build_scheduler(opt, conf)
    traj, _ = _train_short(model, opt, sched, steps=4)
    # 每组各步的 LR 序列不应全等
    for gi in range(len(opt.param_groups)):
        lrs = [step[gi] for step in traj]
        assert len(set(lrs)) > 1, (
            f"warmup LR trajectory for group {opt.param_groups[gi]['name']} is flat: {lrs}"
        )


# ============================================================
# F. fake ckpt 往返：scheduler step / 组角色 / identity 一致
# ============================================================


def test_fake_checkpoint_roundtrip_preserves_scheduler_step_and_roles(tmp_path):
    """保存 optimizer + scheduler + identity → 重新构建 → step/角色/identity 一致。"""
    model = _make_model()
    freeze(model)
    conf = _three_group_conf(scheduler="warmup", warmup_steps=15)
    opt = init_optimizer_emo(model, conf)
    sched = build_scheduler(opt, conf)
    # 跑 3 步以推进 scheduler
    _train_short(model, opt, sched, steps=3, collect_grads=False)
    before_step = sched.last_epoch
    before_summary = summarize_optimizer_identity(model, opt, sched, conf)
    before_group_roles = [pg["name"] for pg in opt.param_groups]
    # 保存
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict(),
        "optimizer_group_roles": before_group_roles,
        "optimizer_identity": before_summary,
    }
    ckpt_path = tmp_path / "fake.pt"
    torch.save(ckpt, ckpt_path)
    # 重建
    model2 = _make_model()
    freeze(model2)
    opt2 = init_optimizer_emo(model2, conf)
    sched2 = build_scheduler(opt2, conf)
    loaded = torch.load(ckpt_path, weights_only=False)
    opt2.load_state_dict(loaded["optimizer_state"])
    sched2.load_state_dict(loaded["scheduler_state"])
    # 角色
    after_roles = [pg["name"] for pg in opt2.param_groups]
    assert after_roles == before_group_roles
    # scheduler 步进
    assert sched2.last_epoch == before_step
    # identity 一致
    after_summary = summarize_optimizer_identity(model2, opt2, sched2, conf)
    # 结构性比对（param_count / role / scheduler type 一致）
    before_by_role = {g["role"]: g for g in before_summary["param_groups"]}
    after_by_role = {g["role"]: g for g in after_summary["param_groups"]}
    for role in before_by_role:
        assert before_by_role[role]["tensor_count"] == after_by_role[role]["tensor_count"]
        assert before_by_role[role]["param_count"] == after_by_role[role]["param_count"]
    assert before_summary["scheduler"] == after_summary["scheduler"]


def test_train_identity_json_roundtrip(tmp_path):
    """write_emofilm_train_identity 写出可读回的 identity JSON。"""
    from tools.write_emofilm_run_identity import (
        code_identity,
        write_emofilm_train_identity,
    )

    model = _make_model()
    freeze(model)
    conf = _three_group_conf(scheduler="warmup", warmup_steps=12)
    opt = init_optimizer_emo(model, conf)
    sched = build_scheduler(opt, conf)
    summary = summarize_optimizer_identity(model, opt, sched, conf)

    # fake base ckpt 路径
    base_ckpt = tmp_path / "base.pt"
    torch.save({"w": torch.zeros(2)}, base_ckpt)
    # fake output ckpt 路径
    out_ckpt = tmp_path / "out.pt"
    torch.save(model.state_dict(), out_ckpt)

    identity_path = tmp_path / "identity.json"
    identity = write_emofilm_train_identity(
        identity_path,
        run_kind="train",
        code_root=ROOT,
        contract_dir=None,  # 允许 None（测试无合同目录）
        command="pytest fake",
        seed=1986,
        base_checkpoint=base_ckpt,
        resolved_config=conf,
        optimizer_identity=summary,
        output_checkpoint=out_ckpt,
    )
    # 写出
    assert identity_path.is_file()
    reloaded = json.loads(identity_path.read_text())
    assert reloaded["schema_version"] == 2
    assert reloaded["contract_name"] == "emofilm"
    assert reloaded["seed"] == 1986
    assert reloaded["optimizer_identity"] == summary
    assert reloaded["resolved_config"] == conf
    assert reloaded["base_checkpoint"]["sha256"] == hashlib.sha256(
        base_ckpt.read_bytes()
    ).hexdigest()
    assert "source" in reloaded  # git_head 或 patch_bundle
    # patch-bundle 能力暴露（即使 clean 也应记录 source 结构）
    assert "git_head" in reloaded["source"] or "patch_bundle" in reloaded["source"]


def test_identity_preserves_v1_entry_signature():
    """v1 ``write_run_identity`` 签名未被改变（MAP §0 v1 只读）。"""
    import inspect
    from tools.write_emofilm_run_identity import write_run_identity

    sig = inspect.signature(write_run_identity)
    params = set(sig.parameters)
    # v1 入口必须保留这些参数
    expected = {
        "output_path", "run_kind", "code_root", "contract_dir",
        "command", "seed", "base_checkpoint", "extra",
    }
    assert expected.issubset(params), f"v1 signature lost params: {expected - params}"


# ============================================================
# G. 反转语义锁：v1 反模式已从活跃训练代码删除（ADR-0020）
# ============================================================


def test_v1_dead_optimizer_patterns_removed_from_active_code():
    """v1 死分组（恒空 base 组 / 单一 emotion_new 组）与 freeze_all_except /
    EMOFILM_TRAINABLE_MODULES 反模式必须已从活跃 train_utils_emo.py 删除。

    ADR-0020 禁源码哈希标定文件，改为断言反模式不存在。v1 基线由 git 锚点
    ``9c6d84b`` 保证。
    """
    src = (ROOT / "cosyvoice" / "utils" / "train_utils_emo.py").read_text(
        encoding="utf-8"
    )
    # v1 死分组构造 / 死函数 / 死常量已删（断言 v1 定义性构造语句，注释中的
    # 历史引用不计——ADR-0020 反模式锁针对的是活跃代码不再构造这些死结构）
    assert '"name": "emotion_new"' not in src, "v1 单一 emotion_new 死分组必须删除"
    assert '"name": "base"' not in src, "v1 恒空 base 死分组必须删除"
    assert "new_params_lr" not in src, "v1 死 new_params_lr 字段必须删除"
    assert "def freeze_all_except" not in src, "v1 freeze_all_except 必须由 freeze 取代"
    assert "EMOFILM_TRAINABLE_MODULES" not in src, (
        "v1 EMOFILM_TRAINABLE_MODULES 必须由 OPTIMIZER_PARAM_GROUPS 取代"
    )


def test_v1_identity_writer_signature_and_behavior_preserved():
    """v1 ``write_run_identity`` 入口签名 + 行为不变（brief 07：允许新增 v2 函数，
    但不得改 v1 行为）。检查：签名参数集 + 产出 schema_version=1 + contract_name=emofilm_v1。"""
    import inspect
    from tools.write_emofilm_run_identity import write_run_identity

    sig = inspect.signature(write_run_identity)
    expected_params = {
        "output_path", "run_kind", "code_root", "contract_dir",
        "command", "seed", "base_checkpoint", "extra",
    }
    assert expected_params.issubset(set(sig.parameters))

    # v1 行为：write_run_identity 走 v1 路径（schema_version=1, contract_name=emofilm_v1）
    # 用 fake 合同目录验证（避免依赖真实合同文件）。
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        contract_dir = td_path / "contract"
        provenance = contract_dir / "provenance"
        provenance.mkdir(parents=True)
        (provenance / "contract.json").write_text("{}")
        (provenance / "sources.json").write_text("[]")
        (provenance / "membership.json").write_text("{}")
        out = td_path / "ident.json"
        identity = write_run_identity(
            out,
            run_kind="train",
            code_root=ROOT,
            contract_dir=contract_dir,
            command="pytest v1 behavior",
            seed=42,
        )
        assert identity["schema_version"] == 1
        assert identity["contract_name"] == "emofilm_v1"
        reloaded = json.loads(out.read_text())
        assert reloaded["schema_version"] == 1
        assert reloaded["contract_name"] == "emofilm_v1"


# ============================================================
# H. v2 配置通过 assert_no_dead_config
# ============================================================


def _extract_train_conf_block(yaml_text: str) -> dict:
    """从 HyperPyYAML 配置文本中提取 ``train_conf:`` 段为纯 yaml dict。

    HyperPyYAML 的 ``!apply:`` / ``!new:`` / ``!name:`` 标签会触发对象实例化
    （需真实权重 / 路径），不适合 CPU 合同测试。``train_conf:`` 段不含这些标签
    （只有 scalar / dict），故可用 ``yaml.safe_load`` 单独解析。
    """
    import yaml as _yaml

    lines = yaml_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("train_conf:") and not line.startswith(" "):
            start = i
            break
    assert start is not None, "train_conf: block not found in v2 yaml"
    # 截取到下一个顶层 key（非缩进、非空行）
    block = [lines[start]]
    for line in lines[start + 1:]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        block.append(line)
    parsed = _yaml.safe_load("\n".join(block))
    return parsed["train_conf"]


def test_config_train_conf_has_three_groups_and_scheduler():
    """v2 配置含三组 LR/WD + scheduler 合法字段（非死字段）。

    HyperPyYAML 的 ``!apply:`` / ``!new:`` / ``!name:`` 标签触发对象实例化
    （需真实权重/路径），不适合 CPU 合同测试；``train_conf:`` 段不含这些标签，
    用 ``yaml.safe_load`` 单独解析。``assert_no_dead_config`` 只接受 mapping，
    不通过文本子串检查（避免 ``nsf_alpha`` 等无关字段命中 ``alpha`` 死键）。
    """
    from tools.build_emofilm_contract import assert_no_dead_config

    text = (ROOT / "conf" / "emo_film.yaml").read_text()
    # 三组角色名出现
    for role in ("random_new_condition", "downstream_heads", "pretrained_decoder"):
        assert role in text, f"config missing optimizer group role {role!r}"
    # scheduler = warmup（v2 修复死 warmup）
    assert "scheduler: warmup" in text
    assert "warmup_steps:" in text

    # train_conf 段纯 yaml 解析（不含 !new/!apply/!ref）
    train_conf = _extract_train_conf_block(text)
    assert set(train_conf["optim_conf"]["groups"].keys()) == {
        "random_new_condition", "downstream_heads", "pretrained_decoder",
    }
    for role, gcfg in train_conf["optim_conf"]["groups"].items():
        assert "lr" in gcfg and "weight_decay" in gcfg
    assert train_conf["scheduler"] == "warmup"
    assert train_conf["scheduler_conf"]["warmup_steps"] > 0

    # assert_no_dead_config 对 train_conf 段（mapping）—— 不通过文本子串
    assert_no_dead_config(train_conf)
    assert_no_dead_config(train_conf["optim_conf"])

    # 额外：train_conf 通过 v2 optimizer/scheduler 启动校验
    validate_optim_scheduler_conf(train_conf)
