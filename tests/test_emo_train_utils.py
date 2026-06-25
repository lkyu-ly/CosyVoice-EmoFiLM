"""freeze + optimizer 单测。"""
import torch
import torch.nn as nn
from cosyvoice.utils.train_utils_emo import freeze_all_except, init_optimizer_emo


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emotion_encoder = nn.Linear(10, 10)
        self.emotion_adapter = nn.Linear(10, 10)
        self.emotion_classifier = nn.Linear(10, 5)
        self.llm_decoder = nn.Linear(10, 20)
        self.llm_backbone = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        )
        self.flow = nn.Linear(10, 10)
        self.hift = nn.Linear(10, 10)


def test_freeze_all_except_unfrozen_modules():
    """裸 Module 模式：精确前缀匹配解冻指定模块。"""
    model = FakeModel()
    target_modules = ["emotion_encoder", "emotion_adapter", "emotion_classifier", "llm_decoder"]
    n_trainable = freeze_all_except(model, target_modules)

    assert n_trainable > 0
    for name, p in model.named_parameters():
        # 用 startswith 精确匹配模块前缀（防止 "llm_decoder_backup" 等误匹配）
        should_train = any(name == tm or name.startswith(tm + ".") for tm in target_modules)
        if should_train:
            assert p.requires_grad is True, f"{name} should be trainable"
        else:
            assert p.requires_grad is False, f"{name} should be frozen"


def test_freeze_returns_count():
    """精确解冻 emotion_encoder 一个模块。"""
    model = FakeModel()
    target_modules = ["emotion_encoder"]
    n_trainable = freeze_all_except(model, target_modules)
    # emotion_encoder: 1 Linear = weight(10,10) + bias(10) = 110 params
    assert n_trainable == 110


def test_freeze_with_ddp_wrapper():
    """DDP 包装后 model.module.named_parameters() 应正常工作。

    核实 train_utils.py:100 `DistributedDataParallel(model, find_unused_parameters=True)`:
    DDP 包装后 named_parameters 会带 'module.' 前缀。
    freeze_all_except 应自动 unwrap（hasattr model.module）。

    本测试用 mock wrapper 模拟 DDP 的 `module` 属性（CPU 环境下无需 init_process_group）。
    """
    class MockDDP(nn.Module):
        """Mock DDP wrapper：仅暴露 .module 属性，复现 DDP 的关键 unwrap 接口。"""
        def __init__(self, module):
            super().__init__()
            self.module = module

    model = FakeModel()
    ddp_wrapped = MockDDP(model)
    target_modules = ["emotion_encoder", "emotion_classifier"]
    n_trainable = freeze_all_except(ddp_wrapped, target_modules)
    # DDP unwrap 后访问 .module 检查 requires_grad
    for name, p in ddp_wrapped.module.named_parameters():
        should_train = any(name == tm or name.startswith(tm + ".") for tm in target_modules)
        assert p.requires_grad is should_train, f"{name}: expected {should_train}, got {p.requires_grad}"


def test_freeze_prefix_match_no_false_positive():
    """子串匹配 'llm_decoder' 不应误匹配 'llm_decoder_backup' 或 'sub_llm_decoder_aux'。"""
    class TrickyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.llm_decoder = nn.Linear(10, 20)
            self.llm_decoder_backup = nn.Linear(10, 20)  # 不应被解冻
            self.sub_llm_decoder_aux = nn.Linear(10,20)  # 不应被解冻（前缀不符）

    model = TrickyModel()
    freeze_all_except(model, ["llm_decoder"])
    for name, p in model.named_parameters():
        if name.startswith("llm_decoder."):  # 仅 llm_decoder 直接子模块
            assert p.requires_grad is True, f"{name} should be trainable"
        else:
            assert p.requires_grad is False, f"{name} should be frozen"


def test_init_optimizer_emo_creates_param_groups():
    """optimizer param_groups 应按 emotion / base 分两组（lr 可不同）。"""
    model = FakeModel()
    freeze_all_except(model, ["emotion_encoder", "emotion_adapter", "emotion_classifier", "llm_decoder"])
    configs = {"train_conf": {
        "optim": "adam",
        "optim_conf": {"lr": 1e-5, "new_params_lr": 1e-5},
    }}
    optimizer = init_optimizer_emo(model, configs)
    assert isinstance(optimizer, torch.optim.Adam)
    # 应该有 1 个 group（4 个 emotion 模块全 train，base 全 frozen → 无 base group）
    # 或 2 个 group（如果未来还有 train 模块如 llm_decoder 与 emotion 同列）
    assert 1 <= len(optimizer.param_groups) <= 2


def test_init_optimizer_emo_distinct_lrs():
    """new_params_lr ≠ base_lr 时两组学习率应分别生效。"""
    model = FakeModel()
    # 故意解冻一个非 emotion 模块以产生 base group
    freeze_all_except(model, ["emotion_encoder", "emotion_adapter", "emotion_classifier", "llm_decoder", "flow"])
    configs = {"train_conf": {
        "optim": "adam",
        "optim_conf": {"lr": 1e-5, "new_params_lr": 5e-5},
    }}
    optimizer = init_optimizer_emo(model, configs)
    groups_by_name = {g["name"]: g for g in optimizer.param_groups}
    assert "emotion_new" in groups_by_name
    assert "base" in groups_by_name
    assert groups_by_name["emotion_new"]["lr"] == 5e-5
    assert groups_by_name["base"]["lr"] == 1e-5
