"""Task 5 — 训练入口 write_training_identity / update_training_identity v2 测试。

核心修复（brief 05 §B + MAP §2 identity 段）：
  - v1 ``write_training_identity`` 调 ``write_run_identity``（schema_version=1，
    contract_name="emofilm_v1"），未记录 optimizer 分组 / resolved config /
    patch_bundle。→ v2 改调 ``write_emofilm_train_identity``（schema_version=2，
    contract_name="emofilm"），绑定 optimizer_identity + resolved_config + patch_bundle。
  - v1 ``update_training_identity`` 写 v1 extra。→ v2 改写 v2 extra（final_checkpoint
    + sha256）；若读到旧 v1 identity，raise 明确提示重训。
  - v1 ``write_run_identity`` 8 参数签名兼容锁不变（ADR-0020）。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# helpers：合成 model + optimizer（带命名组）+ scheduler
# ============================================================


class _FakeModel(nn.Module):
    """简单模型：两个子模块（模拟 emotion_encoder + backbone）。"""

    def __init__(self):
        super().__init__()
        self.emotion_encoder = nn.Linear(4, 4)
        self.backbone = nn.Linear(4, 4)


def _make_named_optimizer(model):
    """构造带 ``name`` 键的三组 Adam（模拟 init_optimizer_emo 的产出）。"""
    groups = [
        {
            "name": "random_new_condition",
            "params": list(model.emotion_encoder.parameters()),
            "lr": 1e-4,
            "weight_decay": 0.01,
        },
        {
            "name": "downstream_heads",
            "params": list(model.backbone.parameters()),
            "lr": 5e-4,
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.Adam(groups)


def _make_scheduler(optimizer):
    from cosyvoice.utils.scheduler import ConstantLR

    return ConstantLR(optimizer)


def _init_git_repo(td: Path) -> Path:
    """初始化一个临时 git 仓库，做一个干净 commit。"""
    td.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=td, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=td, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=td, check=True, capture_output=True,
    )
    (td / "README.md").write_text("test repo\n")
    subprocess.run(["git", "add", "-A"], cwd=td, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=td, check=True, capture_output=True,
    )
    return td


@pytest.fixture
def fake_model_optim_sched():
    """提供 (model, optimizer, scheduler) 三元组。"""
    model = _FakeModel()
    optimizer = _make_named_optimizer(model)
    scheduler = _make_scheduler(optimizer)
    return model, optimizer, scheduler


# ============================================================
# A. write_training_identity → v2 schema
# ============================================================


class TestWriteTrainingIdentityV2:
    """A. write_training_identity 产出 v2 schema（emofilm / schema_version=2）。"""

    def test_training_identity_is_v2_schema(
        self, tmp_path, fake_model_optim_sched
    ):
        """train_identity.json 的 contract_name="emofilm" / schema_version=2，
        含 optimizer_identity + param_groups + resolved_config。"""
        from cosyvoice.bin.train_emo import write_training_identity

        model, optimizer, scheduler = fake_model_optim_sched

        # 准备 resolved.yaml
        resolved_path = tmp_path / "resolved.yaml"
        resolved_payload = {
            "config_source": "conf/emo_film.yaml",
            "arguments": {"seed": 1986},
            "train_conf": {"max_epoch": 5, "batch_size": 4},
        }
        resolved_path.write_text(
            yaml.safe_dump(resolved_payload, sort_keys=True), encoding="utf-8"
        )

        # 准备 git code_root（干净 repo）
        code_root = _init_git_repo(tmp_path / "repo")

        identity_path = tmp_path / "train_identity.json"
        write_training_identity(
            identity_path,
            model=model,
            code_root=code_root,
            contract_dir=None,
            command="torchrun cosyvoice/bin/train_emo.py",
            seed=1986,
            base_checkpoint=None,
            resolved_config=resolved_path,
            checkpoint_role="fresh",
            optimizer=optimizer,
            scheduler=scheduler,
        )

        data = json.loads(identity_path.read_text(encoding="utf-8"))
        assert data["contract_name"] == "emofilm"
        assert data["schema_version"] == 2
        # optimizer_identity 绑定到顶层
        assert "optimizer_identity" in data
        assert data["optimizer_identity"] is not None
        assert "param_groups" in data["optimizer_identity"]
        # resolved_config 绑定到顶层（dict）
        assert "resolved_config" in data
        assert data["resolved_config"] is not None
        assert data["resolved_config"]["train_conf"]["max_epoch"] == 5

    def test_optimizer_identity_matches_actual_groups(self, tmp_path, fake_model_optim_sched):
        """optimizer_identity 的 param_groups 与 optimizer 实际一致。"""
        from cosyvoice.bin.train_emo import write_training_identity

        model, optimizer, scheduler = fake_model_optim_sched

        resolved_path = tmp_path / "resolved.yaml"
        resolved_path.write_text(
            yaml.safe_dump({"train_conf": {}}, sort_keys=True), encoding="utf-8"
        )
        code_root = _init_git_repo(tmp_path / "repo")

        identity = write_training_identity(
            tmp_path / "train_identity.json",
            model=model,
            code_root=code_root,
            contract_dir=None,
            command="torchrun ...",
            seed=1986,
            resolved_config=resolved_path,
            checkpoint_role="fresh",
            optimizer=optimizer,
            scheduler=scheduler,
        )

        opt_id = identity["optimizer_identity"]
        roles = {g["role"] for g in opt_id["param_groups"]}
        assert "random_new_condition" in roles
        assert "downstream_heads" in roles
        # 每组的 initial_lr / weight_decay / tensor_count / param_count 与 optimizer 一致
        opt_by_role = {pg["name"]: pg for pg in optimizer.param_groups}
        for stat in opt_id["param_groups"]:
            pg = opt_by_role[stat["role"]]
            assert stat["tensor_count"] == len(pg["params"])
            assert stat["param_count"] == sum(p.numel() for p in pg["params"])
            assert stat["initial_lr"] == pytest.approx(pg["lr"])
            assert stat["weight_decay"] == pytest.approx(pg.get("weight_decay", 0.0))
        # scheduler 统计
        assert opt_id["scheduler"]["type"] == "ConstantLR"

    def test_identity_records_source_and_patch_bundle_when_dirty(
        self, tmp_path, fake_model_optim_sched
    ):
        """dirty worktree → source.dirty=True；patch_bundle 保存实际 diff bytes。"""
        from cosyvoice.bin.train_emo import write_training_identity

        model, optimizer, scheduler = fake_model_optim_sched

        resolved_path = tmp_path / "resolved.yaml"
        resolved_path.write_text(
            yaml.safe_dump({"train_conf": {}}, sort_keys=True), encoding="utf-8"
        )
        code_root = _init_git_repo(tmp_path / "repo")
        # 制造 dirty
        (code_root / "README.md").write_text("modified\n")

        identity_path = tmp_path / "train_identity.json"
        identity = write_training_identity(
            identity_path,
            model=model,
            code_root=code_root,
            contract_dir=None,
            command="torchrun ...",
            seed=1986,
            resolved_config=resolved_path,
            checkpoint_role="fresh",
            optimizer=optimizer,
            scheduler=scheduler,
        )

        assert identity["source"]["dirty"] is True
        assert identity["source"]["patch_bundle"] is not None
        patch_path = Path(identity["source"]["patch_bundle"]["path"])
        assert patch_path.is_file()
        assert len(patch_path.read_bytes()) > 0

    def test_extra_records_checkpoint_role_and_parameter_hash(
        self, tmp_path, fake_model_optim_sched
    ):
        """extra 仍记录 checkpoint_role + parameter_hash。"""
        from cosyvoice.bin.train_emo import write_training_identity
        from cosyvoice.utils.emo_checkpoint import hash_model_state

        model, optimizer, scheduler = fake_model_optim_sched

        resolved_path = tmp_path / "resolved.yaml"
        resolved_path.write_text(
            yaml.safe_dump({"train_conf": {}}, sort_keys=True), encoding="utf-8"
        )
        code_root = _init_git_repo(tmp_path / "repo")

        identity = write_training_identity(
            tmp_path / "train_identity.json",
            model=model,
            code_root=code_root,
            contract_dir=None,
            command="torchrun ...",
            seed=1986,
            resolved_config=resolved_path,
            checkpoint_role="fresh",
            optimizer=optimizer,
            scheduler=scheduler,
        )

        assert identity["extra"]["checkpoint_role"] == "fresh"
        assert identity["extra"]["parameter_hash"] == hash_model_state(model)

    def test_no_optimizer_produces_none_optimizer_identity(self, tmp_path):
        """不传 optimizer/scheduler → optimizer_identity=None（向后兼容）。"""
        from cosyvoice.bin.train_emo import write_training_identity

        model = _FakeModel()
        resolved_path = tmp_path / "resolved.yaml"
        resolved_path.write_text(
            yaml.safe_dump({"train_conf": {}}, sort_keys=True), encoding="utf-8"
        )
        code_root = _init_git_repo(tmp_path / "repo")

        identity = write_training_identity(
            tmp_path / "train_identity.json",
            model=model,
            code_root=code_root,
            contract_dir=None,
            command="torchrun ...",
            seed=1986,
            resolved_config=resolved_path,
            checkpoint_role="fresh",
        )

        assert identity["schema_version"] == 2
        assert identity["contract_name"] == "emofilm"
        assert identity["optimizer_identity"] is None

    def test_base_checkpoint_recorded_with_sha256(
        self, tmp_path, fake_model_optim_sched
    ):
        """base_checkpoint 在 identity 中记录 path + sha256。"""
        from cosyvoice.bin.train_emo import write_training_identity

        model, optimizer, scheduler = fake_model_optim_sched
        base_ckpt = tmp_path / "llm.pt"
        torch.save(model.state_dict(), base_ckpt)

        resolved_path = tmp_path / "resolved.yaml"
        resolved_path.write_text(
            yaml.safe_dump({"train_conf": {}}, sort_keys=True), encoding="utf-8"
        )
        code_root = _init_git_repo(tmp_path / "repo")

        identity = write_training_identity(
            tmp_path / "train_identity.json",
            model=model,
            code_root=code_root,
            contract_dir=None,
            command="torchrun ...",
            seed=1986,
            base_checkpoint=base_ckpt,
            resolved_config=resolved_path,
            checkpoint_role="base",
            optimizer=optimizer,
            scheduler=scheduler,
        )

        assert identity["base_checkpoint"] is not None
        assert identity["base_checkpoint"]["sha256"]
        assert identity["base_checkpoint"]["path"] == str(base_ckpt.resolve())


# ============================================================
# B. update_training_identity → v2 extra + v1 拒绝
# ============================================================


class TestUpdateTrainingIdentityV2:
    """B. update_training_identity 写 v2 extra；v1 identity raise。"""

    def test_update_writes_final_checkpoint_and_hash(self, tmp_path):
        """update 写入 final_checkpoint（path+sha256）+ final_parameter_hash。"""
        from cosyvoice.bin.train_emo import update_training_identity
        from cosyvoice.utils.emo_checkpoint import hash_model_state

        model = _FakeModel()
        final_ckpt = tmp_path / "final.pt"
        torch.save(model.state_dict(), final_ckpt)

        # 先写一个 v2 identity
        identity_path = tmp_path / "train_identity.json"
        identity_path.write_text(
            json.dumps({
                "schema_version": 2,
                "contract_name": "emofilm",
                "extra": {"checkpoint_role": "fresh"},
            }),
            encoding="utf-8",
        )

        identity = update_training_identity(
            identity_path,
            model=model,
            final_checkpoint=final_ckpt,
        )

        assert identity["extra"]["final_parameter_hash"] == hash_model_state(model)
        assert identity["extra"]["final_checkpoint"]["sha256"]
        assert identity["extra"]["final_checkpoint"]["path"] == str(final_ckpt.resolve())

    def test_update_raises_on_v1_identity(self, tmp_path):
        """读到 v1 identity（schema_version=1）→ raise 明确提示重训。"""
        from cosyvoice.bin.train_emo import update_training_identity

        model = _FakeModel()
        final_ckpt = tmp_path / "final.pt"
        torch.save(model.state_dict(), final_ckpt)

        identity_path = tmp_path / "train_identity.json"
        identity_path.write_text(
            json.dumps({
                "schema_version": 1,
                "contract_name": "emofilm_v1",
                "extra": {"checkpoint_role": "base"},
            }),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="schema_version=1|v1|retrain|重训"):
            update_training_identity(
                identity_path,
                model=model,
                final_checkpoint=final_ckpt,
            )

    def test_update_raises_on_missing_schema_version(self, tmp_path):
        """读到无 schema_version 的旧 identity → 视为 v1 → raise。"""
        from cosyvoice.bin.train_emo import update_training_identity

        model = _FakeModel()
        final_ckpt = tmp_path / "final.pt"
        torch.save(model.state_dict(), final_ckpt)

        identity_path = tmp_path / "train_identity.json"
        identity_path.write_text(
            json.dumps({"extra": {"checkpoint_role": "base"}}),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="schema_version|v1|retrain|重训"):
            update_training_identity(
                identity_path,
                model=model,
                final_checkpoint=final_ckpt,
            )

    def test_update_preserves_v2_top_level_fields(self, tmp_path):
        """update 不破坏 v2 顶层字段（optimizer_identity / resolved_config 等）。"""
        from cosyvoice.bin.train_emo import update_training_identity

        model = _FakeModel()
        final_ckpt = tmp_path / "final.pt"
        torch.save(model.state_dict(), final_ckpt)

        identity_path = tmp_path / "train_identity.json"
        original = {
            "schema_version": 2,
            "contract_name": "emofilm",
            "optimizer_identity": {
                "param_groups": [{"role": "random_new_condition"}],
                "scheduler": {"type": "ConstantLR", "key_params": {}},
            },
            "resolved_config": {"train_conf": {"max_epoch": 5}},
            "source": {"git_head": "abc123", "dirty": False},
            "extra": {"checkpoint_role": "fresh"},
        }
        identity_path.write_text(json.dumps(original), encoding="utf-8")

        identity = update_training_identity(
            identity_path,
            model=model,
            final_checkpoint=final_ckpt,
        )

        # v2 顶层字段不变
        assert identity["optimizer_identity"] == original["optimizer_identity"]
        assert identity["resolved_config"] == original["resolved_config"]
        assert identity["source"] == original["source"]
        # extra 新增 final 字段
        assert "final_checkpoint" in identity["extra"]
        assert "final_parameter_hash" in identity["extra"]
