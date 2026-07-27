"""Ticket 11 — EmoFiLM v2 逐条运行身份与安全恢复链 focused 测试（CPU）。

核心修复（MAP §2 identity 段 + brief 11）：
  - v1 ``write_emofilm_run_identity.py`` 只存 diff SHA-256 hash 不存 bytes
    → dirty worktree 不可重建。v2 保存可应用的 patch bundle（实际字节）。
  - v1 ``inference_emo_film.py`` L182-184 skip-existing 只检查 ``os.path.isfile``
    → 134 条无验证恢复。v2 ``check_skip_existing`` 要求完整逐条身份一致。
  - v1 评测不持久化逐样本 → v2 绑定 eval-row 身份 + aggregate 集合 hash。

覆盖（brief 11 DoD A-E）：
  A. 干净 revision 记录；dirty⇒patch bundle 保存（可重建，不只 SHA）。
  B. 身份匹配→skip-existing 复用；checkpoint/control/prompt/config/source
     任一不一致→拒绝。
  C. eval-row 绑定齐全；aggregate rows 被替换/遗漏/混入→检测。
  D. generation/evaluation identity 写入器 roundtrip + 与 07 train identity 协调。
  E. v1 identity 入口签名 + 行为原样保留。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools.build_emofilm_contract import (
    FINISH_REASONS,
    validate_eval_row,
    validate_generation_row,
)
from tools.write_emofilm_run_identity import (
    SkipDecision,
    _save_patch_bundle,
    build_source_revision_or_patch,
    capture_source_identity,
    check_skip_existing,
    compute_aggregate_identity,
    eval_row_identity_fingerprint,
    generation_request_fingerprint,
    generation_row_identity_fingerprint,
    verify_aggregate_identity,
    write_emofilm_evaluation_identity,
    write_emofilm_generation_identity,
)

ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# helpers：合成 generation row / eval row
# ============================================================

_CKPT_SHA = "a" * 64
_CKPT_SHA_2 = "b" * 64
_SOURCE_REV = "c" * 40
_SOURCE_REV_2 = "d" * 40

_BASE_DECODE_CONFIG = {
    "min_token_text_ratio": 0.25,
    "max_token_text_ratio": 5.0,
    "max_len_hard_cap": 220,
    "top_p": 0.8,
}

_BASE_CONTROL_ROW = {
    "utt_id": "utt_001",
    "control_emotion_id": 2,
    "control_intensity_id": 2,
}

_BASE_PROMPT_ROW = {
    "utt_id": "prompt_001",
    "speaker_id": "spk_01",
    "flow_path": "artifacts/flow/flow.pt",
    "hift_path": "artifacts/hift/hift.pt",
}


def _make_generation_row(
    *,
    utt_id: str = "utt_001",
    finish_reason: str = "eos",
    source_revision: str = _SOURCE_REV,
    checkpoint_sha256: str = _CKPT_SHA,
    control_row_ref: str = "manifests/control.jsonl#utt_001",
    prompt_row_ref: str = "manifests/prompt.jsonl#prompt_001",
    decode_config: dict | None = None,
    seed: int = 1986,
    wav_path: str = "wavs/utt_001.wav",
) -> dict:
    row: dict = {
        "utt_id": utt_id,
        "finish_reason": finish_reason,
        "source_revision": source_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "control_row_ref": control_row_ref,
        "prompt_row_ref": prompt_row_ref,
        "decode_config": decode_config or dict(_BASE_DECODE_CONFIG),
        "seed": seed,
    }
    if finish_reason == "eos":
        row["wav_path"] = wav_path
    return row


def _make_generation_row_with_inline(
    *,
    utt_id: str = "utt_001",
    finish_reason: str = "eos",
    source_revision: str = _SOURCE_REV,
    checkpoint_sha256: str = _CKPT_SHA,
    control_row: dict | None = None,
    prompt_row: dict | None = None,
    decode_config: dict | None = None,
    seed: int = 1986,
    wav_path: str = "wavs/utt_001.wav",
) -> dict:
    """Generation row with inline control_row/prompt_row dicts."""
    row: dict = {
        "utt_id": utt_id,
        "finish_reason": finish_reason,
        "source_revision": source_revision,
        "checkpoint_sha256": checkpoint_sha256,
        "control_row": control_row or dict(_BASE_CONTROL_ROW),
        "prompt_row": prompt_row or dict(_BASE_PROMPT_ROW),
        "decode_config": decode_config or dict(_BASE_DECODE_CONFIG),
        "seed": seed,
    }
    if finish_reason == "eos":
        row["wav_path"] = wav_path
    return row


def _make_eval_row(
    *,
    utt_id: str = "utt_001",
    generation_row_ref: str = "gen_manifest.jsonl#utt_001",
    control_span_ref: str = "control.jsonl#span_001",
    evaluator_name: str = "emotion2vec-v2",
    evaluator_version: str = "frozen-2026-01",
    boundary_evidence_tier: str = "exact",
    metrics: dict | None = None,
) -> dict:
    return {
        "utt_id": utt_id,
        "generation_row_ref": generation_row_ref,
        "control_span_ref": control_span_ref,
        "evaluator": {
            "name": evaluator_name,
            "version": evaluator_version,
            "label_space": ["ang", "hap", "neu", "sad", "sur"],
            "sample_rate_hz": 16000,
            "frame_rate_hz": 50.0,
        },
        "boundary_evidence_tier": boundary_evidence_tier,
        "metrics": metrics or {"emo_sim": 0.85, "arousal_diff": 0.12},
    }


def _make_request_fingerprint(
    *,
    source_revision: str = _SOURCE_REV,
    checkpoint_sha256: str = _CKPT_SHA,
    control_row_ref: str = "manifests/control.jsonl#utt_001",
    prompt_row_ref: str = "manifests/prompt.jsonl#prompt_001",
    decode_config: dict | None = None,
    seed: int = 1986,
) -> str:
    return generation_request_fingerprint(
        source=source_revision,
        checkpoint_sha256=checkpoint_sha256,
        control_row_ref=control_row_ref,
        prompt_row_ref=prompt_row_ref,
        decode_config=decode_config or dict(_BASE_DECODE_CONFIG),
        seed=seed,
    )


def _create_wav(tmp_path: Path, name: str = "utt_001.wav") -> str:
    """在 tmp_path 下创建一个假的 WAV 文件，返回绝对路径（满足 isfile 检查）。"""
    wav = tmp_path / name
    wav.write_bytes(b"fake wav content")
    return str(wav)


# ============================================================
# temp git repo helper（用于测试 source identity）
# ============================================================


def _init_git_repo(td: Path) -> Path:
    """初始化一个临时 git 仓库，做一个干净 commit。"""
    td.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=td,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=td,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=td,
        check=True,
        capture_output=True,
    )
    (td / "README.md").write_text("test repo\n")
    subprocess.run(["git", "add", "-A"], cwd=td, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=td,
        check=True,
        capture_output=True,
    )
    return td


def _get_head_sha(td: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=td,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# ============================================================
# A. 源码身份（干净 revision + dirty⇒patch bundle）
# ============================================================


class TestSourceIdentity:
    """A. 源码身份：干净 revision 记录；dirty⇒patch bundle 保存。"""

    def test_clean_revision_recorded(self, tmp_path):
        """干净 worktree → git_head 记录；dirty=False；无 patch_bundle。"""
        repo = _init_git_repo(tmp_path / "repo")
        head = _get_head_sha(repo)

        source = capture_source_identity(repo)

        assert source["git_head"] == head
        assert source["dirty"] is False
        assert source["patch_bundle"] is None

    def test_dirty_saves_patch_bundle_bytes(self, tmp_path):
        """dirty worktree + patch_bundle_path → 保存实际 diff 字节，不只 SHA。"""
        repo = _init_git_repo(tmp_path / "repo")
        # 制造 dirty：修改文件但不 commit
        (repo / "README.md").write_text("modified content\n")
        (repo / "new_file.py").write_text("print('hello')\n")

        patch_path = tmp_path / "patches" / "run.patch"
        source = capture_source_identity(repo, patch_bundle_path=patch_path)

        assert source["dirty"] is True
        assert source["patch_bundle"] is not None
        # patch bundle 文件实际存在且包含 diff 字节
        assert patch_path.is_file()
        patch_bytes = patch_path.read_bytes()
        assert len(patch_bytes) > 0
        # 验证不是空 patch（实际包含 diff 内容）
        assert b"modified content" in patch_bytes or b"new_file" in patch_bytes
        # SHA-256 匹配
        expected_sha = hashlib.sha256(patch_bytes).hexdigest()
        assert source["patch_bundle"]["sha256"] == expected_sha
        assert source["patch_bundle"]["size_bytes"] == len(patch_bytes)

    def test_dirty_patch_bundle_is_reproducible(self, tmp_path):
        """保存的 patch bundle 可通过 git apply 重建修改（可重建性验证）。"""
        repo = _init_git_repo(tmp_path / "repo")
        original = (repo / "README.md").read_text()

        # 制造 dirty：修改 tracked 文件 + 添加新 tracked 文件
        (repo / "README.md").write_text("modified content\n")
        (repo / "extra.py").write_text("# extra tracked file\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)

        patch_path = tmp_path / "run.patch"
        capture_source_identity(repo, patch_bundle_path=patch_path)

        # 回退到干净状态
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=repo, check=True, capture_output=True)
        assert (repo / "README.md").read_text() == original
        assert not (repo / "extra.py").exists()

        # 应用 patch bundle 重建修改
        result = subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=repo,
            capture_output=True,
        )
        assert result.returncode == 0, (
            f"git apply failed: {result.stderr.decode()}"
        )
        assert (repo / "README.md").read_text() == "modified content\n"
        assert (repo / "extra.py").read_text() == "# extra tracked file\n"

    def test_dirty_without_patch_path_records_sha_only(self, tmp_path):
        """dirty worktree 无 patch_bundle_path → 记录 SHA 但不保存 bytes。"""
        repo = _init_git_repo(tmp_path / "repo")
        (repo / "README.md").write_text("modified\n")

        source = capture_source_identity(repo)  # 无 patch_bundle_path

        assert source["dirty"] is True
        assert source["worktree_diff_sha256"] is not None
        assert source["patch_bundle"] is None  # 未保存 bytes

    def test_patch_bundle_includes_untracked(self, tmp_path):
        """patch_bundle 必须覆盖 untracked 新文件（未 git add），且不污染真实 index/worktree。

        旧实现 ``git diff --binary HEAD`` 只含 tracked 改动，漏 untracked →
        dirty worktree 不可重建（EmoFiLM 主线补救 #7）。修复用 GIT_INDEX_FILE
        隔离临时 index：read-tree HEAD → add -A（写临时 index）→ diff --cached HEAD。
        真实 ``.git/index`` 与 worktree 均不动。
        """
        repo = _init_git_repo(tmp_path / "repo")
        # tracked 改动 + untracked 新文件（关键：未 git add）
        (repo / "README.md").write_text("modified content\n")
        (repo / "untracked_new.py").write_text("b=1\n")
        # 记录调用前的真实 git status，用于验证 index 未被污染
        status_before = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, check=True, capture_output=True, text=True,
        ).stdout

        # patch_bundle 落在 repo 外（实际使用：artifacts 目录，不在 code_root 内）
        patch_out = tmp_path / "artifacts" / "patch.bundle"
        info = _save_patch_bundle(repo, patch_out)

        bundle = patch_out.read_bytes()
        # 1. untracked 新文件必须出现在 bundle（核心修复点）
        assert b"untracked_new.py" in bundle, (
            "patch_bundle 必须含 untracked 新文件（#7 修复目标）"
        )
        # tracked 改动也仍包含
        assert b"modified content" in bundle
        # 返回的 sha256 与落盘 bytes 一致（产物完整性哈希；ADR-0020 §3）
        assert info["sha256"] == hashlib.sha256(bundle).hexdigest()
        assert info["size_bytes"] == len(bundle)
        # 2. worktree 未被改动（GIT_INDEX_FILE 隔离：真实 worktree 不动）
        assert (repo / "untracked_new.py").read_text() == "b=1\n"
        assert (repo / "README.md").read_text() == "modified content\n"
        # 3. 真实 git status 未变（临时 index 未污染真实 .git/index）
        status_after = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, check=True, capture_output=True, text=True,
        ).stdout
        assert status_after == status_before, (
            "真实 git index 未被污染：调用前后 status --porcelain 必须一致"
        )
        # worktree 仍 dirty（两行：tracked 改动 + untracked 新文件）
        assert status_after.count("\n") >= 2

    def test_build_source_revision_clean(self):
        """干净 source identity → source_revision 字段。"""
        source = {
            "git_head": _SOURCE_REV,
            "dirty": False,
            "worktree_diff_sha256": None,
            "patch_bundle": None,
        }
        result = build_source_revision_or_patch(source)
        assert result == {"source_revision": _SOURCE_REV}

    def test_build_source_revision_dirty_with_bundle(self):
        """dirty + patch_bundle → source_patch_bundle + source_patch_sha256。"""
        bundle = {
            "path": "/tmp/run.patch",
            "sha256": "e" * 64,
            "size_bytes": 1024,
        }
        source = {
            "git_head": _SOURCE_REV,
            "dirty": True,
            "worktree_diff_sha256": "f" * 64,
            "patch_bundle": bundle,
        }
        result = build_source_revision_or_patch(source)
        assert result["source_patch_bundle"] == bundle
        assert result["source_patch_sha256"] == "e" * 64


# ============================================================
# B. 逐条 generation 身份 + 安全 skip-existing
# ============================================================


class TestSkipExisting:
    """B. skip-existing 仅在完整逐条身份一致时复用。"""

    def test_identity_match_skips(self, tmp_path):
        """完整身份匹配 + WAV 文件存在 → 安全复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(wav_path=wav)
        fp = _make_request_fingerprint()
        decision = check_skip_existing(row, fp)
        assert decision.skip is True
        assert "identity match" in decision.reason

    def test_inline_rows_identity_match_skips(self, tmp_path):
        """使用 inline control_row/prompt_row 的 row 也正确匹配。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row_with_inline(wav_path=wav)
        fp = generation_request_fingerprint(
            source=_SOURCE_REV,
            checkpoint_sha256=_CKPT_SHA,
            control_row=dict(_BASE_CONTROL_ROW),
            prompt_row=dict(_BASE_PROMPT_ROW),
            decode_config=dict(_BASE_DECODE_CONFIG),
            seed=1986,
        )
        decision = check_skip_existing(row, fp)
        assert decision.skip is True

    def test_checkpoint_mismatch_rejects(self, tmp_path):
        """checkpoint 不同 → 拒绝复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(wav_path=wav, checkpoint_sha256=_CKPT_SHA)
        fp = _make_request_fingerprint(checkpoint_sha256=_CKPT_SHA_2)
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "identity mismatch" in decision.reason

    def test_control_mismatch_rejects(self, tmp_path):
        """控制 row 不同 → 拒绝复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(
            wav_path=wav, control_row_ref="manifests/control.jsonl#utt_001"
        )
        fp = _make_request_fingerprint(control_row_ref="manifests/control.jsonl#utt_002")
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "identity mismatch" in decision.reason

    def test_prompt_mismatch_rejects(self, tmp_path):
        """prompt row 不同 → 拒绝复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(
            wav_path=wav, prompt_row_ref="manifests/prompt.jsonl#prompt_001"
        )
        fp = _make_request_fingerprint(prompt_row_ref="manifests/prompt.jsonl#prompt_002")
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "identity mismatch" in decision.reason

    def test_decode_config_mismatch_rejects(self, tmp_path):
        """解码配置不同 → 拒绝复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(wav_path=wav)
        different_config = dict(_BASE_DECODE_CONFIG)
        different_config["top_p"] = 0.9
        fp = _make_request_fingerprint(decode_config=different_config)
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "identity mismatch" in decision.reason

    def test_source_mismatch_rejects(self, tmp_path):
        """源码身份不同 → 拒绝复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(wav_path=wav, source_revision=_SOURCE_REV)
        fp = _make_request_fingerprint(source_revision=_SOURCE_REV_2)
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "identity mismatch" in decision.reason

    def test_non_eos_rejects(self):
        """finish_reason != eos → 拒绝（无正式 WAV）。"""
        row = _make_generation_row(finish_reason="max_len_reached")
        row.pop("wav_path", None)
        fp = _make_request_fingerprint()
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "finish_reason" in decision.reason

    def test_missing_wav_path_rejects(self):
        """eos 但 wav_path 缺失 → 拒绝。"""
        row = _make_generation_row()
        row["wav_path"] = ""
        fp = _make_request_fingerprint()
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "wav_path" in decision.reason

    def test_fingerprint_stable_on_same_row(self):
        """相同 row → 相同指纹（确定性）。"""
        row = _make_generation_row()
        fp1 = generation_row_identity_fingerprint(row)
        fp2 = generation_row_identity_fingerprint(dict(row))
        assert fp1 == fp2

    def test_fingerprint_changes_on_decode_config(self):
        """decode_config 变化 → 不同指纹。"""
        row = _make_generation_row()
        fp1 = generation_row_identity_fingerprint(row)
        modified = dict(row)
        modified["decode_config"] = {**row["decode_config"], "top_p": 0.95}
        fp2 = generation_row_identity_fingerprint(modified)
        assert fp1 != fp2

    # -- Task 4 / #5：seed 纳入身份指纹 ----------------------------------------

    def test_fingerprint_changes_on_seed(self):
        """seed 变化 → 不同指纹（per-request 固定种子是身份的一部分）。"""
        row = _make_generation_row(seed=1986)
        fp1 = generation_row_identity_fingerprint(row)
        modified = dict(row)
        modified["seed"] = 42
        fp2 = generation_row_identity_fingerprint(modified)
        assert fp1 != fp2

    def test_seed_mismatch_rejects_skip(self, tmp_path):
        """既有 row seed=1986，请求 seed=42 → 指纹不同 → 拒绝复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(wav_path=wav, seed=1986)
        fp = _make_request_fingerprint(seed=42)  # 不同 seed
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "mismatch" in decision.reason

    def test_seed_match_allows_skip(self, tmp_path):
        """既有 row seed=1986，请求 seed=1986 → 指纹一致 → 安全复用。"""
        wav = _create_wav(tmp_path)
        row = _make_generation_row(wav_path=wav, seed=1986)
        fp = _make_request_fingerprint(seed=1986)
        decision = check_skip_existing(row, fp)
        assert decision.skip is True


class TestSkipExistingEdgeCases:
    """B 边界 case：source patch bundle 比对 + WAV 存在性检查。"""

    def test_source_patch_bundle_identity_match(self, tmp_path):
        """使用 source_patch_bundle 的 row 与请求匹配。"""
        wav = _create_wav(tmp_path)
        bundle = {"path": "patches/run.patch", "sha256": "e" * 64, "size_bytes": 1024}
        row = _make_generation_row(wav_path=wav)
        row.pop("source_revision")
        row["source_patch_bundle"] = bundle
        row["source_patch_sha256"] = "e" * 64

        fp = generation_request_fingerprint(
            source={"git_head": _SOURCE_REV, "dirty": True, "patch_bundle": bundle},
            checkpoint_sha256=_CKPT_SHA,
            control_row_ref="manifests/control.jsonl#utt_001",
            prompt_row_ref="manifests/prompt.jsonl#prompt_001",
            decode_config=dict(_BASE_DECODE_CONFIG),
            seed=1986,
        )
        decision = check_skip_existing(row, fp)
        assert decision.skip is True

    def test_wav_file_not_found_rejects(self, tmp_path):
        """WAV 文件不存在（已删除/被替换）+ 身份匹配 → 不 skip，重新生成。

        ADR-0020 §3：safe-resume 新增 os.path.isfile 存在性检查——
        已删除/被替换的 WAV 不再被安全跳过并复用。
        """
        # 构造一个不存在的路径（不创建文件）
        missing_wav = str(tmp_path / "deleted.wav")
        assert not os.path.isfile(missing_wav)

        row = _make_generation_row(wav_path=missing_wav)
        fp = _make_request_fingerprint()  # 身份完全匹配
        decision = check_skip_existing(row, fp)
        assert decision.skip is False
        assert "not found on disk" in decision.reason


# ============================================================
# C. 逐条 evaluation 身份 + aggregate 身份
# ============================================================


class TestEvalRowIdentity:
    """C. eval row 绑定齐全。"""

    def test_eval_fingerprint_stable(self):
        """相同 eval row → 相同指纹。"""
        row = _make_eval_row()
        fp1 = eval_row_identity_fingerprint(row)
        fp2 = eval_row_identity_fingerprint(dict(row))
        assert fp1 == fp2

    def test_eval_fingerprint_changes_on_evaluator(self):
        """evaluator version 变化 → 不同指纹。"""
        row = _make_eval_row()
        fp1 = eval_row_identity_fingerprint(row)
        modified = dict(row)
        modified["evaluator"] = {**row["evaluator"], "version": "frozen-2026-02"}
        fp2 = eval_row_identity_fingerprint(modified)
        assert fp1 != fp2

    def test_eval_fingerprint_changes_on_generation_ref(self):
        """generation row 引用变化 → 不同指纹。"""
        row = _make_eval_row()
        fp1 = eval_row_identity_fingerprint(row)
        modified = dict(row)
        modified["generation_row_ref"] = "gen_manifest.jsonl#utt_002"
        fp2 = eval_row_identity_fingerprint(modified)
        assert fp1 != fp2

    def test_eval_fingerprint_changes_on_control_span(self):
        """控制 span 引用变化 → 不同指纹。"""
        row = _make_eval_row()
        fp1 = eval_row_identity_fingerprint(row)
        modified = dict(row)
        modified["control_span_ref"] = "control.jsonl#span_002"
        fp2 = eval_row_identity_fingerprint(modified)
        assert fp1 != fp2

    def test_eval_fingerprint_changes_on_metrics(self):
        """指标变化 → 不同指纹（绑定逐样本指标用于审计）。"""
        row = _make_eval_row()
        fp1 = eval_row_identity_fingerprint(row)
        modified = dict(row)
        modified["metrics"] = {"emo_sim": 0.90, "arousal_diff": 0.05}
        fp2 = eval_row_identity_fingerprint(modified)
        assert fp1 != fp2

    def test_eval_fingerprint_changes_on_evidence_tier(self):
        """boundary_evidence_tier 变化 → 不同指纹。"""
        row = _make_eval_row(boundary_evidence_tier="exact")
        fp1 = eval_row_identity_fingerprint(row)
        modified = _make_eval_row(boundary_evidence_tier="approximate")
        fp2 = eval_row_identity_fingerprint(modified)
        assert fp1 != fp2


class TestAggregateIdentity:
    """C. aggregate identity 检测 rows 被替换/遗漏/混入。"""

    def test_aggregate_stable_on_same_rows(self):
        """相同 rows → 相同 aggregate hash（确定性）。"""
        rows = [
            _make_eval_row(utt_id="utt_001"),
            _make_eval_row(utt_id="utt_002", generation_row_ref="gen.jsonl#utt_002"),
            _make_eval_row(utt_id="utt_003", generation_row_ref="gen.jsonl#utt_003"),
        ]
        h1 = compute_aggregate_identity(rows)
        h2 = compute_aggregate_identity(list(reversed(rows)))
        assert h1 == h2  # 顺序无关（按 utt_id 排序）

    def test_aggregate_detects_replaced_row(self):
        """一行被替换（同 utt_id 不同 evaluator）→ aggregate hash 变化。"""
        rows = [
            _make_eval_row(utt_id="utt_001"),
            _make_eval_row(utt_id="utt_002", generation_row_ref="gen.jsonl#utt_002"),
        ]
        original = compute_aggregate_identity(rows)

        # 替换第二行的 evaluator version
        replaced = [
            _make_eval_row(utt_id="utt_001"),
            _make_eval_row(
                utt_id="utt_002",
                generation_row_ref="gen.jsonl#utt_002",
                evaluator_version="frozen-2026-02",
            ),
        ]
        actual = compute_aggregate_identity(replaced)
        match, reason = verify_aggregate_identity(replaced, original)
        assert match is False
        assert "mismatch" in reason

    def test_aggregate_detects_missing_row(self):
        """一行遗漏 → n_rows 变化 → aggregate hash 变化。"""
        rows = [
            _make_eval_row(utt_id="utt_001"),
            _make_eval_row(utt_id="utt_002", generation_row_ref="gen.jsonl#utt_002"),
            _make_eval_row(utt_id="utt_003", generation_row_ref="gen.jsonl#utt_003"),
        ]
        original = compute_aggregate_identity(rows)

        missing = rows[:2]  # 去掉最后一行
        match, reason = verify_aggregate_identity(missing, original)
        assert match is False
        assert "mismatch" in reason

    def test_aggregate_detects_extra_row(self):
        """多出混入的行 → n_rows 变化 → aggregate hash 变化。"""
        rows = [
            _make_eval_row(utt_id="utt_001"),
            _make_eval_row(utt_id="utt_002", generation_row_ref="gen.jsonl#utt_002"),
        ]
        original = compute_aggregate_identity(rows)

        mixed_in = rows + [_make_eval_row(utt_id="utt_003", generation_row_ref="gen.jsonl#utt_003")]
        match, reason = verify_aggregate_identity(mixed_in, original)
        assert match is False
        assert "mismatch" in reason

    def test_aggregate_detects_mixed_run_rows(self):
        """混入来自不同运行的行（不同 generation_ref）→ 检测到。"""
        rows_run_a = [
            _make_eval_row(utt_id="utt_001", generation_row_ref="run_a.jsonl#utt_001"),
            _make_eval_row(utt_id="utt_002", generation_row_ref="run_a.jsonl#utt_002"),
        ]
        original = compute_aggregate_identity(rows_run_a)

        # 第二行换成 run_b 的
        mixed = [
            _make_eval_row(utt_id="utt_001", generation_row_ref="run_a.jsonl#utt_001"),
            _make_eval_row(utt_id="utt_002", generation_row_ref="run_b.jsonl#utt_002"),
        ]
        match, reason = verify_aggregate_identity(mixed, original)
        assert match is False

    def test_aggregate_verify_match(self):
        """相同 rows → verify 返回 True。"""
        rows = [_make_eval_row(utt_id="utt_001"), _make_eval_row(utt_id="utt_002")]
        agg = compute_aggregate_identity(rows)
        match, reason = verify_aggregate_identity(rows, agg)
        assert match is True
        assert "matches" in reason

    def test_aggregate_empty_rows(self):
        """空 rows 列表 → 有效 hash（边界 case）。"""
        agg = compute_aggregate_identity([])
        assert isinstance(agg, str)
        assert len(agg) == 64
        match, _ = verify_aggregate_identity([], agg)
        assert match is True


# ============================================================
# D. v2 identity 写入器（generation + evaluation roundtrip）
# ============================================================


class TestIdentityWriters:
    """D. generation/evaluation identity 写入器 roundtrip。"""

    def test_generation_identity_roundtrip(self, tmp_path):
        """generation identity JSON 写出 + 读回。"""
        out = tmp_path / "gen_identity.json"
        identity = write_emofilm_generation_identity(
            out,
            code_root=ROOT,
            command="python inference_v2.py",
            checkpoint_sha256=_CKPT_SHA,
            decode_config_defaults=dict(_BASE_DECODE_CONFIG),
            generation_manifest_path="manifests/gen.jsonl",
            n_generation_rows=100,
            train_identity_ref="artifacts/train_identity.json",
        )
        assert out.is_file()
        reloaded = json.loads(out.read_text())
        assert reloaded["schema_version"] == 2
        assert reloaded["run_kind"] == "generate"
        assert reloaded["contract_name"] == "emofilm"
        assert reloaded["checkpoint"]["sha256"] == _CKPT_SHA
        assert reloaded["n_generation_rows"] == 100
        assert reloaded["train_identity_ref"] == "artifacts/train_identity.json"
        assert "source" in reloaded
        assert "git_head" in reloaded["source"]

    def test_evaluation_identity_roundtrip(self, tmp_path):
        """evaluation identity JSON 写出 + 读回。"""
        eval_rows = [_make_eval_row(utt_id="utt_001"), _make_eval_row(utt_id="utt_002")]
        agg = compute_aggregate_identity(eval_rows)

        out = tmp_path / "eval_identity.json"
        identity = write_emofilm_evaluation_identity(
            out,
            code_root=ROOT,
            command="python eval_v2.py",
            generation_identity_ref="artifacts/gen_identity.json",
            eval_manifest_path="manifests/eval.jsonl",
            n_eval_rows=2,
            aggregate_identity=agg,
            evidence_tier="exact",
            evaluator_info={"name": "emotion2vec-v2", "version": "frozen-2026-01"},
        )
        assert out.is_file()
        reloaded = json.loads(out.read_text())
        assert reloaded["schema_version"] == 2
        assert reloaded["run_kind"] == "evaluate"
        assert reloaded["aggregate_identity"] == agg
        assert reloaded["evidence_tier"] == "exact"
        assert reloaded["generation_identity_ref"] == "artifacts/gen_identity.json"

    def test_generation_identity_with_patch_bundle(self, tmp_path):
        """dirty worktree → generation identity 保存 patch bundle。"""
        repo = _init_git_repo(tmp_path / "repo")
        (repo / "README.md").write_text("dirty change\n")

        patch_path = tmp_path / "gen.patch"
        out = tmp_path / "gen_identity.json"
        identity = write_emofilm_generation_identity(
            out,
            code_root=repo,
            command="python inference_v2.py",
            checkpoint_sha256=_CKPT_SHA,
            patch_bundle_path=patch_path,
        )
        assert identity["source"]["dirty"] is True
        assert identity["source"]["patch_bundle"] is not None
        assert patch_path.is_file()
        assert len(patch_path.read_bytes()) > 0

    def test_train_identity_ref_inline_dict(self, tmp_path):
        """train_identity_ref 接受内联 dict。"""
        out = tmp_path / "gen_identity.json"
        train_ref = {"schema_version": 2, "output_checkpoint": {"sha256": _CKPT_SHA}}
        identity = write_emofilm_generation_identity(
            out,
            code_root=ROOT,
            command="infer",
            train_identity_ref=train_ref,
        )
        reloaded = json.loads(out.read_text())
        assert reloaded["train_identity_ref"] == train_ref

    def test_eval_identity_references_generation_identity(self, tmp_path):
        """eval identity 通过 generation_identity_ref 关联 generation identity。"""
        gen_out = tmp_path / "gen_identity.json"
        gen_identity = write_emofilm_generation_identity(
            gen_out,
            code_root=ROOT,
            command="infer",
        )
        eval_out = tmp_path / "eval_identity.json"
        eval_identity = write_emofilm_evaluation_identity(
            eval_out,
            code_root=ROOT,
            command="eval",
            generation_identity_ref=str(gen_out),
        )
        reloaded = json.loads(eval_out.read_text())
        assert reloaded["generation_identity_ref"] == str(gen_out)

    def test_generation_to_eval_chain(self, tmp_path):
        """端到端身份链：train → generation → evaluation 引用链完整。"""
        # 1. Generation identity
        gen_rows = [_make_generation_row(utt_id=f"utt_{i:03d}") for i in range(3)]
        gen_out = tmp_path / "gen_identity.json"
        write_emofilm_generation_identity(
            gen_out,
            code_root=ROOT,
            command="infer",
            generation_manifest_path="gen.jsonl",
            n_generation_rows=3,
            train_identity_ref="train_identity.json",
        )

        # 2. Evaluation identity 引用 generation identity
        eval_rows = [_make_eval_row(utt_id=f"utt_{i:03d}") for i in range(3)]
        agg = compute_aggregate_identity(eval_rows)
        eval_out = tmp_path / "eval_identity.json"
        write_emofilm_evaluation_identity(
            eval_out,
            code_root=ROOT,
            command="eval",
            generation_identity_ref=str(gen_out),
            aggregate_identity=agg,
            n_eval_rows=3,
        )

        # 3. 验证链条可追溯
        gen_reloaded = json.loads(gen_out.read_text())
        eval_reloaded = json.loads(eval_out.read_text())
        assert gen_reloaded["train_identity_ref"] == "train_identity.json"
        assert eval_reloaded["generation_identity_ref"] == str(gen_out)
        assert eval_reloaded["aggregate_identity"] == agg

        # 4. aggregate 可以验证
        match, _ = verify_aggregate_identity(eval_rows, agg)
        assert match is True


# ============================================================
# E. v1 identity 原样保留
# ============================================================


class TestV1IdentityPreserved:
    """E. v1 ``write_run_identity`` 入口签名 + 行为不变。"""

    def test_v1_entry_signature_unchanged(self):
        """v1 ``write_run_identity`` 参数集不变。"""
        from tools.write_emofilm_run_identity import write_run_identity

        sig = inspect.signature(write_run_identity)
        params = set(sig.parameters)
        expected = {
            "output_path", "run_kind", "code_root", "contract_dir",
            "command", "seed", "base_checkpoint", "extra",
        }
        assert expected.issubset(params), (
            f"v1 signature lost params: {expected - params}"
        )

    def test_v1_entry_produces_schema_v1(self, tmp_path):
        """v1 ``write_run_identity`` 仍产出 schema_version=1 / contract_name=emofilm_v1。"""
        from tools.write_emofilm_run_identity import write_run_identity

        # 构造最小合同目录
        contract_dir = tmp_path / "contract"
        prov = contract_dir / "provenance"
        prov.mkdir(parents=True)
        (prov / "contract.json").write_text("{}")
        (prov / "sources.json").write_text("[]")
        (prov / "membership.json").write_text("{}")

        out = tmp_path / "v1_identity.json"
        identity = write_run_identity(
            out,
            run_kind="generate",
            code_root=ROOT,
            contract_dir=contract_dir,
            command="v1 command",
        )
        reloaded = json.loads(out.read_text())
        assert reloaded["schema_version"] == 1
        assert reloaded["contract_name"] == "emofilm_v1"

    def test_v1_code_identity_still_sha_only(self):
        """v1 ``code_identity`` 仍只存 diff SHA（不存 bytes）——v1 行为不变。

        v2 的修复在 ``capture_source_identity`` + patch_bundle，不改 v1 入口。
        """
        from tools.write_emofilm_run_identity import code_identity

        code_id = code_identity(ROOT)
        # v1 code_identity 的结构不变
        assert "git_head" in code_id
        assert "worktree_diff_sha256" in code_id
        assert "dirty" in code_id
        # v1 不返回 patch_bundle 字段（这不是 v1 的职责）
        assert "patch_bundle" not in code_id

    def test_v1_identity_not_backfilled(self, tmp_path):
        """v1 历史 identity 不被补写 patch bundle 或逐条身份。

        验证：``write_run_identity`` 的输出 JSON 不含 v2 新增字段
        （``source.patch_bundle`` / 逐条身份等）。
        """
        from tools.write_emofilm_run_identity import write_run_identity

        contract_dir = tmp_path / "contract"
        prov = contract_dir / "provenance"
        prov.mkdir(parents=True)
        (prov / "contract.json").write_text("{}")
        (prov / "sources.json").write_text("[]")
        (prov / "membership.json").write_text("{}")

        out = tmp_path / "v1_identity.json"
        write_run_identity(
            out,
            run_kind="generate",
            code_root=ROOT,
            contract_dir=contract_dir,
            command="v1",
        )
        reloaded = json.loads(out.read_text())
        # v1 不含 v2 新增的 source 块
        assert "source" not in reloaded
        # v1 code 块保持原有结构（不是 v2 的 source 块）
        assert "code" in reloaded
        assert "patch_bundle" not in reloaded.get("code", {})
