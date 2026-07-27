#!/usr/bin/env python3
"""EmoFiLM 运行身份单一权威模块（训练 / 生成 / 评测）。

ADR-0020 扁平化整合：本文件合并原 v2 identity 副本的逐条
generation/evaluation 身份与安全恢复链（ticket 11），成为唯一的 identity 模块。
v1 入口 ``write_run_identity`` 签名与行为原样保留（schema_version=1，
contract_name="emofilm_v1"——v1 历史产物身份，冻结只读）；
活跃 EmoFiLM 入口（``write_emofilm_train_identity`` /
``write_emofilm_generation_identity`` / ``write_emofilm_evaluation_identity``）
绑定 schema_version=2 逐条身份，活跃合同名 ``contract_name="emofilm"``。
产物身份用 ``wav_path`` + 结构化身份字段，不含 WAV 内容哈希字段（ADR-0020 §3）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple


_CONTRACT_REQUIRED = (
    "provenance/contract.json",
    "provenance/sources.json",
    "provenance/membership.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256(contract_dir: str | Path) -> str:
    """Hash contract metadata and core manifests, excluding large derived data."""
    root = Path(contract_dir).resolve()
    required = [root / relative for relative in _CONTRACT_REQUIRED]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing contract identity files: {missing}")

    paths = set(required)
    paths.update(root.glob("provenance/*"))
    for pattern in (
        "sources/**/manifest.jsonl",
        "sources/**/tagged.jsonl",
        "eval/**/manifest.jsonl",
        "splits/**/manifest.jsonl",
        "splits/**/parquet/data.list",
    ):
        paths.update(root.glob(pattern))

    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_value(code_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=code_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def code_identity(code_root: str | Path) -> dict[str, Any]:
    """Return immutable commit plus current worktree-diff identity."""
    root = Path(code_root).resolve()
    head = _git_value(root, "rev-parse", "HEAD")
    diff = None
    status = None
    if head is not None:
        try:
            diff_bytes = subprocess.run(
                ["git", "diff", "--binary", "HEAD", "--"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
            diff = hashlib.sha256(diff_bytes).hexdigest()
        except (OSError, subprocess.CalledProcessError):
            diff = None
        status = _git_value(root, "status", "--porcelain")
    return {
        "root": str(root),
        "git_head": head,
        "worktree_diff_sha256": diff,
        "dirty": bool(status),
    }


def _package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "torchaudio",
        "transformers",
        "fairseq",
        "timm",
        "HyperPyYAML",
        "pyarrow",
        "numpy",
    )
    return {
        name: next(
            (version for version in (importlib.metadata.version(name),) if version),
            None,
        )
        if _has_distribution(name)
        else None
        for name in names
    }


def _has_distribution(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _hardware_identity() -> dict[str, Any]:
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
        torch_version = torch.__version__
    except Exception as exc:  # pragma: no cover - environment-specific
        cuda = {"available": False, "error": str(exc)}
        torch_version = None
    return {
        "torch_version": torch_version,
        "cuda": cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def write_run_identity(
    output_path: str | Path,
    *,
    run_kind: str,
    code_root: str | Path,
    contract_dir: str | Path,
    command: str,
    seed: int | None = None,
    base_checkpoint: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and atomically write a run identity JSON document."""
    output = Path(output_path)
    checkpoint = Path(base_checkpoint).resolve() if base_checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    identity: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": run_kind,
        "contract_name": "emofilm_v1",
        "contract_hash": contract_sha256(contract_dir),
        "code": code_identity(code_root),
        "command": command,
        "seed": seed,
        "base_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
        if checkpoint is not None
        else None,
        "python": sys.version,
        "packages": _package_versions(),
        "hardware": _hardware_identity(),
    }
    if extra:
        identity["extra"] = dict(extra)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return identity


# ============================================================
# ticket 07 — EmoFiLM 训练身份（扩展；保留 v1 ``write_run_identity`` 入口签名不变）
# ============================================================
#
# 活跃 EmoFiLM 身份（schema_version=2，contract_name="emofilm"）绑定：
#   - 数据合同（contract_hash 或 None）；
#   - base checkpoint（path + sha256）；
#   - resolved config（实际使用的 train_conf，含三组 LR/WD + scheduler）；
#   - 参数组统计（来自 ``summarize_optimizer_identity``：每组 tensor_count /
#     param_count / initial_lr / weight_decay + scheduler type/key_params）；
#   - seed；
#   - 输出 checkpoint（path + sha256）；
#   - 源码身份（干净 git revision → source.git_head；dirty worktree → patch_bundle
#     能力暴露，保存 ``git diff --binary HEAD`` 到 ``patch_bundle_path`` 并记录
#     sha256；与 ticket 11 协调）。
#
# v1 入口 ``write_run_identity`` 完全保留（contract_name="emofilm_v1" 历史产物身份；
# ADR-0020：v1 基线锚 git commit 9c6d84b，活跃代码可演化）。**关键不变量**是
# ``write_run_identity`` 的**行为与签名**不变：测试
# ``test_v1_entry_signature_unchanged`` 断言签名参数集合不变（8 参数兼容锁）。


def _save_patch_bundle(code_root: Path, output_patch: Path) -> dict[str, Any]:
    """保存 dirty worktree 的完整 ``git diff --binary HEAD`` 到不可变 patch bundle。

    覆盖 **tracked 改动 + untracked 新文件**（EmoFiLM 主线补救 #7）：旧实现
    ``git diff --binary HEAD`` 只比对 worktree 与真实 index，遗漏未 ``git add``
    的新文件 → dirty worktree 不可重建。

    修复用 **GIT_INDEX_FILE 隔离临时 index** 方案：
      1. ``read-tree HEAD`` 把临时 index 初始化为 HEAD 状态；
      2. ``add -A`` 把 worktree 全部（含 untracked）写入**临时 index**；
      3. ``diff --binary --cached HEAD`` 输出完整 patch（tracked 改动 + 新增文件）。

    真实 ``.git/index`` 与 worktree **均不动**——``git add -A`` 是在隔离的临时
    index 上操作，不是真实 ``git add``（ADR-0020 §7 禁止改真实 git 状态）。

    返回 ``{path, sha256, size_bytes}``；sha256 是产物完整性哈希（bundle 内容
    sha256），用于 v2 identity 的 ``source.patch_bundle`` 字段（ticket 11 续训/重建）。
    """
    # 临时 index 文件路径放 patch 同目录（同 fs，可原子 unlink；不污染 .git/index）
    output_patch.parent.mkdir(parents=True, exist_ok=True)
    tmp_index = output_patch.parent / f".git_index_{os.getpid()}.tmp"
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp_index)}
    try:
        subprocess.run(
            ["git", "read-tree", "HEAD"],
            cwd=code_root, check=True, env=env,
            capture_output=True,
        )
        # 写入临时 index（含 untracked）；真实 .git/index 不动
        subprocess.run(
            ["git", "add", "-A"],
            cwd=code_root, check=True, env=env,
            capture_output=True,
        )
        diff_bytes = subprocess.run(
            ["git", "diff", "--binary", "--cached", "HEAD", "--"],
            cwd=code_root, check=True, env=env,
            capture_output=True,
        ).stdout
    finally:
        # 无论成功/失败都清理临时 index，不残留
        try:
            tmp_index.unlink(missing_ok=True)
        except OSError:
            pass
    output_patch.write_bytes(diff_bytes)
    return {
        "path": str(output_patch),
        "sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "size_bytes": len(diff_bytes),
    }


def write_emofilm_train_identity(
    output_path: str | Path,
    *,
    run_kind: str,
    code_root: str | Path,
    contract_dir: str | Path | None,
    command: str,
    seed: int | None = None,
    base_checkpoint: str | Path | None = None,
    resolved_config: dict[str, Any] | None = None,
    optimizer_identity: dict[str, Any] | None = None,
    output_checkpoint: str | Path | None = None,
    patch_bundle_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建并原子写入 EmoFiLM 训练身份 JSON。

    保留 v1 ``write_run_identity`` 入口签名与行为不变（contract_name="emofilm_v1"
    历史产物身份；ADR-0020 v1 基线锚 git commit 9c6d84b）；本函数是活跃 EmoFiLM
    入口（brief 07 §B）。

    身份族（schema_version=2）绑定：
        - ``contract_name`` = ``"emofilm"``（活跃合同名）；``contract_hash`` 来自
          ``contract_dir``（None → 字段为 None，允许测试/无合同场景）。
        - ``resolved_config``：实际使用的 train_conf（三组 LR/WD + scheduler）。
        - ``optimizer_identity``：``summarize_optimizer_identity`` 产出（每组
          tensor/param count + initial LR + weight_decay + scheduler type/key_params）。
        - ``source``：``{git_head, dirty, patch_bundle?}``。dirty worktree 且提供
          ``patch_bundle_path`` → 保存 ``git diff --binary HEAD`` 并记录 sha256
          （patch-bundle 能力；与 ticket 11 协调）。
        - ``base_checkpoint`` / ``output_checkpoint``：path + sha256（若提供）。
        - ``seed`` / ``command`` / ``python`` / ``packages`` / ``hardware``。
    """
    output = Path(output_path)
    root = Path(code_root).resolve()

    # 合同 hash（允许 None：测试 / 无合同目录场景）
    c_hash = None
    if contract_dir is not None:
        c_hash = contract_sha256(contract_dir)

    # base ckpt
    base_block = None
    if base_checkpoint is not None:
        base_path = Path(base_checkpoint).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        base_block = {"path": str(base_path), "sha256": sha256_file(base_path)}

    # output ckpt
    output_block = None
    if output_checkpoint is not None:
        out_path = Path(output_checkpoint).resolve()
        if not out_path.is_file():
            raise FileNotFoundError(out_path)
        output_block = {"path": str(out_path), "sha256": sha256_file(out_path)}

    # source 身份（干净 revision 或 patch bundle）
    code_id = code_identity(root)
    source_block: dict[str, Any] = {
        "git_head": code_id["git_head"],
        "dirty": code_id["dirty"],
        "worktree_diff_sha256": code_id["worktree_diff_sha256"],
    }
    if code_id["dirty"] and patch_bundle_path is not None:
        source_block["patch_bundle"] = _save_patch_bundle(
            root, Path(patch_bundle_path)
        )

    identity: dict[str, Any] = {
        "schema_version": 2,
        "run_kind": run_kind,
        "contract_name": "emofilm",
        "contract_hash": c_hash,
        "code_root": str(root),
        "source": source_block,
        "command": command,
        "seed": seed,
        "base_checkpoint": base_block,
        "resolved_config": resolved_config,
        "optimizer_identity": optimizer_identity,
        "output_checkpoint": output_block,
        "python": sys.version,
        "packages": _package_versions(),
        "hardware": _hardware_identity(),
    }
    if extra:
        identity["extra"] = dict(extra)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="write emofilm_v1 run identity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-kind", required=True, choices=["train", "smoke", "generate", "evaluate"])
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-checkpoint", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    identity = write_run_identity(
        args.output,
        run_kind=args.run_kind,
        code_root=args.code_root,
        contract_dir=args.contract_dir,
        command=args.command,
        seed=args.seed,
        base_checkpoint=args.base_checkpoint,
    )
    print(json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True))


# ============================================================
# ticket 11 — 逐条运行身份与安全恢复链（ADR-0020 自原 v2 identity 副本合并）
# ============================================================
#
# 本段闭合训练 / 生成 / 评测的逐条身份绑定，修复 v1 的三个确定缺陷
# （MAP §2 identity 段）：
#
# 1. **dirty-worktree 不可重建**：v1 ``code_identity`` 只存 diff SHA-256 hash，
#    不存实际 diff bytes → dirty worktree 无法重建。本段保存可应用的不可变
#    source patch bundle（``git diff --binary HEAD`` 实际字节），使运行可复现。
# 2. **skip-existing 无逐条身份**：v1 ``inference_emo_film.py`` L182-184 仅检查
#    ``os.path.isfile(out_wav)`` → 134 条无验证恢复。本段的 ``check_skip_existing``
#    仅当既有 WAV **及其完整逐条身份**（checkpoint/control/prompt/decode_config/
#    源码）与当前请求全部一致时才复用；任一不一致 → 明确拒绝。
# 3. **评测不消费 generation manifests**：v1 评测只返回 aggregate，不持久化逐样本，
#    无逐条 eval-row 身份。本段绑定 generation-row/control-span/evaluator/metrics，
#    并通过 aggregate identity（有序集合 hash）检测 rows 被替换/遗漏/混入其他运行。
#
# v1 入口 ``write_run_identity`` 签名与行为完全保留（MAP §0 v1 只读）。


# ============================================================
# 常量
# ============================================================

# Generation row 中决定可重建性的身份字段族。
# 这些字段参与 generation_row_identity_fingerprint 的计算。
GENERATION_SOURCE_KEYS = (
    "source_revision",
    "source_patch_bundle",
    "source_patch_sha256",
    "source",
)
GENERATION_CHECKPOINT_KEYS = (
    "checkpoint_sha256",
    "checkpoint_ref",
)
GENERATION_CONTROL_KEYS = (
    "control_row_ref",
    "control_row",
)
GENERATION_PROMPT_KEYS = (
    "prompt_row_ref",
    "prompt_row",
)

# Evaluation row 中决定身份的字段族。
EVAL_GENERATION_KEYS = (
    "generation_row_ref",
    "generation_row",
)
EVAL_CONTROL_KEYS = (
    "control_span_ref",
    "control_span",
)


class SkipDecision(NamedTuple):
    """skip-existing 判定结果。

    Attributes:
        skip: True 表示可以安全复用既有 WAV；False 表示必须拒绝/重新生成。
        reason: 人类可读的判定原因（用于日志 / hard-fail 消息）。
    """

    skip: bool
    reason: str


# ============================================================
# 内部 helper：规范化序列化与哈希
# ============================================================


def _canonical_json(obj: Any) -> str:
    """确定性 JSON 序列化（排序键、紧凑分隔符）。

    用于身份指纹计算：相同内容 → 相同字节 → 相同 SHA-256。
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_source_digest(row: Mapping[str, Any]) -> str:
    """从 generation row 或请求中提取源码身份摘要。

    支持多种源码身份表达方式，选取最具体的不可变标识符：
    - ``source_revision``: 干净 git revision sha（字符串）。
    - ``source_patch_sha256``: patch bundle 的 sha256（字符串）。
    - ``source_patch_bundle``: ``{path, sha256, size_bytes}`` 字典。
    - ``source``: 07 的 source 块 ``{git_head, dirty, worktree_diff_sha256, patch_bundle?}``。

    对 ``source`` 块：dirty+patch_bundle → patch sha；dirty 无 bundle → diff sha；
    clean → git_head。确保同一运行的不同表达方式产生相同摘要。
    """
    if "source_revision" in row:
        rev = row["source_revision"]
        if isinstance(rev, str) and rev.strip():
            return rev

    if "source_patch_sha256" in row:
        sha = row["source_patch_sha256"]
        if isinstance(sha, str) and sha.strip():
            return sha

    bundle = row.get("source_patch_bundle")
    if isinstance(bundle, Mapping):
        sha = bundle.get("sha256")
        if isinstance(sha, str) and sha.strip():
            return sha

    source = row.get("source")
    if isinstance(source, Mapping):
        # dirty + patch bundle → patch sha（最具体的不可变标识符）
        pb = source.get("patch_bundle")
        if isinstance(pb, Mapping):
            sha = pb.get("sha256")
            if isinstance(sha, str) and sha.strip():
                return sha
        # dirty 无 bundle → diff sha
        diff_sha = source.get("worktree_diff_sha256")
        if isinstance(diff_sha, str) and diff_sha.strip():
            return diff_sha
        # clean → git_head
        head = source.get("git_head")
        if isinstance(head, str) and head.strip():
            return head

    return ""


def _extract_ref_or_mapping_digest(
    row: Mapping[str, Any],
    *,
    str_ref_key: str | None = None,
    mapping_key: str | None = None,
    extra_str_key: str | None = None,
    inner_sha_key: str | None = None,
) -> str:
    """从 row 中提取身份摘要的统一模板。

    选取最具体的不可变标识符，优先级：

    1. ``extra_str_key``（如 ``checkpoint_sha256``）：顶层非空字符串。
    2. ``str_ref_key``（如 ``control_row_ref``）：非空字符串引用。
    3. ``mapping_key``（如 ``control_row``）：Mapping → 若 ``inner_sha_key``
       指定的内层 sha（如 ``checkpoint_ref.sha256``）非空则取之，否则取
       规范化 JSON。Mapping 为空或类型不符 → ``""``。

    该模板覆盖 checkpoint / control / prompt / generation_ref / control_span
    五种同构身份摘要；``_extract_source_digest`` 因多级嵌套结构不同而独立保留。
    """
    if extra_str_key is not None:
        val = row.get(extra_str_key)
        if isinstance(val, str) and val.strip():
            return val

    if str_ref_key is not None:
        ref = row.get(str_ref_key)
        if isinstance(ref, str) and ref.strip():
            return ref

    if mapping_key is not None:
        obj = row.get(mapping_key)
        if isinstance(obj, Mapping):
            if inner_sha_key is not None:
                inner = obj.get(inner_sha_key)
                if isinstance(inner, str) and inner.strip():
                    return inner
            return _canonical_json(dict(obj))

    return ""


# ============================================================
# A. 源码身份（修复 v1 SHA-only 不可重建）
# ============================================================


def capture_source_identity(
    code_root: str | Path,
    patch_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """捕获源码身份：干净 revision 或 dirty worktree 的不可变 patch bundle。

    返回::

        {
            "git_head": str | None,
            "dirty": bool,
            "worktree_diff_sha256": str | None,
            "patch_bundle": {            # 仅当 dirty 且 patch_bundle_path 提供
                "path": str,
                "sha256": str,
                "size_bytes": int,
            } | None,
        }

    与本模块 ``write_emofilm_train_identity`` 的 ``source`` 块结构一致；
    生成/评测 identity 通过引用本函数的产出来绑定源码身份。

    若 ``patch_bundle_path`` 不为 None **且** worktree dirty，则保存
    ``git diff --binary HEAD`` 的**实际字节**到 ``patch_bundle_path``（不只是
    SHA-256 hash），使运行可通过 ``git apply`` 重建。
    """
    root = Path(code_root).resolve()
    code_id = code_identity(root)

    source_block: dict[str, Any] = {
        "git_head": code_id["git_head"],
        "dirty": code_id["dirty"],
        "worktree_diff_sha256": code_id["worktree_diff_sha256"],
        "patch_bundle": None,
    }

    if code_id["dirty"] and patch_bundle_path is not None:
        source_block["patch_bundle"] = _save_patch_bundle(
            root, Path(patch_bundle_path)
        )

    return source_block


def build_source_revision_or_patch(
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """将 ``capture_source_identity`` 的产出规范化为 GenerationRow 的源码身份字段。

    干净 worktree → ``source_revision = git_head``。
    Dirty worktree 且有 patch_bundle → ``source_patch_bundle`` + ``source_patch_sha256``。
    Dirty worktree 无 patch_bundle → ``source_patch_sha256 = worktree_diff_sha256``
    （记录 SHA 但标记为不可完全重建；仍可用于身份比对）。
    """
    result: dict[str, Any] = {}
    git_head = source_identity.get("git_head")
    dirty = source_identity.get("dirty", False)
    bundle = source_identity.get("patch_bundle")
    diff_sha = source_identity.get("worktree_diff_sha256")

    if not dirty and isinstance(git_head, str) and git_head.strip():
        result["source_revision"] = git_head
    elif isinstance(bundle, Mapping):
        result["source_patch_bundle"] = dict(bundle)
        sha = bundle.get("sha256")
        if isinstance(sha, str) and sha.strip():
            result["source_patch_sha256"] = sha
    elif isinstance(diff_sha, str) and diff_sha.strip():
        # dirty 但未保存 patch bytes：记录 SHA 用于比对，但不可重建。
        result["source_patch_sha256"] = diff_sha

    return result


# ============================================================
# B. 逐条 generation 身份 + 安全 skip-existing
# ============================================================


def generation_row_identity_components(row: Mapping[str, Any]) -> dict[str, Any]:
    """提取 generation row 的身份组件（每族摘要，供 fingerprint 与 skip 共用，DRY）。

    覆盖决定可重建性的完整身份族 + 合成输入摘要：
    - 源码 / checkpoint / 控制 row / prompt row / 解码配置 / 随机种子
    - ``text_digest``（合成文本摘要）+ ``prompt_audio_ref``（prompt 音频身份）：
      B6 纳入指纹以反映**实际合成内容**——内容变更 → 指纹变 → 不复用旧 WAV
      （v1 仅锁 ref 字符串，改 manifest 续跑会静默复用旧条件音频）。

    ``finish_reason`` / ``wav_path`` 是输出，不纳入请求身份指纹。
    """
    return {
        "source": _extract_source_digest(row),
        "checkpoint": _extract_ref_or_mapping_digest(
            row,
            extra_str_key="checkpoint_sha256",
            mapping_key="checkpoint_ref",
            inner_sha_key="sha256",
        ),
        "control": _extract_ref_or_mapping_digest(
            row, str_ref_key="control_row_ref", mapping_key="control_row",
        ),
        "prompt": _extract_ref_or_mapping_digest(
            row, str_ref_key="prompt_row_ref", mapping_key="prompt_row",
        ),
        "decode_config": _canonical_json(dict(row.get("decode_config", {}))),
        "seed": row.get("seed"),
        "text_digest": row.get("text_digest") or "",
        "prompt_audio_ref": row.get("prompt_audio_ref") or "",
    }


def generation_row_identity_fingerprint(row: Mapping[str, Any]) -> str:
    """计算 generation row 的逐条身份指纹（SHA-256），覆盖四族身份 + 解码配置 +
    seed + 合成输入摘要（text_digest / prompt_audio_ref）。

    注意（ADR-0020 §3）：WAV 内容哈希已移除，产物身份用 ``wav_path`` + 结构化
    身份字段；``text_digest`` 是**输入文本**摘要（不在哈希禁区——禁区是源码
    哈希锁与 WAV 产物哈希）。``check_skip_existing`` 单独验证输出有效性。
    """
    return _sha256_text(_canonical_json(generation_row_identity_components(row)))


def generation_request_fingerprint(
    *,
    source: Mapping[str, Any] | str,
    checkpoint_sha256: str | None = None,
    checkpoint_ref: Mapping[str, Any] | None = None,
    control_row_ref: str | None = None,
    control_row: Mapping[str, Any] | None = None,
    prompt_row_ref: str | None = None,
    prompt_row: Mapping[str, Any] | None = None,
    decode_config: Mapping[str, Any] | None = None,
    source_revision: str | None = None,
    source_patch_sha256: str | None = None,
    source_patch_bundle: Mapping[str, Any] | None = None,
    seed: int | None = None,
    text_digest: str | None = None,
    prompt_audio_ref: str | None = None,
) -> str:
    """计算生成请求的身份指纹，用于与既有 generation row 比对。

    接受灵活的源码/控制/prompt 表达方式；内部统一走
    ``generation_row_identity_fingerprint`` 的规范化逻辑。

    ``seed``：per-request 固定随机种子。传入时纳入指纹 payload；
    不传时 ``row.get("seed")`` 返回 ``None``（与无 seed 的 row 匹配）。
    """
    row: dict[str, Any] = {
        "decode_config": dict(decode_config or {}),
    }
    if seed is not None:
        row["seed"] = seed

    # 源码身份
    if isinstance(source, Mapping):
        row["source"] = dict(source)
    elif isinstance(source, str) and source.strip():
        row["source_revision"] = source
    if source_revision:
        row["source_revision"] = source_revision
    if source_patch_sha256:
        row["source_patch_sha256"] = source_patch_sha256
    if source_patch_bundle is not None:
        row["source_patch_bundle"] = dict(source_patch_bundle)

    # checkpoint
    if checkpoint_sha256:
        row["checkpoint_sha256"] = checkpoint_sha256
    if checkpoint_ref is not None:
        row["checkpoint_ref"] = dict(checkpoint_ref)

    # control
    if control_row_ref:
        row["control_row_ref"] = control_row_ref
    if control_row is not None:
        row["control_row"] = dict(control_row)

    # prompt
    if prompt_row_ref:
        row["prompt_row_ref"] = prompt_row_ref
    if prompt_row is not None:
        row["prompt_row"] = dict(prompt_row)

    # B6: 合成输入摘要（文本摘要 + prompt 音频身份），纳入指纹
    if text_digest:
        row["text_digest"] = text_digest
    if prompt_audio_ref:
        row["prompt_audio_ref"] = prompt_audio_ref

    return generation_row_identity_fingerprint(row)


def check_skip_existing(
    existing_row: Mapping[str, Any],
    request_fingerprint: str,
    workspace_root: str | Path | None = None,
) -> SkipDecision:
    """判定是否可以安全复用既有 generation row 的 WAV。

    安全复用条件（**全部**满足）：
    1. 既有 row 的 ``finish_reason == "eos"``（只有 eos 进正式 WAV）。
    2. 既有 row 有有效的 ``wav_path``。
    3. ``wav_path`` 指向的 WAV 文件在磁盘上实际存在（ADR-0020 §3：safe-resume
       仅加 ``os.path.isfile`` 存在性检查，**禁止** wav_sha256 内容比对——
       已删除/被替换的 WAV 不再被安全跳过并复用）。
    4. 既有 row 的逐条身份指纹与请求指纹**完全一致**
       （checkpoint/control/prompt/decode_config/源码 全部匹配）。

    任一不满足 → ``skip=False``，返回明确的拒绝原因（修复 v1 L182-184
    的 134 条无验证恢复）。

    ``workspace_root``：可选，用于解析 **workspace-relative** 的 ``wav_path``
    （schema 要求 POSIX 相对路径，如 ``"wav/esd-0001.wav"``）。传入时 isfile
    检查以 ``workspace_root / wav_path`` 为绝对路径；为 ``None`` 时按既有行为
    用进程 CWD 解析（兼容存绝对路径 ``wav_path`` 的旧调用方）。
    """
    finish_reason = existing_row.get("finish_reason")
    if finish_reason != "eos":
        return SkipDecision(
            skip=False,
            reason=(
                f"existing row finish_reason={finish_reason!r} is not 'eos'; "
                "cannot reuse non-eos output (only eos enters formal WAV)"
            ),
        )

    wav_path = existing_row.get("wav_path")
    if not isinstance(wav_path, str) or not wav_path.strip():
        return SkipDecision(
            skip=False,
            reason="existing row has no valid wav_path",
        )

    # ADR-0020 §3 哈希边界：safe-resume 仅做 os.path.isfile 存在性检查，
    # 不引入 wav_sha256 内容比对。WAV 已删除/被替换 → 必须重新生成。
    # wav_path 通常是 workspace-relative POSIX 路径，需用 workspace_root 解析
    # 为绝对路径再 isfile（否则按进程 CWD 解析，workspace 相对路径永远找不到）。
    if workspace_root is not None:
        resolved_wav_path: Path = Path(workspace_root) / wav_path
    else:
        resolved_wav_path = Path(wav_path)
    if not resolved_wav_path.is_file():
        return SkipDecision(
            skip=False,
            reason=(
                f"existing wav file not found on disk: wav_path={wav_path!r} "
                "(file deleted or replaced) — must regenerate"
            ),
        )

    # B6: 既有 row 四族身份任一缺失 → 视为无 existing（防"无身份"被当"身份匹配"，
    # 退化路径：缺身份 row 之间 ""=="" 会误复用任意内容）。
    existing_comps = generation_row_identity_components(existing_row)
    for _fam in ("source", "checkpoint", "control", "prompt"):
        if not existing_comps[_fam]:
            return SkipDecision(
                skip=False,
                reason=(
                    f"existing row missing {_fam} identity (empty digest) "
                    "— treated as no existing (regenerate)"
                ),
            )

    existing_fp = generation_row_identity_fingerprint(existing_row)
    if existing_fp != request_fingerprint:
        return SkipDecision(
            skip=False,
            reason=(
                f"per-row identity mismatch: "
                f"existing={existing_fp[:16]}… request={request_fingerprint[:16]}… "
                "(checkpoint/control/prompt/decode_config/source differ) — "
                "must regenerate (v1 would have silently reused)"
            ),
        )

    return SkipDecision(
        skip=True,
        reason="per-row identity match: safe to reuse existing WAV",
    )


# ============================================================
# C. 逐条 evaluation 身份 + aggregate 身份
# ============================================================


def eval_row_identity_fingerprint(row: Mapping[str, Any]) -> str:
    """计算 evaluation row 的逐条身份指纹（SHA-256）。

    指纹覆盖：
    - generation row 引用（generation_row_ref / generation_row）
    - 控制 span 引用（control_span_ref / control_span）
    - evaluator 身份（name + version + label_space 等）
    - boundary_evidence_tier（exact / approximate）
    - metrics（逐样本/逐 span 指标——绑定用于审计）

    任一变化 → 不同指纹 → aggregate 检测到不一致。
    """
    payload = _canonical_json({
        "generation": _extract_ref_or_mapping_digest(
            row, str_ref_key="generation_row_ref", mapping_key="generation_row",
        ),
        "control_span": _extract_ref_or_mapping_digest(
            row, str_ref_key="control_span_ref", mapping_key="control_span",
        ),
        "evaluator": _canonical_json(dict(row.get("evaluator", {}))),
        "boundary_evidence_tier": str(row.get("boundary_evidence_tier", "")),
        "metrics": _canonical_json(dict(row.get("metrics", {}))),
    })
    return _sha256_text(payload)


def compute_aggregate_identity(
    eval_rows: list[Mapping[str, Any]],
) -> str:
    """计算 evaluation rows 集合的有序身份 hash。

    确定性派生：
    - 按 ``utt_id`` 排序（保证顺序无关）。
    - 每行计算 ``eval_row_identity_fingerprint``。
    - 对 ``{n_rows, row_fingerprints}`` 做规范化 JSON 哈希。

    能检测：
    - **rows 被替换**：同一 utt_id 位置的行指纹变化 → 集合 hash 变化。
    - **rows 遗漏**：n_rows 减少 + 指纹列表变化 → 集合 hash 变化。
    - **混入其他运行**：来自不同 evaluator/generation 的行指纹不同 → 集合 hash 变化。
    """
    sorted_rows = sorted(
        eval_rows,
        key=lambda r: str(r.get("utt_id", "")),
    )
    fingerprints = [eval_row_identity_fingerprint(r) for r in sorted_rows]
    payload = _canonical_json({
        "n_rows": len(fingerprints),
        "row_fingerprints": fingerprints,
    })
    return _sha256_text(payload)


def verify_aggregate_identity(
    eval_rows: list[Mapping[str, Any]],
    expected: str,
) -> tuple[bool, str]:
    """验证 evaluation rows 集合是否与预期的 aggregate identity 一致。

    返回 ``(match, reason)``。``match=False`` 时 ``reason`` 描述差异类型
    （行数变化 / 行被替换 / 混入其他运行）。
    """
    actual = compute_aggregate_identity(eval_rows)
    if actual == expected:
        return True, "aggregate identity matches"

    # 差异诊断
    n_actual = len(eval_rows)
    sorted_rows = sorted(eval_rows, key=lambda r: str(r.get("utt_id", "")))
    actual_fps = [eval_row_identity_fingerprint(r) for r in sorted_rows]

    return False, (
        f"aggregate identity mismatch: expected={expected[:16]}… "
        f"actual={actual[:16]}… "
        f"(n_rows={n_actual}; "
        "rows replaced/missing/mixed — recompute to diagnose)"
    )


# ============================================================
# D. 生成 / 评测 identity 写入器（与训练 identity 协调）
# ============================================================


def _write_atomic_json(output_path: Path, data: dict[str, Any]) -> None:
    """原子写入 JSON（避免部分写入）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, output_path)


def write_emofilm_generation_identity(
    output_path: str | Path,
    *,
    code_root: str | Path,
    command: str,
    train_identity_ref: str | Mapping[str, Any] | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_ref: Mapping[str, Any] | None = None,
    decode_config_defaults: Mapping[str, Any] | None = None,
    generation_manifest_path: str | None = None,
    n_generation_rows: int | None = None,
    aggregate_generation_fingerprint: str | None = None,
    patch_bundle_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建并原子写入 EmoFiLM 生成运行身份 JSON。

    串联 A（源码身份）+ B（逐条 generation 身份）：
    - 捕获 source identity（干净 revision 或 dirty→patch bundle）。
    - 引用训练 identity（``train_identity_ref``）以关联 checkpoint 来源。
    - 记录生成 manifest 路径 + 行数 + generation rows 的集合指纹。

    ``train_identity_ref`` 可以是 identity JSON 的路径（str）或内联 dict，
    指向 ``write_emofilm_train_identity`` 的产出。
    """
    source_identity = capture_source_identity(code_root, patch_bundle_path)

    # checkpoint 引用规范化
    ckpt_block: dict[str, Any] | None = None
    if checkpoint_sha256:
        ckpt_block = {"sha256": checkpoint_sha256}
        if checkpoint_ref is not None:
            ckpt_block.update(dict(checkpoint_ref))
    elif checkpoint_ref is not None:
        ckpt_block = dict(checkpoint_ref)

    identity: dict[str, Any] = {
        "schema_version": 2,
        "run_kind": "generate",
        "contract_name": "emofilm",
        "source": source_identity,
        "command": command,
        "train_identity_ref": (
            str(train_identity_ref)
            if isinstance(train_identity_ref, (str, Path))
            else dict(train_identity_ref)
            if train_identity_ref is not None
            else None
        ),
        "checkpoint": ckpt_block,
        "decode_config_defaults": dict(decode_config_defaults or {}),
        "generation_manifest_path": generation_manifest_path,
        "n_generation_rows": n_generation_rows,
        "aggregate_generation_fingerprint": aggregate_generation_fingerprint,
        "python": sys.version,
    }
    if extra:
        identity["extra"] = dict(extra)

    _write_atomic_json(Path(output_path), identity)
    return identity


def write_emofilm_evaluation_identity(
    output_path: str | Path,
    *,
    code_root: str | Path,
    command: str,
    generation_identity_ref: str | Mapping[str, Any] | None = None,
    eval_manifest_path: str | None = None,
    n_eval_rows: int | None = None,
    aggregate_identity: str | None = None,
    evidence_tier: str | None = None,
    evaluator_info: Mapping[str, Any] | None = None,
    patch_bundle_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建并原子写入 EmoFiLM 评测运行身份 JSON。

    串联 A（源码身份）+ C（逐条 eval 身份 + aggregate 身份）：
    - 捕获 source identity。
    - 引用 generation identity（``generation_identity_ref``）。
    - 记录 eval manifest + 行数 + aggregate identity（有序集合 hash）。
    - 记录 evidence_tier（exact / approximate）以支持 aggregate 分离。

    ``aggregate_identity`` 来自 ``compute_aggregate_identity(eval_rows)``；
    消费方通过 ``verify_aggregate_identity`` 检测 rows 被替换/遗漏/混入。
    """
    source_identity = capture_source_identity(code_root, patch_bundle_path)

    identity: dict[str, Any] = {
        "schema_version": 2,
        "run_kind": "evaluate",
        "contract_name": "emofilm",
        "source": source_identity,
        "command": command,
        "generation_identity_ref": (
            str(generation_identity_ref)
            if isinstance(generation_identity_ref, (str, Path))
            else dict(generation_identity_ref)
            if generation_identity_ref is not None
            else None
        ),
        "eval_manifest_path": eval_manifest_path,
        "n_eval_rows": n_eval_rows,
        "aggregate_identity": aggregate_identity,
        "evidence_tier": evidence_tier,
        "evaluator_info": dict(evaluator_info) if evaluator_info else None,
        "python": sys.version,
    }
    if extra:
        identity["extra"] = dict(extra)

    _write_atomic_json(Path(output_path), identity)
    return identity


# 逐条身份 / 安全恢复链的公开 API（自原 v2 identity 副本合并）。
__all__ = [
    "SkipDecision",
    "GENERATION_SOURCE_KEYS",
    "GENERATION_CHECKPOINT_KEYS",
    "GENERATION_CONTROL_KEYS",
    "GENERATION_PROMPT_KEYS",
    "EVAL_GENERATION_KEYS",
    "EVAL_CONTROL_KEYS",
    "capture_source_identity",
    "build_source_revision_or_patch",
    "generation_row_identity_fingerprint",
    "generation_request_fingerprint",
    "check_skip_existing",
    "eval_row_identity_fingerprint",
    "compute_aggregate_identity",
    "verify_aggregate_identity",
    "write_emofilm_train_identity",
    "write_emofilm_generation_identity",
    "write_emofilm_evaluation_identity",
]


if __name__ == "__main__":
    main()
