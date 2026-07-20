#!/usr/bin/env python3
"""写入 emofilm_v1 训练、生成和评测运行身份。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


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


if __name__ == "__main__":
    main()
