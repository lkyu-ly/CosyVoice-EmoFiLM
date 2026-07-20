#!/usr/bin/env python3
"""使用 EmoFiLM emotion2vec-base 合同提取 768d/50Hz 帧特征。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


EMOFILM_FEATURE_DIM = 768
EMOFILM_FRAME_RATE_HZ = 50.0
EMOFILM_FRAME_STEP_MS = 20.0


@dataclass
class UserDirModule:
    user_dir: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative file names and bytes in a deterministic directory order."""
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file() or "__pycache__" in child.parts or child.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract EmoFiLM emotion2vec-base frame features"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream_dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument(
        "--workspace_root",
        type=Path,
        default=Path.cwd(),
        help="workspace root used to resolve relative wav_path values",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    return parser


def resolve_workspace_wav(path: str | Path, workspace_root: str | Path) -> Path:
    """Resolve a manifest wav path against the explicit workspace root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(workspace_root) / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def validate_provenance(
    path: Path,
    *,
    checkpoint: Path | None = None,
    upstream_dir: Path | None = None,
) -> dict[str, Any]:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_id", "revision", "checkpoint_sha256", "upstream"}
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"missing provenance fields: {sorted(missing)}")
    digest = str(provenance["checkpoint_sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("checkpoint_sha256 must be a SHA-256 hex string")
    if provenance["model_id"] != "emotion2vec-base":
        raise ValueError("EmoFiLM frame extractor requires emotion2vec-base")
    upstream = provenance["upstream"]
    if not isinstance(upstream, dict):
        raise ValueError("upstream provenance must contain path and sha256")
    upstream_digest = str(upstream.get("sha256", ""))
    if len(upstream_digest) != 64:
        raise ValueError("upstream sha256 must be a SHA-256 hex string")
    if checkpoint is not None and file_sha256(checkpoint) != digest:
        raise ValueError("checkpoint SHA-256 mismatch")
    if upstream_dir is not None and directory_sha256(upstream_dir) != upstream_digest:
        raise ValueError("upstream directory SHA-256 mismatch")
    return provenance


def _validate_features(features: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(features):
        raise ValueError("emotion2vec features must be a tensor")
    if features.ndim != 2 or features.shape[1] != EMOFILM_FEATURE_DIM:
        raise ValueError(
            f"EmoFiLM emotion2vec features must have shape (T, {EMOFILM_FEATURE_DIM}), "
            f"got {tuple(features.shape)}"
        )
    return features.detach().cpu().float().contiguous()


def extract_frame_features(
    model: Any,
    task: Any,
    wav_path: Path,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    """提取并验证单条音频的 EmoFiLM 帧特征。"""
    import soundfile as sf
    import torch.nn.functional as F

    wav, actual_rate = sf.read(str(wav_path), dtype="float32")
    if actual_rate != sample_rate:
        raise ValueError(
            f"sample rate mismatch for {wav_path}: expected {sample_rate}, "
            f"got {actual_rate}"
        )
    if wav.ndim == 2:
        if wav.shape[1] != 1:
            raise ValueError(f"expected mono wav, got {wav.shape[1]} channels: {wav_path}")
        wav = wav[:, 0]
    if wav.ndim != 1:
        raise ValueError(f"expected mono waveform: {wav_path}")

    source = torch.from_numpy(np.asarray(wav)).float().to(device).view(1, -1)
    if getattr(getattr(task, "cfg", None), "normalize", False):
        source = F.layer_norm(source, source.shape)
    with torch.no_grad():
        result = model.extract_features(source, padding_mask=None)
    if not isinstance(result, dict) or "x" not in result:
        raise ValueError("EmoFiLM emotion2vec extract_features must return an 'x' tensor")

    features = result["x"]
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError(f"unexpected batch shape from emotion2vec: {tuple(features.shape)}")
        features = features.squeeze(0)
    return _validate_features(features)


def save_frame_artifact(path: Path, features: torch.Tensor) -> None:
    features = _validate_features(features)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feats": features,
            "frame_rate_hz": EMOFILM_FRAME_RATE_HZ,
            "frame_step_ms": EMOFILM_FRAME_STEP_MS,
        },
        path,
    )


def load_emotion2vec(
    upstream_dir: Path,
    checkpoint: Path,
    device: torch.device,
):
    """通过作者随包 fairseq upstream 严格加载外部官方 checkpoint。"""
    if checkpoint.stat().st_size == 0:
        raise ValueError(f"emotion2vec checkpoint is empty: {checkpoint}")
    if not upstream_dir.is_dir():
        raise FileNotFoundError(f"fairseq upstream directory not found: {upstream_dir}")

    sys.path.insert(0, str(upstream_dir))
    try:
        import fairseq
    except ImportError as exc:
        raise RuntimeError(
            "fairseq is required for EmoFiLM emotion2vec-base extraction; "
            "install/use the validated upstream environment"
        ) from exc

    fairseq.utils.import_user_module(UserDirModule(str(upstream_dir)))
    models, _cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint)]
    )
    if len(models) != 1:
        raise ValueError(f"expected one emotion2vec model, got {len(models)}")
    model = models[0].eval().to(device)
    return model, task


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = build_parser().parse_args()
    provenance = validate_provenance(
        args.provenance,
        checkpoint=args.checkpoint,
        upstream_dir=args.upstream_dir,
    )
    if args.provenance.resolve() == args.checkpoint.resolve():
        raise ValueError("provenance must be a JSON file, not the checkpoint")
    model, task = load_emotion2vec(
        args.upstream_dir,
        args.checkpoint,
        torch.device(args.device),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    samples = _load_manifest(args.manifest)
    for sample in samples:
        utt_id = sample["utt_id"]
        output_path = args.output_dir / f"{utt_id}.pt"
        if output_path.is_file():
            print(f"skip {utt_id}")
            continue
        features = extract_frame_features(
            model,
            task,
            resolve_workspace_wav(sample["wav_path"], args.workspace_root),
            sample_rate=args.sample_rate,
            device=torch.device(args.device),
        )
        save_frame_artifact(output_path, features)

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
