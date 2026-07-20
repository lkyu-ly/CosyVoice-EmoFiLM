#!/usr/bin/env python3
"""构建和验证基础 EmoFiLM 语义数据合同。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow.parquet as pq
import torch

from cosyvoice_emo.emo_annotator import WordSequenceModel


CONTRACT_NAME = "emofilm_v1"
_EMOTION_TAG_RE = re.compile(r"</?emotion\b[^>]*>", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*", re.IGNORECASE)


def normalize_workspace_path(path: str | Path, workspace_root: str | Path) -> str:
    """将工作区内路径规范化为 POSIX 相对路径，禁止绝对路径泄漏。"""
    workspace = Path(workspace_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {candidate}") from exc
    return relative.as_posix()


def normalize_manifest_row(
    row: Mapping[str, object],
    *,
    dataset: str,
    workspace_root: str | Path,
    label_source: str | None = None,
) -> dict[str, object]:
    """统一来源/训练/eval manifest 行的路径、文本和来源字段。"""
    result = dict(row)
    utt_id = str(result.get("utt_id", "")).strip()
    if not utt_id:
        raise ValueError("manifest row requires utt_id")

    wav_value = result.get("wav_path") or result.get("audio_filepath")
    if not wav_value:
        raise ValueError(f"manifest row {utt_id} requires wav_path")
    result["wav_path"] = normalize_workspace_path(str(wav_value), workspace_root)

    for key in ("prompt_wav", "reference_wav", "target_wav"):
        if result.get(key):
            result[key] = normalize_workspace_path(str(result[key]), workspace_root)

    tagged_text = result.get("tagged_text")
    text_value = str(result.get("text", "") or "")
    plain_text = str(result.get("plain_text", "") or "")
    if not plain_text:
        plain_text = text_value if "<emotion" not in text_value else ""
    if not tagged_text and "<emotion" in text_value:
        tagged_text = text_value
    if not plain_text:
        raise ValueError(f"manifest row {utt_id} requires plain text")

    result["utt_id"] = utt_id
    result["text"] = plain_text
    result["plain_text"] = plain_text
    if tagged_text:
        result["tagged_text"] = str(tagged_text)
    result["speaker_id"] = str(result.get("speaker_id", ""))
    result["source_dataset"] = dataset
    result["label_source"] = label_source or str(result.get("label_source", ""))
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_contract_provenance(
    contract_dir: str | Path,
    *,
    contract: Mapping[str, object],
    sources: Sequence[Mapping[str, object]],
    membership: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """写入新版合同要求的四个 provenance 文件。"""
    contract_dir = Path(contract_dir)
    provenance_dir = contract_dir / "provenance"
    contract_value = {"contract_name": CONTRACT_NAME, **dict(contract)}
    _write_json(provenance_dir / "contract.json", contract_value)
    _write_json(provenance_dir / "sources.json", list(sources))
    _write_json(provenance_dir / "membership.json", dict(membership))
    provenance_dir.mkdir(parents=True, exist_ok=True)
    with (provenance_dir / "artifacts.jsonl").open("w", encoding="utf-8") as handle:
        for artifact in artifacts:
            handle.write(json.dumps(dict(artifact), ensure_ascii=False) + "\n")
    return contract_value


def validate_frame_artifact(frame_path: Path, provenance_path: Path) -> dict:
    """验证 emotion2vec-base 帧特征为 768d/50Hz/20ms 且有来源记录。"""
    artifact = torch.load(frame_path, map_location="cpu", weights_only=True)
    feats = artifact.get("feats")
    if not torch.is_tensor(feats) or feats.ndim != 2 or feats.shape[1] != 768:
        raise ValueError(f"frame artifact must have shape (T, 768): {frame_path}")
    frame_rate_hz = float(artifact.get("frame_rate_hz", 0.0))
    frame_step_ms = float(artifact.get("frame_step_ms", 1000.0 / frame_rate_hz if frame_rate_hz else 0.0))
    if abs(frame_rate_hz - 50.0) > 1e-6 or abs(frame_step_ms - 20.0) > 1e-6:
        raise ValueError(f"frame artifact must be 50Hz/20ms: {frame_path}")

    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    required = {"model_id", "revision", "checkpoint_sha256", "upstream"}
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"missing frame provenance fields: {sorted(missing)}")
    if len(provenance["checkpoint_sha256"]) != 64:
        raise ValueError("checkpoint_sha256 must be a SHA-256 hex string")
    upstream = provenance["upstream"]
    if not isinstance(upstream, dict) or len(str(upstream.get("sha256", ""))) != 64:
        raise ValueError("upstream provenance must contain a SHA-256 hash")
    return {
        "feature_dim": int(feats.shape[1]),
        "frame_rate_hz": frame_rate_hz,
        "frame_step_ms": frame_step_ms,
    }


def load_word_sequence_model(checkpoint: Path, device: str = "cpu") -> WordSequenceModel:
    """strict load EmoFiLM 768d/5 类/3D VAD WordSequenceModel。"""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = WordSequenceModel(input_dim=768, num_classes=5, num_heads=8, dropout_rate=0.3, reg_dim=3)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _tag(emotion: str, intensity: str, words: Sequence[str]) -> str:
    text = " ".join(words)
    return f"<emotion type='{emotion}' intensity='{intensity}'>{text}</emotion>"


def merge_word_predictions(words: Iterable[Mapping[str, str]]) -> str:
    """按相邻词的 emotion 和 intensity 双键合并。"""
    segments = []
    current_key = None
    current_words = []
    for item in words:
        key = (item["predicted_emotion"], item["predicted_intensity"])
        if current_key != key:
            if current_key is not None:
                segments.append(_tag(*current_key, current_words))
            current_key = key
            current_words = []
        current_words.append(item["word"])
    if current_key is not None:
        segments.append(_tag(*current_key, current_words))
    return " ".join(segments)


def text_tokens(text: str) -> list[str]:
    """提取用于音频、TextGrid 与 tagged text 覆盖比较的规范词序。"""
    plain = html.unescape(_EMOTION_TAG_RE.sub(" ", text)).lower()
    return _WORD_RE.findall(plain.replace("’", "'"))


def classify_text_coverage(plain_text: str, compared_text: str) -> dict[str, object]:
    """只放行精确词覆盖或撇号切分等价；其他差异统一拒绝。"""
    plain_tokens = text_tokens(plain_text)
    compared_tokens = text_tokens(compared_text)
    if plain_tokens == compared_tokens:
        return {"decision": "keep", "category": "exact"}
    if "".join(plain_tokens).replace("'", "") == "".join(compared_tokens).replace("'", ""):
        return {"decision": "keep", "category": "apostrophe_tokenization"}

    compared_counts: dict[str, int] = {}
    for token in compared_tokens:
        compared_counts[token] = compared_counts.get(token, 0) + 1
    missing = []
    for token in plain_tokens:
        if compared_counts.get(token, 0):
            compared_counts[token] -= 1
        else:
            missing.append(token)
    return {
        "decision": "reject",
        "category": "audio_text_mismatch",
        "plain_tokens": plain_tokens,
        "tagged_tokens": compared_tokens,
        "missing_from_tagged": missing,
    }


def validate_membership(
    train_ids: set[str],
    cv_ids: set[str],
    rejected_ids: set[str],
    frozen_train_ids: set[str],
    frozen_cv_ids: set[str],
) -> None:
    """验证 train/cv 只移除 rejected，且两边无交集。"""
    frozen_union = frozen_train_ids | frozen_cv_ids
    if not rejected_ids <= frozen_union:
        raise ValueError("rejected ids must come from frozen union membership")
    if train_ids & cv_ids:
        raise ValueError("train/cv membership overlap")
    if train_ids != frozen_train_ids - rejected_ids:
        raise ValueError("train membership differs from frozen ids minus rejected")
    if cv_ids != frozen_cv_ids - rejected_ids:
        raise ValueError("cv membership differs from frozen ids minus rejected")


def validate_rejected_manifest(rejected: Sequence[Mapping[str, str]], original: Sequence[Mapping[str, str]]) -> dict:
    """验证 rejected 逐条有原因、来自 source 且分布可审计。"""
    total = len(original)
    if total == 0:
        raise ValueError("original manifest is empty")
    if any(not row.get("utt_id") or not row.get("reason") for row in rejected):
        raise ValueError("each rejected row needs utt_id and reason")
    fraction = len(rejected) / total
    original_by_id = {row["utt_id"]: row for row in original}
    if len(original_by_id) != total:
        raise ValueError("original manifest contains duplicate utt_id")
    rejected_ids = [row["utt_id"] for row in rejected]
    if len(set(rejected_ids)) != len(rejected_ids):
        raise ValueError("rejected manifest contains duplicate utt_id")
    if not set(row["utt_id"] for row in rejected) <= original_by_id.keys():
        raise ValueError("rejected row is not in original manifest")
    speaker_counts = Counter(row.get("speaker_id") for row in rejected)
    emotion_counts = Counter(row.get("sentence_emotion") for row in rejected)
    concentration_limit = 0.75
    if len(rejected) >= 2:
        for dimension, counts in (("speaker", speaker_counts), ("emotion", emotion_counts)):
            if any(count / len(rejected) > concentration_limit for count in counts.values()):
                raise ValueError(f"rejected samples are concentrated by {dimension}")
    return {
        "rejected_count": len(rejected),
        "fraction": fraction,
        "speaker_counts": dict(speaker_counts),
        "emotion_counts": dict(emotion_counts),
    }


def validate_eval_assets(
    rows: Sequence[Mapping[str, str]],
    expected_count: int,
    workspace_root: str | Path | None = None,
) -> dict:
    """验证固定评测 population 的 target/reference/prompt/文本/标签完整。"""
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} eval rows, got {len(rows)}")
    required = ("utt_id", "target_wav", "reference_wav", "prompt_wav", "text", "label", "prompt_text")
    missing = []
    seen = set()
    for row in rows:
        utt_id = row.get("utt_id")
        if not utt_id or utt_id in seen:
            missing.append(f"duplicate-or-empty:{utt_id}")
        seen.add(utt_id)
        for key in required[1:]:
            value = row.get(key)
            if key.endswith("wav") and value and workspace_root is not None:
                value = Path(workspace_root) / value if not Path(value).is_absolute() else Path(value)
            if not value or (key.endswith("wav") and not Path(value).is_file()):
                missing.append(f"{utt_id}:{key}")
    if missing:
        raise ValueError(f"incomplete eval assets: {missing}")
    return {"count": len(rows), "missing": missing}


def _read_shard_list(data_list: Path) -> list[Path]:
    paths = []
    for raw in data_list.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            shard = Path(raw.strip())
            if shard.is_absolute():
                paths.append(shard)
                continue
            paths.append(data_list.parent / shard)
    return paths


def validate_train_cv_parquet(
    train_list: Path,
    cv_list: Path,
) -> dict:
    """读取 train/cv 全部 shard，并拒绝共享 shard。"""
    train_shards = _read_shard_list(train_list)
    cv_shards = _read_shard_list(cv_list)
    shared = sorted(str(path.resolve()) for path in set(train_shards) & set(cv_shards))
    if shared:
        raise ValueError(f"train/cv share parquet shards: {shared}")
    train_rows = sum(pq.read_table(path).num_rows for path in train_shards)
    cv_rows = sum(pq.read_table(path).num_rows for path in cv_shards)
    return {"train_rows": train_rows, "cv_rows": cv_rows, "shared_shards": shared}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_train_cv_contract(
    contract_dir: Path,
    *,
    train_rows: Sequence[Mapping[str, object]],
    cv_rows: Sequence[Mapping[str, object]],
    frozen_train_ids: set[str],
    frozen_cv_ids: set[str],
    rejected_rows: Sequence[Mapping[str, object]],
    num_utts_per_parquet: int = 1000,
    num_processes: int = 1,
    original_rows: Sequence[Mapping[str, object]] | None = None,
    source_root: str | Path | None = None,
    optional_maps: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> dict[str, dict[str, object]]:
    """构建新版独立 train/cv 合同目录，不复制全量音频视图。"""
    from tools.jsonl_to_cosyvoice_src import write_src_dir
    from tools.make_parquet_list import pack_src_dir

    contract_dir = Path(contract_dir)
    rejected_ids = {str(row["utt_id"]) for row in rejected_rows}
    train_ids = {str(row["utt_id"]) for row in train_rows}
    cv_ids = {str(row["utt_id"]) for row in cv_rows}
    validate_membership(train_ids, cv_ids, rejected_ids, frozen_train_ids, frozen_cv_ids)
    if original_rows is not None:
        validate_rejected_manifest(rejected_rows, original_rows)

    splits_dir = contract_dir / "splits"
    staging_dir = contract_dir / ".splits.staging"
    backup_dir = contract_dir / ".splits.backup"
    for path in (staging_dir, backup_dir):
        if path.exists():
            shutil.rmtree(path)

    reports: dict[str, dict[str, object]] = {}
    try:
        for split, rows in (("train", train_rows), ("cv", cv_rows)):
            split_root = staging_dir / split
            split_root.mkdir(parents=True, exist_ok=True)
            _write_jsonl(split_root / "manifest.jsonl", rows)
            split_src = split_root / "src"
            write_src_dir(str(split_src), [dict(row) for row in rows], use_tagged_text=True)
            for filename, mapping in (optional_maps or {}).get(split, {}).items():
                if not isinstance(mapping, Mapping):
                    raise ValueError(f"optional map must be a mapping: {split}/{filename}")
                if filename == "spk2embedding.pt":
                    expected_keys = {str(row.get("speaker_id", "")) for row in rows}
                else:
                    expected_keys = {str(row["utt_id"]) for row in rows}
                actual_keys = {str(key) for key in mapping}
                if actual_keys != expected_keys:
                    raise ValueError(
                        f"optional map coverage mismatch for {split}/{filename}: "
                        f"missing={sorted(expected_keys - actual_keys)[:5]} "
                        f"extra={sorted(actual_keys - expected_keys)[:5]}"
                    )
                torch.save(dict(mapping), split_src / filename)
            pack_report = pack_src_dir(
                split_src,
                split_root / "parquet",
                num_utts_per_parquet=num_utts_per_parquet,
                num_processes=num_processes,
                source_root=source_root,
                shard_prefix=f"{split}_",
            )
            data_list = split_root / "parquet" / "data.list"
            reports[split] = {
                "rows": len(rows),
                "shards": pack_report["shards"],
                "data_list": str(contract_dir / "splits" / split / "parquet" / "data.list"),
                "data_list_sha256": _sha256_file(data_list),
            }

        if splits_dir.exists():
            splits_dir.rename(backup_dir)
        staging_dir.rename(splits_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if backup_dir.exists():
            if splits_dir.exists():
                shutil.rmtree(splits_dir)
            backup_dir.rename(splits_dir)
        raise

    _write_json(
        contract_dir / "provenance" / "membership.json",
        {
            "train": sorted(train_ids),
            "cv": sorted(cv_ids),
            "rejected": sorted(rejected_ids),
            "frozen_train": sorted(frozen_train_ids),
            "frozen_cv": sorted(frozen_cv_ids),
        },
    )
    return reports


def build_annotation_parser() -> argparse.ArgumentParser:
    """生产 EmoFiLM 标注 CLI；刻意不提供历史 smoothing/majority 开关。"""
    parser = argparse.ArgumentParser(description="build EmoFiLM word-level emotion tags")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--textgrid_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="validate emofilm_v1 data contract")
    parser.add_argument("--contract_dir", type=Path)
    parser.add_argument("--word_sequence_checkpoint", type=Path)
    return parser


if __name__ == "__main__":
    build_parser().parse_args()
