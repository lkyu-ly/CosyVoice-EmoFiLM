#!/usr/bin/env python3
"""按冻结成员构建 emofilm_v1 train/cv src 与 parquet。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from tools.build_emofilm_contract import (
    build_train_cv_contract,
    validate_train_cv_parquet,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _id_hash(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for utt_id in sorted(ids):
        digest.update(utt_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_map(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping checkpoint: {path}")
    return dict(value)


def _validate_embedding_map(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = {str(key) for key in mapping}
    if actual != expected:
        raise ValueError(
            f"{name} coverage mismatch: missing={sorted(expected - actual)[:5]} "
            f"extra={sorted(actual - expected)[:5]}"
        )
    for key, value in mapping.items():
        tensor = torch.as_tensor(value)
        if tensor.shape != (192,) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} invalid value for {key}: {tuple(tensor.shape)}")


def _validate_speech_map(mapping: Mapping[str, Any], expected: set[str]) -> None:
    actual = {str(key) for key in mapping}
    if actual != expected:
        raise ValueError(
            f"speech_token coverage mismatch: missing={sorted(expected - actual)[:5]} "
            f"extra={sorted(actual - expected)[:5]}"
        )
    for key, value in mapping.items():
        tokens = list(value)
        if not tokens or not all(isinstance(token, int) and 0 <= token < 6561 for token in tokens):
            raise ValueError(f"speech_token invalid for {key}")


def _merge_rows(
    iemocap_manifest: list[dict[str, Any]],
    iemocap_tagged: list[dict[str, Any]],
    esd_tagged: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    iemocap_meta = {row["utt_id"]: row for row in iemocap_manifest}
    result: dict[str, dict[str, Any]] = {}
    for tagged in iemocap_tagged:
        utt_id = tagged["utt_id"]
        if utt_id in result:
            raise ValueError(f"duplicate source utt_id: {utt_id}")
        row = dict(iemocap_meta.get(utt_id, {}))
        row.update(tagged)
        row["tagged_text"] = row.get("tagged_text") or row.get("text", "")
        row["source_dataset"] = "iemocap"
        result[utt_id] = row
    for tagged in esd_tagged:
        utt_id = tagged["utt_id"]
        if utt_id in result:
            raise ValueError(f"duplicate source utt_id: {utt_id}")
        row = dict(tagged)
        row["tagged_text"] = row.get("tagged_text") or row.get("text", "")
        row["source_dataset"] = "esd"
        result[utt_id] = row
    return result


def build_splits(
    *,
    workspace_root: Path,
    contract_dir: Path,
    reuse_cache_dir: Path,
    num_utts_per_parquet: int = 1000,
    num_processes: int = 4,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    contract_dir = contract_dir.resolve()
    reuse_cache_dir = reuse_cache_dir.resolve()
    provenance = json.loads((contract_dir / "provenance/membership.json").read_text())
    frozen_train = [str(value) for value in provenance["frozen_train"]]
    frozen_cv = [str(value) for value in provenance["frozen_cv"]]
    train_set, cv_set = set(frozen_train), set(frozen_cv)
    if train_set & cv_set:
        raise ValueError("frozen train/cv overlap")

    rejected_rows = read_jsonl(contract_dir / "sources/iemocap/rejected.jsonl")
    rejected_ids = {str(row["utt_id"]) for row in rejected_rows}
    active_train = [utt_id for utt_id in frozen_train if utt_id not in rejected_ids]
    active_cv = [utt_id for utt_id in frozen_cv if utt_id not in rejected_ids]
    active_union = set(active_train) | set(active_cv)

    rows = _merge_rows(
        read_jsonl(contract_dir / "sources/iemocap/manifest.jsonl"),
        read_jsonl(contract_dir / "sources/iemocap/tagged.jsonl"),
        read_jsonl(contract_dir / "sources/esd/tagged.jsonl"),
    )
    if set(rows) & active_union != active_union:
        missing = active_union - set(rows)
        raise ValueError(f"missing frozen tagged rows: {sorted(missing)[:10]}")

    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, ids in (("train", active_train), ("cv", active_cv)):
        selected = []
        for utt_id in ids:
            row = dict(rows[utt_id])
            wav_path = Path(str(row["wav_path"]))
            if wav_path.is_absolute():
                raise ValueError(f"absolute wav path in {split}: {utt_id}")
            if not (workspace_root / wav_path).is_file():
                raise FileNotFoundError(workspace_root / wav_path)
            if not row.get("plain_text") or not row.get("tagged_text"):
                raise ValueError(f"missing text fields in {split}: {utt_id}")
            selected.append(row)
        split_rows[split] = selected

    utt_embedding = _load_map(reuse_cache_dir / "utt2embedding.pt")
    spk_embedding = _load_map(reuse_cache_dir / "spk2embedding.pt")
    speech_token = _load_map(reuse_cache_dir / "utt2speech_token.pt")
    repair_path = contract_dir / "provenance/speech_token_repairs.json"
    if repair_path.is_file():
        repairs = json.loads(repair_path.read_text(encoding="utf-8"))["repairs"]
        for repair in repairs:
            speech_token[str(repair["utt_id"])] = repair["tokens"]

    if not active_union <= set(utt_embedding):
        raise ValueError(f"utt_embedding missing active ids: {sorted(active_union - set(utt_embedding))[:5]}")
    if not active_union <= set(speech_token):
        raise ValueError(f"speech_token missing active ids: {sorted(active_union - set(speech_token))[:5]}")
    utt_embedding = {key: utt_embedding[key] for key in active_union}
    speech_token = {key: speech_token[key] for key in active_union}
    _validate_embedding_map(utt_embedding, active_union, "utt_embedding")
    _validate_speech_map(speech_token, active_union)
    speakers = {str(row["speaker_id"]) for row in rows.values() if row["utt_id"] in active_union}
    _validate_embedding_map(spk_embedding, speakers, "spk_embedding")

    optional_maps = {}
    for split in ("train", "cv"):
        ids = {row["utt_id"] for row in split_rows[split]}
        split_speakers = {str(row["speaker_id"]) for row in split_rows[split]}
        optional_maps[split] = {
            "utt2embedding.pt": {key: utt_embedding[key] for key in ids},
            "utt2speech_token.pt": {key: speech_token[key] for key in ids},
            "spk2embedding.pt": {key: spk_embedding[key] for key in split_speakers},
        }

    report = build_train_cv_contract(
        contract_dir,
        train_rows=split_rows["train"],
        cv_rows=split_rows["cv"],
        frozen_train_ids=train_set,
        frozen_cv_ids=cv_set,
        rejected_rows=rejected_rows,
        original_rows=read_jsonl(contract_dir / "sources/iemocap/manifest.jsonl"),
        optional_maps=optional_maps,
        source_root=workspace_root,
        num_utts_per_parquet=num_utts_per_parquet,
        num_processes=num_processes,
    )
    loaded = validate_train_cv_parquet(
        contract_dir / "splits/train/parquet/data.list",
        contract_dir / "splits/cv/parquet/data.list",
    )
    summary = {
        "train_count": len(split_rows["train"]),
        "cv_count": len(split_rows["cv"]),
        "union_count": len(active_union),
        "rejected_count": len(rejected_ids),
        "train_id_sha256": _id_hash(active_train),
        "cv_id_sha256": _id_hash(active_cv),
        "union_id_sha256": _id_hash(list(active_union)),
        "parquet": loaded,
        "report": report,
    }
    (contract_dir / "provenance/split_build.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="build frozen emofilm_v1 train/cv splits")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--reuse-cache-dir", type=Path, required=True)
    parser.add_argument("--num-utts-per-parquet", type=int, default=1000)
    parser.add_argument("--num-processes", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(build_splits(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
