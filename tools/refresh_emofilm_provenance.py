#!/usr/bin/env python3
"""刷新 emofilm_v1 的 provenance 摘要，不重建数据产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_set_sha256(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in ids):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _list_entries(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(Path(entry).is_absolute() for entry in entries):
        raise ValueError(f"data.list must contain relative entries: {path}")
    return entries


def _parquet_summary(data_list: Path) -> dict:
    entries = _list_entries(data_list)
    shards = [data_list.parent / entry for entry in entries]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing parquet shards for {data_list}: {missing[:5]}")

    row_count = 0
    schema_columns: list[str] | None = None
    schema_mismatch = []
    for shard in shards:
        schema = pq.read_schema(shard)
        columns = list(schema.names)
        if schema_columns is None:
            schema_columns = columns
        elif columns != schema_columns:
            schema_mismatch.append(str(shard))
        row_count += pq.read_table(shard, columns=[schema_columns[0] if schema_columns else columns[0]]).num_rows
    if schema_mismatch:
        raise ValueError(f"parquet schema mismatch in {data_list}: {schema_mismatch[:5]}")

    return {
        "data_list_sha256": sha256_file(data_list),
        "data_list_entries": entries,
        "shard_count": len(shards),
        "shard_bytes": sum(path.stat().st_size for path in shards),
        "row_count": row_count,
        "schema_columns": schema_columns or [],
    }


def refresh_split_summary(
    contract_dir: str | Path,
    *,
    train_ids: Sequence[str],
    cv_ids: Sequence[str],
) -> dict:
    """读取当前 train/cv 列表和 parquet，生成不改变产物的摘要。"""
    contract_dir = Path(contract_dir).resolve()
    train_list = contract_dir / "splits/train/parquet/data.list"
    cv_list = contract_dir / "splits/cv/parquet/data.list"
    train = _parquet_summary(train_list)
    cv = _parquet_summary(cv_list)

    train_shards = {(train_list.parent / entry).resolve() for entry in train["data_list_entries"]}
    cv_shards = {(cv_list.parent / entry).resolve() for entry in cv["data_list_entries"]}
    shared = sorted(str(path) for path in train_shards & cv_shards)
    if shared:
        raise ValueError(f"train/cv share parquet shards: {shared}")

    return {
        "contract_name": "emofilm_v1",
        "train_count": len(train_ids),
        "cv_count": len(cv_ids),
        "union_count": len(set(train_ids) | set(cv_ids)),
        "train_id_sha256": id_set_sha256(train_ids),
        "cv_id_sha256": id_set_sha256(cv_ids),
        "union_id_sha256": id_set_sha256(set(train_ids) | set(cv_ids)),
        "train": train,
        "cv": cv,
        "parquet": {
            "train_rows": train["row_count"],
            "cv_rows": cv["row_count"],
            "shared_shards": shared,
        },
    }


def _quick_directory_record(contract_dir: Path, relative_path: str, directory: Path) -> dict:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    relative_files = [path.relative_to(directory).as_posix() for path in files]
    suffix_counts: dict[str, int] = {}
    for path in files:
        suffix = path.suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

    sample_paths = []
    for path in files[:2] + files[-2:]:
        if path not in sample_paths:
            sample_paths.append(path)
    samples = [
        {
            "path": path.relative_to(contract_dir).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sample_paths
    ]
    return {
        "path": relative_path,
        "kind": "directory_quick",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "id_set_sha256": id_set_sha256(relative_files),
        "suffix_counts": suffix_counts,
        "sample_hashes": samples,
    }


def build_artifact_records(
    contract_dir: str | Path,
    *,
    core_paths: Sequence[Path],
    quick_dirs: Mapping[str, Path],
) -> list[dict]:
    """生成核心文件全 SHA-256 和大型目录分层快速摘要。"""
    contract_dir = Path(contract_dir).resolve()
    records = []
    for path in sorted((Path(path).resolve() for path in core_paths), key=lambda item: str(item)):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": path.relative_to(contract_dir).as_posix(),
                "kind": "file_sha256",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for relative_path, directory in sorted(quick_dirs.items()):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        records.append(_quick_directory_record(contract_dir, relative_path, directory))
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refresh_contract_counts(
    contract_path: Path,
    *,
    train_ids: Sequence[str],
    cv_ids: Sequence[str],
    rejected_ids: Sequence[str],
) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "active_train_count": len(train_ids),
            "active_cv_count": len(cv_ids),
            "active_union_count": len(set(train_ids) | set(cv_ids)),
            "rejected_count": len(rejected_ids),
        }
    )
    _write_json(contract_path, contract)
    return contract


def refresh_contract_provenance(contract_dir: str | Path) -> dict:
    contract_dir = Path(contract_dir).resolve()
    provenance_dir = contract_dir / "provenance"
    membership = json.loads((provenance_dir / "membership.json").read_text(encoding="utf-8"))
    train_ids = [str(value) for value in membership["train"]]
    cv_ids = [str(value) for value in membership["cv"]]
    rejected_ids = [str(value) for value in membership["rejected"]]
    refresh_contract_counts(
        provenance_dir / "contract.json",
        train_ids=train_ids,
        cv_ids=cv_ids,
        rejected_ids=rejected_ids,
    )
    split_summary = refresh_split_summary(contract_dir, train_ids=train_ids, cv_ids=cv_ids)

    core_relative_paths = [
        "provenance/contract.json",
        "provenance/sources.json",
        "provenance/membership.json",
        "provenance/speech_token_repairs.json",
        "sources/iemocap/manifest.jsonl",
        "sources/iemocap/frozen_manifest.jsonl",
        "sources/iemocap/rejected.jsonl",
        "sources/iemocap/tagged.jsonl",
        "provenance/iemocap_word_boundaries.jsonl",
        "sources/esd/manifest.jsonl",
        "sources/esd/tagged.jsonl",
        "sources/fedd/manifest/part_a.jsonl",
        "sources/fedd/manifest/part_b.jsonl",
        "eval/esd/manifest.jsonl",
        "eval/fedd_a/manifest.jsonl",
        "eval/fedd_b/manifest.jsonl",
        "splits/train/manifest.jsonl",
        "splits/cv/manifest.jsonl",
        "splits/train/parquet/data.list",
        "splits/cv/parquet/data.list",
        "splits/train/parquet/utt2data.list",
        "splits/cv/parquet/utt2data.list",
        "splits/train/parquet/spk2data.list",
        "splits/cv/parquet/spk2data.list",
    ]
    core_paths = [contract_dir / relative_path for relative_path in core_relative_paths]
    quick_dirs = {
        "sources/iemocap/emotion2vec_base": contract_dir / "sources/iemocap/emotion2vec_base",
        "sources/iemocap/word_blocks": contract_dir / "sources/iemocap/word_blocks",
        "sources/fedd/target_wav": contract_dir / "sources/fedd/target_wav",
        "sources/fedd/prompts": contract_dir / "sources/fedd/prompts",
        "splits/train/src": contract_dir / "splits/train/src",
        "splits/cv/src": contract_dir / "splits/cv/src",
        "splits/train/parquet": contract_dir / "splits/train/parquet",
        "splits/cv/parquet": contract_dir / "splits/cv/parquet",
    }
    records = build_artifact_records(contract_dir, core_paths=core_paths, quick_dirs=quick_dirs)
    _write_json(provenance_dir / "split_build.json", split_summary)
    with (provenance_dir / "artifacts.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"split_build": split_summary, "artifact_count": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description="refresh emofilm_v1 provenance without rebuilding data")
    parser.add_argument("--contract-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refresh_contract_provenance(args.contract_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
