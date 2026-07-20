#!/usr/bin/env python3
"""准备 emofilm_v1 来源级合同和固定评测清单。

该步骤只规范化/复用已有来源资产，不生成 IEMOCAP 情感特征或训练 parquet。
IEMOCAP 的 frames、word blocks、词级标签和 train/cv src/parquet 在后续步骤生成。
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from tools.build_emofilm_contract import (
    CONTRACT_NAME,
    _sha256_file,
    normalize_manifest_row,
    normalize_workspace_path,
    write_contract_provenance,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_input_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def frozen_split_ids(train_parquet: Path, cv_parquet: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split, parquet_path in (("train", train_parquet), ("cv", cv_parquet)):
        if not parquet_path.is_file():
            raise FileNotFoundError(parquet_path)
        ids = pq.read_table(parquet_path, columns=["utt"])["utt"].to_pylist()
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate frozen IDs in {parquet_path}")
        result[split] = [str(utt_id) for utt_id in ids]
    if set(result["train"]) & set(result["cv"]):
        raise ValueError("frozen train/cv IDs overlap")
    return result


def frozen_source_rows(
    rows: Iterable[dict[str, Any]],
    *,
    frozen_train_ids: set[str],
    frozen_cv_ids: set[str],
) -> list[dict[str, Any]]:
    """选择冻结来源成员并显式记录其原始 split。"""
    result = []
    for row in rows:
        utt_id = str(row["utt_id"])
        if utt_id in frozen_train_ids:
            split = "train"
        elif utt_id in frozen_cv_ids:
            split = "cv"
        else:
            continue
        prepared = dict(row)
        prepared["original_split"] = split
        result.append(prepared)
    expected = frozen_train_ids | frozen_cv_ids
    actual = {str(row["utt_id"]) for row in result}
    if actual != expected:
        raise ValueError(
            f"frozen source coverage mismatch: missing={sorted(expected - actual)[:5]} "
            f"extra={sorted(actual - expected)[:5]}"
        )
    return result


def _normalize_row(
    row: dict[str, Any],
    *,
    dataset: str,
    repo_root: Path,
    workspace_root: Path,
    label_source: str,
) -> dict[str, Any]:
    prepared = dict(row)
    for key in ("wav_path", "audio_filepath", "prompt_wav", "reference_wav", "target_wav"):
        if prepared.get(key):
            prepared[key] = str(resolve_input_path(prepared[key], repo_root))
    return normalize_manifest_row(
        prepared,
        dataset=dataset,
        workspace_root=workspace_root,
        label_source=label_source,
    )


def _eval_row(row: dict[str, Any], *, source_dataset: str) -> dict[str, Any]:
    result = dict(row)
    result["target_wav"] = result["wav_path"]
    result["reference_wav"] = result["wav_path"]
    result["label"] = result.get("sentence_emotion") or result.get("emo_to", "")
    return result


def _copy_fedd_target(src: Path, destination: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size != src.stat().st_size:
            raise ValueError(f"existing FEDD target differs in size: {destination}")
        return
    shutil.copy2(src, destination)


def prepare_contract(
    repo_root: Path,
    workspace_root: Path,
    contract_dir: Path,
    *,
    iemocap_manifest: Path,
    esd_manifest: Path,
    esd_tagged_manifest: Path,
    esd_eval_manifest: Path,
    fedd_manifest: Path,
    frozen_train_parquet: Path,
    frozen_cv_parquet: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    workspace_root = workspace_root.resolve()
    contract_dir = contract_dir.resolve()
    contract_dir.mkdir(parents=True, exist_ok=True)

    frozen = frozen_split_ids(
        resolve_input_path(frozen_train_parquet, repo_root),
        resolve_input_path(frozen_cv_parquet, repo_root),
    )
    frozen_union = set(frozen["train"]) | set(frozen["cv"])

    raw_iemocap = read_jsonl(resolve_input_path(iemocap_manifest, repo_root))
    raw_esd = read_jsonl(resolve_input_path(esd_manifest, repo_root))
    tagged_esd = read_jsonl(resolve_input_path(esd_tagged_manifest, repo_root))
    tagged_esd_test = read_jsonl(resolve_input_path(esd_eval_manifest, repo_root))

    iemocap_rows = [
        _normalize_row(
            row,
            dataset="iemocap",
            repo_root=repo_root,
            workspace_root=workspace_root,
            label_source="iemocap_sentence_label_reference",
        )
        for row in raw_iemocap
    ]
    esd_rows = [
        _normalize_row(
            row,
            dataset="esd",
            repo_root=repo_root,
            workspace_root=workspace_root,
            label_source="dataset_global_label",
        )
        for row in raw_esd
    ]
    esd_tagged_rows = [
        _normalize_row(
            row,
            dataset="esd",
            repo_root=repo_root,
            workspace_root=workspace_root,
            label_source="dataset_global_label",
        )
        for row in tagged_esd
    ]

    iemocap_dir = contract_dir / "sources/iemocap"
    esd_dir = contract_dir / "sources/esd"
    write_jsonl(iemocap_dir / "manifest.jsonl", iemocap_rows)
    write_jsonl(
        iemocap_dir / "frozen_manifest.jsonl",
        frozen_source_rows(
            iemocap_rows,
            frozen_train_ids=set(frozen["train"]),
            frozen_cv_ids=set(frozen["cv"]),
        ),
    )
    write_jsonl(iemocap_dir / "rejected.jsonl", [])
    write_jsonl(esd_dir / "manifest.jsonl", esd_rows)
    write_jsonl(esd_dir / "tagged.jsonl", esd_tagged_rows)

    esd_eval_rows = []
    for row in tagged_esd_test:
        normalized = _normalize_row(
            row,
            dataset="esd",
            repo_root=repo_root,
            workspace_root=workspace_root,
            label_source="dataset_global_label",
        )
        esd_eval_rows.append(_eval_row(normalized, source_dataset="esd"))
    write_jsonl(contract_dir / "eval/esd/manifest.jsonl", esd_eval_rows)

    fedd_source_rows = read_jsonl(resolve_input_path(fedd_manifest, repo_root))
    fedd_dir = contract_dir / "sources/fedd"
    fedd_by_part: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    fedd_eval_by_part: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for row in fedd_source_rows:
        part = str(row.get("part", "")).upper()
        if part not in fedd_by_part:
            raise ValueError(f"unexpected FEDD part: {part!r}")
        source_wav = resolve_input_path(row["wav_path"], repo_root)
        destination = fedd_dir / "target_wav" / f"part_{part.lower()}" / source_wav.name
        _copy_fedd_target(source_wav, destination)

        prepared = dict(row)
        prepared["wav_path"] = str(destination)
        if prepared.get("prompt_wav"):
            prompt = resolve_input_path(prepared["prompt_wav"], repo_root)
            if prompt.is_relative_to(repo_root / "data/fedd_rebuilt/prompts"):
                prompt_destination = fedd_dir / "prompts" / prompt.name
                prompt_destination.parent.mkdir(parents=True, exist_ok=True)
                if not prompt_destination.is_file():
                    shutil.copy2(prompt, prompt_destination)
                prepared["prompt_wav"] = str(prompt_destination)
            else:
                prepared["prompt_wav"] = str(prompt)
        normalized = _normalize_row(
            prepared,
            dataset="fedd_rebuilt",
            repo_root=repo_root,
            workspace_root=workspace_root,
            label_source="construction_known_transition",
        )
        fedd_by_part[part].append(normalized)
        fedd_eval_by_part[part].append(_eval_row(normalized, source_dataset="fedd_rebuilt"))

    for part in ("A", "B"):
        write_jsonl(fedd_dir / "manifest" / f"part_{part.lower()}.jsonl", fedd_by_part[part])
        write_jsonl(contract_dir / "eval" / f"fedd_{part.lower()}" / "manifest.jsonl", fedd_eval_by_part[part])

    anchor_manifest = repo_root / "data/fedd_rebuilt/prompts/anchor_manifest.jsonl"
    if anchor_manifest.is_file():
        anchors = []
        for row in read_jsonl(anchor_manifest):
            prepared = dict(row)
            prepared["prompt_wav"] = str(resolve_input_path(row["prompt_wav"], repo_root))
            if Path(prepared["prompt_wav"]).is_relative_to(repo_root / "data/fedd_rebuilt/prompts"):
                destination = fedd_dir / "prompts" / Path(prepared["prompt_wav"]).name
                if not destination.is_file():
                    shutil.copy2(prepared["prompt_wav"], destination)
                prepared["prompt_wav"] = str(destination)
            prepared["prompt_wav"] = normalize_workspace_path(prepared["prompt_wav"], workspace_root)
            anchors.append(prepared)
        write_jsonl(fedd_dir / "prompts/anchor_manifest.jsonl", anchors)

    membership = {
        "frozen_train": frozen["train"],
        "frozen_cv": frozen["cv"],
        "frozen_union_count": len(frozen_union),
        "train_cv_disjoint": True,
    }
    sources = [
        {"dataset": "iemocap", "manifest": "sources/iemocap/manifest.jsonl", "count": len(iemocap_rows)},
        {"dataset": "esd", "manifest": "sources/esd/manifest.jsonl", "count": len(esd_rows)},
        {"dataset": "fedd_rebuilt", "manifest": "sources/fedd/manifest/part_a.jsonl", "count": len(fedd_by_part["A"])},
        {"dataset": "fedd_rebuilt", "manifest": "sources/fedd/manifest/part_b.jsonl", "count": len(fedd_by_part["B"])},
    ]
    artifacts = []
    for path in (
        iemocap_dir / "manifest.jsonl",
        esd_dir / "manifest.jsonl",
        esd_dir / "tagged.jsonl",
        contract_dir / "eval/esd/manifest.jsonl",
        fedd_dir / "manifest/part_a.jsonl",
        fedd_dir / "manifest/part_b.jsonl",
        contract_dir / "eval/fedd_a/manifest.jsonl",
        contract_dir / "eval/fedd_b/manifest.jsonl",
    ):
        artifacts.append(
            {
                "path": path.relative_to(contract_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    contract = {
        "contract_name": CONTRACT_NAME,
        "schema_version": 1,
        "workspace_root": str(workspace_root),
        "project_root": str(repo_root),
        "frozen_train_count": len(frozen["train"]),
        "frozen_cv_count": len(frozen["cv"]),
        "frozen_union_count": len(frozen_union),
        "eval_counts": {"esd": len(esd_eval_rows), "fedd_a": len(fedd_eval_by_part["A"]), "fedd_b": len(fedd_eval_by_part["B"])},
        "path_base": "workspace_root",
    }
    write_contract_provenance(
        contract_dir,
        contract=contract,
        sources=sources,
        membership=membership,
        artifacts=artifacts,
    )
    return {
        "contract": contract,
        "sources": sources,
        "eval_counts": contract["eval_counts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="prepare emofilm_v1 source/eval contract")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--iemocap-manifest", type=Path, required=True)
    parser.add_argument("--esd-manifest", type=Path, required=True)
    parser.add_argument("--esd-tagged-manifest", type=Path, required=True)
    parser.add_argument("--esd-eval-manifest", type=Path, required=True)
    parser.add_argument("--fedd-manifest", type=Path, required=True)
    parser.add_argument("--frozen-train-parquet", type=Path, required=True)
    parser.add_argument("--frozen-cv-parquet", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = prepare_contract(
        args.project_root,
        args.workspace_root,
        args.contract_dir,
        iemocap_manifest=args.iemocap_manifest,
        esd_manifest=args.esd_manifest,
        esd_tagged_manifest=args.esd_tagged_manifest,
        esd_eval_manifest=args.esd_eval_manifest,
        fedd_manifest=args.fedd_manifest,
        frozen_train_parquet=args.frozen_train_parquet,
        frozen_cv_parquet=args.frozen_cv_parquet,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
