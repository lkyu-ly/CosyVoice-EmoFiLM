#!/usr/bin/env python3
"""将一个 CosyVoice src_dir 直接打包为独立 parquet shards。"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"invalid {path}:{line_number}")
            key, value = parts
            if key in values:
                raise ValueError(f"duplicate key {key!r} in {path}")
            values[key] = value
    return values


def _load_optional_map(src_dir: Path, name: str) -> dict[str, Any] | None:
    path = src_dir / name
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _write_parquet_chunk(
    utt_list: list[str],
    wav_map: dict[str, str],
    text_map: dict[str, str],
    spk_map: dict[str, str],
    optional_maps: dict[str, dict[str, Any] | None],
    parquet_file: str,
    utt2parquet_file: str,
    spk2parquet_file: str,
    source_root: str | None,
) -> dict[str, Any]:
    audio_data = []
    for utt in utt_list:
        wav_path = Path(wav_map[utt])
        if not wav_path.is_absolute() and source_root is not None:
            wav_path = Path(source_root) / wav_path
        with wav_path.open("rb") as handle:
            audio_data.append(handle.read())

    columns: dict[str, Any] = {
        "utt": utt_list,
        "audio_data": audio_data,
        "wav": [wav_map[utt] for utt in utt_list],
        "text": [text_map[utt] for utt in utt_list],
        "spk": [spk_map[utt] for utt in utt_list],
    }
    for column, mapping in optional_maps.items():
        if mapping is not None:
            keys = [spk_map[utt] for utt in utt_list] if column == "spk_embedding" else utt_list
            columns[column] = [mapping[key] for key in keys]

    pd.DataFrame(columns).to_parquet(parquet_file)
    with open(utt2parquet_file, "w", encoding="utf-8") as handle:
        json.dump({utt: Path(parquet_file).name for utt in utt_list}, handle, ensure_ascii=False, indent=2)
    speakers = sorted({spk_map[utt] for utt in utt_list})
    with open(spk2parquet_file, "w", encoding="utf-8") as handle:
        json.dump({speaker: Path(parquet_file).name for speaker in speakers}, handle, ensure_ascii=False, indent=2)
    return {"rows": len(utt_list), "parquet": parquet_file}


def wait_for_async_jobs(jobs):
    """等待所有 parquet worker，并向调用方传播 worker 异常。"""
    for result in jobs:
        result.get()


def _write_data_lists(
    output_dir: Path,
    shard_names: list[str],
    utt_map_names: list[str],
    spk_map_names: list[str],
) -> None:
    for filename, names in (
        ("data.list", shard_names),
        ("utt2data.list", utt_map_names),
        ("spk2data.list", spk_map_names),
    ):
        (output_dir / filename).write_text(
            "".join(f"{name}\n" for name in names), encoding="utf-8"
        )


def pack_src_dir(
    src_dir: str | Path,
    output_dir: str | Path,
    *,
    num_utts_per_parquet: int = 1000,
    num_processes: int = 1,
    source_root: str | Path | None = None,
    shard_prefix: str = "",
) -> dict[str, Any]:
    """直接打包 src_dir；任何 shard 失败都不会写最终 data.list。"""
    if num_utts_per_parquet <= 0:
        raise ValueError("num_utts_per_parquet must be positive")
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")

    src_dir = Path(src_dir)
    output_dir = Path(output_dir)
    if (output_dir / "data.list").exists():
        raise FileExistsError(f"refusing to overwrite existing data.list: {output_dir / 'data.list'}")

    wav_map = _read_key_value_file(src_dir / "wav.scp")
    text_map = _read_key_value_file(src_dir / "text")
    spk_map = _read_key_value_file(src_dir / "utt2spk")
    utts = list(wav_map)
    if not utts:
        raise ValueError(f"src_dir has no utterances: {src_dir}")
    if set(utts) != set(text_map) or set(utts) != set(spk_map):
        raise ValueError("wav.scp, text and utt2spk must have identical utterance IDs")

    optional_maps = {
        "utt_embedding": _load_optional_map(src_dir, "utt2embedding.pt"),
        "spk_embedding": _load_optional_map(src_dir, "spk2embedding.pt"),
        "speech_token": _load_optional_map(src_dir, "utt2speech_token.pt"),
        "instruct": _read_key_value_file(src_dir / "instruct") if (src_dir / "instruct").is_file() else None,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent)))
    chunks = [utts[index:index + num_utts_per_parquet] for index in range(0, len(utts), num_utts_per_parquet)]
    shard_names = [f"{shard_prefix}parquet_{index:09d}.tar" for index in range(len(chunks))]
    utt_map_names = [
        f"{shard_prefix}utt2parquet_{index:09d}.json"
        for index in range(len(chunks))
    ]
    spk_map_names = [
        f"{shard_prefix}spk2parquet_{index:09d}.json"
        for index in range(len(chunks))
    ]

    jobs = []
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        for index, chunk in enumerate(chunks):
            jobs.append(
                pool.apply_async(
                    _write_parquet_chunk,
                    (
                        chunk,
                        wav_map,
                        text_map,
                        spk_map,
                        optional_maps,
                        str(staging / shard_names[index]),
                        str(staging / utt_map_names[index]),
                        str(staging / spk_map_names[index]),
                        str(source_root) if source_root is not None else None,
                    ),
                )
            )
        pool.close()
        wait_for_async_jobs(jobs)
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        for path in staging.iterdir():
            os.replace(path, output_dir / path.name)
        staging.rmdir()
        _write_data_lists(
            output_dir,
            shard_names,
            utt_map_names,
            spk_map_names,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "rows": len(utts),
        "shards": shard_names,
        "data_list": str(output_dir / "data.list"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_utts_per_parquet", type=int, default=1000)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--src_dir", type=Path, required=True)
    parser.add_argument("--des_dir", type=Path, required=True)
    args = parser.parse_args()
    report = pack_src_dir(
        args.src_dir,
        args.des_dir,
        num_utts_per_parquet=args.num_utts_per_parquet,
        num_processes=args.num_processes,
    )
    print(f"Wrote {report['rows']} rows in {len(report['shards'])} shards")


if __name__ == "__main__":
    main()
