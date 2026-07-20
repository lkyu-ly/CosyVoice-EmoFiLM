#!/usr/bin/env python3
"""从现有 MFA TextGrid 构建 EmoFiLM 768d/50Hz 词级 frame blocks。"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.build_emofilm_contract import classify_text_coverage


EMOFILM_FEATURE_DIM = 768
EMOFILM_FRAME_RATE_HZ = 50.0
EMOFILM_FRAME_STEP_MS = 20.0


def parse_word_intervals(textgrid_path: Path) -> list[tuple[str, float, float]]:
    """解析 long TextGrid 中名称含 word 的 tier。"""
    lines = textgrid_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tier_is_word = False
    intervals: list[tuple[str, float, float]] = []
    current_xmin = None
    current_xmax = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("name = "):
            tier_name = line.split("=", 1)[1].strip().strip('"').lower()
            tier_is_word = "word" in tier_name
            current_xmin = None
            current_xmax = None
            continue
        if not tier_is_word:
            continue
        if line.startswith("xmin = "):
            try:
                current_xmin = float(line.split("=", 1)[1].strip())
            except ValueError:
                current_xmin = None
        elif line.startswith("xmax = "):
            try:
                current_xmax = float(line.split("=", 1)[1].strip())
            except ValueError:
                current_xmax = None
        elif line.startswith("text = "):
            word = line.split("=", 1)[1].strip().strip('"')
            if current_xmin is None or current_xmax is None or not word:
                current_xmin = None
                current_xmax = None
                continue
            clean = re.sub(r"\(\d+\)", "", word).strip()
            if clean and not clean.startswith("<"):
                intervals.append((clean, current_xmin, current_xmax))
            current_xmin = None
            current_xmax = None
    return intervals


def _load_frame_artifact(path: Path) -> tuple[torch.Tensor, float, float]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError(f"invalid frame artifact: {path}")
    features = artifact.get("feats")
    if not torch.is_tensor(features) or features.ndim != 2 or features.shape[1] != EMOFILM_FEATURE_DIM:
        raise ValueError(
            f"frame artifact must have shape (T, {EMOFILM_FEATURE_DIM}): {path}"
        )
    frame_rate_hz = float(artifact.get("frame_rate_hz", 0.0))
    frame_step_ms = float(artifact.get("frame_step_ms", 0.0))
    if frame_rate_hz != EMOFILM_FRAME_RATE_HZ or frame_step_ms != EMOFILM_FRAME_STEP_MS:
        raise ValueError(f"frame artifact must be 50Hz/20ms: {path}")
    return features.float().contiguous(), frame_rate_hz, frame_step_ms


def _rejected_row(
    sample: dict[str, Any],
    reason: str,
    split: str,
    reason_details: dict[str, object] | None = None,
) -> dict[str, Any]:
    row = dict(sample)
    row.update(
        {
            "utt_id": sample.get("utt_id", ""),
            "reason": reason,
            "speaker_id": sample.get("speaker_id", ""),
            "sentence_emotion": sample.get("sentence_emotion", sample.get("emotion", "")),
            "original_split": sample.get("original_split", split),
        }
    )
    if reason_details is not None:
        row["reason_details"] = reason_details
    return row


def _handle_failure(
    sample: dict[str, Any],
    reason: str,
    dataset: str,
    split: str,
    rejected: list[dict[str, Any]],
    reason_details: dict[str, object] | None = None,
) -> None:
    effective_split = str(sample.get("original_split") or split).lower()
    if dataset.lower() == "iemocap" and effective_split in {"train", "cv"}:
        rejected.append(_rejected_row(sample, reason, effective_split, reason_details))
        return
    raise RuntimeError(
        f"hard-fail: {dataset}/{effective_split} sample {sample.get('utt_id', '')} "
        f"cannot build word blocks ({reason})"
    )


def _write_rejected(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_word_blocks(
    manifest_path: Path,
    features_dir: Path,
    textgrid_dir: Path,
    output_dir: Path,
    *,
    dataset: str,
    split: str,
    rejected_manifest: Path | None = None,
    boundary_manifest: Path | None = None,
) -> dict[str, Any]:
    """完整验证每条 utterance 后原子发布 blocks，并记录量化边界事件。"""
    with manifest_path.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]

    rejected: list[dict[str, Any]] = []
    boundary_events: list[dict[str, Any]] = []
    done = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        utt_id = sample["utt_id"]
        feature_path = features_dir / f"{utt_id}.pt"
        textgrid_path = textgrid_dir / f"{utt_id}.TextGrid"
        if not feature_path.is_file():
            _handle_failure(sample, "missing_frame_artifact", dataset, split, rejected)
            continue
        if not textgrid_path.is_file():
            _handle_failure(sample, "missing_textgrid", dataset, split, rejected)
            continue

        try:
            features, frame_rate_hz, frame_step_ms = _load_frame_artifact(feature_path)
        except (OSError, ValueError) as exc:
            _handle_failure(sample, f"invalid_frame_artifact:{exc}", dataset, split, rejected)
            continue

        intervals = parse_word_intervals(textgrid_path)
        if not intervals:
            _handle_failure(sample, "empty_word_tier", dataset, split, rejected)
            continue

        coverage = classify_text_coverage(
            str(sample.get("plain_text") or sample.get("text", "")),
            " ".join(word for word, _, _ in intervals),
        )
        if coverage["decision"] == "reject":
            coverage_sec = features.shape[0] / EMOFILM_FRAME_RATE_HZ
            for index, (word, start_sec, end_sec) in enumerate(intervals):
                if end_sec <= coverage_sec:
                    continue
                start_frame = max(
                    0,
                    min(math.floor(start_sec * EMOFILM_FRAME_RATE_HZ), features.shape[0]),
                )
                end_frame = max(
                    start_frame,
                    min(math.ceil(end_sec * EMOFILM_FRAME_RATE_HZ), features.shape[0]),
                )
                boundary_events.append(
                    {
                        "utt_id": utt_id,
                        "word": word,
                        "word_index": index,
                        "event": "tail_clipped" if end_frame > start_frame else "tail_empty",
                        "interval_start_sec": float(start_sec),
                        "interval_end_sec": float(end_sec),
                        "frame_coverage_sec": coverage_sec,
                        "overrun_ms": (float(end_sec) - coverage_sec) * 1000.0,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "disposition": "rejected",
                        "rejection_reason": str(coverage["category"]),
                    }
                )
            utt_dir = output_dir / utt_id
            if utt_dir.exists():
                shutil.rmtree(utt_dir)
            _handle_failure(
                sample,
                str(coverage["category"]),
                dataset,
                split,
                rejected,
                coverage,
            )
            continue

        total_frames = features.shape[0]
        coverage_sec = total_frames / EMOFILM_FRAME_RATE_HZ
        planned_blocks: list[dict[str, Any]] = []
        sample_events: list[dict[str, Any]] = []
        failure_reason = None
        for index, (word, start_sec, end_sec) in enumerate(intervals):
            start_frame = max(0, min(math.floor(start_sec * EMOFILM_FRAME_RATE_HZ), total_frames))
            end_frame = max(start_frame, min(math.ceil(end_sec * EMOFILM_FRAME_RATE_HZ), total_frames))
            if end_frame <= start_frame:
                failure_reason = f"empty_word_interval:{word}"
                if end_sec > coverage_sec:
                    sample_events.append(
                        {
                            "utt_id": utt_id,
                            "word": word,
                            "word_index": index,
                            "event": "tail_empty",
                            "interval_start_sec": float(start_sec),
                            "interval_end_sec": float(end_sec),
                            "frame_coverage_sec": coverage_sec,
                            "overrun_ms": (float(end_sec) - coverage_sec) * 1000.0,
                            "start_frame": int(start_frame),
                            "end_frame": int(end_frame),
                        }
                    )
                break
            planned_blocks.append(
                {
                    "frames": features[start_frame:end_frame].clone(),
                    "word": word,
                    "padding_mask": torch.zeros(end_frame - start_frame, dtype=torch.bool),
                    "frame_rate_hz": frame_rate_hz,
                    "frame_step_ms": frame_step_ms,
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "filename": f"{index:04d}_{start_frame}_{end_frame}.pt",
                },
            )
            if end_sec > coverage_sec:
                sample_events.append(
                    {
                        "utt_id": utt_id,
                        "word": word,
                        "word_index": index,
                        "event": "tail_clipped",
                        "interval_start_sec": float(start_sec),
                        "interval_end_sec": float(end_sec),
                        "frame_coverage_sec": coverage_sec,
                        "overrun_ms": (float(end_sec) - coverage_sec) * 1000.0,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                    }
                )

        utt_dir = output_dir / utt_id
        if failure_reason is not None or not planned_blocks:
            if utt_dir.exists():
                shutil.rmtree(utt_dir)
            _handle_failure(
                sample,
                failure_reason or "no_valid_word_blocks",
                dataset,
                split,
                rejected,
            )
            boundary_events.extend(
                {
                    **event,
                    "disposition": "rejected",
                    "rejection_reason": failure_reason or "no_valid_word_blocks",
                }
                for event in sample_events
            )
            continue

        staging_dir = output_dir / f".{utt_id}.staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        for block in planned_blocks:
            filename = block.pop("filename")
            torch.save(block, staging_dir / filename)
        if utt_dir.exists():
            shutil.rmtree(utt_dir)
        staging_dir.rename(utt_dir)
        boundary_events.extend({**event, "disposition": "kept"} for event in sample_events)
        done += 1

    _write_rejected(rejected_manifest, rejected)
    _write_jsonl(boundary_manifest, boundary_events)
    return {
        "done": done,
        "rejected": rejected,
        "boundary_events": len(boundary_events),
        "total": len(samples),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build EmoFiLM word frame blocks")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--textgrid_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--rejected_manifest", type=Path)
    parser.add_argument("--boundary_manifest", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_word_blocks(
        args.manifest,
        args.features_dir,
        args.textgrid_dir,
        args.output_dir,
        dataset=args.dataset,
        split=args.split,
        rejected_manifest=args.rejected_manifest,
        boundary_manifest=args.boundary_manifest,
    )
    print(f"Done: {report['done']} utterances, {len(report['rejected'])} rejected")


if __name__ == "__main__":
    main()
