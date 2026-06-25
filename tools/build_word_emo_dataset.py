#!/usr/bin/env python3
"""从 MFA TextGrid + emotion2vec frame feats 构建 per-word frame blocks。

用法: python tools/build_word_emo_dataset.py --manifest data/esd_manifest_train.jsonl \
  --features_dir features/esd_frame/ --textgrid_dir align/esd/ --output_dir word_blocks/esd/
"""
import argparse
import json
import math
import os
import re
from pathlib import Path
import torch


def parse_word_intervals(textgrid_path):
    """解析 long TextGrid 的 words tier。返回 [(word, start_sec, end_sec), ...]."""
    with open(textgrid_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    tier_is_word = False
    intervals = []
    cur_xmin, cur_xmax = None, None

    for raw in lines:
        line = raw.strip()
        if line.startswith("name = "):
            tier_name = line.split("=", 1)[1].strip().strip('"').lower()
            tier_is_word = "word" in tier_name
            continue
        if not tier_is_word:
            continue
        if line.startswith("xmin = "):
            try:
                cur_xmin = float(line.split("=", 1)[1].strip())
            except ValueError:
                cur_xmin = None
        elif line.startswith("xmax = "):
            try:
                cur_xmax = float(line.split("=", 1)[1].strip())
            except ValueError:
                cur_xmax = None
        elif line.startswith("text = "):
            word = line.split("=", 1)[1].strip().strip('"')
            if cur_xmin is not None and cur_xmax is not None and word:
                clean = re.sub(r"\(\d+\)", "", word).strip()
                if clean and not clean.startswith("<"):
                    intervals.append((clean, cur_xmin, cur_xmax))
            cur_xmin, cur_xmax = None, None
    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--features_dir", type=str, required=True)
    parser.add_argument("--textgrid_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rejected_manifest", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    rejected = []
    done = 0
    for s in samples:
        utt_id = s["utt_id"]
        feat_path = os.path.join(args.features_dir, f"{utt_id}.pt")
        tg_path = os.path.join(args.textgrid_dir, f"{utt_id}.TextGrid")
        if not os.path.isfile(feat_path) or not os.path.isfile(tg_path):
            rejected.append(s)
            continue

        word_intervals = parse_word_intervals(tg_path)
        if not word_intervals:
            rejected.append(s)
            continue

        data = torch.load(feat_path, map_location="cpu")
        feats = data["feats"]  # (T, D)
        fps = data.get("frame_rate_hz", 50.0)
        total_frames = feats.shape[0]

        utt_dir = os.path.join(args.output_dir, utt_id)
        os.makedirs(utt_dir, exist_ok=True)

        word_count = 0
        for idx, (word, start_sec, end_sec) in enumerate(word_intervals):
            start_f = int(math.floor(start_sec * fps))
            end_f = int(math.ceil(end_sec * fps))
            start_f = max(0, min(start_f, total_frames))
            end_f = max(start_f, min(end_f, total_frames))
            if end_f <= start_f:
                continue
            block = feats[start_f:end_f].clone()
            out_file = os.path.join(utt_dir, f"{idx:04d}_{start_f}_{end_f}.pt")
            torch.save({
                "frames": block,
                "word": word,
                "padding_mask": torch.zeros(block.shape[0], dtype=torch.bool),
            }, out_file)
            word_count += 1

        if word_count == 0:
            rejected.append(s)
        else:
            done += 1

    print(f"Done: {done} utterances, {len(rejected)} rejected")

    if args.rejected_manifest and rejected:
        with open(args.rejected_manifest, "w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Rejected manifest written to {args.rejected_manifest}")


if __name__ == "__main__":
    main()
