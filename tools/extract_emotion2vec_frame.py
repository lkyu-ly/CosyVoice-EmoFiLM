#!/usr/bin/env python3
"""从 manifest jsonl 批量提取 emotion2vec_plus_large 帧级特征。

用法: python tools/extract_emotion2vec_frame.py --manifest data/esd_manifest_train.jsonl --output_dir features/esd_frame/
"""
import argparse
import json
import os
import time
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from funasr import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str, default="iic/emotion2vec_plus_large")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    with open(args.manifest, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} utterances from {args.manifest}")

    model = AutoModel(model=args.model_id, disable_update=True)
    if device == "cuda":
        model.model.to(device)

    skipped = 0
    done = 0
    for sample in tqdm(samples, desc="extract"):
        out_path = os.path.join(args.output_dir, f"{sample['utt_id']}.pt")
        if os.path.exists(out_path):
            print(f"skip {sample['utt_id']} (already exists)")
            skipped += 1
            continue

        wav_path = sample["wav_path"]
        res = model.generate(wav_path, granularity="frame", extract_embedding=True)
        feats = res[0]["feats"]

        if isinstance(feats, torch.Tensor):
            feats = feats.cpu()
        else:
            feats = torch.from_numpy(feats)

        info = torchaudio.info(wav_path)
        dur = info.num_frames / info.sample_rate
        frame_rate_hz = feats.shape[0] / dur if dur > 0 else 50.0

        torch.save({"feats": feats, "frame_rate_hz": frame_rate_hz}, out_path)
        done += 1

    print(f"Done: {done} new, {skipped} skipped, {len(samples)} total")


if __name__ == "__main__":
    main()
