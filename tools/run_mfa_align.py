#!/usr/bin/env python3
"""MFA 批量强制对齐工具。

PATH 注入: 确保 emofilm/bin (含 fstcompile) 与 conda base bin (含 sqlite3) 在 PATH 中。
复用 Stage 0 smoke_test_mfa.py 的封装模式。

用法: python tools/run_mfa_align.py --manifest data/esd_manifest_train.jsonl --output_dir align/esd/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


CONDA_BASE = "/home/lkyu/miniconda3"
EMOFILM_BIN = os.path.join(CONDA_BASE, "envs", "emofilm", "bin")


def _inject_path_env():
    env = os.environ.copy()
    current_path = env.get("PATH", "")
    parts = [EMOFILM_BIN, os.path.join(CONDA_BASE, "bin")] + [current_path]
    env["PATH"] = ":".join(parts)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="align")
    parser.add_argument("--dictionary", type=str, default="english_mfa")
    parser.add_argument("--acoustic_model", type=str, default="english_mfa")
    parser.add_argument("--temp_corpus_dir", type=str, default="/tmp/mfa_corpus_batch")
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--mfa_bin", type=str, default=f"{EMOFILM_BIN}/mfa")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} utterances")

    # prepare corpus
    corpus = Path(args.temp_corpus_dir)
    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True, exist_ok=True)

    for s in samples:
        wav_src = Path(s["wav_path"])
        shutil.copy2(wav_src, corpus / f"{s['utt_id']}.wav")
        lab = corpus / f"{s['utt_id']}.lab"
        lab.write_text(s["text"] + "\n", encoding="utf-8")

    env = _inject_path_env()

    if args.validate_only:
        cmd = [args.mfa_bin, "validate", str(corpus), args.dictionary, args.acoustic_model, "--clean"]
        print(" ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
        print("Validate OK")
        return

    output = Path(args.output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.mfa_bin, "align",
        str(corpus), args.dictionary, args.acoustic_model,
        str(output),
        "--num_jobs", str(args.num_jobs),
        "--output_format", "long_textgrid",
        "--overwrite",
    ]
    if args.clean:
        cmd.append("--clean")
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    # 统计
    n_ok = sum(1 for s in samples if (output / f"{s['utt_id']}.TextGrid").is_file())
    print(f"Aligned: {n_ok}/{len(samples)}")


if __name__ == "__main__":
    main()
