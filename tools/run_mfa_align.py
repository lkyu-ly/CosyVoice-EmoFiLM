#!/usr/bin/env python3
"""MFA 批量强制对齐工具。

用法: python tools/run_mfa_align.py --manifest data/esd_manifest_train.jsonl --output_dir align/esd/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

def resolve_mfa_bin(cli_value=None):
    """按 CLI > MFA_BIN > PATH 的优先级解析 MFA 可执行文件。"""
    mfa_bin = cli_value or os.environ.get("MFA_BIN") or shutil.which("mfa")
    if mfa_bin:
        return mfa_bin
    raise FileNotFoundError(
        "MFA executable not found; pass --mfa_bin, set MFA_BIN, or add mfa to PATH"
    )


def build_subprocess_env(mfa_bin):
    """将 MFA 同目录置于 PATH 前部，同时保留当前环境。"""
    env = os.environ.copy()
    mfa_parent = str(Path(mfa_bin).parent)
    current_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(filter(None, (mfa_parent, current_path)))
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
    parser.add_argument("--mfa_bin", type=str)
    args = parser.parse_args()
    mfa_bin = resolve_mfa_bin(args.mfa_bin)
    subprocess_env = build_subprocess_env(mfa_bin)

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

    if args.validate_only:
        cmd = [mfa_bin, "validate", str(corpus), args.dictionary, args.acoustic_model, "--clean"]
        print(" ".join(cmd))
        subprocess.run(cmd, env=subprocess_env, check=True)
        print("Validate OK")
        return

    output = Path(args.output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        mfa_bin, "align",
        str(corpus), args.dictionary, args.acoustic_model,
        str(output),
        "--num_jobs", str(args.num_jobs),
        "--output_format", "long_textgrid",
        "--overwrite",
    ]
    if args.clean:
        cmd.append("--clean")
    print(" ".join(cmd))
    subprocess.run(cmd, env=subprocess_env, check=True)

    # 统计
    n_ok = sum(1 for s in samples if (output / f"{s['utt_id']}.TextGrid").is_file())
    print(f"Aligned: {n_ok}/{len(samples)}")


if __name__ == "__main__":
    main()
