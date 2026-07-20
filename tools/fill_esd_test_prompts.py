#!/usr/bin/env python3
"""填 ESD test tagged 输入的 prompt 字段（2026-07-13）。

ESD test 输入（data/tagged_jsonl/esd_test.jsonl）无 part 字段，按 speaker_id
填 ESD same-speaker Neutral wav + 其真实转写，供 ESD 推理重跑用真实 prompt_text
（替代 'reference audio' 占位）。原地更新，先备份 .bak_pre_real_prompt_text。幂等。

原则（ADR-0001）：prompt_text = prompt_wav 真实转写。

用法:
  python tools/fill_esd_test_prompts.py
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fill_fedd_part_a_prompts import load_esd_neutral_lookup  # noqa: E402


def fill_esd_test(esd_test_path, esd_manifest_paths, backup_path=None):
    """原地填 ESD test 每条的 prompt_wav/prompt_text/prompt_source（按 speaker_id）。

    仅在所有行处理成功后写回（中途 raise 不破坏 manifest）。
    """
    esd_lookup = load_esd_neutral_lookup(esd_manifest_paths)

    if backup_path is None:
        backup_path = esd_test_path + ".bak_pre_real_prompt_text"
    if not os.path.isfile(backup_path):
        shutil.copy2(esd_test_path, backup_path)
        print(f"[fill_esd] backup -> {backup_path}")

    rows = [json.loads(l) for l in open(esd_test_path, encoding="utf-8") if l.strip()]
    filled = 0
    for r in rows:
        spk = r.get("speaker_id", "")
        e = esd_lookup.get(spk)
        if e is None:
            raise RuntimeError(
                f"ESD test speaker_id={spk} 的 Neutral _000001 未找到于 {esd_manifest_paths}")
        r["prompt_wav"] = e["wav_path"]
        r["prompt_text"] = e["text"]
        r["prompt_source"] = "esd_same_speaker_neutral"
        filled += 1

    with open(esd_test_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[fill_esd] {filled}/{len(rows)} 条已填 -> {esd_test_path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="填 ESD test 输入的 prompt 字段")
    ap.add_argument("--manifest", default="data/tagged_jsonl/esd_test.jsonl")
    ap.add_argument(
        "--esd_manifest",
        default="data/raw_manifests/esd_train.jsonl,data/raw_manifests/esd_train_full_backup.jsonl",
        help="逗号分隔，按序 fallback 查 {spk}_000001")
    ap.add_argument("--backup", default=None)
    args = ap.parse_args()
    fill_esd_test(
        args.manifest,
        [p.strip() for p in args.esd_manifest.split(",") if p.strip()],
        args.backup)


if __name__ == "__main__":
    main()
