#!/usr/bin/env python3
"""填 FEDD manifest 的 prompt 字段（2026-07-13）。

为 FEDD Part A（MiMo voice）填 neutral anchor 的 prompt_wav/prompt_text；
为 FEDD Part B（ESD speaker）填 ESD same-speaker Neutral wav + 其真实转写。
原地更新 manifest.jsonl，先备份 .bak_pre_anchor。幂等。

原则（ADR-0001）：prompt_text = prompt_wav 的真实转写，不用占位。

用法:
  python tools/fill_fedd_part_a_prompts.py
  # 默认读 data/fedd_rebuilt/manifest.jsonl + prompts/anchor_manifest.jsonl
  # ESD 查表用 data/raw_manifests/esd_train.jsonl（缺则 fallback esd_train_full_backup.jsonl）
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_anchor_map(anchor_manifest_path):
    """{voice: anchor_entry}"""
    m = {}
    for l in open(anchor_manifest_path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        m[r["voice"]] = r
    return m


def load_esd_neutral_lookup(esd_manifest_paths):
    """{spk: {wav_path, text}} for {spk}_000001（Neutral 第 1 句）。

    遍历所有 esd_manifest_paths（按序 fallback），取每 speaker 的 _000001 且 emo=neu。
    """
    lookup = {}
    for path in esd_manifest_paths:
        if not path or not os.path.isfile(path):
            continue
        for l in open(path, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            uid = r.get("utt_id", "")
            if not uid.endswith("_000001"):
                continue
            if r.get("sentence_emotion") != "neu":
                continue
            spk = uid.rsplit("_", 1)[0]
            if spk in lookup:
                continue  # 已有，保留首个
            lookup[spk] = {"wav_path": r["wav_path"], "text": r["text"]}
    return lookup


def fill_manifest(manifest_path, anchor_manifest_path, esd_manifest_paths, backup_path=None):
    """原地填 manifest.jsonl 的 Part A/B prompt 字段，先备份。

    仅在所有行处理成功后写回（中途 raise 不会破坏 manifest）。
    """
    anchor_map = load_anchor_map(anchor_manifest_path)
    esd_lookup = load_esd_neutral_lookup(esd_manifest_paths)

    if backup_path is None:
        backup_path = manifest_path + ".bak_pre_anchor"
    if not os.path.isfile(backup_path):
        shutil.copy2(manifest_path, backup_path)
        print(f"[fill] backup -> {backup_path}")

    rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8") if l.strip()]
    filled = 0
    for r in rows:
        part = r.get("part")
        spk = r.get("speaker_id", "")
        if part == "A":
            a = anchor_map.get(spk)
            if a is None:
                raise RuntimeError(f"Part A speaker_id={spk} 在 anchor_manifest 中未找到")
            r["prompt_wav"] = a["prompt_wav"]
            r["prompt_text"] = a["prompt_text"]
            r["prompt_source"] = a["prompt_source"]
            filled += 1
        elif part == "B":
            e = esd_lookup.get(spk)
            if e is None:
                raise RuntimeError(
                    f"Part B speaker_id={spk} 的 ESD Neutral _000001 未找到于 {esd_manifest_paths}")
            r["prompt_wav"] = e["wav_path"]
            r["prompt_text"] = e["text"]
            r["prompt_source"] = "esd_same_speaker_neutral"
            filled += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[fill] {filled}/{len(rows)} 条已填 prompt 字段 -> {manifest_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="填 FEDD manifest prompt 字段")
    parser.add_argument("--manifest", default="data/fedd_rebuilt/manifest.jsonl")
    parser.add_argument(
        "--anchor_manifest", default="data/fedd_rebuilt/prompts/anchor_manifest.jsonl")
    parser.add_argument(
        "--esd_manifest",
        default="data/raw_manifests/esd_train.jsonl,data/raw_manifests/esd_train_full_backup.jsonl",
        help="逗号分隔，按序 fallback 查 {spk}_000001")
    parser.add_argument("--backup", default=None)
    args = parser.parse_args()

    fill_manifest(
        manifest_path=args.manifest,
        anchor_manifest_path=args.anchor_manifest,
        esd_manifest_paths=[p.strip() for p in args.esd_manifest.split(",") if p.strip()],
        backup_path=args.backup,
    )


if __name__ == "__main__":
    main()
