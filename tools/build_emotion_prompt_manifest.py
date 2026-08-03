#!/usr/bin/env python3
"""生成"情感匹配 prompt"验证 manifest：5 情感 × N 句完整同文本组。

从 sources/esd 的 tagged.jsonl（含 tagged_text，15127 行）挑出 (speaker, text)
5 情感齐全的组，每句 5 条 target 全保留；prompt_wav 用同说话人同情感的另一条
wav（不同句，避免自我克隆/内容泄漏）。输出 jsonl 供
tools/inference_emo_film.py 直接使用（manifest prompt_wav 优先）。
"""
import argparse
import collections
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True,
                        help="sources/esd/tagged.jsonl（含 tagged_text）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sentences", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.input_manifest, encoding="utf-8") if l.strip()]
    groups = collections.defaultdict(dict)  # (spk, text.lower()) -> {emotion: row}
    for r in rows:
        emo = r.get("sentence_emotion") or r.get("label")
        if emo:
            groups[(r["speaker_id"], r["text"].lower())][emo] = r

    # 跨句索引：(speaker, emotion) -> [该说话人该情感的所有行（不同句）]。
    # prompt 必须是同说话人同情感但【不同句】的 wav，避免与 target 同句造成自我克隆/内容泄漏。
    by_speaker_emo = collections.defaultdict(list)
    for r in rows:
        emo = r.get("sentence_emotion") or r.get("label")
        if emo:
            by_speaker_emo[(r["speaker_id"], emo)].append(r)

    full_groups = [k for k, v in groups.items() if len(v) >= 5]
    rng = random.Random(args.seed)
    chosen = rng.sample(full_groups, min(args.sentences, len(full_groups)))

    out = []
    for spk, text_lower in chosen:
        by_emo = groups[(spk, text_lower)]
        for emo, r in by_emo.items():
            target_text_lower = (r.get("text") or "").lower()
            prompt_candidates = [
                rr for rr in by_speaker_emo.get((spk, emo), [])
                if rr["utt_id"] != r["utt_id"]
                and (rr.get("text") or "").lower() != target_text_lower
            ]
            # 确定性选取：按 utt_id 排序后取第一条，保证可复现
            prompt_candidates.sort(key=lambda rr: rr["utt_id"])
            prompt = prompt_candidates[0] if prompt_candidates else r
            out.append({
                "utt_id": r["utt_id"],
                "wav_path": r["wav_path"],
                "target_wav": r["wav_path"],
                "reference_wav": r["wav_path"],
                "text": r["text"],
                "tagged_text": r["tagged_text"],
                "plain_text": r.get("plain_text") or r["text"],
                "sentence_emotion": emo,
                "label": emo,
                "speaker_id": spk,
                "prompt_wav": prompt["wav_path"],
                "prompt_text": prompt.get("text") or "",
                "prompt_source": "esd_same_speaker_same_emotion",
                "source_dataset": "esd",
            })
    with open(args.output, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows ({len(out) // 5} sentences x 5 emotions) -> {args.output}")


if __name__ == "__main__":
    main()
