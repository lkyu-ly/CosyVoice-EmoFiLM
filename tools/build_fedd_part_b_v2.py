#!/usr/bin/env python3
"""FEDD Part B v2：ESD 平行文本 + MFA 词边界切拼（strong transitions）。

对齐论文（学位论文 §FEDD / arXiv 2509.20378 §4.1）：
- 5 名说话人（0011/0012/0013 男 + 0015/0016 女，F0 实测判性别）
- 同说话人同文本在两种情感下的渲染，于词边界处切分拼接（近似"语义边界"），
  50ms sin²/cos² cross-fade，产物为**完整句子**（含真实转写，支撑 WER 评测）
- 5 spk × 20 有序情感对 × num_per_pair(5) = 500 条

替代已归档的 build_fedd.py Part B（其 bug 导致产物只有 50ms cross-fade 窗口）。
不修改 build_fedd.py（spec 不变量 #6）。

用法:
  python tools/build_fedd_part_b_v2.py \
    --esd_dir datasets/ESD \
    --mfa_dirs data/mfa_alignments/esd_train data/mfa_alignments/esd_test \
    --output_dir data/fedd_rebuilt/ \
    --num_per_pair 5 --seed 42
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
CROSSFADE_S = 0.05
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
EMO_SHORT = {"Angry": "ang", "Happy": "hap", "Neutral": "neu", "Sad": "sad", "Surprise": "sur"}
DEFAULT_SPEAKERS = ["0011", "0012", "0013", "0015", "0016"]  # 3M + 2F
GROUP_SIZE = 350  # ESD 每情感每说话人 350 条，utt 编号按情感分段


def load_parallel_groups(esd_dir, spk):
    """解析 {spk}.txt，返回 {text_group: {emotion: (utt_id, text)}}。

    仅保留 5 情感文本**严格相等**（strip 后）的组——ESD 转写存在少量措辞差异
    （如 "many many prisoners"），这类组直接跳过。
    """
    txt = Path(esd_dir) / spk / f"{spk}.txt"
    per_group = {}
    for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        utt_id, text, emo = parts[0], parts[1].strip(), parts[2].strip()
        if emo not in EMOTIONS:
            continue
        num = int(utt_id.split("_")[1])
        group = (num - 1) % GROUP_SIZE + 1
        per_group.setdefault(group, {})[emo] = (utt_id, text)

    result = {}
    for group, emo_map in per_group.items():
        if set(emo_map.keys()) != set(EMOTIONS):
            continue
        texts = {t for _, t in emo_map.values()}
        if len(texts) == 1:
            result[group] = emo_map
    return result


def _parse_words_tier(textgrid_path):
    """极简 TextGrid 解析：返回 words tier 的非空 interval 列表 [(xmin, xmax, word)]。"""
    content = Path(textgrid_path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r'name\s*=\s*"words"(.*?)(?:item\s*\[\d+\]:|\Z)', content, re.DOTALL)
    if not m:
        raise ValueError(f"no words tier in {textgrid_path}")
    words = []
    for iv in re.finditer(
        r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*\"(.*?)\"",
        m.group(1),
        re.DOTALL,
    ):
        x0, x1, w = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip()
        if w:
            words.append((x0, x1, w))
    return words


def word_boundary_times(textgrid_path, word_index, side="end"):
    """第 word_index 个词（1-indexed）的边界时刻。

    side="end": 该词右边界（A 侧切出点）。
    side="start_next": 第 word_index+1 个词的左边界（B 侧切入点）。
    返回 (time, n_words)。
    """
    words = _parse_words_tier(textgrid_path)
    n = len(words)
    if word_index < 1 or word_index >= n:
        raise ValueError(f"word_index {word_index} out of range (n_words={n})")
    if side == "end":
        return words[word_index - 1][1], n
    if side == "start_next":
        return words[word_index][0], n
    raise ValueError(f"bad side {side}")


def _find_textgrid(mfa_dirs, utt_id):
    for d in mfa_dirs:
        p = Path(d) / f"{utt_id}.TextGrid"
        if p.is_file():
            return str(p)
    return None


def _splice(y_a, t_a, y_b, t_b):
    """A[0:t_a] + 50ms cross-fade + B[t_b:end]，返回完整音频。"""
    n_fade = int(CROSSFADE_S * SR)
    seg_a = y_a[: int(round(t_a * SR))]
    seg_b = y_b[int(round(t_b * SR)):]
    if len(seg_a) <= n_fade or len(seg_b) <= n_fade:
        raise ValueError("segment shorter than cross-fade window")
    fade_out = np.cos(np.linspace(0, np.pi / 2, n_fade)) ** 2
    fade_in = np.sin(np.linspace(0, np.pi / 2, n_fade)) ** 2
    overlap = seg_a[-n_fade:] * fade_out + seg_b[:n_fade] * fade_in
    return np.concatenate([seg_a[:-n_fade], overlap, seg_b[n_fade:]])


def build_part_b_v2(esd_dir, mfa_dirs, output_dir, speakers=None,
                    num_per_pair=5, seed=42):
    """生成 Part B strong transitions，返回 manifest 条目列表。

    产物: {output_dir}/wav/fedd_b_*.wav + manifest.jsonl + part_b_source_utts.txt
    """
    speakers = speakers or DEFAULT_SPEAKERS
    rng = random.Random(seed)
    wav_dir = Path(output_dir) / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    source_utts = set()
    skipped = 0
    for spk in speakers:
        groups = load_parallel_groups(esd_dir, spk)
        group_ids = sorted(groups.keys())
        for e_from in EMOTIONS:
            for e_to in EMOTIONS:
                if e_from == e_to:
                    continue
                candidates = group_ids[:]
                rng.shuffle(candidates)
                made = 0
                for g in candidates:
                    if made >= num_per_pair:
                        break
                    utt_a, text = groups[g][e_from]
                    utt_b, _ = groups[g][e_to]
                    tg_a = _find_textgrid(mfa_dirs, utt_a)
                    tg_b = _find_textgrid(mfa_dirs, utt_b)
                    if not tg_a or not tg_b:
                        skipped += 1
                        continue
                    try:
                        n_words = len(_parse_words_tier(tg_a))
                        if n_words < 2 or len(_parse_words_tier(tg_b)) < 2:
                            skipped += 1
                            continue
                        k = max(1, min(n_words - 1, n_words // 2))
                        t_a, _ = word_boundary_times(tg_a, k, side="end")
                        t_b, _ = word_boundary_times(tg_b, min(k, len(_parse_words_tier(tg_b)) - 1),
                                                     side="start_next")
                        wav_a = Path(esd_dir) / spk / e_from / f"{utt_a}.wav"
                        wav_b = Path(esd_dir) / spk / e_to / f"{utt_b}.wav"
                        y_a, sr_a = sf.read(str(wav_a))
                        y_b, sr_b = sf.read(str(wav_b))
                        if sr_a != SR or sr_b != SR:
                            skipped += 1
                            continue
                        if y_a.ndim > 1:
                            y_a = y_a.mean(axis=1)
                        if y_b.ndim > 1:
                            y_b = y_b.mean(axis=1)
                        y = _splice(y_a, t_a, y_b, t_b)
                    except (ValueError, RuntimeError) as ex:
                        skipped += 1
                        continue
                    utt_id = (f"fedd_b_{spk}_{EMO_SHORT[e_from]}2{EMO_SHORT[e_to]}"
                              f"_{len(entries):04d}")
                    wav_path = str(wav_dir / f"{utt_id}.wav")
                    sf.write(wav_path, y.astype(np.float32), SR)
                    entries.append({
                        "utt_id": utt_id,
                        "wav_path": wav_path,
                        "text": text,
                        "emo_from": EMO_SHORT[e_from],
                        "emo_to": EMO_SHORT[e_to],
                        "emotion_transition": f"{e_from}→{e_to}",
                        "speaker_id": spk,
                        "boundary_word_index": k,
                        "source": "esd_parallel_word_boundary",
                        "part": "B",
                        "level": "strong",
                        "src_utt_from": utt_a,
                        "src_utt_to": utt_b,
                    })
                    source_utts.update([utt_a, utt_b])
                    made += 1

    out = Path(output_dir)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out / "part_b_source_utts.txt", "w") as f:
        f.write("\n".join(sorted(source_utts)) + "\n")
    with open(out / "metadata.json", "w") as f:
        json.dump({
            "part_b": len(entries),
            "speakers": speakers,
            "num_per_pair": num_per_pair,
            "seed": seed,
            "skipped": skipped,
            "source_utts": len(source_utts),
            "disclaimer": "This is FEDD-rebuilt, NOT the original FEDD dataset. "
                          "Part B splices ESD parallel-text renditions at MFA word "
                          "boundaries (paper: GPT-4o Audio speech cut at semantic "
                          "boundaries).",
        }, f, indent=2, ensure_ascii=False)
    print(f"Part B v2: {len(entries)} entries, {len(source_utts)} source utts, "
          f"{skipped} skipped -> {out / 'manifest.jsonl'}")
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esd_dir", required=True)
    ap.add_argument("--mfa_dirs", nargs="+", required=True,
                    help="按顺序查找 TextGrid 的目录列表")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    ap.add_argument("--num_per_pair", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_part_b_v2(args.esd_dir, args.mfa_dirs, args.output_dir,
                    speakers=args.speakers, num_per_pair=args.num_per_pair,
                    seed=args.seed)


if __name__ == "__main__":
    main()
