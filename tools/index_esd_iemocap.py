#!/usr/bin/env python3
"""ESD / IEMOCAP 数据索引：扫描原始目录，生成带统一 schema 的 JSONL manifest。

ESD 布局: {speaker}/{Emotion}/{speaker}_{seq:06d}.wav + {speaker}/{speaker}.txt
IEMOCAP 布局: labels.csv + wav/ 目录

输出每行: {"utt_id": str, "wav_path": str, "text": str, "sentence_emotion": str, "speaker_id": str}
sentence_emotion ∈ {ang, hap, neu, sad, sur}
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path


ESD_EMOTION_MAP = {
    "angry": "ang", "happy": "hap", "neutral": "neu",
    "sad": "sad", "surprise": "sur",
}

IEMOCAP_MAP = {
    "angry": "ang", "happy": "hap", "neutral": "neu",
    "sad": "sad", "surprise": "sur", "excited": "hap",
}

ENGLISH_SPEAKERS = {f"{i:04d}" for i in range(11, 21)}


def index_esd(data_dir: Path, english_only: bool = True):
    """返回 list[dict]."""
    samples = []

    # 先读所有 .txt 文件
    txt_index = {}
    for spk_dir in sorted(data_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        if english_only and spk_dir.name not in ENGLISH_SPEAKERS:
            continue
        txt_file = spk_dir / f"{spk_dir.name}.txt"
        if txt_file.is_file():
            with open(txt_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        utt_id, text, emotion = parts[0], parts[1], parts[2]
                        # emotion 英文大写 → 小写 → id
                        emo_id = ESD_EMOTION_MAP.get(emotion.lower())
                        if emo_id is None:
                            continue
                        txt_index[utt_id] = (text, emo_id)

    # 扫描 wav
    for spk_dir in sorted(data_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        if english_only and spk_dir.name not in ENGLISH_SPEAKERS:
            continue
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir():
                continue
            for wav_file in sorted(emo_dir.glob("*.wav")):
                utt_id = wav_file.stem
                info = txt_index.get(utt_id)
                if info is None:
                    continue
                text, emo_id = info
                samples.append({
                    "utt_id": utt_id,
                    "wav_path": str(wav_file.resolve()),
                    "text": text,
                    "sentence_emotion": emo_id,
                    "speaker_id": spk_dir.name,
                })
    return samples


def index_iemocap(data_dir: Path):
    """返回 list[dict]。

    每条记录可选包含 `sentence_vad: [valence, arousal, dominance]`（原始 [1,5]）。
    来源：IEMOCAP labels.csv 的 EmoVal/EmoAct/EmoDom 字段（spec 6.2 + 8.3）。
    Task 6 训练器会做 (v-1)/4 归一化到 [0,1]。
    """
    labels_csv = data_dir / "labels.csv"
    if not labels_csv.is_file():
        raise FileNotFoundError(f"missing {labels_csv}")

    samples = []
    with open(labels_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emo_raw = row.get("major_emotion", "").strip().lower()
            emo_id = IEMOCAP_MAP.get(emo_raw)
            if emo_id is None:
                continue
            wav_path = row.get("wav_path", "")
            if not wav_path or not Path(wav_path).is_file():
                # 尝试 ../datasets/IEMOCAP/wav/ 下相对路径
                wav_rel = row.get("file", "")
                if wav_rel:
                    alt = data_dir / "wav" / wav_rel
                    if alt.is_file():
                        wav_path = str(alt.resolve())
                    else:
                        continue
                else:
                    continue
            # speaker_id: 从 wav 文件名提取 Session+speaker 前缀 (如 Ses01F)
            fn = Path(wav_path).stem
            spk_match = re.match(r"^(Ses\d+[FM])", fn)
            speaker_id = spk_match.group(1) if spk_match else "unknown"
            # VAD：csv 顺序 EmoAct/EmoVal/EmoDom → 输出 [valence, arousal, dominance]
            # 与 spec 8.3 + Emo_PA pipeline_word_emotion.py:264-276 一致
            rec = {
                "utt_id": fn,
                "wav_path": str(Path(wav_path).resolve()),
                "text": row.get("transcription", "").strip(),
                "sentence_emotion": emo_id,
                "speaker_id": speaker_id,
            }
            try:
                emo_val = float(row.get("EmoVal", ""))
                emo_act = float(row.get("EmoAct", ""))
                emo_dom = float(row.get("EmoDom", ""))
                # 范围校验：[1,5]，越界视为缺失
                if all(1.0 <= v <= 5.0 for v in (emo_val, emo_act, emo_dom)):
                    rec["sentence_vad"] = [emo_val, emo_act, emo_dom]
            except (ValueError, TypeError):
                pass  # 字段缺失或非数值，跳过 VAD 标签
            samples.append(rec)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["esd", "iemocap"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--test_output", type=str, default=None)
    parser.add_argument("--test_per_speaker", type=int, default=30)
    parser.add_argument("--english_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    if args.dataset == "esd":
        all_samples = index_esd(data_dir, english_only=args.english_only)
    else:
        all_samples = index_iemocap(data_dir)

    if args.dataset == "esd" and args.test_output:
        # split: 每 speaker 每 emotion 30 条 test
        train, test = [], []
        grouped = {}
        for s in all_samples:
            key = (s["speaker_id"], s["sentence_emotion"])
            grouped.setdefault(key, []).append(s)
        for key, lst in grouped.items():
            random.shuffle(lst)
            test.extend(lst[:args.test_per_speaker])
            train.extend(lst[args.test_per_speaker:])
        with open(args.output, "w", encoding="utf-8") as f:
            for s in train:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        with open(args.test_output, "w", encoding="utf-8") as f:
            for s in test:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"ESD: {len(train)} train, {len(test)} test")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            for s in all_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{args.dataset}: {len(all_samples)} samples")


if __name__ == "__main__":
    main()
