#!/usr/bin/env python3
"""FEDD-rebuilt 构建脚本。

用法:
  python tools/build_fedd.py --output_dir fedd_rebuilt/ --esd_dir datasets/ESD/ --mode smoke
  python tools/build_fedd.py --output_dir fedd_rebuilt/ --esd_dir datasets/ESD/ --mode full --azure_key $KEY

smoke 模式: 从 ESD 构造最小 10 条以验证流程。
full 模式: 1000 条 (500 mild + 500 strong)。
"""
import argparse
import json
import os
import random
import numpy as np
import soundfile as sf
import torchaudio
from pathlib import Path


ESD_SPEAKERS = [f"{i:04d}" for i in range(11, 21)]
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]


def build_part_b_crossfade(esd_dir, manifest, rng, n):
    """从 ESD 同 speaker 不同情感构造 strong transition (cross-fade 50ms)。

    spec 6.3 要求：按 phone 边界切分并 50ms cross-fade 拼接。
    **smoke 模式限制（与 Part A 同）**：
    - smoke 模式：按音频中点切片做 sin²/cos² cross-fade（不依赖 MFA 对齐，便于快速验证）。
      标注 source='esd_crossfade_smoke'。
    - full 模式：必须先用 MFA 对齐 ESD 句子取得 phone 边界，从相邻 phone 边界处切分。
      标注 source='esd_crossfade_phone_boundary'。需在调用前用 MFA 对齐 ESD 全部英文子集。

    当前实现为 smoke 模式（中点切片）；full 模式 phone-boundary 切分需前置 MFA 对齐，
    由用户在拿到外部 TTS 凭证时切换到 full 模式时手工执行。
    """
    crossfade_samples = int(0.05 * 16000)  # 50ms @ 16kHz

    entries = []
    pairs = []
    for spk in ESD_SPEAKERS:
        spk_dir = Path(esd_dir) / spk
        emo_dirs = {e: sorted((spk_dir / e).glob("*.wav")) for e in EMOTIONS}
        for e1, wavs1 in emo_dirs.items():
            for e2, wavs2 in emo_dirs.items():
                if e1 == e2:
                    continue
                if wavs1 and wavs2:
                    pairs.append((spk, e1, e2, wavs1, wavs2))

    rng.shuffle(pairs)
    for spk, e1, e2, wavs1, wavs2 in pairs[:n]:
        w1 = rng.choice(wavs1)
        w2 = rng.choice(wavs2)
        y1, sr1 = torchaudio.load(str(w1))
        y2, sr2 = torchaudio.load(str(w2))
        if sr1 != 16000 or sr2 != 16000:
            continue
        y1, y2 = y1.squeeze().numpy(), y2.squeeze().numpy()
        half1 = len(y1) // 2
        half2 = len(y2) // 2
        part1 = y1[half1 - crossfade_samples // 2 : half1 + crossfade_samples // 2] if len(y1) > crossfade_samples else y1
        part2_start = half2 - crossfade_samples // 2
        part2 = y2[max(0, part2_start) : max(0, part2_start) + len(part1)] if len(y2) > crossfade_samples else y2
        min_len = min(len(part1), len(part2))
        part1, part2 = part1[:min_len], part2[:min_len]
        fade_out = np.cos(np.linspace(0, np.pi / 2, min_len)) ** 2
        fade_in = np.sin(np.linspace(0, np.pi / 2, min_len)) ** 2
        transition = part1 * fade_out + part2 * fade_in

        utt_id = f"fedd_b_{spk}_{e1}_{e2}_{len(entries):04d}"
        wav_path = os.path.join(manifest["output_dir"], "wav", f"{utt_id}.wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        sf.write(wav_path, transition, 16000)
        entries.append({
            "utt_id": utt_id, "wav_path": wav_path,
            "text": f"[{e1}→{e2} cross-fade]",
            "emotion_transition": f"{e1}→{e2}",
            "source": "esd_crossfade_smoke", "part": "B", "level": "strong",
        })
    return entries


def build_part_a_mild(manifest, rng, n):
    """Part A mild: 优先用外部 TTS。不可用时用 ESD 同 emotion 句子间 cross-fade 模拟。

    spec 6.3 要求：Part A 用外部 TTS 生成 mild transitions。
    smoke 模式（外部 TTS 不可用时）：用 ESD 同说话人同 emotion 两句做 sin²/cos² cross-fade。
    这是 spec 6.3 的降级实现，**source 字段必须标注 'esd_crossfade_mild_smoke'**
    以便后续筛选/排除。full 模式应使用 azure/openai TTS API（source='azure' 或 'openai'）。
    """
    entries = []
    esd_dir = Path(manifest["esd_dir"])
    crossfade_samples = 800  # 50ms @ 16kHz
    for spk in ESD_SPEAKERS:
        spk_dir = esd_dir / spk
        for emo_dir in sorted(spk_dir.glob("*")):
            if not emo_dir.is_dir():
                continue
            wavs = sorted(emo_dir.glob("*.wav"))
            if len(wavs) < 2:
                continue
            # smoke 模式：同 emotion 两句做 sin²/cos² cross-fade（**修复：不再用 hard concat + 静音**）
            y1, _ = torchaudio.load(str(wavs[0]))
            y2, _ = torchaudio.load(str(wavs[-1]))
            y1_np = y1.squeeze().numpy()
            y2_np = y2.squeeze().numpy()
            # 取前半段 y1 + 后半段 y2，在边界 50ms 做 cross-fade
            half1 = len(y1_np) // 2
            half2 = len(y2_np) // 2
            part1 = y1_np[:half1 + crossfade_samples // 2]
            part2 = y2_np[max(0, half2 - crossfade_samples // 2):]
            min_overlap = min(len(part1), len(part2), crossfade_samples)
            if min_overlap < crossfade_samples // 2:
                continue  # 音频过短无法 cross-fade，跳过
            fade_out = np.cos(np.linspace(0, np.pi / 2, min_overlap)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, min_overlap)) ** 2
            pre = part1[:-min_overlap] if len(part1) > min_overlap else np.array([], dtype=y1_np.dtype)
            transition = part1[-min_overlap:] * fade_out + part2[:min_overlap] * fade_in
            post = part2[min_overlap:] if len(part2) > min_overlap else np.array([], dtype=y2_np.dtype)
            combined = np.concatenate([pre, transition, post]).astype(np.float32)
            utt_id = f"fedd_a_{spk}_{emo_dir.name}_{len(entries):04d}"
            wav_path = os.path.join(manifest["output_dir"], "wav", f"{utt_id}.wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            sf.write(wav_path, combined, 16000)
            entries.append({
                "utt_id": utt_id, "wav_path": wav_path,
                "text": f"[{emo_dir.name} mild transition]",
                "emotion_transition": f"{emo_dir.name}→{emo_dir.name}",
                "source": "esd_crossfade_mild_smoke",  # 标注 smoke 模式降级
                "part": "A", "level": "mild",
            })
            if len(entries) >= n:
                break
        if len(entries) >= n:
            break
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--esd_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--azure_key", type=str, default="")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.output_dir, "wav"), exist_ok=True)
    manifest_meta = {"output_dir": args.output_dir, "esd_dir": args.esd_dir}

    entries = []
    n_a = min(args.num // 2, 5) if args.mode == "smoke" else 500
    n_b = args.num - n_a if args.mode == "smoke" else 500
    entries += build_part_a_mild(manifest_meta, rng, n_a)
    entries += build_part_b_crossfade(args.esd_dir, manifest_meta, rng, n_b)

    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "date": "2026-06-25",
            "mode": args.mode,
            "total": len(entries),
            "part_a": sum(1 for e in entries if e["part"] == "A"),
            "part_b": sum(1 for e in entries if e["part"] == "B"),
            "disclaimer": "This is FEDD-rebuilt, NOT the original FEDD dataset.",
        }, f, indent=2)

    print(f"Done. {len(entries)} entries in {manifest_path}")


if __name__ == "__main__":
    main()
