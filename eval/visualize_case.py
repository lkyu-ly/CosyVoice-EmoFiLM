#!/usr/bin/env python3
"""F0/mel可视化: GT / CosyVoice2 baseline / Emo-FiLM 三列对比。

用法: python eval/visualize_case.py --gt_wav a.wav --cosyvoice2_wav b.wav --emofilm_wav c.wav --output case.png
"""
import argparse
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_one(ax_f0, ax_mel, wav_path, title, sr=24000):
    y, _ = librosa.load(wav_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    times = librosa.times_like(f0, sr=sr)

    ax_f0.plot(times, f0, label=title)
    ax_f0.set_ylabel("F0 (Hz)")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1920, hop_length=480, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, hop_length=480, x_axis="time", y_axis="mel", ax=ax_mel)
    ax_mel.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_wav", type=str, required=True)
    parser.add_argument("--cosyvoice2_wav", type=str, required=True)
    parser.add_argument("--emofilm_wav", type=str, required=True)
    parser.add_argument("--output", type=str, default="case_study.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for wav, title, row in [(args.gt_wav, "GT", 0), (args.cosyvoice2_wav, "CosyVoice2", 1), (args.emofilm_wav, "Emo-FiLM", 2)]:
        plot_one(axes[row, 0], axes[row, 1], wav, title)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
