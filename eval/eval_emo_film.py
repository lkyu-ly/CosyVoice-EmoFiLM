#!/usr/bin/env python3
"""客观评测指标计算: Emo-SIM, DTW, WER。

用法: python eval/eval_emo_film.py --ref_dir wav_ref/ --hyp_dir wav_hyp/ --output result.json

DTW 距离度量选择（spec 12.1 + Self-Review 决策记录）：
- 默认 Euclidean（与论文 DTW 数值 23.98/49.62/133.97/73.96 单位量级一致）
- 备选 cosine：通过 --dtw_dist cosine 切换（适合 emotion2vec 嵌入空间内比较）
- 同时保存 raw accumulated distance（path 累加）和 path-normalized distance

Emo-SIM 实现（spec 12.1）：
- utterance feats 已是 (1024,) 向量（Stage 0 实测），**不需要均值池化**
- 直接 L2 normalize → 余弦相似度 ×100（单位：百分比）

WER normalization（spec 12.1）：
- 论文未公开 normalization 细节，作者源码 evaluate.py 只实现 ACC-E/SS，未实现 WER
- 本实现采用：lowercase + 去标点 + 数字→英文 + 多空格合并 + strip
"""
import os
# 离线模式：跳过 modelscope 更新检查（避免每次启动重新下载 2.88G 模型）
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
import argparse
import json
import re
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
from fastdtw import fastdtw
from scipy.spatial.distance import cosine, euclidean
import whisper


def compute_emo_sim(model, wav_path, device):
    """提取 utterance feats (1024,) → L2 normalize 返回单位向量。

    Stage 0 实测确认 utterance feats 已是 (1024,)，无需均值池化（spec 12.1 措辞修正）。
    """
    res = model.generate(wav_path, granularity="utterance", extract_embedding=True)
    feats = res[0]["feats"]
    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().numpy()
    feats = feats.reshape(-1)  # utterance feats 已是 1D, reshape 为 no-op 但保险
    return feats / (np.linalg.norm(feats) + 1e-8)


def compute_dtw(model, ref_path, hyp_path, device, dist_fn="euclidean"):
    """计算 fastdtw 距离，返回 (raw_distance, path_normalized_distance)。

    Args:
        dist_fn: 'euclidean' (默认) 或 'cosine'。Euclidean 与论文数值量级一致；
                 cosine 适合在 emotion2vec 嵌入空间内比较。
    """
    ref_res = model.generate(ref_path, granularity="frame", extract_embedding=True)
    hyp_res = model.generate(hyp_path, granularity="frame", extract_embedding=True)
    ref_feats = ref_res[0]["feats"]
    hyp_feats = hyp_res[0]["feats"]
    if isinstance(ref_feats, torch.Tensor):
        ref_feats = ref_feats.cpu().numpy()
    if isinstance(hyp_feats, torch.Tensor):
        hyp_feats = hyp_feats.cpu().numpy()
    dist_fn_impl = euclidean if dist_fn == "euclidean" else cosine
    dist, path = fastdtw(ref_feats, hyp_feats, dist=dist_fn_impl)
    path_len = len(path) if path else 1
    return dist, dist / path_len


def compute_wer(whisper_model, wav_path):
    """Whisper-large-v3 转写为参考文本（不是 WER 本身）。"""
    result = whisper_model.transcribe(wav_path)
    return result["text"].strip().lower() if result else ""


_NUM_WORD_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _normalize_digits(text):
    """简单数字→英文（0-9 单字符）。两位以上数字保持原样（论文未公开规则）。"""
    for d, w in _NUM_WORD_MAP.items():
        text = text.replace(d, w)
    return text


def normalize_text(text):
    """WER normalization (论文未公开, Emo_PA 未实现 WER)。

    规则：lowercase + 去标点 + 数字转英文 + 多空格合并 + strip。
    不做 contraction expansion（don't → do not），因 Whisper 转写很少产出缩写。
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = _normalize_digits(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--hyp_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtw_dist", choices=["euclidean", "cosine"], default="euclidean",
                        help="DTW 距离度量：euclidean (默认, 论文量级) / cosine (嵌入空间)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    # emotion2vec_plus_large + whisper-large-v3 显式指定 device
    emo_model = AutoModel(
        model="iic/emotion2vec_plus_large",
        disable_update=True,
        device=device,
    )
    whisper_model = whisper.load_model("large-v3", device=device)

    ref_files = sorted([f for f in os.listdir(args.ref_dir) if f.endswith(".wav")])
    hyp_files = sorted([f for f in os.listdir(args.hyp_dir) if f.endswith(".wav")])
    common = [f for f in ref_files if f in hyp_files]
    if not common:
        common = ref_files[: min(len(ref_files), len(hyp_files))]

    emo_sims, dtws, dtw_norms, wers = [], [], [], []
    for fname in common:
        ref_path = os.path.join(args.ref_dir, fname)
        hyp_path = os.path.join(args.hyp_dir, fname)
        try:
            ref_feats = compute_emo_sim(emo_model, ref_path, device)
            hyp_feats = compute_emo_sim(emo_model, hyp_path, device)
            emo_sims.append(float(np.dot(ref_feats, hyp_feats) * 100))

            dist, dist_norm = compute_dtw(emo_model, ref_path, hyp_path, device, args.dtw_dist)
            dtws.append(dist)
            dtw_norms.append(dist_norm)

            ref_text = compute_wer(whisper_model, ref_path)
            hyp_text = compute_wer(whisper_model, hyp_path)
            from jiwer import wer
            wers.append(wer(normalize_text(ref_text), normalize_text(hyp_text)))
        except Exception as e:
            print(f"WARN: {fname} failed: {e}")

    result = {
        "emo_sim": float(np.mean(emo_sims)) if emo_sims else 0,
        "dtw": float(np.mean(dtws)) if dtws else 0,
        "dtw_normalized": float(np.mean(dtw_norms)) if dtw_norms else 0,
        "wer": float(np.mean(wers)) if wers else 1,
        "n_samples": len(common),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done. {result}")


if __name__ == "__main__":
    main()
