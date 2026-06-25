#!/usr/bin/env python3
"""Per-emotion 分类准确率: 用 WordSequenceModel 预测生成音频的 5 类情感。

用法: python eval/per_emo_accuracy.py --hyp_dir wav_hyp/ --annotator_ckpt checkpoints/word_sequence_model_best.pt --gt_manifest data/esd_manifest_test.jsonl --output accuracy.json

输出 JSON schema (与 plan 4 Task 2 接口声明一致):
{
    "accuracy": float,                  # 总体准确率 = trace / total
    "confusion_matrix": list[list[int]],# 5×5 矩阵, rows=gt, cols=pred
    "per_class": {                      # per-class precision/recall
        "ang": {"precision": float, "recall": float, "f1": float, "support": int},
        ...
    },
    "predictions": list[dict],          # 每条预测明细
    "n_total": int
}
"""
import os
# 离线模式：跳过 modelscope 更新检查（与 eval_emo_film.py 一致）
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
import numpy as np
import torch
from cosyvoice_emo.emo_annotator import WordSequenceModel
from funasr import AutoModel


LABELS = ["ang", "hap", "neu", "sad", "sur"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for i, l in enumerate(LABELS)}

# ESD 文件名 → emotion 映射（spec 6.1 + 实测 ESD 命名约定）
# 备选: 通过 --gt_manifest 提供显式 utt_id → emotion 映射
ESD_FILENAME_EMOTION_RE = re.compile(r"_([A-Za-z]+)\.wav$")


def parse_gt_emotion(wav_name, gt_manifest=None):
    """从 wav_name 或 gt_manifest 推断 ground-truth emotion label。

    优先级：
    1. gt_manifest（含 utt_id → sentence_emotion 显式映射，最可靠）
    2. wav_name 启发式（ESD: 0011_000351.wav → 需查 txt；FEDD-rebuilt: case_001_hap.wav → 解析）
    """
    if gt_manifest is not None:
        # utt_id 去除 .wav 后缀
        utt_id = os.path.splitext(wav_name)[0]
        if utt_id in gt_manifest:
            return gt_manifest[utt_id]
    # 启发式 fallback：尝试从文件名尾部解析 emotion 关键字
    m = ESD_FILENAME_EMOTION_RE.search(wav_name)
    if m:
        raw = m.group(1).lower()
        short = {"angry": "ang", "happy": "hap", "neutral": "neu",
                 "sad": "sad", "surprise": "sur"}.get(raw)
        if short in LABELS:
            return short
    return None


def compute_confusion_matrix(gt_labels, pred_labels):
    """构造 5×5 混淆矩阵，rows=gt, cols=pred。缺失 gt 的样本不计入。"""
    n = len(LABELS)
    cm = [[0] * n for _ in range(n)]
    for gt, pred in zip(gt_labels, pred_labels):
        if gt is None or gt not in LABEL_TO_IDX:
            continue
        if pred not in LABEL_TO_IDX:
            continue
        cm[LABEL_TO_IDX[gt]][LABEL_TO_IDX[pred]] += 1
    return cm


def compute_per_class_metrics(cm):
    """从混淆矩阵计算 per-class precision/recall/f1/support。"""
    metrics = {}
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(LABELS))) - tp
        fn = sum(cm[i][c] for c in range(len(LABELS))) - tp
        support = sum(cm[i])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_dir", type=str, required=True)
    parser.add_argument("--annotator_ckpt", type=str, required=True)
    parser.add_argument("--gt_manifest", type=str, default=None,
                        help="ESD/IEMOCAP manifest jsonl 提供显式 utt_id → emotion 映射")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # 显式校验 ckpt 文件存在（spec 13 测试矩阵要求）
    if not os.path.isfile(args.annotator_ckpt):
        raise FileNotFoundError(
            f"annotator_ckpt not found: {args.annotator_ckpt}. "
            f"Stage 1 WordSequenceModel checkpoint is required."
        )

    # 用 inspect 验证 WordSequenceModel 构造签名（防止 Plan 1 与 Plan 4 接口漂移）
    import inspect
    sig = inspect.signature(WordSequenceModel.__init__)
    expected_params = {"input_dim", "num_classes", "num_heads", "dropout_rate"}
    actual_params = set(sig.parameters.keys()) - {"self"}
    assert expected_params.issubset(actual_params), (
        f"WordSequenceModel.__init__ 签名与 plan 4 期望不符。"
        f"expected >= {expected_params}, got {actual_params}. "
        f"Plan 1 与 Plan 4 接口漂移，请核对。"
    )

    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    model.load_state_dict(torch.load(args.annotator_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # emotion2vec_plus_large 显式指定 device，避免 CPU smoke 失败
    emo_model = AutoModel(
        model="iic/emotion2vec_plus_large",
        disable_update=True,
        device=device,
    )

    # 加载 gt_manifest（如提供）
    gt_manifest = None
    if args.gt_manifest and os.path.isfile(args.gt_manifest):
        gt_manifest = {}
        with open(args.gt_manifest, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                utt_id = rec.get("utt_id") or os.path.splitext(os.path.basename(rec.get("wav_path", "")))[0]
                gt_manifest[utt_id] = rec.get("sentence_emotion")

    wav_files = sorted([f for f in os.listdir(args.hyp_dir) if f.endswith(".wav")])
    predictions = []
    gt_labels = []
    pred_labels = []
    for wav_name in wav_files:
        wav_path = os.path.join(args.hyp_dir, wav_name)
        try:
            res = emo_model.generate(wav_path, granularity="frame", extract_embedding=True)
            feats = res[0]["feats"]
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu()
            feats_t = feats.unsqueeze(0).float().to(device)
            mask = torch.zeros(1, feats_t.shape[1], dtype=torch.bool, device=device)
            with torch.no_grad():
                logits, _ = model(feats_t, mask)
            pred_idx = int(logits.argmax(dim=1).item())
            pred_label = IDX_TO_LABEL.get(pred_idx, "unknown")
            gt_label = parse_gt_emotion(wav_name, gt_manifest)
            predictions.append({"file": wav_name, "predicted": pred_label, "gt": gt_label})
            pred_labels.append(pred_label)
            gt_labels.append(gt_label)
        except Exception as e:
            print(f"WARN: {wav_name} failed: {e}")

    # 计算混淆矩阵 + per-class metrics + 总体 accuracy
    cm = compute_confusion_matrix(gt_labels, pred_labels)
    per_class = compute_per_class_metrics(cm)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(LABELS)))
    accuracy = correct / total if total > 0 else 0.0

    output = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class": per_class,
        "predictions": predictions,
        "n_total": len(predictions),
        "labels": LABELS,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Done. accuracy={accuracy:.4f} ({correct}/{total}), {len(predictions)} predictions in {args.output}")


if __name__ == "__main__":
    main()
