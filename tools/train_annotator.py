#!/usr/bin/env python3
"""WordSequenceModel 词级情感分类器训练入口。

用法: python tools/train_annotator.py --data_dir word_blocks/ --manifest data/iemocap_manifest.jsonl --save_dir checkpoints/ --epochs=3

VAD 监督策略（spec 8.3, 6.2）：
- IEMOCAP 样本含 sentence_vad（来自 labels.csv 的 EmoVal/EmoAct/EmoDom），
  归一化 (raw - 1) / 4 从 [1,5] 映射到 [0,1]，对每个词复用句级 VAD 作为目标。
- ESD 样本无 VAD 标签，仅贡献 CE loss；MSE loss 按样本 mask 跳过。
- 训练日志显式打印 vad_n / no_vad_n 防止 VAD 标签被静默丢失。
"""
import argparse
import json
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 让脚本可作为子进程独立调用：把项目根加入 sys.path 以导入 cosyvoice_emo
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cosyvoice_emo.emo_annotator import WordSequenceModel

LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3, "sur": 4}
EMOTION_TO_LABEL = LABEL_MAP


class WordEmoDataset(Dataset):
    """每个样本返回 (frames, mask, label_id, vad_target_or_None)。

    vad_target 为长度 3 的 float32 张量 [valence, arousal, dominance]（归一化到 [0,1]），
    或 None（ESD 等无 VAD 标签数据集）。
    """

    def __init__(self, data_dir, manifest_path):
        self.samples = []
        with open(manifest_path, encoding="utf-8") as f:
            lines = f.readlines()
        manifest = {}
        for l in lines:
            rec = json.loads(l)
            manifest[rec["utt_id"]] = rec

        for utt_id, rec in manifest.items():
            sent_emo = rec["sentence_emotion"]
            utt_dir = os.path.join(data_dir, utt_id)
            if not os.path.isdir(utt_dir):
                continue
            word_files = sorted(os.listdir(utt_dir))
            if not word_files:
                continue
            label_id = EMOTION_TO_LABEL.get(sent_emo.strip())
            if label_id is None:
                continue
            # VAD 标签处理：IEMOCAP sentence_vad=[v,a,d] (raw [1,5]) → normalize [0,1]
            # ESD 无 sentence_vad 字段 → None（CE-only）
            sentence_vad = rec.get("sentence_vad", None)
            if sentence_vad is not None and len(sentence_vad) == 3:
                # 顺序 [valence, arousal, dominance]，与 spec 8.3 + Emo_PA pipeline_word_emotion.py:264-276 一致
                vad_tensor = torch.tensor(sentence_vad, dtype=torch.float32)
                vad_tensor = (vad_tensor - 1.0) / 4.0  # [1,5] → [0,1]
                vad_tensor = vad_tensor.clamp(0.0, 1.0)
            else:
                vad_tensor = None
            for wf in word_files:
                pt_path = os.path.join(utt_dir, wf)
                self.samples.append((pt_path, label_id, sent_emo, vad_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, label_id, sent_emo, vad_tensor = self.samples[idx]
        data = torch.load(pt_path, map_location="cpu")
        frames = data["frames"]
        mask = data.get("padding_mask", torch.zeros(frames.shape[0], dtype=torch.bool))
        return frames, mask, torch.tensor(label_id, dtype=torch.long), vad_tensor


def collate_fn(batch):
    """Pad 到 batch 内最大帧数。vad_target 为 None 的样本用 0 占位 + mask 标记跳过。"""
    max_len = max(b[0].shape[0] for b in batch)
    dim = batch[0][0].shape[1]
    frames_padded = []
    masks = []
    labels = []
    vad_targets = []
    vad_valid = []  # bool mask: True = 该样本有 VAD 监督
    for frames, mask, label, vad_tensor in batch:
        t = frames.shape[0]
        pad_len = max_len - t
        frames_padded.append(torch.cat([frames, torch.zeros(pad_len, dim)], dim=0))
        masks.append(torch.cat([mask, torch.ones(pad_len, dtype=torch.bool)], dim=0))
        labels.append(label)
        if vad_tensor is not None:
            vad_targets.append(vad_tensor)
            vad_valid.append(True)
        else:
            vad_targets.append(torch.zeros(3, dtype=torch.float32))
            vad_valid.append(False)
    return (
        torch.stack(frames_padded),
        torch.stack(masks),
        torch.stack(labels),
        torch.stack(vad_targets),
        torch.tensor(vad_valid, dtype=torch.bool),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "label_map.json"), "w") as f:
        json.dump(EMOTION_TO_LABEL, f)

    dataset = WordEmoDataset(args.data_dir, args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # 关键：分别统计含/不含 VAD 标签的样本数，防止静默丢失
    vad_n = sum(1 for s in dataset.samples if s[3] is not None)
    no_vad_n = len(dataset.samples) - vad_n
    print(f"Dataset: {len(dataset)} word samples (vad_n={vad_n}, no_vad_n={no_vad_n})")
    print(f"  含 VAD 监督样本占比 = {vad_n / max(1, len(dataset)):.2%}")

    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    model.to(args.device)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss(reduction="none")  # 自行按 mask 平均
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if vad_n == 0:
        print("[WARNING] manifest 中没有任何 sentence_vad 标签，λ_reg 项将完全失效！"
              "强度回归头不会被训练，Task 7 的 arousal 分桶将退化为单一 intensity。")

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for frames, masks, labels, vad_targets, vad_valid in loader:
            frames = frames.to(args.device)
            masks = masks.to(args.device)
            labels = labels.to(args.device)
            vad_targets = vad_targets.to(args.device)
            vad_valid = vad_valid.to(args.device)
            class_logits, vad_pred = model(frames, padding_mask=masks)
            loss_cls = cls_criterion(class_logits, labels)
            # 仅在含 VAD 标签的样本上计算 MSE（per-sample mean），再取有效样本均值
            if vad_valid.any():
                per_sample_mse = reg_criterion(vad_pred, vad_targets).mean(dim=-1)  # (B,)
                loss_reg = per_sample_mse[vad_valid].mean()
            else:
                loss_reg = torch.zeros((), device=args.device)
            loss = args.lambda_cls * loss_cls + args.lambda_reg * loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}, loss={avg_loss:.4f}, loss_reg={loss_reg.item():.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "word_sequence_model_best.pt"))
    print(f"Done. Best loss={best_loss:.4f}. Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
