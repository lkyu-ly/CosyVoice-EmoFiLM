#!/usr/bin/env python3
"""WordSequenceModel 词级情感分类器训练入口。

用法（仅保留作者 WordSequenceModel 结构检查/历史兼容训练）：
  python tools/train_annotator.py \
    --data_dir word_blocks_iemocap/ word_blocks_esd/ \
    --manifest data/raw_manifests/iemocap_train.jsonl data/raw_manifests/esd_train.jsonl \
    --save_dir checkpoints/ --epochs 3 --batch_size 4

数据源（spec 表2-1 标注预测阶段 = IEMOCAP + ESD）：
- --data_dir / --manifest 均可传多个（一一对应），合并训练。
- IEMOCAP 样本含 sentence_vad → 贡献强度回归监督；ESD 无 VAD → 仅 CE，MSE 按样本 mask 跳过。

作者合同固定为 768d 输入、5 类情感和 3D VAD；本入口不再暴露旧的 1024d/1D 分支。
"""
import argparse
import json
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# 让脚本可作为子进程独立调用：把项目根加入 sys.path 以导入 cosyvoice_emo
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cosyvoice_emo.emo_annotator import WordSequenceModel

LABEL_MAP = {"ang": 0, "hap": 1, "neu": 2, "sad": 3, "sur": 4}
EMOTION_TO_LABEL = LABEL_MAP


class WordEmoDataset(Dataset):
    """每个样本返回 (frames, mask, label_id, reg_target_or_None)。

    reg_target 为长度 3 的 float32 VAD 张量（归一化到 [0,1]）：
    [valence, arousal, dominance]，与作者 WordSequenceModel checkpoint 一致。
    或 None（ESD 等无 VAD 标签数据集）。

    data_dirs / manifest_paths 支持多源（str 或等长 list），合并所有源的样本。
    """

    def __init__(self, data_dirs, manifest_paths):
        self.samples = []
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        if isinstance(manifest_paths, str):
            manifest_paths = [manifest_paths]
        assert len(data_dirs) == len(manifest_paths), \
            f"data_dir 与 manifest 数量须一致: {len(data_dirs)} vs {len(manifest_paths)}"

        for data_dir, manifest_path in zip(data_dirs, manifest_paths):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = {json.loads(l)["utt_id"]: json.loads(l) for l in f if l.strip()}
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
                    vad = torch.tensor(sentence_vad, dtype=torch.float32)
                    vad = ((vad - 1.0) / 4.0).clamp(0.0, 1.0)  # [1,5] → [0,1]
                    reg_target = vad
                else:
                    reg_target = None
                for wf in word_files:
                    self.samples.append((os.path.join(utt_dir, wf), label_id, sent_emo, reg_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, label_id, sent_emo, vad_tensor = self.samples[idx]
        data = torch.load(pt_path, map_location="cpu")
        frames = data["frames"]
        mask = data.get("padding_mask", torch.zeros(frames.shape[0], dtype=torch.bool))
        return frames, mask, torch.tensor(label_id, dtype=torch.long), vad_tensor


def make_collate_fn():
    """返回 collate_fn：pad 到 batch 内最大帧数；reg_target 为 None 的样本用 0 占位 + mask 标记跳过。"""
    def collate_fn(batch):
        max_len = max(b[0].shape[0] for b in batch)
        dim = batch[0][0].shape[1]
        frames_padded, masks, labels, reg_targets, reg_valid = [], [], [], [], []
        for frames, mask, label, reg_tensor in batch:
            pad_len = max_len - frames.shape[0]
            frames_padded.append(torch.cat([frames, torch.zeros(pad_len, dim)], dim=0))
            masks.append(torch.cat([mask, torch.ones(pad_len, dtype=torch.bool)], dim=0))
            labels.append(label)
            if reg_tensor is not None:
                reg_targets.append(reg_tensor)
                reg_valid.append(True)
            else:
                reg_targets.append(torch.zeros(3, dtype=torch.float32))
                reg_valid.append(False)
        return (
            torch.stack(frames_padded),
            torch.stack(masks),
            torch.stack(labels),
            torch.stack(reg_targets),
            torch.tensor(reg_valid, dtype=torch.bool),
        )
    return collate_fn


@torch.no_grad()
def evaluate(model, loader, device, cls_criterion, reg_criterion, lambda_reg):
    """在 val/test loader 上计算 (avg_loss, accuracy)。

    loss 口径与训练一致（lambda_cls·CE + lambda_reg·MSE，仅含 VAD 样本计入 MSE）；
    accuracy = 分类 top-1 准确率（5 类）。抽函数便于 plan Task1 Step1.5 的 val_acc 验收。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for frames, masks, labels, vad_targets, vad_valid in loader:
        frames = frames.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        vad_targets = vad_targets.to(device)
        vad_valid = vad_valid.to(device)
        class_logits, vad_pred = model(frames, padding_mask=masks)
        loss_cls = cls_criterion(class_logits, labels)
        if vad_valid.any():
            per_sample_mse = reg_criterion(vad_pred, vad_targets).mean(dim=-1)
            loss_reg = per_sample_mse[vad_valid].mean()
        else:
            loss_reg = torch.zeros((), device=device)
        total_loss += (loss_cls + lambda_reg * loss_reg).item()
        correct += (class_logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, len(loader)), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs="+", required=True,
                        help="词块特征目录，可多源（与 --manifest 一一对应），如 IEMOCAP + ESD")
    parser.add_argument("--manifest", type=str, nargs="+", required=True,
                        help="manifest jsonl，可多源（与 --data_dir 一一对应）")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)  # 论文表2-1 = 4
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="从训练样本划出 val 比例（plan Task1 验收 val_acc 用）")
    parser.add_argument("--seed", type=int, default=42, help="val 划分随机种子（可复现）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "label_map.json"), "w") as f:
        json.dump(EMOTION_TO_LABEL, f)

    dataset = WordEmoDataset(args.data_dir, args.manifest)
    # 关键：分别统计含/不含 VAD 标签的样本数，防止静默丢失
    vad_n = sum(1 for s in dataset.samples if s[3] is not None)
    no_vad_n = len(dataset.samples) - vad_n
    print(f"Dataset: {len(dataset)} word samples (vad_n={vad_n}, no_vad_n={no_vad_n})")
    print(f"  含 VAD 监督样本占比 = {vad_n / max(1, len(dataset)):.2%}")

    # val 划分（固定种子可复现）；val 太小（<1 batch）时退化为全量训练 + 同集评测
    n_val = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    if n_val and n_val < len(dataset):
        val_set, train_set = random_split(
            dataset, [n_val, len(dataset) - n_val],
            generator=torch.Generator().manual_seed(args.seed))
    else:
        train_set, val_set = dataset, None
    collate = make_collate_fn()
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = (DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
                  if val_set is not None else None)
    if val_loader is not None:
        print(f"  train={len(train_set)} val={len(val_set)}（val_ratio={args.val_ratio}）")

    model = WordSequenceModel()
    model.to(args.device)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss(reduction="none")  # 自行按 mask 平均
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if vad_n == 0:
        print("[WARNING] manifest 中没有任何 sentence_vad 标签，λ_reg 项将完全失效！"
              "强度回归头不会被训练，Task 7 的 arousal 分桶将退化为单一 intensity。")

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
        last_reg = 0.0
        for frames, masks, labels, vad_targets, vad_valid in pbar:
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
            last_reg = loss_reg.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", reg=f"{last_reg:.4f}")
        avg_loss = total_loss / len(loader)

        # 每 epoch 末 val 评测（plan Task1 Step1.3/1.5 验收依赖）
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, args.device,
                                         cls_criterion, reg_criterion, args.lambda_reg)
            print(f"[Epoch {epoch+1}] train_loss={avg_loss:.4f}, loss_reg={last_reg:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
        else:
            # 无 val 集：按 train loss 存（兜底，正常规模不会走到这）
            print(f"[Epoch {epoch+1}] train_loss={avg_loss:.4f}, loss_reg={last_reg:.4f} (no val set)")
            if -avg_loss > best_val_acc:
                best_val_acc = -avg_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))

    if val_loader is not None:
        print(f"Done. Best val_acc={best_val_acc:.4f}. Saved best.pt to {args.save_dir}")
    else:
        print(f"Done. Best train_loss={-best_val_acc:.4f}. Saved best.pt to {args.save_dir}")


if __name__ == "__main__":
    main()
