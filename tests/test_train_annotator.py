"""WordSequenceModel 训练入口单测（仅验证可加载 + val 评测契约，不实际大规模训练）。"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

EMOFILM_PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "tools" / "train_annotator.py")


def test_script_importable():
    """脚本 --help 可运行。"""
    r = subprocess.run([EMOFILM_PY, SCRIPT, "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_evaluate_metrics():
    """evaluate(model, loader, ...) 返回合法 (loss, acc)：acc∈[0,1]，完美可分≈1.0。

    抽函数便于 plan Task1 Step1.5 的 val_acc>0.6 验收可量化、可单测。
    """
    from tools.train_annotator import evaluate, make_collate_fn
    from cosyvoice_emo.emo_annotator import WordSequenceModel
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn

    class Const(Dataset):
        """每类用恒定 one-hot 向量，便于模型快速学出完美分类，验证 acc 计算。"""
        def __init__(self, n=20):
            self.items = []
            for i in range(n):
                lab = i % 5
                vec = torch.zeros(8, 768)
                vec[:, lab] = 1.0
                self.items.append((vec, torch.zeros(8, dtype=torch.bool),
                                   torch.tensor(lab), None))
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    model = WordSequenceModel()
    loader = DataLoader(Const(20), batch_size=4, collate_fn=make_collate_fn())
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    for _ in range(30):
        for f, m, y, _, _ in loader:
            opt.zero_grad()
            logits, _ = model(f, padding_mask=m)
            crit(logits, y).backward()
            opt.step()

    loss, acc = evaluate(model, loader, device="cpu",
                         cls_criterion=crit,
                         reg_criterion=nn.MSELoss(reduction="none"),
                         lambda_reg=0.5)
    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0
    assert acc > 0.8, f"完美可分数据 acc 应接近 1，实际 {acc}"


def test_mini_training_smoke():
    """最小 fake 数据训练 1 epoch（含 VAD 混合 + val 评测 + best.pt）。

    覆盖：
    - IEMOCAP 子集（含 sentence_vad）→ CE + MSE；ESD 子集（无 VAD）→ 仅 CE
    - val 划分 + 每 epoch 打印 val_loss/val_acc（plan Task1 Step1.3/1.5 验收）
    - checkpoint 文件名 = best.pt（plan Task1 产物 / Task2 命令契约）
    """
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)

    tmp = tempfile.mkdtemp(prefix="anno_train_")
    for utt_id in ["utt_a", "utt_b", "utt_c", "utt_d"]:
        utt_dir = os.path.join(tmp, "word_blocks", utt_id)
        os.makedirs(utt_dir)
        for wi in range(5):
            n_frames = np.random.randint(3, 15)
            torch.save({
                "frames": torch.randn(n_frames, 768),
                "word": f"w{wi}",
                "padding_mask": torch.zeros(n_frames, dtype=torch.bool),
            }, os.path.join(utt_dir, f"{wi:04d}_0_{n_frames}.pt"))

    manifest = os.path.join(tmp, "manifest.jsonl")
    labels = ["ang", "hap", "neu", "sad"]
    sentence_vads = [
        [3.2, 4.1, 2.8],  # utt_a IEMOCAP, high arousal
        [2.5, 2.0, 3.0],  # utt_b IEMOCAP, low arousal
        [4.0, 3.7, 4.2],  # utt_c IEMOCAP, medium arousal
        None,             # utt_d ESD, 无 VAD 标签
    ]
    with open(manifest, "w") as f:
        for i, utt_id in enumerate(["utt_a", "utt_b", "utt_c", "utt_d"]):
            rec = {"utt_id": utt_id, "sentence_emotion": labels[i]}
            if sentence_vads[i] is not None:
                rec["sentence_vad"] = sentence_vads[i]
            f.write(json.dumps(rec) + "\n")

    ckpt_dir = os.path.join(tmp, "ckpt")
    cmd = [
        EMOFILM_PY, SCRIPT,
        f"--data_dir={os.path.join(tmp, 'word_blocks')}",
        f"--manifest={manifest}",
        f"--save_dir={ckpt_dir}",
        "--epochs=1",
        "--batch_size=2",
        "--lr=1e-4",
        "--device=cpu",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    assert r.returncode == 0, r.stderr

    # checkpoint 文件名 = best.pt（对齐 plan Task1 产物 + Task2 --checkpoint 契约）
    ckpt_file = os.path.join(ckpt_dir, "best.pt")
    assert os.path.isfile(ckpt_file), f"missing {ckpt_file}"
    label_file = os.path.join(ckpt_dir, "label_map.json")
    assert os.path.isfile(label_file), f"missing {label_file}"
    with open(label_file) as f:
        lm = json.load(f)
    assert lm == {"ang": 0, "hap": 1, "neu": 2, "sad": 3, "sur": 4}

    # 训练日志必须显式打印两类样本数（防止 VAD 标签被静默丢失）
    assert "vad_n=" in r.stdout, "训练日志必须打印含 VAD 标签的样本数"
    assert "no_vad_n=" in r.stdout, "训练日志必须打印无 VAD 标签的样本数（ESD 子集）"
    # plan Task1 Step1.3/1.5 验收：必须打印 val_acc / val_loss
    assert "val_acc=" in r.stdout, "训练日志必须打印 val_acc（plan 验收依赖）"
    assert "val_loss=" in r.stdout, "训练日志必须打印 val_loss"
