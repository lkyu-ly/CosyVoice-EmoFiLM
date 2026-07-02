"""WordSequenceModel 训练入口单测（仅验证可加载，不实际训练）。"""
import json
import os
import subprocess
import tempfile

import torch

EMOFILM_PY = "/home/hanlvyuan/miniconda3/envs/emofilm/bin/python"
ROOT = "/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM"
SCRIPT = f"{ROOT}/tools/train_annotator.py"


def test_script_importable():
    """脚本 --help 可运行。"""
    r = subprocess.run([EMOFILM_PY, SCRIPT, "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_mini_training_smoke():
    """最小 fake 数据训练 1 epoch（含 VAD 标签混合场景）。

    覆盖：
    - IEMOCAP 子集样本（含 sentence_vad）→ 同时计算 CE + MSE
    - ESD 子集样本（无 sentence_vad）→ 仅计算 CE，loss_reg 跳过
    """
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)

    tmp = tempfile.mkdtemp(prefix="anno_train_")
    # generate fake word_blocks for 4 utterances
    # utt_a/utt_b/utt_c 模拟 IEMOCAP（含 VAD），utt_d 模拟 ESD（无 VAD）
    for utt_id in ["utt_a", "utt_b", "utt_c", "utt_d"]:
        utt_dir = os.path.join(tmp, "word_blocks", utt_id)
        os.makedirs(utt_dir)
        for wi in range(5):
            n_frames = np.random.randint(3, 15)
            torch.save({
                "frames": torch.randn(n_frames, 1024),
                "word": f"w{wi}",
                "padding_mask": torch.zeros(n_frames, dtype=torch.bool),
            }, os.path.join(utt_dir, f"{wi:04d}_0_{n_frames}.pt"))

    # manifest with sentence labels + 可选 VAD（IEMOCAP 有，ESD 无）
    # VAD 顺序 [valence, arousal, dominance]，原始 [1,5]，归一化到 [0,1] 用 (v-1)/4
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
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
    assert r.returncode == 0, r.stderr

    ckpt_file = os.path.join(ckpt_dir, "word_sequence_model_best.pt")
    assert os.path.isfile(ckpt_file), f"missing {ckpt_file}"
    label_file = os.path.join(ckpt_dir, "label_map.json")
    assert os.path.isfile(label_file), f"missing {label_file}"
    with open(label_file) as f:
        lm = json.load(f)
    assert lm == {"ang": 0, "hap": 1, "neu": 2, "sad": 3, "sur": 4}

    # 关键断言：训练日志必须显式打印两类样本数（防止 VAD 标签被静默丢失）
    assert "vad_n=" in r.stdout, "训练日志必须打印含 VAD 标签的样本数"
    assert "no_vad_n=" in r.stdout, "训练日志必须打印无 VAD 标签的样本数（ESD 子集）"
