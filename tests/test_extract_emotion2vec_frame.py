"""emotion2vec frame feats 批量提取单测。"""
import json
import os
import subprocess

import torch

EMOFILM_PY = "/home/hanlvyuan/miniconda3/envs/emofilm/bin/python"
ROOT = "/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM"
SCRIPT = f"{ROOT}/tools/extract_emotion2vec_frame.py"
TEST_MANIFEST = "/tmp/test_emo_manifest.jsonl"
TEST_FEAT_DIR = "/tmp/test_emo_features"


def setup_module():
    """用 ESD speaker 0011 的 2 条 wav 生成最小 manifest。"""
    esd = "/home/hanlvyuan/LLM-Audio/datasets/ESD/0011/Angry"
    wavs = sorted(os.listdir(esd))[:2]
    with open(TEST_MANIFEST, "w", encoding="utf-8") as f:
        for w in wavs:
            row = {
                "utt_id": w.replace(".wav", ""),
                "wav_path": os.path.join(esd, w),
                "text": "test",
                "sentence_emotion": "ang",
                "speaker_id": "0011",
            }
            f.write(json.dumps(row) + "\n")
    assert len(wavs) == 2


def test_extract_output_shape():
    """产出 .pt 的 feats 维度正确 (T,1024)。"""
    cmd = [
        EMOFILM_PY, SCRIPT,
        f"--manifest={TEST_MANIFEST}",
        f"--output_dir={TEST_FEAT_DIR}",
        "--device=cpu",
    ]
    subprocess.run(cmd, check=True)

    for wav_name in sorted(os.listdir("/home/hanlvyuan/LLM-Audio/datasets/ESD/0011/Angry"))[:2]:
        utt_id = wav_name.replace(".wav", "")
        pt_file = os.path.join(TEST_FEAT_DIR, f"{utt_id}.pt")
        assert os.path.isfile(pt_file), f"missing {pt_file}"
        data = torch.load(pt_file, map_location="cpu")
        assert "feats" in data, f"missing 'feats' in {pt_file}"
        feats = data["feats"]
        assert feats.ndim == 2, f"feats ndim={feats.ndim}"
        assert feats.shape[1] == 1024, f"feats dim={feats.shape[1]}"
        assert feats.shape[0] > 0, f"zero frames"
        assert "frame_rate_hz" in data
        assert 40 <= data["frame_rate_hz"] <= 60


def test_skip_existing():
    """已有 .pt 的 utterance 跳过。"""
    # 第二次跑应直接 skip 并打印 skip 信息
    result = subprocess.run([
        EMOFILM_PY, SCRIPT,
        f"--manifest={TEST_MANIFEST}",
        f"--output_dir={TEST_FEAT_DIR}",
        "--device=cpu",
    ], capture_output=True, text=True)
    assert "skip" in (result.stdout + result.stderr).lower() or \
           "already" in (result.stdout + result.stderr).lower()
