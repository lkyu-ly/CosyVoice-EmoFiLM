"""per-word frame blocks 构建工具单测。"""
import json
import os
import shutil
import subprocess
from pathlib import Path
import torch

EMOFILM_PY = "/home/hanlvyuan/miniconda3/envs/emofilm/bin/python"
ROOT = "/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM"
SCRIPT = f"{ROOT}/tools/build_word_emo_dataset.py"

# 手动构造 TextGrid + .pt feats 测试数据
TMP = "/tmp/test_word_blocks"


def setup_module():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)


def _write_textgrid(utt_id, word_intervals):
    """手工构造最小 TextGrid（long format）。"""
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
             "xmin = 0", "xmax = 2.0", "tiers? <exists>", "size = 1", 'item []:',
             '    item [1]:',
             '        class = "IntervalTier"',
             '        name = "words"',
             '        xmin = 0',
             '        xmax = 2.0',
             f'        intervals: size = {len(word_intervals)}']
    for i, (word, start, end) in enumerate(word_intervals):
        lines += [f'            intervals [{i+1}]:',
                  f'                xmin = {start}',
                  f'                xmax = {end}',
                  f'                text = "{word}"']
    tg = os.path.join(TMP, f"{utt_id}.TextGrid")
    with open(tg, "w") as f:
        f.write("\n".join(lines))
    return tg


def _write_feats(utt_id, n_frames, dim=1024):
    """生成伪 feats。"""
    feats = {
        "feats": torch.randn(n_frames, dim),
        "frame_rate_hz": 50.0,
    }
    feat_path = os.path.join(TMP, f"{utt_id}.pt")
    torch.save(feats, feat_path)
    return feat_path


def _write_manifest(utt_id, wav_path):
    """生成最小 manifest。"""
    p = os.path.join(TMP, "test_manifest.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({
            "utt_id": utt_id,
            "wav_path": wav_path,
            "text": "hello world test",
            "sentence_emotion": "neu",
            "speaker_id": "0011",
        }) + "\n")
    return p


def test_build_word_blocks():
    utt_id = "test001"
    # 3 words, 每个 ~0.5s → 25 frames
    _write_textgrid(utt_id, [("hello", 0.0, 0.5), ("world", 0.6, 1.1), ("test", 1.2, 1.9)])
    _write_feats(utt_id, 100)
    wav_path = os.path.join(TMP, "test001.wav")
    Path(wav_path).touch()
    manifest = _write_manifest(utt_id, wav_path)

    out_dir = os.path.join(TMP, "word_blocks")
    cmd = [
        EMOFILM_PY, SCRIPT,
        f"--manifest={manifest}",
        f"--features_dir={TMP}",
        f"--textgrid_dir={TMP}",
        f"--output_dir={out_dir}",
    ]
    subprocess.run(cmd, check=True)

    utt_dir = os.path.join(out_dir, utt_id)
    assert os.path.isdir(utt_dir)
    word_files = sorted(os.listdir(utt_dir))
    assert len(word_files) == 3, f"expected 3 word files, got {len(word_files)}"
    for wf in word_files:
        data = torch.load(os.path.join(utt_dir, wf), map_location="cpu")
        assert "frames" in data
        assert "word" in data
        assert "padding_mask" in data
        assert data["frames"].ndim == 2
        assert data["frames"].shape[1] == 1024


def test_empty_word_filtered():
    """区间为空的 word 被过滤。"""
    utt_id = "test002"
    # word "x" has xmin=xmax → no frames
    _write_textgrid(utt_id, [("hello", 0.0, 0.5), ("x", 0.6, 0.6), ("world", 0.61, 1.0)])
    _write_feats(utt_id, 60)
    wav_path = os.path.join(TMP, "test002.wav")
    Path(wav_path).touch()
    manifest = _write_manifest(utt_id, wav_path)

    out_dir = os.path.join(TMP, "word_blocks2")
    subprocess.run([
        EMOFILM_PY, SCRIPT,
        f"--manifest={manifest}",
        f"--features_dir={TMP}",
        f"--textgrid_dir={TMP}",
        f"--output_dir={out_dir}",
    ], check=True)

    word_files = sorted(os.listdir(os.path.join(out_dir, utt_id)))
    assert len(word_files) == 2  # x filtered
