"""FEDD-rebuilt smoke 模式单测。"""
import json
import os
import shutil
import subprocess

EMOFILM_PY = "/home/hanlvyuan/miniconda3/envs/emofilm/bin/python"
ROOT = "/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM"
SCRIPT = f"{ROOT}/tools/build_fedd.py"
OUTDIR = "/tmp/test_fedd_smoke"


def setup_module():
    if os.path.isdir(OUTDIR):
        shutil.rmtree(OUTDIR)


def test_fedd_smoke_mode_runs():
    """smoke 模式成功产生 manifest + wav。"""
    cmd = [
        EMOFILM_PY, SCRIPT,
        f"--output_dir={OUTDIR}",
        f"--esd_dir=/home/hanlvyuan/LLM-Audio/datasets/ESD",
        "--mode=smoke",
        "--num=10",
        "--seed=42",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr

    manifest_path = os.path.join(OUTDIR, "manifest.jsonl")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as f:
        lines = f.readlines()
    assert len(lines) >= 4  # 至少几条测试

    wav_dir = os.path.join(OUTDIR, "wav")
    assert os.path.isdir(wav_dir)
    wav_count = len([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    assert wav_count >= len(lines) * 0.5


def test_fedd_manifest_schema():
    """manifest 每行 schema 正确，标注 FEDD-rebuilt。"""
    manifest_path = os.path.join(OUTDIR, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        return
    with open(manifest_path) as f:
        for line in f:
            row = json.loads(line)
            assert "utt_id" in row
            assert "wav_path" in row
            assert "text" in row
            assert "emotion_transition" in row
            assert "source" in row
            assert "part" in row
            assert row["source"] != "original_FEDD"  # 不允许冒充原始数据


def test_fedd_wav_duration():
    """FEDD wav 时长 ≥ 1s ≤ 30s。"""
    import torchaudio
    wav_dir = os.path.join(OUTDIR, "wav")
    if not os.path.isdir(wav_dir):
        return
    for wav_name in sorted(os.listdir(wav_dir))[:4]:
        p = os.path.join(wav_dir, wav_name)
        info = torchaudio.info(p)
        dur = info.num_frames / info.sample_rate
        assert 1.0 <= dur <= 30.0, f"{wav_name} dur={dur:.1f}s out of range [1,30]"
