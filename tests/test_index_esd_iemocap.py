"""数据索引脚本单测：验证 manifest 格式与统计量。"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

EMOFILM_PY = sys.executable
ROOT_PATH = Path(__file__).resolve().parents[1]
ROOT = str(ROOT_PATH)
INDEX_SCRIPT = f"{ROOT}/tools/index_esd_iemocap.py"
ESD_DIR = str(ROOT_PATH / "datasets" / "ESD")
IEMOCAP_DIR = str(ROOT_PATH / "datasets" / "IEMOCAP")


def test_esd_manifest_format():
    """ESD manifest 每行符合 schema, speaker 范围正确, emotion 值有效。"""
    esd_path = "/tmp/test_esd_manifest.jsonl"
    cmd = [
        EMOFILM_PY, INDEX_SCRIPT,
        "--dataset", "esd",
        f"--data_dir={ESD_DIR}",
        f"--output={esd_path}",
        "--test_per_speaker=30",
        "--english_only"
    ]
    subprocess.run(cmd, check=True)
    assert os.path.isfile(esd_path)
    with open(esd_path) as f:
        lines = f.readlines()
    assert len(lines) > 1000
    speakers = set()
    emotions = set()
    for line in lines:
        row = json.loads(line)
        assert "utt_id" in row
        assert "wav_path" in row
        assert "text" in row
        assert "sentence_emotion" in row
        assert "speaker_id" in row
        assert os.path.isfile(row["wav_path"]), f"missing {row['wav_path']}"
        speakers.add(row["speaker_id"])
        emotions.add(row["sentence_emotion"])
    assert all(s in speakers for s in [f"{i:04d}" for i in range(11, 21)]), f"missing english speakers: {speakers}"
    assert {"ang", "hap", "neu", "sad", "sur"}.issubset(emotions), f"missing emotions: {emotions}"


def test_iemocap_manifest_format():
    """IEMOCAP manifest 只含 5 类 emotion 值。"""
    iemocap_path = "/tmp/test_iemocap_manifest.jsonl"
    cmd = [
        EMOFILM_PY, INDEX_SCRIPT,
        "--dataset", "iemocap",
        f"--data_dir={IEMOCAP_DIR}",
        f"--output={iemocap_path}"
    ]
    subprocess.run(cmd, check=True)
    assert os.path.isfile(iemocap_path)
    with open(iemocap_path) as f:
        lines = f.readlines()
    assert len(lines) > 1000
    allowed = {"ang", "hap", "neu", "sad", "sur"}
    for line in lines:
        row = json.loads(line)
        assert row["sentence_emotion"] in allowed, f"unexpected emotion: {row['sentence_emotion']}"


def test_esd_train_test_split():
    """ESD train/test 无重叠 utt_id, 每 speaker 150 条 test。"""
    train_path = "/tmp/test_esd_train.jsonl"
    test_path = "/tmp/test_esd_test.jsonl"
    subprocess.run([
        EMOFILM_PY, INDEX_SCRIPT,
        "--dataset", "esd",
        f"--data_dir={ESD_DIR}",
        f"--output={train_path}",
        f"--test_output={test_path}",
        "--test_per_speaker=30",
        "--english_only"
    ], check=True)
    with open(train_path) as f:
        train_ids = {json.loads(l)["utt_id"] for l in f}
    with open(test_path) as f:
        test_ids = {json.loads(l)["utt_id"] for l in f}
    assert len(train_ids & test_ids) == 0, "train/test overlap"
    # 每 speaker 30 条 test × 5 emotions = 150
    for s in [f"{i:04d}" for i in range(11, 21)]:
        spk_test = [t for t in test_ids if t.startswith(s)]
        assert len(spk_test) == 150, f"speaker {s} has {len(spk_test)} test samples (expected 150=30*5)"
