"""emotion_metrics 纯函数核心测试（v3 评测契约）。"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from emotion_metrics import (
    build_emotion_ref_index,
    compute_dtw_normalized,
    compute_frame_mean_emo_sim,
    compute_per_emotion_mean_sim,
    normalize_text,
)


def test_frame_mean_emo_sim_identity_is_100():
    feats = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    assert compute_frame_mean_emo_sim(feats, feats) == pytest.approx(100.0, abs=1e-6)


def test_frame_mean_emo_sim_orthogonal_is_0():
    ref = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    hyp = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    assert compute_frame_mean_emo_sim(ref, hyp) == pytest.approx(0.0, abs=1e-6)


def test_dtw_normalized_identity_is_zero():
    feats = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert compute_dtw_normalized(feats, feats) == pytest.approx(0.0, abs=1e-9)


def test_dtw_normalized_orthogonal_cosine_is_one():
    ref = np.array([[1.0, 0.0]], dtype=np.float32)
    hyp = np.array([[0.0, 1.0]], dtype=np.float32)
    assert compute_dtw_normalized(ref, hyp) == pytest.approx(1.0, abs=1e-9)


def test_normalize_text_rules():
    assert normalize_text("Hello, World! 42") == "hello world forty two"


def test_build_emotion_ref_index(tmp_path):
    p = tmp_path / "src.jsonl"
    p.write_text(
        json.dumps({"utt_id": "a1", "speaker_id": "s1", "text": "Hi",
                    "sentence_emotion": "ang", "wav_path": "/x/a1.wav"}) + "\n" +
        json.dumps({"utt_id": "a2", "speaker_id": "s1", "text": "hi",
                    "sentence_emotion": "hap", "wav_path": "/x/a2.wav"}) + "\n",
        encoding="utf-8",
    )
    idx = build_emotion_ref_index(str(p))
    assert idx[("s1", "hi")] == {"ang": "/x/a1.wav", "hap": "/x/a2.wav"}


def test_per_emotion_mean_sim_groups_by_emotion():
    rows = [
        {"emotion": "ang", "emo_sim": 10.0},
        {"emotion": "ang", "emo_sim": 20.0},
        {"emotion": "hap", "emo_sim": 90.0},
    ]
    out = compute_per_emotion_mean_sim(rows)
    assert out == {"ang": 15.0, "hap": 90.0}
