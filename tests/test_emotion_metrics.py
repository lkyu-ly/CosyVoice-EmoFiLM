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
    compute_discriminability,
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


class _MapEmoModel:
    """按路径映射返回 utterance/frame 向量的 fake（判别测试用，不读音频）。"""

    def __init__(self, vec_by_path):
        self.vec_by_path = vec_by_path

    def generate(self, inp, **kw):
        lst = [inp] if isinstance(inp, str) else list(inp)
        if kw.get("granularity") == "utterance":
            return [{"feats": self.vec_by_path[p]} for p in lst]
        return [{"feats": self.vec_by_path[p][None, :]} for p in lst]


def test_discriminability_merges_reference_wav():
    """目标情感参考从 eval_rows.reference_wav 合并进候选，判别可算且方向正确。"""
    import numpy as np
    e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    e3 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    model = _MapEmoModel({
        "/refs/a_ang.wav": e0,  # 目标情感参考（reference_wav）
        "/refs/a_hap.wav": e1,
        "/refs/a_sad.wav": e2,
        "/refs/a_sur.wav": e3,
        "/hyp/a.wav": e0,       # 生成音频与目标参考同向 → 判别应选 ang
    })
    ref_index = {("s1", "hi"): {
        "hap": "/refs/a_hap.wav",
        "sad": "/refs/a_sad.wav",
        "sur": "/refs/a_sur.wav",
    }}
    eval_rows = [{
        "utt_id": "a", "speaker_id": "s1", "text": "hi",
        "emotion": "ang", "reference_wav": "/refs/a_ang.wav",
    }]
    out = compute_discriminability(["/hyp/a.wav"], eval_rows, ref_index, model)
    assert out["n_valid"] == 1
    assert out["n_scored"] == 1
    assert out["nearest_ref_acc_pct"] == 100.0
    # same/cross/gap 用原始余弦（与 prompt_match 既有四模型口径一致：
    # same≈0.85、gap≈0.38）；仅 nearest_ref_acc_pct 是百分比。勿与 emo_sim ×100 混淆。
    assert out["same_emotion_mean"] == pytest.approx(1.0, abs=1e-6)
    assert out["cross_emotion_mean"] == pytest.approx(0.0, abs=1e-6)
    assert out["gap_same_minus_cross"] == pytest.approx(1.0, abs=1e-6)
    assert out["n_way_distribution"] == {"4": 1}


def test_discriminability_skips_rows_without_enough_refs():
    """参考不足（含 target 后仍 <3）→ 进 n_skipped，不写 NaN、不误导。"""
    import numpy as np
    model = _MapEmoModel({
        "/refs/a_hap.wav": np.array([0.0, 1.0], dtype=np.float32),
        "/hyp/a.wav": np.array([1.0, 0.0], dtype=np.float32),
    })
    ref_index = {("s1", "hi"): {"hap": "/refs/a_hap.wav"}}
    eval_rows = [{
        "utt_id": "a", "speaker_id": "s1", "text": "hi",
        "emotion": "ang",  # 无 reference_wav → 合并后仅 1 个候选
    }]
    out = compute_discriminability(["/hyp/a.wav"], eval_rows, ref_index, model)
    assert out["n_valid"] == 0
    assert out["n_skipped"] == 1
    assert "reason" in out
