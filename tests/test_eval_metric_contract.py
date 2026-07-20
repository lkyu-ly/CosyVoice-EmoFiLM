import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eval_emo_film as ev


def test_frame_mean_emo_sim_uses_time_mean():
    ref = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    hyp = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    expected = 100 / np.sqrt(2)
    assert ev.compute_frame_mean_emo_sim(ref, hyp) == pytest.approx(expected)


def test_dtw_metrics_identity_is_zero():
    feats = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = ev.compute_dtw_metrics(feats, feats)
    assert result == {
        "dtw": 0.0,
        "dtw_normalized": 0.0,
        "dtw_euclidean": 0.0,
        "dtw_euclidean_normalized": 0.0,
    }


def test_dtw_formal_metric_is_accumulated_cosine():
    ref = np.array([[1.0, 0.0]], dtype=np.float32)
    hyp = np.array([[0.0, 1.0]], dtype=np.float32)
    result = ev.compute_dtw_metrics(ref, hyp)
    assert result["dtw"] == pytest.approx(1.0)
    assert result["dtw_normalized"] == pytest.approx(1.0)
    assert result["dtw_euclidean"] == pytest.approx(np.sqrt(2))


# ============================================================
# Task 2: 严格配对 + v2 聚合 schema
# ============================================================

def test_pair_wavs_rejects_mismatched_ids(tmp_path):
    """ref/hyp ID 集合不一致 → ValueError，禁止静默排序回退。"""
    ref = tmp_path / "ref"
    hyp = tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    (ref / "a.wav").write_bytes(b"")
    (hyp / "b.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="wav ID mismatch"):
        ev.pair_wavs_strict(str(ref), str(hyp), expected_count=1)


def test_pair_wavs_rejects_wrong_count(tmp_path):
    """expected_count 不符 → ValueError。"""
    ref = tmp_path / "ref"
    hyp = tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    for directory in (ref, hyp):
        (directory / "a.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="expected 2"):
        ev.pair_wavs_strict(str(ref), str(hyp), expected_count=2)


def test_pair_wavs_rejects_empty(tmp_path):
    """空目录 → ValueError。"""
    ref = tmp_path / "ref"
    hyp = tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    with pytest.raises(ValueError, match="empty"):
        ev.pair_wavs_strict(str(ref), str(hyp))


def test_pair_wavs_rejects_duplicate_ids(tmp_path):
    """同目录下重复 utt_id（理论上不可能因文件系统覆盖，但 _wav_map 仍显式校验）。"""
    ref = tmp_path / "ref"
    hyp = tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    (ref / "a.wav").write_bytes(b"")
    (hyp / "a.wav").write_bytes(b"")
    pairs = ev.pair_wavs_strict(str(ref), str(hyp), expected_count=1)
    assert len(pairs) == 1
    assert pairs[0][0] == "a"


def test_pair_wavs_strict_ok(tmp_path):
    """正常配对返回 (utt_id, ref_path, hyp_path) 三元组列表，按 utt_id 排序。"""
    ref = tmp_path / "ref"
    hyp = tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    for n in ("b.wav", "a.wav"):
        (ref / n).write_bytes(b"")
        (hyp / n).write_bytes(b"")
    pairs = ev.pair_wavs_strict(str(ref), str(hyp), expected_count=2)
    assert [p[0] for p in pairs] == ["a", "b"]
    assert pairs[0][1].endswith("ref/a.wav")
    assert pairs[0][2].endswith("hyp/a.wav")


def test_result_schema_has_ratio_and_percent_wer():
    """aggregate_metric_rows 产 v2 九字段 schema；wer 是 [0,1] 比例，wer_percent 仅展示。"""
    result = ev.aggregate_metric_rows([{
        "emo_sim": 100.0, "dtw": 0.0, "dtw_normalized": 0.0,
        "dtw_euclidean": 0.0, "dtw_euclidean_normalized": 0.0,
        "wer": 0.0312,
    }])
    assert result["metric_contract_version"] == "emofilm-eval-v2"
    assert result["wer"] == pytest.approx(0.0312)
    assert result["wer_percent"] == pytest.approx(3.12)


def test_aggregate_metric_rows_has_nine_fields():
    """v2 schema 严格 9 字段：metric_contract_version, emo_sim, dtw, dtw_normalized,
    dtw_euclidean, dtw_euclidean_normalized, wer, n_samples, wer_percent。"""
    result = ev.aggregate_metric_rows([{
        "emo_sim": 1.0, "dtw": 2.0, "dtw_normalized": 3.0,
        "dtw_euclidean": 4.0, "dtw_euclidean_normalized": 5.0,
        "wer": 0.0,
    }])
    assert set(result.keys()) == {
        "metric_contract_version", "emo_sim", "dtw", "dtw_normalized",
        "dtw_euclidean", "dtw_euclidean_normalized", "wer", "n_samples",
        "wer_percent",
    }
    assert result["n_samples"] == 1


def test_aggregate_metric_rows_rejects_empty():
    """空 rows → ValueError（禁止返回貌似合法的默认值）。"""
    with pytest.raises(ValueError, match="empty"):
        ev.aggregate_metric_rows([])


# ============================================================
# Task 8: 校准受控扰动 helpers（append_silence / crop_tail / time_stretch）
# ============================================================

def test_append_silence_is_deterministic():
    """append_silence: 尾部追加指定秒数静音，长度和内容确定性。"""
    from calibrate_eval_contract import append_silence
    wav = np.ones(16000, dtype=np.float32)
    out = append_silence(wav, sample_rate=16000, seconds=0.5)
    assert len(out) == 24000
    assert np.all(out[-8000:] == 0)


def test_crop_tail_rejects_empty_output():
    """crop_tail: keep_ratio=0.0 → 空输出 → ValueError。"""
    from calibrate_eval_contract import crop_tail
    with pytest.raises(ValueError):
        crop_tail(np.ones(10), keep_ratio=0.0)


def test_time_stretch_has_requested_length():
    """time_stretch: factor=1.5 → 长度按因子缩放。"""
    from calibrate_eval_contract import time_stretch
    wav = np.arange(100, dtype=np.float32)
    assert len(time_stretch(wav, factor=1.5)) == 150


def test_crop_tail_keep_ratio_one_keeps_all():
    """crop_tail: keep_ratio=1.0 → 原样保留全部样本。"""
    from calibrate_eval_contract import crop_tail
    wav = np.arange(100, dtype=np.float32)
    out = crop_tail(wav, keep_ratio=1.0)
    assert len(out) == 100


def test_time_stretch_factor_one_preserves_length():
    """time_stretch: factor=1.0 → 长度不变（恒等）。"""
    from calibrate_eval_contract import time_stretch
    wav = np.arange(100, dtype=np.float32)
    assert len(time_stretch(wav, factor=1.0)) == 100
