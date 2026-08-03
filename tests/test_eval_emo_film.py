"""eval_emo_film v3 CLI/聚合测试（mock 模型，不加载真模型）。

只保留核心契约：严格配对、v3 schema、判别指标透传、mock 端到端数值一致性。
旧 v2 九字段锁定与批处理细节断言已删除（v3 不向前兼容）。
"""
import os
import sys
import json

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import eval_emo_film as ev
import emotion_metrics as em


def _make_wavs(tmp_path, names):
    ref = tmp_path / "ref"
    hyp = tmp_path / "hyp"
    ref.mkdir()
    hyp.mkdir()
    for n in names:
        (ref / n).write_bytes(b"")
        (hyp / n).write_bytes(b"")
    return str(ref), str(hyp)


class _FakeEmoModel:
    """确定性 fake：utterance 向量与 frame 矩阵都按 basename 派生。"""

    def __init__(self, dim=8):
        self.dim = dim

    def _vec(self, key):
        rng = np.random.RandomState(sum(ord(c) for c in key) % 10000)
        return rng.randn(self.dim).astype(np.float32)

    def generate(self, inp, **kw):
        inp_list = [inp] if isinstance(inp, str) else list(inp)
        if kw.get("granularity") == "utterance":
            return [{"feats": self._vec(os.path.basename(w))} for w in inp_list]
        return [{"feats": self._vec(os.path.basename(w))[None, :]} for w in inp_list]


class _FakeWhisper:
    def transcribe(self, wav_path):
        return {"text": os.path.basename(wav_path)}


def test_pair_wavs_rejects_mismatch(tmp_path):
    ref, hyp = _make_wavs(tmp_path, ["a.wav"])
    (tmp_path / "hyp" / "b.wav").write_bytes(b"")
    with pytest.raises(ValueError, match="wav ID mismatch"):
        ev.pair_wavs_strict(ref, hyp, expected_count=1)


def test_pair_wavs_strict_ok(tmp_path):
    ref, hyp = _make_wavs(tmp_path, ["b.wav", "a.wav"])
    pairs = ev.pair_wavs_strict(ref, hyp, expected_count=2)
    assert [p[0] for p in pairs] == ["a", "b"]


def test_aggregate_v3_schema():
    rows = [{
        "utt_id": "u1", "emotion": "ang", "emo_sim": 50.0,
        "dtw_normalized": 0.5, "wer": 0.1,
    }]
    out = ev.aggregate_metric_rows(rows)
    assert set(out.keys()) == {
        "metric_contract_version", "n_samples", "emo_sim", "dtw_normalized",
        "wer", "wer_percent", "per_emotion_emo_sim",
    }
    assert out["metric_contract_version"] == "emofilm-eval-v3"
    assert out["per_emotion_emo_sim"] == {"ang": 50.0}


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        ev.aggregate_metric_rows([])


def test_run_evaluation_end_to_end_v3(tmp_path):
    """mock 端到端：同 wav → emo_sim=100；per_emotion 无 manifest 情感时为 {}。"""
    ref, hyp = _make_wavs(tmp_path, ["a.wav", "b.wav"])
    emo = _FakeEmoModel(dim=8)
    whisper = _FakeWhisper()
    text_map = {"a": {"text": "x"}, "b": {"text": "y"}}
    result = ev.run_evaluation(
        emo, whisper, ref, hyp, text_map,
        batch_size=2, wer_fn=lambda r, h: 0.0, expected_count=2,
    )
    assert result["n_samples"] == 2
    assert abs(result["emo_sim"] - 100.0) < 1e-2
    assert result["per_emotion_emo_sim"] == {}


def test_run_evaluation_with_emotion_ref_manifest(tmp_path):
    """提供情感参考索引时输出 discriminability（结构存在且 n_valid>0）。"""
    ref, hyp = _make_wavs(tmp_path, ["a.wav"])
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"utt_id": "a", "speaker_id": "s1", "text": "hi",
                    "sentence_emotion": "ang", "wav_path": "/x/a.wav"}) + "\n" +
        json.dumps({"utt_id": "a2", "speaker_id": "s1", "text": "hi",
                    "sentence_emotion": "hap", "wav_path": "/x/a2.wav"}) + "\n" +
        json.dumps({"utt_id": "a3", "speaker_id": "s1", "text": "hi",
                    "sentence_emotion": "sad", "wav_path": "/x/a3.wav"}) + "\n",
        encoding="utf-8",
    )
    eval_manifest = tmp_path / "eval.jsonl"
    eval_manifest.write_text(
        json.dumps({"utt_id": "a", "speaker_id": "s1", "text": "hi",
                    "sentence_emotion": "ang"}) + "\n",
        encoding="utf-8",
    )
    emo = _FakeEmoModel(dim=8)
    whisper = _FakeWhisper()
    text_map = {"a": {"text": "hi", "speaker_id": "s1", "emotion": "ang"}}
    result = ev.run_evaluation(
        emo, whisper, ref, hyp, text_map,
        batch_size=1, wer_fn=lambda r, h: 0.0, expected_count=1,
        emotion_ref_index=em.build_emotion_ref_index(str(src)),
        eval_rows=[json.loads(l) for l in open(eval_manifest)],
    )
    assert result["discriminability"]["n_valid"] == 1
    assert 0.0 <= result["discriminability"]["nearest_ref_acc_pct"] <= 100.0


def test_batch_size_cli():
    parser = ev.build_arg_parser()
    # v3：--ref_text_manifest 为必填项，基线 argv 必须包含
    common = ["--ref_dir=/r", "--hyp_dir=/h", "--output=/o/out.json",
              "--ref_text_manifest=/m.jsonl"]
    assert parser.parse_args(common + ["--batch_size", "8"]).batch_size == 8
    assert parser.parse_args(common).batch_size == 16
    assert parser.parse_args(common + ["--expected_count", "10"]).expected_count == 10
    assert parser.parse_args(common).emotion_ref_manifest is None
