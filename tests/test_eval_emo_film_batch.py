"""eval_emo_film 批处理优化单测（mock 模型，不加载真 emotion2vec/Whisper）。

验证 ADR-0002 的 eval 批量化：emotion2vec funasr batch generate（句+帧）+
Whisper ThreadPoolExecutor 并行；聚合逻辑与逐条等价、顺序对齐、输出 schema 不变。
真实模型 batch-vs-sequential parity 由主循环冒烟验收（不在本任务）。
"""
import os
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import eval_emo_film as ev


# ---------------- fake emotion2vec ----------------

class _FakeEmoModel:
    """按 wav 路径确定性派生 embedding 的 fake emotion2vec。

    同一 wav 无论批量还是逐条调用都返回同一输出（RandomState(seed) 确定性，
    与 PYTHONHASHSEED 无关）。utterance → (dim,) 向量；frame → (T, dim) 矩阵。
    记录 generate 调用形式，供测试断言批量契约。
    """

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []

    def _key(self, wav):
        # 按 basename 派生：run_evaluation 中 ref/hyp 同名 → 同 embedding（验证批量
        # 提取一致性）；不同 utt 名仍不同。逐条 vs 批量同一 wav 也给同一输出。
        return os.path.basename(wav)

    def _utt_vec(self, wav):
        seed = sum(ord(c) for c in self._key(wav)) % 10000
        rng = np.random.RandomState(seed)
        return rng.randn(self.dim).astype(np.float32)

    def _frame_mat(self, wav):
        seed = sum(ord(c) for c in self._key(wav)) % 10000
        rng = np.random.RandomState(seed)
        t = (seed % 4) + 3  # T ∈ [3,6]
        return rng.randn(t, self.dim).astype(np.float32)

    def generate(self, inp, **kw):
        if isinstance(inp, str):
            inp_list = [inp]
        else:
            inp_list = list(inp)
        self.calls.append({
            "granularity": kw.get("granularity"),
            "extract_embedding": kw.get("extract_embedding"),
            "batch_size": kw.get("batch_size"),
            "n": len(inp_list),
        })
        if kw.get("granularity") == "utterance":
            return [{"feats": self._utt_vec(w)} for w in inp_list]
        return [{"feats": self._frame_mat(w)} for w in inp_list]


# ---------------- fake whisper ----------------

class _FakeWhisper:
    """transcribe(wav) 返回 basename 文本；可选 sleep 模拟耗时。"""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.transcribe_calls = []

    def transcribe(self, wav_path):
        if self.delay:
            time.sleep(self.delay)
        self.transcribe_calls.append(wav_path)
        return {"text": os.path.basename(wav_path)}


# ============================================================
# 1. emotion2vec 句级 embedding：批量 == 逐条
# ============================================================

def test_batch_emotion2vec_utt_aggregation():
    """N 个 wav 批量提取 utterance embedding == 逐条提取（同一 fake）。"""
    model = _FakeEmoModel(dim=16)
    wavs = [f"utt{i}.wav" for i in range(5)]

    embs_batch = ev.extract_utt_embeddings(model, wavs, batch_size=4)
    assert len(embs_batch) == len(wavs)

    # 逐条（同一 fake）
    embs_seq = [ev.extract_utt_embeddings(model, [w], batch_size=1)[0] for w in wavs]

    for b, s in zip(embs_batch, embs_seq):
        assert b.shape == (16,)
        assert np.allclose(b, s, atol=1e-6)

    # 批量契约：funasr batch — 一次 generate（列表输入）、透传 batch_size/granularity/extract_embedding
    batch_calls = [c for c in model.calls if c["n"] > 1]
    assert len(batch_calls) == 1
    assert batch_calls[0]["granularity"] == "utterance"
    assert batch_calls[0]["extract_embedding"] is True
    assert batch_calls[0]["batch_size"] == 4


# ============================================================
# 2. emotion2vec 帧级特征：批量 == 逐条
# ============================================================

def test_batch_emotion2vec_frame_aggregation():
    """N 个 wav 批量提取 frame 特征 == 逐条提取。"""
    model = _FakeEmoModel(dim=8)
    wavs = [f"clip{i}.wav" for i in range(4)]

    frames_batch = ev.extract_frame_embeddings(model, wavs, batch_size=2)
    assert len(frames_batch) == len(wavs)

    frames_seq = [ev.extract_frame_embeddings(model, [w], batch_size=1)[0] for w in wavs]

    for b, s in zip(frames_batch, frames_seq):
        assert b.ndim == 2 and b.shape[1] == 8
        assert np.allclose(b, s, atol=1e-6)

    batch_calls = [c for c in model.calls if c["n"] > 1]
    assert len(batch_calls) == 1
    assert batch_calls[0]["granularity"] == "frame"
    assert batch_calls[0]["extract_embedding"] is True
    assert batch_calls[0]["batch_size"] == 2


# ============================================================
# 3. Whisper ThreadPoolExecutor 并行转写
# ============================================================

def test_batch_whisper_threadpool():
    """batch_size 个 wav 经 ThreadPoolExecutor：每条正确收集 + 顺序对齐输入。

    注：whisper 模型实例线程不安全（decoder 共享 kv_cache），transcribe_parallel
    内部加锁串行化模型调用，故不验证并行加速——只验完整收集 + 顺序对齐（任务要求）。
    delay 模拟单条 transcribe 耗时。
    """
    whisper = _FakeWhisper(delay=0.05)
    wavs = [f"/d/sample_{i}.wav" for i in range(4)]

    texts = ev.transcribe_parallel(whisper, wavs, max_workers=4)

    # 完整收集 + 顺序对齐（compute_wer 会 strip+lower，basename 无空格无标点）
    assert len(texts) == len(wavs)
    for w, t in zip(wavs, texts):
        assert t == os.path.basename(w).lower()
    assert set(whisper.transcribe_calls) == set(wavs)


def test_batch_whisper_threadpool_empty():
    """空输入：不调 transcribe、返回空列表。"""
    whisper = _FakeWhisper()
    assert ev.transcribe_parallel(whisper, [], max_workers=4) == []
    assert whisper.transcribe_calls == []


# ============================================================
# 4. --batch_size CLI 透传
# ============================================================

def test_batch_size_cli():
    """--batch_size 8 透传到 args；默认 16；--expected_count 透传。"""
    parser = ev.build_arg_parser()

    common = ["--ref_dir=/r", "--hyp_dir=/h", "--output=/o/out.json"]
    args8 = parser.parse_args(common + ["--batch_size", "8"])
    assert args8.batch_size == 8

    args_default = parser.parse_args(common)
    assert args_default.batch_size == 16

    # v2 新增 --expected_count（默认 None，正式运行必须显式提供）
    assert args_default.expected_count is None
    args_cnt = parser.parse_args(common + ["--expected_count", "10"])
    assert args_cnt.expected_count == 10

    # 兼容保留字段仍可解析（正式口径不再可选，但 CLI 参数保留向后兼容）
    assert args_default.ref_text_manifest is None
    assert args_default.device == "cuda"


# ============================================================
# 5. run_evaluation v2 schema（mock 模型）
# ============================================================

# v2 九字段 schema
_V2_KEYS = {
    "metric_contract_version", "emo_sim", "dtw", "dtw_normalized",
    "dtw_euclidean", "dtw_euclidean_normalized", "wer", "n_samples",
    "wer_percent",
}


def test_metrics_structure_v2_schema(tmp_path):
    """小集合（mock 模型）batch 模式输出 v2 九字段 schema，数值由 mock 决定。

    wer_fn 注入避免依赖环境缺失的 jiwer；ref_text == hyp_text → wer=0。
    """
    ref_dir = tmp_path / "ref"
    hyp_dir = tmp_path / "hyp"
    ref_dir.mkdir()
    hyp_dir.mkdir()
    for name in ["s1.wav", "s2.wav", "s3.wav"]:
        (ref_dir / name).write_bytes(b"")  # 空文件，fake 模型按名派生不读音频
        (hyp_dir / name).write_bytes(b"")

    emo = _FakeEmoModel(dim=8)
    whisper = _FakeWhisper()

    # manifest 覆盖 s1/s2（gt 路径），s3 缺失（回退转写 ref）
    text_map = {"s1": "hello", "s2": "world"}
    # fake whisper 返回 basename，normalize → "s1.wav" 等；ref gt="hello"≠hyp"wav"
    wer_fn = lambda r, h: 0.0 if r == h else 1.0

    result = ev.run_evaluation(
        emo, whisper, str(ref_dir), str(hyp_dir), text_map,
        batch_size=8, wer_fn=wer_fn, expected_count=3,
    )

    assert set(result.keys()) == _V2_KEYS
    assert result["metric_contract_version"] == "emofilm-eval-v2"
    assert result["n_samples"] == 3
    assert 0.0 <= result["emo_sim"] <= 100.0 + 1e-2
    assert result["dtw"] >= 0.0
    assert result["dtw_normalized"] >= 0.0
    assert result["dtw_euclidean"] >= 0.0
    assert result["dtw_euclidean_normalized"] >= 0.0
    assert 0.0 <= result["wer"] <= 1.0
    assert result["wer_percent"] == pytest.approx(result["wer"] * 100.0)

    # emo_sim 数值由 mock 决定：同 wav ref/hyp 同名 → 批量提取同名 frame 特征相同
    # → 均值池化 L2 后自相似 → sim=100
    assert abs(result["emo_sim"] - 100.0) < 1e-2


def test_metrics_structure_all_manifest_present(tmp_path):
    """全部 manifest 命中：Whisper 仅转写 hyp（不回退转写 ref），v2 schema 正确。"""
    ref_dir = tmp_path / "ref"
    hyp_dir = tmp_path / "hyp"
    ref_dir.mkdir()
    hyp_dir.mkdir()
    for name in ["a.wav", "b.wav"]:
        (ref_dir / name).write_bytes(b"")
        (hyp_dir / name).write_bytes(b"")

    emo = _FakeEmoModel(dim=4)
    whisper = _FakeWhisper()
    text_map = {"a": "x", "b": "y"}

    # gt != hyp → wer=1；v2 schema + n_samples 校验（wer 值无约束）
    result = ev.run_evaluation(
        emo, whisper, str(ref_dir), str(hyp_dir), text_map,
        batch_size=2, wer_fn=lambda r, h: 1.0, expected_count=2,
    )
    assert set(result.keys()) == _V2_KEYS
    assert result["n_samples"] == 2
    assert result["wer"] == 1.0
    assert result["wer_percent"] == 100.0
    # v2: dtw 是 cosine 口径，dtw_euclidean 是诊断口径（独立输出）
    assert result["dtw"] >= 0.0
    assert result["dtw_euclidean"] >= 0.0


def test_run_evaluation_rejects_id_mismatch(tmp_path):
    """v2: ref/hyp ID 不一致 → hard-fail（替代 v1 静默排序回退）。"""
    ref_dir = tmp_path / "ref"
    hyp_dir = tmp_path / "hyp"
    ref_dir.mkdir()
    hyp_dir.mkdir()
    (ref_dir / "a.wav").write_bytes(b"")
    (hyp_dir / "b.wav").write_bytes(b"")

    emo = _FakeEmoModel()
    whisper = _FakeWhisper()
    with pytest.raises(ValueError, match="wav ID mismatch"):
        ev.run_evaluation(
            emo, whisper, str(ref_dir), str(hyp_dir), {},
            batch_size=2, wer_fn=lambda r, h: 0.0, expected_count=1,
        )


def test_run_evaluation_rejects_wrong_count(tmp_path):
    """v2: expected_count 不符 → hard-fail。"""
    ref_dir = tmp_path / "ref"
    hyp_dir = tmp_path / "hyp"
    ref_dir.mkdir()
    hyp_dir.mkdir()
    for d in (ref_dir, hyp_dir):
        (d / "a.wav").write_bytes(b"")

    emo = _FakeEmoModel()
    whisper = _FakeWhisper()
    with pytest.raises(ValueError, match="expected 2"):
        ev.run_evaluation(
            emo, whisper, str(ref_dir), str(hyp_dir), {},
            batch_size=2, wer_fn=lambda r, h: 0.0, expected_count=2,
        )


def test_run_evaluation_propagates_sample_error(tmp_path):
    """v2: 逐条异常不再被吞掉返回部分均值；必须立即抛出携带 utt_id。

    构造 wer_fn 对特定文本抛异常 → run_evaluation 的逐样本 try 必须把它包成
    RuntimeError 并附 utt_id 向上抛（替代 v1 的 print WARN + 跳过）。
    frame 提取本身正常，确保失败发生在逐样本循环内，验证 utt_id wrapping。
    """
    ref_dir = tmp_path / "ref"
    hyp_dir = tmp_path / "hyp"
    ref_dir.mkdir()
    hyp_dir.mkdir()
    for d in (ref_dir, hyp_dir):
        (d / "ok.wav").write_bytes(b"")
        (d / "boom.wav").write_bytes(b"")

    emo = _FakeEmoModel(dim=4)
    whisper = _FakeWhisper()

    def wer_fn_boom(ref_text, hyp_text):
        if "boom" in ref_text or "boom" in hyp_text:
            raise ValueError("synthetic wer failure")
        return 0.0

    # text_map 覆盖 boom，使 ref_text=normalize("boom")="boom" → 触发 wer_fn 抛错
    text_map = {"ok": "ok", "boom": "boom"}
    with pytest.raises(RuntimeError, match="'boom'"):
        ev.run_evaluation(
            emo, whisper, str(ref_dir), str(hyp_dir), text_map,
            batch_size=2, wer_fn=wer_fn_boom, expected_count=2,
        )
