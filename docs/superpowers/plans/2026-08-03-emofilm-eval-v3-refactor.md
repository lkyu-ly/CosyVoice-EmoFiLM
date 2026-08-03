# EmoFiLM 评测侧 v3 重构 + 情感匹配 prompt 验证 + CosyVoice3 基线 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把评测侧重构为干净可扩展的 v3（Emo-SIM/WER/DTW-normalized + per-emotion + n-way 判别），同步跑通"情感匹配 prompt 四模型验证"与"CosyVoice3 官方基线"两个实验。

**Architecture:** 纯指标函数与 CLI 分离（`eval/emotion_metrics.py` + 薄 `eval/eval_emo_film.py`）；评测输出全新 `emofilm-eval-v3` schema（不向前兼容，旧 schema 与旧产物保持历史冻结）；验证实验复用现有推理脚本（manifest 自带 prompt_wav，零推理代码改动）；CV3 基线下载官方权重后先冒烟确认推理入口再小规模生成。

**Tech Stack:** Python 3.10 / torch / funasr（emotion2vec_plus_large）/ whisper large-v3 / jiwer / fastdtw / pytest / ModelScope。

**执行约定（用户约束，替代 writing-plans 模板的 commit 步骤）：**
- 不主动 git commit；工作树即交付物。每个 Task 的收尾步骤改为"跑验证命令确认通过"。
- 敏捷 + 精简：旧测试与旧合同直接删除/重写，不为兼容留双轨；测试只覆盖核心正确性与流程接入，不追求全覆盖。
- 所有新代码/注释使用中文。
- 环境：`/home/hanlvyuan/miniconda3/envs/emofilm/bin/python`，`PYTHONPATH=.:third_party/Matcha-TTS`。

**决策确认（2026-08-03 用户逐项认可推荐项）：**
1. 全新 `emofilm-eval-v3` schema，不向前兼容。
2. DTW 精简为 `dtw_normalized` 一个字段。
3. 旧三个评测测试整体删除，重写精简核心测试。
4. 删除过时合同 `docs/contracts/emofilm_v2_evaluators.md`。
5. 三个 run_eval.sh 更新为新 CLI，v3 输出到 `eval/v3/`，历史 json 不覆盖。
6. 情感匹配 prompt 验证：5 情感 × 12 句完整组 = 60 条 × 4 模型。
7. CV3 基线：ESD 150 条（5 情感 × 30 句）。
8. CV3 权重 ModelScope 下载；推理先试本仓库 CLI（inference_instruct2），不兼容再 clone 官方仓库脚本。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `eval/emotion_metrics.py`（新建） | 全部纯指标函数：特征提取、Emo-SIM、DTW、WER normalize、参考索引、n-way 判别、per-emotion 聚合 |
| `eval/eval_emo_film.py`（重写） | 薄 CLI + run_evaluation 流程编排，输出 v3 schema |
| `tests/test_emotion_metrics.py`（新建） | 核心 TDD：纯函数正确性（手工样例） |
| `tests/test_eval_emo_film.py`（新建） | CLI/配对/聚合/mock 端到端（替换旧 3 个测试文件） |
| `tests/test_eval_smoke.py`（修改） | 更新 v3 schema 断言 |
| `tools/calibrate_eval_contract.py`（修改） | 适配新函数名与新字段 |
| `tools/build_emotion_prompt_manifest.py`（新建） | 生成情感匹配 prompt 验证 manifest |
| `exp/*/run_eval.sh`（修改 × 3） | 新 CLI + v3 输出到 `eval/v3/`，不覆盖历史 json |
| `docs/contracts/emofilm_v3_eval.md`（新建） | v3 CLI/schema 合同 |
| `docs/contracts/emofilm_v2_evaluators.md`（删除） | 过时文档（引用不存在的 `acoustic_evaluators.py`） |
| `README.md`（修改） | 评测命令示例更新 |

---

## Phase 1：评测侧 v3 重构（核心 TDD）

### Task 1: 新建纯函数模块 `eval/emotion_metrics.py`

**Files:**
- Create: `eval/emotion_metrics.py`
- Test: `tests/test_emotion_metrics.py`

- [ ] **Step 1: 写失败测试（核心指标正确性）**

创建 `tests/test_emotion_metrics.py`：

```python
"""emotion_metrics 纯函数核心测试（v3 评测契约）。"""
import json
import numpy as np
import pytest

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
```

说明：`compute_per_emotion_mean_sim` 输入行带 `emotion` 与 `emo_sim` 字段；`build_emotion_ref_index` 以 `(speaker_id, text.lower())` 为键；`normalize_text("Hello, World! 42")` 按现有规则（lowercase + 去标点 + 数字转英文）应为 `"hello world forty two"`。

- [ ] **Step 2: 运行确认失败**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=eval /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/test_emotion_metrics.py -v
```

Expected: FAIL（`ModuleNotFoundError: No module named 'emotion_metrics'`）。

- [ ] **Step 3: 实现 `eval/emotion_metrics.py`**

```python
"""EmoFiLM v3 评测纯指标函数（不加载模型、不含 CLI）。

v3 契约（emofilm-eval-v3）相对 v2 的精简与新增：
- 保留：frame 均值池化 Emo-SIM、DTW cosine-normalized、WER（manifest GT 文本）。
- 精简：删除 dtw/dtw_euclidean/dtw_euclidean_normalized 三个从未用于决策的字段。
- 新增：per-emotion 均值、n-way nearest-ref 情感判别、同/跨情感余弦。
"""
import json
import re

import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import cosine


# ============================================================
# 特征提取（funasr emotion2vec）
# ============================================================

def extract_utt_embeddings(model, wav_paths, batch_size=16):
    """批量提取 utterance embedding，返回 list of (dim,) 单位向量（顺序对齐）。"""
    if not wav_paths:
        return []
    results = model.generate(wav_paths, granularity="utterance",
                             extract_embedding=True, batch_size=batch_size)
    out = []
    for res in results:
        feats = res["feats"]
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        feats = feats.reshape(-1)
        out.append(feats / (np.linalg.norm(feats) + 1e-8))
    return out


def extract_frame_embeddings(model, wav_paths, batch_size=16):
    """批量提取 frame 级特征，返回 list of (T, dim) ndarray（顺序对齐）。"""
    if not wav_paths:
        return []
    results = model.generate(wav_paths, granularity="frame",
                             extract_embedding=True, batch_size=batch_size)
    out = []
    for res in results:
        feats = res["feats"]
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        out.append(feats)
    return out


# ============================================================
# 基础指标
# ============================================================

def _l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / (np.linalg.norm(vector) + 1e-8)


def compute_frame_mean_emo_sim(ref_feats, hyp_feats):
    """frame 特征均值池化 → L2 normalize → 余弦 ×100。"""
    ref = _l2_normalize(np.asarray(ref_feats).mean(axis=0))
    hyp = _l2_normalize(np.asarray(hyp_feats).mean(axis=0))
    return float(np.dot(ref, hyp) * 100.0)


def compute_dtw_normalized(ref_feats, hyp_feats):
    """cosine 距离的 fastdtw，按路径长度归一化（v3 唯一 DTW 口径）。"""
    raw, path = fastdtw(np.asarray(ref_feats, dtype=np.float64),
                        np.asarray(hyp_feats, dtype=np.float64), dist=cosine)
    path_len = len(path)
    if path_len == 0:
        raise ValueError("DTW path must not be empty")
    return float(raw / path_len)


# ============================================================
# WER
# ============================================================

_NUM_WORD_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _normalize_digits(text):
    for d, w in _NUM_WORD_MAP.items():
        text = text.replace(d, w)
    return text


def normalize_text(text):
    """lowercase + 去标点 + 数字转英文 + 多空格合并 + strip。"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = _normalize_digits(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 情感判别指标
# ============================================================

def build_emotion_ref_index(manifest_path):
    """读 jsonl（sources 级 manifest），返回 {(speaker_id, text.lower()): {emotion: wav_path}}。

    用于 n-way nearest-ref 判别：对每条 hyp，用同说话人同文本的其他情感参考音频
    做嵌入余弦对比。emotion 字段优先 sentence_emotion，回退 label。
    """
    index = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            spk = rec.get("speaker_id")
            text = rec.get("text")
            emotion = rec.get("sentence_emotion") or rec.get("label")
            wav = rec.get("wav_path")
            if not (spk and text and emotion and wav):
                continue
            index.setdefault((spk, text.lower()), {})[emotion] = wav
    return index


def compute_discriminability(hyp_paths, eval_rows, ref_index, emo_model,
                             batch_size=16):
    """对每条 hyp 计算与可用情感参考的余弦，输出 n-way 判别指标。

    前置契约：eval_rows 必须与 hyp_paths 按 utt_id 对齐（由调用方保证，
    见 run_evaluation 的重排）。
    返回 dict：n_valid（参考数>=3 的样本数）、n_skipped（参考不足被跳过的样本数）、
    n_way_avg、nearest_ref_acc_pct、same_emotion_mean / cross_emotion_mean /
    gap_same_minus_cross、n_way_distribution、mean_sim_by_ref_emotion。
    参考不足（n<3）的样本计入 n_skipped，不硬性失败（诚实口径）。
    """
    emotions = ["ang", "hap", "neu", "sad", "sur"]
    path_set = set()
    for i, row in enumerate(eval_rows):
        key = (row.get("speaker_id"), (row.get("text") or "").lower())
        refs = ref_index.get(key, {})
        if len(refs) < 3:
            continue
        for emo in emotions:
            p = refs.get(emo) or refs.get(
                row.get("emotion") or row.get("sentence_emotion") or row.get("label"))
            if p:
                path_set.add(p)
    if not path_set:
        return {"n_valid": 0, "reason": "no reference groups with >=3 emotions"}

    ref_embs = extract_utt_embeddings(emo_model, sorted(path_set), batch_size)
    ref_emb_by_path = dict(zip(sorted(path_set), ref_embs))
    hyp_embs = extract_utt_embeddings(emo_model, hyp_paths, batch_size)

    sims = {}  # row_idx -> {emotion: sim}
    for i, row in enumerate(eval_rows):
        key = (row.get("speaker_id"), (row.get("text") or "").lower())
        refs = ref_index.get(key, {})
        if len(refs) < 3:
            continue
        target = row.get("emotion") or row.get("sentence_emotion") or row.get("label")
        valid = [e for e in emotions if e in refs]
        row_sims = {}
        for e in valid:
            row_sims[e] = float(np.dot(hyp_embs[i], ref_emb_by_path[refs[e]]))
        sims[i] = row_sims

    valid_idx = list(sims.keys())
    n = len(valid_idx)
    if n == 0:
        return {"n_valid": 0, "reason": "no samples with usable references"}

    same = np.mean([sims[i].get(
        eval_rows[i].get("emotion") or eval_rows[i].get("sentence_emotion")
        or eval_rows[i].get("label"),
        np.nan) for i in valid_idx])
    cross = []
    acc = 0.0
    n_way_counts = {}
    for i in valid_idx:
        target = (eval_rows[i].get("emotion") or eval_rows[i].get("sentence_emotion")
                  or eval_rows[i].get("label"))
        candidates = {e: s for e, s in sims[i].items()}
        if target not in candidates:
            continue
        acc += int(max(candidates, key=candidates.get) == target)
        n_way = len(candidates)
        n_way_counts[str(n_way)] = n_way_counts.get(str(n_way), 0) + 1
        cross += [s for e, s in candidates.items() if e != target]
    acc_pct = acc / n * 100.0
    mean_sim_by_emo = {
        e: float(np.nanmean([sims[i].get(e, np.nan) for i in valid_idx]))
        for e in emotions
    }
    return {
        "n_valid": n,
        "n_skipped": len(eval_rows) - n,
        "n_way_avg": float(sum(int(k) * v for k, v in n_way_counts.items())
                           / sum(n_way_counts.values())) if n_way_counts else 0.0,
        "n_way_distribution": n_way_counts,
        "nearest_ref_acc_pct": round(acc_pct, 2),
        "same_emotion_mean": round(float(np.nanmean(same)), 2),
        "cross_emotion_mean": round(float(np.mean(cross)), 2),
        "gap_same_minus_cross": round(float(np.nanmean(same) - np.mean(cross)), 2),
        "mean_sim_by_ref_emotion": {
            e: round(mean_sim_by_emo[e], 2) for e in emotions
        },
    }


def compute_per_emotion_mean_sim(rows):
    """按 emotion 分组的 emo_sim 均值。输入行需含 emotion 与 emo_sim。"""
    groups = {}
    for r in rows:
        emo = r.get("emotion")
        if not emo:
            continue
        groups.setdefault(emo, []).append(r["emo_sim"])
    return {e: float(np.mean(v)) for e, v in groups.items()}


METRIC_CONTRACT_VERSION = "emofilm-eval-v3"
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=eval /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/test_emotion_metrics.py -v
```

Expected: 7 passed。

### Task 2: 重写 CLI `eval/eval_emo_film.py`（v3 schema + 判别指标接线）

**Files:**
- Modify: `eval/eval_emo_film.py`（整体重写，446 行 → ~150 行）
- Test: `tests/test_eval_emo_film.py`

- [ ] **Step 1: 写失败测试（聚合 schema + 配对 + mock 端到端）**

创建 `tests/test_eval_emo_film.py`：

```python
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
    common = ["--ref_dir=/r", "--hyp_dir=/h", "--output=/o/out.json"]
    assert parser.parse_args(common + ["--batch_size", "8"]).batch_size == 8
    assert parser.parse_args(common).batch_size == 16
    assert parser.parse_args(common + ["--expected_count", "10"]).expected_count == 10
    assert parser.parse_args(common).emotion_ref_manifest is None
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=eval /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/test_eval_emo_film.py -v
```

Expected: FAIL（`eval_emo_film` 仍为 v2 实现：缺 `emotion_ref_manifest` 参数、schema 不符）。

- [ ] **Step 3: 重写 `eval/eval_emo_film.py`**

```python
#!/usr/bin/env python3
"""EmoFiLM v3 评测 CLI（emofilm-eval-v3 契约）。

用法:
  python eval/eval_emo_film.py --ref_dir wav_ref/ --hyp_dir wav_hyp/ \\
      --output result.json --expected_count N \\
      --ref_text_manifest manifest.jsonl [--emotion_ref_manifest sources.jsonl]

v3 相对 v2 的破坏性变更（不向前兼容）：
- 输出 schema 精简为 metric_contract_version / n_samples / emo_sim /
  dtw_normalized / wer / wer_percent / per_emotion_emo_sim /
  discriminability（提供情感参考时）。
- 删除 dtw / dtw_euclidean / dtw_euclidean_normalized / --dtw_dist。
- --ref_text_manifest 必填；不再支持转写 ref 回退 WER。
- 新增 --emotion_ref_manifest → n-way nearest-ref 判别。
"""
import os
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from funasr import AutoModel
import whisper

from emotion_metrics import (
    METRIC_CONTRACT_VERSION,
    build_emotion_ref_index,
    compute_discriminability,
    compute_dtw_normalized,
    compute_frame_mean_emo_sim,
    compute_per_emotion_mean_sim,
    extract_frame_embeddings,
    normalize_text,
)


def _wav_map(directory):
    result = {}
    for name in os.listdir(directory):
        if not name.endswith(".wav"):
            continue
        utt_id = os.path.splitext(name)[0]
        if utt_id in result:
            raise ValueError(f"duplicate wav ID: {utt_id}")
        result[utt_id] = os.path.join(directory, name)
    return result


def pair_wavs_strict(ref_dir, hyp_dir, expected_count=None):
    """按 utt_id 严格交集配对；集合不等/空/数量不符 → hard-fail。"""
    refs, hyps = _wav_map(ref_dir), _wav_map(hyp_dir)
    if set(refs) != set(hyps):
        raise ValueError(
            f"wav ID mismatch: ref_only={sorted(set(refs)-set(hyps))[:5]} "
            f"hyp_only={sorted(set(hyps)-set(refs))[:5]}"
        )
    ids = sorted(refs)
    if not ids:
        raise ValueError("wav pair set must not be empty")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"expected {expected_count} pairs, got {len(ids)}")
    return [(utt_id, refs[utt_id], hyps[utt_id]) for utt_id in ids]


def load_manifest(path):
    """读 jsonl manifest，返回 {utt_id: {"text","speaker_id","emotion"}}。"""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get("utt_id")
            if uid is None:
                continue
            mapping[uid] = {
                "text": rec.get("text") or "",
                "speaker_id": rec.get("speaker_id"),
                "emotion": (rec.get("sentence_emotion")
                            or rec.get("label") or rec.get("emo_to")),
            }
    return mapping


def aggregate_metric_rows(rows):
    """v3 聚合。rows 需含 utt_id/emotion/emo_sim/dtw_normalized/wer。"""
    if not rows:
        raise ValueError("metric rows must not be empty")
    n = len(rows)
    return {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "n_samples": n,
        "emo_sim": float(np.mean([r["emo_sim"] for r in rows])),
        "dtw_normalized": float(np.mean([r["dtw_normalized"] for r in rows])),
        "wer": float(np.mean([r["wer"] for r in rows])),
        "wer_percent": float(np.mean([r["wer"] for r in rows])) * 100.0,
        "per_emotion_emo_sim": compute_per_emotion_mean_sim(rows),
    }


def compute_wer(whisper_model, wav_path):
    result = whisper_model.transcribe(wav_path)
    return result["text"].strip().lower() if result else ""


def transcribe_parallel(whisper_model, wav_paths, max_workers=16):
    """并行转写（whisper 实例线程不安全，加锁串行化模型调用）。"""
    if not wav_paths:
        return []
    lock = threading.Lock()

    def _safe(w):
        with lock:
            return compute_wer(whisper_model, w)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        return list(ex.map(_safe, wav_paths))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--hyp_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--expected_count", type=int, default=None)
    parser.add_argument("--ref_text_manifest", type=str, required=True,
                        help="jsonl（utt_id/text/可选 speaker_id+sentence_emotion）")
    parser.add_argument("--emotion_ref_manifest", type=str, default=None,
                        help="sources 级 jsonl（speaker_id/text/sentence_emotion/wav_path）")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser


def run_evaluation(emo_model, whisper_model, ref_dir, hyp_dir, text_map,
                   batch_size=16, wer_fn=None, expected_count=None,
                   emotion_ref_index=None, eval_rows=None):
    """v3 批量评测主流程，返回聚合 dict（含可选 discriminability）。"""
    pairs = pair_wavs_strict(ref_dir, hyp_dir, expected_count=expected_count)
    utt_ids = [p[0] for p in pairs]
    ref_paths = [p[1] for p in pairs]
    hyp_paths = [p[2] for p in pairs]

    interleaved = []
    for ref_p, hyp_p in zip(ref_paths, hyp_paths):
        interleaved += [ref_p, hyp_p]
    frame_feats = extract_frame_embeddings(emo_model, interleaved, batch_size=batch_size)

    hyp_texts = transcribe_parallel(whisper_model, hyp_paths, max_workers=batch_size)

    if wer_fn is None:
        try:
            from jiwer import wer as wer_fn
        except ImportError:
            def wer_fn(_r, _h):
                raise ImportError("jiwer not installed")

    rows = []
    for idx, utt_id in enumerate(utt_ids):
        ref_frame = frame_feats[idx * 2]
        hyp_frame = frame_feats[idx * 2 + 1]
        meta = text_map.get(utt_id, {})
        try:
            emo_sim = compute_frame_mean_emo_sim(ref_frame, hyp_frame)
            dtw_norm = compute_dtw_normalized(ref_frame, hyp_frame)
            if not meta.get("text"):
                raise ValueError("missing GT text (v3 不再支持转写 ref 回退)")
            wer = wer_fn(normalize_text(meta["text"]), normalize_text(hyp_texts[idx]))
        except Exception as e:
            raise RuntimeError(f"sample '{utt_id}' failed: {e}") from e
        rows.append({
            "utt_id": utt_id,
            "emotion": meta.get("emotion"),
            "emo_sim": emo_sim,
            "dtw_normalized": dtw_norm,
            "wer": float(wer),
        })

    result = aggregate_metric_rows(rows)
    if emotion_ref_index is not None and eval_rows is not None:
        # 判别输入必须与 hyp_paths 按 utt_id 对齐（hyp_paths 按 pair_wavs_strict 排序）
        rows_by_id = {r["utt_id"]: r for r in eval_rows}
        eval_rows = [rows_by_id[uid] for uid in utt_ids]
        result["discriminability"] = compute_discriminability(
            hyp_paths, eval_rows, emotion_ref_index, emo_model, batch_size)
    return result


def main():
    args = build_arg_parser().parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    emo_model = AutoModel(model="iic/emotion2vec_plus_large",
                          disable_update=True, device=device)
    whisper_model = whisper.load_model("large-v3", device=device)

    text_map = load_manifest(args.ref_text_manifest)
    eval_rows = [{"utt_id": uid, **v} for uid, v in text_map.items()]
    emotion_ref_index = (build_emotion_ref_index(args.emotion_ref_manifest)
                         if args.emotion_ref_manifest else None)

    result = run_evaluation(
        emo_model, whisper_model, args.ref_dir, args.hyp_dir, text_map,
        batch_size=args.batch_size, expected_count=args.expected_count,
        emotion_ref_index=emotion_ref_index, eval_rows=eval_rows,
    )
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Done. {result}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=eval /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/test_emotion_metrics.py tests/test_eval_emo_film.py -v
```

Expected: 全部通过（14 passed）。

### Task 3: 清理旧测试与过时合同，适配调用方

**Files:**
- Delete: `tests/test_eval_metric_contract.py`、`tests/test_eval_emo_film_batch.py`、`tests/test_eval_wer.py`
- Delete: `docs/contracts/emofilm_v2_evaluators.md`
- Modify: `tests/test_eval_smoke.py`、`tools/calibrate_eval_contract.py`、`README.md`
- Modify: `exp/emofilm_film_only/run_eval.sh`、`exp/emofilm_film_only_longepoch/run_eval.sh`、`exp/emofilm_sentlvl/run_eval.sh`
- Create: `docs/contracts/emofilm_v3_eval.md`

- [ ] **Step 1: 删除旧测试与过时合同**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && rm tests/test_eval_metric_contract.py tests/test_eval_emo_film_batch.py tests/test_eval_wer.py docs/contracts/emofilm_v2_evaluators.md
```

删除原因（写入最终执行汇报）：v2 九字段 schema 锁定与批处理细节断言已过时；`emofilm_v2_evaluators.md` 声明的权威实现 `eval/acoustic_evaluators.py` 不存在（ADR-0020 已确认对应模块移除），是未落地设计的残留合同。

- [ ] **Step 2: 更新 `tests/test_eval_smoke.py`**

把 `test_eval_runs_and_outputs_valid_json` 的 v2 断言块替换为：

```python
    assert data["metric_contract_version"] == "emofilm-eval-v3"
    for key in ("emo_sim", "dtw_normalized", "wer", "wer_percent",
                "n_samples", "per_emotion_emo_sim"):
        assert key in data, f"missing v3 field: {key}"
    assert 0 <= data["emo_sim"] <= 100 + 1e-2
    assert data["dtw_normalized"] >= 0
    assert 0 <= data["wer"] <= 1.0
    assert data["wer_percent"] == pytest.approx(data["wer"] * 100.0)
    assert data["n_samples"] >= 1
```

`setup_module()` 末尾追加（v3 必填 manifest；tmp 由返回值带出）：

```python
    manifest = os.path.join(tmp, "manifest.jsonl")
    with open(manifest, "w", encoding="utf-8") as f:
        for name in ["smoke_zh.wav", "smoke_en.wav"]:
            uid = os.path.splitext(name)[0]
            f.write(json.dumps({"utt_id": uid, "text": "smoke text",
                                "sentence_emotion": "neu"}) + "\n")
    return tmp, manifest
```

`test_eval_runs_and_outputs_valid_json` 同步改为 `tmp, manifest = setup_module()`，并在 `cmd` 中加入：

```python
        f"--ref_text_manifest={manifest}",
```

- [ ] **Step 3: 适配 `tools/calibrate_eval_contract.py`**

把 `from eval_emo_film import (...)` 改为（`compute_wer`/`normalize_text` 仍在 CLI 模块，`emotion_metrics` 不持有 whisper 依赖）：

```python
    from emotion_metrics import (
        extract_frame_embeddings,
        compute_frame_mean_emo_sim,
        compute_dtw_normalized,
    )
    from eval_emo_film import (
        compute_wer,
        normalize_text,
    )
```

逐变体循环中 `dtw_metrics = compute_dtw_metrics(ref_feats, hyp_feats)` 改为：

```python
        dtw_normalized = compute_dtw_normalized(ref_feats, hyp_feats)
```

`results[name] = {..., **dtw_metrics, ...}` 改为 `results[name] = {"emo_sim": emo_sim, "dtw_normalized": dtw_normalized, "wer": float(wer_value), "hyp_text": hyp_text}`；`_check_identity_hard_fail` 的 `dtw_keys` 改为 `("dtw_normalized",)`。

- [ ] **Step 4: 更新三个 `run_eval.sh`**

每个脚本开头加 `EXP=exp/<目录名>`，`run_eval` 函数替换为：

```bash
run_eval() {
  local ds=$1 cnt=$2 gpu=$3
  mkdir -p ${EXP}/eval/v3
  CUDA_VISIBLE_DEVICES=$gpu $PY eval/eval_emo_film.py \
    --ref_dir ${EXP}/eval_refs/$ds \
    --hyp_dir ${EXP}/full/$ds \
    --ref_text_manifest $WR/data/contracts/emofilm_v1/eval/$ds/manifest.jsonl \
    --emotion_ref_manifest $WR/data/contracts/emofilm_v1/sources/esd/manifest.jsonl \
    --output ${EXP}/eval/v3/${ds}_metrics.json \
    --device cuda --expected_count $cnt --batch_size 16 \
    > ${EXP}/eval/v3/${ds}.log 2>&1 &
}
```

说明：`--emotion_ref_manifest` 指向 ESD sources 全集；FEDD 无同文本跨情感参考，其 `discriminability` 会返回 `n_valid=0`（诚实口径，不失败）。历史 `eval/*_metrics.json` 保留不覆盖。

- [ ] **Step 5: 新建 `docs/contracts/emofilm_v3_eval.md`**

```markdown
# EmoFiLM v3 评测契约（emofilm-eval-v3）

权威实现：`eval/eval_emo_film.py`（CLI）+ `eval/emotion_metrics.py`（纯函数）。

## CLI

`python eval/eval_emo_film.py --ref_dir REF --hyp_dir HYP --output OUT
--expected_count N --ref_text_manifest EVAL.jsonl
[--emotion_ref_manifest SOURCES.jsonl] [--batch_size 16] [--device cuda]`

## 输出 schema

| 字段 | 语义 |
|---|---|
| `metric_contract_version` | 恒 `emofilm-eval-v3` |
| `n_samples` | 参与聚合样本数 |
| `emo_sim` | frame 均值池化余弦 ×100 |
| `dtw_normalized` | cosine fastdtw 按路径长度归一化 |
| `wer` / `wer_percent` | WER 比例 / 展示百分比（GT 文本 vs hyp 转写） |
| `per_emotion_emo_sim` | 按 emotion 分组的 emo_sim 均值 |
| `discriminability`（可选） | n-way 判别：n_valid / n_way_avg / nearest_ref_acc_pct / same_emotion_mean / cross_emotion_mean / gap_same_minus_cross / n_way_distribution / mean_sim_by_ref_emotion |

## 破坏性变更（相对 v2）

- 删除 dtw / dtw_euclidean / dtw_euclidean_normalized / `--dtw_dist`。
- `--ref_text_manifest` 必填；不再支持转写 ref 回退 WER。
- 新增 `--emotion_ref_manifest` 与判别指标。
```

- [ ] **Step 6: 更新 README 命令示例**

README 第 74-86 行的评测命令改为 v3 CLI 形态（含 `--emotion_ref_manifest`），并把 "v2 九字段" 表述改为 v3 schema。

- [ ] **Step 7: 全量验证（评测模块接入实验流程）**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=.:third_party/Matcha-TTS /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/ -q
```

Expected: 全绿（`test_eval_smoke` 缺 `/tmp/smoke_*.wav` 时 skip；`test_extract_emotion2vec_frame` 缺环境变量时 skip）。若出现引用 `emofilm_v2_evaluators` 或旧函数名/旧 schema 的失败，直接删除或精简对应断言（旧测试清理约定）。

---

## Phase 2：情感匹配 prompt 验证（R1 生成侧）

### Task 4: 生成验证 manifest（5 情感 × 12 句完整同文本组）

**Files:**
- Create: `tools/build_emotion_prompt_manifest.py`

- [ ] **Step 1: 实现脚本**

```python
#!/usr/bin/env python3
"""生成"情感匹配 prompt"验证 manifest：5 情感 × N 句完整同文本组。

从 sources/esd 的 tagged.jsonl（含 tagged_text，15127 行）挑出 (speaker, text)
5 情感齐全的组，每句 5 条 target 全保留；prompt_wav 用同说话人同情感的另一条
wav（不同句，避免自我克隆/内容泄漏）。输出 jsonl 供
tools/inference_emo_film.py 直接使用（manifest prompt_wav 优先）。
"""
import argparse
import collections
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True,
                        help="sources/esd/tagged.jsonl（含 tagged_text）")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sentences", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.input_manifest, encoding="utf-8") if l.strip()]
    groups = collections.defaultdict(dict)  # (spk, text.lower()) -> {emotion: row}
    for r in rows:
        emo = r.get("sentence_emotion") or r.get("label")
        if emo:
            groups[(r["speaker_id"], r["text"].lower())][emo] = r

    full_groups = [k for k, v in groups.items() if len(v) >= 5]
    rng = random.Random(args.seed)
    chosen = rng.sample(full_groups, min(args.sentences, len(full_groups)))

    out = []
    for spk, text_lower in chosen:
        by_emo = groups[(spk, text_lower)]
        for emo, r in by_emo.items():
            prompt_candidates = [
                rr for ee, rr in by_emo.items()
                if ee == emo and rr["utt_id"] != r["utt_id"]
            ]
            prompt = prompt_candidates[0] if prompt_candidates else r
            out.append({
                "utt_id": r["utt_id"],
                "wav_path": r["wav_path"],
                "target_wav": r["wav_path"],
                "reference_wav": r["wav_path"],
                "text": r["text"],
                "tagged_text": r["tagged_text"],
                "plain_text": r.get("plain_text") or r["text"],
                "sentence_emotion": emo,
                "label": emo,
                "speaker_id": spk,
                "prompt_wav": prompt["wav_path"],
                "prompt_text": prompt.get("text") or "",
                "prompt_source": "esd_same_speaker_same_emotion",
                "source_dataset": "esd",
            })
    with open(args.output, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows ({len(out) // 5} sentences x 5 emotions) -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行生成并抽查**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && \
  /home/hanlvyuan/miniconda3/envs/emofilm/bin/python tools/build_emotion_prompt_manifest.py \
  --input_manifest data/contracts/emofilm_v1/sources/esd/tagged.jsonl \
  --output data/contracts/emofilm_v1/eval/esd_prompt_match_60.jsonl --sentences 12
```

Expected: `wrote 60 rows (12 sentences x 5 emotions)`；`python -c "..."` 抽查 3 行确认 `prompt_source=esd_same_speaker_same_emotion` 且 prompt_wav 与 target 不同 utt。

### Task 5: 四模型生成 + v3 评测 + 对比汇总

**Files:**
- Create: `exp/prompt_match/run_infer.sh`、`exp/prompt_match/run_eval.sh`

- [ ] **Step 1: 写推理脚本（4 GPU 并行）**

`exp/prompt_match/run_infer.sh`：

```bash
#!/bin/bash
# 情感匹配 prompt 验证：4 模型 × 60 条（5 情感 × 12 句），同情感 prompt。
set -o pipefail
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
WR=/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
PY=/home/hanlvyuan/miniconda3/envs/emofilm/bin/python
export PYTHONPATH=.:third_party/Matcha-TTS
export MODELSCOPE_OFFLINE=1
MANIFEST=$WR/data/contracts/emofilm_v1/eval/esd_prompt_match_60.jsonl

run_model() {
  local name=$1 ckpt=$2 gpu=$3
  mkdir -p exp/prompt_match/$name/esd
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/inference_emo_film.py \
    --test_manifest $MANIFEST \
    --output_dir exp/prompt_match/$name/esd \
    --model_dir $WR/pretrained_models/CosyVoice2-0.5B --llm_ckpt $ckpt \
    --esd_root $WR/datasets/ESD --workspace_root $WR \
    --device cuda --fp16 --seed 1986 --save_every 20 --skip_existing \
    > exp/prompt_match/$name/infer.log 2>&1 &
}

run_model v1            exp/emofilm_v1/final.pt                    0
run_model film_only     exp/emofilm_film_only/final.pt             1
run_model longepoch     exp/emofilm_film_only_longepoch/final.pt   2
run_model sentlvl       exp/emofilm_sentlvl/final.pt               3
wait
echo "ALL PROMPT-MATCH INFERENCE DONE $(date)"
```

- [ ] **Step 2: 运行推理**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && bash exp/prompt_match/run_infer.sh
```

Expected: 4 个目录各 60 个 wav；`inference_manifest.jsonl` 60 行全部 `finish_reason=eos`（约 1 小时，4 GPU）。

- [ ] **Step 3: 建 60 条 ref 视图 + 写评测脚本**

```bash
mkdir -p exp/prompt_match/refs/esd && \
/home/hanlvyuan/miniconda3/envs/emofilm/bin/python - <<'EOF'
import json, os
rows=[json.loads(l) for l in open('data/contracts/emofilm_v1/eval/esd_prompt_match_60.jsonl')]
for r in rows:
    dst=os.path.join('exp/prompt_match/refs/esd', r['utt_id']+'.wav')
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(r['reference_wav']), dst)
EOF
```

`exp/prompt_match/run_eval.sh`：

```bash
#!/bin/bash
set -o pipefail
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
WR=/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
PY=/home/hanlvyuan/miniconda3/envs/emofilm/bin/python
export PYTHONPATH=.:third_party/Matcha-TTS
export MODELSCOPE_OFFLINE=1
MANIFEST=$WR/data/contracts/emofilm_v1/eval/esd_prompt_match_60.jsonl

for name in v1 film_only longepoch sentlvl; do
  mkdir -p exp/prompt_match/$name/eval
  CUDA_VISIBLE_DEVICES=0 $PY eval/eval_emo_film.py \
    --ref_dir $WR/exp/prompt_match/refs/esd \
    --hyp_dir exp/prompt_match/$name/esd \
    --ref_text_manifest $MANIFEST \
    --emotion_ref_manifest $WR/data/contracts/emofilm_v1/sources/esd/manifest.jsonl \
    --output exp/prompt_match/$name/eval/esd_v3_metrics.json \
    --device cuda --expected_count 60 --batch_size 16 \
    > exp/prompt_match/$name/eval/esd_v3.log 2>&1
done
echo "ALL PROMPT-MATCH EVAL DONE $(date)"
```

- [ ] **Step 4: 运行评测并写汇总**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && bash exp/prompt_match/run_eval.sh
```

Expected: 4 个 `esd_v3_metrics.json`，`discriminability.n_valid=60`、`n_way_distribution={"5": 60}`（验证组保证 5-way 完整）。

写 `exp/prompt_match/summary.md` 对比表：每模型 Emo-SIM / per-emotion / 5-way acc / same−cross gap / WER，并对照原中性 prompt 全量评测（`exp/*/eval/*_metrics.json`），给出结论"情感匹配 prompt 是否绕过声学钳制、每模型实际上限"。

---

## Phase 3：CosyVoice3 官方基线测试

### Task 6: 下载官方权重 + 冒烟确认推理入口

- [ ] **Step 1: 下载权重（ModelScope）**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && \
  /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -c "
from modelscope import snapshot_download
p = snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512')
print('downloaded to', p)
"
```

Expected: 缓存目录含 `cosyvoice3.yaml / llm.pt / flow.pt / hift.pt / speech_tokenizer_v3.onnx / CosyVoice-BlankEN / campplus.onnx / spk2info.pt`。若网络不可达，改用镜像 `git clone https://gitserver.onethingai.com/ai-models/Fun-CosyVoice3-0.5B-2512.git` 并记录。

- [ ] **Step 2: 冒烟：CV3 指令推理**

从官方 README（HF 模型卡 / FunAudioLLM/CosyVoice 仓库）取 CV3 instruct 示例模板，先试本仓库 CLI：

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && \
  MODELSCOPE_OFFLINE=1 /home/hanlvyuan/miniconda3/envs/emofilm/bin/python - <<'EOF'
from cosyvoice.cli.cosyvoice import AutoModel
model = AutoModel(model_dir='<下载缓存目录>')
for chunk in model.inference_instruct2(
        'Rare rabbit had a little apron.',
        'Speak with angry emotion.',
        'datasets/ESD/0011/Neutral/0011_000001.wav'):
    print('finish_reason=', chunk['finish_reason'],
          'audio=', chunk['tts_speech'].shape)
    break
EOF
```

Expected: 输出 audio 且 finish_reason=eos。若 `frontend_instruct2` 与 CV3 协议不兼容（LLM 输入 key/形状错误），则 clone 官方仓库最新版到 `/tmp/CosyVoice-official`，按其 README 的 CosyVoice3 示例脚本跑通；执行汇报中记录实际入口、instruct 模板与加载耗时。

- [ ] **Step 3: 写冒烟结论**

入口结论、instruct 模板、加载耗时写入 `exp/cosyvoice3_baseline/smoke.md`。

### Task 7: CV3 150 条 ESD 子集生成 + 评测 + 对比报告

- [ ] **Step 1: 生成 150 条子集 manifest 与 ref 视图**

从 `data/contracts/emofilm_v1/eval/esd/manifest.jsonl` 按情感分层抽 5 × 30（seed=7），保留原 `prompt_wav`（中性），输出 `data/contracts/emofilm_v1/eval/esd_cv3_150.jsonl`；建 `exp/cosyvoice3_baseline/refs/esd/` symlink 视图。

- [ ] **Step 2: 生成 150 条音频**

`exp/cosyvoice3_baseline/run_infer.sh`（本仓库 CLI 版模板；若 Task 6 冒烟采用官方脚本，则仅替换调用行、保留 manifest/输出结构）：

```bash
#!/bin/bash
# CV3 官方基线：150 条 ESD（5 情感 × 30 句），instruct 情感指令 + 中性 prompt。
set -o pipefail
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
WR=/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
PY=/home/hanlvyuan/miniconda3/envs/emofilm/bin/python
export PYTHONPATH=.:third_party/Matcha-TTS
export MODELSCOPE_OFFLINE=1
CV3_DIR=<Task 6 确认的模型缓存目录>
MANIFEST=$WR/data/contracts/emofilm_v1/eval/esd_cv3_150.jsonl
mkdir -p exp/cosyvoice3_baseline/esd_150

$PY - <<'EOF' > exp/cosyvoice3_baseline/infer.log 2>&1
import json
from cosyvoice.cli.cosyvoice import AutoModel
import torch, torchaudio

model = AutoModel(model_dir='''<CV3_DIR>''')
EMO_INSTRUCT = {'ang': 'Speak with angry emotion.',
                'hap': 'Speak with happy emotion.',
                'neu': 'Speak with neutral emotion.',
                'sad': 'Speak with sad emotion.',
                'sur': 'Speak with surprised emotion.'}
rows = [json.loads(l) for l in open('''<MANIFEST>''')]
for i, r in enumerate(rows):
    out = f"exp/cosyvoice3_baseline/esd_150/{r['utt_id']}.wav"
    torch.manual_seed(1986 + i)
    for chunk in model.inference_instruct2(
            r['text'], EMO_INSTRUCT[r['sentence_emotion']], r['prompt_wav']):
        if chunk.get('finish_reason') == 'eos' and chunk.get('tts_speech') is not None:
            torchaudio.save(out, chunk['tts_speech'].cpu(), model.sample_rate)
            print(r['utt_id'], 'OK')
        break
EOF
echo "CV3 INFER DONE $(date)"
```

`<CV3_DIR>` 与 `<MANIFEST>` 在落盘时替换为实际路径；instruct 模板以 Task 6 冒烟确认的官方模板为准。1 GPU，预计 2-3 小时；Expected: 150 个 wav，日志 150 行 `OK`。

- [ ] **Step 3: v3 评测**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && \
  CUDA_VISIBLE_DEVICES=0 /home/hanlvyuan/miniconda3/envs/emofilm/bin/python eval/eval_emo_film.py \
  --ref_dir exp/cosyvoice3_baseline/refs/esd \
  --hyp_dir exp/cosyvoice3_baseline/esd_150 \
  --ref_text_manifest data/contracts/emofilm_v1/eval/esd_cv3_150.jsonl \
  --emotion_ref_manifest data/contracts/emofilm_v1/sources/esd/manifest.jsonl \
  --output exp/cosyvoice3_baseline/eval/esd_v3_metrics.json \
  --device cuda --expected_count 150 --batch_size 16
```

Expected: `n_samples=150`；`discriminability.n_valid>0`（eval 抽样可用参考 n≥3）。

- [ ] **Step 4: 写四方 + CV3 对比报告**

`docs/reports/2026-08-03-cosyvoice3-baseline-comparison.md`：同一 v3 口径下列出 v1 / 5ep / 27ep / sentlvl（原中性 prompt 全量）+ CV3 150 条子集（Emo-SIM / per-emotion / n-way acc / WER），并给结论：CV3 官方基线是否已超过 ~66 平台；若未超，说明问题在生成/评测协议而非基座（报告 §6.4 决策树）。

---

## Self-Review

**Spec 覆盖：**
- 评测侧重构（第一块任务核心）→ Phase 1 Task 1-3。
- 加验证（R1 生成侧：情感匹配 prompt 四模型）→ Phase 2 Task 4-5。
- CosyVoice3 替换测试（报告 §6.4）→ Phase 3 Task 6-7。
- 旧测试/合同清理 + 测试保证接入流程 → Task 3（删旧测试、冒烟更新、run_eval.sh、全量 pytest）。
- 核心 TDD → Task 1/2 先写测试后实现；Task 3 Step 7 为流程级验证。

**无占位符检查：** 所有代码块均给出完整实现或具体替换片段；CV3 推理入口的不确定性以 Task 6 Step 2 的"双路径冒烟"显式覆盖，不依赖未来信息。

**类型一致性：** `load_manifest` 返回 `{utt_id: {"text","speaker_id","emotion"}}`，`run_evaluation` 的 `text_map`/`eval_rows` 用法在 Task 2 测试与实现中一致；`compute_discriminability(hyp_paths, eval_rows, ref_index, emo_model, batch_size)` 签名在 emotion_metrics 与 CLI 调用处一致。

**已知风险与对策：**
- CV3 推理入口不确定 → Task 6 双路径冒烟，先本仓库 CLI 后官方仓库。
- 情感匹配 prompt 评测的 ref 配对 → Task 5 Step 3 新建 60 条 ref 视图，避免全量配对失败。
- v3 强制 `--ref_text_manifest` → 所有 run_eval.sh 与 smoke 测试同步更新。
- FEDD 无同文本跨情感参考 → `discriminability` 返回 `n_valid=0`（诚实口径），不影响基础指标。
