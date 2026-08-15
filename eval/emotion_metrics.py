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
        # 与 _l2_normalize 一致：max(norm, eps) 避免给单位向量引入系统偏差
        out.append(feats / np.maximum(np.linalg.norm(feats), 1e-8))
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
    # 用 max(norm, eps) 而非 norm+eps：后者给每个非零向量的归一化引入 ~eps/norm
    # 的系统性偏差，使恒等对照 emo_sim 偏离 100（v3 测试 1e-6 硬约束）。
    return vector / np.maximum(np.linalg.norm(vector), 1e-8)


def compute_frame_mean_emo_sim(ref_feats, hyp_feats):
    """frame 特征均值池化 → L2 normalize → 余弦 ×100。

    均值前显式转 float64：float32 输入在 float32 下累积均值会引入 ~1e-6 量级
    误差，恒等对照 emo_sim 会偏离 100 超出 1e-6 容差（v3 测试硬约束）。
    """
    ref = _l2_normalize(np.asarray(ref_feats, dtype=np.float64).mean(axis=0))
    hyp = _l2_normalize(np.asarray(hyp_feats, dtype=np.float64).mean(axis=0))
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

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n):
    """非负整数 → 英文单词（如 42 → 'forty two'，1001 → 'one thousand one'）。

    支持到百万量级，覆盖 WER 文本中可能出现的数字；负数/超大值原样返回 str(n)。
    """
    if n < 0:
        return str(n)

    def _under_thousand(num):
        parts = []
        if num >= 100:
            parts.append(_ONES[num // 100] + " hundred")
            num %= 100
        if num >= 20:
            parts.append(_TENS[num // 10])
            if num % 10:
                parts.append(_ONES[num % 10])
        elif num > 0:
            parts.append(_ONES[num])
        return " ".join(parts)

    if n == 0:
        return "zero"
    parts = []
    if n >= 1000:
        parts.append(_under_thousand(n // 1000) + " thousand")
        n %= 1000
    parts.append(_under_thousand(n))
    return " ".join(p for p in parts if p)


def _normalize_digits(text):
    """连续数字段 → 英文单词（如 42 → 'forty two'）；非数字段原样。

    按完整数字段（\\d+）整体转换，而非逐字符替换——后者会把 42 错写成 'fourtwo'。
    """
    return re.sub(r"\d+", lambda m: _int_to_words(int(m.group(0))), text)


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


EMOTIONS = ["ang", "hap", "neu", "sad", "sur"]


def _merged_refs(ref_index, row):
    """合并 sources 索引与 eval 行自带 reference_wav（目标情感参考）。

    ESD train/eval 划分会把 (speaker, text) 组的目标情感单独留到 eval，sources
    索引缺目标情感；reference_wav 是目标情感真实音频，合并后判别候选才包含
    "正确答案"。返回 {emotion: wav_path}；reference_wav 存在则覆盖目标情感。
    """
    target = (row.get("emotion") or row.get("sentence_emotion")
              or row.get("label"))
    refs = dict(ref_index.get(
        (row.get("speaker_id"), (row.get("text") or "").lower()), {}))
    rw = row.get("reference_wav")
    if target and rw:
        refs[target] = rw
    return refs


def compute_discriminability(hyp_paths, eval_rows, ref_index, emo_model,
                             batch_size=16):
    """对每条 hyp 计算与可用情感参考的余弦，输出 n-way 判别指标。

    前置契约：eval_rows 必须与 hyp_paths 按 utt_id 对齐（由调用方保证）。
    流程：合并参考 → 过滤候选>=3 → 批量提取嵌入 → 单遍判别 → 聚合。
    所有返回分支使用同一 schema；不可算时 reason 说明原因，不写 NaN。
    """
    # 1) 预处理：每行合并参考并过滤候选 >=3
    refs_by_idx = {}
    for i, row in enumerate(eval_rows):
        refs = _merged_refs(ref_index, row)
        if len(refs) >= 3:
            refs_by_idx[i] = refs
    if not refs_by_idx:
        return {
            "n_valid": 0, "n_skipped": len(eval_rows), "n_scored": 0,
            "n_way_avg": 0.0, "n_way_distribution": {},
            "nearest_ref_acc_pct": 0.0, "same_emotion_mean": None,
            "cross_emotion_mean": None, "gap_same_minus_cross": None,
            "mean_sim_by_ref_emotion": {},
            "reason": "no reference groups with >=3 emotions",
        }

    # 2) 批量提取参考嵌入（路径去重后一次提取）
    ref_paths = sorted({p for refs in refs_by_idx.values() for p in refs.values()})
    ref_embs = extract_utt_embeddings(emo_model, ref_paths, batch_size)
    ref_emb_by_path = dict(zip(ref_paths, ref_embs))
    hyp_embs = extract_utt_embeddings(emo_model, hyp_paths, batch_size)

    # 3) 单遍判别聚合
    n_way_counts = {}
    acc = n_scored = 0
    same_vals, cross_vals = [], []
    for i, refs in refs_by_idx.items():
        target = (eval_rows[i].get("emotion") or eval_rows[i].get("sentence_emotion")
                  or eval_rows[i].get("label"))
        if target not in refs:
            continue
        sims = {e: float(np.dot(hyp_embs[i], ref_emb_by_path[p]))
                for e, p in refs.items()}
        n_scored += 1
        n_way = len(sims)
        n_way_counts[str(n_way)] = n_way_counts.get(str(n_way), 0) + 1
        acc += int(max(sims, key=sims.get) == target)
        same_vals.append(sims[target])
        cross_vals += [s for e, s in sims.items() if e != target]
    if n_scored == 0:
        return {
            "n_valid": len(refs_by_idx),
            "n_skipped": len(eval_rows) - len(refs_by_idx),
            "n_scored": 0, "n_way_avg": 0.0, "n_way_distribution": {},
            "nearest_ref_acc_pct": 0.0, "same_emotion_mean": None,
            "cross_emotion_mean": None, "gap_same_minus_cross": None,
            "mean_sim_by_ref_emotion": {},
            "reason": "target emotion absent from merged references",
        }

    present = {e for refs in refs_by_idx.values() for e in refs}
    mean_sim_by_emo = {
        e: float(np.mean([np.dot(hyp_embs[i], ref_emb_by_path[refs[e]])
                          for i, refs in refs_by_idx.items() if e in refs]))
        for e in EMOTIONS if e in present
    }
    return {
        "n_valid": len(refs_by_idx),
        "n_skipped": len(eval_rows) - len(refs_by_idx),
        "n_scored": n_scored,
        "n_way_avg": float(sum(int(k) * v for k, v in n_way_counts.items())
                           / sum(n_way_counts.values())),
        "n_way_distribution": n_way_counts,
        "nearest_ref_acc_pct": round(acc / n_scored * 100.0, 2),
        "same_emotion_mean": round(float(np.mean(same_vals)), 2),
        "cross_emotion_mean": round(float(np.mean(cross_vals)), 2),
        "gap_same_minus_cross": round(
            float(np.mean(same_vals) - np.mean(cross_vals)), 2),
        "mean_sim_by_ref_emotion": mean_sim_by_emo,
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
