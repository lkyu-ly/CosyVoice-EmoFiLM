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
        valid = [e for e in emotions if e in refs]
        row_sims = {}
        for e in valid:
            row_sims[e] = float(np.dot(hyp_embs[i], ref_emb_by_path[refs[e]]))
        sims[i] = row_sims

    valid_idx = list(sims.keys())
    n = len(valid_idx)
    if n == 0:
        return {"n_valid": 0, "reason": "no samples with usable references"}

    same_vals = []  # 目标情感参考相似度（仅 target 在参考中的行）
    cross = []
    acc = 0.0
    n_scored = 0
    n_way_counts = {}
    for i in valid_idx:
        target = (eval_rows[i].get("emotion") or eval_rows[i].get("sentence_emotion")
                  or eval_rows[i].get("label"))
        candidates = {e: s for e, s in sims[i].items()}
        if target not in candidates:
            continue
        n_scored += 1
        acc += int(max(candidates, key=candidates.get) == target)
        same_vals.append(candidates[target])
        n_way = len(candidates)
        n_way_counts[str(n_way)] = n_way_counts.get(str(n_way), 0) + 1
        cross += [s for e, s in candidates.items() if e != target]
    if n_scored == 0:
        # 退化：无任何样本的目标情感在可用参考中。ESD train/eval 划分会把每组的
        # 目标情感单独留到 eval，sources 参考恰好缺该情感 → nearest-ref acc/gap
        # 结构上不可计算。诚实返回 reason，不写 NaN、不报误导性 0%。
        return {
            "n_valid": n,
            "n_skipped": len(eval_rows) - n,
            "n_scored": 0,
            "n_way_distribution": {},
            "reason": "target emotion absent from all reference groups "
                      "(eval/train split excludes target; use full 5-emotion groups for discriminability)",
        }
    acc_pct = acc / n_scored * 100.0
    # mean_sim_by_ref_emotion：仅收录至少在一个可用参考组中出现的情感，
    # 避免对缺失情感写入非法 NaN（诚实口径：无参考 → 不报告该情感）。
    present_emotions = {e for i in valid_idx for e in sims[i]}
    mean_sim_by_emo = {}
    for e in emotions:
        if e not in present_emotions:
            continue
        vals = [sims[i].get(e, np.nan) for i in valid_idx]
        mean_sim_by_emo[e] = float(np.nanmean(vals))
    return {
        "n_valid": n,
        "n_skipped": len(eval_rows) - n,
        "n_scored": n_scored,
        "n_way_avg": float(sum(int(k) * v for k, v in n_way_counts.items())
                           / sum(n_way_counts.values())),
        "n_way_distribution": n_way_counts,
        "nearest_ref_acc_pct": round(acc_pct, 2),
        "same_emotion_mean": round(float(np.mean(same_vals)), 2),
        "cross_emotion_mean": round(float(np.mean(cross)), 2),
        "gap_same_minus_cross": round(float(np.mean(same_vals) - np.mean(cross)), 2),
        "mean_sim_by_ref_emotion": {
            e: round(mean_sim_by_emo[e], 2) for e in emotions if e in mean_sim_by_emo
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
