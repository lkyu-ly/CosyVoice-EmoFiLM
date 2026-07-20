#!/usr/bin/env python3
"""客观评测指标计算（emofilm-eval-v2 契约）: Emo-SIM, DTW, WER。

用法: python eval/eval_emo_film.py --ref_dir wav_ref/ --hyp_dir wav_hyp/ \
        --output result.json --expected_count N

v2 评测契约（emofilm-eval-v2）：
- 配对：pair_wavs_strict 按 utt_id 取严格交集；空/数量不符/重复 ID → hard-fail，
  禁止 v1 的静默 sorted 回退。CLI 新增 --expected_count，正式运行必须显式提供。
- Emo-SIM：从 frame features 均值池化 → L2 normalize → 余弦 ×100
  （compute_frame_mean_emo_sim，替代 v1 的 utterance-level feats）。
- DTW：compute_dtw_metrics 同时输出 cosine（dtw / dtw_normalized，正式口径）
  与 euclidean（dtw_euclidean / dtw_euclidean_normalized，诊断口径）。
  v1 --dtw_dist 选择正式口径已移除（仅作兼容保留参数）。
- WER：wer ∈ [0,1] 比例，wer_percent = wer × 100 仅用于展示。
  空 rows → aggregate_metric_rows hard-fail（替代 v1 的 wers 空 → wer=1 韧性）。
- 逐条异常：立即抛出携带 utt_id，不再 try/except 吞掉返回部分均值。

输出 JSON 9 字段（aggregate_metric_rows）：
  metric_contract_version, emo_sim, dtw, dtw_normalized, dtw_euclidean,
  dtw_euclidean_normalized, wer, n_samples, wer_percent

批处理吞吐优化（ADR-0002，保留）：
- frame features 一次性批量提取（Emo-SIM/DTW 共用，不再分别句级/帧级）。
- Whisper 转写 hyp（及回退 ref）：ThreadPoolExecutor 并行；单条 transcribe 语义不变。

WER normalization（spec 12.1）：
- 论文未公开 normalization 细节，作者源码 evaluate.py 只实现 ACC-E/SS，未实现 WER
- 本实现采用：lowercase + 去标点 + 数字→英文 + 多空格合并 + strip
"""
import os
# 离线模式：跳过 modelscope 更新检查（避免每次启动重新下载 2.88G 模型）
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchaudio
from funasr import AutoModel
from fastdtw import fastdtw
from scipy.spatial.distance import cosine, euclidean
import whisper


# ============================================================
# emotion2vec 特征批量提取
# ============================================================

def extract_utt_embeddings(model, wav_paths, batch_size=16):
    """批量提取 utterance embedding，返回 list of (dim,) 单位向量（顺序对齐 wav_paths）。

    funasr AutoModel.generate 接受 wav 路径**列表** + batch_size 做真分批，返回等长
    list of {"feats": ...}（顺序对齐输入）。逐 feats → numpy → reshape(-1) → L2 normalize。

    等价于对每条单独 generate(wav, granularity="utterance", extract_embedding=True)
    再 normalize（ADR-0002：batch padding+mask 逐样本输出与串行一致）。
    """
    if not wav_paths:
        return []
    results = model.generate(wav_paths, granularity="utterance",
                             extract_embedding=True, batch_size=batch_size)
    out = []
    for res in results:
        feats = res["feats"]
        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()
        feats = feats.reshape(-1)  # utterance feats 已是 1D, reshape 为 no-op 但保险
        out.append(feats / (np.linalg.norm(feats) + 1e-8))
    return out


def extract_frame_embeddings(model, wav_paths, batch_size=16):
    """批量提取 frame 级特征，返回 list of (T, dim) ndarray（顺序对齐 wav_paths）。

    供 DTW 输入。funasr generate(wav_list, granularity="frame", extract_embedding=True,
    batch_size=N)。逐 feats → numpy（保留 2D 帧维度，不 reshape）。
    """
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
# v2 指标原语（emofilm-eval-v2 契约）
# ============================================================

METRIC_CONTRACT_VERSION = "emofilm-eval-v2"


def _l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / (np.linalg.norm(vector) + 1e-8)


def compute_frame_mean_emo_sim(ref_feats, hyp_feats):
    ref = _l2_normalize(np.asarray(ref_feats).mean(axis=0))
    hyp = _l2_normalize(np.asarray(hyp_feats).mean(axis=0))
    return float(np.dot(ref, hyp) * 100.0)


def _dtw_pair(ref_feats, hyp_feats, distance):
    raw, path = fastdtw(ref_feats, hyp_feats, dist=distance)
    path_len = len(path)
    if path_len == 0:
        raise ValueError("DTW path must not be empty")
    return float(raw), float(raw / path_len)


def compute_dtw_metrics(ref_feats, hyp_feats):
    cosine_raw, cosine_norm = _dtw_pair(ref_feats, hyp_feats, cosine)
    euclidean_raw, euclidean_norm = _dtw_pair(ref_feats, hyp_feats, euclidean)
    return {
        "dtw": cosine_raw,
        "dtw_normalized": cosine_norm,
        "dtw_euclidean": euclidean_raw,
        "dtw_euclidean_normalized": euclidean_norm,
    }


def compute_emo_sim(model, wav_path, device=None):
    """提取单条 utterance feats (dim,) → L2 normalize 返回单位向量（薄包装）。

    Stage 0 实测确认 utterance feats 已是 (1024,)，无需均值池化（spec 12.1 措辞修正）。
    保留供逐条调用；批量场景请用 extract_utt_embeddings。
    """
    return extract_utt_embeddings(model, [wav_path], batch_size=1)[0]


def compute_dtw_distance(ref_feats, hyp_feats, dist_fn="euclidean"):
    """fastdtw 距离，返回 (raw_distance, path_normalized_distance)。纯函数。

    输入为已提取的 frame 特征 ndarray。距离计算逻辑与原逐条 compute_dtw 完全一致：
    euclidean（默认, 论文量级）或 cosine（嵌入空间）。批量场景配合
    extract_frame_embeddings 使用；逐条场景见 compute_dtw。
    """
    dist_fn_impl = euclidean if dist_fn == "euclidean" else cosine
    dist, path = fastdtw(ref_feats, hyp_feats, dist=dist_fn_impl)
    path_len = len(path) if path else 1
    return dist, dist / path_len


def compute_dtw(model, ref_path, hyp_path, device=None, dist_fn="euclidean"):
    """单条 frame DTW（薄包装），返回 (raw_distance, path_normalized_distance)。

    保留供逐条调用；批量场景请用 extract_frame_embeddings + compute_dtw_distance。
    """
    ref_feats = extract_frame_embeddings(model, [ref_path], batch_size=1)[0]
    hyp_feats = extract_frame_embeddings(model, [hyp_path], batch_size=1)[0]
    return compute_dtw_distance(ref_feats, hyp_feats, dist_fn)


# ============================================================
# v2 严格配对 + 聚合（emofilm-eval-v2 契约，Task 2）
# ============================================================

def _wav_map(directory):
    """扫描目录返回 {utt_id: path}；同 utt_id 重复直接抛错（hard-fail）。"""
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
    """严格配对 ref/hyp：按 utt_id 取交集，集合不等 / 空 / 数量不符均 hard-fail。

    返回 [(utt_id, ref_path, hyp_path), ...]（按 utt_id 排序），禁止旧实现的
    『空交集回退排序前 N』静默行为。
    """
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


def aggregate_metric_rows(rows):
    """v2 聚合：逐样本指标行 → result dict（9 字段 emofilm-eval-v2 schema）。

    - 空 rows → hard-fail（禁止返回貌似合法的默认值）
    - wer: [0,1] 比例（论文口径），wer_percent = wer * 100 仅用于展示
    """
    if not rows:
        raise ValueError("metric rows must not be empty")
    result = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "emo_sim": float(np.mean([r["emo_sim"] for r in rows])),
        "dtw": float(np.mean([r["dtw"] for r in rows])),
        "dtw_normalized": float(np.mean([r["dtw_normalized"] for r in rows])),
        "dtw_euclidean": float(np.mean([r["dtw_euclidean"] for r in rows])),
        "dtw_euclidean_normalized": float(np.mean([r["dtw_euclidean_normalized"] for r in rows])),
        "wer": float(np.mean([r["wer"] for r in rows])),
        "n_samples": len(rows),
    }
    result["wer_percent"] = result["wer"] * 100.0
    return result


# ============================================================
# Whisper 转写
# ============================================================

def compute_wer(whisper_model, wav_path):
    """Whisper-large-v3 转写音频为文本。

    单条调用线程安全；whisper 模型实例的 decoder 共享 kv_cache / mask buffer，
    并发 transcribe 同一实例会竞争崩溃（见 transcribe_parallel 的锁）。
    """
    result = whisper_model.transcribe(wav_path)
    return result["text"].strip().lower() if result else ""


def transcribe_parallel(whisper_model, wav_paths, max_workers=16):
    """ThreadPoolExecutor 并行 Whisper 转写，返回文本列表（顺序对齐输入）。

    Whisper transcribe 为单条 API；不改单条转写语义（见 compute_wer），用线程池
    并行下发 + ThreadPoolExecutor.map 按输入顺序收集结果，保证顺序对齐。

    线程安全：whisper 模型实例 decoder 共享 kv_cache/mask 状态，并发 transcribe
    会触发 RuntimeError（tensor 尺寸错配），故对每次 transcribe 加锁串行化模型
    调用。保留线程池结构 + 顺序对齐 + 单条语义不变；whisper 真并行需进程隔离或
    多模型副本（超范围）。eval 主要吞吐收益来自 emotion2vec funasr batch。
    """
    if not wav_paths:
        return []
    workers = max(1, max_workers)
    lock = threading.Lock()

    def _safe(w):
        with lock:
            return compute_wer(whisper_model, w)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_safe, wav_paths))


# ============================================================
# WER 参考文本
# ============================================================

def load_text_manifest(path):
    """读 jsonl manifest，返回 {utt_id: text}。用于 WER 的 ground-truth 参考文本。"""
    import json as _json
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            uid = rec.get("utt_id")
            if uid is not None and rec.get("text"):
                mapping[uid] = rec["text"]
    return mapping


def wer_reference_text(stem, text_map, whisper_model, ref_path):
    """WER 参考文本选择（论文口径：ground-truth 文本 vs 合成音频转写）。

    优先用 manifest 的 ground-truth text（论文定义的可懂度 WER）；
    manifest 缺该条时回退为转写参考音频（旧行为，degraded，仅保冒烟可跑），返回 used_gt=False。
    """
    if stem in text_map:
        return text_map[stem], True
    return compute_wer(whisper_model, ref_path), False


_NUM_WORD_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _normalize_digits(text):
    """简单数字→英文（0-9 单字符）。两位以上数字保持原样（论文未公开规则）。"""
    for d, w in _NUM_WORD_MAP.items():
        text = text.replace(d, w)
    return text


def normalize_text(text):
    """WER normalization (论文未公开, Emo_PA 未实现 WER)。

    规则：lowercase + 去标点 + 数字转英文 + 多空格合并 + strip。
    不做 contraction expansion（don't → do not），因 Whisper 转写很少产出缩写。
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = _normalize_digits(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 批量评测核心
# ============================================================

def build_arg_parser():
    """构建 CLI parser（抽出以便测试 --batch_size / --expected_count 透传）。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--hyp_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--expected_count", type=int, default=None,
                        help="ref/hyp 期望配对数；正式运行必须显式提供，"
                             "与 pair_wavs_strict 配合做 hard-fail 校验")
    parser.add_argument("--dtw_dist", choices=["euclidean", "cosine"], default="euclidean",
                        help="[兼容保留] DTW 距离度量；v2 正式口径固定为 cosine（dtw 字段），"
                             "euclidean 仅作 dtw_euclidean 诊断输出，不再可选")
    parser.add_argument("--ref_text_manifest", type=str, default=None,
                        help="jsonl (含 utt_id/text)，提供则 WER 用 ground-truth 文本作参考"
                             "（论文口径）；缺省则回退为转写参考音频（degraded，仅供冒烟）")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="emotion2vec funasr 批大小 / Whisper 线程池并发数（默认 16）")
    return parser


def run_evaluation(emo_model, whisper_model, ref_dir, hyp_dir, text_map,
                   batch_size=16, wer_fn=None, expected_count=None):
    """v2 批量评测（emofilm-eval-v2 契约），返回 aggregate_metric_rows 聚合 dict（9 字段）。

    与 v1 关键差异：
    1) 严格配对：pair_wavs_strict 按 utt_id 交集校验；空/数量不符/重复 ID → hard-fail。
    2) frame features 只提取一次：Emo-SIM 从 frame features 均值池化
       (compute_frame_mean_emo_sim)；DTW 用 compute_dtw_metrics（cosine 为 dtw 正式口径，
       euclidean 为 dtw_euclidean 诊断）。
    3) 逐条异常立即抛出并携带 utt_id，不再吞 Exception 返回部分均值。
    4) WER: 空 rows → aggregate_metric_rows hard-fail；不再返回 wer=1 默认值。

    wer_fn=None → 延迟 `from jiwer import wer`（生产路径，保持原 WER 实现）。
    """
    # 1) 严格配对（替代 v1 的 sorted + common 静默回退）
    pairs = pair_wavs_strict(ref_dir, hyp_dir, expected_count=expected_count)
    utt_ids = [p[0] for p in pairs]
    ref_paths = [p[1] for p in pairs]
    hyp_paths = [p[2] for p in pairs]

    # 2) frame features 只提取一次（ref/hyp 交织批量提取，结果按 idx*2=ref, idx*2+1=hyp 回填）
    interleaved = []
    for ref_p, hyp_p in zip(ref_paths, hyp_paths):
        interleaved.append(ref_p)
        interleaved.append(hyp_p)
    frame_feats = extract_frame_embeddings(emo_model, interleaved, batch_size=batch_size)

    # 3) Whisper 并行转写所有 hyp + manifest 缺失的 ref（回退路径）
    hyp_texts = transcribe_parallel(whisper_model, hyp_paths, max_workers=batch_size)
    ref_fallback_idx = [i for i, uid in enumerate(utt_ids) if uid not in text_map]
    ref_fallback_paths = [ref_paths[i] for i in ref_fallback_idx]
    ref_fallback_texts = (transcribe_parallel(whisper_model, ref_fallback_paths,
                                              max_workers=batch_size)
                          if ref_fallback_paths else [])

    if wer_fn is None:
        # 生产路径：延迟 import jiwer（缺失则逐样本调用时抛错 → hard-fail 带 utt_id）
        try:
            from jiwer import wer as wer_fn
        except ImportError:
            def wer_fn(_r, _h):
                raise ImportError("jiwer not installed")

    # 4) 逐样本：Emo-SIM 从 frame 均值池化；DTW 用 compute_dtw_metrics；WER 论文口径
    rows = []
    for idx, utt_id in enumerate(utt_ids):
        ref_frame = frame_feats[idx * 2]
        hyp_frame = frame_feats[idx * 2 + 1]
        try:
            emo_sim = compute_frame_mean_emo_sim(ref_frame, hyp_frame)
            dtw_metrics = compute_dtw_metrics(ref_frame, hyp_frame)

            hyp_text = hyp_texts[idx]
            if utt_id in text_map:
                ref_text = text_map[utt_id]
            else:
                pos = ref_fallback_idx.index(idx)
                ref_text = ref_fallback_texts[pos]
            wer = wer_fn(normalize_text(ref_text), normalize_text(hyp_text))
        except Exception as e:
            # v2: 不再吞掉返回部分均值；逐条异常带 utt_id 立即抛出
            raise RuntimeError(f"sample '{utt_id}' failed: {e}") from e

        rows.append({
            "emo_sim": emo_sim,
            "dtw": dtw_metrics["dtw"],
            "dtw_normalized": dtw_metrics["dtw_normalized"],
            "dtw_euclidean": dtw_metrics["dtw_euclidean"],
            "dtw_euclidean_normalized": dtw_metrics["dtw_euclidean_normalized"],
            "wer": float(wer),
        })

    return aggregate_metric_rows(rows)


def main():
    args = build_arg_parser().parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    # emotion2vec_plus_large + whisper-large-v3 显式指定 device
    emo_model = AutoModel(
        model="iic/emotion2vec_plus_large",
        disable_update=True,
        device=device,
    )
    whisper_model = whisper.load_model("large-v3", device=device)

    text_map = load_text_manifest(args.ref_text_manifest) if args.ref_text_manifest else {}
    if not text_map:
        print("WARN: 未提供 --ref_text_manifest，WER 回退为『参考音频转写 vs 合成音频转写』"
              "（非论文可懂度 WER，仅供冒烟）。正式复现请传 manifest。")

    result = run_evaluation(emo_model, whisper_model,
                            args.ref_dir, args.hyp_dir, text_map,
                            batch_size=args.batch_size,
                            expected_count=args.expected_count)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Done. {result}")


if __name__ == "__main__":
    main()
