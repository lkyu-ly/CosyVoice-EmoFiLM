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
    """读 jsonl manifest，返回 {utt_id: {"text","speaker_id","emotion","reference_wav"}}。"""
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
                "reference_wav": (os.path.abspath(rec["reference_wav"])
                                  if rec.get("reference_wav") else None),
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
    parser.add_argument("--whisper_device", type=str, default=None,
                        help="whisper 加载设备；默认同 --device。共享 GPU 饱和时可设 cpu "
                             "以把 whisper 移出 GPU（emotion2vec 仍在 --device），降峰值。")
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
    whisper_device = args.whisper_device or device
    emo_model = AutoModel(model="iic/emotion2vec_plus_large",
                          disable_update=True, device=device)
    whisper_model = whisper.load_model("large-v3", device=whisper_device)

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
