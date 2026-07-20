#!/usr/bin/env python3
"""FEDD 全局情感打标 + 一致性校验（emotion2vec-plus-large）。

对齐论文（学位论文 §FEDD）："所有样本均利用 emotion2vec-plus-large 模型进行
特征提取与标签一致性校验"。

- 对 manifest 每条 wav 做 utterance 级情感分类，写回 emo2vec_label / emo2vec_score
- 一致性：top 标签（9 类映射到 5 类）∈ {emo_from, emo_to} 记 pass
- 产出 label_check_report.json（整体 / 分 part 通过率）

性能：funasr AutoModel.generate 支持传 wav 路径**列表** + batch_size 做真 GPU 分批，
并内置 tqdm 进度条。故一次性把待打标 wav 组成列表批量推理（默认 batch_size=32），
充分利用显存，避免逐条调用（每次 batch=1 + 满屏 1/1 小进度条）。

用法:
  MODELSCOPE_OFFLINE=true CUDA_VISIBLE_DEVICES=0 python tools/label_fedd_emotion2vec.py \
    --manifest data/fedd_rebuilt/manifest.jsonl \
    --report data/fedd_rebuilt/label_check_report.json \
    --device cuda --batch_size 32
支持增量：--skip_labeled 跳过已有 emo2vec_label 的条目（Part A 后补跑时用）。
"""
import argparse
import json
import os

# emotion2vec_plus 输出 9 类 "中文/english" 标签；FEDD 只关心 5 类
LABEL_MAP = {
    "angry": "ang",
    "happy": "hap",
    "neutral": "neu",
    "sad": "sad",
    "surprised": "sur",
}


def map_label(raw_label: str) -> str:
    """"生气/angry" → "ang"；不在 5 类内（fearful/disgusted/<unk> 等）→ "other"。"""
    en = raw_label.split("/")[-1].strip().lower()
    return LABEL_MAP.get(en, "other")


def label_manifest(model, entries, skip_labeled=False, batch_size=32):
    """批量推理打标，返回 (new_entries, report)。

    model: funasr AutoModel（或兼容 fake），一次以 wav 路径**列表**调用：
        generate([wav, ...], granularity="utterance", extract_embedding=False,
                 batch_size=N) → [{"labels": [...], "scores": [...]}, ...]（顺序对齐输入）。
    skip_labeled=True 时仅对无 emo2vec_label 的条目推理（Part A 后补跑）；结果按序回填。
    """
    new_entries = [dict(e) for e in entries]

    # 选出需要推理的条目下标，组成一个 wav 列表批量下发
    todo = [i for i, e in enumerate(new_entries)
            if not (skip_labeled and "emo2vec_label" in e)]
    if todo:
        wavs = [new_entries[i]["wav_path"] for i in todo]
        results = model.generate(wavs, granularity="utterance",
                                 extract_embedding=False, batch_size=batch_size)
        for i, res in zip(todo, results):
            labels, scores = res["labels"], res["scores"]
            top = max(range(len(scores)), key=lambda k: scores[k])
            new_entries[i]["emo2vec_label"] = map_label(labels[top])
            new_entries[i]["emo2vec_score"] = float(scores[top])

    # 一致性统计（覆盖全部条目，含 skip 复用的旧标签）
    stats = {}
    for e in new_entries:
        part = e.get("part", "?")
        s = stats.setdefault(part, {"total": 0, "passed": 0})
        s["total"] += 1
        if e["emo2vec_label"] in (e.get("emo_from"), e.get("emo_to")):
            s["passed"] += 1

    total = sum(s["total"] for s in stats.values())
    passed = sum(s["passed"] for s in stats.values())
    report = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_part": {
            p: {**s, "pass_rate": round(s["passed"] / s["total"], 4) if s["total"] else 0.0}
            for p, s in sorted(stats.items())
        },
    }
    return new_entries, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_id", default="iic/emotion2vec_plus_large")
    ap.add_argument("--skip_labeled", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32, help="GPU 批大小（默认 32）")
    args = ap.parse_args()

    entries = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    todo = sum(1 for e in entries if not (args.skip_labeled and "emo2vec_label" in e))
    print(f"[label] manifest={args.manifest}  共 {len(entries)} 条，"
          f"待打标 {todo} 条（skip_labeled={args.skip_labeled}）")
    print(f"[label] 加载 {args.model_id} @ {args.device}，batch_size={args.batch_size} …")

    from funasr import AutoModel
    model = AutoModel(model=args.model_id, disable_update=True, device=args.device)

    new_entries, report = label_manifest(
        model, entries, skip_labeled=args.skip_labeled, batch_size=args.batch_size)

    tmp = args.manifest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, args.manifest)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[label] 完成 → manifest 已回写，报告 → {args.report}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
