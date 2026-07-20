#!/usr/bin/env python3
"""为 ESD test manifest 生成句级 tagged_text（论文表2-3 ESD "Global Label" 条件）。

论文 ESD 主结果在每句显式给出全局情感标签驱动合成。此脚本把整句用 sentence_emotion
包成 <emotion type='X' intensity='Y'>text</emotion>，写入 tagged_text 字段，供推理注入句级情感。
不注入标签则推理回退纯文本（默认 neu/low），ESD 复现无效——这是本脚本要解决的 blocking。

用法:
  python tools/build_esd_tagged_text.py \
    --manifest data/raw_manifests/esd_test.jsonl \
    --output data/raw_manifests/esd_test_tagged.jsonl \
    --intensity medium
"""
import argparse
import json

EMOTIONS = ["ang", "hap", "neu", "sad", "sur"]
INTENSITIES = ["low", "medium", "high"]


def sentence_tagged_text(text, emotion, intensity="medium"):
    """整句包裹为单个情感标签段。emotion 必须 ∈ EMOTIONS，intensity ∈ INTENSITIES。"""
    if emotion not in EMOTIONS:
        raise ValueError(f"emotion {emotion!r} 不在 {EMOTIONS}")
    if intensity not in INTENSITIES:
        raise ValueError(f"intensity {intensity!r} 不在 {INTENSITIES}")
    return f"<emotion type='{emotion}' intensity='{intensity}'>{text}</emotion>"


def build(records, intensity="medium"):
    """给每条记录添加 tagged_text 字段，返回新列表。

    ESD Global Label 必须用数据集已知 sentence_emotion（ADR-0003 label-provenance 合同）；
    缺失或非法 emotion 时 hard-fail，避免回退到 target-derived 伪标签。
    """
    out = []
    for rec in records:
        rec = dict(rec)
        emo = rec.get("sentence_emotion")
        if emo not in EMOTIONS:
            raise ValueError(
                f"ESD 记录 {rec.get('utt_id')} 的 sentence_emotion={emo!r} 不在 {EMOTIONS}；"
                "ESD Global Label 必须取数据集已知标签（ADR-0003），不得用标注器预测。")
        rec["tagged_text"] = sentence_tagged_text(rec["text"], emo, intensity)
        # 标签来源戳记（audit_label_provenance.py 据此执法）
        rec["label_role"] = "control"
        rec["label_source"] = "dataset_global_label"
        rec["intensity_policy"] = f"fixed_{intensity}"
        rec["granularity"] = "utterance"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--intensity", default="medium", choices=INTENSITIES,
                    help="句级强度（论文 ESD 全局标签未给逐句强度，默认 medium=论文第2章默认档）")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    tagged = build(records, args.intensity)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in tagged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(tagged)} tagged records -> {args.output}")


if __name__ == "__main__":
    main()
