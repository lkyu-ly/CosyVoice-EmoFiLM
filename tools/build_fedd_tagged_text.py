#!/usr/bin/env python3
"""为 FEDD manifest 生成词级 tagged_text（论文表2-3 FEDD "Fine-grained Label" 条件）。

论文 FEDD 主结果由句内词级情感标注驱动，评估词级动态调制能力（本文核心贡献）。
不注入词级标签则推理 --plain_text（默认 neu/low），词级控制未展示——这是本脚本要解决的 blocking。

- Part B（strong）：在 boundary_word_index 处把句子切两段，前段 emo_from、后段 emo_to。
- Part A（mild，若存在）：无词边界，退化为按词数中点两段过渡；更精细可后续用
  generate_tagged_jsonl.py（标注器逐词预测）替代。

用法:
  python tools/build_fedd_tagged_text.py \
    --manifest data/fedd_rebuilt/manifest.jsonl \
    --output data/fedd_rebuilt/manifest_tagged.jsonl \
    --intensity medium
"""
import argparse
import json

EMOTIONS = ["ang", "hap", "neu", "sad", "sur"]
INTENSITIES = ["low", "medium", "high"]


def word_level_tagged_text(text, emo_from, emo_to, boundary_word_index=None, intensity="medium"):
    """把句子按词边界切两段，前段 emo_from、后段 emo_to。

    boundary_word_index=前段词数 k；None 时取词数中点。k 会被 clamp 到 [1, n_words-1]。
    单词句无法切分时退化为单段 emo_from（返回时不抛错，交由调用方决定）。
    """
    for e in (emo_from, emo_to):
        if e not in EMOTIONS:
            raise ValueError(f"emotion {e!r} 不在 {EMOTIONS}")
    if intensity not in INTENSITIES:
        raise ValueError(f"intensity {intensity!r} 不在 {INTENSITIES}")
    words = text.split()
    n = len(words)
    if n <= 1:
        return f"<emotion type='{emo_from}' intensity='{intensity}'>{text}</emotion>"
    k = boundary_word_index if boundary_word_index is not None else n // 2
    k = max(1, min(int(k), n - 1))
    first = " ".join(words[:k])
    second = " ".join(words[k:])
    return (f"<emotion type='{emo_from}' intensity='{intensity}'>{first}</emotion> "
            f"<emotion type='{emo_to}' intensity='{intensity}'>{second}</emotion>")


def build(records, intensity="medium"):
    """给每条记录添加 tagged_text + method 戳记；Part A 无 boundary 时用中点。返回新列表。

    ADR-0003：FEDD Fine-grained 控制必须来自构造已知的 emo_from→emo_to；每条必须有转折
    （emo_from≠emo_to）。Part B 用真实 boundary_word_index（exact_concatenation_boundary），
    Part A 无词边界用词数中点（midpoint_two_span_approximation，工程近似，须如此命名）。
    """
    out = []
    for rec in records:
        rec = dict(rec)
        emo_from = rec.get("emo_from")
        emo_to = rec.get("emo_to")
        if not emo_from or not emo_to:
            raise ValueError(
                f"FEDD 记录 {rec.get('utt_id')} 缺 emo_from/emo_to；"
                "FEDD 控制标签必须来自构造已知转折（ADR-0003）。")
        if emo_from == emo_to:
            raise ValueError(
                f"FEDD 记录 {rec.get('utt_id')} emo_from==emo_to={emo_from}，非转折（ADR-0003）。")
        has_boundary = rec.get("boundary_word_index") is not None
        rec["tagged_text"] = word_level_tagged_text(
            rec["text"], emo_from, emo_to, rec.get("boundary_word_index"), intensity)
        rec["method"] = "exact_concatenation_boundary" if has_boundary else "midpoint_two_span_approximation"
        rec["label_role"] = "control"
        rec["label_source"] = "construction_known_transition"
        rec["intensity_policy"] = f"fixed_{intensity}"
        rec["granularity"] = "word"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--intensity", default="medium", choices=INTENSITIES)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    tagged = build(records, args.intensity)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in tagged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(tagged)} tagged records -> {args.output}")


if __name__ == "__main__":
    main()
