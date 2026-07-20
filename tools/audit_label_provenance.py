#!/usr/bin/env python3
"""标签来源合同执法器（ADR-0003）。

按 profile 校验一份 tagged manifest（jsonl）是否符合 label-provenance 合同，违反即 hard-fail
（退出码 1）。作为数据重建 → 训练/推理前的 gate 运行。

用法:
  python tools/audit_label_provenance.py --manifest data/contracts/emofilm_v1/eval/esd/manifest.jsonl \\
      --profile esd_control --reference data/contracts/emofilm_v1/eval/esd/manifest.jsonl
  python tools/audit_label_provenance.py --manifest data/fedd_rebuilt/manifest_tagged.jsonl \\
      --profile fedd_control
  python tools/audit_label_provenance.py --manifest data/contracts/emofilm_v1/sources/iemocap/tagged.jsonl \\
      --profile iemocap_supervision
  python tools/audit_label_provenance.py --manifest <no_word_src 的 ESD jsonl> --profile no_word

profile 语义见 ADR-0003 与 CONTEXT.md「标签来源」。
"""
import argparse
import json
import re
import sys

EMOTIONS = {"ang", "hap", "neu", "sad", "sur"}
NON_NEUTRAL = {"ang", "hap", "sad", "sur"}
EMO_TAG_RE = re.compile(r"<emotion type='(\w+)' intensity='(\w+)'>(.*?)</emotion>")


def _load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _spans(text):
    return EMO_TAG_RE.findall(text or "")


def _tagged_text(r):
    """取标签文本：优先 tagged_text 字段（build_esd/build_fedd 产物），回退 text（generate_tagged_jsonl 产物）。
    与 inference_emo_film._pick_text / jsonl_to_cosyvoice_src 同口径（ADR-0003 schema 统一）。"""
    return r.get("tagged_text") or r.get("text", "")


def audit_esd_control(records, reference=None):
    errs = []
    ref_map = {r["utt_id"]: r.get("sentence_emotion") for r in _load(reference)} if reference else {}
    for r in records:
        uid = r.get("utt_id")
        if r.get("label_source") != "dataset_global_label":
            errs.append(f"{uid}: label_source={r.get('label_source')!r}，应为 dataset_global_label")
        if r.get("intensity_policy") != "fixed_medium":
            errs.append(f"{uid}: intensity_policy={r.get('intensity_policy')!r}，应为 fixed_medium")
        if r.get("granularity") != "utterance":
            errs.append(f"{uid}: granularity={r.get('granularity')!r}，应为 utterance")
        spans = _spans(_tagged_text(r))
        if len(spans) != 1:
            errs.append(f"{uid}: ESD Global Label 必须单 span，实际 {len(spans)} 段")
        elif spans[0][0] not in EMOTIONS:
            errs.append(f"{uid}: emotion={spans[0][0]!r} 不在 {sorted(EMOTIONS)}")
        if ref_map:
            gt = ref_map.get(uid)
            pred = spans[0][0] if spans else None
            if gt and pred and gt != pred:
                errs.append(f"{uid}: emotion={pred!r} ≠ 数据集 sentence_emotion={gt!r}")
    if ref_map and len(records) != len(ref_map):
        errs.append(f"条数 {len(records)} ≠ 参考 {len(ref_map)}（ESD test 必须 1500，拒 MFA 丢句）")
    return errs


def audit_iemocap_supervision(records):
    errs = []
    for r in records:
        if r.get("label_source") != "word_annotator_pseudo_label":
            errs.append(f"{r.get('utt_id')}: label_source={r.get('label_source')!r}，应为 word_annotator_pseudo_label")
    return errs


def audit_fedd_control(records):
    errs = []
    for r in records:
        uid = r.get("utt_id")
        if r.get("label_source") != "construction_known_transition":
            errs.append(f"{uid}: label_source={r.get('label_source')!r}，应为 construction_known_transition")
        if r.get("granularity") != "word":
            errs.append(f"{uid}: granularity={r.get('granularity')!r}，应为 word")
        if r.get("method") not in ("exact_concatenation_boundary", "midpoint_two_span_approximation"):
            errs.append(f"{uid}: method={r.get('method')!r} 缺失/非法")
        ef, et = r.get("emo_from"), r.get("emo_to")
        if not ef or not et or ef == et:
            errs.append(f"{uid}: emo_from={ef!r} emo_to={et!r}，须为不同转折两端")
        spans = _spans(_tagged_text(r))
        if len(spans) < 2:
            errs.append(f"{uid}: FEDD Fine-grained 须 ≥2 span（转折），实际 {len(spans)}")
        # emo2vec_label 只能作 metadata，不得进入控制文本
        if "emo2vec_label" in r.get("text", "") or "emo2vec" in (r.get("tagged_text") or ""):
            errs.append(f"{uid}: emo2vec_label 泄漏进控制文本（应仅 metadata）")
    return errs


def audit_no_word(records):
    """no_word = 仅 global-label ESD；非中性情感须覆盖 ang/hap/sad/sur（禁 neu/low 全塌）。"""
    errs = []
    seen_emotions = set()
    for r in records:
        if r.get("label_source") != "dataset_global_label":
            errs.append(f"{r.get('utt_id')}: no_word 须仅含 global-label ESD，实际 label_source={r.get('label_source')!r}")
        for emo, _int, _txt in _spans(_tagged_text(r)):
            seen_emotions.add(emo)
    missing = NON_NEUTRAL - seen_emotions
    if missing:
        errs.append(f"no_word 非中性情感未覆盖 {sorted(missing)}（实际仅 {sorted(seen_emotions)}）；"
                    "数据应为 ESD Global Label 而非 plain text（plain text 会全塌 neu/low）")
    return errs


PROFILES = {
    "esd_control": audit_esd_control,
    "iemocap_supervision": audit_iemocap_supervision,
    "fedd_control": audit_fedd_control,
    "no_word": audit_no_word,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--reference", help="esd_control 时传 raw manifest 校 emotion=count")
    args = ap.parse_args()

    records = _load(args.manifest)
    errs = PROFILES[args.profile](records, reference=args.reference) if args.profile == "esd_control" \
        else PROFILES[args.profile](records)
    print(f"[audit] profile={args.profile} manifest={args.manifest} records={len(records)}")
    if errs:
        print(f"FAIL: {len(errs)} 项违反 ADR-0003 合同：")
        for e in errs[:20]:
            print(f"  - {e}")
        if len(errs) > 20:
            print(f"  ... 还有 {len(errs) - 20} 项")
        sys.exit(1)
    print("PASS: 符合 label-provenance 合同。")


if __name__ == "__main__":
    main()
