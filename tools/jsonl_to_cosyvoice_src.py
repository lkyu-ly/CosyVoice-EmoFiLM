#!/usr/bin/env python3
"""tagged jsonl → CosyVoice 原生 src_dir 格式转换器。

CosyVoice make_parquet_list.py 期望 src_dir 含（每行 "utt_id value"，utt_id 作 key）：
- wav.scp:   "utt_id wav_path"
- text:      "utt_id tagged_text 或 plain_text"（下游 emo_tokenizer 解析 <emotion> 标签）
- utt2spk:   "utt_id speaker_id"
- instruct:  "utt_id instruct_text"（可选；zero-shot prompt 段）

entries schema = generate_tagged_jsonl.py 实际输出：
  {utt_id, audio_filepath(=wav_path), text(tagged, <emotion>标签), plain_text(原), speaker_id, 可选 instruct}

用法:
  python tools/jsonl_to_cosyvoice_src.py \\
    --input data/contracts/emofilm_v1/splits/train/manifest.jsonl \\
    --src_dir data/contracts/emofilm_v1/splits/train/src/ \\
    --use_tagged_text
"""
import argparse
import json
import os


def write_src_dir(src_dir, entries, use_tagged_text=True, write_instruct=False):
    """把 tagged jsonl 条目写入 CosyVoice src_dir 格式。

    Args:
        src_dir: 目标目录（如 data/contracts/emofilm_v1/splits/train/src/）。
        entries: list of tagged jsonl dict（generate_tagged_jsonl.py 输出）。
        use_tagged_text: True 用 text（含 <emotion> 词级标签）；False 用 plain_text（消融 no_word）。
        write_instruct: True 且 entry 含 instruct 字段时，额外写 instruct 文件。
    """
    os.makedirs(src_dir, exist_ok=True)

    wav_lines, text_lines, spk_lines, instruct_lines = [], [], [], []
    has_instruct = False
    for e in entries:
        utt_id = e["utt_id"]
        wav = e.get("audio_filepath") or e.get("wav_path", "")
        spk = e.get("speaker_id", "")
        # 优先 tagged_text 字段（build_esd/build_fedd_tagged_text 产物），回退 text（generate_tagged_jsonl 产物）。
        # 与 inference_emo_film._pick_text 同口径，统一两种 producer schema（ADR-0003）。
        if use_tagged_text:
            text = e.get("tagged_text") or e.get("text", "")
        else:
            text = e.get("plain_text", e.get("text", ""))
        wav_lines.append(f"{utt_id} {wav}")
        text_lines.append(f"{utt_id} {text}")
        spk_lines.append(f"{utt_id} {spk}")
        if write_instruct and "instruct" in e and e["instruct"]:
            instruct_lines.append(f"{utt_id} {e['instruct']}")
            has_instruct = True

    with open(os.path.join(src_dir, "wav.scp"), "w", encoding="utf-8") as f:
        f.write("\n".join(wav_lines) + "\n")
    with open(os.path.join(src_dir, "text"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines) + "\n")
    with open(os.path.join(src_dir, "utt2spk"), "w", encoding="utf-8") as f:
        f.write("\n".join(spk_lines) + "\n")
    if has_instruct:
        with open(os.path.join(src_dir, "instruct"), "w", encoding="utf-8") as f:
            f.write("\n".join(instruct_lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="tagged jsonl（generate_tagged_jsonl.py 输出）")
    ap.add_argument("--src_dir", required=True, help="CosyVoice src_dir 输出目录")
    tagged = ap.add_mutually_exclusive_group()
    tagged.add_argument("--use_tagged_text", dest="use_tagged_text", action="store_true",
                        help="text 用 tagged（默认，含 <emotion> 词级标签）")
    tagged.add_argument("--plain_text", dest="use_tagged_text", action="store_false",
                        help="text 用 plain_text（消融 no_word：无词级标签）")
    ap.set_defaults(use_tagged_text=True)
    ap.add_argument("--write_instruct", action="store_true",
                    help="写 instruct 文件（需 entry 含 instruct 字段）")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]
    write_src_dir(args.src_dir, entries,
                  use_tagged_text=args.use_tagged_text, write_instruct=args.write_instruct)
    print(f"Wrote src_dir -> {args.src_dir} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
