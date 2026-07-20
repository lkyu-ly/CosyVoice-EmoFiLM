#!/usr/bin/env python3
"""使用 EmoFiLM WordSequenceModel 生成无平滑的词级标签。"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


# 让脚本可作为子进程独立调用。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.build_emofilm_contract import (  # noqa: E402
    classify_text_coverage,
    load_word_sequence_model,
    merge_word_predictions,
)


LABEL_REVERSE = {0: "ang", 1: "hap", 2: "neu", 3: "sad", 4: "sur"}


def arousal_to_intensity(arousal_val):
    if arousal_val > 3.5:
        return "high"
    if arousal_val > 2.5:
        return "medium"
    return "low"


def predict_words(model, word_files, utt_dir, device):
    results = []
    for word_file in word_files:
        data = torch.load(
            os.path.join(utt_dir, word_file),
            map_location=device,
            weights_only=True,
        )
        frames = data["frames"].unsqueeze(0).float().to(device)
        padding_mask = torch.zeros(
            1,
            frames.shape[1],
            dtype=torch.bool,
            device=device,
        )
        with torch.no_grad():
            class_logits, vad_pred = model(frames, padding_mask=padding_mask)

        pred_idx = int(class_logits.argmax(dim=1).item())
        vad_scaled = vad_pred.squeeze(0).cpu() * 4.0 + 1.0
        arousal = float(vad_scaled[1].item())
        results.append(
            {
                "word": data["word"],
                "predicted_emotion": LABEL_REVERSE.get(pred_idx, "neu"),
                "predicted_arousal": arousal,
                "predicted_intensity": arousal_to_intensity(arousal),
            }
        )
    return results


def generate_tagged_dataset(
    *,
    data_dir: Path,
    manifest_path: Path,
    checkpoint: Path,
    output_jsonl: Path,
    rejected_jsonl: Path,
    device: str,
    existing_rejected_jsonl: Path | None = None,
) -> dict[str, int]:
    model = load_word_sequence_model(checkpoint, device=device)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing_rejected = []
    if existing_rejected_jsonl is not None and existing_rejected_jsonl.is_file():
        with existing_rejected_jsonl.open(encoding="utf-8") as handle:
            existing_rejected = [json.loads(line) for line in handle if line.strip()]
    excluded_ids = {str(row["utt_id"]) for row in existing_rejected}
    kept = 0
    rejected_count = len(existing_rejected)
    with (
        output_jsonl.open("w", encoding="utf-8") as output,
        rejected_jsonl.open("w", encoding="utf-8") as rejected_output,
    ):
        for row in existing_rejected:
            rejected_output.write(json.dumps(row, ensure_ascii=False) + "\n")
        for sample in manifest:
            utt_id = sample["utt_id"]
            if utt_id in excluded_ids:
                continue
            utt_dir = data_dir / utt_id
            if not utt_dir.is_dir():
                raise FileNotFoundError(f"missing word blocks for {utt_id}: {utt_dir}")
            word_files = sorted(path.name for path in utt_dir.glob("*.pt"))
            if not word_files:
                raise ValueError(f"empty word blocks for {utt_id}: {utt_dir}")

            predictions = predict_words(model, word_files, utt_dir, device)
            tagged = merge_word_predictions(predictions)
            coverage = classify_text_coverage(str(sample.get("text", "")), tagged)
            if coverage["decision"] == "reject":
                rejected_row = dict(sample)
                rejected_row.update(
                    {
                        "reason": str(coverage["category"]),
                        "reason_details": coverage,
                        "original_split": sample.get("original_split", ""),
                    }
                )
                rejected_output.write(json.dumps(rejected_row, ensure_ascii=False) + "\n")
                rejected_count += 1
                continue

            output.write(
                json.dumps(
                    {
                        "utt_id": utt_id,
                        "wav_path": sample.get("wav_path", ""),
                        "text": tagged,
                        "plain_text": sample.get("text", ""),
                        "speaker_id": sample.get("speaker_id", ""),
                        "source_dataset": sample.get("source_dataset", "iemocap"),
                        "label_role": "supervision",
                        "label_source": "word_annotator_pseudo_label",
                        "intensity_policy": "predicted_arousal",
                        "granularity": "word",
                        "text_coverage": str(coverage["category"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    return {
        "kept": kept,
        "rejected": rejected_count,
        "total": len(manifest),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate EmoFiLM word-level emotion tags"
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--rejected_jsonl", type=Path)
    parser.add_argument("--existing_rejected_jsonl", type=Path)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main():
    args = build_parser().parse_args()
    rejected_jsonl = args.rejected_jsonl or args.output_jsonl.with_name("rejected.jsonl")
    report = generate_tagged_dataset(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output_jsonl=args.output_jsonl,
        rejected_jsonl=rejected_jsonl,
        device=args.device,
        existing_rejected_jsonl=args.existing_rejected_jsonl,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
