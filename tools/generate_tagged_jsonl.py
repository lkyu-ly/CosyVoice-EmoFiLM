#!/usr/bin/env python3
"""合并连续同标签词 + arousal 分桶 → 带 <emotion> 标签的 JSONL。

分桶阈值 (Emo_PA): arousal > 3.5 → high, > 2.5 → medium, else → low.
VAD 顺序: [valence, arousal, dominance]，取 vad[1] 作为 arousal。
VAD 缩放: sigmoid 输出 [0,1] × 4 + 1 → [1,5]。
"""
import argparse
import json
import os
import sys
import torch

# 让脚本可作为子进程独立调用
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cosyvoice_emo.emo_annotator import WordSequenceModel


LABEL_REVERSE = {0: "ang", 1: "hap", 2: "neu", 3: "sad", 4: "sur"}


def arousal_to_intensity(arousal_val):
    if arousal_val > 3.5:
        return "high"
    elif arousal_val > 2.5:
        return "medium"
    else:
        return "low"


def merge_continuous_tags(words_with_predictions):
    """合并连续同 emotion 词为一个 tag 段。

    合并粒度按 emotion（intensity 在段内取平均 arousal 后重新分桶）。
    这样即使 arousal 数值有抖动，连续相同 emotion 的词仍能合并，
    符合 spec 8.4 (a) "连续相同 emotion/intensity 合并" 的语义：
    intensity 由段内代表值统一化，而非逐词判定。

    Args:
        words_with_predictions: list[dict], 每项 {"word": str, "predicted_emotion": str, "predicted_arousal": float}
    Returns:
        str: tagged text
    """
    if not words_with_predictions:
        return ""

    segments = []
    cur_emo = None
    cur_words = []
    cur_arousals = []

    for wp in words_with_predictions:
        emo = wp["predicted_emotion"]
        word = wp["word"]
        arousal = wp["predicted_arousal"]

        if emo == cur_emo:
            cur_words.append(word)
            cur_arousals.append(arousal)
        else:
            if cur_words:
                inner = " ".join(cur_words)
                avg_arousal = sum(cur_arousals) / len(cur_arousals)
                intensity = arousal_to_intensity(avg_arousal)
                segments.append(f"<emotion type='{cur_emo}' intensity='{intensity}'>{inner}</emotion>")
            cur_emo = emo
            cur_words = [word]
            cur_arousals = [arousal]

    if cur_words:
        inner = " ".join(cur_words)
        avg_arousal = sum(cur_arousals) / len(cur_arousals)
        intensity = arousal_to_intensity(avg_arousal)
        segments.append(f"<emotion type='{cur_emo}' intensity='{intensity}'>{inner}</emotion>")

    return " ".join(segments)


def smooth_labels(words, window=3):
    """孤立单词标签按邻域多数投票平滑 + 强度数值短窗口均值平滑。

    spec 8.4 (b)(c) 完整要求：
    - (b) 孤立单词 emotion 标签按邻域多数投票平滑（窗口=3）
    - (c) 强度（arousal 数值）做短窗口均值平滑后再分桶合并
    两个平滑都必须在 merge_continuous_tags 之前完成，否则 arousal 数值抖动
    会让"连续相同 intensity"几乎不可能成立，使合并退化为每词一个 tag。
    """
    if len(words) <= 2:
        return words
    from collections import Counter
    smoothed = [dict(w) for w in words]
    half = window // 2
    for i in range(len(words)):
        left = max(0, i - half)
        right = min(len(words), i + half + 1)
        # (b) emotion 多数投票
        neighbor_emos = [words[j]["predicted_emotion"] for j in range(left, right)]
        smoothed[i]["predicted_emotion"] = Counter(neighbor_emos).most_common(1)[0][0]
        # (c) arousal 数值窗口均值平滑
        neighbor_arousals = [words[j]["predicted_arousal"] for j in range(left, right)]
        smoothed[i]["predicted_arousal"] = sum(neighbor_arousals) / len(neighbor_arousals)
    return smoothed


def predict_words(model, word_files, utt_dir, device):
    results = []
    for wf in word_files:
        data = torch.load(os.path.join(utt_dir, wf), map_location=device)
        frames = data["frames"].unsqueeze(0).float().to(device)
        mask = torch.zeros(1, frames.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad():
            class_logits, vad_pred = model(frames, padding_mask=mask)
        pred_idx = int(class_logits.argmax(dim=1).item())
        vad_scaled = vad_pred.squeeze(0).cpu().numpy() * 4.0 + 1.0
        results.append({
            "word": data["word"],
            "predicted_emotion": LABEL_REVERSE.get(pred_idx, "neu"),
            "predicted_arousal": float(vad_scaled[1]),  # arousal is index 1
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    model.eval()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f]

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for s in manifest:
            utt_id = s["utt_id"]
            utt_dir = os.path.join(args.data_dir, utt_id)
            if not os.path.isdir(utt_dir):
                continue
            word_files = sorted(os.listdir(utt_dir))
            if not word_files:
                continue

            words = predict_words(model, word_files, utt_dir, args.device)
            if not args.no_smooth:
                words = smooth_labels(words)

            tagged = merge_continuous_tags(words)

            fout.write(json.dumps({
                "audio_filepath": s.get("wav_path", ""),
                "text": tagged,
                "speaker_id": s.get("speaker_id", ""),
            }, ensure_ascii=False) + "\n")

    print(f"Done. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
