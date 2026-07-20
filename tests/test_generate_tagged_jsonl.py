"""EmoFiLM 768d/3D、无平滑词级标签生成测试。"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PYTHON = sys.executable
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/generate_tagged_jsonl.py"


def test_generate_jsonl_format(tmp_path):
    """符合合同 checkpoint 形状的最小数据可以生成带来源戳记的 JSONL。"""
    np.random.seed(42)
    torch.manual_seed(42)

    from cosyvoice_emo.emo_annotator import WordSequenceModel

    model = WordSequenceModel(input_dim=768, num_classes=5, num_heads=8, dropout_rate=0.3, reg_dim=3)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)

    word_blocks = tmp_path / "word_blocks"
    for utt_id in ("utt_1", "utt_2"):
        utt_dir = word_blocks / utt_id
        utt_dir.mkdir(parents=True)
        for word_index in range(4):
            n_frames = np.random.randint(5, 20)
            torch.save(
                {
                    "frames": torch.randn(n_frames, 768),
                    "word": f"word{word_index}",
                    "padding_mask": torch.zeros(n_frames, dtype=torch.bool),
                },
                utt_dir / f"{word_index:04d}_0_{n_frames}.pt",
            )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "utt_id": utt_id,
                    "wav_path": f"/tmp/{utt_id}.wav",
                    "text": "word0 word1 word2 word3",
                    "speaker_id": "0011",
                }
            )
            + "\n"
            for utt_id in ("utt_1", "utt_2")
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tagged.jsonl"
    subprocess.run(
        [
            PYTHON,
            str(SCRIPT),
            f"--data_dir={word_blocks}",
            f"--manifest={manifest}",
            f"--checkpoint={checkpoint}",
            f"--output_jsonl={output}",
            "--device=cpu",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["label_source"] == "word_annotator_pseudo_label"
        assert row["granularity"] == "word"
        assert row["text"]
        assert "<emotion" in row["text"]
        assert "</emotion>" in row["text"]


def test_merge_requires_matching_emotion_and_intensity():
    from tools.build_emofilm_contract import merge_word_predictions

    tagged = merge_word_predictions(
        [
            {"word": "I", "predicted_emotion": "ang", "predicted_intensity": "high"},
            {"word": "am", "predicted_emotion": "ang", "predicted_intensity": "high"},
            {"word": "happy", "predicted_emotion": "hap", "predicted_intensity": "medium"},
            {"word": "today", "predicted_emotion": "hap", "predicted_intensity": "low"},
        ]
    )
    assert tagged.count("<emotion") == 3
    assert "I am" in tagged
    assert "happy</emotion>" in tagged
    assert "today</emotion>" in tagged


def test_arousal_bucketing():
    from tools.generate_tagged_jsonl import arousal_to_intensity

    assert arousal_to_intensity(4.0) == "high"
    assert arousal_to_intensity(3.0) == "medium"
    assert arousal_to_intensity(2.0) == "low"
    assert arousal_to_intensity(1.5) == "low"


def test_text_coverage_accepts_only_exact_or_apostrophe_equivalent_pairs():
    from tools.generate_tagged_jsonl import classify_text_coverage

    assert classify_text_coverage(
        "Everybody's told the story.",
        "<emotion type='neu' intensity='low'>everybody</emotion> "
        "<emotion type='neu' intensity='low'>'s told the story</emotion>",
    )["decision"] == "keep"
    mismatch = classify_text_coverage(
        "Mmhmm. Yeah.",
        "<emotion type='neu' intensity='low'>yeah</emotion>",
    )
    assert mismatch["decision"] == "reject"
    assert mismatch["category"] == "audio_text_mismatch"
    assert mismatch["missing_from_tagged"] == ["mmhmm"]


def test_generate_filters_rejected_rows_without_reintroducing_existing_rejections(tmp_path, monkeypatch):
    from tools import generate_tagged_jsonl as module

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"utt_id": "keep", "text": "Keep me.", "original_split": "train"}),
                json.dumps({"utt_id": "reject", "text": "Missing it.", "original_split": "cv"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "blocks"
    for utt_id in ("keep", "reject"):
        (data_dir / utt_id).mkdir(parents=True)
    torch.save(
        {"frames": torch.zeros(2, 768), "word": "keep", "end_sec": 0.041, "end_frame": 2},
        data_dir / "keep/0000_0_2.pt",
    )
    torch.save(
        {"frames": torch.zeros(2, 768), "word": "me", "end_sec": 0.081, "end_frame": 4},
        data_dir / "keep/0001_2_4.pt",
    )
    torch.save(
        {"frames": torch.zeros(2, 768), "word": "missing", "end_sec": 0.04, "end_frame": 2},
        data_dir / "reject/0000_0_2.pt",
    )

    monkeypatch.setattr(module, "load_word_sequence_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module,
        "predict_words",
        lambda model, word_files, utt_dir, device: [
            {
                "word": torch.load(utt_dir / word_file, weights_only=True)["word"],
                "predicted_emotion": "neu",
                "predicted_intensity": "low",
            }
            for word_file in word_files
        ],
    )

    output = tmp_path / "tagged.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    existing_rejected = tmp_path / "existing_rejected.jsonl"
    existing_rejected.write_text(
        json.dumps({"utt_id": "reject", "reason": "audio_text_mismatch", "original_split": "cv"})
        + "\n",
        encoding="utf-8",
    )
    report = module.generate_tagged_dataset(
        data_dir=data_dir,
        manifest_path=manifest,
        checkpoint=tmp_path / "checkpoint.pt",
        output_jsonl=output,
        rejected_jsonl=rejected,
        device="cpu",
        existing_rejected_jsonl=existing_rejected,
    )

    assert [json.loads(line)["utt_id"] for line in output.read_text().splitlines()] == ["keep"]
    rejected_row = json.loads(rejected.read_text())
    assert rejected_row["utt_id"] == "reject"
    assert rejected_row["original_split"] == "cv"
    assert rejected_row["reason"] == "audio_text_mismatch"
    assert report == {"kept": 1, "rejected": 1, "total": 2}
