"""作者 768d/50Hz 词级 frame block 合同测试。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]


def _write_textgrid(path: Path, intervals):
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        "xmax = 2.0",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "words"',
        "        xmin = 0",
        "        xmax = 2.0",
        f"        intervals: size = {len(intervals)}",
    ]
    for index, (word, start, end) in enumerate(intervals, 1):
        lines.extend(
            [
                f"            intervals [{index}]:",
                f"                xmin = {start}",
                f"                xmax = {end}",
                f'                text = "{word}"',
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(path: Path, utt_id="utt-1", split="train", text="hello world"):
    path.write_text(
        json.dumps(
            {
                "utt_id": utt_id,
                "wav_path": f"/tmp/{utt_id}.wav",
                "text": text,
                "speaker_id": "spk-a",
                "sentence_emotion": "ang",
                "original_split": split,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_frame_artifact(path: Path, dim=768):
    torch.save(
        {
            "feats": torch.zeros(100, dim),
            "frame_rate_hz": 50.0,
            "frame_step_ms": 20.0,
        },
        path,
    )


def test_word_block_builder_cli_starts_from_script_path():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_word_emo_dataset.py"), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--boundary_manifest" in result.stdout


def test_word_blocks_keep_emofilm_frame_contract(tmp_path):
    from tools.build_word_emo_dataset import build_word_blocks

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    features_dir = tmp_path / "features"
    textgrid_dir = tmp_path / "textgrid"
    output_dir = tmp_path / "blocks"
    rejected_path = tmp_path / "rejected.jsonl"
    features_dir.mkdir()
    textgrid_dir.mkdir()
    _write_frame_artifact(features_dir / "utt-1.pt")
    _write_textgrid(textgrid_dir / "utt-1.TextGrid", [("hello", 0.0, 0.5), ("world", 0.6, 1.0)])

    report = build_word_blocks(
        manifest,
        features_dir,
        textgrid_dir,
        output_dir,
        dataset="iemocap",
        split="train",
        rejected_manifest=rejected_path,
    )

    assert report["done"] == 1
    assert report["rejected"] == []
    blocks = sorted((output_dir / "utt-1").glob("*.pt"))
    assert len(blocks) == 2
    for block_path in blocks:
        block = torch.load(block_path, map_location="cpu", weights_only=True)
        assert block["frames"].shape[1] == 768
        assert block["frame_rate_hz"] == 50.0
        assert block["frame_step_ms"] == 20.0


def test_missing_training_alignment_is_rejected_with_reason(tmp_path):
    from tools.build_word_emo_dataset import build_word_blocks

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    features_dir = tmp_path / "features"
    textgrid_dir = tmp_path / "textgrid"
    output_dir = tmp_path / "blocks"
    rejected_path = tmp_path / "rejected.jsonl"
    features_dir.mkdir()
    textgrid_dir.mkdir()
    _write_frame_artifact(features_dir / "utt-1.pt")

    report = build_word_blocks(
        manifest,
        features_dir,
        textgrid_dir,
        output_dir,
        dataset="iemocap",
        split="train",
        rejected_manifest=rejected_path,
    )

    assert report["done"] == 0
    assert report["rejected"][0]["reason"] == "missing_textgrid"
    row = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert row["utt_id"] == "utt-1"
    assert row["original_split"] == "train"


def test_eval_alignment_failure_is_hard_error(tmp_path):
    from tools.build_word_emo_dataset import build_word_blocks

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, split="test")
    with pytest.raises(RuntimeError, match="hard-fail"):
        build_word_blocks(
            manifest,
            tmp_path / "features",
            tmp_path / "textgrid",
            tmp_path / "blocks",
            dataset="esd",
            split="test",
            rejected_manifest=tmp_path / "rejected.jsonl",
        )


def test_empty_word_interval_rejects_whole_iemocap_utterance_without_partial_blocks(tmp_path):
    from tools.build_word_emo_dataset import build_word_blocks

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    features_dir = tmp_path / "features"
    textgrid_dir = tmp_path / "textgrid"
    output_dir = tmp_path / "blocks"
    rejected_path = tmp_path / "rejected.jsonl"
    boundary_path = tmp_path / "boundary.jsonl"
    features_dir.mkdir()
    textgrid_dir.mkdir()
    _write_frame_artifact(features_dir / "utt-1.pt")
    _write_textgrid(
        textgrid_dir / "utt-1.TextGrid",
        [("hello", 0.0, 1.0), ("world", 2.0, 2.01)],
    )

    report = build_word_blocks(
        manifest,
        features_dir,
        textgrid_dir,
        output_dir,
        dataset="iemocap",
        split="train",
        rejected_manifest=rejected_path,
        boundary_manifest=boundary_path,
    )

    assert report["done"] == 0
    assert report["rejected"][0]["reason"] == "empty_word_interval:world"
    assert not (output_dir / "utt-1").exists()
    event = json.loads(boundary_path.read_text())
    assert event["event"] == "tail_empty"
    assert event["disposition"] == "rejected"
    assert event["rejection_reason"] == "empty_word_interval:world"


def test_tail_clipping_is_kept_and_recorded(tmp_path):
    from tools.build_word_emo_dataset import build_word_blocks

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, text="hello")
    features_dir = tmp_path / "features"
    textgrid_dir = tmp_path / "textgrid"
    output_dir = tmp_path / "blocks"
    boundary_path = tmp_path / "boundary.jsonl"
    features_dir.mkdir()
    textgrid_dir.mkdir()
    _write_frame_artifact(features_dir / "utt-1.pt")
    _write_textgrid(textgrid_dir / "utt-1.TextGrid", [("hello", 1.9, 2.01)])

    report = build_word_blocks(
        manifest,
        features_dir,
        textgrid_dir,
        output_dir,
        dataset="iemocap",
        split="train",
        rejected_manifest=tmp_path / "rejected.jsonl",
        boundary_manifest=boundary_path,
    )

    assert report["done"] == 1
    assert report["boundary_events"] == 1
    event = json.loads(boundary_path.read_text())
    assert event["event"] == "tail_clipped"
    assert event["word"] == "hello"
    assert event["disposition"] == "kept"


def test_audio_text_mismatch_is_rejected_before_word_blocks_are_published(tmp_path):
    from tools.build_word_emo_dataset import build_word_blocks

    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    features_dir = tmp_path / "features"
    textgrid_dir = tmp_path / "textgrid"
    output_dir = tmp_path / "blocks"
    features_dir.mkdir()
    textgrid_dir.mkdir()
    _write_frame_artifact(features_dir / "utt-1.pt")
    _write_textgrid(textgrid_dir / "utt-1.TextGrid", [("hello", 1.9, 2.01)])

    report = build_word_blocks(
        manifest,
        features_dir,
        textgrid_dir,
        output_dir,
        dataset="iemocap",
        split="train",
        rejected_manifest=tmp_path / "rejected.jsonl",
        boundary_manifest=tmp_path / "boundary.jsonl",
    )

    assert report["done"] == 0
    assert report["rejected"][0]["reason"] == "audio_text_mismatch"
    assert report["rejected"][0]["reason_details"]["missing_from_tagged"] == ["world"]
    assert not (output_dir / "utt-1").exists()
    event = json.loads((tmp_path / "boundary.jsonl").read_text())
    assert event["event"] == "tail_clipped"
    assert event["disposition"] == "rejected"
    assert event["rejection_reason"] == "audio_text_mismatch"
