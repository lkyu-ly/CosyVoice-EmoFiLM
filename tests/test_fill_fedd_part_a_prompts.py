"""fill_fedd_part_a_prompts 测试。

验证：Part A 按 voice 填 anchor；Part B 按 speaker 填 ESD Neutral 真实转写；
计数不变；备份生成；幂等。
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


ANCHORS = [
    {"voice": "Mia", "prompt_wav": "/anchors/Mia.wav", "prompt_text": "calm text",
     "prompt_source": "mimo_same_voice_neutral_anchor", "instruction": "x"},
    {"voice": "Chloe", "prompt_wav": "/anchors/Chloe.wav", "prompt_text": "calm text",
     "prompt_source": "mimo_same_voice_neutral_anchor", "instruction": "x"},
]

ESD = [
    {"utt_id": "0011_000001", "wav_path": "/ESD/0011/Neutral/0011_000001.wav",
     "text": "The nine the eggs, I keep.", "sentence_emotion": "neu", "speaker_id": "0011"},
    {"utt_id": "0011_000002", "wav_path": "/ESD/0011/Neutral/0011_000002.wav",
     "text": "other.", "sentence_emotion": "neu", "speaker_id": "0011"},
]

MANIFEST = [
    {"utt_id": "fedd_a_x1", "speaker_id": "Mia", "part": "A",
     "text": "t1", "wav_path": "/w/a1.wav"},
    {"utt_id": "fedd_b_x1", "speaker_id": "0011", "part": "B",
     "text": "tb", "wav_path": "/w/b1.wav"},
]


def test_fill_part_a_anchor_and_part_b_esd_neutral(tmp_path):
    import tools.fill_fedd_part_a_prompts as mod

    anchor = tmp_path / "anchor_manifest.jsonl"
    esd = tmp_path / "esd_train.jsonl"
    man = tmp_path / "manifest.jsonl"
    bak = str(tmp_path / "manifest.jsonl.bak_pre_anchor")
    _write(anchor, ANCHORS)
    _write(esd, ESD)
    _write(man, MANIFEST)

    mod.fill_manifest(
        manifest_path=str(man),
        anchor_manifest_path=str(anchor),
        esd_manifest_paths=[str(esd)],
        backup_path=bak,
    )

    rows = _read(man)
    a = [r for r in rows if r["part"] == "A"][0]
    b = [r for r in rows if r["part"] == "B"][0]
    # Part A: anchor by voice
    assert a["prompt_wav"] == "/anchors/Mia.wav"
    assert a["prompt_text"] == "calm text"
    assert a["prompt_source"] == "mimo_same_voice_neutral_anchor"
    # Part B: ESD same-speaker Neutral real transcript
    assert b["prompt_wav"] == "/ESD/0011/Neutral/0011_000001.wav"
    assert b["prompt_text"] == "The nine the eggs, I keep."
    assert b["prompt_source"] == "esd_same_speaker_neutral"
    # backup created, count preserved
    assert os.path.isfile(bak)
    assert len(rows) == len(MANIFEST)
    # target text/wav_path/utt_id 未被破坏
    assert a["utt_id"] == "fedd_a_x1" and a["wav_path"] == "/w/a1.wav" and a["text"] == "t1"
    assert b["utt_id"] == "fedd_b_x1" and b["wav_path"] == "/w/b1.wav" and b["text"] == "tb"


def test_fill_is_idempotent(tmp_path):
    import tools.fill_fedd_part_a_prompts as mod

    anchor = tmp_path / "anchor_manifest.jsonl"
    esd = tmp_path / "esd_train.jsonl"
    man = tmp_path / "manifest.jsonl"
    bak = str(tmp_path / "manifest.jsonl.bak_pre_anchor")
    _write(anchor, ANCHORS)
    _write(esd, ESD)
    _write(man, MANIFEST)

    mod.fill_manifest(str(man), str(anchor), [str(esd)], bak)
    first = _read(man)
    # 第二次：备份已存在，不应覆盖原始备份；输出应一致
    mod.fill_manifest(str(man), str(anchor), [str(esd)], bak)
    second = _read(man)
    assert first == second


def test_fill_raises_if_esd_neutral_missing_for_part_b(tmp_path):
    import tools.fill_fedd_part_a_prompts as mod

    anchor = tmp_path / "anchor_manifest.jsonl"
    esd = tmp_path / "esd_train.jsonl"
    man = tmp_path / "manifest.jsonl"
    bak = str(tmp_path / "manifest.jsonl.bak_pre_anchor")
    _write(anchor, ANCHORS)
    _write(esd, [])  # 无 ESD 记录 → Part B speaker 0011 查不到
    _write(man, MANIFEST)

    with pytest.raises(RuntimeError, match="0011"):
        mod.fill_manifest(str(man), str(anchor), [str(esd)], bak)
