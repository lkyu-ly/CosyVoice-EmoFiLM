"""fill_esd_test_prompts 测试。

ESD test 输入无 part 字段，按 speaker_id 填 ESD same-speaker Neutral 真实转写。
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


ESD_TRAIN = [
    {"utt_id": "0011_000001", "wav_path": "/ESD/0011/Neutral/0011_000001.wav",
     "text": "The nine the eggs, I keep.", "sentence_emotion": "neu", "speaker_id": "0011"},
    {"utt_id": "0012_000001", "wav_path": "/ESD/0012/Neutral/0012_000001.wav",
     "text": "Second speaker text.", "sentence_emotion": "neu", "speaker_id": "0012"},
]

ESD_TEST = [
    {"utt_id": "0011_000484", "speaker_id": "0011",
     "text": "<emotion type='ang' intensity='medium'>Rare rabbit had a little apron.</emotion>",
     "plain_text": "Rare rabbit had a little apron.", "audio_filepath": "/w/0011_000484.wav"},
    {"utt_id": "0012_000001", "speaker_id": "0012",
     "text": "<emotion type='hap' intensity='medium'>Hello.</emotion>",
     "plain_text": "Hello.", "audio_filepath": "/w/0012_000001.wav"},
]


def test_fill_esd_test_by_speaker(tmp_path):
    import tools.fill_esd_test_prompts as mod

    esd = tmp_path / "esd_train.jsonl"
    test = tmp_path / "esd_test.jsonl"
    bak = str(tmp_path / "esd_test.jsonl.bak")
    _write(esd, ESD_TRAIN)
    _write(test, ESD_TEST)

    mod.fill_esd_test(str(test), [str(esd)], bak)

    rows = _read(test)
    assert rows[0]["prompt_wav"] == "/ESD/0011/Neutral/0011_000001.wav"
    assert rows[0]["prompt_text"] == "The nine the eggs, I keep."
    assert rows[0]["prompt_source"] == "esd_same_speaker_neutral"
    assert rows[1]["prompt_wav"] == "/ESD/0012/Neutral/0012_000001.wav"
    assert rows[1]["prompt_text"] == "Second speaker text."
    # text/plain_text/audio_filepath 未破坏
    assert "<emotion" in rows[0]["text"]
    assert rows[0]["plain_text"] == "Rare rabbit had a little apron."
    assert rows[0]["audio_filepath"] == "/w/0011_000484.wav"
    assert os.path.isfile(bak)


def test_fill_esd_test_raises_on_missing_speaker(tmp_path):
    import tools.fill_esd_test_prompts as mod

    esd = tmp_path / "esd_train.jsonl"
    test = tmp_path / "esd_test.jsonl"
    bak = str(tmp_path / "esd_test.jsonl.bak")
    _write(esd, [])  # 空 → 查不到
    _write(test, ESD_TEST)

    with pytest.raises(RuntimeError, match="0011"):
        mod.fill_esd_test(str(test), [str(esd)], bak)
