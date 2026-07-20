"""eval_emo_film WER 参考文本选择单测（不调用 whisper）。

验证论文口径修复：WER 参考 = manifest ground-truth 文本，hypothesis = 合成音频转写；
manifest 缺条时回退转写参考音频（degraded）。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import eval_emo_film as ev


def test_load_text_manifest(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps({"utt_id": "a", "text": "Hello world"}) + "\n" +
        json.dumps({"utt_id": "b", "text": "Good morning"}) + "\n" +
        json.dumps({"utt_id": "c", "text": ""}) + "\n",  # 空 text 跳过
        encoding="utf-8",
    )
    m = ev.load_text_manifest(str(p))
    assert m == {"a": "Hello world", "b": "Good morning"}


class _FakeWhisper:
    def transcribe(self, wav_path):
        return {"text": "TRANSCRIBED_" + os.path.basename(wav_path)}


def test_wer_reference_uses_ground_truth():
    """manifest 命中时用 ground-truth 文本、不转写参考音频。"""
    text_map = {"utt1": "the ground truth sentence"}
    ref_text, used_gt = ev.wer_reference_text("utt1", text_map, _FakeWhisper(), "/x/utt1.wav")
    assert used_gt is True
    assert ref_text == "the ground truth sentence"


def test_wer_reference_falls_back_when_missing():
    """manifest 缺条时回退转写参考音频，标记 used_gt=False。"""
    ref_text, used_gt = ev.wer_reference_text("utt_missing", {}, _FakeWhisper(), "/x/utt_missing.wav")
    assert used_gt is False
    assert ref_text == "transcribed_utt_missing.wav"  # compute_wer lower()
