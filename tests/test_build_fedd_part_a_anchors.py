"""FEDD Part A neutral anchor 生成测试。

不触真网络：注入假 http_post；不依赖 ffmpeg：monkeypatch _convert_audio_to_wav。
"""
import base64
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _ok_payload(audio_bytes=b"RIFF\x00\x00"):
    return {"choices": [{"message": {"audio": {"data": base64.b64encode(audio_bytes).decode()}}}]}


def _patch_convert(mod, monkeypatch):
    def fake_convert(audio_bytes, wav_path, sr=16000):
        import numpy as np
        import soundfile as sf
        sf.write(wav_path, np.zeros(sr, dtype="float32"), sr)
    monkeypatch.setattr(mod, "_convert_audio_to_wav", fake_convert)


def test_generate_anchors_produces_four_with_correct_fields(tmp_path, monkeypatch):
    import tools.build_fedd_part_a_anchors as mod
    _patch_convert(mod, monkeypatch)
    from tools.build_fedd_part_a_mimo import MiMoConfig

    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen[json["audio"]["voice"]] = json
        return _FakeResp(_ok_payload(b"RIFF\x00\x00"))

    out = str(tmp_path / "prompts")
    manifest_path = str(tmp_path / "anchor_manifest.jsonl")
    entries = mod.generate_anchors(
        output_dir=out, cfg=MiMoConfig(api_key="k"),
        http_post=fake_post, anchor_manifest=manifest_path,
    )

    assert len(entries) == 4
    assert set(e["voice"] for e in entries) == {"Mia", "Chloe", "Milo", "Dean"}
    for e in entries:
        assert e["prompt_source"] == "mimo_same_voice_neutral_anchor"
        assert e["prompt_text"] == mod.ANCHOR_TEXT
        assert e["instruction"] == mod.ANCHOR_INSTRUCTION
        assert os.path.isfile(e["prompt_wav"])
        assert e["prompt_wav"].endswith(f"{e['voice']}_neutral_anchor.wav")
    rows = [json.loads(l) for l in open(manifest_path) if l.strip()]
    assert len(rows) == 4
    assert set(r["voice"] for r in rows) == {"Mia", "Chloe", "Milo", "Dean"}
    # 每个 voice 下发的是中性指令 + 中性文本（非 target 的 transition 指令）
    for v in ["Mia", "Chloe", "Milo", "Dean"]:
        assert seen[v]["messages"][0]["content"] == mod.ANCHOR_INSTRUCTION
        assert seen[v]["messages"][1]["content"] == mod.ANCHOR_TEXT
        assert seen[v]["audio"]["voice"] == v
