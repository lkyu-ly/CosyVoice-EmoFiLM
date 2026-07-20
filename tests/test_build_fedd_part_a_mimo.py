"""FEDD Part A MiMo-V2.5-TTS 指令 TTS 合成器测试。

不触真网络：注入假 http_post；不依赖 ffmpeg：monkeypatch _convert_audio_to_wav。
"""
import base64
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


PROMPTS = [
    {"prompt_id": "pa_ang2hap_000", "text": "Test sentence one.",
     "emo_from": "ang", "emo_to": "hap", "voice": "Dean",
     "tts_instructions": "Start angry, gradually become happy."},
    {"prompt_id": "pa_sad2sur_000", "text": "Test sentence two.",
     "emo_from": "sad", "emo_to": "sur", "voice": "Mia",
     "tts_instructions": "Start sad, gradually become surprised."},
]


def _patch_convert(mod, monkeypatch):
    def fake_convert(audio_bytes, wav_path, sr=16000):
        import numpy as np
        import soundfile as sf
        sf.write(wav_path, np.zeros(sr, dtype="float32"), sr)
    monkeypatch.setattr(mod, "_convert_audio_to_wav", fake_convert)


# ── Task 1: 配置 / 请求体 / synth_one ──────────────────────────────

def test_build_payload_two_stage_messages():
    from tools.build_fedd_part_a_mimo import MiMoConfig, build_payload
    cfg = MiMoConfig(api_key="k")
    pl = build_payload("Hello world.", "Chloe", "start angry, end happy", cfg)
    assert pl["model"] == "mimo-v2.5-tts"
    assert pl["messages"][0] == {"role": "user", "content": "start angry, end happy"}
    assert pl["messages"][1] == {"role": "assistant", "content": "Hello world."}
    assert pl["audio"]["voice"] == "Chloe"
    assert pl["audio"]["format"] == "wav"


def test_synth_one_decodes_base64_and_sends_key():
    from tools.build_fedd_part_a_mimo import MiMoConfig, synth_one
    cfg = MiMoConfig(api_key="secret-key")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        return _FakeResp(_ok_payload(b"ABCD"))

    audio = synth_one("Hi.", "Dean", "be sad", cfg, http_post=fake_post)
    assert audio == b"ABCD"
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["api-key"] == "secret-key"
    assert seen["json"]["messages"][0]["content"] == "be sad"
    assert seen["json"]["messages"][1]["content"] == "Hi."
    assert seen["json"]["audio"]["voice"] == "Dean"


def test_synth_one_raises_on_missing_audio():
    from tools.build_fedd_part_a_mimo import MiMoConfig, synth_one
    cfg = MiMoConfig(api_key="k")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp({"error": {"message": "quota exceeded"}})

    with pytest.raises(RuntimeError, match="quota exceeded"):
        synth_one("Hi.", "Dean", "be sad", cfg, http_post=fake_post)


# ── Task 2: 并发循环 / manifest / 失败重试 ─────────────────────────

def test_concurrent_manifest_and_instruction(tmp_path, monkeypatch):
    import tools.build_fedd_part_a_mimo as mod
    _patch_convert(mod, monkeypatch)
    from tools.build_fedd_part_a_mimo import MiMoConfig

    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        # 以 assistant 文本为键记录该条实际下发的 payload
        seen[json["messages"][1]["content"]] = json
        return _FakeResp(_ok_payload(b"RIFF\x00\x00"))

    out = str(tmp_path / "wav")
    entries = mod.generate_part_a_mimo(
        prompts=PROMPTS, output_dir=out, cfg=MiMoConfig(api_key="k"),
        num_samples=2, concurrency=2, http_post=fake_post,
    )
    assert len(entries) == 2
    for p in PROMPTS:
        body = seen[p["text"]]
        assert body["audio"]["voice"] == p["voice"]
        assert body["messages"][0]["content"] == p["tts_instructions"]
    by_id = {e["utt_id"]: e for e in entries}
    for p in PROMPTS:
        e = by_id[f"fedd_a_{p['prompt_id']}"]
        assert e["source"] == "mimo_api" and e["part"] == "A" and e["level"] == "mild"
        assert e["speaker_id"] == p["voice"]
        assert e["emo_from"] == p["emo_from"] and e["emo_to"] == p["emo_to"]
        assert os.path.isfile(e["wav_path"])
    expected = {"utt_id", "wav_path", "text", "emo_from", "emo_to",
                "speaker_id", "source", "part", "level", "model"}
    assert expected.issubset(entries[0].keys())


def test_failed_prompts_logged(tmp_path, monkeypatch):
    import tools.build_fedd_part_a_mimo as mod
    _patch_convert(mod, monkeypatch)
    monkeypatch.setattr(mod, "RETRY_SLEEP_S", 0)
    from tools.build_fedd_part_a_mimo import MiMoConfig

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResp({"error": {"message": "rate limit"}})

    out = str(tmp_path / "wav")
    entries = mod.generate_part_a_mimo(
        prompts=PROMPTS[:1], output_dir=out, cfg=MiMoConfig(api_key="k"),
        num_samples=1, concurrency=1, failed_log=str(tmp_path / "failed.jsonl"),
        http_post=fake_post,
    )
    assert entries == []
    assert os.path.isfile(tmp_path / "failed.jsonl")
    assert (not os.listdir(out)) if os.path.isdir(out) else True
