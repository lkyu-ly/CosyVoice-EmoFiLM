import os
import sys

ROOT = str(__import__("pathlib").Path(__file__).parents[1])
sys.path.insert(0, ROOT)


def test_build_fedd_tagged_text_preserves_prompt_metadata():
    from tools.build_fedd_tagged_text import build

    records = [{
        "utt_id": "fedd_a_pa_ang2hap_000",
        "text": "demo text",
        "emo_from": "ang",
        "emo_to": "hap",
        "prompt_wav": "/tmp/prompt.wav",
        "prompt_text": "reference prompt text",
        "part": "A",
        "speaker_id": "Mia",
    }]

    tagged = build(records, intensity="medium")

    assert tagged[0]["prompt_wav"] == "/tmp/prompt.wav"
    assert tagged[0]["prompt_text"] == "reference prompt text"
    assert "tagged_text" in tagged[0]
