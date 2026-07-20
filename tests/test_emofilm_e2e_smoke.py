"""三分区 Emo-FiLM 端到端 smoke：ESD、FEDD-A、FEDD-B 各一条。"""

import json

import torch


def test_three_sample_smoke_writes_audio_manifest_and_identity(tmp_path, monkeypatch):
    import tools.inference_emo_film as inference
    from tools.inference_emo_film import run_inference
    from tools.write_emofilm_run_identity import write_run_identity

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"prompt")
    manifest = tmp_path / "eval.jsonl"
    rows = [
        {
            "utt_id": "esd-1",
            "source_dataset": "esd",
            "text": "one",
            "prompt_wav": str(prompt),
            "prompt_text": "reference",
        },
        {
            "utt_id": "fedd-a-1",
            "source_dataset": "fedd_rebuilt",
            "part": "A",
            "text": "two",
            "prompt_wav": str(prompt),
            "prompt_text": "reference",
        },
        {
            "utt_id": "fedd-b-1",
            "source_dataset": "fedd_rebuilt",
            "part": "B",
            "text": "three",
            "prompt_wav": str(prompt),
            "prompt_text": "reference",
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    def fake_save(path, *_args, **_kwargs):
        path = path if hasattr(path, "write_bytes") else __import__("pathlib").Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"wav")

    monkeypatch.setattr(inference.torchaudio, "save", fake_save)

    class _FakeCosyVoice:
        sample_rate = 24000

        def inference_emo_film(self, **kwargs):
            yield {"tts_speech": torch.zeros(1, 8)}

    output_dir = tmp_path / "wav"
    result = run_inference(
        _FakeCosyVoice(), str(manifest), str(tmp_path), str(output_dir)
    )

    contract_dir = tmp_path / "contract"
    provenance = contract_dir / "provenance"
    provenance.mkdir(parents=True)
    for filename, value in (
        ("contract.json", {"contract_name": "emofilm_v1"}),
        ("sources.json", []),
        ("membership.json", {"train": [], "cv": []}),
    ):
        (provenance / filename).write_text(json.dumps(value), encoding="utf-8")
    identity_path = tmp_path / "run_identity.json"
    identity = write_run_identity(
        identity_path,
        run_kind="smoke",
        code_root=tmp_path,
        contract_dir=contract_dir,
        command="synthetic smoke",
    )

    assert len(result) == 3
    assert {row["utt_id"] for row in result} == {"esd-1", "fedd-a-1", "fedd-b-1"}
    assert all((output_dir / f"{row['utt_id']}.wav").is_file() for row in result)
    assert (tmp_path / "inference_wav.jsonl").is_file()
    assert identity["run_kind"] == "smoke"
    assert identity["contract_hash"]
    assert json.loads(identity_path.read_text(encoding="utf-8"))["contract_hash"]
