"""作者 emotion2vec-base 帧提取合同测试。"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


def _write_wav(path: Path):
    sf.write(path, np.zeros(3200, dtype=np.float32), 16000)


class _Task:
    class cfg:
        normalize = False


class _Model:
    def __init__(self, feature_dim=768):
        self.feature_dim = feature_dim

    def extract_features(self, source, padding_mask=None):
        return {"x": torch.zeros(1, 10, self.feature_dim)}


def test_parser_requires_emofilm_checkpoint_and_upstream():
    from tools.extract_emotion2vec_frame import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--manifest", "m", "--output_dir", "o"])

    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--checkpoint" in options
    assert "--upstream_dir" in options
    assert "--model_id" not in options


def test_emofilm_frame_extraction_writes_768d_50hz_artifact(tmp_path):
    from tools.extract_emotion2vec_frame import extract_frame_features, save_frame_artifact

    wav_path = tmp_path / "utt.wav"
    artifact_path = tmp_path / "utt.pt"
    _write_wav(wav_path)

    features = extract_frame_features(
        _Model(),
        _Task(),
        wav_path,
        sample_rate=16000,
        device=torch.device("cpu"),
    )
    save_frame_artifact(artifact_path, features)

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert artifact["feats"].shape == (10, 768)
    assert artifact["frame_rate_hz"] == 50.0
    assert artifact["frame_step_ms"] == 20.0


def test_manifest_relative_wav_is_resolved_against_workspace(tmp_path):
    from tools.extract_emotion2vec_frame import resolve_workspace_wav

    workspace = tmp_path / "workspace"
    wav_path = workspace / "datasets" / "IEMOCAP" / "wav" / "utt.wav"
    wav_path.parent.mkdir(parents=True)
    _write_wav(wav_path)

    assert resolve_workspace_wav("datasets/IEMOCAP/wav/utt.wav", workspace) == wav_path
    with pytest.raises(FileNotFoundError):
        resolve_workspace_wav("datasets/IEMOCAP/wav/missing.wav", workspace)


def test_non_768_emofilm_features_are_rejected(tmp_path):
    from tools.extract_emotion2vec_frame import extract_frame_features

    wav_path = tmp_path / "utt.wav"
    _write_wav(wav_path)

    with pytest.raises(ValueError, match="768"):
        extract_frame_features(
            _Model(feature_dim=1024),
            _Task(),
            wav_path,
            sample_rate=16000,
            device=torch.device("cpu"),
        )


def test_provenance_requires_emofilm_model_identity(tmp_path):
    from tools.extract_emotion2vec_frame import validate_provenance

    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "model_id": "emotion2vec-base",
                "revision": "r1",
                "checkpoint_sha256": "a" * 64,
                "upstream": {
                    "path": "fairseq-test",
                    "sha256": "b" * 64,
                    "hash_algorithm": "relative_path_and_bytes_sha256",
                },
            }
        ),
        encoding="utf-8",
    )
    assert validate_provenance(provenance)["model_id"] == "emotion2vec-base"


def test_provenance_verifies_checkpoint_and_upstream_hashes(tmp_path):
    from tools.extract_emotion2vec_frame import (
        directory_sha256,
        file_sha256,
        validate_provenance,
    )

    checkpoint = tmp_path / "emotion2vec_base.pt"
    checkpoint.write_bytes(b"checkpoint")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "models.py").write_bytes(b"validated upstream")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "model_id": "emotion2vec-base",
                "revision": "r1",
                "checkpoint_sha256": file_sha256(checkpoint),
                "upstream": {
                    "path": str(upstream),
                    "sha256": directory_sha256(upstream),
                    "hash_algorithm": "relative_path_and_bytes_sha256",
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_provenance(provenance, checkpoint=checkpoint, upstream_dir=upstream)

    assert result["checkpoint_sha256"] == file_sha256(checkpoint)

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        validate_provenance(provenance, checkpoint=checkpoint, upstream_dir=upstream)


def test_emofilm_dependency_declarations_are_present():
    root = Path(__file__).parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    setup = (root / "setup.py").read_text(encoding="utf-8")

    assert "fairseq==0.12.2" in requirements
    assert "timm==1.0.28" in requirements
    assert '"fairseq==0.12.2"' in setup
    assert '"timm==1.0.28"' in setup


def test_real_emofilm_emotion2vec_loader_smoke():
    from tools.extract_emotion2vec_frame import extract_frame_features, load_emotion2vec

    project_root_value = os.environ.get("EMOFILM_PROJECT_ROOT")
    upstream_value = os.environ.get("EMOFILM_UPSTREAM")
    assert project_root_value, "set EMOFILM_PROJECT_ROOT to the checkout containing the downloaded model"
    assert upstream_value, "set EMOFILM_UPSTREAM to the validated fairseq upstream directory"
    project_root = Path(project_root_value)
    checkpoint = project_root / "pretrained_models/emotion2vec_base/emotion2vec_base.pt"
    upstream = Path(upstream_value)
    wav = project_root / "pretrained_models/emotion2vec_base/example/test.wav"

    assert checkpoint.is_file(), f"missing official checkpoint: {checkpoint}"
    assert upstream.is_dir(), f"missing validated upstream: {upstream}"
    assert wav.is_file(), f"missing smoke wav: {wav}"

    model, task = load_emotion2vec(upstream, checkpoint, torch.device("cpu"))
    features = extract_frame_features(model, task, wav, 16000, torch.device("cpu"))

    assert features.ndim == 2
    assert features.shape[1] == 768
    assert features.shape[0] > 0
    assert torch.isfinite(features).all()
