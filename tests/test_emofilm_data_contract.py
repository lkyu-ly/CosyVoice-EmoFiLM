"""emofilm_v1 数据合同测试。"""
import json
import os
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORD_SEQUENCE_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "word_sequence_model" / "author_best_model.pth"
)


def test_emofilm_frame_contract_is_768d_50hz(tmp_path):
    from tools.build_emofilm_contract import validate_frame_artifact

    frame_path = tmp_path / "utt.pt"
    provenance_path = tmp_path / "provenance.json"
    torch.save(
        {
            "feats": torch.zeros(4, 768),
            "frame_rate_hz": 50.0,
            "frame_step_ms": 20.0,
        },
        frame_path,
    )
    provenance_path.write_text(
        json.dumps(
            {
                "model_id": "emotion2vec-base",
                "revision": "test-revision",
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

    result = validate_frame_artifact(frame_path, provenance_path)

    assert result["feature_dim"] == 768
    assert result["frame_rate_hz"] == pytest.approx(50.0)
    assert result["frame_step_ms"] == pytest.approx(20.0)


def test_workspace_paths_are_normalized_without_absolute_leak(tmp_path):
    from tools.build_emofilm_contract import normalize_workspace_path

    workspace = tmp_path / "workspace"
    path = workspace / "datasets" / "ESD" / "0011" / "Neutral" / "a.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFF")

    assert normalize_workspace_path(path, workspace) == (
        "datasets/ESD/0011/Neutral/a.wav"
    )
    with pytest.raises(ValueError, match="outside workspace"):
        normalize_workspace_path(tmp_path / "outside.wav", workspace)


def test_normalize_manifest_row_unifies_text_and_source_metadata(tmp_path):
    from tools.build_emofilm_contract import normalize_manifest_row

    workspace = tmp_path / "workspace"
    wav = workspace / "datasets" / "ESD" / "0011" / "Angry" / "a.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"RIFF")
    row = normalize_manifest_row(
        {
            "utt_id": "u1",
            "wav_path": str(wav),
            "text": "Hello.",
            "tagged_text": "<emotion type='ang' intensity='medium'>Hello.</emotion>",
            "speaker_id": "0011",
        },
        dataset="esd",
        workspace_root=workspace,
        label_source="dataset_global_label",
    )

    assert row["wav_path"] == "datasets/ESD/0011/Angry/a.wav"
    assert row["text"] == "Hello."
    assert row["plain_text"] == "Hello."
    assert row["tagged_text"].startswith("<emotion")
    assert row["source_dataset"] == "esd"
    assert row["label_source"] == "dataset_global_label"


def test_write_contract_provenance_uses_frozen_layout(tmp_path):
    from tools.build_emofilm_contract import write_contract_provenance

    report = write_contract_provenance(
        tmp_path / "contract",
        contract={"contract_name": "emofilm_v1"},
        sources=[{"dataset": "iemocap", "count": 1}],
        membership={"train": ["u1"], "cv": ["u2"]},
        artifacts=[{"path": "sources/iemocap/manifest.jsonl", "count": 1}],
    )

    provenance = tmp_path / "contract" / "provenance"
    assert (provenance / "contract.json").is_file()
    assert (provenance / "sources.json").is_file()
    assert (provenance / "membership.json").is_file()
    assert (provenance / "artifacts.jsonl").is_file()
    assert report["contract_name"] == "emofilm_v1"


def test_emofilm_word_sequence_checkpoint_is_768_5_3():
    from tools.build_emofilm_contract import load_word_sequence_model

    checkpoint = Path(
        os.environ.get(
            "EMOFILM_WORD_SEQUENCE_CHECKPOINT",
            DEFAULT_WORD_SEQUENCE_CHECKPOINT,
        )
    )
    assert checkpoint.is_file(), f"missing WordSequence checkpoint: {checkpoint}"

    model = load_word_sequence_model(checkpoint, device="cpu")
    class_logits, vad = model(torch.zeros(1, 2, 768))

    assert class_logits.shape == (1, 5)
    assert vad.shape == (1, 3)


def test_emofilm_word_sequence_state_dict_shapes_match_model_definition():
    from tools.build_emofilm_contract import load_word_sequence_model

    checkpoint = Path(
        os.environ.get(
            "EMOFILM_WORD_SEQUENCE_CHECKPOINT",
            DEFAULT_WORD_SEQUENCE_CHECKPOINT,
        )
    )
    assert checkpoint.is_file(), f"missing WordSequence checkpoint: {checkpoint}"

    state = load_word_sequence_model(checkpoint, device="cpu").state_dict()

    assert state["attention.in_proj_weight"].shape == (2304, 768)
    assert state["ffn.0.weight"].shape == (3072, 768)
    assert state["ffn.3.weight"].shape == (768, 3072)
    assert state["classification_head.weight"].shape == (5, 768)
    assert state["regression_head.0.weight"].shape == (3, 768)


def test_merge_key_is_emotion_and_intensity():
    from tools.build_emofilm_contract import merge_word_predictions

    tagged = merge_word_predictions(
        [
            {"word": "a", "predicted_emotion": "ang", "predicted_intensity": "low"},
            {"word": "b", "predicted_emotion": "ang", "predicted_intensity": "high"},
            {"word": "c", "predicted_emotion": "ang", "predicted_intensity": "high"},
        ]
    )

    assert tagged.count("<emotion") == 2
    assert "a</emotion> <emotion type='ang' intensity='high'>b c" in tagged


def test_no_smoothing_option_exists_in_production_cli():
    from tools.generate_tagged_jsonl import build_parser

    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--majority" not in options
    assert "--smooth" not in options
    assert "--no-smooth" not in options
    source = Path(__file__).parents[1].joinpath("tools", "generate_tagged_jsonl.py").read_text()
    assert "smooth_labels" not in source


def test_train_cv_membership_matches_frozen_ids():
    from tools.build_emofilm_contract import validate_membership

    validate_membership(
        train_ids={"train-a"},
        cv_ids={"cv-a"},
        rejected_ids={"train-rejected", "cv-rejected"},
        frozen_train_ids={"train-a", "train-rejected"},
        frozen_cv_ids={"cv-a", "cv-rejected"},
    )


def test_rejected_iemocap_reports_cleaning_fraction_without_legacy_one_percent_cap():
    from tools.build_emofilm_contract import validate_rejected_manifest

    original = [
        {
            "utt_id": f"utt-{i}",
            "speaker_id": "spk-a" if i % 2 == 0 else "spk-b",
            "sentence_emotion": "ang" if i % 2 == 0 else "hap",
        }
        for i in range(200)
    ]
    rejected = [
        {
            "utt_id": "utt-0",
            "speaker_id": "spk-a",
            "sentence_emotion": "ang",
            "reason": "missing word tier",
            "original_split": "train",
        },
        {
            "utt_id": "utt-1",
            "speaker_id": "spk-b",
            "sentence_emotion": "hap",
            "reason": "empty word tier",
            "original_split": "train",
        },
    ]

    report = validate_rejected_manifest(rejected, original)

    assert report["rejected_count"] == 2
    assert report["fraction"] == pytest.approx(0.01)

    expanded = rejected + [
        {
            "utt_id": "utt-2",
            "speaker_id": "spk-a",
            "sentence_emotion": "ang",
            "reason": "audio_text_mismatch:non_speech_marker",
            "original_split": "cv",
        }
    ]
    expanded_report = validate_rejected_manifest(expanded, original)
    assert expanded_report["fraction"] == pytest.approx(0.015)


def test_rejected_iemocap_concentration_is_rejected():
    from tools.build_emofilm_contract import validate_rejected_manifest

    original = [
        {
            "utt_id": f"utt-{i}",
            "speaker_id": "spk-a" if i < 100 else "spk-b",
            "sentence_emotion": "ang" if i < 100 else "hap",
        }
        for i in range(200)
    ]
    rejected = [
        {
            "utt_id": "utt-0",
            "speaker_id": "spk-a",
            "sentence_emotion": "ang",
            "reason": "missing word tier",
        },
        {
            "utt_id": "utt-1",
            "speaker_id": "spk-a",
            "sentence_emotion": "ang",
            "reason": "empty word tier",
        },
    ]

    with pytest.raises(ValueError, match="concentrated"):
        validate_rejected_manifest(rejected, original)


def test_esd_and_fedd_eval_assets_are_complete(tmp_path):
    from tools.build_emofilm_contract import validate_eval_assets

    rows = []
    for i in range(3):
        files = {}
        for name in ("target_wav", "reference_wav", "prompt_wav"):
            path = tmp_path / f"{name}-{i}.wav"
            path.write_bytes(b"RIFF")
            files[name] = str(path)
        rows.append(
            {
                "utt_id": f"utt-{i}",
                **files,
                "text": "test text",
                "label": "ang",
                "prompt_text": "prompt",
            }
        )

    report = validate_eval_assets(rows, expected_count=3)

    assert report["count"] == 3
    assert report["missing"] == []


def test_eval_assets_resolve_relative_paths_against_workspace(tmp_path):
    from tools.build_emofilm_contract import validate_eval_assets

    workspace = tmp_path / "workspace"
    wav = workspace / "datasets" / "ESD" / "u.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"RIFF")
    rows = [
        {
            "utt_id": "u",
            "target_wav": "datasets/ESD/u.wav",
            "reference_wav": "datasets/ESD/u.wav",
            "prompt_wav": "datasets/ESD/u.wav",
            "text": "text",
            "label": "ang",
            "prompt_text": "prompt",
        }
    ]

    report = validate_eval_assets(rows, expected_count=1, workspace_root=workspace)

    assert report["count"] == 1


def test_build_train_cv_contract_carries_reused_optional_maps(tmp_path):
    from tools.build_emofilm_contract import build_train_cv_contract

    rows = _write_contract_rows(tmp_path)
    optional = {
        "train": {
            "utt2embedding.pt": {"train-a": [1.0]},
            "utt2speech_token.pt": {"train-a": [1, 2]},
            "spk2embedding.pt": {"spk-train": [3.0]},
        },
        "cv": {
            "utt2embedding.pt": {"cv-a": [4.0]},
            "utt2speech_token.pt": {"cv-a": [5, 6]},
            "spk2embedding.pt": {"spk-cv": [7.0]},
        },
    }
    build_train_cv_contract(
        tmp_path / "contract",
        train_rows=rows["train"],
        cv_rows=rows["cv"],
        frozen_train_ids={"train-a"},
        frozen_cv_ids={"cv-a"},
        rejected_rows=[],
        optional_maps=optional,
        num_utts_per_parquet=1,
        num_processes=1,
    )

    assert torch.load(
        tmp_path / "contract/splits/train/src/utt2speech_token.pt",
        weights_only=True,
    )["train-a"] == [1, 2]
    assert torch.load(
        tmp_path / "contract/splits/cv/src/utt2embedding.pt",
        weights_only=True,
    )["cv-a"] == [4.0]


def test_train_and_cv_parquet_are_directly_loadable(tmp_path):
    from tools.build_emofilm_contract import validate_train_cv_parquet

    train_dir = tmp_path / "train"
    cv_dir = tmp_path / "cv"
    train_dir.mkdir()
    cv_dir.mkdir()
    train_shard = train_dir / "parquet_000000000.tar"
    cv_shard = cv_dir / "parquet_000000000.tar"
    pq.write_table(pa.table({"utt": ["train-a"], "text": ["a"]}), train_shard)
    pq.write_table(pa.table({"utt": ["cv-a"], "text": ["b"]}), cv_shard)
    train_list = train_dir / "data.list"
    cv_list = cv_dir / "data.list"
    train_list.write_text("parquet_000000000.tar\n", encoding="utf-8")
    cv_list.write_text("parquet_000000000.tar\n", encoding="utf-8")

    report = validate_train_cv_parquet(train_list, cv_list)

    assert report["train_rows"] == 1
    assert report["cv_rows"] == 1
    assert report["shared_shards"] == []


def _write_contract_rows(tmp_path):
    rows = {}
    for split, utt_id in (("train", "train-a"), ("cv", "cv-a")):
        wav_path = tmp_path / f"{utt_id}.wav"
        wav_path.write_bytes(b"RIFF-fake-wav")
        rows[split] = [
            {
                "utt_id": utt_id,
                "audio_filepath": str(wav_path),
                "text": "<emotion type='neu' intensity='low'>hello</emotion>",
                "plain_text": "hello",
                "speaker_id": f"spk-{split}",
                "label_source": "esd_global_label",
            }
        ]
    return rows


def test_build_train_cv_contract_creates_independent_artifacts(tmp_path):
    from tools.build_emofilm_contract import build_train_cv_contract

    rows = _write_contract_rows(tmp_path)
    report = build_train_cv_contract(
        tmp_path / "contract",
        train_rows=rows["train"],
        cv_rows=rows["cv"],
        frozen_train_ids={"train-a"},
        frozen_cv_ids={"cv-a"},
        rejected_rows=[],
        num_utts_per_parquet=1,
        num_processes=1,
    )

    contract = tmp_path / "contract"
    assert (contract / "splits" / "train" / "manifest.jsonl").is_file()
    assert (contract / "splits" / "cv" / "manifest.jsonl").is_file()
    assert (contract / "splits" / "train" / "src" / "wav.scp").is_file()
    assert (contract / "splits" / "cv" / "src" / "text").is_file()
    assert (contract / "splits" / "train" / "parquet" / "data.list").is_file()
    assert (contract / "splits" / "cv" / "parquet" / "data.list").is_file()
    assert report["train"]["rows"] == 1
    assert report["cv"]["rows"] == 1
    assert report["train"]["data_list_sha256"] != report["cv"]["data_list_sha256"]

    from tools.build_emofilm_contract import validate_train_cv_parquet

    loaded = validate_train_cv_parquet(
        contract / "splits" / "train" / "parquet" / "data.list",
        contract / "splits" / "cv" / "parquet" / "data.list",
    )
    assert loaded == {"train_rows": 1, "cv_rows": 1, "shared_shards": []}


def test_build_train_cv_contract_rebuilds_existing_splits_without_staging_residue(tmp_path):
    from tools.build_emofilm_contract import build_train_cv_contract

    rows = _write_contract_rows(tmp_path)
    kwargs = {
        "train_rows": rows["train"],
        "cv_rows": rows["cv"],
        "frozen_train_ids": {"train-a"},
        "frozen_cv_ids": {"cv-a"},
        "rejected_rows": [],
        "num_utts_per_parquet": 1,
        "num_processes": 1,
    }

    first = build_train_cv_contract(tmp_path / "contract", **kwargs)
    second = build_train_cv_contract(tmp_path / "contract", **kwargs)

    assert first["train"]["data_list_sha256"] == second["train"]["data_list_sha256"]
    assert first["cv"]["data_list_sha256"] == second["cv"]["data_list_sha256"]
    assert not (tmp_path / "contract/.splits.staging").exists()
    assert not (tmp_path / "contract/.splits.backup").exists()


def test_pack_src_dir_does_not_write_data_list_after_worker_failure(tmp_path):
    from tools.make_parquet_list import pack_src_dir

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "wav.scp").write_text("utt-missing /not/a/real/file.wav\n", encoding="utf-8")
    (src_dir / "text").write_text("utt-missing hello\n", encoding="utf-8")
    (src_dir / "utt2spk").write_text("utt-missing spk\n", encoding="utf-8")
    output_dir = tmp_path / "parquet"

    with pytest.raises(FileNotFoundError):
        pack_src_dir(src_dir, output_dir, num_utts_per_parquet=1, num_processes=1)

    assert not (output_dir / "data.list").exists()


def test_pack_src_dir_resolves_speaker_embeddings_by_speaker_id(tmp_path):
    from tools.make_parquet_list import pack_src_dir

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    wav_path = tmp_path / "utt.wav"
    wav_path.write_bytes(b"RIFF-fake-wav")
    (src_dir / "wav.scp").write_text(f"utt-a {wav_path}\n", encoding="utf-8")
    (src_dir / "text").write_text("utt-a hello\n", encoding="utf-8")
    (src_dir / "utt2spk").write_text("utt-a speaker-a\n", encoding="utf-8")
    torch.save({"utt-a": [1.0]}, src_dir / "utt2embedding.pt")
    torch.save({"speaker-a": [2.0]}, src_dir / "spk2embedding.pt")

    report = pack_src_dir(src_dir, tmp_path / "parquet", num_processes=1)
    table = pq.read_table(report["shards"][0] if Path(report["shards"][0]).is_absolute() else tmp_path / "parquet" / report["shards"][0])
    assert table.column_names[-2:] == ["utt_embedding", "spk_embedding"]
    assert table["spk_embedding"][0].as_py() == [2.0]


def test_pack_src_dir_resolves_relative_wav_against_source_root(tmp_path):
    from tools.make_parquet_list import pack_src_dir

    workspace = tmp_path / "workspace"
    wav_path = workspace / "datasets" / "ESD" / "u.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"RIFF-fake-wav")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "wav.scp").write_text(
        "utt-a datasets/ESD/u.wav\n", encoding="utf-8"
    )
    (src_dir / "text").write_text("utt-a hello\n", encoding="utf-8")
    (src_dir / "utt2spk").write_text("utt-a speaker-a\n", encoding="utf-8")

    report = pack_src_dir(
        src_dir,
        tmp_path / "parquet",
        source_root=workspace,
        num_processes=1,
    )
    shard = tmp_path / "parquet" / report["shards"][0]
    table = pq.read_table(shard)
    assert table["wav"][0].as_py() == "datasets/ESD/u.wav"
    assert table["audio_data"][0].as_py() == b"RIFF-fake-wav"


def test_data_list_uses_paths_relative_to_list_file(tmp_path):
    from tools.make_parquet_list import pack_src_dir

    wav_path = tmp_path / "datasets" / "u.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(b"RIFF-fake-wav")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "wav.scp").write_text("utt-a datasets/u.wav\n", encoding="utf-8")
    (src_dir / "text").write_text("utt-a hello\n", encoding="utf-8")
    (src_dir / "utt2spk").write_text("utt-a speaker-a\n", encoding="utf-8")
    output_dir = tmp_path / "data" / "contracts" / "split" / "parquet"

    pack_src_dir(
        src_dir,
        output_dir,
        source_root=tmp_path,
        num_processes=1,
    )

    assert (output_dir / "data.list").read_text() == "parquet_000000000.tar\n"
    assert (output_dir / (output_dir / "data.list").read_text().strip()).is_file()


def test_rejected_ids_must_come_from_frozen_union_membership():
    from tools.build_emofilm_contract import validate_membership

    validate_membership(
        train_ids={"train-a"},
        cv_ids=set(),
        rejected_ids={"cv-a"},
        frozen_train_ids={"train-a"},
        frozen_cv_ids={"cv-a"},
    )

    with pytest.raises(ValueError, match="frozen union"):
        validate_membership(
            train_ids={"train-a"},
            cv_ids={"cv-a"},
            rejected_ids={"outside"},
            frozen_train_ids={"train-a"},
            frozen_cv_ids={"cv-a"},
        )


def test_prepare_parser_requires_explicit_input_contracts():
    from tools.prepare_emofilm_v1_data import build_parser

    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert "--iemocap-manifest" in options
    assert "--esd-manifest" in options
    assert "--esd-tagged-manifest" in options
    assert "--esd-eval-manifest" in options
    assert "--fedd-manifest" in options
    assert "--frozen-train-parquet" in options
    assert "--frozen-cv-parquet" in options


def test_frozen_source_rows_record_original_split():
    from tools.prepare_emofilm_v1_data import frozen_source_rows

    rows = [
        {"utt_id": "train-a", "source_dataset": "iemocap"},
        {"utt_id": "cv-a", "source_dataset": "iemocap"},
        {"utt_id": "source-only", "source_dataset": "iemocap"},
    ]

    frozen = frozen_source_rows(
        rows,
        frozen_train_ids={"train-a"},
        frozen_cv_ids={"cv-a"},
    )

    assert frozen == [
        {"utt_id": "train-a", "source_dataset": "iemocap", "original_split": "train"},
        {"utt_id": "cv-a", "source_dataset": "iemocap", "original_split": "cv"},
    ]


def test_split_parser_requires_explicit_reuse_cache_dir():
    from tools.build_emofilm_splits import build_parser

    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    assert "--reuse-cache-dir" in options
    assert "--project-root" not in options


def test_run_identity_records_contract_and_code_hash(tmp_path):
    from tools.write_emofilm_run_identity import write_run_identity

    contract = tmp_path / "contract"
    provenance = contract / "provenance"
    provenance.mkdir(parents=True)
    for filename, value in (
        ("contract.json", {"contract_name": "emofilm_v1"}),
        ("sources.json", []),
        ("membership.json", {"train": [], "cv": []}),
    ):
        (provenance / filename).write_text(json.dumps(value), encoding="utf-8")

    identity = write_run_identity(
        tmp_path / "identity.json",
        run_kind="train",
        code_root=Path(__file__).parents[1],
        contract_dir=contract,
        command="pytest",
        seed=1986,
    )

    assert identity["contract_name"] == "emofilm_v1"
    assert len(identity["contract_hash"]) == 64
    assert identity["code"]["git_head"]
