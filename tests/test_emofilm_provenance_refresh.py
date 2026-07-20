"""emofilm_v1 provenance 摘要刷新合同。"""

import json

import pyarrow as pa
import pyarrow.parquet as pq


def test_refresh_split_summary_resolves_local_lists_and_records_schema(tmp_path):
    from tools.refresh_emofilm_provenance import refresh_split_summary

    contract = tmp_path / "contract"
    train = contract / "splits" / "train" / "parquet"
    cv = contract / "splits" / "cv" / "parquet"
    train.mkdir(parents=True)
    cv.mkdir(parents=True)
    pq.write_table(pa.table({"utt": ["train-a"], "text": ["a"]}), train / "train_000.tar")
    pq.write_table(pa.table({"utt": ["cv-a"], "text": ["b"]}), cv / "cv_000.tar")
    (train / "data.list").write_text("train_000.tar\n", encoding="utf-8")
    (cv / "data.list").write_text("cv_000.tar\n", encoding="utf-8")

    summary = refresh_split_summary(
        contract,
        train_ids=["train-a"],
        cv_ids=["cv-a"],
    )

    assert summary["train_count"] == 1
    assert summary["cv_count"] == 1
    assert summary["train"]["data_list_entries"] == ["train_000.tar"]
    assert summary["cv"]["data_list_entries"] == ["cv_000.tar"]
    assert summary["train"]["schema_columns"] == ["utt", "text"]
    assert summary["parquet"]["shared_shards"] == []


def test_build_artifact_records_includes_manifest_hashes_and_directory_quick_stats(tmp_path):
    from tools.refresh_emofilm_provenance import build_artifact_records

    contract = tmp_path / "contract"
    manifest = contract / "sources" / "iemocap" / "tagged.jsonl"
    frames = contract / "sources" / "iemocap" / "emotion2vec_base"
    manifest.parent.mkdir(parents=True)
    frames.mkdir(parents=True)
    manifest.write_text(json.dumps({"utt_id": "u1"}) + "\n", encoding="utf-8")
    (frames / "u1.pt").write_bytes(b"frame")

    records = build_artifact_records(
        contract,
        core_paths=[manifest],
        quick_dirs={"sources/iemocap/emotion2vec_base": frames},
    )

    by_path = {record["path"]: record for record in records}
    assert by_path["sources/iemocap/tagged.jsonl"]["sha256"]
    assert by_path["sources/iemocap/emotion2vec_base"]["file_count"] == 1
    assert by_path["sources/iemocap/emotion2vec_base"]["id_set_sha256"]


def test_refresh_contract_counts_records_active_and_rejected_membership(tmp_path):
    from tools.refresh_emofilm_provenance import refresh_contract_counts

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_name": "emofilm_v1",
                "frozen_train_count": 3,
                "frozen_cv_count": 2,
                "frozen_union_count": 5,
            }
        ),
        encoding="utf-8",
    )

    refreshed = refresh_contract_counts(
        contract_path,
        train_ids=["t1", "t2"],
        cv_ids=["c1"],
        rejected_ids=["t3", "c2"],
    )

    assert refreshed["active_train_count"] == 2
    assert refreshed["active_cv_count"] == 1
    assert refreshed["active_union_count"] == 3
    assert refreshed["rejected_count"] == 2
    assert json.loads(contract_path.read_text()) == refreshed
