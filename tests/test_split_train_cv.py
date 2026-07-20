"""parquet train/cv 切分测试。"""
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = str(__import__("pathlib").Path(__file__).parents[1])
sys.path.insert(0, ROOT)


def test_import():
    from tools.split_train_cv import split_parquet_train_cv
    assert callable(split_parquet_train_cv)


def test_split_ratio(tmp_path):
    """100 行 parquet 切 5% cv → train=95, cv=5；utt_id 不重叠。"""
    from tools.split_train_cv import split_parquet_train_cv

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    table = pa.table({
        "utt_id": [f"u{i:03d}" for i in range(100)],
        "text": [f"text {i}" for i in range(100)],
    })
    pq.write_table(table, src_dir / "parquet_000000000.tar")

    n_train, n_cv = split_parquet_train_cv(
        src_dir=str(src_dir),
        train_dir=str(tmp_path / "train"),
        cv_dir=str(tmp_path / "cv"),
        cv_ratio=0.05, seed=42,
    )
    assert n_train == 95
    assert n_cv == 5

    train_table = pq.read_table(tmp_path / "train" / "parquet_000000000.tar")
    cv_table = pq.read_table(tmp_path / "cv" / "parquet_000000000.tar")
    assert train_table.num_rows == 95
    assert cv_table.num_rows == 5

    train_ids = set(train_table["utt_id"].to_pylist())
    cv_ids = set(cv_table["utt_id"].to_pylist())
    assert train_ids.isdisjoint(cv_ids)


def test_seed_reproducibility(tmp_path):
    """同 seed 切两次结果一致。"""
    from tools.split_train_cv import split_parquet_train_cv

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    table = pa.table({"utt_id": [f"u{i}" for i in range(20)], "text": ["t"] * 20})
    pq.write_table(table, src_dir / "parquet_000000000.tar")

    split_parquet_train_cv(src_dir=str(src_dir), train_dir=str(tmp_path / "t1"),
                           cv_dir=str(tmp_path / "c1"), seed=42)
    split_parquet_train_cv(src_dir=str(src_dir), train_dir=str(tmp_path / "t2"),
                           cv_dir=str(tmp_path / "c2"), seed=42)
    t1 = pq.read_table(tmp_path / "t1" / "parquet_000000000.tar")
    t2 = pq.read_table(tmp_path / "t2" / "parquet_000000000.tar")
    assert t1["utt_id"].to_pylist() == t2["utt_id"].to_pylist()
