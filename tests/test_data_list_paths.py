"""训练 data.list 路径解析合同。"""

from pathlib import Path


def test_read_lists_resolves_relative_entries_from_list_file_directory(tmp_path):
    from cosyvoice.utils.file_utils import read_lists

    list_dir = tmp_path / "contract" / "train" / "parquet"
    list_dir.mkdir(parents=True)
    shard = list_dir / "train_parquet_000000000.tar"
    shard.write_bytes(b"parquet")
    data_list = list_dir / "data.list"
    data_list.write_text("train_parquet_000000000.tar\n", encoding="utf-8")

    assert read_lists(data_list) == [str(shard)]
