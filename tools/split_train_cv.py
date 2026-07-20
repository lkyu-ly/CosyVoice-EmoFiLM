#!/usr/bin/env python3
"""parquet 目录 train/cv 切分。

读 src_dir 下所有 parquet_*.tar，按 cv_ratio 随机切分为 train/cv 两组
（每 shard 内独立 shuffle，固定 seed 可复现），各合并为一个 parquet + data.list。

用法:
  python tools/split_train_cv.py \\
    --src_dir data/parquets/full \\
    --train_dir data/parquets/full/train \\
    --cv_dir data/parquets/full/cv \\
    --cv_ratio 0.05 --seed 42
"""
import argparse
import glob
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq


def split_parquet_train_cv(src_dir, train_dir, cv_dir, cv_ratio=0.05, seed=42):
    """切分 src 目录所有 parquet_*.tar 为 train/cv 两组。

    每个 shard 内独立 shuffle 后按 cv_ratio 取 cv（至少 1 条），其余归 train；
    跨 shard 合并后各写一个 parquet + data.list。固定 seed 保证可复现。

    Returns: (train_count, cv_count)。
    """
    rng = random.Random(seed)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(cv_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src_dir, "parquet_*.tar")))
    if not files:
        raise FileNotFoundError(f"No parquet_*.tar in {src_dir}")

    train_rows, cv_rows = [], []
    for f in files:
        table = pq.read_table(f)
        n = table.num_rows
        idx = list(range(n))
        rng.shuffle(idx)
        cv_n = max(1, int(n * cv_ratio))  # 至少 1 条 cv
        cv_idx = set(idx[:cv_n])

        cv_rows.append(table.take(list(cv_idx)))
        train_mask = pa.array([i not in cv_idx for i in range(n)])
        train_rows.append(table.filter(train_mask))

    full_train = pa.concat_tables(train_rows)
    full_cv = pa.concat_tables(cv_rows)
    pq.write_table(full_train, os.path.join(train_dir, "parquet_000000000.tar"))
    pq.write_table(full_cv, os.path.join(cv_dir, "parquet_000000000.tar"))

    with open(os.path.join(train_dir, "data.list"), "w") as f:
        f.write(os.path.join(train_dir, "parquet_000000000.tar") + "\n")
    with open(os.path.join(cv_dir, "data.list"), "w") as f:
        f.write(os.path.join(cv_dir, "parquet_000000000.tar") + "\n")

    return full_train.num_rows, full_cv.num_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--cv_dir", required=True)
    parser.add_argument("--cv_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_train, n_cv = split_parquet_train_cv(
        args.src_dir, args.train_dir, args.cv_dir,
        cv_ratio=args.cv_ratio, seed=args.seed,
    )
    print(f"train={n_train}, cv={n_cv}, ratio={n_cv / (n_train + n_cv):.3f}")


if __name__ == "__main__":
    main()
