#!/usr/bin/env python3
"""多 GPU 数据并行推理 launcher。

把测试 manifest 按条目切片分给 K 个 GPU，每个 GPU 跑一个独立的
inference_emo_film.py 子进程（数据并行，非模型级 batch）。所有子进程结束后，
按 utt_id 合并各分片 manifest 为单个 inference_{base}.jsonl。

关键：子进程 env 必须设 PYTHONPATH=$REPO:$REPO/third_party/Matcha-TTS，
否则 Matcha-TTS import 失败（见 ADR-0001 / ADR-0002）。

用法:
  python tools/run_inference_parallel.py \\
    --gpus 1,2,3,4 \\
    --model_dir pretrained_models/CosyVoice2-0.5B \\
    --llm_ckpt exp/emofilm_v1/final.pt \\
    --test_manifest data/contracts/emofilm_v1/eval/esd/manifest.jsonl \\
    --esd_root datasets/ESD \\
    --output_dir exp/emofilm_v1/wav_esd
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
_DEFAULT_INFER_SCRIPT = "tools/inference_emo_film.py"


def _shard_manifest_path(parent, base, shard_idx):
    return os.path.join(parent, f"inference_{base}.shard{shard_idx}.jsonl")


def _read_jsonl(path):
    """读 jsonl，返回行列表；文件不存在返回 []。"""
    rows = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def run_inference_parallel(model_dir, llm_ckpt, test_manifest, esd_root, output_dir,
                           gpus, use_tagged_text=True, skip_existing=False,
                           save_every=50, workspace_root=REPO,
                           python=PY, inference_script=None):
    """每 GPU 启动一个分片子进程，并发推理，结束后合并 manifest。

    Args:
        gpus: GPU id 列表，如 [1,2,3,4]，K=len(gpus)。
        其余参数透传给 inference_emo_film.py。

    Returns:
        {"per_shard": [{"gpu","shard_idx","returncode","n_out"}],
         "merged_count": int, "failed_shards": [shard_idx, ...]}
    """
    if not gpus:
        raise ValueError("gpus must be a non-empty list")
    if inference_script is None:
        inference_script = _DEFAULT_INFER_SCRIPT

    K = len(gpus)
    base = os.path.basename(output_dir.rstrip("/"))
    parent = os.path.dirname(output_dir.rstrip("/"))

    # 子进程 env：复用当前环境（PATH / CUDA libs），覆盖 PYTHONPATH（Matcha-TTS 关键）
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = f"{REPO}:{REPO}/third_party/Matcha-TTS"

    # 并发启动：每 GPU 一个 Popen
    procs = []
    for i, gpu in enumerate(gpus):
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            python, inference_script,
            "--model_dir", model_dir,
            "--llm_ckpt", llm_ckpt,
            "--test_manifest", test_manifest,
            "--esd_root", esd_root,
            "--workspace_root", workspace_root,
            "--output_dir", output_dir,
            "--device", "cuda",
            "--shard_idx", str(i),
            "--num_shards", str(K),
            "--save_every", str(save_every),
        ]
        if skip_existing:
            cmd.append("--skip_existing")
        if not use_tagged_text:
            cmd.append("--plain_text")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env)
        procs.append({"gpu": gpu, "shard_idx": i, "proc": proc, "cmd": cmd})

    # 等全部结束，收 returncode
    for p in procs:
        p["returncode"] = p["proc"].wait()

    failed_shards = [p["shard_idx"] for p in procs if p["returncode"] != 0]
    if failed_shards:
        raise RuntimeError(f"inference shard {failed_shards[0]} failed")

    # 合并各分片 manifest（按 index 0..K-1 读，失败分片的缺失文件直接跳过）
    merged = {}
    for i in range(K):
        for r in _read_jsonl(_shard_manifest_path(parent, base, i)):
            merged[r.get("utt_id")] = r

    merged_path = os.path.join(parent, f"inference_{base}.jsonl")
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 组装返回（n_out = 各分片 manifest 行数）
    per_shard = []
    failed_shards = []
    for p in procs:
        shard_rows = _read_jsonl(_shard_manifest_path(parent, base, p["shard_idx"]))
        entry = {"gpu": p["gpu"], "shard_idx": p["shard_idx"],
                 "returncode": p["returncode"], "n_out": len(shard_rows)}
        per_shard.append(entry)
        if p["returncode"] != 0:
            failed_shards.append(p["shard_idx"])

    return {"per_shard": per_shard, "merged_count": len(merged),
            "failed_shards": failed_shards}


def main():
    parser = argparse.ArgumentParser(description="多 GPU 数据并行推理 launcher")
    parser.add_argument("--gpus", required=True,
                        help="逗号分隔的 GPU id，如 1,2,3,4")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--llm_ckpt", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--esd_root", required=True)
    parser.add_argument("--workspace_root", default=REPO)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--skip_existing", action="store_true")
    tagged = parser.add_mutually_exclusive_group()
    tagged.add_argument("--use_tagged_text", dest="use_tagged_text", action="store_true",
                        help="用 tagged（<emotion> 词级标签，默认）")
    tagged.add_argument("--plain_text", dest="use_tagged_text", action="store_false",
                        help="用 plain_text（无词级标签）")
    parser.set_defaults(use_tagged_text=True)
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    result = run_inference_parallel(
        args.model_dir, args.llm_ckpt, args.test_manifest, args.esd_root,
        args.output_dir, gpus,
        use_tagged_text=args.use_tagged_text,
        skip_existing=args.skip_existing,
        save_every=args.save_every,
        workspace_root=args.workspace_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
