#!/usr/bin/env python3
# 一次性下载 AbstractTTS/IEMOCAP HF 数据集并解包到 datasets/IEMOCAP/
#
# 用法：
#   python tools/download_iemocap.py
#
# 若 SSL 失败（cdn-lfs 域名被代理拦截），请设置环境变量后重试：
#   export HF_ENDPOINT=https://hf-mirror.com
#   或
#   export HTTPS_PROXY=http://your-proxy:port
#
# 说明：
#   不使用 `datasets` 库（emofilm env 中 torch 2.3 与 datasets 5.0 冲突）。
#   直接走 huggingface_hub + pyarrow + soundfile。
import csv
import io
from pathlib import Path

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
import soundfile as sf

REPO_ID = "AbstractTTS/IEMOCAP"
OUT_DIR = Path("/home/lkyu/LLM-Audio/datasets/IEMOCAP")
WAV_DIR = OUT_DIR / "wav"
HF_CACHE = OUT_DIR / "_hf_parquet"  # parquet 原始缓存，可手动删除

COLUMNS = [
    "file", "gender", "major_emotion", "transcription",
    "angry", "happy", "sad", "neutral", "excited", "surprise",
    "frustrated", "disgust", "fear",
    "EmoAct", "EmoVal", "EmoDom",
    "speaking_rate", "pitch_mean", "pitch_std", "rms", "relative_db",
    "wav_path",
]


def main():
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUT_DIR / "labels.csv"

    print(f"[1/4] snapshot_download({REPO_ID!r}) ...", flush=True)
    print(f"      若长时间无进展或 SSL 失败，设置 HF_ENDPOINT=https://hf-mirror.com 后重试", flush=True)
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(HF_CACHE),
        allow_patterns=["data/*.parquet"],
    )
    parquet_files = sorted(Path(local_dir).glob("data/*.parquet"))
    print(f"      parquet shards: {len(parquet_files)}", flush=True)

    print(f"[2/4] 探测 schema ...", flush=True)
    schema = pq.read_schema(parquet_files[0])
    audio_type = schema.field("audio").type
    print(f"      audio arrow type: {audio_type}", flush=True)

    print(f"[3/4] 解码音频并写入 {WAV_DIR} ...", flush=True)
    n_total = 0
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for pf in parquet_files:
            tbl = pq.read_table(pf, columns=[c for c in COLUMNS if c != "wav_path"] + ["audio"])
            for row in tbl.to_pylist():
                wav_path = WAV_DIR / row["file"]
                if not wav_path.exists():
                    audio_struct = row["audio"]
                    # HF Audio feature: {'bytes': bytes|None, 'path': str|None}
                    audio_bytes = audio_struct.get("bytes")
                    if not audio_bytes:
                        raise RuntimeError(f"audio bytes missing for {row['file']}")
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(wav_path), data, sr)
                r = {k: row.get(k) for k in COLUMNS if k != "wav_path"}
                r["wav_path"] = str(wav_path)
                w.writerow(r)
                n_total += 1
                if n_total % 500 == 0:
                    print(f"      {n_total}", flush=True)

    n_wav = sum(1 for _ in WAV_DIR.rglob("*.wav"))
    print(f"[4/4] 完成。labels={labels_path}  wavs={n_wav}", flush=True)


if __name__ == "__main__":
    main()
