#!/usr/bin/env python3
"""FEDD Part A 音频合成器（小米 MiMo-V2.5-TTS 版，2026-07-08）。

对齐论文（学位论文 §FEDD）：mild transition 语音由 TTS 按自然语言指令合成。
原计划用 GPT-4o Audio；供应商无 GPT audio、MiniMax 无自由指令（句内无渐变），
改用 MiMo-V2.5-TTS —— OpenAI 兼容 chat/completions，支持"同一段语音内风格转场"，
真正实现 Part A 的句内 emo_from→emo_to 渐变。
- 两段式 messages：user=情感过渡指令（不进语音），assistant=待合成文本（逐字合成）。
- 复用 generate_fedd_part_a_texts.py 已产出的 tts_instructions（本就是渐变指令）。
- 每条 prompt 用自己的 voice（4 个 MiMo 英文预置音色 = Part A 说话人）。
- 非流式返回 base64 wav → ffmpeg 转 16kHz 单声道 wav。
- 并发（默认 10）+ tqdm 进度条；失败重试 3 次后记 failed_log 跳过，不写静音占位。

用法:
  # 硬编码：把 Key 填进下方 HARDCODED_MIMO_API_KEY 即可直接跑；或设环境变量 MIMO_API_KEY。
  # 冒烟（10 条 → /tmp）:
  python tools/build_fedd_part_a_mimo.py \
    --prompts /tmp/pa_smoke.jsonl \
    --output_dir /tmp/fedd_part_a_smoke/wav \
    --manifest /tmp/fedd_part_a_smoke/manifest.jsonl --num_samples 10
  # 全量（并发 10）:
  python tools/build_fedd_part_a_mimo.py \
    --prompts data/fedd_part_a_prompts.jsonl \
    --output_dir data/fedd_rebuilt/wav \
    --manifest data/fedd_rebuilt/manifest.jsonl --num_samples 500 --concurrency 10
"""
import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests
from tqdm import tqdm

# ── 快速测试硬编码（留空则回退环境变量）────────────────────────────
HARDCODED_MIMO_API_KEY = ""       # 已脱敏；设环境变量 MIMO_API_KEY（_resolve_config 回退读取）
HARDCODED_MIMO_MODEL = ""         # 留空 = mimo-v2.5-tts
HARDCODED_MIMO_BASE_URL = ""      # 留空 = https://api.xiaomimimo.com/v1
# ────────────────────────────────────────────────────────────────

RETRY_SLEEP_S = 2
REQUEST_TIMEOUT_S = 180


@dataclass
class MiMoConfig:
    api_key: str
    model: str = "mimo-v2.5-tts"
    base_url: str = "https://api.xiaomimimo.com/v1"


def build_payload(text: str, voice: str, instruction: str, cfg: MiMoConfig) -> dict:
    """构造 MiMo chat/completions 请求体。

    两段式：user 放情感过渡指令（不进语音），assistant 放待合成文本（逐字合成）。
    """
    return {
        "model": cfg.model,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": text},
        ],
        "audio": {"format": "wav", "voice": voice},
    }


def synth_one(text: str, voice: str, instruction: str, cfg: MiMoConfig,
              http_post=requests.post) -> bytes:
    """调 MiMo 合成一条，返回原始音频字节（wav）。

    http_post 可注入（默认 requests.post）便于单测离线。返回缺失/报错抛 RuntimeError。
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "api-key": cfg.api_key,
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = build_payload(text, voice, instruction, cfg)
    resp = http_post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S)
    data = resp.json()
    try:
        audio_b64 = data["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError):
        err = data.get("error") if isinstance(data, dict) else None
        raise RuntimeError(f"MiMo TTS no audio in response: {err or data}")
    if not audio_b64:
        raise RuntimeError("MiMo TTS returned empty audio")
    return base64.b64decode(audio_b64)


def _convert_audio_to_wav(audio_bytes: bytes, wav_path: str, sr: int = 16000) -> None:
    """API 返回的音频 bytes（wav，ffmpeg 自动嗅探）→ 16kHz mono wav。

    转换失败抛 RuntimeError，由调用方重试/记录。要求 ffmpeg 在 PATH。
    """
    import subprocess
    import tempfile

    fd, tmp_in = tempfile.mkstemp(suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_in,
             "-ar", str(sr), "-ac", "1", wav_path],
            capture_output=True,
        )
        if r.returncode != 0 or not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr.decode(errors='ignore')[:300]}")
    finally:
        if os.path.exists(tmp_in):
            os.unlink(tmp_in)


def _synth_record(p: dict, output_dir: str, cfg: MiMoConfig, http_post) -> tuple:
    """单条：3 次重试 → 合成 → 转 wav → 返回 ("ok", entry) 或 ("fail", failure)。"""
    last_err = None
    for attempt in range(3):
        try:
            audio = synth_one(p["text"], p["voice"], p["tts_instructions"], cfg,
                              http_post=http_post)
            utt_id = f"fedd_a_{p['prompt_id']}"
            wav_path = os.path.join(output_dir, f"{utt_id}.wav")
            _convert_audio_to_wav(audio, wav_path)
            return ("ok", {
                "utt_id": utt_id,
                "wav_path": wav_path,
                "text": p["text"],
                "emo_from": p["emo_from"],
                "emo_to": p["emo_to"],
                "emotion_transition": f"{p['emo_from']}→{p['emo_to']}",
                "speaker_id": p["voice"],
                "source": "mimo_api",
                "part": "A",
                "level": "mild",
                "model": cfg.model,
            })
        except Exception as e:
            last_err = e
            print(f"[retry {attempt+1}/3] {p['prompt_id']}: {e}")
            time.sleep(RETRY_SLEEP_S)
    return ("fail", {"prompt_id": p["prompt_id"], "text": p["text"], "error": str(last_err)})


def generate_part_a_mimo(
    prompts: list,
    output_dir: str,
    cfg: MiMoConfig,
    num_samples: int = 500,
    concurrency: int = 10,
    failed_log: str = None,
    http_post=requests.post,
) -> list:
    """按 prompts 并发调 MiMo 指令 TTS，返回成功的 manifest 条目列表。

    Args:
        prompts: list of {prompt_id, text, emo_from, emo_to, voice, tts_instructions}
                 （generate_fedd_part_a_texts.py 产出；voice 为 MiMo voice、
                  tts_instructions 为句内渐变指令）。
        output_dir: wav 输出目录。
        cfg: MiMoConfig。
        num_samples: 最大生成数（取前 N；冒烟用 10）。
        concurrency: 并发线程数（默认 10）。
        failed_log: 失败 prompt 记录文件路径（可选）。
        http_post: 可注入的 HTTP POST（默认 requests.post）。
    """
    selected = prompts[:num_samples]
    os.makedirs(output_dir, exist_ok=True)

    entries = []
    failures = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_synth_record, p, output_dir, cfg, http_post) for p in selected]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="MiMo TTS"):
            kind, payload = fut.result()
            if kind == "ok":
                entries.append(payload)
            else:
                failures.append(payload)

    if failed_log and failures:
        os.makedirs(os.path.dirname(failed_log) or ".", exist_ok=True)
        with open(failed_log, "w", encoding="utf-8") as f:
            for fail in failures:
                f.write(json.dumps(fail, ensure_ascii=False) + "\n")

    return entries


def _resolve_config() -> MiMoConfig:
    api_key = HARDCODED_MIMO_API_KEY or os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set HARDCODED_MIMO_API_KEY or MIMO_API_KEY env var")
    model = HARDCODED_MIMO_MODEL or os.environ.get("MIMO_MODEL") or "mimo-v2.5-tts"
    base_url = (HARDCODED_MIMO_BASE_URL or os.environ.get("MIMO_BASE_URL")
                or "https://api.xiaomimimo.com/v1")
    return MiMoConfig(api_key=api_key, model=model, base_url=base_url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True,
                        help="generate_fedd_part_a_texts.py 产出的 jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", required=True, help="manifest.jsonl（追加写入）")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--failed_log", default=None)
    args = parser.parse_args()

    cfg = _resolve_config()

    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    if not prompts:
        raise SystemExit(f"ERROR: {args.prompts} 为空 —— 请先用 generate_fedd_part_a_texts.py 生成")
    required = {"prompt_id", "text", "emo_from", "emo_to", "voice", "tts_instructions"}
    missing = required - set(prompts[0].keys())
    if missing:
        raise SystemExit(f"ERROR: prompts 缺字段 {missing} —— 请先用 "
                         f"generate_fedd_part_a_texts.py 生成")

    failed_log = args.failed_log or os.path.join(
        os.path.dirname(args.manifest) or ".", "failed_prompts.jsonl")

    entries = generate_part_a_mimo(
        prompts=prompts, output_dir=args.output_dir, cfg=cfg,
        num_samples=args.num_samples, concurrency=args.concurrency, failed_log=failed_log,
    )

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Done. {len(entries)}/{min(args.num_samples, len(prompts))} Part A entries "
          f"appended to {args.manifest}")
    print(f"Failures logged to {failed_log}")


if __name__ == "__main__":
    main()
