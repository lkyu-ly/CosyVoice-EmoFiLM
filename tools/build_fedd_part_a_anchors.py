#!/usr/bin/env python3
"""FEDD Part A neutral anchor 生成器（2026-07-13）。

为 4 个 MiMo 英文预置 voice（Mia/Chloe/Milo/Dean）各生成 1 条独立中性参考语音
（neutral anchor），作为该 voice 全部 125 条 Part A 样本的 prompt_wav（speaker
conditioning）。anchor 独立于 target、中性语气、不含情感转折，避免用 target 当
prompt 造成评测泄漏。

复用 build_fedd_part_a_mimo.py 的合成原语（synth_one / MiMoConfig /
_convert_audio_to_wav / _resolve_config），不重复 API 逻辑。

决策见 docs/adr/0001-fedd-part-a-neutral-anchor.md。

用法:
  python tools/build_fedd_part_a_anchors.py
  # 默认输出 data/fedd_rebuilt/prompts/，需 HARDCODED_MIMO_API_KEY 或 MIMO_API_KEY
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from tools.build_fedd_part_a_mimo import (  # noqa: E402
    MiMoConfig,
    synth_one,
    _convert_audio_to_wav,
    _resolve_config,
)

VOICES = ["Mia", "Chloe", "Milo", "Dean"]
ANCHOR_TEXT = "The weather is calm today, and I am reading a simple sentence."
ANCHOR_INSTRUCTION = (
    "Speak this sentence in a calm, neutral tone with no emotional transition."
)
PROMPT_SOURCE = "mimo_same_voice_neutral_anchor"


def generate_anchors(output_dir, cfg, http_post=requests.post, anchor_manifest=None):
    """为每个 voice 合成 1 条 neutral anchor，返回 entries，可选写 anchor_manifest.jsonl。

    Args:
        output_dir: anchor wav 输出目录。
        cfg: MiMoConfig。
        http_post: 可注入 HTTP POST（默认 requests.post），便于单测离线。
        anchor_manifest: anchor_manifest.jsonl 路径（可选；给则写）。
    """
    os.makedirs(output_dir, exist_ok=True)
    entries = []
    for voice in VOICES:
        audio = synth_one(ANCHOR_TEXT, voice, ANCHOR_INSTRUCTION, cfg, http_post=http_post)
        wav_path = os.path.join(output_dir, f"{voice}_neutral_anchor.wav")
        _convert_audio_to_wav(audio, wav_path)
        entries.append({
            "voice": voice,
            "prompt_wav": wav_path,
            "prompt_text": ANCHOR_TEXT,
            "prompt_source": PROMPT_SOURCE,
            "instruction": ANCHOR_INSTRUCTION,
        })
        print(f"[anchor] {voice} -> {wav_path}")
    if anchor_manifest:
        os.makedirs(os.path.dirname(anchor_manifest) or ".", exist_ok=True)
        with open(anchor_manifest, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"[anchor] manifest -> {anchor_manifest} ({len(entries)} entries)")
    return entries


def main():
    parser = argparse.ArgumentParser(description="生成 FEDD Part A neutral anchors")
    parser.add_argument("--output_dir", default="data/fedd_rebuilt/prompts")
    parser.add_argument(
        "--anchor_manifest", default="data/fedd_rebuilt/prompts/anchor_manifest.jsonl")
    args = parser.parse_args()

    cfg = _resolve_config()
    generate_anchors(
        output_dir=args.output_dir, cfg=cfg,
        anchor_manifest=args.anchor_manifest,
    )


if __name__ == "__main__":
    main()
