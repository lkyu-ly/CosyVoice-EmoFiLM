#!/usr/bin/env python3
"""Emo-FiLM 批量推理脚本。

加载 CosyVoice2_Emotion + 训练后 LLM ckpt（final.pt），对测试 manifest 批量合成 wav。
text 逻辑兼容 Stage 2 tagged jsonl schema（text=tagged <emotion>，plain_text=原）：
- --use_tagged_text（默认）：用 tagged（text 或 tagged_text 字段），下游 emo_tokenizer 解析词级标签
- --plain_text：用 plain_text（消融/默认 neu-low）

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/inference_emo_film.py \\
    --model_dir pretrained_models/CosyVoice2-0.5B \\
    --llm_ckpt exp/emofilm_v1/final.pt \\
    --test_manifest data/contracts/emofilm_v1/eval/esd/manifest.jsonl \\
    --esd_root datasets/ESD \\
    --output_dir exp/emofilm_v1/wav_esd \\
    --device cuda

产物: {output_dir}/{utt_id}.wav + inference_manifest.jsonl
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def filter_state_dict(ckpt: dict) -> dict:
    """过滤 ckpt 字典中的 epoch/step 元信息，返回纯 state_dict。

    train_emo.py 保存格式: {**model.module.state_dict(), 'epoch': N, 'step': M}
    load_state_dict(strict=True) 拒绝额外 key，所以必须过滤。
    """
    return {k: v for k, v in ckpt.items() if k not in ("epoch", "step")}


def select_prompt_wav(utt: dict, esd_root: str) -> str:
    """为 zero-shot 推理选 prompt wav：同 speaker 的 Neutral 情感 wav。

    Args:
        utt: manifest 行，含 speaker_id。
        esd_root: ESD 数据根目录（结构 {esd_root}/{spk}/{Emotion}/*.wav）。

    Returns:
        prompt wav 绝对路径。缺 Neutral 时 fallback 到第一个可用 emotion。
    """
    spk = utt["speaker_id"]
    spk_dir = Path(esd_root) / spk
    if not spk_dir.is_dir():
        raise FileNotFoundError(f"speaker dir not found: {spk_dir}")

    # 优先 Neutral
    neutral_dir = spk_dir / "Neutral"
    if neutral_dir.is_dir():
        wavs = sorted(neutral_dir.glob("*.wav"))
        if wavs:
            return str(wavs[0])

    # Fallback: 第一个有 wav 的 emotion 目录
    for emo_dir in sorted(spk_dir.iterdir()):
        if emo_dir.is_dir():
            wavs = sorted(emo_dir.glob("*.wav"))
            if wavs:
                return str(wavs[0])

    raise FileNotFoundError(f"No prompt wav found for speaker {spk} under {spk_dir}")


def resolve_prompt(utt: dict, esd_root: str, workspace_root: str | None = None) -> dict:
    """显式 prompt 解析：manifest 自带优先；Part A 无 prompt 失败；ESD/Part B 用 ESD Neutral。

    Returns dict with ok/status/prompt_wav/prompt_text/prompt_source（失败时 reason）。
    """
    prompt_wav = utt.get("prompt_wav")
    prompt_text = utt.get("prompt_text")
    if prompt_wav:
        prompt_path = Path(prompt_wav)
        if not prompt_path.is_absolute() and workspace_root is not None:
            prompt_path = Path(workspace_root) / prompt_path
        prompt_path = prompt_path.resolve()
        if prompt_path.is_file():
            if not prompt_text:
                raise ValueError(f"prompt_text missing for prompt_wav: {prompt_path}")
            return {"ok": True, "prompt_wav": str(prompt_path),
                    "prompt_text": prompt_text,
                    "prompt_source": "manifest", "status": "success"}
        raise FileNotFoundError(f"prompt_wav not found: {prompt_path}")

    part = utt.get("part")
    if part == "A":
        raise FileNotFoundError("FEDD Part A requires manifest prompt_wav and prompt_text")

    # ESD / FEDD Part B：回退到 ESD same-speaker Neutral
    prompt_wav = select_prompt_wav(utt, esd_root)
    if not prompt_text:
        raise ValueError(f"prompt_text missing for speaker: {utt.get('speaker_id', '')}")
    return {"ok": True, "prompt_wav": prompt_wav,
            "prompt_text": prompt_text,
            "prompt_source": "esd_same_speaker_neutral", "status": "success"}


def load_emofilm_model(model_dir: str, llm_ckpt: str, fp16: bool = False, device: str = "cuda"):
    """加载 CosyVoice2_Emotion 并替换 LLM 权重为训练后 ckpt。

    device 指定 ckpt 加载时的 map_location，默认 "cuda"。
    """
    from cosyvoice.cli.cosyvoice_emo import CosyVoice2_Emotion
    from cosyvoice.utils.emo_checkpoint import load_trained_state

    cv2 = CosyVoice2_Emotion(model_dir, fp16=fp16)
    ckpt = torch.load(llm_ckpt, map_location=device, weights_only=True)
    state_dict = filter_state_dict(ckpt)
    load_trained_state(cv2.model.llm, state_dict)
    return cv2


def _pick_text(utt: dict, use_tagged_text: bool) -> str:
    """按 use_tagged_text 选文本，兼容 Stage 2 schema（text=tagged, plain_text=原）与其他。"""
    if use_tagged_text:
        return utt.get("tagged_text") or utt.get("text", "")
    return utt.get("plain_text") or utt.get("text", "")


def _manifest_path_for(output_dir, shard_idx, num_shards):
    """计算 manifest 路径：num_shards>1 时带 .shard{idx}.jsonl，否则保持原名。"""
    base = os.path.basename(output_dir.rstrip("/"))
    parent = os.path.dirname(output_dir.rstrip("/"))
    if num_shards > 1:
        name = f"inference_{base}.shard{shard_idx}.jsonl"
    else:
        name = f"inference_{base}.jsonl"
    return os.path.join(parent, name)


def _write_manifest(manifest_path, results):
    """把当前累计 results 覆盖写到 manifest_path（增量保存与最终保存共用）。"""
    parent = os.path.dirname(manifest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_inference(cv2, test_manifest, esd_root, output_dir,
                  use_tagged_text=True, max_samples=None,
                  shard_idx=0, num_shards=1, skip_existing=False, save_every=50,
                  workspace_root=None):
    """批量推理，返回 manifest 条目列表。

    多 GPU 数据并行：每个进程在 max_samples 截取后处理 entries[shard_idx::num_shards]。
    skip_existing：out_wav 已存在则记录 skipped_existing 并跳过合成（续跑）。
    save_every：每处理 N 条把累计 results 覆盖写回 manifest（增量保存），循环结束再写一次。
    manifest 命名：num_shards>1 → inference_{base}.shard{shard_idx}.jsonl；
    num_shards==1 → 保持原 inference_{base}.jsonl。
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(test_manifest) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    if max_samples:
        entries = entries[:max_samples]
    entries = entries[shard_idx::num_shards]

    manifest_path = _manifest_path_for(output_dir, shard_idx, num_shards)
    results = []
    started_at = time.perf_counter()
    for utt in tqdm(entries, desc="Emo-FiLM infer"):
        utt_id = utt["utt_id"]
        out_wav = os.path.join(output_dir, f"{utt_id}.wav")

        if skip_existing and os.path.isfile(out_wav):
            results.append({"utt_id": utt_id, "status": "skipped_existing",
                            "wav_path": out_wav})
        else:
            text_with_emo = _pick_text(utt, use_tagged_text)
            resolved = resolve_prompt(utt, esd_root, workspace_root=workspace_root)
            prompt_wav = resolved["prompt_wav"]
            prompt_text = resolved["prompt_text"]
            prompt_source = resolved["prompt_source"]
            t0 = time.time()
            wrote_output = False
            for chunk in cv2.inference_emo_film(
                text_with_emo=text_with_emo,
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav,
            ):
                torchaudio.save(out_wav, chunk["tts_speech"].cpu(), cv2.sample_rate)
                wrote_output = True
                break  # 只取第一个 chunk（非流式）
            if not wrote_output:
                raise RuntimeError(f"generation returned no audio for {utt_id}")
            dt = time.time() - t0
            results.append({"utt_id": utt_id, "wav_path": out_wav,
                            "prompt_wav": prompt_wav, "prompt_text": prompt_text,
                            "prompt_source": prompt_source, "status": "success",
                            "duration_s": dt})

        if save_every and len(results) % save_every == 0:
            _write_manifest(manifest_path, results)
            elapsed = time.perf_counter() - started_at
            LOGGER.info(
                "shard=%s progress=%s/%s elapsed=%.1fs avg_s_per_sample=%.2f",
                shard_idx,
                len(results),
                len(entries),
                elapsed,
                elapsed / len(results),
            )

    _write_manifest(manifest_path, results)
    print(f"Done. {len(results)}/{len(entries)} synthesized -> {output_dir}")
    print(f"Manifest -> {manifest_path}")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="pretrained CosyVoice2 dir")
    parser.add_argument("--llm_ckpt", required=True, help="trained LLM ckpt (final.pt)")
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--esd_root", required=True, help="ESD root for prompt selection")
    parser.add_argument("--workspace_root", default=ROOT,
                        help="workspace root for relative manifest paths")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="本进程处理的分片索引（多 GPU 数据并行）")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="总分片数（=GPU 数）；>1 时 manifest 带 .shard{idx}")
    parser.add_argument("--skip_existing", action="store_true",
                        help="out_wav 已存在则跳过合成，支持断点续跑")
    parser.add_argument("--save_every", type=int, default=50,
                        help="每 N 条增量覆盖写 manifest（0 关闭，仍会最终写一次）")
    tagged = parser.add_mutually_exclusive_group()
    tagged.add_argument("--use_tagged_text", dest="use_tagged_text", action="store_true",
                        help="用 tagged（<emotion> 词级标签，默认）")
    tagged.add_argument("--plain_text", dest="use_tagged_text", action="store_false",
                        help="用 plain_text（无词级标签）")
    parser.set_defaults(use_tagged_text=True)
    args = parser.parse_args()

    cv2 = load_emofilm_model(args.model_dir, args.llm_ckpt, fp16=args.fp16, device=args.device)
    run_inference(cv2, args.test_manifest, args.esd_root, args.output_dir,
                  use_tagged_text=args.use_tagged_text, max_samples=args.max_samples,
                  shard_idx=args.shard_idx, num_shards=args.num_shards,
                  skip_existing=args.skip_existing, save_every=args.save_every,
                  workspace_root=args.workspace_root)


if __name__ == "__main__":
    main()
