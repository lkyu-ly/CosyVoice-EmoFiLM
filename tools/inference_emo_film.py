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

manifest 每行是一条合法 GenerationRow（utt_id/finish_reason/source_revision/
checkpoint_sha256/decode_config/seed/control_row_ref/prompt_row_ref/wav_path）。
仅 finish_reason=eos 的 row 携 wav_path；非 eos 仅诊断 row。per-utt 生成前重置
torch+cuda RNG（seed 可复现）。skip_existing 基于逐条身份指纹（含 seed）比对。
"""
import argparse
import hashlib
import json
import logging
import os
import subprocess
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

from tools.build_emofilm_contract import validate_generation_row  # noqa: E402
from tools.write_emofilm_run_identity import (  # noqa: E402
    check_skip_existing,
    generation_request_fingerprint,
    sha256_file,
)


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

    v2 单流协议下 ``prompt_text`` 不进 LLM 条件（仅 target text + emotion/intensity
    控制），因此不再强制必填——缺失时回退空串（声学侧 prompt conditioning 仍由
    prompt_wav 提供）。Returns dict with ok/status/prompt_wav/prompt_text/prompt_source。
    """
    prompt_wav = utt.get("prompt_wav")
    prompt_text = utt.get("prompt_text") or ""
    if prompt_wav:
        prompt_path = Path(prompt_wav)
        if not prompt_path.is_absolute() and workspace_root is not None:
            prompt_path = Path(workspace_root) / prompt_path
        prompt_path = prompt_path.resolve()
        if prompt_path.is_file():
            return {"ok": True, "prompt_wav": str(prompt_path),
                    "prompt_text": prompt_text,
                    "prompt_source": "manifest", "status": "success"}
        raise FileNotFoundError(f"prompt_wav not found: {prompt_path}")

    part = utt.get("part")
    if part == "A":
        raise FileNotFoundError("FEDD Part A requires manifest prompt_wav")

    # ESD / FEDD Part B：回退到 ESD same-speaker Neutral
    prompt_wav = select_prompt_wav(utt, esd_root)
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
                  workspace_root=None,
                  seed=1986, decode_config=None,
                  source_revision=None, llm_ckpt_sha=None):
    """批量推理，返回 GenerationRow 列表。

    多 GPU 数据并行：每个进程在 max_samples 截取后处理 entries[shard_idx::num_shards]。

    关键不变量（Task 4 / #5）：
    - **per-utt RNG 重置**：每个 utt 生成前 ``torch.manual_seed(seed)`` +
      ``torch.cuda.manual_seed_all(seed)``，保证同 seed + 同输入 → 输出可复现。
    - **仅 eos 落 WAV**：取首 chunk 的 ``finish_reason``（T1 暴露）；eos+audio 才
      ``torchaudio.save`` + 携 ``wav_path``；非 eos 仅诊断 row（无 ``wav_path``）。
    - **身份完整**：row 含 ``source_revision`` / ``checkpoint_sha256`` /
      ``control_row_ref`` / ``prompt_row_ref`` / ``decode_config`` / ``seed``
      四族身份 + seed，可通过 ``validate_generation_row``。
    - **安全 skip**：``skip_existing=True`` 时既有 manifest row 的逐条身份指纹
      （含 seed）与请求指纹比对；seed 或任何身份变→指纹不同→不 skip（重生成）。
      既有 row 无身份（v1 manifest）→当作无 existing（重生成）。

    ``decode_config`` 回退优先级：显式参数 > ``cv2.decode_config``（T2 抽取）。
    """
    # decode_config 回退：参数 > cv2.decode_config（T2 从 yaml 抽取到实例属性）
    if decode_config is None:
        decode_config = getattr(cv2, "decode_config", None)

    os.makedirs(output_dir, exist_ok=True)
    with open(test_manifest) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    if max_samples:
        entries = entries[:max_samples]
    entries = entries[shard_idx::num_shards]

    manifest_path = _manifest_path_for(output_dir, shard_idx, num_shards)

    # 加载既有 manifest rows 用于 identity-based skip（既有 row 无身份→不 skip）
    existing_rows_by_id: dict[str, dict] = {}
    if skip_existing and os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                    uid = existing.get("utt_id")
                    if uid:
                        existing_rows_by_id[uid] = existing
                except json.JSONDecodeError:
                    pass

    results = []
    started_at = time.perf_counter()
    ref_root = workspace_root or os.path.dirname(output_dir.rstrip("/"))
    for utt in tqdm(entries, desc="Emo-FiLM infer"):
        utt_id = utt["utt_id"]
        out_wav = os.path.join(output_dir, f"{utt_id}.wav")

        # 合成输入（提前计算，供 skip 指纹与生成本身共用——B6 要求 skip 指纹也
        # 反映实际合成内容，否则改 manifest 续跑会静默复用旧条件 WAV）。
        text_with_emo = _pick_text(utt, use_tagged_text)
        resolved = resolve_prompt(utt, esd_root, workspace_root=workspace_root)
        prompt_wav = resolved["prompt_wav"]
        prompt_text = resolved["prompt_text"]
        prompt_source = resolved["prompt_source"]

        # B6: 合成输入摘要进指纹（文本摘要 + prompt 音频 workspace-relative 身份）
        text_digest = hashlib.sha256(text_with_emo.encode("utf-8")).hexdigest()
        prompt_audio_ref = os.path.relpath(prompt_wav, ref_root).replace(os.sep, "/")

        # B5: control/prompt 身份引用——eval manifest 若无 ref 字段则就地合成
        # （稳定标识；内容变更由 text_digest/prompt_audio_ref 在指纹侧捕获）。
        control_row_ref = utt.get("control_row_ref") or f"control/{utt_id}"
        prompt_row_ref = utt.get("prompt_row_ref") or f"prompt/{Path(prompt_wav).name}"

        # --- 身份-based skip（仅完整逐条身份一致时复用）---
        skip_decision = None
        existing_row = None
        if skip_existing:
            request_fp = generation_request_fingerprint(
                source=source_revision,
                checkpoint_sha256=llm_ckpt_sha,
                control_row_ref=control_row_ref,
                prompt_row_ref=prompt_row_ref,
                decode_config=decode_config,
                seed=seed,
                text_digest=text_digest,
                prompt_audio_ref=prompt_audio_ref,
            )
            existing_row = existing_rows_by_id.get(utt_id)
            if existing_row is not None:
                skip_decision = check_skip_existing(
                    existing_row, request_fp, workspace_root=workspace_root,
                )

        if skip_decision is not None and skip_decision.skip:
            # 安全复用既有 row（已是合法 GenerationRow）
            skipped_row = dict(existing_row)
            skipped_row["status"] = "skipped_existing"
            results.append(skipped_row)
        else:
            if skip_decision is not None:
                LOGGER.info("utt=%s skip rejected: %s", utt_id, skip_decision.reason)

            # per-utt RNG 重置（per-request 固定 seed → 可复现）
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            t0 = time.time()

            # 取首 chunk 的 finish_reason（T1 在 model_emo.tts 门控后暴露）。
            # 非 eos 时 tts_speech=None（T1 yield），不调用 .cpu() 避免 AttributeError。
            finish_reason = "sampler_error"
            tts_speech = None
            for chunk in cv2.inference_emo_film(
                text_with_emo=text_with_emo,
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav,
            ):
                finish_reason = chunk.get("finish_reason", "sampler_error")
                tts_speech = chunk.get("tts_speech")
                break  # 只取首 chunk（非流式）

            dt = time.time() - t0

            # wav_path: workspace-relative POSIX
            wav_rel = os.path.relpath(out_wav, ref_root).replace(os.sep, "/")

            # GenerationRow 构造（四族身份 + seed + decode_config + 合成输入摘要）
            row: dict = {
                "utt_id": utt_id,
                "finish_reason": finish_reason,
                "source_revision": source_revision,
                "checkpoint_sha256": llm_ckpt_sha,
                "decode_config": decode_config,
                "seed": seed,
                "control_row_ref": control_row_ref,
                "prompt_row_ref": prompt_row_ref,
                "text_digest": text_digest,
                "prompt_audio_ref": prompt_audio_ref,
            }

            if finish_reason == "eos" and tts_speech is not None:
                torchaudio.save(out_wav, tts_speech.cpu(), cv2.sample_rate)
                row["wav_path"] = wav_rel
                row["prompt_source"] = prompt_source
                row["duration_s"] = dt
                row["status"] = "success"
            else:
                # B12: 非 eos 清除可能残留的同名旧 WAV——保证目录内容 == manifest
                # eos 集合（v1 回归门按目录 glob 配对，旧 WAV 会混源污染）。
                if os.path.isfile(out_wav):
                    os.remove(out_wav)
                    LOGGER.warning(
                        "utt=%s removed stale WAV before non-eos diagnostic",
                        utt_id,
                    )
                # 非 eos 仅诊断 row（schema 强制不得携 wav_path）
                row["prompt_source"] = prompt_source
                row["duration_s"] = dt
                row["status"] = "non_eos_diagnostic"
                LOGGER.warning(
                    "utt=%s finish_reason=%s (no WAV written; "
                    "only eos enters acoustics)",
                    utt_id, finish_reason,
                )

            # B5: 写盘前合同自检——不合格携 utt_id fail-fast（不在评测端才崩）。
            validate_generation_row(row)
            results.append(row)

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
    parser.add_argument("--seed", type=int, default=1986,
                        help="per-request 固定随机种子（per-utt 重置 torch+cuda RNG）")
    tagged = parser.add_mutually_exclusive_group()
    tagged.add_argument("--use_tagged_text", dest="use_tagged_text", action="store_true",
                        help="用 tagged（<emotion> 词级标签，默认）")
    tagged.add_argument("--plain_text", dest="use_tagged_text", action="store_false",
                        help="用 plain_text（无词级标签）")
    parser.set_defaults(use_tagged_text=True)
    args = parser.parse_args()

    cv2 = load_emofilm_model(args.model_dir, args.llm_ckpt, fp16=args.fp16, device=args.device)

    # 源码身份：git HEAD（干净 revision）
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()

    # checkpoint 身份：llm_ckpt 内容 sha256
    llm_ckpt_sha = sha256_file(Path(args.llm_ckpt))

    run_inference(cv2, args.test_manifest, args.esd_root, args.output_dir,
                  use_tagged_text=args.use_tagged_text, max_samples=args.max_samples,
                  shard_idx=args.shard_idx, num_shards=args.num_shards,
                  skip_existing=args.skip_existing, save_every=args.save_every,
                  workspace_root=args.workspace_root,
                  seed=args.seed, source_revision=source_revision,
                  llm_ckpt_sha=llm_ckpt_sha)


if __name__ == "__main__":
    main()
