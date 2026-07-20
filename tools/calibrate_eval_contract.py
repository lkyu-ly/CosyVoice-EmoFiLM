#!/usr/bin/env python3
"""评测契约校准 CLI（设计 §10 "恒等对照 + 受控扰动"）。

生成一条参考音频的 4 种确定性变体，对每对 (ref, variant) 调用 v2 纯指标函数，
输出 calibration.json。identity 对做硬校验：Emo-SIM 必须 ∈ [100±1e-4]，
所有 DTW 分量必须为零——否则立即退出非零。

这不是论文阈值校准，而是指标实现正确性的自检：
- identity (原音频 vs 原音频)：特征提取确定性 → 恒等指标必须精确为零偏差
- append-silence-0.5s：尾部追加 0.5s 静音，测尾部不变性
- crop-tail-20pct：裁掉尾部 20%（keep_ratio=0.8），测截断鲁棒性
- stretch-1.25x：时间拉伸 1.25 倍，测时长变化鲁棒性

identity WER 使用显式 --reference_text（ground-truth 文本）与 Whisper 转写结果对比，
衡量真实转写质量——禁止用 ASR 自转写制造必然为零的假校准。

重模型 import 全部延迟到 run_calibration() 内部，保证 --help 不触发模型加载。

用法:
  python tools/calibrate_eval_contract.py \\
    --input_wav ref.wav \\
    --output_dir exp/diagnostics/<run_id>/metric_calibration \\
    --device cuda \\
    --reference_text "hello world"
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 受控扰动 helpers（纯函数，仅依赖 numpy）
# ============================================================

def append_silence(wav, sample_rate, seconds):
    """尾部追加指定秒数的静音（零值），返回新数组。

    wav: 1D ndarray；sample_rate: Hz；seconds: 追加秒数。
    """
    n_silence = int(sample_rate * seconds)
    silence = np.zeros(n_silence, dtype=wav.dtype)
    return np.concatenate([wav, silence])


def crop_tail(wav, keep_ratio):
    """保留前 keep_ratio 比例的样本，裁掉尾部。

    keep_ratio ∈ (0, 1]；keep_ratio=0.0 或裁剪后为空 → ValueError。
    """
    n = int(len(wav) * keep_ratio)
    if n == 0:
        raise ValueError(
            f"crop_tail 产出空数组 (len={len(wav)}, keep_ratio={keep_ratio})"
        )
    return wav[:n]


def time_stretch(wav, factor):
    """按 factor 线性插值重采样，改变时长但不改变音高。

    factor > 1 → 变长（变慢）；factor < 1 → 变短（变快）；factor=1.0 → 恒等。
    """
    new_len = int(len(wav) * factor)
    if new_len == 0:
        raise ValueError(
            f"time_stretch 产出空数组 (len={len(wav)}, factor={factor})"
        )
    old_indices = np.arange(len(wav))
    new_indices = np.linspace(0, len(wav) - 1, new_len)
    return np.interp(new_indices, old_indices, wav).astype(wav.dtype)


# ============================================================
# 变体生成
# ============================================================

def generate_variants(wav, sample_rate):
    """生成 4 种确定性变体，返回 {name: wav_array} 字典。

    变体列表（固定）：
    - identity: 原音频（恒等对照）
    - append_silence_0.5s: 尾部追加 0.5s 静音
    - crop_tail_20pct: 保留前 80%（裁掉尾部 20%）
    - stretch_1.25x: 时间拉伸 1.25 倍
    """
    return {
        "identity": wav,
        "append_silence_0.5s": append_silence(wav, sample_rate, 0.5),
        "crop_tail_20pct": crop_tail(wav, keep_ratio=0.8),
        "stretch_1.25x": time_stretch(wav, factor=1.25),
    }


# ============================================================
# CLI
# ============================================================

def build_arg_parser():
    """构建 CLI parser。--help 仅解析参数，不加载任何模型。"""
    parser = argparse.ArgumentParser(
        description="评测契约校准：生成受控扰动变体，计算 v2 指标，输出 calibration.json"
    )
    parser.add_argument("--input_wav", type=str, required=True,
                        help="参考音频路径（wav）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录（calibration.json 及变体 wav 写入此目录）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="模型设备（cuda / cpu）")
    parser.add_argument("--reference_text", type=str, required=True,
                        help="identity WER 参考文本（ground-truth）；"
                             "禁止用 ASR 转写自身制造必然为零的假校准")
    return parser


def _save_variant_wavs(variant_wavs, sample_rate, output_dir):
    """将非 identity 变体写为 wav 文件；identity 直接复用 input_wav 路径。"""
    import torch
    import torchaudio

    for name, wav_arr in variant_wavs.items():
        if name == "identity":
            continue  # identity 不重复保存，直接用 input_wav
        path = os.path.join(output_dir, f"{name}.wav")
        tensor = torch.from_numpy(np.asarray(wav_arr)).float().unsqueeze(0)
        torchaudio.save(path, tensor, sample_rate)


def _check_identity_hard_fail(identity_metrics):
    """identity 硬校验：Emo-SIM ∉ [100±1e-4] 或任一 DTW ≠ 0 → 退出非零。"""
    emo_sim = identity_metrics["emo_sim"]
    if abs(emo_sim - 100.0) > 1e-4:
        print(
            f"FAIL: identity Emo-SIM = {emo_sim}，超出 [100±1e-4]",
            file=sys.stderr,
        )
        sys.exit(1)

    dtw_keys = ("dtw", "dtw_normalized", "dtw_euclidean", "dtw_euclidean_normalized")
    for key in dtw_keys:
        val = identity_metrics[key]
        if abs(val) > 1e-4:
            print(
                f"FAIL: identity {key} = {val}，必须为零",
                file=sys.stderr,
            )
            sys.exit(1)


def run_calibration(input_wav, output_dir, device, reference_text):
    """执行校准主流程：加载模型 → 生成变体 → 计算指标 → 硬校验 → 写 JSON。

    重模型 import 全部在本函数内，保证模块导入（含 --help）不触发加载。
    """
    import torch
    import torchaudio

    # eval/ 加入 sys.path，使 from eval_emo_film import ... 在 CLI 独立运行时可用
    sys.path.insert(0, os.path.join(ROOT, "eval"))

    from eval_emo_film import (
        extract_frame_embeddings,
        compute_frame_mean_emo_sim,
        compute_dtw_metrics,
        compute_wer,
        normalize_text,
    )
    from funasr import AutoModel
    import whisper

    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    emo_model = AutoModel(
        model="iic/emotion2vec_plus_large",
        disable_update=True,
        device=device,
    )
    whisper_model = whisper.load_model("large-v3", device=device)

    # 读取输入 wav
    wav_tensor, sr = torchaudio.load(input_wav)  # (1, T)
    wav_np = wav_tensor.squeeze(0).numpy()

    # 生成变体
    variant_wavs = generate_variants(wav_np, sr)

    # 保存变体 wav（identity 复用 input_wav 路径）
    variant_paths = {"identity": input_wav}
    _save_variant_wavs(variant_wavs, sr, output_dir)
    for name in variant_wavs:
        if name != "identity":
            variant_paths[name] = os.path.join(output_dir, f"{name}.wav")

    # 批量提取 frame features：ref (input_wav) + 4 个 hyp 变体
    variant_names = list(variant_wavs.keys())
    all_paths = [input_wav] + [variant_paths[n] for n in variant_names]
    all_feats = extract_frame_embeddings(emo_model, all_paths)
    ref_feats = all_feats[0]

    # WER 计算函数（延迟 import jiwer）
    try:
        from jiwer import wer as wer_fn
    except ImportError:
        def wer_fn(ref, hyp):
            raise ImportError("jiwer 未安装，无法计算 WER")

    # 逐变体计算指标
    results = {}
    for idx, name in enumerate(variant_names):
        hyp_feats = all_feats[idx + 1]
        emo_sim = compute_frame_mean_emo_sim(ref_feats, hyp_feats)
        dtw_metrics = compute_dtw_metrics(ref_feats, hyp_feats)

        # WER: --reference_text vs Whisper 转写变体音频
        hyp_text = compute_wer(whisper_model, variant_paths[name])
        wer_value = wer_fn(
            normalize_text(reference_text),
            normalize_text(hyp_text),
        )

        results[name] = {
            "emo_sim": emo_sim,
            **dtw_metrics,
            "wer": float(wer_value),
            "hyp_text": hyp_text,
        }

    # identity 硬校验（不通过则 sys.exit(1)）
    _check_identity_hard_fail(results["identity"])

    # 写 calibration.json（exclusive create：文件已存在则报错）
    calibration = {
        "input_wav": os.path.abspath(input_wav),
        "sample_rate": sr,
        "reference_text": reference_text,
        "variants": results,
    }
    out_path = os.path.join(output_dir, "calibration.json")
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)

    print(f"校准通过。结果已写入 {out_path}")


def main():
    args = build_arg_parser().parse_args()
    run_calibration(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        device=args.device,
        reference_text=args.reference_text,
    )


if __name__ == "__main__":
    main()
