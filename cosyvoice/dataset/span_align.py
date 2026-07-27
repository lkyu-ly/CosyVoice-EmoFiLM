#!/usr/bin/env python3
"""EmoFiLM 监督 span → speech-token 对齐（纯函数）。

本模块是 EmoFiLM 主线的时间→token 对齐层（ADR-0020 扁平化后；原 v2 修复
模块去后缀为正式模块，v1 无对应）。它把监督 span 生成器产出的
SupervisionSpan 的 ``[start_sec, end_sec]`` 稳定映射到 teacher-forced
speech-token 序列上的 token 区间，供下游在生成因果链按 span 池化 hidden state。

设计要点（MAP.md §2 speech-token / §3 supervision-span 不变量；brief 03；issue 03）：

- **纯函数，CPU 可测，不依赖 GPU**：核心 ``align_spans_to_tokens`` 仅用 stdlib；
  ``collate_aligned_spans`` 用 torch CPU 张量（与下游训练侧兼容）。

- **speech-token 帧率 = 25 Hz（审计确认，多处来源一致）**：
    * ``conf/emo_film.yaml:20``: ``token_frame_rate: 25``
    * ``cosyvoice/dataset/processor.py:193`` 注释："align speech to 25hz first"
    * ``cosyvoice/cli/frontend.py:176-177``: ``token_len = min(speech_feat.shape[1] / 2,
      speech_token.shape[1])``，speech_feat 在 50 Hz（emotion2vec mel），``token_mel_ratio=2``
      → speech token = 50/2 = 25 Hz。
    * ``cosyvoice/flow/flow.py:126``: ``mel_len = token_len / input_frame_rate * 22050 / 256``，
      ``input_frame_rate=25``（token_frame_rate）→ 用 25 Hz 把 token 数换算成时长。
    * ``cosyvoice/flow/flow.py:32/156/291``: ``input_frame_rate: int = 50`` 是 flow 的**mel**
      帧率默认值，但 yaml 用 ``!ref <token_frame_rate>``（=25）覆盖；不要混淆两者。
  **关键区分**：IEMOCAP span 里的 ``frame_rate_hz: 50.0`` 是 word-block emotion2vec 特征率
  （50 Hz），**不是** speech-token 率。本函数**只**用调用方传入的 ``token_frame_rate_hz``
  参数（训练入口会传 25.0），**绝不**读 ``span["frame_rate_hz"]``——那是另一个空间。

- **映射规则**：``tok = round(time_sec * token_frame_rate_hz)``，clip 到 ``[0, speech_token_len)``；
  保证单调（``tok_end_prev <= tok_start_next``，允许相邻接）；``tok_end`` 为 Python 切片式
  exclusive end（``hidden[:, tok_start:tok_end, :]``）。

- **utterance-level span 覆盖全部有效 speech-token 列** ``[0, speech_token_len)``，不覆盖
  padding / 文本前缀 / 特殊 token（这些由训练侧 ``IGNORE_ID`` 标记，本函数只关心
  speech-token 段）。

- **fail-fast / 明确无效 mask**：空 span（``end<=start``）、反向边界、越界（整段在音频外）、
  零 token 覆盖（区间映射后 ``tok_start>=tok_end``）→ ``valid=False`` + 明确 ``invalid_reason``；
  **禁止静默扩展到整句**。调用方可据此 mask 或拒绝样本。

- **透传监督字段**：每条对齐结果独立携带 ``emotion_mask, intensity_mask,
  emotion_soft_distribution, arousal, raw_score, calibrated, supervision_weight,
  control_emotion_id, control_intensity_id``（透传自监督 span，保持合同校验语义）。

- 不实现模型 / 任务头（归训练侧）。
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


# ============================================================
# 常量与 strategy_version
# ============================================================

#: 对齐策略标识。bumped 当映射规则发生 breaking 变化（如 round→floor、clip 策略变更）。
STRATEGY_ID = "emofilm_span_align.v1"

#: 默认 speech-token 帧率（来源见模块 docstring；调用方应显式传 token_frame_rate_hz）。
DEFAULT_TOKEN_FRAME_RATE_HZ = 25.0


def build_strategy_version(token_frame_rate_hz: float) -> str:
    """构造可审计的 strategy_version 字符串，记录帧率与映射规则。

    含 ``token_frame_rate_hz`` 以便下游从对齐结果反查帧率来源；调用方应把帧率来源
    （如 ``conf/emo_film.yaml:20``）记入更上层 provenance。
    """
    return (
        f"{STRATEGY_ID}"
        f"|token_frame_rate_hz={float(token_frame_rate_hz):g}"
        f"|rule=time_sec->tok:round_then_clip[0,N)_then_monotonic_clamp_tok_start>=prev_tok_end"
        f"|quant_error=max(|err_start|,|err_end|)_sec"
        f"|tok_end=exclusive_slice_end"
    )


# ============================================================
# 核心对齐纯函数
# ============================================================


def _round_bankers(value: float) -> int:
    """Python 3 ``round``：banker's rounding（half-to-even），与 numpy/HALF_EVEN 一致。"""
    return round(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _align_one_span(
    span: Mapping[str, Any],
    *,
    speech_token_len: int,
    audio_duration_sec: float,
    token_frame_rate_hz: float,
    prev_tok_end: int,
    strategy_version: str,
) -> dict[str, Any]:
    """对齐单条 span；返回携带对齐 + 透传字段的 dict（valid 可能 False）。"""
    utt_id = str(span.get("utt_id", ""))
    granularity = str(span.get("supervision_granularity", ""))
    start_sec = _safe_float(span.get("start_sec"), float("nan"))
    end_sec = _safe_float(span.get("end_sec"), float("nan"))

    # 透传监督字段（合同的诚实语义）
    emotion_mask = bool(span.get("emotion_mask", False))
    intensity_mask = bool(span.get("intensity_mask", False))
    calibrated = bool(span.get("calibrated", False))
    supervision_weight = _safe_float(span.get("supervision_weight"), 1.0)
    control_emotion_id = span.get("control_emotion_id", 0)
    control_intensity_id = span.get("control_intensity_id", 0)
    emotion_soft_distribution = span.get("emotion_soft_distribution")
    arousal = span.get("arousal")
    raw_score = span.get("raw_score")

    base: dict[str, Any] = {
        "utt_id": utt_id,
        "supervision_granularity": granularity,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "strategy_version": strategy_version,
        "emotion_mask": emotion_mask,
        "intensity_mask": intensity_mask,
        "calibrated": calibrated,
        "supervision_weight": supervision_weight,
        "control_emotion_id": control_emotion_id,
        "control_intensity_id": control_intensity_id,
        "emotion_soft_distribution": (
            [float(p) for p in emotion_soft_distribution]
            if emotion_soft_distribution is not None else None
        ),
        "arousal": float(arousal) if arousal is not None else None,
        "raw_score": float(raw_score) if raw_score is not None else None,
    }

    # --- fail-fast：反向 / 空 span ---
    if not (end_sec > start_sec):
        base.update({
            "tok_start": 0, "tok_end": 0, "quant_error_sec": 0.0,
            "valid": False, "invalid_reason": "empty_or_reversed_boundary",
        })
        return base

    # --- fail-fast：整段在音频外（越界） ---
    if end_sec <= 0.0 or start_sec >= audio_duration_sec:
        base.update({
            "tok_start": 0, "tok_end": 0, "quant_error_sec": 0.0,
            "valid": False, "invalid_reason": "out_of_range",
        })
        return base

    # --- utterance-level span：覆盖全部有效 speech-token 列 [0, N) ---
    if granularity == "utterance":
        base.update({
            "tok_start": 0,
            "tok_end": speech_token_len,
            "quant_error_sec": 0.0,  # 全覆盖，无量化误差
            "valid": True,
            "invalid_reason": None,
        })
        return base

    # --- word / span level：时间→token 映射 ---
    ideal_tok_start = start_sec * token_frame_rate_hz
    ideal_tok_end = end_sec * token_frame_rate_hz
    raw_tok_start = _round_bankers(ideal_tok_start)
    raw_tok_end = _round_bankers(ideal_tok_end)

    # clip 到 [0, speech_token_len)
    clamped_tok_start = max(0, min(raw_tok_start, speech_token_len))
    clamped_tok_end = max(0, min(raw_tok_end, speech_token_len))

    # 单调保证：tok_start >= prev_tok_end（允许相邻接 ==）
    monotonic_tok_start = max(clamped_tok_start, prev_tok_end)
    tok_end = clamped_tok_end
    # tok_end 也至少不低于（单调后）tok_start
    if tok_end < monotonic_tok_start:
        tok_end = monotonic_tok_start

    # 量化误差：实际 token 边界回译秒与原始秒之差（取两端最大绝对值）
    err_start = abs(monotonic_tok_start / token_frame_rate_hz - start_sec)
    err_end = abs(tok_end / token_frame_rate_hz - end_sec)
    quant_error_sec = max(err_start, err_end)

    tok_start = monotonic_tok_start

    # --- fail-fast：零 token 覆盖 ---
    if tok_start >= tok_end:
        base.update({
            "tok_start": tok_start, "tok_end": tok_end,
            "quant_error_sec": quant_error_sec,
            "valid": False,
            "invalid_reason": "zero_coverage_after_clip_and_monotonic",
        })
        return base

    base.update({
        "tok_start": tok_start,
        "tok_end": tok_end,
        "quant_error_sec": quant_error_sec,
        "valid": True,
        "invalid_reason": None,
    })
    return base


def align_spans_to_tokens(
    spans: list[Mapping[str, Any]],
    speech_token_len: int,
    audio_duration_sec: float,
    token_frame_rate_hz: float,
    *,
    strategy_version: str,
) -> list[dict[str, Any]]:
    """把 SupervisionSpan 的 ``[start_sec, end_sec]`` 映射到 speech-token 区间。

    Args:
        spans: 监督 span 生成器产出的 SupervisionSpan 列表（按时间序）。
        speech_token_len: 本条音频的有效 speech-token 长度（不含 padding/文本前缀/特殊 token）。
        audio_duration_sec: 本条音频的有效时长（秒）。
        token_frame_rate_hz: speech-token 帧率（CosyVoice2 = 25 Hz；见模块 docstring 来源）。
        strategy_version: 调用方传入的映射策略版本（用 ``build_strategy_version`` 构造）。

    Returns:
        每条 span 一个 dict，携带 ``tok_start, tok_end, quant_error_sec, valid,
        invalid_reason, strategy_version`` 与透传监督字段。输出按输入顺序（时间序）单调。

    不变量：
        - ``tok = round(time_sec * token_frame_rate_hz)``，clip 到 ``[0, speech_token_len)``。
        - 单调：``result[i]["tok_end"] <= result[i+1]["tok_start"]``（允许相邻接 ==）。
        - utterance-level span → ``[0, speech_token_len)``。
        - 空/反向/越界/零覆盖 → ``valid=False`` + ``invalid_reason``，**绝不静默扩展到整句**。
        - ``tok_end`` 为 exclusive 切片端（``hidden[:, tok_start:tok_end, :]``）。
    """
    if speech_token_len <= 0:
        raise ValueError(
            f"speech_token_len must be > 0 (got {speech_token_len}); "
            "alignment requires at least one valid speech-token column"
        )
    if audio_duration_sec <= 0.0:
        raise ValueError(
            f"audio_duration_sec must be > 0 (got {audio_duration_sec})"
        )
    if token_frame_rate_hz <= 0.0:
        raise ValueError(
            f"token_frame_rate_hz must be > 0 (got {token_frame_rate_hz})"
        )

    results: list[dict[str, Any]] = []
    prev_tok_end = 0  # 单调下界：首 span tok_start 至少为 0
    for span in spans:
        aligned = _align_one_span(
            span,
            speech_token_len=speech_token_len,
            audio_duration_sec=audio_duration_sec,
            token_frame_rate_hz=token_frame_rate_hz,
            prev_tok_end=prev_tok_end,
            strategy_version=strategy_version,
        )
        # 更新 prev_tok_end 用于下一 span 的单调约束（即使本 span 无效，仍推进下界
        # 以避免后续 span 倒退；对无效 span 用其（已 clamp 的）tok_end）。
        if aligned["tok_end"] > prev_tok_end:
            prev_tok_end = aligned["tok_end"]
        results.append(aligned)
    return results


# ============================================================
# collate：可变 span 数 batch（一一对应）
# ============================================================


def _pad_or_none(value: Any) -> Any:
    return value


def collate_aligned_spans(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """把多条样本的 aligned_spans 拼成 batch 张量，保持 span ↔ mask ↔ target ↔ source 1-1。

    每条 sample 形如 ``{"utt_id": str, "aligned_spans": list[dict]}``（dict 字段由
    ``align_spans_to_tokens`` 产出）。支持不同样本的可变 span 数：pad span 维度到
    ``max_spans``，并产 ``span_mask`` 标识真实 span（非 padding）。

    Returns:
        batch dict（torch CPU 张量），键：
            - ``span_mask``: (B, max_spans) bool — 真实 span（非 padding）。
            - ``span_valid``: (B, max_spans) bool — 对齐有效（非 invalid）。
            - ``span_tok_start`` / ``span_tok_end``: (B, max_spans) long.
            - ``span_emotion_mask`` / ``span_intensity_mask`` / ``span_calibrated``:
              (B, max_spans) bool.
            - ``span_emotion_soft_dist``: (B, max_spans, 5) float（None→zeros）。
            - ``span_arousal`` / ``span_raw_score`` / ``span_supervision_weight``:
              (B, max_spans) float（None→0.0）。
            - ``span_control_emotion_id`` / ``span_control_intensity_id``:
              (B, max_spans) long.
            - ``span_invalid_reason``: (B, max_spans) list[list[Optional[str]]]（非张量）。
            - ``span_count``: (B,) long.
            - ``utt_id``: list[str].
            - ``strategy_version``: str（所有 span 应一致；取首条）。
    """
    import torch

    if not isinstance(samples, (list, tuple)) or len(samples) == 0:
        raise ValueError("samples must be a non-empty list of {utt_id, aligned_spans}")

    utt_ids: list[str] = []
    per_sample_spans: list[list[dict[str, Any]]] = []
    strategy_version: Optional[str] = None
    for sample in samples:
        utt_ids.append(str(sample.get("utt_id", "")))
        spans = list(sample.get("aligned_spans", []) or [])
        per_sample_spans.append(spans)
        if strategy_version is None and spans:
            strategy_version = spans[0].get("strategy_version")

    batch_size = len(samples)
    max_spans = max((len(s) for s in per_sample_spans), default=0)

    def _tensor(dtype: torch.dtype, fill: Any) -> torch.Tensor:
        if max_spans == 0:
            return torch.empty((batch_size, 0), dtype=dtype)
        return torch.full((batch_size, max_spans), fill, dtype=dtype)

    def _tensor_3d(dim3: int, fill: float) -> torch.Tensor:
        if max_spans == 0:
            return torch.empty((batch_size, 0, dim3), dtype=torch.float32)
        return torch.full((batch_size, max_spans, dim3), fill, dtype=torch.float32)

    span_mask = _tensor(torch.bool, False)
    span_valid = _tensor(torch.bool, False)
    span_tok_start = _tensor(torch.long, 0)
    span_tok_end = _tensor(torch.long, 0)
    span_emotion_mask = _tensor(torch.bool, False)
    span_intensity_mask = _tensor(torch.bool, False)
    span_calibrated = _tensor(torch.bool, False)
    span_emotion_soft_dist = _tensor_3d(5, 0.0)
    span_arousal = _tensor(torch.float32, 0.0)
    span_raw_score = _tensor(torch.float32, 0.0)
    span_supervision_weight = _tensor(torch.float32, 0.0)
    span_control_emotion_id = _tensor(torch.long, 0)
    span_control_intensity_id = _tensor(torch.long, 0)
    span_count = torch.zeros((batch_size,), dtype=torch.long)
    span_invalid_reason: list[list[Optional[str]]] = [
        [None] * max_spans for _ in range(batch_size)
    ]

    for i, spans in enumerate(per_sample_spans):
        span_count[i] = len(spans)
        for j, a in enumerate(spans):
            span_mask[i, j] = True
            span_valid[i, j] = bool(a.get("valid", False))
            span_tok_start[i, j] = int(a.get("tok_start", 0))
            span_tok_end[i, j] = int(a.get("tok_end", 0))
            span_emotion_mask[i, j] = bool(a.get("emotion_mask", False))
            span_intensity_mask[i, j] = bool(a.get("intensity_mask", False))
            span_calibrated[i, j] = bool(a.get("calibrated", False))
            dist = a.get("emotion_soft_distribution")
            if dist is not None:
                for k in range(min(5, len(dist))):
                    span_emotion_soft_dist[i, j, k] = float(dist[k])
            arousal = a.get("arousal")
            if arousal is not None:
                span_arousal[i, j] = float(arousal)
            rs = a.get("raw_score")
            if rs is not None:
                span_raw_score[i, j] = float(rs)
            span_supervision_weight[i, j] = float(a.get("supervision_weight", 0.0))
            ceid = a.get("control_emotion_id", 0)
            span_control_emotion_id[i, j] = int(ceid) if ceid is not None else 0
            ciid = a.get("control_intensity_id", 0)
            span_control_intensity_id[i, j] = int(ciid) if ciid is not None else 0
            span_invalid_reason[i][j] = a.get("invalid_reason")

    return {
        "utt_id": utt_ids,
        "span_mask": span_mask,
        "span_valid": span_valid,
        "span_tok_start": span_tok_start,
        "span_tok_end": span_tok_end,
        "span_emotion_mask": span_emotion_mask,
        "span_intensity_mask": span_intensity_mask,
        "span_calibrated": span_calibrated,
        "span_emotion_soft_dist": span_emotion_soft_dist,
        "span_arousal": span_arousal,
        "span_raw_score": span_raw_score,
        "span_supervision_weight": span_supervision_weight,
        "span_control_emotion_id": span_control_emotion_id,
        "span_control_intensity_id": span_control_intensity_id,
        "span_count": span_count,
        "span_invalid_reason": span_invalid_reason,
        "strategy_version": strategy_version,
        "max_spans": max_spans,
    }


__all__ = [
    "STRATEGY_ID",
    "DEFAULT_TOKEN_FRAME_RATE_HZ",
    "build_strategy_version",
    "align_spans_to_tokens",
    "collate_aligned_spans",
]
