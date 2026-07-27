#!/usr/bin/env python3
"""EmoFiLM v2 FEDD 局部转场逐 span 评测（ticket 09）。

消费控制 manifest（FEDD construction records: emo_from/emo_to/
boundary_word_index/method/text）+ generation manifest（v2 GenerationRow:
finish_reason/WAV/身份）+ WAV，产出逐样本 / 逐 span EvaluationRow 与
按 evidence_tier 分离的 Aggregate。

与 v1 ``eval/eval_emo_film.py``（整体质量：Emo-SIM/DTW/WER）**互补**：
v1 保持只读，本模块聚焦 FEDD 局部转场控制质量（前后段 emotion 命中、
transition 方向、边界时间误差）。

设计要点（MAP §3 评测不变量 / brief 09）：
- **严格配对**：utt_id / checkpoint / 控制条件 / finish_reason / WAV 一一对应；
  任一缺失 / 重复 / 非 EOS / 身份不一致 / evaluator 失败 → 携 utt_id hard-fail，
  禁止跳过算部分均值（沿用 v1 ``pair_wavs_strict`` + 逐样本异常传播先例）。
- **FEDD-B（exact）**：前/后 emotion 命中、transition direction、相对精确构造
  词边界的时间误差。生成音频的目标词边界由**固定文本强制对齐**获得；无有效
  对齐 → 不伪造边界误差（null + reason）。
- **FEDD-A（approximate）**：保留 midpoint 近似标签，可报告趋势与整体质量，
  **不进** exact-boundary aggregate。
- **aggregate 确定性派生**：从已持久化 EvaluationRow 重算，分别输出 exact /
  approximate 两个 aggregate。
- **强制对齐为可注入接口**（``ForcedAligner`` Protocol）：测试用
  ``FakeForcedAligner``（``tests/_emofilm_fakes.py``，无 MFA / GPU 依赖）；
  真实 MFA 由 ``MfaForcedAligner`` 薄封装，不在测试中调用。

真实 MFA/GPU smoke 延后（spec Out of Scope）；不实现强度三元（归 ticket 10）。
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tools.build_emofilm_contract import validate_eval_row, validate_aggregate


# ============================================================
# 常量
# ============================================================

METRIC_CONTRACT_VERSION = "emofilm_v2_eval"

# FEDD 构造方法（来自 build_fedd_tagged_text.py）。
EXACT_METHOD = "exact_concatenation_boundary"
APPROX_METHOD = "midpoint_two_span_approximation"

# 方法 → 证据等级映射。
_METHOD_TO_TIER = {
    EXACT_METHOD: "exact",
    APPROX_METHOD: "approximate",
}


# ============================================================
# 强制对齐接口（可注入）
# ============================================================


@dataclass(frozen=True)
class WordBoundary:
    """单个词的强制对齐时间区间。"""
    start_sec: float
    end_sec: float
    word: str


@dataclass
class AlignmentResult:
    """强制对齐结果。

    Attributes:
        status: ``"aligned"``（成功）/ ``"failed"``（对齐失败或分数过低）/
            ``"not_attempted"``（未请求对齐，如 Part A 近似边界）。
        words: 词级时间边界列表（成功时按文本顺序）。
        reason: 失败原因（仅 status != aligned 时有意义）。
    """
    status: str
    words: list[WordBoundary] = field(default_factory=list)
    reason: str | None = None


@runtime_checkable
class ForcedAligner(Protocol):
    """强制对齐器接口（可注入；测试用 Fake，生产用 MFA wrapper）。

    实现者保证：
    - ``align(wav_path, text)`` 返回 ``AlignmentResult``；
    - 对齐失败时 ``status="failed"`` + ``reason`` 非空，而非抛异常
      （异常由调用方决定如何传播）。
    """

    def align(self, wav_path: str, text: str) -> AlignmentResult: ...


class MfaForcedAligner:
    """MFA 强制对齐薄封装（生产用；不在测试中调用）。

    封装 ``tools/run_mfa_align.py`` 的子进程调用 + TextGrid 解析。
    构造时不加载 MFA；对齐时临时建 corpus、跑 ``mfa align``、解析输出 TextGrid。

    **不在 CPU 合同测试中使用**（需要 MFA 二进制 + acoustic model）。
    """

    def __init__(
        self,
        mfa_bin: str | None = None,
        dictionary: str = "english_mfa",
        acoustic_model: str = "english_mfa",
        temp_dir: str = "/tmp/mfa_eval_v2",
    ):
        self._mfa_bin = mfa_bin
        self._dictionary = dictionary
        self._acoustic_model = acoustic_model
        self._temp_dir = temp_dir

    def align(self, wav_path: str, text: str) -> AlignmentResult:
        import os
        import shutil
        import subprocess
        from pathlib import Path

        try:
            from tools.run_mfa_align import resolve_mfa_bin, build_subprocess_env
        except Exception as exc:
            return AlignmentResult(
                status="failed", reason=f"mfa_import_error: {exc}",
            )

        try:
            mfa_bin = resolve_mfa_bin(self._mfa_bin)
        except FileNotFoundError as exc:
            return AlignmentResult(
                status="failed", reason=f"mfa_not_found: {exc}",
            )

        utt_id = Path(wav_path).stem
        corpus = Path(self._temp_dir) / f"corpus_{utt_id}"
        output = Path(self._temp_dir) / f"aligned_{utt_id}"
        if corpus.exists():
            shutil.rmtree(corpus)
        if output.exists():
            shutil.rmtree(output)
        corpus.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)

        shutil.copy2(wav_path, corpus / f"{utt_id}.wav")
        (corpus / f"{utt_id}.lab").write_text(text + "\n", encoding="utf-8")

        env = build_subprocess_env(mfa_bin)
        cmd = [
            mfa_bin, "align", str(corpus),
            self._dictionary, self._acoustic_model, str(output),
            "--num_jobs", "1", "--output_format", "long_textgrid",
            "--overwrite", "--clean",
        ]
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            return AlignmentResult(
                status="failed",
                reason=f"mfa_exit_{exc.returncode}: {exc.stderr[:200]}",
            )

        tg_path = output / f"{utt_id}.TextGrid"
        if not tg_path.is_file():
            return AlignmentResult(
                status="failed", reason="no_textgrid_output",
            )

        words = _parse_textgrid_words(tg_path)
        if not words:
            return AlignmentResult(
                status="failed", reason="empty_words_tier",
            )
        return AlignmentResult(status="aligned", words=words)


def _parse_textgrid_words(textgrid_path: str | Path) -> list[WordBoundary]:
    """极简 TextGrid 解析：返回 words tier 的非空 interval。"""
    import re
    content = Path(textgrid_path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r'name\s*=\s*"words"(.*?)(?:item\s*\[\d+\]:|\Z)', content, re.DOTALL)
    if not m:
        return []
    words = []
    for iv in re.finditer(
        r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*\"(.*?)\"",
        m.group(1), re.DOTALL,
    ):
        x0, x1, w = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip()
        if w:
            words.append(WordBoundary(start_sec=x0, end_sec=x1, word=w))
    return words


# ============================================================
# 核心评测逻辑（纯函数）
# ============================================================


def derive_evidence_tier(
    method: str | None = None,
    boundary_word_index: int | None = None,
    part: str | None = None,
) -> str:
    """从 FEDD 构造记录推断边界证据等级。

    优先使用 ``method`` 字段；其次 ``boundary_word_index``（非 None → exact）；
    最后 ``part``（"B" → exact, "A" → approximate）。

    Returns:
        ``"exact"`` 或 ``"approximate"``。
    """
    if method is not None:
        if method in _METHOD_TO_TIER:
            return _METHOD_TO_TIER[method]
        if "exact" in method.lower():
            return "exact"
        if "approx" in method.lower():
            return "approximate"
    if boundary_word_index is not None:
        return "exact"
    if part is not None:
        return "exact" if part.upper() == "B" else "approximate"
    return "approximate"


def evaluate_spans_from_frames(
    frames: np.ndarray,
    times_sec: np.ndarray,
    boundary_sec: float,
    emo_from: str,
    emo_to: str,
    label_space: list[str],
) -> dict[str, Any]:
    """在合成 / 真实 frame 轨迹上计算前 / 后 span emotion 命中与 transition direction。

    纯函数：不调用 evaluator、不读取文件。接收已提取的 ``(T, K)`` 分布矩阵。

    Args:
        frames: ``(T, K)`` 逐帧情感分布（K = 标签数）。
        times_sec: ``(T,)`` 帧中心时间戳。
        boundary_sec: 前 / 后 span 分割时刻。
        emo_from: 构造前段目标情感（标签名）。
        emo_to: 构造后段目标情感。
        label_space: 标签列表（与 frames 列顺序对齐）。

    Returns:
        包含 ``front`` / ``back`` / ``transition_direction`` / ``front_back_both_hit``
        / ``valid`` 的 metrics dict。Task #11: 空 frames 或全非有限（NaN/inf）
        → ``valid=False``，调用方据此不计入 aggregate 分母。
    """
    frames = np.asarray(frames, dtype=np.float64)
    times_sec = np.asarray(times_sec, dtype=np.float64)
    n_frames = len(frames)
    label_to_idx = {lab: i for i, lab in enumerate(label_space)}

    # Task #11: 空 / 全非有限 / 列数与 label_space 不对齐 → span metrics invalid。
    # NaN 进 argmax 行为未定义，且会在 mean 中传播，污染 front/back 命中率与 score。
    invalid_metrics: dict[str, Any] = {
        "front": {
            "target_emotion": emo_from,
            "predicted_emotion": None,
            "hit": False,
            "score": 0.0,
            "start_sec": 0.0,
            "end_sec": float(boundary_sec),
        },
        "back": {
            "target_emotion": emo_to,
            "predicted_emotion": None,
            "hit": False,
            "score": 0.0,
            "start_sec": float(boundary_sec),
            "end_sec": 0.0,
        },
        "transition_direction": "other",
        "front_back_both_hit": False,
        "valid": False,
    }
    if n_frames == 0:
        return invalid_metrics
    if frames.ndim != 2 or frames.shape[1] != len(label_space):
        return invalid_metrics
    if not np.isfinite(frames).any():
        return invalid_metrics

    # 分割帧索引：times_sec[i] < boundary_sec 的帧为 front，
    # times_sec[i] >= boundary_sec 的帧为 back。
    boundary_frame = int(np.searchsorted(times_sec, boundary_sec, side="left"))
    boundary_frame = max(1, min(boundary_frame, n_frames - 1))

    front_frames = frames[:boundary_frame]
    back_frames = frames[boundary_frame:]

    # 防 front / back 为空（boundary_frame == 0 或 == n_frames）
    if len(front_frames) == 0:
        front_frames = frames[:1]
    if len(back_frames) == 0:
        back_frames = frames[-1:]

    # 仅用全有限行做 mean（丢弃零星 NaN 行，全行 NaN 已被前面拦截）。
    front_finite = np.all(np.isfinite(front_frames), axis=1)
    back_finite = np.all(np.isfinite(back_frames), axis=1)
    if not front_finite.any() or not back_finite.any():
        return invalid_metrics
    front_mean = front_frames[front_finite].mean(axis=0)
    back_mean = back_frames[back_finite].mean(axis=0)

    front_pred_idx = int(np.argmax(front_mean))
    back_pred_idx = int(np.argmax(back_mean))
    front_pred = label_space[front_pred_idx]
    back_pred = label_space[back_pred_idx]

    front_hit = front_pred == emo_from
    back_hit = back_pred == emo_to

    from_idx = label_to_idx.get(emo_from)
    to_idx = label_to_idx.get(emo_to)
    front_score = float(front_mean[from_idx]) if from_idx is not None else 0.0
    back_score = float(back_mean[to_idx]) if to_idx is not None else 0.0

    # Transition direction 判定。
    if front_pred == emo_from and back_pred == emo_to:
        direction = "correct"
    elif front_pred == back_pred:
        direction = "maintained"
    elif front_pred == emo_to and back_pred == emo_from:
        direction = "reverse"
    else:
        direction = "other"

    return {
        "front": {
            "target_emotion": emo_from,
            "predicted_emotion": front_pred,
            "hit": bool(front_hit),
            "score": front_score,
            "start_sec": float(times_sec[0]),
            "end_sec": float(boundary_sec),
        },
        "back": {
            "target_emotion": emo_to,
            "predicted_emotion": back_pred,
            "hit": bool(back_hit),
            "score": back_score,
            "start_sec": float(boundary_sec),
            "end_sec": float(times_sec[-1]),
        },
        "transition_direction": direction,
        "front_back_both_hit": bool(front_hit and back_hit),
        "valid": True,
    }


def detect_transition_from_frames(
    frames: np.ndarray,
    times_sec: np.ndarray,
    from_label: str,
    to_label: str,
    label_space: list[str],
    frame_rate_hz: float = 50.0,
) -> dict[str, Any]:
    """在逐帧 argmax 轨迹中检测 from→to 情感切换时刻。

    检测策略：
    1. 找首个相邻帧 ``t-1 == from_idx, t == to_idx`` 的 ``t``（严格切换）。
    2. 退化：若无严格相邻切换，找首个 ``== to_idx`` 的帧。

    Args:
        frames: ``(T, K)`` 分布矩阵。
        times_sec: ``(T,)`` 帧时间戳。
        from_label: 切换前情感。
        to_label: 切换后情感。
        label_space: 标签列表。
        frame_rate_hz: 帧率（用于退化情况下将帧索引换算为秒）。

    Returns:
        ``{"detected": bool, "detected_sec": float | None, ...}``
    """
    frames = np.asarray(frames, dtype=np.float64)
    times_sec = np.asarray(times_sec, dtype=np.float64)
    label_to_idx = {lab: i for i, lab in enumerate(label_space)}

    from_idx = label_to_idx.get(from_label)
    to_idx = label_to_idx.get(to_label)
    if from_idx is None or to_idx is None:
        return {
            "detected": False,
            "detected_sec": None,
            "from_label": from_label,
            "to_label": to_label,
            "reason": "label_not_in_label_space",
        }

    per_frame = np.argmax(frames, axis=1) if len(frames) > 0 else np.array([], dtype=int)

    # 1) 严格相邻切换：per_frame[t-1] == from_idx and per_frame[t] == to_idx.
    detected_frame = None
    for t in range(1, len(per_frame)):
        if per_frame[t - 1] == from_idx and per_frame[t] == to_idx:
            detected_frame = t
            break

    # 2) 退化：首个 == to_idx 的帧.
    if detected_frame is None:
        to_frames = np.where(per_frame == to_idx)[0]
        # 仅当序列中确实存在 from 帧在 to 帧之前时才报告退化检测
        from_frames = np.where(per_frame == from_idx)[0]
        if len(to_frames) > 0 and len(from_frames) > 0:
            if from_frames[0] < to_frames[-1]:
                detected_frame = int(to_frames[0])

    if detected_frame is None:
        return {
            "detected": False,
            "detected_sec": None,
            "from_label": from_label,
            "to_label": to_label,
            "reason": "no_transition_in_argmax",
        }

    detected_sec = float(times_sec[detected_frame])
    return {
        "detected": True,
        "detected_sec": detected_sec,
        "detected_frame": detected_frame,
        "from_label": from_label,
        "to_label": to_label,
    }


def compute_boundary_time_error(
    detected_sec: float | None,
    aligned_sec: float | None,
) -> dict[str, Any]:
    """计算检测到的 transition 与对齐词边界的时间误差。

    error = detected - aligned。正值 = 转场偏晚（transition 在词边界之后），
    负值 = 转场偏早（transition 在词边界之前）。

    若 detected 或 aligned 为 None（未检测到 transition 或对齐失败），
    不伪造误差 → ``boundary_error_sec=None`` + reason。
    """
    if detected_sec is None and aligned_sec is None:
        return {
            "boundary_error_sec": None,
            "detected_sec": None,
            "aligned_sec": None,
            "reason": "no_detection_and_no_alignment",
        }
    if detected_sec is None:
        return {
            "boundary_error_sec": None,
            "detected_sec": None,
            "aligned_sec": aligned_sec,
            "reason": "transition_not_detected",
        }
    if aligned_sec is None:
        return {
            "boundary_error_sec": None,
            "detected_sec": detected_sec,
            "aligned_sec": None,
            "reason": "alignment_unavailable",
        }
    error = float(detected_sec) - float(aligned_sec)
    return {
        "boundary_error_sec": error,
        "detected_sec": float(detected_sec),
        "aligned_sec": float(aligned_sec),
    }


def _normalize_word(w: str) -> str:
    """归一化词面用于词序同构比对：保留字母/数字/撇号，去标点 + 小写。"""
    import re
    return re.sub(r"[^\w']", "", w).lower()


def resolve_aligned_boundary_sec(
    alignment_result: AlignmentResult,
    boundary_word_index: int,
    text: str | None = None,
) -> tuple[float | None, str]:
    """从对齐结果中提取第 ``boundary_word_index`` 词的右边界时刻。

    boundary_word_index 为 1-indexed（FEDD 构造约定）：k 表示前 k 个词属于
    emo_from 段。因此边界时刻 = 第 k 个词的 end_sec（0-indexed words[k-1].end）。

    ``text``（B11）：提供时校验 MFA 对齐词序与文本词序同构，不同构返回
    ``(None, "word_sequence_mismatch")``（clitic 拆分等会使 words[k-1] 错位）。

    Returns:
        (boundary_sec, status)。boundary_sec 为 None 时 status 给出原因。
    """
    if alignment_result.status != "aligned":
        return None, alignment_result.status

    words = alignment_result.words
    if not words:
        return None, "failed"

    k = int(boundary_word_index)
    if k < 1 or k >= len(words):
        # word_index 超出对齐词数范围（可能是对齐不完整）
        return None, "failed"

    if text is not None and text.strip():
        # B11: 校验前 k 词（到 boundary）逐词同构——clitic 拆分（如 you'll→you+'ll）
        # 使 words 前 k 词与 text 前 k 词不等，words[k-1] 指错词，boundary_sec 静默
        # 偏移约一个词长。仅校验前 k 词（boundary 之后的词不影响 boundary_sec）。
        expected = [_normalize_word(w) for w in text.split()[:k]]
        actual = [_normalize_word(w.word) for w in words[:k]]
        if expected != actual:
            return None, "word_sequence_mismatch"

    boundary_sec = float(words[k - 1].end_sec)
    return boundary_sec, "aligned"


# ============================================================
# EvaluationRow 构建
# ============================================================


def build_eval_row(
    utt_id: str,
    control_record: dict[str, Any],
    generation_row: dict[str, Any],
    evaluator_identity: dict[str, Any],
    frames_output: dict[str, Any],
    alignment_result: AlignmentResult,
    evidence_tier: str,
) -> dict[str, Any]:
    """从控制记录、生成行、evaluator 输出、对齐结果构建一条 EvaluationRow。

    返回的 dict 通过 ``validate_eval_row``。metrics 包含前 / 后 span 的
    {target, predicted, hit, score, start_sec, end_sec}、transition_direction、
    boundary_error_sec（exact tier 且有效对齐时）、alignment_status 等。
    """
    frames = np.asarray(frames_output["frames"])
    times_sec = np.asarray(frames_output["times_sec"])
    frame_rate = float(frames_output.get("frame_rate_hz", 50.0))
    label_space = frames_output.get("label_space", [])

    emo_from = control_record["emo_from"]
    emo_to = control_record["emo_to"]
    boundary_word_index = control_record.get("boundary_word_index")
    text = control_record.get("text", "")

    duration_sec = float(times_sec[-1]) if len(times_sec) > 0 else 0.0

    # 确定 boundary_sec：
    # - exact: 从对齐结果提取词边界（word_index k 的右边界）。
    # - approximate: duration / 2（midpoint 近似）。
    if evidence_tier == "exact":
        if boundary_word_index is not None:
            boundary_sec, align_status = resolve_aligned_boundary_sec(
                alignment_result, boundary_word_index, text,
            )
            if boundary_sec is None:
                # 对齐失败 → 无法确定精确边界时间。
                # 回退到 duration/2 做 span 分割，但记录对齐状态。
                boundary_sec = duration_sec / 2.0
                # 优先使用 AlignmentResult.reason（如 "low_score"），
                # 否则用 status（如 "failed"）。
                alignment_reason = (
                    alignment_result.reason if alignment_result.reason
                    else align_status
                )
            else:
                alignment_reason = None
        else:
            boundary_sec = duration_sec / 2.0
            align_status = "not_attempted"
            alignment_reason = "no_boundary_word_index"
    else:
        # approximate: 始终用 midpoint
        boundary_sec = duration_sec / 2.0
        align_status = "not_attempted"
        alignment_reason = "approximate_tier"

    # 核心 span 评测.
    span_metrics = evaluate_spans_from_frames(
        frames, times_sec, boundary_sec,
        emo_from, emo_to, label_space,
    )

    # transition 检测.
    transition = detect_transition_from_frames(
        frames, times_sec, emo_from, emo_to, label_space, frame_rate,
    )

    # boundary time error（仅 exact tier 且有对齐时计算）.
    boundary_error: dict[str, Any]
    if evidence_tier == "exact" and align_status == "aligned":
        aligned_boundary = resolve_aligned_boundary_sec(
            alignment_result, boundary_word_index, text,
        )[0]
        boundary_error = compute_boundary_time_error(
            transition["detected_sec"], aligned_boundary,
        )
        if boundary_error["boundary_error_sec"] is None:
            alignment_reason = boundary_error.get("reason")
    elif evidence_tier == "exact":
        # 对齐未成功 → 不伪造边界误差
        boundary_error = {
            "boundary_error_sec": None,
            "detected_sec": transition["detected_sec"],
            "aligned_sec": None,
            "reason": alignment_reason or f"alignment_{align_status}",
        }
    else:
        # approximate tier → 不计算边界误差
        boundary_error = {
            "boundary_error_sec": None,
            "detected_sec": transition["detected_sec"],
            "aligned_sec": None,
            "reason": "approximate_tier_no_precise_boundary",
        }

    # 组装 metrics dict.
    # Task #11: span_metrics["valid"] 标识 evaluator 输出（frames）是否可用；
    # False → aggregate 跳过该样本（不计入 hit / direction / score 分母）。
    metrics: dict[str, Any] = {
        "front_span": span_metrics["front"],
        "back_span": span_metrics["back"],
        "transition_direction": span_metrics["transition_direction"],
        "front_back_both_hit": span_metrics["front_back_both_hit"],
        "valid": bool(span_metrics.get("valid", True)),
        "boundary_error_sec": boundary_error["boundary_error_sec"],
        "detected_transition_sec": boundary_error["detected_sec"],
        "aligned_boundary_sec": boundary_error["aligned_sec"],
        "alignment_status": align_status if evidence_tier == "exact" else "not_attempted",
        "alignment_reason": alignment_reason,
        "construction_boundary_word_index": boundary_word_index,
        "construction_method": control_record.get("method"),
        "duration_sec": duration_sec,
        "frame_rate_hz": frame_rate,
    }

    # 构建 evaluator dict（合同 Evaluator TypedDict 子集）.
    evaluator_dict = {
        "name": evaluator_identity.get("name", ""),
        "version": evaluator_identity.get("version", ""),
        "label_space": evaluator_identity.get("label_space"),
        "sample_rate_hz": evaluator_identity.get("sample_rate_hz"),
        "frame_rate_hz": evaluator_identity.get("frame_rate_hz"),
        "self_evidence_risk": evaluator_identity.get("self_evidence_risk"),
    }

    row: dict[str, Any] = {
        "utt_id": utt_id,
        "generation_row": generation_row,
        "control_span": control_record,
        "evaluator": evaluator_dict,
        "boundary_evidence_tier": evidence_tier,
        "metrics": metrics,
    }
    validate_eval_row(row)
    return row


# ============================================================
# 严格配对
# ============================================================


def _extract_utt_id_set(
    records: list[dict[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    """构建 {utt_id: record}；重复 utt_id → ValueError 携带 utt_id。"""
    mapping: dict[str, dict[str, Any]] = {}
    for rec in records:
        uid = rec.get("utt_id")
        if not uid or not isinstance(uid, str):
            raise ValueError(
                f"{source_name} record missing non-empty utt_id: {rec!r}"
            )
        if uid in mapping:
            raise ValueError(
                f"duplicate utt_id '{uid}' in {source_name}"
            )
        mapping[uid] = rec
    return mapping


def _identity_core(token: Any) -> str:
    """规范化身份 token 为可比较的核心标识。

    处理 ``control_row_ref`` / ``prompt_row_ref`` 的常见约定：

    - ``"control/{utt_id}"`` → ``"{utt_id}"``
    - ``"prompt/{speaker_id}"`` → ``"{speaker_id}"``
    - ``"{utt_id}"`` → ``"{utt_id}"``

    取最后一段 ``/`` 分割结果，使 control_record 的 ``utt_id``（裸 ID）能与
    generation row 的 ``"control/{utt_id}"``（带前缀引用）在同一身份空间比较。
    非字符串 / 空字符串 → ``""``。
    """
    if not isinstance(token, str):
        return ""
    token = token.strip()
    if not token:
        return ""
    return token.split("/")[-1]


def _extract_gen_identity_core(
    gen_row: Mapping[str, Any],
    *,
    str_key: str,
    mapping_key: str,
) -> str:
    """从 generation row 内嵌身份提取核心标识（ticket 07 / 核查 #5）。

    优先级：
    1. ``str_key``（如 ``control_row_ref`` / ``prompt_row_ref``）：非空字符串
       → ``_identity_core`` 规范化。
    2. ``mapping_key``（如 ``control_row`` / ``prompt_row``）：Mapping → 取其
       同名 ``str_key`` 子字段或 ``utt_id`` 子字段规范化；均缺失 → 规范化 JSON
       作为身份指纹（确保非空 Mapping 也能产生可比 token）。

    返回 ``""`` 表示 generation row 未携带该身份（无法校验，调用方跳过）。
    """
    ref = gen_row.get(str_key)
    if isinstance(ref, str) and ref.strip():
        return _identity_core(ref)
    mapping = gen_row.get(mapping_key)
    if isinstance(mapping, Mapping) and len(mapping) > 0:
        inner_ref = mapping.get(str_key)
        if isinstance(inner_ref, str) and inner_ref.strip():
            return _identity_core(inner_ref)
        inner_uid = mapping.get("utt_id")
        if isinstance(inner_uid, str) and inner_uid.strip():
            return _identity_core(inner_uid)
        # Mapping 无可提取子身份 → 用规范化 JSON 作为唯一指纹
        return json.dumps(dict(mapping), sort_keys=True, default=str)
    return ""


def _extract_ctrl_control_core(ctrl: Mapping[str, Any]) -> str:
    """从控制记录提取期望的 control 身份核心标识。

    控制记录可能显式携带 ``control_row_ref``（与 generation row 同约定）；
    否则控制记录自身即控制 row，用其 ``utt_id`` 作为身份键。
    """
    ref = ctrl.get("control_row_ref")
    if isinstance(ref, str) and ref.strip():
        return _identity_core(ref)
    uid = ctrl.get("utt_id")
    return _identity_core(uid)


def _strict_pair(
    control_records: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """严格 1:1 配对控制记录与生成行（by utt_id）。

    校验（任一失败 → 携 utt_id hard-fail）：
    - utt_id 非空、无重复（控制 / 生成各自）；
    - utt_id 集合完全相等（无单侧遗漏）；
    - 每条生成行 ``finish_reason == "eos"``（非 EOS 不进声学）；
    - 每条生成行有 wav_path（EOS 必需）；
    - 所有生成行 checkpoint_sha256 一致（同一模型评测）；
    - **control 身份对称强制（ticket 09 / Grilling 决策 #3）**：每对
      (control, generation) 校验 generation row 内嵌的 ``control_row_ref`` /
      ``control_row`` 与配对 control_record 的 control 身份一致。**任一侧缺失
      → hard-fail**（旧逻辑静默跳过，漏检 gen 未内嵌 control；schema §2
      GenerationRow 的 control 族 ``control_row_ref`` / ``control_row`` 必需，
      任一不得缺失）。防止 gen 的 WAV 实际来自另一控制条件却被按当前 utt_id
      配对打分。

    Note: gen 的 prompt 族（``prompt_row_ref`` / ``prompt_row``）存在性由
    ``validate_generation_row`` 保证（schema §2 四族各≥1），**不在 per-pair
    校验**——schema §1 SupervisionSpan 无 ``prompt_row_ref`` 字段，ctrl 这侧
    无法提供期望 prompt 身份，per-pair prompt 校验是死代码（ticket 09 已删）。
    """
    ctrl_map = _extract_utt_id_set(control_records, "control_manifest")
    gen_map = _extract_utt_id_set(generation_rows, "generation_manifest")

    ctrl_ids = set(ctrl_map)
    gen_ids = set(gen_map)

    if ctrl_ids != gen_ids:
        ctrl_only = sorted(ctrl_ids - gen_ids)
        gen_only = sorted(gen_ids - ctrl_ids)
        detail_utt = (ctrl_only + gen_only)[0] if (ctrl_only or gen_only) else "?"
        raise ValueError(
            f"utt_id set mismatch (hard-fail utt='{detail_utt}'): "
            f"control_only={ctrl_only[:5]} generation_only={gen_only[:5]}"
        )

    # checkpoint 一致性.
    checkpoint_shas = {
        g.get("checkpoint_sha256") for g in gen_map.values()
        if g.get("checkpoint_sha256") is not None
    }
    if len(checkpoint_shas) > 1:
        raise ValueError(
            f"checkpoint_sha256 mismatch across generation rows: "
            f"{len(checkpoint_shas)} distinct values — "
            "all rows in one evaluation must come from the same checkpoint"
        )

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for uid in sorted(ctrl_map):
        ctrl = ctrl_map[uid]
        gen = gen_map[uid]

        finish_reason = gen.get("finish_reason")
        if finish_reason != "eos":
            raise RuntimeError(
                f"sample '{uid}' hard-fail: finish_reason='{finish_reason}' "
                f"(only eos enters acoustic evaluation)"
            )

        wav_path = gen.get("wav_path")
        if not wav_path or not isinstance(wav_path, str):
            raise RuntimeError(
                f"sample '{uid}' hard-fail: eos row missing wav_path"
            )

        # —— control 身份对称强制校验（ticket 09 / Grilling 决策 #3）——
        # expected / gen 任一缺失 → hard-fail（schema §2 control 族必需）；
        # 都非空 → 比对，不等 hard-fail。
        expected_ctrl_core = _extract_ctrl_control_core(ctrl)
        gen_ctrl_core = _extract_gen_identity_core(
            gen, str_key="control_row_ref", mapping_key="control_row",
        )
        if not expected_ctrl_core or not gen_ctrl_core:
            raise ValueError(
                f"sample '{uid}' hard-fail: control identity missing "
                f"(expected={expected_ctrl_core!r}, gen={gen_ctrl_core!r}) — "
                "schema §2 GenerationRow control 族（control_row_ref / "
                "control_row）必需，任一缺失不得静默跳过"
            )
        if expected_ctrl_core != gen_ctrl_core:
            raise ValueError(
                f"sample '{uid}' hard-fail: control_row_ref mismatch — "
                f"generation row 内嵌 control 身份 '{gen_ctrl_core}' "
                f"与配对 control_record 身份 '{expected_ctrl_core}' 不一致"
            )

        pairs.append((ctrl, gen))

    return pairs


# ============================================================
# Aggregate 构建
# ============================================================


def build_aggregate_from_rows(
    rows: list[dict[str, Any]],
    evidence_tier: str,
) -> dict[str, Any]:
    """从已持久化 EvaluationRow 确定性派生 Aggregate（按 evidence_tier 分离）。

    计算：
    - front/back emotion hit rate；
    - transition direction 分布（correct/reverse/maintained/other）；
    - mean front/back score；
    - boundary error 统计（仅 valid boundary_error_sec 的样本参与）。

    exact tier 对齐失败样本（``alignment_status != "aligned"``）**不计入**
    exact aggregate 分母（hit / direction / score / boundary_error）——
    它们的 boundary_sec 来自 midpoint 回退而非真实词边界，不能参与精确结论
    （spec L106）。对齐失败样本单独归入 ``n_exact_alignment_failed`` 计数。
    approximate tier 行为不变（alignment_status 恒为 ``not_attempted``）。

    空 rows → ``n_samples=0`` 的合法 aggregate（该 tier 无样本）。
    """
    tier_rows = [r for r in rows if r.get("boundary_evidence_tier") == evidence_tier]

    # Task #11: evaluator 输出为空 / 全 NaN 的样本（metrics.valid is False）
    # 不计入任何 hit / direction / score 分母，单独计数 n_invalid_output。
    # 兼容旧 row（无 metrics.valid 字段）→ 视为 True（既有路径不破）。
    n_invalid_output = 0
    valid_output_rows: list[dict[str, Any]] = []
    for r in tier_rows:
        if r.get("metrics", {}).get("valid") is False:
            n_invalid_output += 1
        else:
            valid_output_rows.append(r)
    tier_rows = valid_output_rows

    # exact tier：区分"未尝试"（aligner 缺位 / approximate）/"失败"（对齐或词序
    # 同构失败）/"成功"（aligned）。只有 aligned 进精确结论分母（B10 可见性）。
    n_exact_alignment_not_attempted = 0
    n_exact_alignment_failed = 0
    if evidence_tier == "exact":
        aligned_rows: list[dict[str, Any]] = []
        for r in tier_rows:
            status = r["metrics"].get("alignment_status")
            if status == "aligned":
                aligned_rows.append(r)
            elif status == "not_attempted":
                n_exact_alignment_not_attempted += 1
            else:
                n_exact_alignment_failed += 1
        tier_rows = aligned_rows

    n = len(tier_rows)

    if n == 0:
        agg: dict[str, Any] = {
            "evidence_tier": evidence_tier,
            "metric_contract_version": METRIC_CONTRACT_VERSION,
            "n_samples": 0,
            "metrics": {},
        }
        if n_invalid_output > 0:
            agg["n_invalid_output"] = n_invalid_output
        if evidence_tier == "exact":
            agg["n_exact_alignment_failed"] = n_exact_alignment_failed
            agg["n_exact_alignment_not_attempted"] = n_exact_alignment_not_attempted
        validate_aggregate(agg)
        return agg

    front_hits = sum(1 for r in tier_rows if r["metrics"]["front_span"]["hit"])
    back_hits = sum(1 for r in tier_rows if r["metrics"]["back_span"]["hit"])
    both_hits = sum(1 for r in tier_rows if r["metrics"]["front_back_both_hit"])

    directions = [r["metrics"]["transition_direction"] for r in tier_rows]
    n_correct = sum(1 for d in directions if d == "correct")
    n_reverse = sum(1 for d in directions if d == "reverse")
    n_maintained = sum(1 for d in directions if d == "maintained")
    n_other = sum(1 for d in directions if d == "other")

    front_scores = [r["metrics"]["front_span"]["score"] for r in tier_rows]
    back_scores = [r["metrics"]["back_span"]["score"] for r in tier_rows]

    boundary_errors = [
        r["metrics"]["boundary_error_sec"]
        for r in tier_rows
        if r["metrics"]["boundary_error_sec"] is not None
    ]

    abs_errors = [abs(e) for e in boundary_errors]

    metrics: dict[str, Any] = {
        "front_emotion_hit_rate": front_hits / n,
        "back_emotion_hit_rate": back_hits / n,
        "front_back_both_hit_rate": both_hits / n,
        "transition_correct_rate": n_correct / n,
        "transition_reverse_rate": n_reverse / n,
        "transition_maintained_rate": n_maintained / n,
        "transition_other_rate": n_other / n,
        "mean_front_score": float(np.mean(front_scores)),
        "mean_back_score": float(np.mean(back_scores)),
        "n_boundary_errors": len(boundary_errors),
    }
    if boundary_errors:
        metrics["mean_boundary_error_sec"] = float(np.mean(boundary_errors))
        metrics["median_boundary_error_sec"] = float(statistics.median(boundary_errors))
        metrics["mean_abs_boundary_error_sec"] = float(np.mean(abs_errors))
    else:
        metrics["mean_boundary_error_sec"] = None
        metrics["median_boundary_error_sec"] = None
        metrics["mean_abs_boundary_error_sec"] = None

    agg = {
        "evidence_tier": evidence_tier,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "n_samples": n,
        "metrics": metrics,
    }
    if n_invalid_output > 0:
        agg["n_invalid_output"] = n_invalid_output
    if evidence_tier == "exact":
        agg["n_exact_alignment_failed"] = n_exact_alignment_failed
        agg["n_exact_alignment_not_attempted"] = n_exact_alignment_not_attempted
    validate_aggregate(agg)
    return agg


# ============================================================
# 全管线编排
# ============================================================


def evaluate_fedd_dataset(
    control_records: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    evaluator: Any,
    *,
    aligner: ForcedAligner | None = None,
) -> dict[str, Any]:
    """完整 FEDD 逐 span 评测管线。

    Args:
        control_records: FEDD 控制记录列表（每条含 utt_id / emo_from / emo_to /
            boundary_word_index / method / text）。
        generation_rows: v2 GenerationRow 列表（每条含 utt_id / finish_reason /
            wav_path / checkpoint_sha256 等）。
        evaluator: 实现 ``EmotionEvaluator`` 接口的对象（Fake 或真实）。
        aligner: 可选 ``ForcedAligner``。exact tier 样本需要对齐来计算精确
            boundary error；None 时 exact 样本的 boundary_error 为 null。

    Returns:
        ``{"rows": [EvaluationRow, ...], "aggregate_exact": Aggregate,
        "aggregate_approximate": Aggregate, "n_samples": int}``

    Raises:
        ValueError / RuntimeError: 任何配对 / 身份 / finish_reason 异常
            （携 utt_id hard-fail）。
    """
    pairs = _strict_pair(control_records, generation_rows)
    evaluator_identity = evaluator.identity()

    rows: list[dict[str, Any]] = []
    for control, gen_row in pairs:
        utt_id = control["utt_id"]
        wav_path = gen_row["wav_path"]
        text = control.get("text", "")
        evidence_tier = derive_evidence_tier(
            method=control.get("method"),
            boundary_word_index=control.get("boundary_word_index"),
            part=control.get("part"),
        )

        # 调用 evaluator 获取 frame 轨迹.
        try:
            frames_output = evaluator.predict_frames(wav_path)
        except Exception as exc:
            raise RuntimeError(
                f"sample '{utt_id}' hard-fail: evaluator.predict_frames failed: {exc}"
            ) from exc

        # 对齐（仅 exact tier 请求）.
        if evidence_tier == "exact" and aligner is not None:
            try:
                alignment_result = aligner.align(wav_path, text)
            except Exception as exc:
                alignment_result = AlignmentResult(
                    status="failed", reason=f"aligner_exception: {exc}",
                )
        else:
            alignment_result = AlignmentResult(
                status="not_attempted",
                reason="approximate_tier" if evidence_tier == "approximate"
                else "no_aligner",
            )

        row = build_eval_row(
            utt_id, control, gen_row, evaluator_identity,
            frames_output, alignment_result, evidence_tier,
        )
        rows.append(row)

    aggregate_exact = build_aggregate_from_rows(rows, "exact")
    aggregate_approximate = build_aggregate_from_rows(rows, "approximate")

    return {
        "rows": rows,
        "aggregate_exact": aggregate_exact,
        "aggregate_approximate": aggregate_approximate,
        "n_samples": len(rows),
    }


# ============================================================
# manifest 加载
# ============================================================


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读 JSONL manifest，返回 list of dict。"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ============================================================
# 命令行入口
# ============================================================


def main():
    """命令行入口（生产用；测试不调用）。"""
    import argparse

    ap = argparse.ArgumentParser(description="EmoFiLM v2 FEDD span/transition evaluation")
    ap.add_argument("--control_manifest", required=True,
                    help="JSONL: FEDD control records (utt_id/emo_from/emo_to/"
                         "boundary_word_index/method/text)")
    ap.add_argument("--generation_manifest", required=True,
                    help="JSONL: v2 GenerationRow (utt_id/finish_reason/wav_path/...)")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--evaluator", default="emotion2vec",
                    help="Evaluator to use (emotion2vec/fake). Default: emotion2vec")
    ap.add_argument("--mfa_bin", default=None, help="MFA binary path (for exact tier)")
    ap.add_argument("--no_aligner", action="store_true",
                    help="跳过 MFA 对齐（仅 approximate 评测；有 exact 样本时需显式确认）")
    ap.add_argument("--exclude_utts_file", default=None,
                    help="JSON report（label_fedd_emotion2vec 产）含 failed_utts；"
                         "这些 utt 从评测排除（B14，不进 exact 分母）")
    args = ap.parse_args()

    control_records = load_jsonl(args.control_manifest)
    generation_rows = load_jsonl(args.generation_manifest)

    # B14: 排除 emotion2vec 一致性失败的 utt（参考构造音频情感模糊，不进 exact 分母）。
    # control 与 generation 同步过滤（_strict_pair 要求两侧 utt 集合配对）。
    if args.exclude_utts_file:
        from pathlib import Path
        rep = json.loads(Path(args.exclude_utts_file).read_text(encoding="utf-8"))
        exclude = {f["utt_id"] for f in rep.get("failed_utts", [])}
        if exclude:
            control_records = [c for c in control_records if c.get("utt_id") not in exclude]
            generation_rows = [g for g in generation_rows if g.get("utt_id") not in exclude]
            print(f"[B14] 排除 {len(exclude)} 个一致性失败 utt，"
                  f"剩余 control {len(control_records)} / generation {len(generation_rows)}")

    if args.evaluator == "fake":
        from eval.acoustic_evaluators import FakeAcousticEvaluator
        evaluator = FakeAcousticEvaluator(kind="emotion")
    elif args.evaluator == "external_ser":
        # 决策3 A：外部异源 SER（消除自证风险）+ 滑窗 frame-level
        import torch as _torch
        from eval.acoustic_evaluators import ExternalSerEmotionEvaluator
        evaluator = ExternalSerEmotionEvaluator(
            device="cuda" if _torch.cuda.is_available() else "cpu",
        )
    else:
        # 生产路径：注入真实 emotion2vec wrapper（08 §3.2，门禁 NOT MET）
        # 此处仅占位；真实音频验收需外部独立 evaluator（08 §4）
        raise RuntimeError(
            "real emotion2vec evaluator gate NOT MET (see "
            "docs/contracts/emofilm_v2_evaluators.md §3.2); "
            "use --evaluator external_ser (independent) or fake for synthetic testing"
        )

    aligner = MfaForcedAligner(mfa_bin=args.mfa_bin) if args.mfa_bin else None

    # B10: aligner 缺位检测——有 exact-tier 样本（需 MFA 定位切换点）但未提供
    # aligner → 显式失败（防产出 n_samples=0 的空 exact aggregate 静默 exit 0）。
    if aligner is None and not args.no_aligner:
        needs_aligner = any(
            derive_evidence_tier(c.get("method")) == "exact" for c in control_records
        )
        if needs_aligner:
            raise RuntimeError(
                "control records 含 exact-tier 样本（需 MFA 对齐定位切换点）但未提供 "
                "--mfa_bin，aligner 缺位将产出 n_samples=0 的空 exact aggregate。"
                "请传 --mfa_bin（或设 MFA_BIN 环境变量），或显式 --no_aligner 确认仅 "
                "approximate 评测。"
            )

    result = evaluate_fedd_dataset(
        control_records, generation_rows, evaluator, aligner=aligner,
    )

    # B10 后置门禁：exact aggregate 全灭（曾有 exact 样本但 0 aligned）→ 非零退出，
    # 禁止静默产出空结论（MFA 全失败 / 词序全不匹配等）。
    agg_exact = result["aggregate_exact"]
    exact_attempted = (
        agg_exact.get("n_samples", 0)
        + agg_exact.get("n_exact_alignment_failed", 0)
        + agg_exact.get("n_exact_alignment_not_attempted", 0)
    )
    if exact_attempted > 0 and agg_exact.get("n_samples", 0) == 0:
        raise RuntimeError(
            f"exact tier 0/{exact_attempted} aligned（failed="
            f"{agg_exact.get('n_exact_alignment_failed', 0)}, not_attempted="
            f"{agg_exact.get('n_exact_alignment_not_attempted', 0)}）——旗舰证据层"
            "无可用样本，禁止静默产出空结论。"
        )

    # B15: aggregate 身份绑定确定 rows 集合（检测 rows 被替换/遗漏/混入其他运行）。
    from tools.write_emofilm_run_identity import (
        compute_aggregate_identity, write_emofilm_evaluation_identity,
    )

    aggregate_identity = compute_aggregate_identity(result["rows"])
    evaluator_info = evaluator.identity() if hasattr(evaluator, "identity") else None

    output = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "rows": result["rows"],
        "aggregate_exact": result["aggregate_exact"],
        "aggregate_approximate": result["aggregate_approximate"],
        "n_samples": result["n_samples"],
        "aggregate_identity": aggregate_identity,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # B15: 评测运行身份 sidecar（command + 输入 manifest + aggregate 身份 + evaluator）。
    import os as _os
    import sys as _sys
    _code_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    write_emofilm_evaluation_identity(
        str(args.output) + ".identity.json",
        code_root=_code_root,
        command=" ".join(_sys.argv),
        eval_manifest_path=args.control_manifest,
        n_eval_rows=len(result["rows"]),
        aggregate_identity=aggregate_identity,
        evaluator_info=evaluator_info,
    )
    print(f"Wrote {result['n_samples']} rows -> {args.output}")


__all__ = [
    # 常量
    "METRIC_CONTRACT_VERSION",
    "EXACT_METHOD",
    "APPROX_METHOD",
    # 对齐接口
    "WordBoundary",
    "AlignmentResult",
    "ForcedAligner",
    "MfaForcedAligner",
    # 核心纯函数
    "derive_evidence_tier",
    "evaluate_spans_from_frames",
    "detect_transition_from_frames",
    "compute_boundary_time_error",
    "resolve_aligned_boundary_sec",
    # 编排
    "build_eval_row",
    "build_aggregate_from_rows",
    "evaluate_fedd_dataset",
    "load_jsonl",
]


if __name__ == "__main__":
    # 代码层门禁（ADR-0020 §6）：补 CLI 入口，使 ``python -m eval.eval_local_control``
    # 进入 main() 而非静默退出（原 v2 局部评测副本末尾缺该块）。
    main()
