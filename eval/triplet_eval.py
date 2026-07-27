#!/usr/bin/env python3
"""EmoFiLM v2 低/中/高强度三元评测（ticket 10）。

为固定文本 / emotion / prompt / speaker / checkpoint / seed-policy / 解码配置
的 base case 建立只改变 ``intensity`` 的三元生成与评测链，直接量化
low → medium → high 的声学 arousal 响应、emotion 保持与可懂度（WER），
而非从全 medium aggregate 推断强度能力。是 11（身份链）的前提之一。

设计要点（MAP §3 评测不变量 / brief 10）：
- **稳定 group_id**：三条记录除 ``intensity`` 外完全一致并写入 group metadata
  （text / emotion / speaker / prompt / checkpoint / seed_policy / decode_config）。
- **整组有效门槛**：三条均须 EOS 完成 + 身份一致；缺一条或任一失败 →
  整组标 ``valid=False``，不进单调率分母（显式计数 failure_reason）。
- **逐样本 row**（通过 ``validate_eval_row``）：arousal score、emotion prediction、
  WER、span 范围、evaluator 版本、generation 引用。
- **逐组 row**：low→med→high arousal 单调性、有效跨度（high−low arousal 差）、
  emotion 保持率（三档 emotion 预测是否一致）、分档 WER。
- **aggregate**（通过 ``validate_aggregate``）：valid/invalid 组数、单调率、
  跨度统计、emotion 保持率、分档 WER、failure-reason 分布。
- **强度结论** = 独立 evaluator 下的相对控制响应；IEMOCAP weak arousal **不是**
  词级真值（继承 08 §2 自证风险）。旧整句 WER/Emo-SIM/DTW 保持整体质量指标
  身份，不与三元强度指标混合。

与 09（``eval/eval_local_control.py`` FEDD span/transition）**互补**：
- 09 聚焦前后段 emotion 命中 + transition 方向 + 边界时间误差（FEDD 构造）。
- 10 聚焦同一 base case 下三档强度的 arousal 单调响应（非 FEDD 构造）。
- 复用 09 的 ``ClipMappedEvaluator`` 测试辅助（``tests/_emofilm_fakes.py``）
  + 08 的 ``FakeAcousticEvaluator`` + ``SyntheticReferenceClip``。核心 per-sample
  行结构与 09 一致（``validate_eval_row``），只是 metrics 聚焦 arousal / emotion
  / WER / span range。

CPU 合同 / 行为测试用 ``FakeAcousticEvaluator`` + ``FakeWerEvaluator``
（``tests/_emofilm_fakes.py``）+ 合成 arousal/emotion 轨迹；不加载真实模型、
不调用真实 ASR/MFA。
"""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tools.build_emofilm_contract import (
    BOUNDARY_EVIDENCE_TIERS,
    validate_aggregate,
    validate_eval_row,
    validate_generation_row,
)


# ============================================================
# 常量
# ============================================================

METRIC_CONTRACT_VERSION = "emofilm_v2_eval"

# 强度档位（与 cosyvoice/tokenizer/emo_tokenizer.py 一致；
# control_intensity_id 1=low / 2=medium / 3=high）。
INTENSITY_TIERS = ("low", "medium", "high")
INTENSITY_TIER_TO_ID = {"low": 1, "medium": 2, "high": 3}
INTENSITY_ID_TO_TIER = {v: k for k, v in INTENSITY_TIER_TO_ID.items()}

# 三元组 per-sample 行的 evidence tier：控制 span 为整句（无 FEDD 式近似边界），
# 因此使用 ``exact``。这使逐样本行可通过 ``validate_eval_row``。
TRIPLET_EVIDENCE_TIER = "exact"

# 强度结论的语义标签 —— 明确不当词级真值。
INTENSITY_CONCLUSION_SCOPE = (
    "relative_control_response_under_independent_evaluator"
)


# ============================================================
# WER evaluator 接口（可注入）
# ============================================================


@runtime_checkable
class WerEvaluator(Protocol):
    """冻结的 ASR 转写评测器接口（用于逐样本 WER）。

    实现者保证：
    - ``is_frozen`` 恒 True；
    - ``identity()`` 返回身份记录（name / version 非空）；
    - ``transcribe(wav_path)`` 返回 ``{"hypothesis_text": str}``，不抛异常
      （失败时返回空字符串 + ``status`` 字段）。
    """

    @property
    def is_frozen(self) -> bool: ...

    def identity(self) -> dict[str, Any]: ...

    def transcribe(self, wav_path: Any) -> dict[str, Any]: ...


# ============================================================
# 纯函数：per-sample 指标计算
# ============================================================


def compute_arousal_score(arousal_output: dict[str, Any]) -> float | None:
    """逐帧 arousal 轨迹的均值（单一标量 per sample）。

    Task #11: 空轨迹或全非有限（NaN/inf）→ 返回 ``None``（标记无效样本，
    由调用方记 ``metrics.valid=False``，不计入 aggregate 分母，避免 0.0
    假数据污染单调性 / span gap 统计）。输入为 ``{"frames": np.ndarray (T,)}``。
    """
    frames = np.asarray(arousal_output.get("frames", []), dtype=np.float64)
    if frames.size == 0:
        return None
    # 全非有限（NaN/inf）→ 无可用观测，返回 None。
    if not np.isfinite(frames).any():
        return None
    # 仅取有限帧的均值（丢弃零星 NaN，保证可用样本仍可统计）。
    finite_mask = np.isfinite(frames)
    if not finite_mask.all():
        return float(np.mean(frames[finite_mask]))
    return float(np.mean(frames))


def compute_emotion_prediction(
    emotion_output: dict[str, Any],
    label_space: list[str],
) -> str | None:
    """逐帧情感分布的 frame-mean argmax（单一标签 per sample）。

    Task #11: 空分布 / 1D 单列（无标签维度）/ 全非有限 → 返回 ``None``
    （调用方据 None 标 ``metrics.valid=False``，不进 emotion 命中分母）。
    平局时取 label_space 中最靠前的标签（确定性）。
    """
    frames = np.asarray(emotion_output.get("frames", []), dtype=np.float64)
    if frames.size == 0:
        return None
    if frames.ndim == 1:
        # 1D 单列无法对齐到 K 标签维度 → 视为无效输出。
        return None
    # 全非有限 → 无可用观测。
    if not np.isfinite(frames).any():
        return None
    # 仅用有限行做 mean（nanmean 会因全行 NaN 报警，前面已过滤）。
    finite_rows = np.all(np.isfinite(frames), axis=1)
    if not finite_rows.all():
        frames = frames[finite_rows]
    mean_dist = frames.mean(axis=0)
    idx = int(np.argmax(mean_dist))
    return label_space[idx] if idx < len(label_space) else label_space[0]


def compute_wer(reference: str, hypothesis: str) -> dict[str, Any]:
    """词级 WER = (S + D + I) / N，N = 参考词数。

    使用动态规划词级编辑距离。边界：
    - 空参考 + 空假设 → WER = 0.0；
    - 空参考 + 非空假设 → WER = 1.0（全部为插入，归一化为上限 1.0）；
    - N > 0 且 (S+D+I) > N → WER 可超过 1.0（标准定义）。

    Returns:
        ``{wer: float, substitutions: int, deletions: int, insertions: int,
        n_reference_words: int}``
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)

    if n_ref == 0:
        if n_hyp == 0:
            return {
                "wer": 0.0, "substitutions": 0, "deletions": 0,
                "insertions": 0, "n_reference_words": 0,
            }
        # 非空假设 + 空参考：归一化为 1.0（全部插入，超过 N=0 → 上限 1.0）
        return {
            "wer": 1.0, "substitutions": 0, "deletions": 0,
            "insertions": n_hyp, "n_reference_words": 0,
        }

    # 词级 Levenshtein（dp[i][j] = ref[:i] vs hyp[:j] 的最小编辑距离）。
    # 操作码：0=match（匹配），1=substitution（替换），2=deletion（删除，ref→无），
    # 3=insertion（插入，无←hyp）。substitution 与 match 走同一对角单元但编码
    # 不同，避免回溯时把替换误算为 match。
    dp = np.zeros((n_ref + 1, n_hyp + 1), dtype=np.int32)
    op = np.zeros((n_ref + 1, n_hyp + 1), dtype=np.int8)
    for i in range(1, n_ref + 1):
        dp[i, 0] = i
        op[i, 0] = 2  # deletion
    for j in range(1, n_hyp + 1):
        dp[0, j] = j
        op[0, j] = 3  # insertion
    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
                op[i, j] = 0  # match
            else:
                sub_cost = dp[i - 1, j - 1] + 1
                del_cost = dp[i - 1, j] + 1
                ins_cost = dp[i, j - 1] + 1
                best = sub_cost
                best_op = 1  # substitution
                if del_cost < best:
                    best = del_cost
                    best_op = 2  # deletion
                if ins_cost < best:
                    best = ins_cost
                    best_op = 3  # insertion
                dp[i, j] = best
                op[i, j] = best_op

    # 回溯统计 S / D / I。
    s = d = ins = 0
    i, j = n_ref, n_hyp
    while i > 0 or j > 0:
        code = int(op[i, j])
        if code == 0:  # match
            i -= 1
            j -= 1
        elif code == 1:  # substitution
            s += 1
            i -= 1
            j -= 1
        elif code == 2:  # deletion
            d += 1
            i -= 1
        else:  # code == 3, insertion
            ins += 1
            j -= 1

    wer = (s + d + ins) / n_ref
    return {
        "wer": float(wer),
        "substitutions": int(s),
        "deletions": int(d),
        "insertions": int(ins),
        "n_reference_words": int(n_ref),
    }


def extract_span_range(times_sec: Any) -> dict[str, float]:
    """从帧时间戳数组提取 span 范围 ``{start_sec, end_sec}``。

    空数组 → ``{0.0, 0.0}``（诚实，不崩溃）。
    """
    times = np.asarray(times_sec, dtype=np.float64).ravel()
    if times.size == 0:
        return {"start_sec": 0.0, "end_sec": 0.0}
    return {
        "start_sec": float(times[0]),
        "end_sec": float(times[-1]),
    }


# ============================================================
# 逐样本 EvaluationRow 构建（per triplet member）
# ============================================================


def build_triplet_member_eval_row(
    utt_id: str,
    base_case: dict[str, Any],
    intensity_tier: str,
    generation_row: dict[str, Any],
    control_span: dict[str, Any],
    arousal_output: dict[str, Any],
    emotion_output: dict[str, Any],
    hypothesis_text: str,
    arousal_evaluator_identity: dict[str, Any],
    emotion_evaluator_identity: dict[str, Any],
    wer_evaluator_identity: dict[str, Any],
) -> dict[str, Any]:
    """构建一条 triplet 成员的 EvaluationRow（通过 ``validate_eval_row``）。

    metrics 携带 arousal score、emotion prediction、WER、span 范围、
    以及三个 evaluator 的完整身份（arousal / emotion / wer）。
    """
    arousal_score = compute_arousal_score(arousal_output)
    emotion_label_space = (
        list(emotion_evaluator_identity.get("label_space") or [])
        or list(emotion_output.get("label_space") or [])
    )
    emotion_pred = compute_emotion_prediction(emotion_output, emotion_label_space)
    wer_result = compute_wer(base_case.get("text", ""), hypothesis_text)
    span_range = extract_span_range(arousal_output.get("times_sec", []))

    # Task #11: compute_* 返回 None 表示 evaluator 输出为空 / 全非有限，
    # 该样本视为无效（不进 aggregate 分母），arousal_score / emotion_prediction
    # 保留 None 以反映"未观测"，而非 0.0 / 首类标签等假数据。
    valid_output = arousal_score is not None and emotion_pred is not None

    metrics: dict[str, Any] = {
        "intensity_tier": intensity_tier,
        "intensity_id": INTENSITY_TIER_TO_ID.get(intensity_tier),
        "valid": bool(valid_output),
        "arousal_score": (float(arousal_score) if arousal_score is not None else None),
        "emotion_prediction": emotion_pred,
        "wer": float(wer_result["wer"]),
        "wer_detail": {
            "substitutions": wer_result["substitutions"],
            "deletions": wer_result["deletions"],
            "insertions": wer_result["insertions"],
            "n_reference_words": wer_result["n_reference_words"],
        },
        "span_range": span_range,
        "hypothesis_text": hypothesis_text,
        "reference_text": base_case.get("text", ""),
        "evaluators": {
            "arousal": _strip_identity(arousal_evaluator_identity),
            "emotion": _strip_identity(emotion_evaluator_identity),
            "wer": _strip_identity(wer_evaluator_identity),
        },
    }

    # 主 evaluator 字段（合同要求 name+version）= arousal evaluator（强度主信号）。
    primary_evaluator = _strip_identity(arousal_evaluator_identity)

    row: dict[str, Any] = {
        "utt_id": utt_id,
        "generation_row": dict(generation_row),
        "control_span": dict(control_span),
        "evaluator": primary_evaluator,
        "boundary_evidence_tier": TRIPLET_EVIDENCE_TIER,
        "metrics": metrics,
    }
    validate_eval_row(row)
    return row


def _strip_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """从 evaluator.identity() 提取合同 Evaluator TypedDict 子集。

    保留 name / version / label_space / sample_rate_hz / frame_rate_hz /
    self_evidence_risk + 其他扩展字段（model_id 等）。
    """
    if not isinstance(identity, dict):
        return {"name": "", "version": ""}
    keys = (
        "name", "version", "label_space", "sample_rate_hz", "frame_rate_hz",
        "self_evidence_risk", "model_id", "revision", "calibration",
        "shares_source_with_iemocap_weak_supervision",
        "output_semantics", "known_limitations",
    )
    return {k: identity[k] for k in keys if k in identity}


# ============================================================
# 逐组 TripletGroupRow 构建
# ============================================================


def _check_group_membership(
    group_id: str,
    base_case: dict[str, Any],
    member_rows: dict[str, dict[str, Any]],
) -> tuple[bool, str | None]:
    """核验三条成员：档位齐全 + EOS + 身份一致 + 控制值匹配。

    Returns:
        (valid, failure_reason)。valid=False 时 failure_reason 非空。
    """
    # 1. 三档齐全且唯一。
    present_tiers = set(member_rows.keys())
    expected = set(INTENSITY_TIERS)
    missing = expected - present_tiers
    if missing:
        return False, f"missing_member: {sorted(missing)}"
    extra = present_tiers - expected
    if extra:
        return False, f"unexpected_member: {sorted(extra)}"

    # 2. 每条 generation_row finish_reason=eos + wav_path.
    for tier in INTENSITY_TIERS:
        gen = member_rows[tier].get("generation_row", {})
        fr = gen.get("finish_reason")
        if fr != "eos":
            return False, f"non_eos: tier={tier} finish_reason={fr!r}"
        if not gen.get("wav_path"):
            return False, f"missing_wav_path: tier={tier}"

    # 2b. Task #11: 任一 member 的 evaluator 输出为空 / 全非有限 → 整组 invalid。
    # 该检查紧跟 EOS 之后：EOS 行应产出可用 acoustic frames；若 evaluator 返回
    # 空 / 全 NaN（build_triplet_member_eval_row 已置 metrics.valid=False），
    # group 不可参与单调性 / emotion 命中等分母统计。
    for tier in INTENSITY_TIERS:
        m = member_rows[tier].get("metrics", {})
        if m.get("valid") is False:
            return False, f"invalid_member_output: tier={tier}"

    # 3. checkpoint / source / decode_config / seed 一致.
    checkpoints = {
        member_rows[t].get("generation_row", {}).get("checkpoint_sha256")
        for t in INTENSITY_TIERS
    }
    if len(checkpoints) > 1:
        return False, f"checkpoint_mismatch: {len(checkpoints)} distinct values"

    sources = {
        member_rows[t].get("generation_row", {}).get("source_revision")
        for t in INTENSITY_TIERS
    }
    if len(sources) > 1:
        return False, f"source_revision_mismatch: {len(sources)} distinct values"

    # 防御性 double-check（核验 #10）：checkpoint_sha256 与 source_revision
    # 三档「全 None」即视为身份族全缺。入口 ``evaluate_triplet_dataset`` 已对每条
    # generation_row 调 ``validate_generation_row``（schema §2 四族≥1 + seed 非负 int），
    # 此处再兜底以避免绕过入口的脏数据混入 group（set({None}) len=1 恒通过的盲区）.
    cp_all_missing = all(
        member_rows[t].get("generation_row", {}).get("checkpoint_sha256") is None
        for t in INTENSITY_TIERS
    )
    src_all_missing = all(
        member_rows[t].get("generation_row", {}).get("source_revision") is None
        for t in INTENSITY_TIERS
    )
    if cp_all_missing and src_all_missing:
        return False, (
            "missing_identity: checkpoint_sha256 and source_revision both absent "
            "across all tiers"
        )

    decode_configs = [
        json_key(member_rows[t].get("generation_row", {}).get("decode_config"))
        for t in INTENSITY_TIERS
    ]
    if len(set(decode_configs)) > 1:
        return False, "decode_config_mismatch"

    # seed 比对（核验 #10）：三档 per-request 固定种子必须一致；schema GenerationRow
    # §2 从无 seed_policy 字段（旧代码读 .seed_policy 恒 None → set({None}) 永过），
    # 改读 seed 字段（T4 已加入 schema + validate_generation_row）.
    seeds = {
        member_rows[t].get("generation_row", {}).get("seed")
        for t in INTENSITY_TIERS
    }
    if None in seeds or len(seeds) > 1:
        non_none = sorted(s for s in seeds if s is not None)
        return False, f"seed_mismatch: {non_none}"

    # 4. control_span 的 text / emotion / speaker 一致 + 三档 prompt_row_ref 一致
    #    + control_intensity_id 匹配 tier. prompt_row_ref 字段实际位于
    #    generation_row（不在 control_span）；三档必须用同一条 prompt，
    #    否则 prompt 差异会被误归因于 intensity（ticket 08 / 核查 #6：
    #    此前注释声称比较 prompt_ref 但代码并未比较）.
    texts = {
        member_rows[t].get("control_span", {}).get("text")
        for t in INTENSITY_TIERS
    }
    if len(texts) > 1:
        return False, "text_mismatch"
    emotions = {
        member_rows[t].get("control_span", {}).get("emotion")
        for t in INTENSITY_TIERS
    }
    if len(emotions) > 1:
        return False, "emotion_mismatch"
    speakers = {
        member_rows[t].get("control_span", {}).get("speaker")
        for t in INTENSITY_TIERS
    }
    if len(speakers) > 1:
        return False, "speaker_mismatch"
    # prompt_row_ref 一致（与 text/emotion/speaker 并列；从 generation_row 取值）.
    prompt_refs = {
        member_rows[t].get("generation_row", {}).get("prompt_row_ref")
        for t in INTENSITY_TIERS
    }
    if len(prompt_refs) > 1:
        return False, f"prompt_row_ref_mismatch: {len(prompt_refs)} distinct values"

    for tier in INTENSITY_TIERS:
        expected_id = INTENSITY_TIER_TO_ID[tier]
        actual_id = member_rows[tier].get("control_span", {}).get("control_intensity_id")
        if actual_id != expected_id:
            return False, (
                f"intensity_id_mismatch: tier={tier} expected_id={expected_id} "
                f"actual_id={actual_id}"
            )

    return True, None


def json_key(obj: Any) -> str:
    """稳定的 JSON 序列化（用于 dict 比较）。"""
    import json
    return json.dumps(obj, sort_keys=True, default=str)


def evaluate_triplet_group(
    group_id: str,
    base_case: dict[str, Any],
    member_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """构建一条 TripletGroupRow（per-group）。

    流程：
    1. ``_check_group_membership`` 核验三档齐全 + EOS + 身份一致 + 控制值匹配；
       失败 → ``valid=False`` + ``failure_reason``，metrics 为空占位。
    2. 计算 per-group metrics：arousal 单调性、跨度（high−low）、emotion 保持、
       分档 WER、分档 span_range。

    Returns:
        ``{group_id, valid, failure_reason, base_case, members, metrics,
        evaluator, boundary_evidence_tier}``
    """
    valid, failure_reason = _check_group_membership(group_id, base_case, member_rows)

    members_meta: dict[str, Any] = {}
    for tier in INTENSITY_TIERS:
        row = member_rows.get(tier)
        members_meta[tier] = {
            "utt_id": row.get("utt_id") if row else None,
            "intensity_id": INTENSITY_TIER_TO_ID.get(tier),
            "present": row is not None,
        } if row else {
            "utt_id": None,
            "intensity_id": INTENSITY_TIER_TO_ID.get(tier),
            "present": False,
        }

    # evaluator 身份从任一成员提取（已校验一致）；无成员则空.
    sample_row = next(iter(member_rows.values()), None)
    evaluator_identity = (
        _strip_identity(sample_row["evaluator"]) if sample_row
        and isinstance(sample_row.get("evaluator"), dict) else {"name": "", "version": ""}
    )

    group_row: dict[str, Any] = {
        "group_id": group_id,
        "valid": bool(valid),
        "failure_reason": failure_reason,
        "base_case": dict(base_case),
        "members": members_meta,
        "evaluator": evaluator_identity,
        "boundary_evidence_tier": TRIPLET_EVIDENCE_TIER,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "metrics": {},
    }

    if not valid:
        return group_row

    # 提取 per-tier 标量.
    arousal_by_tier = {
        tier: float(member_rows[tier]["metrics"]["arousal_score"])
        for tier in INTENSITY_TIERS
    }
    emotion_by_tier = {
        tier: member_rows[tier]["metrics"]["emotion_prediction"]
        for tier in INTENSITY_TIERS
    }
    wer_by_tier = {
        tier: float(member_rows[tier]["metrics"]["wer"])
        for tier in INTENSITY_TIERS
    }
    span_by_tier = {
        tier: member_rows[tier]["metrics"]["span_range"]
        for tier in INTENSITY_TIERS
    }

    a_low = arousal_by_tier["low"]
    a_med = arousal_by_tier["medium"]
    a_high = arousal_by_tier["high"]

    # 单调性定义：
    # - arousal_monotonic: low < high（净方向正确；start → end 有所上升）。
    # - arousal_strict_monotonic: low < med < high（三档全序）。
    # 平坦（全部相等）→ 两者皆 False（控制无效）。
    monotonic = a_low < a_high
    strict_monotonic = a_low < a_med < a_high
    span_gap = a_high - a_low

    # emotion 保持：三档预测完全一致.
    unique_emotions = set(emotion_by_tier.values())
    emotion_preserved = len(unique_emotions) == 1

    group_row["metrics"] = {
        "arousal_low": a_low,
        "arousal_medium": a_med,
        "arousal_high": a_high,
        "arousal_monotonic": bool(monotonic),
        "arousal_strict_monotonic": bool(strict_monotonic),
        "arousal_span_gap": float(span_gap),
        "emotion_predictions": emotion_by_tier,
        "emotion_preserved": bool(emotion_preserved),
        "wer_low": wer_by_tier["low"],
        "wer_medium": wer_by_tier["medium"],
        "wer_high": wer_by_tier["high"],
        "span_ranges": span_by_tier,
        "member_refs": {
            tier: member_rows[tier].get("utt_id") for tier in INTENSITY_TIERS
        },
    }
    return group_row


# ============================================================
# Aggregate 构建（从逐组行确定性派生）
# ============================================================


def build_triplet_aggregate(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """从 TripletGroupRow 列表确定性派生 Aggregate。

    计算：
    - valid / invalid 组数；
    - monotonicity rate（low<high，分母 = n_valid_groups）；
    - strict monotonicity rate（low<med<high，分母 = n_valid_groups）；
    - arousal span gap 统计（mean / min / max，仅 valid 组）；
    - emotion preservation rate（分母 = n_valid_groups）；
    - 分档 mean WER（low / medium / high，仅 valid 组）；
    - failure-reason 分布（invalid 组的 failure_reason 计数）；
    - intensity_conclusion：明确为相对控制响应 + 自证风险（08 继承）。

    空 group_rows → ``n_samples=0`` 的合法 aggregate。
    """
    n_total = len(group_rows)
    valid_rows = [g for g in group_rows if g.get("valid") is True]
    invalid_rows = [g for g in group_rows if g.get("valid") is not True]
    n_valid = len(valid_rows)
    n_invalid = len(invalid_rows)

    # failure-reason 分布（无效组）.
    failure_reasons = Counter()
    for g in invalid_rows:
        reason = g.get("failure_reason") or "unknown"
        # 取首词作为分布键（例如 "missing_member: ['high']" → "missing_member"）。
        key = reason.split(":", 1)[0].strip()
        failure_reasons[key] += 1
    failure_dist = dict(sorted(failure_reasons.items()))

    metrics: dict[str, Any] = {
        "n_total_groups": n_total,
        "n_valid_groups": n_valid,
        "n_invalid_groups": n_invalid,
        "failure_reason_distribution": failure_dist,
    }

    if n_valid == 0:
        metrics.update({
            "monotonicity_rate": 0.0,
            "strict_monotonicity_rate": 0.0,
            "emotion_preservation_rate": 0.0,
            "mean_span_gap": None,
            "min_span_gap": None,
            "max_span_gap": None,
            "mean_wer_low": None,
            "mean_wer_medium": None,
            "mean_wer_high": None,
        })
    else:
        n_monotonic = sum(
            1 for g in valid_rows if g["metrics"].get("arousal_monotonic")
        )
        n_strict = sum(
            1 for g in valid_rows if g["metrics"].get("arousal_strict_monotonic")
        )
        n_emotion_keep = sum(
            1 for g in valid_rows if g["metrics"].get("emotion_preserved")
        )
        span_gaps = [
            float(g["metrics"].get("arousal_span_gap", 0.0)) for g in valid_rows
        ]
        wer_low = [float(g["metrics"].get("wer_low", 0.0)) for g in valid_rows]
        wer_med = [float(g["metrics"].get("wer_medium", 0.0)) for g in valid_rows]
        wer_high = [float(g["metrics"].get("wer_high", 0.0)) for g in valid_rows]

        metrics.update({
            "monotonicity_rate": n_monotonic / n_valid,
            "strict_monotonicity_rate": n_strict / n_valid,
            "emotion_preservation_rate": n_emotion_keep / n_valid,
            "mean_span_gap": float(np.mean(span_gaps)),
            "min_span_gap": float(min(span_gaps)),
            "max_span_gap": float(max(span_gaps)),
            "mean_wer_low": float(np.mean(wer_low)),
            "mean_wer_medium": float(np.mean(wer_med)),
            "mean_wer_high": float(np.mean(wer_high)),
        })

    # 强度结论：明确不当词级真值，继承 08 自证风险.
    # self_evidence_risk = 任一 valid 组的 arousal evaluator 标记（若无 valid 组，
    # 回退到任一 invalid 组的标记；都无则 False）.
    self_evidence_risk = False
    for g in group_rows:
        ident = g.get("evaluator") or {}
        risk = ident.get("self_evidence_risk")
        if risk is True:
            self_evidence_risk = True
            break

    metrics["intensity_conclusion"] = {
        "scope": INTENSITY_CONCLUSION_SCOPE,
        "self_evidence_risk": bool(self_evidence_risk),
        "iemocap_weak_arousal_is_word_level_truth": False,
        "n_valid_groups_basis": n_valid,
        "overall_quality_metrics_kept_separate": (
            "v1 whole-utterance WER / Emo-SIM / DTW remain overall-quality "
            "identity and are NOT mixed into this triplet intensity score"
        ),
    }

    agg = {
        "evidence_tier": TRIPLET_EVIDENCE_TIER,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "n_samples": n_total,  # aggregate 层面 n_samples = 组数（非样本数）
        "metrics": metrics,
    }
    validate_aggregate(agg)
    return agg


# ============================================================
# 全管线编排
# ============================================================


def evaluate_triplet_dataset(
    triplet_specs: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    *,
    arousal_eval: Any,
    emotion_eval: Any,
    wer_eval: Any | None = None,
) -> dict[str, Any]:
    """完整三元生成 + 评测管线。

    Args:
        triplet_specs: 每条 = ``{group_id, base_case, control_spans: {tier: span}}``。
            base_case 至少包含 ``text`` / ``emotion`` / ``speaker`` / ``prompt_ref``
            / ``checkpoint_sha256`` / ``source_revision`` / ``seed_policy``
            / ``decode_config``。
        generation_rows: v2 GenerationRow 列表。每条携带非合同元数据
            ``group_id`` + ``intensity_tier``（orchestrator 用其分组）。
            未匹配任何 spec 的孤儿行被忽略（不 hard-fail）。
        arousal_eval: 实现 ``ArousalEvaluator`` 接口的对象（Fake 或真实）。
        emotion_eval: 实现 ``EmotionEvaluator`` 接口的对象。
        wer_eval: 可选 ``WerEvaluator``。None 时 WER 全部记 0（用于无 ASR 场景）。

    Returns:
        ``{"member_rows": [EvaluationRow, ...],
        "group_rows": [TripletGroupRow, ...],
        "aggregate": Aggregate, "n_groups": int}``

    Raises:
        ValueError: spec 缺失 group_id / base_case / control_spans。
    """
    # 索引 generation rows：优先 (group_id, intensity_tier) 元数据，回退 utt_id 约定.
    # 入口对每条 generation_row 调 validate_generation_row（核验 #10），保证 schema §2
    # 四族身份各≥1 + seed 非负 int；脏 row 在此 hard-fail，不进 group 评测.
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    by_utt: dict[str, dict[str, Any]] = {}
    for g in generation_rows:
        validate_generation_row(g)  # raises ValueError on bad row
        gid = g.get("group_id")
        tier = g.get("intensity_tier")
        uid = g.get("utt_id")
        if uid:
            by_utt[uid] = g
        if gid and tier:
            by_pair[(gid, tier)] = g

    arousal_identity = arousal_eval.identity()
    emotion_identity = emotion_eval.identity()
    wer_identity = wer_eval.identity() if wer_eval is not None else {
        "name": "no-wer-evaluator",
        "version": "absent",
    }

    member_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for spec in triplet_specs:
        group_id = spec.get("group_id")
        base_case = spec.get("base_case") or {}
        control_spans = spec.get("control_spans") or {}
        if not group_id:
            raise ValueError(f"triplet spec missing group_id: {spec!r}")
        if not base_case:
            raise ValueError(f"triplet spec '{group_id}' missing base_case")
        if not control_spans:
            raise ValueError(f"triplet spec '{group_id}' missing control_spans")

        # 每档 generation row + per-member eval row.
        per_tier_rows: dict[str, dict[str, Any]] = {}
        for tier in INTENSITY_TIERS:
            ctrl = control_spans.get(tier)
            if ctrl is None:
                continue  # spec 未声明该档 → 成员缺失，稍后由 membership 校验判 invalid
            # 优先按元数据匹配；否则按 utt_id 约定（group_id + "_" + tier）.
            gen = by_pair.get((group_id, tier))
            if gen is None:
                conv_utt = f"{group_id}_{tier}"
                gen = by_utt.get(conv_utt)
            if gen is None:
                continue  # 缺失该档生成行 → 成员缺失

            # 控制 span 的 utt_id 优先取 generation_row 的（防止命名漂移）.
            utt_id = gen.get("utt_id") or ctrl.get("utt_id") or f"{group_id}_{tier}"

            # 仅对 EOS 行调用 evaluator（非 EOS 不进声学）.
            fr = gen.get("finish_reason")
            if fr == "eos" and gen.get("wav_path"):
                wav_path = gen["wav_path"]
                try:
                    arousal_out = arousal_eval.predict_frames(wav_path)
                except Exception as exc:
                    raise RuntimeError(
                        f"group '{group_id}' tier '{tier}' hard-fail: "
                        f"arousal evaluator failed: {exc}"
                    ) from exc
                try:
                    emotion_out = emotion_eval.predict_frames(wav_path)
                except Exception as exc:
                    raise RuntimeError(
                        f"group '{group_id}' tier '{tier}' hard-fail: "
                        f"emotion evaluator failed: {exc}"
                    ) from exc
                if wer_eval is not None:
                    try:
                        wer_out = wer_eval.transcribe(wav_path)
                        hypothesis_text = wer_out.get("hypothesis_text", "")
                    except Exception as exc:
                        raise RuntimeError(
                            f"group '{group_id}' tier '{tier}' hard-fail: "
                            f"wer evaluator failed: {exc}"
                        ) from exc
                else:
                    hypothesis_text = ""
            else:
                # 非 EOS 行：跳过 evaluator，使用空输出（Task #11: metrics.valid=False，
                # 整组经 _check_group_membership 判 invalid，不进 aggregate 分母）。
                # emotion_out 也用空 frames（旧实现 np.zeros((1,5)) 会被误判为
                # 有效首类标签，污染 emotion_preservation_rate 分母）。
                arousal_out = {"frames": np.array([], dtype=np.float64),
                               "times_sec": np.array([], dtype=np.float64)}
                emotion_out = {"frames": np.array([], dtype=np.float64),
                               "times_sec": np.array([], dtype=np.float64),
                               "label_space": emotion_identity.get("label_space") or []}
                hypothesis_text = ""

            row = build_triplet_member_eval_row(
                utt_id, base_case, tier, gen, ctrl,
                arousal_out, emotion_out, hypothesis_text,
                arousal_identity, emotion_identity, wer_identity,
            )
            per_tier_rows[tier] = row
            member_rows.append(row)

        group_row = evaluate_triplet_group(group_id, base_case, per_tier_rows)
        group_rows.append(group_row)

    aggregate = build_triplet_aggregate(group_rows)

    return {
        "member_rows": member_rows,
        "group_rows": group_rows,
        "aggregate": aggregate,
        "n_groups": len(group_rows),
    }


__all__ = [
    # 常量
    "METRIC_CONTRACT_VERSION",
    "INTENSITY_TIERS",
    "INTENSITY_TIER_TO_ID",
    "INTENSITY_ID_TO_TIER",
    "TRIPLET_EVIDENCE_TIER",
    "INTENSITY_CONCLUSION_SCOPE",
    # WER 评测器
    "WerEvaluator",
    # 纯函数
    "compute_arousal_score",
    "compute_emotion_prediction",
    "compute_wer",
    "extract_span_range",
    # 行构建
    "build_triplet_member_eval_row",
    "evaluate_triplet_group",
    "build_triplet_aggregate",
    "evaluate_triplet_dataset",
    # 内部 helper（导出供 09-style 复用 / 测试）
    "json_key",
]
