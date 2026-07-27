"""Ticket 08 / 09 — 聚合排除对齐失败 + control 身份严格校验 focused 测试。

Ticket 08（task-8-brief §Step 1）：
- exact tier + 对齐失败（alignment_status != "aligned"）→ 不计入 exact
  aggregate 的 hit / direction / score / boundary_error 分母；
- 对齐失败的 exact 样本单独计入 ``n_exact_alignment_failed``；
- approximate tier 行为不受影响；
- 对齐失败的 exact 样本不影响 approximate aggregate。

Ticket 09（task-9-brief §Step 1，Grilling 决策 #3）：
- control 身份校验改对称强制：expected / gen 任一缺失 → hard-fail；
- per-pair prompt 死代码已删（``_extract_ctrl_prompt_core`` 不再存在）。

CPU 合同 / 行为测试：不加载真实模型，直接构造 dict 驱动被测函数。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from eval.eval_local_control import (
    _strict_pair,
    build_aggregate_from_rows,
    evaluate_spans_from_frames,
)


def _make_eval_row(
    evidence_tier: str,
    *,
    alignment_status: str = "aligned",
    boundary_error_sec: float | None = 0.1,
    front_hit: bool = True,
    back_hit: bool = True,
    both_hit: bool = True,
    direction: str = "correct",
    front_score: float = 0.9,
    back_score: float = 0.8,
) -> dict[str, Any]:
    """构造最小合法 EvaluationRow dict（仅含 aggregate 必需字段）。"""
    return {
        "boundary_evidence_tier": evidence_tier,
        "metrics": {
            "front_span": {"hit": front_hit, "score": front_score},
            "back_span": {"hit": back_hit, "score": back_score},
            "front_back_both_hit": both_hit,
            "transition_direction": direction,
            "boundary_error_sec": boundary_error_sec,
            "alignment_status": alignment_status,
        },
    }


class TestExactAlignmentFailedExcluded:
    """Task #8: exact tier 对齐失败样本不计入 exact aggregate 分母。"""

    def test_alignment_failed_not_in_exact_aggregate(self):
        """exact + alignment failed → n_samples 仅含 aligned 那条。"""
        rows = [
            _make_eval_row("exact", alignment_status="failed",
                            boundary_error_sec=None),
            _make_eval_row("exact", alignment_status="aligned",
                            boundary_error_sec=0.05),
        ]
        agg = build_aggregate_from_rows(rows, "exact")
        # exact aggregate 分母只含真正对齐成功的样本
        assert agg["n_samples"] == 1
        # 对齐失败的 exact 样本单独计数
        assert agg["n_exact_alignment_failed"] == 1

    def test_alignment_failed_excluded_from_hit_rate(self):
        """对齐失败样本的 hit/score 不污染 exact aggregate。"""
        rows = [
            # 失败样本：front/back 均 miss，score 0 → 不应拉低 exact 命中率
            _make_eval_row("exact", alignment_status="failed",
                            front_hit=False, back_hit=False,
                            both_hit=False, direction="other",
                            front_score=0.0, back_score=0.0,
                            boundary_error_sec=None),
            # 对齐成功样本：双 hit
            _make_eval_row("exact", alignment_status="aligned",
                            front_hit=True, back_hit=True,
                            both_hit=True, direction="correct",
                            front_score=1.0, back_score=1.0,
                            boundary_error_sec=0.0),
        ]
        agg = build_aggregate_from_rows(rows, "exact")
        m = agg["metrics"]
        # 分母 = 1（只含对齐成功样本）
        assert agg["n_samples"] == 1
        assert m["front_emotion_hit_rate"] == 1.0
        assert m["back_emotion_hit_rate"] == 1.0
        assert m["front_back_both_hit_rate"] == 1.0
        assert m["transition_correct_rate"] == 1.0
        assert m["mean_front_score"] == 1.0
        assert m["mean_back_score"] == 1.0
        # 对齐失败样本不贡献 boundary_error
        assert m["n_boundary_errors"] == 1

    def test_all_exact_alignment_failed_yields_zero_samples(self):
        """全部 exact 样本对齐失败 → n_samples=0 + n_exact_alignment_failed=N。"""
        rows = [
            _make_eval_row("exact", alignment_status="failed",
                            boundary_error_sec=None),
            _make_eval_row("exact", alignment_status="failed",
                            boundary_error_sec=None),
        ]
        agg = build_aggregate_from_rows(rows, "exact")
        assert agg["n_samples"] == 0
        assert agg["n_exact_alignment_failed"] == 2

    def test_approximate_tier_unaffected_by_alignment_status(self):
        """approximate tier 的 alignment_status 恒为 not_attempted，不影响聚合。"""
        rows = [
            _make_eval_row("approximate", alignment_status="not_attempted",
                            boundary_error_sec=None),
            _make_eval_row("approximate", alignment_status="not_attempted",
                            boundary_error_sec=None),
        ]
        agg = build_aggregate_from_rows(rows, "approximate")
        # approximate 不引入 n_exact_alignment_failed 字段
        assert "n_exact_alignment_failed" not in agg
        assert agg["n_samples"] == 2

    def test_mixed_tiers_only_exact_alignment_failed_counted(self):
        """exact 对齐失败 + approximate 行 → approximate 聚合不受影响。"""
        rows = [
            _make_eval_row("exact", alignment_status="failed",
                            boundary_error_sec=None),
            _make_eval_row("approximate", alignment_status="not_attempted",
                            boundary_error_sec=None),
        ]
        # exact aggregate 仅排除对齐失败样本
        agg_exact = build_aggregate_from_rows(rows, "exact")
        assert agg_exact["n_samples"] == 0
        assert agg_exact["n_exact_alignment_failed"] == 1
        # approximate aggregate 不受 exact 对齐失败影响
        agg_approx = build_aggregate_from_rows(rows, "approximate")
        assert agg_approx["n_samples"] == 1
        assert "n_exact_alignment_failed" not in agg_approx


# ============================================================
# Ticket 09 — control 身份对称强制 hard-fail + 删 prompt 死代码
# ============================================================


def _make_minimal_ctrl(utt_id: str, *, extra: dict | None = None) -> dict[str, Any]:
    """构造最小合法 control 记录（SupervisionSpan：utt_id 必需，无 prompt 字段）。

    schema §1 SupervisionSpan **无** ``prompt_row_ref`` / ``prompt_row`` 字段
    （Grilling 决策 #3）——控制记录只声明 control 身份。
    """
    rec: dict[str, Any] = {"utt_id": utt_id}
    if extra:
        rec.update(extra)
    return rec


def _make_minimal_gen(
    utt_id: str,
    *,
    control_row_ref: str | None = "sentinel",
    prompt_row_ref: str | None = "sentinel",
    extra: dict | None = None,
) -> dict[str, Any]:
    """构造最小合法 generation row（能进入 _strict_pair control 校验段）。

    默认 control/prompt 族用 "sentinel" 占位；测试可传 ``None`` 表示该族缺失，
    或传具体值（如 ``"control/u1"``）模拟内嵌身份。
    """
    row: dict[str, Any] = {
        "utt_id": utt_id,
        "finish_reason": "eos",
        "wav_path": f"wav/{utt_id}.wav",
        "checkpoint_sha256": "a" * 64,
    }
    if control_row_ref is not None:
        row["control_row_ref"] = control_row_ref
    if prompt_row_ref is not None:
        row["prompt_row_ref"] = prompt_row_ref
    if extra:
        row.update(extra)
    return row


class TestStrictPairControlHardFail:
    """Ticket 09: _strict_pair control 身份改对称强制 hard-fail。

    旧逻辑（票据 07）：双方非空才比，缺失静默跳过 → 漏检 gen 未内嵌 control。
    新逻辑（Grilling #3）：expected / gen 任一为空 → hard-fail（携 utt_id）。
    """

    def test_ctrl_declares_but_gen_missing_control_raises(self):
        """ctrl 声明 control_row_ref，gen 缺 control 族 → hard-fail。

        旧逻辑：gen_ctrl_core="" → `if expected_ctrl_core and gen_ctrl_core`
        条件不成立 → 静默跳过（漏检）。新逻辑：对称强制 → raise。
        """
        ctrl = [_make_minimal_ctrl("u1", extra={"control_row_ref": "control/u1"})]
        gen = [_make_minimal_gen("u1", control_row_ref=None)]
        with pytest.raises(ValueError, match=r"control"):
            _strict_pair(ctrl, gen)

    def test_gen_embeds_but_ctrl_mismatch_raises(self):
        """gen 内嵌 control_row_ref 与 ctrl 声明的不一致 → hard-fail。

        对称强制的 mismatch 分支：expected_ctrl_core（来自 ctrl utt_id 回退）
        与 gen_ctrl_core（来自 control_row_ref）不等 → raise。
        """
        # ctrl 无 control_row_ref → 回退到 utt_id="u1" → 核心 "u1"
        ctrl = [_make_minimal_ctrl("u1")]
        # gen 声明 control_row_ref="control/other" → 核心 "other"
        gen = [_make_minimal_gen("u1", control_row_ref="control/other")]
        with pytest.raises(ValueError, match=r"control"):
            _strict_pair(ctrl, gen)

    def test_both_present_and_match_passes(self):
        """正例：双方声明 control 身份且一致 → _strict_pair 返回配对列表。"""
        ctrl = [_make_minimal_ctrl("u1", extra={"control_row_ref": "control/u1"})]
        gen = [_make_minimal_gen("u1", control_row_ref="control/u1")]
        pairs = _strict_pair(ctrl, gen)
        assert len(pairs) == 1
        assert pairs[0][0] is ctrl[0]
        assert pairs[0][1] is gen[0]

    def test_ctrl_utt_id_fallback_matches_gen_passes(self):
        """ctrl 无 control_row_ref 但 utt_id 与 gen 内嵌核心一致 → 通过。

        _extract_ctrl_control_core 回退到 utt_id；gen control_row_ref="control/u1"
        核心也是 "u1" → 一致。
        """
        ctrl = [_make_minimal_ctrl("u1")]  # 无 control_row_ref
        gen = [_make_minimal_gen("u1", control_row_ref="control/u1")]
        pairs = _strict_pair(ctrl, gen)
        assert len(pairs) == 1

    def test_prompt_pair_check_removed(self):
        """``_extract_ctrl_prompt_core`` 已删；_strict_pair 不再校验 prompt 身份。

        即使 ctrl 声明了 prompt_row_ref 且 gen 内嵌不同 prompt_row_ref，
        per-pair prompt 校验也不触发（schema §1 SupervisionSpan 无 prompt 字段，
        gen 的 prompt 族存在性由 ``validate_generation_row`` 保证）。
        """
        import eval.eval_local_control as m

        # 死代码已物理删除
        assert not hasattr(m, "_extract_ctrl_prompt_core")

        # 即便 ctrl 带 prompt_row_ref（schema 非法但 dict 可构造），
        # gen 带不一致 prompt_row_ref → 不应因 prompt raise（control 一致即可）
        ctrl = [
            _make_minimal_ctrl(
                "u1",
                extra={
                    "control_row_ref": "control/u1",
                    "prompt_row_ref": "prompt/speaker_a",  # schema §1 非法字段
                },
            ),
        ]
        gen = [
            _make_minimal_gen(
                "u1",
                control_row_ref="control/u1",
                prompt_row_ref="prompt/speaker_b",  # 与 ctrl 不一致
            ),
        ]
        # 不 raise —— per-pair prompt 校验已删
        pairs = _strict_pair(ctrl, gen)
        assert len(pairs) == 1


# ============================================================
# Ticket 11 — 空 / NaN evaluator 输出 span invalid 不污染分母
# ============================================================


_LABELS = ["ang", "hap", "neu", "sad", "sur"]


class TestSpanMetricsValidity:
    """Task #11: evaluate_spans_from_frames 对空/全 NaN 输出标 valid=False。"""

    def test_nan_frames_span_invalid(self):
        """全 NaN frames → metrics.valid=False（NaN 不进 argmax/mean）。"""
        frames = np.full((4, 5), np.nan)
        times = np.array([0.0, 0.5, 1.0, 1.5])
        metrics = evaluate_spans_from_frames(
            frames, times, 0.5, "ang", "hap", _LABELS,
        )
        assert metrics.get("valid") is False

    def test_empty_frames_span_invalid(self):
        """空 frames → metrics.valid=False。"""
        frames = np.array([], dtype=np.float64)
        times = np.array([], dtype=np.float64)
        metrics = evaluate_spans_from_frames(
            frames, times, 0.5, "ang", "hap", _LABELS,
        )
        assert metrics.get("valid") is False

    def test_normal_frames_marked_valid(self):
        """正常 frames → metrics.valid=True（回归）。"""
        frames = np.zeros((4, 5))
        frames[:, 0] = 0.9  # ang dominant
        times = np.array([0.0, 0.5, 1.0, 1.5])
        metrics = evaluate_spans_from_frames(
            frames, times, 0.5, "ang", "hap", _LABELS,
        )
        assert metrics.get("valid") is True


def _make_eval_row_with_valid(
    evidence_tier: str,
    *,
    valid: bool = True,
    front_hit: bool = True,
    back_hit: bool = True,
    both_hit: bool = True,
    direction: str = "correct",
    front_score: float = 0.9,
    back_score: float = 0.8,
    alignment_status: str = "aligned",
    boundary_error_sec: float | None = 0.1,
) -> dict[str, Any]:
    """构造 EvaluationRow，metrics.valid 控制 Task #11 跳过逻辑。"""
    return {
        "boundary_evidence_tier": evidence_tier,
        "metrics": {
            "front_span": {"hit": front_hit, "score": front_score},
            "back_span": {"hit": back_hit, "score": back_score},
            "front_back_both_hit": both_hit,
            "transition_direction": direction,
            "boundary_error_sec": boundary_error_sec,
            "alignment_status": alignment_status,
            "valid": valid,
        },
    }


class TestAggregateExcludesInvalidOutput:
    """Task #11: metrics.valid=False 行不计入 aggregate 分母。"""

    def test_invalid_output_excluded_from_hit_rate(self):
        """valid=False 的样本（NaN/空 evaluator 输出）不污染 hit/score/direction 分母。"""
        rows = [
            # invalid: evaluator 空输出 → 不应拉低命中率
            _make_eval_row_with_valid(
                "approximate", valid=False,
                front_hit=False, back_hit=False, both_hit=False,
                direction="other", front_score=0.0, back_score=0.0,
                boundary_error_sec=None,
                alignment_status="not_attempted",
            ),
            # valid: 双 hit
            _make_eval_row_with_valid(
                "approximate", valid=True,
                front_hit=True, back_hit=True, both_hit=True,
                direction="correct", front_score=1.0, back_score=1.0,
                boundary_error_sec=None,
                alignment_status="not_attempted",
            ),
        ]
        agg = build_aggregate_from_rows(rows, "approximate")
        m = agg["metrics"]
        # 分母 = 1（仅 valid 样本）
        assert agg["n_samples"] == 1
        assert m["front_emotion_hit_rate"] == pytest.approx(1.0)
        assert m["back_emotion_hit_rate"] == pytest.approx(1.0)
        assert m["transition_correct_rate"] == pytest.approx(1.0)
        assert m["mean_front_score"] == pytest.approx(1.0)
        # invalid 样本计数单独保留
        assert agg.get("n_invalid_output") == 1

    def test_all_invalid_output_yields_zero_samples(self):
        """全部样本 valid=False → n_samples=0 + n_invalid_output=N。"""
        rows = [
            _make_eval_row_with_valid("approximate", valid=False,
                                      boundary_error_sec=None),
            _make_eval_row_with_valid("approximate", valid=False,
                                      boundary_error_sec=None),
        ]
        agg = build_aggregate_from_rows(rows, "approximate")
        assert agg["n_samples"] == 0
        assert agg.get("n_invalid_output") == 2

    def test_valid_flag_absent_treated_as_valid(self):
        """旧 row 无 metrics.valid 字段 → 视为 valid（向后兼容，不破坏既有路径）。"""
        rows = [
            _make_eval_row("approximate", alignment_status="not_attempted",
                           boundary_error_sec=None),
        ]
        agg = build_aggregate_from_rows(rows, "approximate")
        assert agg["n_samples"] == 1
        # 旧路径不出现 n_invalid_output 字段（未观测到 invalid 样本）
        assert "n_invalid_output" not in agg
