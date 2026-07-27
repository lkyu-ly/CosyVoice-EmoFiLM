"""Ticket 06（experiment-readiness）— MFA 对齐守卫测试（B10 计数区分 + B11 词序同构）。

B11：MFA 对齐词序须与文本词序同构，否则 ``words[k-1]`` 可能指向错误词（clitic
拆分如 ``you'll`` → ``you`` + ``'ll``），boundary_sec 静默偏移约一个词长 → 标
``word_sequence_mismatch``（不进 exact 分母）。

B10：exact tier 区分"未尝试"（aligner 缺位）/"失败"（对齐或词序失败）/"成功"
计数，使 aligner 缺位可见（不再静默 n=0 exit 0）。
"""
from __future__ import annotations

from eval.eval_local_control import (
    AlignmentResult,
    WordBoundary,
    build_aggregate_from_rows,
    resolve_aligned_boundary_sec,
)


def _aligned(words):
    return AlignmentResult(status="aligned", words=words)


# ---------------- B11: 词序同构校验 ----------------

def test_resolve_returns_boundary_when_isomorphic():
    """词序同构 → 返回 boundary_sec + aligned。"""
    words = [WordBoundary(0.0, 0.3, "hello"), WordBoundary(0.3, 0.6, "world")]
    sec, status = resolve_aligned_boundary_sec(_aligned(words), 1, "hello world")
    assert status == "aligned"
    assert sec == 0.3


def test_resolve_flags_clitic_split_as_mismatch():
    """clitic 拆分（you'll → you + 'll，词数 1 vs 2）→ word_sequence_mismatch（B11）。"""
    words = [
        WordBoundary(0.0, 0.2, "you"),
        WordBoundary(0.2, 0.4, "'ll"),
        WordBoundary(0.4, 0.7, "world"),
    ]
    sec, status = resolve_aligned_boundary_sec(_aligned(words), 1, "you'll world")
    assert sec is None
    assert status == "word_sequence_mismatch"


def test_resolve_text_none_skips_isomorphism():
    """text=None → 不做词序校验（兼容既有调用方/近似路径）。"""
    words = [WordBoundary(0.0, 0.3, "hello"), WordBoundary(0.3, 0.6, "world")]
    sec, status = resolve_aligned_boundary_sec(_aligned(words), 1, None)
    assert status == "aligned"


def test_resolve_word_face_mismatch():
    """前 k 词词面不同（MFA 误识）→ word_sequence_mismatch（k=2 校验前 2 词）。"""
    words = [
        WordBoundary(0.0, 0.3, "hello"),
        WordBoundary(0.3, 0.6, "word"),  # 应为 "world"
        WordBoundary(0.6, 0.9, "test"),
    ]
    sec, status = resolve_aligned_boundary_sec(_aligned(words), 2, "hello world test")
    assert sec is None
    assert status == "word_sequence_mismatch"


# ---------------- B10: 计数区分 ----------------

def _metric(status, hit=False):
    score = 0.9 if hit else 0.0
    return {
        "valid": True,
        "alignment_status": status,
        "front_span": {"hit": hit, "score": score},
        "back_span": {"hit": hit, "score": score},
        "front_back_both_hit": hit,
        "transition_direction": "correct" if hit else "other",
        "front_score": score,
        "back_score": score,
        "boundary_error_sec": 0.1 if hit else None,
    }


def test_build_aggregate_distinguishes_not_attempted_and_failed():
    """exact tier：not_attempted 与 failed 分别计数（B10 可见性），仅 aligned 进分母。"""
    rows = [
        {"boundary_evidence_tier": "exact", "metrics": _metric("aligned", hit=True)},
        {"boundary_evidence_tier": "exact", "metrics": _metric("not_attempted")},
        {"boundary_evidence_tier": "exact", "metrics": _metric("failed")},
        {"boundary_evidence_tier": "exact", "metrics": _metric("word_sequence_mismatch")},
    ]
    agg = build_aggregate_from_rows(rows, "exact")
    assert agg["n_samples"] == 1
    assert agg["n_exact_alignment_not_attempted"] == 1
    assert agg["n_exact_alignment_failed"] == 2  # failed + word_sequence_mismatch


def test_build_aggregate_all_not_attempted_reports_zero_with_count():
    """全部 not_attempted（aligner 缺位）→ n_samples=0 + not_attempted 计数可见（B10 核心）。"""
    rows = [
        {"boundary_evidence_tier": "exact", "metrics": _metric("not_attempted")},
        {"boundary_evidence_tier": "exact", "metrics": _metric("not_attempted")},
    ]
    agg = build_aggregate_from_rows(rows, "exact")
    assert agg["n_samples"] == 0
    assert agg["n_exact_alignment_not_attempted"] == 2
    assert agg["n_exact_alignment_failed"] == 0
