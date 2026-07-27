"""Ticket 03 — span → speech-token alignment focused tests.

验证 v2 监督 span 的 ``[start_sec, end_sec]`` 稳定映射到 teacher-forced
speech-token 序列上的 token 区间（MAP §3; brief 03; issue 03）。

覆盖：
- 基本映射 ``tok = round(time_sec * token_frame_rate_hz)`` + clip ``[0, N)``。
- utterance-level span 覆盖**全部有效 speech-token 列**（不覆盖 padding/前缀/特殊）。
- 单调性：``tok_end_prev <= tok_start_next``（允许相邻接）。
- 量化误差 ``quant_error_sec`` 记录（边缘恰落 token 边界 → 0）。
- fail-fast：空 span / 反向 / 越界（整段在音频外）/ 零覆盖 → ``valid=False`` + reason，
  **禁止静默扩展到整句**。
- word/span target 透传 emotion_mask / intensity_mask / target / raw_score / calibrated /
  supervision_weight（保持 01/02 的诚实监督语义）。
- collate 保持 span ↔ mask ↔ target ↔ source row **一一对应**；支持可变 span 数。
- 轻量集成：02 产出的 v2 小样本 tagged.jsonl + 合成 speech_token_len 形成可被（06 的）
  池化消费的 batch。

CPU 合成，无 GPU/重模型依赖（MAP §4）。speech-token 帧率 25 Hz 的来源见
``span_align.py`` 模块 docstring 与 ``sdd/reports/03-align-spans-to-tokens.md``。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cosyvoice.dataset.span_align import (
    STRATEGY_ID,
    align_spans_to_tokens,
    build_strategy_version,
    collate_aligned_spans,
)

# 审计确认的 speech-token 帧率（见 report 03）。
TOKEN_FRAME_RATE_HZ = 25.0
SV = build_strategy_version(TOKEN_FRAME_RATE_HZ)


def _span(**overrides: Any) -> dict[str, Any]:
    """最小合法 IEMOCAP 风格测试 span（word 粒度默认）。"""
    base: dict[str, Any] = {
        "utt_id": "utt_test",
        "label_source": "word_annotator_pseudo_label",
        "supervision_granularity": "word",
        "start_sec": 0.0,
        "end_sec": 1.0,
        "control_emotion_id": 3,
        "control_intensity_id": 2,
        "calibrated": False,
        "emotion_mask": True,
        "intensity_mask": True,
        "supervision_weight": 1.0,
        "emotion_soft_distribution": [0.1, 0.2, 0.5, 0.1, 0.1],
        "arousal": 2.5,
    }
    base.update(overrides)
    return base


# ============================================================
# A. 基本映射、量化、clip
# ============================================================


class TestAlignBasic:
    def test_basic_mapping_round(self):
        # 25 Hz: round(0.04*25)=round(1.0)=1; round(1.16*25)=round(29.0)=29
        spans = [_span(start_sec=0.04, end_sec=1.16)]
        result = align_spans_to_tokens(
            spans, speech_token_len=50, audio_duration_sec=2.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert len(result) == 1
        a = result[0]
        assert a["valid"] is True
        assert a["invalid_reason"] is None
        assert a["tok_start"] == 1
        assert a["tok_end"] == 29

    def test_quant_error_recorded(self):
        # 0.05s → round(1.25)=1 → 1/25=0.04 → err=|0.04-0.05|=0.01
        spans = [_span(start_sec=0.05, end_sec=1.0)]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["valid"] is True
        assert a["quant_error_sec"] == pytest.approx(0.01, abs=1e-9)

    def test_edge_exact_token_boundary_zero_error(self):
        # 恰落 token 边界 → 0 量化误差
        spans = [_span(start_sec=0.04, end_sec=1.00)]  # 1/25, 25/25
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["tok_start"] == 1
        assert a["tok_end"] == 25
        assert a["quant_error_sec"] == pytest.approx(0.0, abs=1e-9)

    def test_clip_to_speech_token_len(self):
        # span 超出 speech_token_len → clip tok_end 到 N（部分越界不判 invalid）
        spans = [_span(start_sec=0.0, end_sec=10.0)]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["valid"] is True
        assert a["tok_start"] == 0
        assert a["tok_end"] == 25  # clipped

    def test_padding_text_prefix_isolated(self):
        # speech-token 区段 = [0, speech_token_len)；padding/文本前缀/特殊 token
        # 不在本函数视野内（由训练侧 IGNORE_ID 标记）。utterance span 只覆盖 [0, N)，
        # 绝不覆盖 N 及以上列。
        N = 30
        spans = [_span(supervision_granularity="utterance", start_sec=0.0, end_sec=1.2)]
        result = align_spans_to_tokens(
            spans, speech_token_len=N, audio_duration_sec=1.2,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["tok_start"] == 0
        assert a["tok_end"] == N  # exactly [0, N) — 不含 N（padding 列）

    def test_token_rate_25_not_50_disambiguated(self):
        # 明确区分：span [0, 0.5s]，token_len=13（0.52s 的 25Hz token）。
        # 25Hz: round(0.5*25)=round(12.5)→12或13；50Hz: round(0.5*50)=25 → clip 到 13。
        # 关键：tok_end <= N=13。25Hz 下 round(12.5)=12（Python banker's rounding）。
        spans = [_span(start_sec=0.0, end_sec=0.5)]
        result = align_spans_to_tokens(
            spans, speech_token_len=13, audio_duration_sec=0.52,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        # 25Hz → round(12.5) = 12 (banker's rounding in Python 3)
        assert a["tok_end"] == 12
        # 若误用 50Hz → round(25.0) = 25 → clip 到 13。12 != 13，区分成功。


# ============================================================
# B. utterance-level span 覆盖全部有效 token
# ============================================================


class TestUtteranceSpan:
    def test_utterance_covers_all_valid_tokens(self):
        # 即使 end_sec 量化后 != N，utterance span 仍覆盖 [0, N)
        spans = [_span(supervision_granularity="utterance", start_sec=0.0, end_sec=1.0)]
        result = align_spans_to_tokens(
            spans, speech_token_len=37, audio_duration_sec=1.5,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["valid"] is True
        assert a["tok_start"] == 0
        assert a["tok_end"] == 37

    def test_esd_utterance_style(self):
        # ESD utterance span：intensity_mask=False，无 arousal
        span = _span(
            supervision_granularity="utterance",
            label_source="esd_fixed_medium_control",
            intensity_mask=False,
            intensity_policy="fixed_medium",
            start_sec=0.0, end_sec=1.816,
        )
        span.pop("arousal")
        result = align_spans_to_tokens(
            [span], speech_token_len=45, audio_duration_sec=1.8,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["valid"] is True
        assert a["tok_start"] == 0
        assert a["tok_end"] == 45
        assert a["intensity_mask"] is False
        assert a.get("arousal") is None


# ============================================================
# C. 单调性
# ============================================================


class TestMonotonicity:
    def test_adjacent_spans_ok(self):
        spans = [
            _span(start_sec=0.0, end_sec=0.5),
            _span(start_sec=0.5, end_sec=1.0),
        ]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert all(a["valid"] for a in result)
        assert result[0]["tok_end"] <= result[1]["tok_start"]

    def test_three_spans_tile_no_overlap_no_gap_monotonic(self):
        spans = [
            _span(start_sec=0.0, end_sec=0.4),
            _span(start_sec=0.4, end_sec=0.8),
            _span(start_sec=0.8, end_sec=1.2),
        ]
        result = align_spans_to_tokens(
            spans, speech_token_len=30, audio_duration_sec=1.2,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        for i in range(len(result) - 1):
            assert result[i]["tok_end"] <= result[i + 1]["tok_start"]

    def test_overlapping_input_clamped_monotonic(self):
        # 重叠输入（02 不会产生，但防御）：tok_start 被 clamp 到 prev tok_end
        spans = [
            _span(start_sec=0.0, end_sec=0.6),
            _span(start_sec=0.4, end_sec=1.0),  # 与上一个重叠
        ]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        # 输出仍单调
        assert result[0]["tok_end"] <= result[1]["tok_start"]


# ============================================================
# D. fail-fast：明确无效 mask + 原因
# ============================================================


class TestInvalidCases:
    def test_empty_span_invalid(self):
        spans = [_span(start_sec=1.0, end_sec=1.0)]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert result[0]["valid"] is False
        assert result[0]["invalid_reason"] is not None

    def test_reversed_boundary_invalid(self):
        spans = [_span(start_sec=0.8, end_sec=0.2)]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert result[0]["valid"] is False
        assert "reversed" in result[0]["invalid_reason"] or "empty" in result[0]["invalid_reason"]

    def test_out_of_range_after_invalid(self):
        spans = [_span(start_sec=2.0, end_sec=3.0)]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert result[0]["valid"] is False
        assert "out_of_range" in result[0]["invalid_reason"]

    def test_out_of_range_before_invalid(self):
        spans = [_span(start_sec=-1.0, end_sec=-0.5)]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert result[0]["valid"] is False
        assert "out_of_range" in result[0]["invalid_reason"]

    def test_zero_coverage_invalid(self):
        # span 过短 → 映射后 tok_start >= tok_end
        spans = [_span(start_sec=0.01, end_sec=0.02)]  # < 1 token at 25Hz
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert result[0]["valid"] is False
        assert "zero_coverage" in result[0]["invalid_reason"]

    def test_invalid_never_silently_extends_to_whole_sentence(self):
        # 关键不变量：无效 span **绝不**被静默扩展到 [0, N)
        for bad_span in [
            _span(start_sec=2.0, end_sec=3.0),       # out_of_range
            _span(start_sec=0.01, end_sec=0.02),     # zero_coverage
            _span(start_sec=0.8, end_sec=0.2),       # reversed
        ]:
            result = align_spans_to_tokens(
                [bad_span], speech_token_len=25, audio_duration_sec=1.0,
                token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
            )
            a = result[0]
            assert a["valid"] is False
            assert not (a["tok_start"] == 0 and a["tok_end"] == 25), (
                f"invalid span must not be silently extended to [0, N): "
                f"got tok=[{a['tok_start']},{a['tok_end']}]"
            )


# ============================================================
# E. 透传监督字段（01/02 的诚实语义）
# ============================================================


class TestPassthrough:
    def test_word_span_passthrough(self):
        spans = [_span(
            emotion_mask=True, intensity_mask=True,
            emotion_soft_distribution=[0.0, 0.0, 1.0, 0.0, 0.0],
            arousal=3.5, raw_score=0.9, calibrated=False,
            supervision_weight=0.8, control_emotion_id=3, control_intensity_id=3,
        )]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["emotion_mask"] is True
        assert a["intensity_mask"] is True
        assert a["emotion_soft_distribution"] == [0.0, 0.0, 1.0, 0.0, 0.0]
        assert a["arousal"] == 3.5
        assert a["raw_score"] == 0.9
        assert a["calibrated"] is False
        assert a["supervision_weight"] == 0.8
        assert a["control_emotion_id"] == 3
        assert a["control_intensity_id"] == 3

    def test_esd_no_arousal_no_raw_score(self):
        span = _span(
            supervision_granularity="utterance",
            label_source="esd_fixed_medium_control",
            intensity_mask=False, intensity_policy="fixed_medium",
            start_sec=0.0, end_sec=1.0,
        )
        span.pop("arousal")
        span.pop("raw_score", None)
        result = align_spans_to_tokens(
            [span], speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["intensity_mask"] is False
        assert a.get("arousal") is None
        assert a.get("raw_score") is None


# ============================================================
# F. strategy_version
# ============================================================


class TestStrategyVersion:
    def test_strategy_version_recorded_per_span(self):
        spans = [_span()]
        result = align_spans_to_tokens(
            spans, speech_token_len=25, audio_duration_sec=1.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert result[0]["strategy_version"] == SV

    def test_strategy_version_includes_rate(self):
        sv25 = build_strategy_version(25.0)
        sv12 = build_strategy_version(12.5)
        assert "25" in sv25
        assert "12.5" in sv12
        assert sv25 != sv12

    def test_strategy_id_constant(self):
        assert STRATEGY_ID.startswith("emofilm_span_align")


# ============================================================
# G. collate：可变 span 数，一一对应
# ============================================================


class TestCollate:
    def test_variable_span_count_1to1(self):
        sample_a = {
            "utt_id": "a",
            "aligned_spans": [
                {"tok_start": 0, "tok_end": 10, "valid": True, "invalid_reason": None,
                 "emotion_mask": True, "intensity_mask": False,
                 "emotion_soft_distribution": [1.0, 0, 0, 0, 0],
                 "arousal": None, "raw_score": None, "calibrated": False,
                 "supervision_weight": 1.0, "control_emotion_id": 1, "control_intensity_id": 2},
            ],
        }
        sample_b = {
            "utt_id": "b",
            "aligned_spans": [
                {"tok_start": 0, "tok_end": 5, "valid": True, "invalid_reason": None,
                 "emotion_mask": True, "intensity_mask": False,
                 "emotion_soft_distribution": [0, 1.0, 0, 0, 0],
                 "arousal": None, "raw_score": None, "calibrated": False,
                 "supervision_weight": 1.0, "control_emotion_id": 2, "control_intensity_id": 2},
                {"tok_start": 5, "tok_end": 10, "valid": True, "invalid_reason": None,
                 "emotion_mask": True, "intensity_mask": False,
                 "emotion_soft_distribution": [0, 0, 1.0, 0, 0],
                 "arousal": None, "raw_score": None, "calibrated": False,
                 "supervision_weight": 1.0, "control_emotion_id": 3, "control_intensity_id": 2},
            ],
        }
        batch = collate_aligned_spans([sample_a, sample_b])
        # B=2, max_spans=2
        assert batch["span_mask"].shape == (2, 2)
        assert batch["span_mask"][0].tolist() == [True, False]
        assert batch["span_mask"][1].tolist() == [True, True]
        # 1-1：tok_start 保持
        assert batch["span_tok_start"][0, 0].item() == 0
        assert batch["span_tok_start"][1, 0].item() == 0
        assert batch["span_tok_start"][1, 1].item() == 5
        # tok_end 保持
        assert batch["span_tok_end"][0, 0].item() == 10
        assert batch["span_tok_end"][1, 1].item() == 10
        # span 数量
        assert batch["span_count"].tolist() == [1, 2]
        # soft dist 保持
        assert batch["span_emotion_soft_dist"][0, 0].tolist() == [1.0, 0, 0, 0, 0]
        assert batch["span_emotion_soft_dist"][1, 1].tolist() == [0, 0, 1.0, 0, 0]
        # padded span 的 mask=False
        assert bool(batch["span_mask"][0, 1].item()) is False

    def test_collate_empty_spans(self):
        sample = {"utt_id": "x", "aligned_spans": []}
        batch = collate_aligned_spans([sample])
        assert batch["span_count"].tolist() == [0]
        assert batch["span_mask"].shape == (1, 0)

    def test_collate_preserves_invalid_spans(self):
        # 无效 span 仍进入 batch（valid=False），调用方决定是否 mask 掉
        sample = {
            "utt_id": "x",
            "aligned_spans": [
                {"tok_start": 0, "tok_end": 5, "valid": True, "invalid_reason": None,
                 "emotion_mask": True, "intensity_mask": True,
                 "emotion_soft_distribution": [0.2] * 5, "arousal": 2.0,
                 "raw_score": 0.7, "calibrated": False,
                 "supervision_weight": 1.0, "control_emotion_id": 1, "control_intensity_id": 1},
                {"tok_start": 0, "tok_end": 0, "valid": False,
                 "invalid_reason": "zero_coverage_after_clip",
                 "emotion_mask": False, "intensity_mask": False,
                 "emotion_soft_distribution": None, "arousal": None,
                 "raw_score": None, "calibrated": False,
                 "supervision_weight": 0.0, "control_emotion_id": 1, "control_intensity_id": 1},
            ],
        }
        batch = collate_aligned_spans([sample])
        assert batch["span_valid"][0, 0].item() is True or batch["span_valid"][0, 0].item() == True
        assert bool(batch["span_valid"][0, 1].item()) is False
        assert batch["span_invalid_reason"][0][1] == "zero_coverage_after_clip"


# ============================================================
# H. 集成：02 小样本 tagged.jsonl → 可消费 batch
# ============================================================


class TestIntegration:
    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            pytest.skip(f"02 sample not available: {path}")
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_iemocap_sample_align_and_collate(self):
        repo_root = Path(__file__).resolve().parent.parent
        spans = self._load_jsonl(
            repo_root / "data" / "contracts" / "emofilm_v2" / "sources" / "iemocap" / "tagged.jsonl"
        )
        by_utt: dict[str, list[dict[str, Any]]] = {}
        for s in spans:
            by_utt.setdefault(s["utt_id"], []).append(s)
        utt_id, utt_spans = next(iter(by_utt.items()))
        # 合成：10s 音频 → 250 tokens @25Hz
        result = align_spans_to_tokens(
            utt_spans, speech_token_len=250, audio_duration_sec=10.0,
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        assert len(result) == len(utt_spans)
        for a in result:
            if a["valid"]:
                assert 0 <= a["tok_start"] < a["tok_end"] <= 250
        # 单调
        valid_results = [a for a in result if a["valid"]]
        for i in range(len(valid_results) - 1):
            assert valid_results[i]["tok_end"] <= valid_results[i + 1]["tok_start"]
        # 形成 batch（用 fake，不跑真模型）
        batch = collate_aligned_spans([{"utt_id": utt_id, "aligned_spans": result}])
        assert batch["span_mask"].shape[0] == 1
        assert batch["span_count"][0].item() == len(result)

    def test_esd_sample_utterance_align(self):
        repo_root = Path(__file__).resolve().parent.parent
        spans = self._load_jsonl(
            repo_root / "data" / "contracts" / "emofilm_v2" / "sources" / "esd" / "tagged.jsonl"
        )
        span = spans[0]
        N = 45  # ~1.8s @25Hz
        result = align_spans_to_tokens(
            [span], speech_token_len=N, audio_duration_sec=span["end_sec"],
            token_frame_rate_hz=TOKEN_FRAME_RATE_HZ, strategy_version=SV,
        )
        a = result[0]
        assert a["valid"] is True
        assert a["tok_start"] == 0
        assert a["tok_end"] == N
        assert a["intensity_mask"] is False
        assert a.get("arousal") is None
