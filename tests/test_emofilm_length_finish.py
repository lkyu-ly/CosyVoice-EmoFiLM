"""Ticket 05 — v2 长度合同与结构化 finish reason（CPU fake backbone）。

验证（brief 05 §C / MAP §3 长度不变量 / schema GenerationRow.finish_reason）：
  - 长度由 text_len + min/max_token_text_ratio + max_len_hard_cap 推导；
  - 解码前 ``max_len > min_len`` 否则 ``finish_reason="input_rejected"`` 且**不进采样**；
  - 解码产出结构化 DecodeResult（tokens/finish_reason/min_len/max_len/
    num_valid_speech_tokens/invalid_token_retries）；
  - finish_reason ∈ FINISH_REASONS，互斥稳定；
  - 仅 eos 进 Flow/HiFT + 正式 WAV；非 eos 不得带 wav_path
    （通过 ``validate_generation_row``）；
  - 旧错误边界 text_len>=100：要么 ratio 推导出可行 max_len 正常完成，
    要么 hard cap 不足时 input_rejected；绝不静默成功/截断。

CPU fake 测试（MAP §4）：``_FakeQwen`` 透传 hidden，``_FixedDecoder`` /
脚本采样器控制 token 序列。不加载真实权重，不需 GPU。
"""
from pathlib import Path

import pytest
import torch

from cosyvoice.llm.llm_emotion import (
    DEFAULT_MAX_LEN_HARD_CAP,
    MAX_INVALID_TOKEN_RETRIES,
    DecodeResult,
    Qwen2LM_Emotion,
)
from tests._emofilm_fakes import _FakeBackbone, _FakeHF, _FakeQwen
from tools.build_emofilm_contract import (
    FINISH_REASONS,
    validate_generation_row,
)


# ============================================================
# fakes：从 tests._emofilm_fakes 复用（_FakeBackbone / _FakeHF / _FakeQwen
# 原先在本文件本地定义，现已 DRY 整合到共享测试辅助模块）。
# 注：原本地版 forward 的 mask tensor 未带显式 ``device=xs.device``，
# 整合版统一带 device；CPU 测试下行为完全等价（``xs.device`` 为 CPU、
# ``torch.ones`` 默认也在 CPU）。
# ============================================================


VOCAB = 10
EMO = 6
INTEN = 4


def _make_model(speech_token_size=VOCAB, sampling=None):
    return Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=speech_token_size,
        emotion_vocab_size=EMO,
        intensity_vocab_size=INTEN,
        llm=_FakeQwen(4),
        sampling=sampling or (lambda scores, decoded, s: 2),
    )


def _inputs(text_len=3):
    """Full inference kwargs（含声学侧 prompt 字段）。"""
    return {
        "text_token": torch.tensor([[2 + i for i in range(text_len)]]),
        "text_len": torch.tensor([text_len], dtype=torch.int32),
        "emotion_ids": torch.ones(1, text_len, dtype=torch.long),
        "intensity_ids": torch.ones(1, text_len, dtype=torch.long),
        "prompt_speech_token": torch.zeros(1, 0, dtype=torch.long),
        "prompt_speech_token_len": torch.tensor([0], dtype=torch.int32),
        "embedding": torch.zeros(1, 4),
    }


def _decode_inputs(text_len=3):
    """LLM-level decode kwargs（decode 只接受 target text + emotion/intensity）。"""
    return {
        "text_token": torch.tensor([[2 + i for i in range(text_len)]]),
        "text_len": torch.tensor([text_len], dtype=torch.int32),
        "emotion_ids": torch.ones(1, text_len, dtype=torch.long),
        "intensity_ids": torch.ones(1, text_len, dtype=torch.long),
    }


def _sampler_eos_after(min_len, eos, valid=2):
    """Produce valid tokens until len(decoded) >= min_len, then EOS."""
    def _s(scores, decoded, s):
        if len(decoded) >= min_len:
            return eos
        return valid
    return _s


def _sampler_always(token):
    def _s(scores, decoded, s):
        return token
    return _s


def _generation_row(result, utt_id="u1", wav_path=None):
    row = {
        "utt_id": utt_id,
        "finish_reason": result.finish_reason,
        "source_revision": "deadbeef" * 5,
        "checkpoint_sha256": "0" * 64,
        "control_row_ref": "ctrl/u1.jsonl:0",
        "prompt_row_ref": "prm/u1.jsonl:0",
        "decode_config": {
            "min_token_text_ratio": 2,
            "max_token_text_ratio": 20,
            "max_len_hard_cap": DEFAULT_MAX_LEN_HARD_CAP,
        },
        "seed": 1984,
    }
    if wav_path is not None:
        row["wav_path"] = wav_path
    return row


# ============================================================
# A. 结构化结果 + finish_reason 不变量
# ============================================================


def test_decode_result_is_dataclass_with_required_fields():
    result = DecodeResult(
        tokens=[2, 3], finish_reason="eos", min_len=2, max_len=10,
        num_valid_speech_tokens=2, invalid_token_retries=0, text_len=1,
    )
    assert result.tokens == [2, 3]
    assert result.finish_reason == "eos"
    assert result.min_len == 2
    assert result.max_len == 10
    assert result.num_valid_speech_tokens == 2
    assert result.invalid_token_retries == 0
    assert result.text_len == 1


def test_short_text_eos_finish_reason():
    """短文本：采样器在 min_len 后产 EOS → finish_reason=eos。"""
    eos = VOCAB
    model = _make_model(sampling=_sampler_eos_after(min_len=6, eos=eos))
    result = model.decode(**_decode_inputs(text_len=3),
                          max_token_text_ratio=20, min_token_text_ratio=2)
    assert result.finish_reason == "eos"
    assert result.min_len == 6          # int(3 * 2)
    assert result.max_len == 60         # int(3 * 20)
    assert result.num_valid_speech_tokens == 6
    assert len(result.tokens) == 6
    assert result.invalid_token_retries == 0
    assert result.finish_reason in FINISH_REASONS


def test_eos_finish_reason_yields_tokens_via_inference():
    """inference 生成器在 eos 时产出 speech token 给 Flow/HiFT。"""
    eos = VOCAB
    model = _make_model(sampling=_sampler_eos_after(min_len=4, eos=eos))
    tokens = list(model.inference(**_inputs(text_len=2)))
    assert len(tokens) == 4
    assert all(int(t) == 2 for t in tokens)
    # 结构化结果对调用方可见。
    assert model.last_decode_result.finish_reason == "eos"


def test_old_boundary_text_len_100_completes_via_ratio():
    """v1 在 text_len>=100 确定性失败（max_len=200<=min_len=200 静默截断）。

    v2：足够 hard cap 时 ratio 推导出 max_len=2000 > min_len=200，
    采样器在 min_len 后产 EOS → 正常 eos 完成（绝不静默成功）。
    """
    eos = VOCAB
    model = _make_model(sampling=_sampler_eos_after(min_len=200, eos=eos))
    result = model.decode(**_decode_inputs(text_len=100),
                          max_token_text_ratio=20, min_token_text_ratio=2,
                          max_len_hard_cap=2000)
    assert result.min_len == 200
    assert result.max_len == 2000       # ratio_max=2000, hard_cap=2000 → 2000
    assert result.finish_reason == "eos"
    assert result.num_valid_speech_tokens == 200
    assert result.finish_reason in FINISH_REASONS


def test_old_boundary_text_len_100_input_rejected_when_hard_cap_too_small():
    """同一 text_len=100，但 hard_cap=150 < min_len=200 → 结构化 input_rejected。"""
    sampler_calls = []
    def _s(scores, decoded, s):
        sampler_calls.append(1)
        return 2
    model = _make_model(sampling=_s)
    result = model.decode(**_decode_inputs(text_len=100),
                          max_token_text_ratio=20, min_token_text_ratio=2,
                          max_len_hard_cap=150)
    assert result.finish_reason == "input_rejected"
    assert result.min_len == 200
    assert result.max_len == 150        # min(2000, 150)
    assert result.num_valid_speech_tokens == 0
    assert result.invalid_token_retries == 0
    assert result.text_len == 100
    assert sampler_calls == []          # NO sampling entered


# ============================================================
# B. hard-cap 拒绝（解码前不变量）
# ============================================================


def test_hard_cap_rejection_no_sampling():
    """max_len <= min_len → input_rejected，不进 token sampling。"""
    sampler_calls = []
    def _s(scores, decoded, s):
        sampler_calls.append(1)
        return 2
    model = _make_model(sampling=_s)
    # text_len=10: min_len=20, ratio_max=200; hard_cap=15 → max_len=15 < min_len=20
    result = model.decode(**_decode_inputs(text_len=10),
                          max_token_text_ratio=20, min_token_text_ratio=2,
                          max_len_hard_cap=15)
    assert result.finish_reason == "input_rejected"
    assert result.min_len == 20
    assert result.max_len == 15
    assert sampler_calls == []
    assert result.finish_reason in FINISH_REASONS


def test_hard_cap_equal_min_len_rejected():
    """边界：max_len == min_len 也拒绝（要求严格 >）。"""
    model = _make_model()
    # text_len=10: min_len=20; ratio_max=200; hard_cap=20 → max_len=20 == min_len=20
    result = model.decode(**_decode_inputs(text_len=10),
                          max_token_text_ratio=20, min_token_text_ratio=2,
                          max_len_hard_cap=20)
    assert result.finish_reason == "input_rejected"


# ============================================================
# C. 无 EOS 跑满 max_len → max_len_reached
# ============================================================


def test_no_eos_max_len_reached():
    """采样器恒产合法 speech token、从不产 EOS → max_len_reached（不带 wav）。"""
    model = _make_model(sampling=_sampler_always(2))
    result = model.decode(**_decode_inputs(text_len=3),
                          max_token_text_ratio=20, min_token_text_ratio=2)
    assert result.finish_reason == "max_len_reached"
    assert result.min_len == 6
    assert result.max_len == 60
    assert result.num_valid_speech_tokens == 60
    assert len(result.tokens) == 60
    assert result.finish_reason in FINISH_REASONS


def test_max_len_reached_inference_yields_nothing():
    """非 eos 的 inference 不向 Flow/HiFT 产任何 token（防静默成功）。"""
    model = _make_model(sampling=_sampler_always(2))
    tokens = list(model.inference(**_inputs(text_len=3)))
    assert tokens == []
    assert model.last_decode_result.finish_reason == "max_len_reached"


# ============================================================
# D. 连续非法/辅助 token 重试耗尽 → invalid_token_retry_exhausted
# ============================================================


def test_invalid_aux_token_retry_exhausted():
    """采样器恒产辅助 token (>=speech_token_size) → 100 次重试后耗尽。"""
    model = _make_model(sampling=_sampler_always(VOCAB + 1))   # aux token
    result = model.decode(**_decode_inputs(text_len=3))
    assert result.finish_reason == "invalid_token_retry_exhausted"
    assert result.num_valid_speech_tokens == 0
    # 第一个解码步容忍 MAX_INVALID_TOKEN_RETRIES 次非法重采样，
    # 之后下一次再失败即放弃（与 v1 ``trials > 100`` 边界一致，
    # 见 MAP §2 llm_emotion.py:176）。
    assert result.invalid_token_retries == MAX_INVALID_TOKEN_RETRIES + 1
    assert result.finish_reason in FINISH_REASONS


def test_eos_before_min_len_retry_exhausted():
    """采样器恒产 EOS（但未达 min_len）→ 同样计入重试，耗尽后 invalid_retry。"""
    model = _make_model(sampling=_sampler_always(VOCAB))  # always EOS
    result = model.decode(**_decode_inputs(text_len=3),
                          max_token_text_ratio=20, min_token_text_ratio=2)
    assert result.finish_reason == "invalid_token_retry_exhausted"
    assert result.num_valid_speech_tokens == 0


# ============================================================
# E. 采样器异常 → sampler_error
# ============================================================


def test_sampler_error_caught():
    """采样器抛异常 → finish_reason=sampler_error（不向上传播）。"""
    def _boom(scores, decoded, s):
        raise RuntimeError("sampler exploded")
    model = _make_model(sampling=_boom)
    result = model.decode(**_decode_inputs(text_len=3))
    assert result.finish_reason == "sampler_error"
    assert result.finish_reason in FINISH_REASONS


# ============================================================
# F. 互斥与稳定性
# ============================================================


def test_finish_reasons_are_mutually_exclusive_per_run():
    """同一 decode 结果只有一个 finish_reason；枚举完整。"""
    assert FINISH_REASONS == {
        "eos", "max_len_reached", "invalid_token_retry_exhausted",
        "sampler_error", "input_rejected",
    }


# ============================================================
# G. 生成行校验：非 eos 不得带 wav_path（schema §2）
# ============================================================


def test_eos_generation_row_validates_with_wav_path():
    eos = VOCAB
    model = _make_model(sampling=_sampler_eos_after(min_len=4, eos=eos))
    result = model.decode(**_decode_inputs(text_len=2))
    assert result.finish_reason == "eos"
    row = _generation_row(result, wav_path="out/u1.wav")
    validate_generation_row(row)     # must not raise


def test_non_eos_generation_row_validates_without_wav_path():
    model = _make_model(sampling=_sampler_always(2))
    result = model.decode(**_decode_inputs(text_len=3))
    assert result.finish_reason == "max_len_reached"
    row = _generation_row(result, wav_path=None)
    validate_generation_row(row)     # must not raise


def test_non_eos_generation_row_with_wav_path_rejected():
    """非 eos 携 wav_path → validate_generation_row 必须拒。"""
    model = _make_model(sampling=_sampler_always(2))
    result = model.decode(**_decode_inputs(text_len=3))
    assert result.finish_reason == "max_len_reached"
    row = _generation_row(result, wav_path="out/u1.wav")
    with pytest.raises(ValueError, match="must not carry a formal wav_path"):
        validate_generation_row(row)


def test_input_rejected_generation_row_validates_without_wav_path():
    model = _make_model()
    result = model.decode(**_decode_inputs(text_len=100),
                          max_token_text_ratio=20, min_token_text_ratio=2,
                          max_len_hard_cap=150)
    assert result.finish_reason == "input_rejected"
    row = _generation_row(result, wav_path=None)
    validate_generation_row(row)


# ============================================================
# H. 反转语义锁：v1 死长度字段已从活跃代码删除（ADR-0020）
# ============================================================


def test_v1_dead_length_patterns_removed_from_active_code():
    """历史 ``max_len=200`` 硬编码 bug 与源码哈希锁已从活跃代码删除。

    ADR-0020 禁源码哈希标定文件（删 md5/sha256 基线锁），改为断言反模式不存在。
    活跃长度合同由 ``DEFAULT_MAX_LEN_HARD_CAP`` + ratio 推导（见本文件 length 合同
    测试）。v1 基线由 git 锚点 ``9c6d84b`` 保证。
    """
    _repo_root = Path(__file__).resolve().parent.parent
    llm_src = (_repo_root / "cosyvoice" / "llm" / "llm_emotion.py").read_text(
        encoding="utf-8"
    )
    # v1 硬编码 max_len=200 必须删除（活跃代码用 DEFAULT_MAX_LEN_HARD_CAP 推导）
    assert "max_len = 200" not in llm_src
    config_text = (_repo_root / "conf" / "emo_film.yaml").read_text(encoding="utf-8")
    assert "max_len_hard_cap" in config_text
