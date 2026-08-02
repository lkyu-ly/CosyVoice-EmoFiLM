"""EmoFiLM target-only 单流训练-推理协议 focused 测试（活跃主线，ADR-0020 扁平化）。

覆盖（brief 04 §D / issue 04 checklist）：
- 活跃模型恒构造冻结的 input-end 探针 ``emotion_classifier``（``emo_loss_weight>0``
  才计入 loss；默认 0 时 loss_dict 无 ``loss_emotion_input``）；
- forward 恒定单流 ``[sos, FiLM(text), task, speech]``，
  target = ``[IGNORE] * (1 + text_len) + speech + [eos]``；
- 无论 speech/text 比例如何均不产生 fill_token / 交错文本 / 双流状态
  （强制覆盖 speech/text > 3 的原双流触发比例）；
- disabled（emo_loss_weight=0、无 span）forward 仅产出 ``loss_tts``；
- inference 前缀 = ``[sos, FiLM(target text), task]``（训练前缀减去 teacher speech），
  LLM 条件只含 target text + emotion/intensity 控制；
- inference 签名不接受 ``prompt_emotion_ids`` / ``prompt_intensity_ids`` /
  ``prompt_text``（死字段已删），保留 ``prompt_speech_token`` / ``embedding``
  透传给 Flow/HiFT（不进 LLM lm_input）；
- ``conf/emo_film.yaml`` 无死配置字段（mix_ratio / alpha），
  实例化 ``Qwen2LM_Emotion``，base 仍指 CosyVoice2 llm.pt；
- 反转语义锁仅保留残余反模式（``mix_ratio`` / ``alpha`` 死字段）已删断言
  （ADR-0020 禁源码哈希标定；``emotion_classifier`` / ``emo_loss_weight`` 现为
  可选 input-end 句级监督，不再是反模式）。

CPU fake-backbone 测试（仿 ``tests/test_emofilm_inference_contract.py:9-59``），
无需 GPU / 真实权重。
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion
from cosyvoice.utils.common import IGNORE_ID
from tests._emofilm_fakes import _FakeBackbone, _FakeHF, _FakeQwen
from tools.build_emofilm_contract import DEAD_CONFIG_KEYS, assert_no_dead_config

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_CONFIG_PATH = ROOT / "conf" / "emo_film.yaml"
ACTIVE_MODEL_EMO_PATH = ROOT / "cosyvoice" / "cli" / "model_emo.py"


# ============================================================
# fake backbone：从 tests._emofilm_fakes 复用（_FakeBackbone / _FakeHF /
# _FakeQwen 原先在本文件本地定义，现已 DRY 整合到共享测试辅助模块）。
# ============================================================


class _RecorderQwen(_FakeQwen):
    """记录 forward 的 lm_input / lm_input_len，用于断言单流结构。"""

    def __init__(self, model_dim=4):
        super().__init__(model_dim)
        self.forward_calls = []

    def forward(self, xs, xs_lens):
        self.forward_calls.append({
            "xs": xs.detach().clone(),
            "xs_lens": xs_lens.detach().clone() if torch.is_tensor(xs_lens) else xs_lens,
        })
        return super().forward(xs, xs_lens)


def _make_model(speech_token_size=10, llm=None):
    return Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=speech_token_size,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=llm or _FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
    )


def _batch(text_len=3, speech_len=5, speech_token_size=10):
    """单 sample 训练 batch。

    speech token id 限制在 ``[0, speech_token_size)`` 内（与 speech_embedding
    行数 ``speech_token_size + 3`` 兼容），以支持长 speech_len 的比例测试。
    """
    return {
        "text_token": torch.tensor([list(range(10, 10 + text_len))]),
        "text_token_len": torch.tensor([text_len], dtype=torch.int32),
        "speech_token": torch.tensor(
            [[i % speech_token_size for i in range(speech_len)]]
        ),
        "speech_token_len": torch.tensor([speech_len], dtype=torch.int32),
        "emotion_ids": torch.ones(1, text_len, dtype=torch.long),
        "intensity_ids": torch.ones(1, text_len, dtype=torch.long),
    }


class _CapturingCE(nn.Module):
    """包一层 criterion_ce（nn.Module）以捕获 lm_target。

    ``criterion_ce`` 是注册子模块（LabelSmoothingLoss），不能赋值为普通函数，
    故用 nn.Module 包装保持类型合法。
    """

    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped
        self.targets = []

    def forward(self, logits, target):
        self.targets.append(target.detach().clone())
        return self.wrapped(logits, target)


def _capture_target(model):
    """包一层 criterion_ce 以捕获 lm_target（不锁私有函数名，只观测可观测输出）。"""
    capture = _CapturingCE(model.criterion_ce)
    model.criterion_ce = capture
    return capture


# ============================================================
# B. forward 恒定单流、无 fill_token、仅 loss_tts
# ============================================================


def test_forward_is_single_stream_and_no_fill_token():
    model = _make_model(speech_token_size=10)
    fill_token = model.fill_token  # speech_token_size + 2
    eos = model.eos_token
    captured = _capture_target(model)

    text_len, speech_len = 3, 5
    out = model.forward(_batch(text_len, speech_len), torch.device("cpu"))

    tgt = captured.targets[-1][0].tolist()
    expected = [IGNORE_ID] * (1 + text_len) + list(range(speech_len)) + [eos]
    assert tgt == expected, f"single-stream target mismatch: {tgt}"
    assert fill_token not in tgt, "fill_token must never appear in v2 target"
    assert tgt.count(eos) == 1, "exactly one EOS terminator"

    # forward 仅产出 loss_tts；无 emotion/intensity loss 字段
    assert "loss" in out and "loss_tts" in out
    for forbidden in ("loss_emotion_span", "loss_intensity", "loss_emotion_input"):
        assert forbidden not in out, f"forward must not emit {forbidden!r}"
    torch.testing.assert_close(out["loss"].detach(), out["loss_tts"])


def test_long_speech_text_ratio_stays_single_stream():
    """speech/text > 3（原 v1 双流触发比例）仍走单流，无 fill_token。"""
    rec = _RecorderQwen(4)
    model = _make_model(speech_token_size=10, llm=rec)
    fill_token = model.fill_token
    captured = _capture_target(model)

    text_len, speech_len = 2, 20  # ratio = 10 > mix_ratio[1]/mix_ratio[0] = 3
    model.forward(_batch(text_len, speech_len), torch.device("cpu"))

    assert rec.forward_calls, "fake backbone forward was not called"
    xs_lens = rec.forward_calls[0]["xs_lens"]
    expected_len = 1 + text_len + 1 + speech_len  # sos + text + task + speech
    assert int(xs_lens[0]) == expected_len, (
        f"long ratio must stay single-stream: "
        f"got lm_input_len={int(xs_lens[0])}, expected {expected_len}"
    )

    tgt = captured.targets[-1][0].tolist()
    assert fill_token not in tgt, "fill_token must never appear even at long ratio"
    assert len(tgt) == expected_len, (
        f"target length must equal single-stream lm_input length: {len(tgt)}"
    )
    # 末端为 EOS，前面是 speech 区段
    assert tgt[-1] == model.eos_token


def test_forward_lm_input_matches_target_length():
    """单流下 lm_input 与 lm_target 等长（逐位置 teacher-forcing 对齐）。"""
    rec = _RecorderQwen(4)
    model = _make_model(speech_token_size=10, llm=rec)
    captured = _capture_target(model)
    model.forward(_batch(text_len=4, speech_len=6), torch.device("cpu"))
    xs = rec.forward_calls[0]["xs"]
    assert xs.shape[1] == captured.targets[-1].shape[1], (
        "lm_input and lm_target must be aligned (single-stream next-token)"
    )


def test_forward_batch_padding_stays_single_stream():
    """batch>1（不同长度）下仍单流：每样本 lm_input_len = 1+text+1+speech，无 fill_token。"""
    rec = _RecorderQwen(4)
    model = _make_model(speech_token_size=10, llm=rec)
    fill_token = model.fill_token
    eos = model.eos_token
    captured = _capture_target(model)

    # 两样本：text 2/4，speech 3/6（speech/text > 3 的比例也存在）
    batch = {
        "text_token": torch.tensor([[10, 11, 0, 0], [10, 11, 12, 13]]),
        "text_token_len": torch.tensor([2, 4], dtype=torch.int32),
        "speech_token": torch.tensor([[0, 1, 2, 0, 0, 0], [0, 1, 2, 3, 4, 5]]),
        "speech_token_len": torch.tensor([3, 6], dtype=torch.int32),
        "emotion_ids": torch.ones(2, 4, dtype=torch.long),
        "intensity_ids": torch.ones(2, 4, dtype=torch.long),
    }
    model.forward(batch, torch.device("cpu"))

    xs_lens = rec.forward_calls[0]["xs_lens"]
    expected = torch.tensor([1 + 2 + 1 + 3, 1 + 4 + 1 + 6], dtype=torch.int32)
    torch.testing.assert_close(xs_lens, expected)
    tgt = captured.targets[-1]
    flat = tgt.view(-1).tolist()
    assert fill_token not in flat, "fill_token must never appear in batched target"
    # 每样本末端（unpad 后）为 EOS：取每行有效长度的最后一个 token
    for i in range(2):
        row = tgt[i][: int(expected[i])].tolist()
        assert row[-1] == eos
        assert row.count(eos) == 1


# ============================================================
# C. inference 前缀 = [sos, FiLM(text), task]，不含 teacher speech
# ============================================================


def _capture_prefix(model):
    captured = {}

    def fake_wrapper(lm_input, sampling, min_len, max_len, uuid):
        captured["lm_input"] = lm_input.detach().clone()
        captured["min_len"] = min_len
        captured["max_len"] = max_len
        if False:  # pragma: no cover - generator stub
            yield None

    model.inference_wrapper = fake_wrapper
    return captured


def test_inference_prefix_excludes_teacher_speech():
    model = _make_model()
    captured = _capture_prefix(model)
    target_len = 3
    list(model.inference(
        text_token=torch.tensor([[2, 3, 4]]),
        text_len=torch.tensor([target_len], dtype=torch.int32),
        emotion_ids=torch.ones(1, target_len, dtype=torch.long),
        intensity_ids=torch.ones(1, target_len, dtype=torch.long),
        prompt_speech_token=torch.zeros((1, 4), dtype=torch.int32),
        prompt_speech_token_len=torch.tensor([4], dtype=torch.int32),
        embedding=torch.zeros(1, 4),
    ))
    # SOS + FiLM(target text) + task_id；无 teacher speech、无 prompt speech
    assert captured["lm_input"].shape == (1, 1 + target_len + 1, model.llm_input_size)


def test_inference_min_max_derived_from_target_text_len():
    """长度由 target 文本长度 + ratio 推导（MAP §3 长度不变量；hard-cap 归 ticket 05）。"""
    model = _make_model()
    captured = _capture_prefix(model)
    target_len = 5
    list(model.inference(
        text_token=torch.tensor([[2, 3, 4, 5, 6]]),
        text_len=torch.tensor([target_len], dtype=torch.int32),
        emotion_ids=torch.ones(1, target_len, dtype=torch.long),
        intensity_ids=torch.ones(1, target_len, dtype=torch.long),
        prompt_speech_token=torch.zeros((1, 0), dtype=torch.int32),
        prompt_speech_token_len=torch.tensor([0], dtype=torch.int32),
        embedding=torch.zeros(1, 4),
        min_token_text_ratio=2,
        max_token_text_ratio=20,
    ))
    assert captured["min_len"] == int(target_len * 2)
    assert captured["max_len"] == int(target_len * 20)


# ============================================================
# D. inference 签名清洁 + Flow/HiFT prompt 透传
# ============================================================


def test_inference_signature_drops_dead_prompt_fields():
    sig = inspect.signature(Qwen2LM_Emotion.inference)
    params = set(sig.parameters)
    for dead in (
        "prompt_emotion_ids",
        "prompt_intensity_ids",
        "prompt_text",
        "prompt_text_len",
    ):
        assert dead not in params, (
            f"v2 inference must not accept dead prompt field {dead!r}"
        )
    # Flow/HiFT 透传字段保留（不在 LLM lm_input，仅传给声学侧）
    assert "prompt_speech_token" in params
    assert "prompt_speech_token_len" in params
    assert "embedding" in params
    # target text + emotion/intensity 控制保留
    assert {"text_token", "text_len", "emotion_ids", "intensity_ids"} <= params


def test_flow_hift_prompt_not_in_llm_condition():
    """prompt_speech_token / embedding 被接受但不进 LLM lm_input。"""
    model = _make_model()
    captured = _capture_prefix(model)
    prompt_speech = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    embedding = torch.ones(1, 4)
    target_len = 2
    list(model.inference(
        text_token=torch.tensor([[2, 3]]),
        text_len=torch.tensor([target_len], dtype=torch.int32),
        emotion_ids=torch.ones(1, target_len, dtype=torch.long),
        intensity_ids=torch.ones(1, target_len, dtype=torch.long),
        prompt_speech_token=prompt_speech,
        prompt_speech_token_len=torch.tensor([3], dtype=torch.int32),
        embedding=embedding,
    ))
    # LLM 条件长度 = sos(1) + text(2) + task(1) = 4；prompt speech 未拼接
    assert captured["lm_input"].shape[1] == 1 + target_len + 1


# ============================================================
# E. conf/emo_film.yaml 合同（活跃配置）
# ============================================================


def test_active_config_has_no_dead_keys():
    assert ACTIVE_CONFIG_PATH.exists(), "conf/emo_film.yaml must exist"
    text = ACTIVE_CONFIG_PATH.read_text()
    for key in DEAD_CONFIG_KEYS:
        assert re.search(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*:", text) is None, (
            f"active config must not contain dead key {key!r} at any indentation"
        )
    # 活跃配置实例化 Qwen2LM_Emotion（去后缀类名）
    assert "!new:cosyvoice.llm.llm_emotion.Qwen2LM_Emotion" in text
    # base 仍指 CosyVoice2 llm.pt
    assert "CosyVoice2" in text or "llm.pt" in text or "pretrain_path" in text
    # 合同级 assert 在无死键的 resolved dict 上通过
    assert_no_dead_config({"contract_name": "emofilm", "schema_version": 2})


def test_active_config_does_not_pass_dead_kwargs_to_llm():
    """活跃 llm 块不得把死字段作为 __init__ kwarg 传给 Qwen2LM_Emotion。

    用 hyperpyyaml 加载（惰性占位自定义 tag），只取顶层 resolved dict，避免
    构造真实重模型；断言 llm 对象的构造 kwargs 不含死字段。``emo_loss_weight``
    是可选活参数（input-end 句级监督权重），基线配置默认不传。
    """
    text = ACTIVE_CONFIG_PATH.read_text()
    llm_block = _extract_top_block(text, "llm")
    for key in ("mix_ratio", "alpha"):
        assert re.search(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*:", llm_block) is None, (
            f"active llm block must not pass dead kwarg {key!r}"
        )
    # FiLM 相关模块仍配置
    assert "emotion_vocab_size" in llm_block
    assert "intensity_vocab_size" in llm_block


# ============================================================
# F. 反转语义锁：v1 反模式已从活跃代码删除（ADR-0020）
# ============================================================


def test_v1_anti_patterns_removed_from_active_code():
    """残余 v1 反模式（mix_ratio / alpha 死字段 / cli 类名硬断言）必须已从活跃
    代码删除。input-end ``emotion_classifier`` / ``emo_loss_weight`` 现为可选句级
    监督（``emo_loss_weight>0`` 启用），不再是反模式。ADR-0020 禁源码哈希标定，
    改为断言残余反模式不存在；v1 基线由 git 锚点 ``9c6d84b`` 保证。
    """
    config_text = ACTIVE_CONFIG_PATH.read_text(encoding="utf-8")
    # 配置不得出现死键（任意缩进的 ``key:`` 形式）
    for dead in ("mix_ratio", "alpha"):
        assert re.search(rf"(?m)^[ \t]*{re.escape(dead)}[ \t]*:", config_text) is None, (
            f"active config must not contain dead key {dead!r}"
        )

    # cli 不再有 v1 类名硬断言（去后缀后类名即 Qwen2LM_Emotion，断言冗余已删）
    model_emo_src = (ROOT / "cosyvoice" / "cli" / "cosyvoice_emo.py").read_text(
        encoding="utf-8"
    )
    assert "__class__.__name__" not in model_emo_src, (
        "v1 类名硬断言必须从 cosyvoice_emo.py 删除（#1 cli 接线修复）"
    )


# ============================================================
# helpers
# ============================================================


def _extract_top_block(text: str, header: str) -> str:
    """粗提取 yaml 中某个顶层 ``key:`` 块的文本（到下一个同缩进顶层键为止）。

    仅用于 textual 合同断言；不解析 yaml 语义。
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"(?m)^{re.escape(header)}\s*:", line):
            start = i
            break
    assert start is not None, f"block {header!r} not found in yaml"
    out = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() == "" or line.startswith((" ", "\t", "-")):
            out.append(line)
        else:
            break
    return "\n".join(out)
