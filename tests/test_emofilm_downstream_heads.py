"""Ticket 06 — EmoFiLM v2 下游 speech-token 监督任务头 focused 测试（CPU fake）。

核心：**反捷径结构证明**（brief 06 §C / issue 06 checklist）：

  (a) 任务头输入 feature 仅来自 ``lm_output`` 按 span speech-token 区间的
      masked-mean 池化；**不**来自 ``modulated_text_emb`` / ``emotion_ids`` /
      ``intensity_ids`` / loss target。
  (b) 改变 loss **target**（emotion soft dist / arousal）**不**改变任务头输入
      feature tensor（同一 forward 两次，仅换 target → feature bit-identical）。
  (c) 无代码路径将 ``modulated_text_emb`` / 控制 ID 喂入任务头：
      ``_pool_span_features`` 源码不含这些标识符（结构保证）；动态上，改变
      emotion/intensity 控制 ID（→FiLM→text emb）不改变 head feature（identity
      fake backbone 下 speech 区段与控制 ID 无直接连接）。

覆盖（brief 06 DoD）：
  - span masked-mean 池化正确（合成 lm_output + 已知区间 → 期望 feature）；
  - emotion/intensity mask 独立（emotion_mask=False 不贡献 emotion loss；
    intensity_mask=False 不贡献 intensity loss；ESD-style span intensity loss=0）；
  - soft distribution loss + hard CE（one-hot 特例）+ continuous arousal MSE；
  - 无效 span（03 的 valid=False）不贡献 loss；
  - 总 loss = tts + w_e·emotion + w_i·intensity（数值可校验）；
  - 活跃模型恒构造冻结 ``emotion_classifier``（input-end 句级监督探针，
    ``emo_loss_weight>0`` 才计入 loss）+ 可训练 ``emotion_head``/``arousal_head``
    （随机初始化 Linear，5 类 / 标量回归）；
  - base loader 允许 emotion_head/arousal_head（+ FiLM + emotion_classifier）
    缺失于 base ckpt；trained loader 仅容忍 emotion_classifier 缺失
    （旧 disabled ckpt），仍严格拒绝任务头缺失；
  - v1 基线锚 git ``9c6d84b``（ADR-0020 扁平化后不再用源码 sha256 锁）。

CPU fake-backbone 测试（仿 ``tests/test_emofilm_inference_contract.py:9-59``）。
无需 GPU / 真实权重。
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion
from cosyvoice.utils.emo_checkpoint import (
    ALLOWED_MISSING_PREFIXES,
    load_base_state,
    load_trained_state,
)
from cosyvoice.utils.common import IGNORE_ID
from tests._emofilm_fakes import _FakeBackbone, _FakeHF, _FakeQwen

ROOT = Path(__file__).resolve().parent.parent
ACTIVE_CONFIG_PATH = ROOT / "conf" / "emo_film.yaml"


# ============================================================
# fake backbone：从 tests._emofilm_fakes 复用（_FakeBackbone / _FakeHF /
# _FakeQwen 原先在本文件本地定义，现已 DRY 整合到共享测试辅助模块）。
# ============================================================


class _RecorderQwen(_FakeQwen):
    """记录 backbone 产出的 ``lm_output``，用于反捷径断言（feature 源自 lm_output）。"""

    def __init__(self, model_dim=4):
        super().__init__(model_dim)
        self.lm_outputs = []

    def forward(self, xs, xs_lens):
        self.lm_outputs.append(xs.detach().clone())
        return super().forward(xs, xs_lens)


def _make_model(speech_token_size=10, llm=None, ew=1.0, iw=1.0, downstream_supervision="disabled"):
    return Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=speech_token_size,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=llm or _FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emotion_head_weight=ew,
        intensity_head_weight=iw,
        downstream_supervision=downstream_supervision,
    )


def _base_batch(text_len=3, speech_len=6, speech_token_size=10, B=1):
    """无 span 的基础训练 batch（与 04 协议测试同构）。"""
    return {
        "text_token": torch.tensor([[10 + i for i in range(text_len)]] * B),
        "text_token_len": torch.tensor([text_len] * B, dtype=torch.int32),
        "speech_token": torch.tensor(
            [[i % speech_token_size for i in range(speech_len)]] * B
        ),
        "speech_token_len": torch.tensor([speech_len] * B, dtype=torch.int32),
        "emotion_ids": torch.ones(B, text_len, dtype=torch.long),
        "intensity_ids": torch.ones(B, text_len, dtype=torch.long),
    }


_SPAN_KEYS = (
    "span_mask",
    "span_valid",
    "span_tok_start",
    "span_tok_end",
    "span_emotion_mask",
    "span_intensity_mask",
    "span_emotion_soft_dist",
    "span_arousal",
    "span_supervision_weight",
    "span_control_emotion_id",
    "span_control_intensity_id",
)


def _add_one_span(
    batch,
    tok_start,
    tok_end,
    *,
    emotion_mask=True,
    intensity_mask=True,
    soft_dist=None,
    arousal=0.5,
    valid=True,
    supervision_weight=1.0,
    control_emotion_id=2,
    control_intensity_id=2,
):
    """向 batch 注入单样本单 span 张量（字段名/形状与 03 ``collate_aligned_spans`` 对齐）。"""
    if soft_dist is None:
        soft_dist = [0.0, 1.0, 0.0, 0.0, 0.0]
    batch.update(
        {
            "span_mask": torch.tensor([[True]]),
            "span_valid": torch.tensor([[valid]]),
            "span_tok_start": torch.tensor([[tok_start]]),
            "span_tok_end": torch.tensor([[tok_end]]),
            "span_emotion_mask": torch.tensor([[emotion_mask]]),
            "span_intensity_mask": torch.tensor([[intensity_mask]]),
            "span_emotion_soft_dist": torch.tensor(
                [[list(soft_dist)]], dtype=torch.float32
            ),
            "span_arousal": torch.tensor([[float(arousal)]], dtype=torch.float32),
            "span_supervision_weight": torch.tensor(
                [[float(supervision_weight)]], dtype=torch.float32
            ),
            "span_control_emotion_id": torch.tensor([[control_emotion_id]]),
            "span_control_intensity_id": torch.tensor([[control_intensity_id]]),
        }
    )
    return batch


class _RecordingHead(nn.Module):
    """包一层任务头以捕获其输入张量（反捷径断言用）。"""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return self.inner(x)


def _record_heads(model):
    emo_rec = _RecordingHead(model.emotion_head)
    aro_rec = _RecordingHead(model.arousal_head)
    model.emotion_head = emo_rec
    model.arousal_head = aro_rec
    return emo_rec, aro_rec


def _expected_feature(lm_output, text_token_len, tok_start, tok_end):
    """手工计算 masked-mean：speech 区段起点 = 1 + text_len（IGNORE 前缀长度）。"""
    speech_start = 1 + int(text_token_len[0].item())
    return lm_output[0, speech_start + tok_start : speech_start + tok_end].mean(dim=0)


# ============================================================
# A. 模型结构：可训练下游 heads + 冻结 input-end 探针
# ============================================================


def test_model_has_downstream_heads():
    model = _make_model()
    # 新增下游任务头
    assert isinstance(model.emotion_head, nn.Linear), "emotion_head must be nn.Linear"
    assert model.emotion_head.out_features == 5, "emotion_head outputs 5 emotion classes"
    assert model.emotion_head.in_features == model.llm_output_size
    assert isinstance(model.arousal_head, nn.Linear), "arousal_head must be nn.Linear"
    assert model.arousal_head.out_features == 1, "arousal_head outputs scalar regression"
    # 可训练（非冻结）
    assert all(p.requires_grad for p in model.emotion_head.parameters())
    assert all(p.requires_grad for p in model.arousal_head.parameters())
    # FiLM 保留
    assert hasattr(model, "emotion_encoder") and hasattr(model, "emotion_adapter")


def test_init_accepts_head_weights_distinct_from_v1():
    sig = inspect.signature(Qwen2LM_Emotion.__init__)
    params = set(sig.parameters)
    assert "emotion_head_weight" in params
    assert "intensity_head_weight" in params
    model = _make_model(ew=0.7, iw=0.3)
    assert model.emotion_head_weight == 0.7
    assert model.intensity_head_weight == 0.3


# ============================================================
# B. 池化正确性（_pool_span_features 纯函数）
# ============================================================


def test_pool_span_features_is_masked_mean_over_speech_region():
    """feature = mean(lm_output[:, speech_start+tok_start : speech_start+tok_end, :])。

    speech_start = 1 + text_len（v2 target 的 IGNORE 前缀长度，即 lm_target!=IGNORE
    的首列）。EOS 列被 tok_end<=speech_len 的 exclusive 切片排除。
    """
    model = _make_model()
    D = 3
    # lm_input 布局 (text_len=2, speech_len=2): [SOS, t0, t1, task, s0, s1, EOS]
    # target IGNORE 前缀 = 1+text_len = 3 → speech 区段从 col 3 起
    lm_output = torch.tensor(
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [3, 3, 3], [9, 9, 9],
          [7, 7, 7]]],
        dtype=torch.float32,
    )  # (1, 7, 3)
    speech_token_mask = torch.tensor(
        [[False, False, False, True, True, True, True]]
    )  # cols 3,4,5 = speech; col6 = EOS
    text_token_len = torch.tensor([2], dtype=torch.int32)
    feature = model._pool_span_features(
        lm_output,
        speech_token_mask,
        text_token_len,
        span_tok_start=torch.tensor([[0]]),
        span_tok_end=torch.tensor([[2]]),  # abs [3,5) → speech s0,s1
        span_mask=torch.tensor([[True]]),
        span_valid=torch.tensor([[True]]),
    )
    # mean(col3=[1,1,1], col4=[3,3,3]) = [2,2,2]；EOS(col6) 排除
    torch.testing.assert_close(feature, torch.tensor([[[2.0, 2.0, 2.0]]]))


def test_pool_span_features_excludes_ignore_and_padding_columns():
    """池化仅取 speech_token_mask=True 的列，IGNORE/padding 列不进 feature。"""
    model = _make_model()
    lm_output = torch.tensor(
        [[[0, 0], [10, 10], [1, 1], [3, 3], [9, 9]]], dtype=torch.float32
    )
    # text_len=1 → speech_start=2；但 col1 被标 IGNORE（模拟 padding/文本）
    speech_token_mask = torch.tensor([[False, False, True, True, True]])
    feature = model._pool_span_features(
        lm_output,
        speech_token_mask,
        text_token_len=torch.tensor([1], dtype=torch.int32),
        span_tok_start=torch.tensor([[0]]),
        span_tok_end=torch.tensor([[3]]),  # abs [2,5)
        span_mask=torch.tensor([[True]]),
        span_valid=torch.tensor([[True]]),
    )
    # 仅 col2,3,4 非 IGNORE → mean([1,1],[3,3],[9,9]) = [13/3, 13/3]
    torch.testing.assert_close(feature[0, 0], torch.tensor([13.0 / 3, 13.0 / 3]))


def test_pool_span_features_invalid_span_returns_zero_feature():
    """valid=False 的 span 不贡献（feature 为零向量，且 loss mask 门控）。"""
    model = _make_model()
    lm_output = torch.randn(1, 4, 3)
    feature = model._pool_span_features(
        lm_output,
        torch.tensor([[True, True, True, True]]),
        text_token_len=torch.tensor([0], dtype=torch.int32),
        span_tok_start=torch.tensor([[0]]),
        span_tok_end=torch.tensor([[2]]),
        span_mask=torch.tensor([[True]]),
        span_valid=torch.tensor([[False]]),
    )
    assert torch.count_nonzero(feature).item() == 0


# ============================================================
# C. 反捷径结构证明（核心）
# ============================================================


def test_anti_shortcut_head_input_equals_pooled_lm_output():
    """(a) 任务头输入 == lm_output 按 span 区间 masked-mean 池化。"""
    rec = _RecorderQwen(4)
    model = _make_model(llm=rec)
    emo_rec, aro_rec = _record_heads(model)

    text_len, speech_len = 3, 6
    batch = _base_batch(text_len=text_len, speech_len=speech_len)
    _add_one_span(batch, tok_start=1, tok_end=4)  # abs [1+3+1, 1+3+4) = [5,8)
    model.forward(batch, torch.device("cpu"))

    assert emo_rec.inputs and aro_rec.inputs, "heads were not called"
    lm_output = rec.lm_outputs[-1]
    expected = _expected_feature(lm_output, batch["text_token_len"], 1, 4)

    torch.testing.assert_close(emo_rec.inputs[-1][0, 0], expected)
    torch.testing.assert_close(aro_rec.inputs[-1][0, 0], expected)


def test_anti_shortcut_changing_target_does_not_change_head_input():
    """(b) 同一 forward 两次，仅换监督 target → head 输入 feature bit-identical。"""
    rec = _RecorderQwen(4)
    model = _make_model(llm=rec, ew=1.0, iw=1.0)
    emo_rec, aro_rec = _record_heads(model)
    model.eval()

    text_len, speech_len = 2, 5
    # run A: arousal=0.2, emotion=ang (one-hot col0)
    batch_a = _base_batch(text_len=text_len, speech_len=speech_len)
    _add_one_span(
        batch_a, tok_start=0, tok_end=3,
        soft_dist=[1.0, 0, 0, 0, 0], arousal=0.2,
    )
    out_a = model.forward(batch_a, torch.device("cpu"))
    feat_a = emo_rec.inputs[-1].clone()

    # run B: same lm_output-determining inputs, swap ONLY targets
    batch_b = _base_batch(text_len=text_len, speech_len=speech_len)
    _add_one_span(
        batch_b, tok_start=0, tok_end=3,
        soft_dist=[0.0, 0, 0, 0, 1.0], arousal=0.9,  # different target
    )
    out_b = model.forward(batch_b, torch.device("cpu"))
    feat_b = emo_rec.inputs[-1].clone()

    torch.testing.assert_close(feat_a, feat_b)  # feature unchanged
    # 但 loss 随 target 改变（证明 target 确实进了 loss，而非 feature）
    assert not torch.allclose(out_a["loss_emotion_span"], out_b["loss_emotion_span"])
    assert not torch.allclose(out_a["loss_intensity"], out_b["loss_intensity"])


def test_anti_shortcut_changing_control_id_does_not_directly_change_head_input():
    """(c) 改 emotion/intensity 控制 ID（→FiLM→text emb）不直接改变 head feature。

    identity fake backbone 下 speech 区段 hidden 与控制 ID 无直接连接；
    若 head 直接接收控制 ID 或 modulated_text_emb，feature 必随 ID 改变。
    """
    model = _make_model(llm=_RecorderQwen(4), ew=1.0, iw=1.0)
    emo_rec, aro_rec = _record_heads(model)
    model.eval()

    text_len, speech_len = 2, 4
    # run 1: emotion=1, intensity=1
    batch1 = _base_batch(text_len=text_len, speech_len=speech_len)
    batch1["emotion_ids"] = torch.ones(1, text_len, dtype=torch.long)
    batch1["intensity_ids"] = torch.ones(1, text_len, dtype=torch.long)
    _add_one_span(batch1, tok_start=0, tok_end=2, control_emotion_id=1,
                  control_intensity_id=1, soft_dist=[1, 0, 0, 0, 0], arousal=0.5)
    model.forward(batch1, torch.device("cpu"))
    feat1 = emo_rec.inputs[-1].clone()

    # run 2: emotion=5, intensity=3（控制 ID 全变）
    batch2 = _base_batch(text_len=text_len, speech_len=speech_len)
    batch2["emotion_ids"] = torch.full((1, text_len), 5, dtype=torch.long)
    batch2["intensity_ids"] = torch.full((1, text_len), 3, dtype=torch.long)
    _add_one_span(batch2, tok_start=0, tok_end=2, control_emotion_id=5,
                  control_intensity_id=3, soft_dist=[1, 0, 0, 0, 0], arousal=0.5)
    model.forward(batch2, torch.device("cpu"))
    feat2 = emo_rec.inputs[-1].clone()

    # 控制 ID 改变不直接进 head → speech 区段 feature 不变
    torch.testing.assert_close(feat1, feat2)


def test_anti_shortcut_pool_function_source_has_no_forbidden_features():
    """(c 结构) ``_pool_span_features`` 代码（不含 docstring/comment）不得引用
    modulated_text_emb / 控制 ID / loss target —— 结构上保证 head 输入仅由
    lm_output + span 几何区间决定。用 AST 解析，只看真实代码标识符。
    """
    import ast
    import textwrap

    src = textwrap.dedent(inspect.getsource(Qwen2LM_Emotion._pool_span_features))
    tree = ast.parse(src)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.arg):
            used.add(node.arg)
    forbidden = {
        "modulated_text_emb",
        "emotion_ids",
        "intensity_ids",
        "emotion_features",
        "emotion_adapter",
        "emotion_encoder",
        "span_emotion_soft_dist",
        "span_arousal",
        "loss",
        "control_emotion_id",
        "control_intensity_id",
        "span_emotion_mask",
        "span_intensity_mask",
    }
    leaked = used & forbidden
    assert not leaked, (
        f"_pool_span_features references forbidden identifiers {sorted(leaked)} "
        "(anti-shortcut: head input is lm_output pooled over spans only)"
    )


def test_anti_shortcut_head_input_not_pooled_modulated_text_emb():
    """(c) head 输入 != modulated_text_emb 池化（证明源是 lm_output 不是 text emb）。

    identity backbone 下 lm_output 包含 [SOS, modulated_text_emb, task, speech_emb]；
    head 池化 speech 区段 → 应等于 speech_emb 池化，不等于 text emb 池化。
    """
    rec = _RecorderQwen(4)
    model = _make_model(llm=rec)
    emo_rec, _ = _record_heads(model)

    text_len, speech_len = 3, 4
    batch = _base_batch(text_len=text_len, speech_len=speech_len)
    _add_one_span(batch, tok_start=0, tok_end=2)
    model.forward(batch, torch.device("cpu"))

    lm_output = rec.lm_outputs[-1]
    # modulated_text_emb 区段 = cols [1, 1+text_len)
    text_pool = lm_output[0, 1 : 1 + text_len].mean(dim=0)
    head_in = emo_rec.inputs[-1][0, 0]
    # head 输入应 != text emb 池化（数值上明显不同）
    assert not torch.allclose(head_in, text_pool)


# ============================================================
# D. mask 独立 / 无效 span 不贡献
# ============================================================


def test_emotion_mask_false_contributes_no_emotion_loss():
    model = _make_model(ew=1.0, iw=0.0)
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(batch, tok_start=0, tok_end=2, emotion_mask=False, intensity_mask=False)
    out = model.forward(batch, torch.device("cpu"))
    assert float(out["loss_emotion_span"]) == 0.0
    assert float(out["loss_intensity"]) == 0.0


def test_intensity_mask_false_esd_contributes_no_intensity_loss():
    """ESD-style span: emotion_mask=True, intensity_mask=False → 仅 emotion loss。"""
    model = _make_model(ew=1.0, iw=1.0)
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(
        batch, tok_start=0, tok_end=2,
        emotion_mask=True, intensity_mask=False,
        soft_dist=[0, 1.0, 0, 0, 0], arousal=0.0,  # arousal 应被忽略
    )
    out = model.forward(batch, torch.device("cpu"))
    assert float(out["loss_intensity"]) == 0.0
    assert float(out["loss_emotion_span"]) > 0.0


def test_emotion_mask_independent_from_intensity_mask():
    """emotion_mask=False / intensity_mask=True → 仅 intensity loss。"""
    model = _make_model(ew=1.0, iw=1.0)
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(
        batch, tok_start=0, tok_end=2,
        emotion_mask=False, intensity_mask=True, arousal=0.7,
    )
    out = model.forward(batch, torch.device("cpu"))
    assert float(out["loss_emotion_span"]) == 0.0
    assert float(out["loss_intensity"]) > 0.0


def test_invalid_span_contributes_no_loss():
    """03 标 valid=False 的 span（零覆盖/越界）不贡献任何 loss。"""
    model = _make_model(ew=1.0, iw=1.0)
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(
        batch, tok_start=0, tok_end=2, valid=False,
        soft_dist=[1.0, 0, 0, 0, 0], arousal=0.9,
    )
    out = model.forward(batch, torch.device("cpu"))
    assert float(out["loss_emotion_span"]) == 0.0
    assert float(out["loss_intensity"]) == 0.0


# ============================================================
# E. loss 语义：soft CE / hard CE / continuous arousal MSE / total
# ============================================================


def test_soft_distribution_loss_matches_manual_soft_ce():
    model = _make_model(ew=1.0, iw=0.0)
    rec = _RecorderQwen(4)
    model.llm = rec
    emo_rec, _ = _record_heads(model)
    model.eval()

    soft = [0.2, 0.3, 0.1, 0.25, 0.15]
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(batch, tok_start=0, tok_end=2, soft_dist=soft, arousal=0.0)
    out = model.forward(batch, torch.device("cpu"))

    feature = emo_rec.inputs[-1]
    logits = model.emotion_head.inner(feature)  # 原始 head（未被 wrapper）
    log_probs = F.log_softmax(logits, dim=-1)
    expected = -(torch.tensor(soft) * log_probs[0, 0]).sum()
    torch.testing.assert_close(out["loss_emotion_span"], expected.detach())


def test_hard_ce_one_hot_is_special_case_of_soft_ce():
    """one-hot soft dist → soft CE 等价于标准 CE（覆盖 hard target 路径）。"""
    model = _make_model(ew=1.0, iw=0.0)
    rec = _RecorderQwen(4)
    model.llm = rec
    emo_rec, _ = _record_heads(model)
    model.eval()

    hard_label = 2  # one-hot at index 2
    soft = [0.0, 0.0, 1.0, 0.0, 0.0]
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(batch, tok_start=0, tok_end=2, soft_dist=soft)
    out = model.forward(batch, torch.device("cpu"))

    feature = emo_rec.inputs[-1]
    logits = model.emotion_head.inner(feature)[0, 0]
    manual_ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([hard_label]))
    torch.testing.assert_close(out["loss_emotion_span"], manual_ce.detach())


def test_continuous_arousal_mse_matches_manual():
    model = _make_model(ew=0.0, iw=1.0)
    rec = _RecorderQwen(4)
    model.llm = rec
    _, aro_rec = _record_heads(model)
    model.eval()

    arousal_tgt = 0.42
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(batch, tok_start=0, tok_end=2, arousal=arousal_tgt,
                  soft_dist=[1, 0, 0, 0, 0])
    out = model.forward(batch, torch.device("cpu"))

    feature = aro_rec.inputs[-1]
    pred = model.arousal_head.inner(feature)[0, 0, 0]
    expected = (pred - arousal_tgt) ** 2
    torch.testing.assert_close(out["loss_intensity"], expected.detach())


def test_total_loss_is_tts_plus_weighted_emotion_plus_weighted_intensity():
    model = _make_model(ew=0.5, iw=0.25)
    rec = _RecorderQwen(4)
    model.llm = rec
    emo_rec, aro_rec = _record_heads(model)
    model.eval()

    soft = [0.1, 0.2, 0.3, 0.4, 0.0]
    batch = _base_batch(text_len=2, speech_len=4)
    _add_one_span(batch, tok_start=0, tok_end=2, soft_dist=soft, arousal=0.6)
    out = model.forward(batch, torch.device("cpu"))

    expected = (
        out["loss_tts"]
        + 0.5 * out["loss_emotion_span"]
        + 0.25 * out["loss_intensity"]
    )
    torch.testing.assert_close(out["loss"].detach(), expected)


def test_no_spans_batch_returns_loss_tts_only():
    """无 span 的 batch（04 协议路径）仍只返回 loss_tts（向后兼容，不回归 04）。"""
    model = _make_model()
    batch = _base_batch(text_len=2, speech_len=4)
    out = model.forward(batch, torch.device("cpu"))
    assert "loss" in out and "loss_tts" in out
    for forbidden in ("loss_emotion_span", "loss_intensity", "emotion_logits"):
        assert forbidden not in out
    torch.testing.assert_close(out["loss"].detach(), out["loss_tts"])


# ============================================================
# F. base / trained checkpoint loader
# ============================================================


def test_base_loader_allows_missing_downstream_heads():
    model = _make_model()
    full = model.state_dict()
    # base ckpt：剥离新下游任务头 + FiLM（emotion_encoder/emotion_adapter）
    base = {
        k: v
        for k, v in full.items()
        if not k.startswith(ALLOWED_MISSING_PREFIXES)
    }
    # 允许这些缺失
    fresh = _make_model()
    load_base_state(fresh, base)
    # base ckpt 不得缺失 backbone（非 allowed-missing）→ 失败
    backbone_only = {k: v for k, v in base.items() if not k.startswith("llm_decoder")}
    with pytest.raises(RuntimeError):
        load_base_state(_make_model(), backbone_only)


def test_trained_loader_is_strict():
    model = _make_model()
    full = model.state_dict()
    load_trained_state(_make_model(), full)  # 完整 → OK
    # 缺失 head → strict 失败
    missing_head = {k: v for k, v in full.items() if not k.startswith("emotion_head.")}
    with pytest.raises(RuntimeError):
        load_trained_state(_make_model(), missing_head)
    # 多余键 → 失败
    extra = dict(full)
    extra["foreign.key"] = torch.zeros(2)
    with pytest.raises(RuntimeError):
        load_trained_state(_make_model(), extra)


# ============================================================
# G. 反转语义锁 + 活跃配置含下游任务头权重（非死字段）
# ============================================================


def test_active_config_carries_head_weights_not_dead():
    text = ACTIVE_CONFIG_PATH.read_text()
    assert "emotion_head_weight" in text
    assert "intensity_head_weight" in text
    # emo_film.yaml（disabled 基线 + 推理配置）不得含 emo_loss_weight：句级监督
    # 仅 sentlvl 实验配置启用，disabled 基线保持纯净（零回归）。
    for dead in ("emo_loss_weight",):
        # yaml 顶层或 llm 块缩进下都不得出现该字段
        assert f"{dead}:" not in text


# ============================================================
# H. input-end 句级监督（emo_loss_weight>0 可选路径）
# ============================================================


def test_input_end_loss_emotion_gradient_flows_to_film():
    """input-end 句级监督（emo_loss_weight>0）的核心不变量：

    - emotion_classifier 存在且冻结（requires_grad=False，不进 optimizer）。
    - forward 返回 loss_emotion_input（finite）；loss = loss_tts + emo_loss_weight·loss_emotion_input。
    - loss_emotion_input 的梯度经冻结分类器回流到 FiLM（emotion_encoder/emotion_adapter）。
    """
    # 启用 input-end 句级监督（直接构造，复用 _FakeQwen / _base_batch）
    model = Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=10,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
        downstream_supervision="disabled",
    )
    # 分类器存在且冻结
    assert hasattr(model, "emotion_classifier")
    assert hasattr(model, "criterion_emotion_cls")
    assert model.emo_loss_weight == 0.2
    assert all(not p.requires_grad for p in model.emotion_classifier.parameters())

    batch = _base_batch()
    out = model(batch, torch.device("cpu"))
    # loss_dict 含 loss_emotion_input（finite）
    assert "loss_emotion_input" in out and "loss_tts" in out and "loss" in out
    assert torch.isfinite(out["loss_emotion_input"]).item()
    # loss = loss_tts + 0.2 * loss_emotion_input（数值一致）
    assert torch.allclose(
        out["loss"], out["loss_tts"] + 0.2 * out["loss_emotion_input"], atol=1e-6
    )
    # backward 成功；冻结分类器不累积梯度（反捷径：分类器权重不更新）
    out["loss"].backward()
    assert model.emotion_classifier.weight.grad is None

    # 隔离验证：loss_emotion_input 对 FiLM 的梯度（不经 loss_tts，直接 autograd.grad）。
    # 冻结分类器不阻挡梯度——CE 对 emo_logits 的梯度经分类器固定权重的转置
    # 回流到 modulated_text_emb → emotion_adapter → emotion_encoder。
    # 注意：FiLMLayer 恒等初始化（projection.weight=0）使初始 modulated=text_emb、
    # 不依赖 emotion_features；故打破恒等初始化以验证完整梯度路径无断点（训练后
    # projection.weight 非零时 emotion_encoder 才开始接收 loss_emotion_input 梯度）。
    with torch.no_grad():
        model.emotion_adapter.projection.weight.normal_(0.0, 0.1)
    text_emb = model.llm.model.model.embed_tokens(batch["text_token"])
    emo_feats = model.emotion_encoder(batch["emotion_ids"], batch["intensity_ids"])
    modulated = model.emotion_adapter(text_emb, emo_feats)
    emo_logits = model.emotion_classifier(modulated)
    loss_emotion_input = model.criterion_emotion_cls(
        emo_logits.reshape(-1, emo_logits.size(-1)),
        batch["emotion_ids"].reshape(-1),
    )
    enc_params = list(model.emotion_encoder.parameters())
    ada_params = list(model.emotion_adapter.parameters())
    film_grads = torch.autograd.grad(
        loss_emotion_input, enc_params + ada_params, allow_unused=True
    )
    enc_grads = film_grads[: len(enc_params)]
    ada_grads = film_grads[len(enc_params) :]
    assert any(g is not None and g.abs().sum().item() > 0 for g in enc_grads), (
        "loss_emotion_input 梯度必须经冻结分类器回流 emotion_encoder（FiLM）"
    )
    assert any(g is not None and g.abs().sum().item() > 0 for g in ada_grads), (
        "loss_emotion_input 梯度必须经冻结分类器回流 emotion_adapter（FiLM）"
    )

# ============================================================
# I. 恒定拓扑 + checkpoint 双向兼容（P0 回归）
# ============================================================


def test_constant_topology_always_has_frozen_classifier():
    """恒构造冻结探针：disabled 模型也含 emotion_classifier（训练/推理/基线
    同一 state_dict 键集）；emo_loss_weight 恒为 float 属性。"""
    model = _make_model()
    assert model.emo_loss_weight == 0.0
    assert isinstance(model.emotion_classifier, nn.Linear)
    assert model.emotion_classifier.in_features == model.llm_input_size
    assert model.emotion_classifier.out_features == 6
    assert all(not p.requires_grad for p in model.emotion_classifier.parameters())
    keys = set(model.state_dict())
    assert "emotion_classifier.weight" in keys
    assert "emotion_classifier.bias" in keys


def test_base_load_allows_missing_classifier():
    """训练启动路径：sentlvl 模型（emo_loss_weight>0）加载 base llm.pt
    （无分类器键）必须成功（P0 回归）。"""
    model = Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=10,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
        downstream_supervision="disabled",
    )
    base = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith(ALLOWED_MISSING_PREFIXES)
    }
    load_base_state(model, base)


def test_trained_load_accepts_with_and_without_classifier():
    """推理路径：disabled 模型 strict 加载 sentlvl final.pt（含分类器）成功；
    旧 disabled ckpt（无分类器）也成功（训练期专用模块缺失容忍）。"""
    sentlvl = Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=10,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
        downstream_supervision="disabled",
    )
    infer = _make_model()  # disabled 推理拓扑
    load_trained_state(infer, dict(sentlvl.state_dict()))
    old_ckpt = {
        key: value.clone()
        for key, value in infer.state_dict().items()
        if not key.startswith("emotion_classifier.")
    }
    load_trained_state(infer, old_ckpt)


def test_trained_load_still_rejects_v1_missing_heads():
    """v1 防冒充守卫：trained 加载仍拒绝 emotion_head/arousal_head 缺失。"""
    model = _make_model()
    v1_like = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith(("emotion_head.", "arousal_head."))
    }
    with pytest.raises(RuntimeError, match="emotion_head"):
        load_trained_state(_make_model(), v1_like)


# ============================================================
# J. 两条监督路径可叠加 + loss 键名分离（P1 回归）
# ============================================================


def test_input_end_and_span_losses_coexist_with_distinct_keys():
    """span 分支不再提前 return：input-end loss 不被静默丢弃；键名分离
    （loss_emotion_span / loss_intensity / loss_emotion_input）。"""
    model = Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=10,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
        downstream_supervision="disabled",
    )
    batch = _add_one_span(
        _base_batch(text_len=2, speech_len=4), tok_start=0, tok_end=2
    )
    out = model.forward(batch, torch.device("cpu"))
    assert "loss_emotion_span" in out
    assert "loss_intensity" in out
    assert "loss_emotion_input" in out
    expected = (
        out["loss_tts"]
        + 1.0 * out["loss_emotion_span"]
        + 1.0 * out["loss_intensity"]
        + 0.2 * out["loss_emotion_input"]
    )
    torch.testing.assert_close(out["loss"].detach(), expected)


def test_zero_weight_gates_input_end_loss():
    """emo_loss_weight=0：分类器存在但不计入 loss，loss_dict 键集与基线一致。"""
    model = _make_model()
    out = model.forward(_base_batch(), torch.device("cpu"))
    assert set(out.keys()) == {"loss", "acc", "loss_tts"}
    torch.testing.assert_close(out["loss"].detach(), out["loss_tts"])
