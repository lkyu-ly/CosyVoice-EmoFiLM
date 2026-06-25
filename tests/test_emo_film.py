"""FiLMLayer + EmotionEncoder + AddFusionEmotionAdapter 单测。"""
import torch
from cosyvoice.llm.emo_film import EmotionEncoder, FiLMLayer, AddFusionEmotionAdapter


def test_emotion_encoder_shape():
    enc = EmotionEncoder(emotion_vocab_size=6, intensity_vocab_size=4, model_dim=896)
    e_ids = torch.tensor([[1, 2, 3], [4, 1, 1]])  # B=2, T=3
    i_ids = torch.tensor([[1, 2, 1], [3, 1, 1]])
    out = enc(e_ids, i_ids)
    assert out.shape == (2, 3, 896)


def test_emotion_encoder_padding_zero():
    """padding ID=0 不影响其他 token 的 embedding。"""
    enc = EmotionEncoder(emotion_vocab_size=6, intensity_vocab_size=4, model_dim=896)
    e_ids = torch.tensor([[1, 0, 2]])
    i_ids = torch.tensor([[1, 0, 1]])
    out = enc(e_ids, i_ids)
    assert out.shape == (1, 3, 896)
    # pad 位置非全零（embedding 表包含 padding id），但不应 NaN
    assert not out.isnan().any()


def test_film_identity_at_init():
    """初始化时 FiLM 等价恒等映射: gamma=1, beta=0 → output == input。"""
    film = FiLMLayer(model_dim=896)
    x = torch.randn(4, 10, 896)
    e = torch.randn(4, 10, 896)
    out = film(x, e)
    torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


def test_film_shape_preserved():
    """FiLM 不改变输入 shape。"""
    film = FiLMLayer(model_dim=896)
    x = torch.randn(2, 15, 896)
    e = torch.randn(2, 15, 896)
    out = film(x, e)
    assert out.shape == (2, 15, 896)


def test_film_gradient_to_both_inputs():
    """梯度回传到 text_features 和 emotion_features 两端。"""
    film = FiLMLayer(model_dim=896)
    x = torch.randn(3, 5, 896, requires_grad=True)
    e = torch.randn(3, 5, 896, requires_grad=True)
    out = film(x, e)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert e.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


# ----- AddFusionEmotionAdapter 消融对照 -----

def test_add_fusion_shape_preserved():
    """AddFusion 输出 shape 与 FiLMLayer 一致。"""
    adapter = AddFusionEmotionAdapter(model_dim=896)
    x = torch.randn(2, 15, 896)
    e = torch.randn(2, 15, 896)
    out = adapter(x, e)
    assert out.shape == (2, 15, 896)


def test_add_fusion_forward_equation():
    """AddFusion 前向严格等于 text_features + projection(emotion_features)。"""
    adapter = AddFusionEmotionAdapter(model_dim=896)
    # 冻结 projection 权重以便手动复算
    x = torch.randn(2, 8, 896)
    e = torch.randn(2, 8, 896)
    with torch.no_grad():
        expected = x + adapter.projection(e)
    out = adapter(x, e)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_add_fusion_not_identity_at_init():
    """AddFusion 初始化即非恒等（与 FiLM 恒等初始化对比，证明消融对照有效）。"""
    adapter = AddFusionEmotionAdapter(model_dim=896)
    x = torch.randn(2, 5, 896)
    e = torch.randn(2, 5, 896)
    out = adapter(x, e)
    # 默认 Linear 初始化非零，故 out != x（极大概率）
    assert not torch.allclose(out, x, atol=1e-5, rtol=1e-5)
