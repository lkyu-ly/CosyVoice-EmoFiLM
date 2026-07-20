"""FiLMLayer + EmotionEncoder 单测。"""
import torch
from cosyvoice.llm.emo_film import EmotionEncoder, FiLMLayer


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
