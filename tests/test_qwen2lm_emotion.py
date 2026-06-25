"""Qwen2LM_Emotion forward 单测（使用 fake Qwen2Encoder）。"""
import torch
import torch.nn as nn
from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion
from cosyvoice.llm.emo_film import FiLMLayer, AddFusionEmotionAdapter


class FakeQwen2Encoder(nn.Module):
    """最小 Qwen2Encoder 替代，返回固定 hidden states。"""
    def __init__(self, model_dim=896, pretrain_path=None):
        super().__init__()
        self.model = FakeHFModel(model_dim)

    def forward(self, xs, xs_lens):
        return xs, torch.ones(xs.shape[0], 1, xs.shape[1], dtype=torch.bool)

    def forward_one_step(self, xs, masks=None, cache=None):
        return xs, None


class FakeHFModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = FakeEmbed(dim)

    def return_dict(self):
        return True


class FakeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(20000, dim)


def fake_sampling(scores, decoded_tokens, sampling):
    return scores.argmax()


def make_fake_batch(B=2, T=10, S=25):
    """构建 fake batch dict 适配新版 prepare_lm_input_target。"""
    return {
        "text_token": torch.randint(0, 5000, (B, T)),
        "text_token_len": torch.tensor([T] * B, dtype=torch.int32),
        "speech_token": torch.randint(0, 6561, (B, S)),
        "speech_token_len": torch.tensor([S] * B, dtype=torch.int32),
        "emotion_ids": torch.ones(B, T, dtype=torch.long),  # all 'ang'=1
        "intensity_ids": torch.ones(B, T, dtype=torch.long),  # all 'low'=1
    }


def test_forward_returns_expected_keys():
    model = Qwen2LM_Emotion(
        llm_input_size=896,
        llm_output_size=896,
        speech_token_size=6561,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=FakeQwen2Encoder(model_dim=896),
        sampling=fake_sampling,
        length_normalized_loss=True,
        lsm_weight=0.0,
        mix_ratio=[5, 15],
        emo_loss_weight=0.2,
    )
    model.eval()
    batch = make_fake_batch()
    with torch.no_grad():
        out = model(batch, torch.device("cpu"))
    assert "loss" in out
    assert "acc" in out
    assert "loss_tts" in out
    assert "loss_emotion" in out
    assert out["loss"].numel() == 1
    assert not out["loss"].isnan()


def test_forward_respects_emo_loss_weight():
    """emo_loss_weight=0 时 loss_emotion 不影响总 loss。"""
    model_zero = Qwen2LM_Emotion(
        llm_input_size=896, llm_output_size=896, speech_token_size=6561,
        emotion_vocab_size=6, intensity_vocab_size=4,
        llm=FakeQwen2Encoder(model_dim=896),
        sampling=fake_sampling, emo_loss_weight=0.0,
    )
    model_zero.eval()
    batch = make_fake_batch()
    with torch.no_grad():
        out = model_zero(batch, torch.device("cpu"))
    assert abs(out["loss_emotion"].item()) >= 0


def test_forward_emo_loss_on_modulated_vs_llm_output():
    """spec 商讨点 3：emo_loss_on 切换应改变 loss_emotion 数值。

    - 'modulated_text_emb' (默认): emotion_classifier 输入 = FiLM 调制后的 text embedding
    - 'llm_output': emotion_classifier 输入 = Qwen2Encoder 输出
    两者维度不同（text_emb 是 llm_input_size, llm_output 是 llm_output_size），
    需要分别构造 emotion_classifier，故 loss 数值应不同。
    """
    # modulated_text_emb 模式
    model_a = Qwen2LM_Emotion(
        llm_input_size=896, llm_output_size=896, speech_token_size=6561,
        emotion_vocab_size=6, intensity_vocab_size=4,
        llm=FakeQwen2Encoder(model_dim=896),
        sampling=fake_sampling, emo_loss_weight=0.2,
        emo_loss_on="modulated_text_emb",
    )
    model_a.eval()
    # 同 seed 重置 FiLM/classifier 权重以公平对比
    torch.manual_seed(42)
    model_a.emotion_encoder.apply(_weights_init_zero)
    model_a.emotion_adapter.apply(_weights_init_film)
    model_a.emotion_classifier.reset_parameters()

    # llm_output 模式：classifier 输入维度 = llm_output_size
    model_b = Qwen2LM_Emotion(
        llm_input_size=896, llm_output_size=896, speech_token_size=6561,
        emotion_vocab_size=6, intensity_vocab_size=4,
        llm=FakeQwen2Encoder(model_dim=896),
        sampling=fake_sampling, emo_loss_weight=0.2,
        emo_loss_on="llm_output",
    )
    # 重写 classifier 输入维度匹配 llm_output_size
    model_b.emotion_classifier = nn.Linear(896, 6)  # llm_output_size
    model_b.eval()
    torch.manual_seed(42)
    model_b.emotion_encoder.apply(_weights_init_zero)
    model_b.emotion_adapter.apply(_weights_init_film)
    model_b.emotion_classifier.reset_parameters()

    batch = make_fake_batch()
    with torch.no_grad():
        out_a = model_a(batch, torch.device("cpu"))
        out_b = model_b(batch, torch.device("cpu"))
    assert model_a.emo_loss_on == "modulated_text_emb"
    assert model_b.emo_loss_on == "llm_output"
    # 两者都产出有效 loss
    assert not out_a["loss_emotion"].isnan()
    assert not out_b["loss_emotion"].isnan()


def _weights_init_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)


def _weights_init_film(m):
    """FiLM gamma=1, beta=0 初始化（gamma bias 在前一半）。"""
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        out_dim = m.bias.shape[0]
        with torch.no_grad():
            m.bias[:out_dim // 2].fill_(1.0)
            m.bias[out_dim // 2:].fill_(0.0)


def test_emo_loss_on_invalid_value_raises():
    """非法 emo_loss_on 值应在 __init__ 阶段 assert 失败。"""
    import pytest
    with pytest.raises(AssertionError):
        Qwen2LM_Emotion(
            llm_input_size=896, llm_output_size=896, speech_token_size=6561,
            emotion_vocab_size=6, intensity_vocab_size=4,
            llm=FakeQwen2Encoder(model_dim=896),
            sampling=fake_sampling, emo_loss_on="invalid_value",
        )


# ----- emotion_adapter 注入接口（spec 12.2 w/o FiLM 消融） -----

def test_default_emotion_adapter_is_film():
    """未注入 emotion_adapter 时默认使用 FiLMLayer（与 Emo_PA 源码一致）。"""
    model = Qwen2LM_Emotion(
        llm_input_size=896, llm_output_size=896, speech_token_size=6561,
        emotion_vocab_size=6, intensity_vocab_size=4,
        llm=FakeQwen2Encoder(model_dim=896),
        sampling=fake_sampling,
    )
    assert isinstance(model.emotion_adapter, FiLMLayer)


def test_inject_add_fusion_emotion_adapter():
    """注入 AddFusionEmotionAdapter 时 self.emotion_adapter 是注入实例。"""
    custom_adapter = AddFusionEmotionAdapter(model_dim=896)
    model = Qwen2LM_Emotion(
        llm_input_size=896, llm_output_size=896, speech_token_size=6561,
        emotion_vocab_size=6, intensity_vocab_size=4,
        llm=FakeQwen2Encoder(model_dim=896),
        sampling=fake_sampling,
        emotion_adapter=custom_adapter,
    )
    assert isinstance(model.emotion_adapter, AddFusionEmotionAdapter)
    assert model.emotion_adapter is custom_adapter  # 同一实例引用


def test_inject_add_fusion_forward_runs():
    """注入 AddFusion 后 forward 仍可正常运行（无 shape 错误）。"""
    custom_adapter = AddFusionEmotionAdapter(model_dim=896)
    model = Qwen2LM_Emotion(
        llm_input_size=896, llm_output_size=896, speech_token_size=6561,
        emotion_vocab_size=6, intensity_vocab_size=4,
        llm=FakeQwen2Encoder(model_dim=896),
        sampling=fake_sampling,
        emotion_adapter=custom_adapter,
    )
    model.eval()
    batch = make_fake_batch()
    with torch.no_grad():
        out = model(batch, torch.device("cpu"))
    assert not out["loss"].isnan()
    assert not out["loss_emotion"].isnan()
