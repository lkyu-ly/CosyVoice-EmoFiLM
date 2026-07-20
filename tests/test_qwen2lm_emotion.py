"""Qwen2LM_Emotion forward 单测（使用 fake Qwen2Encoder）。"""
import pytest
import torch
import torch.nn as nn

from cosyvoice.llm.emo_film import FiLMLayer
from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion


class FakeQwen2Encoder(nn.Module):
    def __init__(self, model_dim=896, pretrain_path=None):
        super().__init__()
        self.model = FakeHFModel(model_dim)

    def forward(self, xs, xs_lens):
        return xs, torch.ones(xs.shape[0], 1, xs.shape[1], dtype=torch.bool)

    def forward_one_step(self, xs, masks=None, cache=None):
        return xs, cache


class FakeHFModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = FakeEmbed(dim)


class FakeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(20000, dim)


def fake_sampling(scores, decoded_tokens, sampling):
    return scores.argmax()


def make_fake_batch(B=2, T=10, S=25):
    return {
        "text_token": torch.randint(0, 5000, (B, T)),
        "text_token_len": torch.tensor([T] * B, dtype=torch.int32),
        "speech_token": torch.randint(0, 6561, (B, S)),
        "speech_token_len": torch.tensor([S] * B, dtype=torch.int32),
        "emotion_ids": torch.ones(B, T, dtype=torch.long),
        "intensity_ids": torch.ones(B, T, dtype=torch.long),
    }


def make_model():
    return Qwen2LM_Emotion(
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


def test_forward_returns_expected_keys():
    model = make_model().eval()
    with torch.no_grad():
        output = model(make_fake_batch(), torch.device("cpu"))
    assert {"loss", "acc", "loss_tts", "loss_emotion"} <= set(output)
    assert output["loss"].numel() == 1
    assert not output["loss"].isnan()


def test_emotion_loss_reads_film_output_and_classifier_is_frozen():
    model = make_model()
    adapter_output = []
    classifier_input = []
    model.emotion_adapter.register_forward_hook(
        lambda module, inputs, output: adapter_output.append(output.detach())
    )
    model.emotion_classifier.register_forward_pre_hook(
        lambda module, inputs: classifier_input.append(inputs[0].detach())
    )

    model(make_fake_batch(), torch.device("cpu"))

    torch.testing.assert_close(classifier_input[0], adapter_output[0])
    assert all(not parameter.requires_grad for parameter in model.emotion_classifier.parameters())


def test_default_adapter_is_film():
    assert isinstance(make_model().emotion_adapter, FiLMLayer)


def test_invalid_loss_seam_is_rejected():
    with pytest.raises((TypeError, ValueError)):
        Qwen2LM_Emotion(
            llm_input_size=896,
            llm_output_size=896,
            speech_token_size=6561,
            emotion_vocab_size=6,
            intensity_vocab_size=4,
            llm=FakeQwen2Encoder(model_dim=896),
            sampling=fake_sampling,
            emo_loss_on="llm_output",
        )
