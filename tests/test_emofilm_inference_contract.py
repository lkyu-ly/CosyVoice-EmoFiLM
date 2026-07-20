"""基础 Emo-FiLM 推理合同。"""
import torch
import torch.nn as nn

from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion
from cosyvoice.utils.common import ras_sampling


class _RecorderTokenizer:
    def __init__(self):
        self.calls = []

    def add_special_tokens(self, values):
        return None

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        return [1, 2] if "hello" in text else [3]


class _FakeBackbone(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, model_dim)


class _FakeHF(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model = _FakeBackbone(model_dim)


class _FakeQwen(nn.Module):
    def __init__(self, model_dim=4):
        super().__init__()
        self.model = _FakeHF(model_dim)

    def forward_one_step(self, xs, masks=None, cache=None):
        return xs, cache

    def forward(self, xs, xs_lens):
        return xs, torch.ones(xs.shape[0], 1, xs.shape[1], dtype=torch.bool)


class _FixedDecoder(nn.Module):
    def __init__(self, vocab_size, token):
        super().__init__()
        self.vocab_size = vocab_size
        self.token = token

    def forward(self, hidden):
        logits = torch.full(
            (*hidden.shape[:-1], self.vocab_size),
            -100.0,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        logits[..., self.token] = 100.0
        return logits


def _make_model(speech_token_size=10):
    return Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=speech_token_size,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
    )


def _inference_inputs(model, target_len=3, prompt_len=2, prompt_speech_len=2):
    return {
        "text_token": torch.tensor([[2, 3, 4][:target_len]]),
        "text_len": torch.tensor([target_len], dtype=torch.int32),
        "emotion_ids": torch.ones(1, target_len, dtype=torch.long),
        "intensity_ids": torch.ones(1, target_len, dtype=torch.long),
        "prompt_text": torch.full((1, prompt_len), 77, dtype=torch.long),
        "prompt_text_len": torch.tensor([prompt_len], dtype=torch.int32),
        "prompt_emotion_ids": torch.full((1, prompt_len), 2, dtype=torch.long),
        "prompt_intensity_ids": torch.full((1, prompt_len), 2, dtype=torch.long),
        "prompt_speech_token": torch.full((1, prompt_speech_len), 8, dtype=torch.long),
        "prompt_speech_token_len": torch.tensor([prompt_speech_len], dtype=torch.int32),
        "embedding": torch.zeros(1, 4),
    }


def test_text_is_lowercased(monkeypatch):
    import cosyvoice.tokenizer.emo_tokenizer as module

    recorder = _RecorderTokenizer()
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: recorder,
    )
    tokenizer = module.QwenTokenizer_Emotion("unused")
    result = tokenizer.encode_plus(
        "<emotion type='hap' intensity='high'>Hello WORLD</emotion> Plain TEXT"
    )

    assert result["emotion_ids"].tolist() == [2, 2, 3]
    assert result["intensity_ids"].tolist() == [3, 3, 1]
    assert recorder.calls == ["hello world", " plain text"]


def test_llm_condition_excludes_prompt_text_and_prompt_speech():
    model = _make_model()
    captured = {}

    def fake_wrapper(lm_input, sampling, min_len, max_len, uuid):
        captured["lm_input"] = lm_input.detach().clone()
        captured["min_len"] = min_len
        captured["max_len"] = max_len
        if False:
            yield None

    model.inference_wrapper = fake_wrapper
    list(model.inference(**_inference_inputs(model)))

    # SOS + target FiLM text + task; prompt text and prompt speech are absent.
    assert captured["lm_input"].shape[1] == 1 + 3 + 1


def test_prompt_is_retained_for_flow_and_hift():
    import threading
    from cosyvoice.cli.model_emo import CosyVoice2Model_Emotion

    model = CosyVoice2Model_Emotion.__new__(CosyVoice2Model_Emotion)
    model.device = torch.device("cpu")
    model.fp16 = False
    model.lock = threading.Lock()
    model.tts_speech_token_dict = {}
    model.llm_end_dict = {}
    model.hift_cache_dict = {}
    seen = {}

    def fake_llm_job(*args):
        uuid = args[-1]
        model.tts_speech_token_dict[uuid].append(2)
        model.llm_end_dict[uuid] = True

    def fake_token2wav(token, prompt_token, prompt_feat, embedding, **kwargs):
        seen["prompt_token"] = prompt_token
        seen["prompt_feat"] = prompt_feat
        seen["embedding"] = embedding
        return torch.zeros(1, 4)

    model.llm_job = fake_llm_job
    model.token2wav = fake_token2wav
    prompt_token = torch.tensor([[7, 8]], dtype=torch.int32)
    prompt_feat = torch.ones(1, 3, 80)
    embedding = torch.ones(1, 4)
    list(
        model.tts(
            text=torch.tensor([[2]], dtype=torch.int32),
            emotion_ids=torch.ones(1, 1, dtype=torch.long),
            intensity_ids=torch.ones(1, 1, dtype=torch.long),
            flow_prompt_speech_token=prompt_token,
            prompt_speech_feat=prompt_feat,
            flow_embedding=embedding,
        )
    )

    assert seen["prompt_token"] is prompt_token
    assert seen["prompt_feat"] is prompt_feat
    assert seen["embedding"] is embedding


def test_max_len_is_200():
    model = _make_model()
    model.llm_decoder = _FixedDecoder(model.speech_token_size + 3, token=2)
    outputs = list(model.inference(**_inference_inputs(model)))

    assert len(outputs) == 200


def test_eos_is_resampled_before_min_len():
    model = _make_model()
    eos = model.eos_token
    scores_seen = []
    calls = {"count": 0}

    class _Decoder(nn.Module):
        def forward(self, hidden):
            scores = torch.zeros(
                (*hidden.shape[:-1], model.speech_token_size + 3),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            scores[..., eos] = 5.0
            scores[..., 2] = 4.0
            return scores

    def sampling(scores, decoded, sampling):
        scores_seen.append(scores.detach().clone())
        calls["count"] += 1
        return eos if calls["count"] == 1 else 2

    model.llm_decoder = _Decoder()
    model.sampling = sampling
    inputs = _inference_inputs(model, target_len=1, prompt_len=0, prompt_speech_len=0)
    outputs = list(model.inference(**inputs))

    assert outputs
    assert (outputs[0].item() if torch.is_tensor(outputs[0]) else outputs[0]) == 2
    assert len(scores_seen) >= 2
    assert torch.isfinite(scores_seen[0][eos])
    torch.testing.assert_close(scores_seen[1], scores_seen[0])


def test_auxiliary_special_tokens_do_not_stop_or_extend_prefix():
    model = _make_model()
    sequence = iter([model.speech_token_size + 1, model.speech_token_size + 2, 2, model.eos_token])
    model.sampling = lambda scores, decoded, sampling: next(sequence)
    model.llm_decoder = _FixedDecoder(model.speech_token_size + 3, token=2)

    outputs = list(
        model.inference_wrapper(
            torch.zeros(1, 2, 4), sampling=25, min_len=0, max_len=10, uuid="test"
        )
    )

    assert outputs == [2]


def test_auxiliary_special_tokens_are_resampled_without_advancing_prefix():
    model = _make_model()
    calls = []

    class _RecorderQwen(_FakeQwen):
        def forward_one_step(self, xs, masks=None, cache=None):
            calls.append(xs.detach().clone())
            return super().forward_one_step(xs, masks=masks, cache=cache)

    model.llm = _RecorderQwen(4)
    sequence = iter([model.speech_token_size + 1, model.speech_token_size + 2, 2, model.eos_token])
    model.sampling = lambda scores, decoded, sampling: next(sequence)

    assert list(model.inference_wrapper(
        torch.zeros(1, 2, 4), sampling=25, min_len=0, max_len=10, uuid="test"
    )) == [2]
    assert len(calls) == 2
    assert calls[0].shape == (1, 2, 4)
    assert calls[1].shape == (1, 1, 4)


def test_kv_cache_mask_includes_past_sequence_length():
    model = _make_model()
    captured_masks = []

    class _Cache:
        def __init__(self, length):
            self.length = length

        def get_seq_length(self):
            return self.length

    class _CacheQwen(_FakeQwen):
        def forward_one_step(self, xs, masks=None, cache=None):
            captured_masks.append(masks.detach().clone())
            past_length = 0 if cache is None else cache.get_seq_length()
            return xs, _Cache(past_length + xs.shape[1])

    model.llm = _CacheQwen(4)
    sequence = iter([2, model.eos_token])
    model.sampling = lambda scores, decoded, sampling: next(sequence)

    assert list(model.inference_wrapper(
        torch.zeros(1, 2, 4), sampling=25, min_len=0, max_len=10, uuid="test"
    )) == [2]
    assert captured_masks[0].shape == (1, 2, 2)
    assert captured_masks[1].shape == (1, 1, 3)


def test_ras_fallback_uses_unmodified_scores(monkeypatch):
    original = torch.tensor([0.2, 0.3, 0.5])
    fallback_scores = []

    monkeypatch.setattr(
        "cosyvoice.utils.common.nucleus_sampling",
        lambda scores, top_p=0.8, top_k=25: 2,
    )

    def fallback(scores, decoded_tokens, sampling):
        fallback_scores.append(scores.clone())
        return 1

    monkeypatch.setattr("cosyvoice.utils.common.random_sampling", fallback)
    assert ras_sampling(original.clone(), [2], sampling=25, win_size=1, tau_r=0.1) == 1
    torch.testing.assert_close(fallback_scores[0], original)
