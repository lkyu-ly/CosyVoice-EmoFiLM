"""基础 Emo-FiLM 推理合同。"""
import threading
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from cosyvoice.llm.llm_emotion import DecodeResult, Qwen2LM_Emotion
from cosyvoice.utils.common import ras_sampling
from tests._emofilm_fakes import _FakeBackbone, _FakeHF, _FakeQwen


class _RecorderTokenizer:
    def __init__(self):
        self.calls = []

    def add_special_tokens(self, values):
        return None

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        return [1, 2] if "hello" in text else [3]


# _FakeBackbone / _FakeHF / _FakeQwen 从 tests._emofilm_fakes 复用
# （原先在本文件本地定义，现已 DRY 整合到共享测试辅助模块；原文本地版
# forward 的 mask tensor 未带显式 ``device=xs.device``，整合版统一带
# device，CPU 测试下行为完全等价）。


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


def _inference_inputs(model, target_len=3, prompt_speech_len=2):
    """v2 单流 inference kwargs（无 v1 死字段 prompt_text/prompt_emotion_ids）。

    prompt_speech_token / embedding 透传给 Flow/HiFT，不进 LLM lm_input。
    """
    return {
        "text_token": torch.tensor([[2, 3, 4][:target_len]]),
        "text_len": torch.tensor([target_len], dtype=torch.int32),
        "emotion_ids": torch.ones(1, target_len, dtype=torch.long),
        "intensity_ids": torch.ones(1, target_len, dtype=torch.long),
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


def test_llm_condition_excludes_prompt_speech():
    """v2 单流：LLM 条件 = SOS + FiLM(target text) + task；prompt speech 不进 lm_input。"""
    model = _make_model()
    captured = {}

    def fake_wrapper(lm_input, sampling, min_len, max_len, uuid):
        captured["lm_input"] = lm_input.detach().clone()
        captured["min_len"] = min_len
        captured["max_len"] = max_len
        if False:
            yield None

    model.inference_wrapper = fake_wrapper
    list(model.inference(**_inference_inputs(model, target_len=3, prompt_speech_len=2)))

    # SOS + target FiLM text + task；prompt speech 未拼接（shape 不含 prompt_speech_len）
    assert captured["lm_input"].shape[1] == 1 + 3 + 1


def test_prompt_is_retained_for_flow_and_hift():
    from cosyvoice.cli.model_emo import CosyVoice2Model_Emotion

    model = CosyVoice2Model_Emotion.__new__(CosyVoice2Model_Emotion)
    model.device = torch.device("cpu")
    model.fp16 = False
    model.lock = threading.Lock()
    model.tts_speech_token_dict = {}
    model.llm_end_dict = {}
    model.hift_cache_dict = {}
    seen = {}

    def fake_llm_job(*args, **kwargs):
        uuid = args[-1]
        model.tts_speech_token_dict[uuid].append(2)
        model.llm_end_dict[uuid] = True
        # 模拟真实 llm_job 暴露结构化解码结果（eos 完成），供 tts 门控判定
        model._last_decode_result = DecodeResult(
            tokens=[2], finish_reason="eos",
            min_len=2, max_len=4, num_valid_speech_tokens=1,
            invalid_token_retries=0, text_len=1,
        )

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


def test_max_len_is_ratio_derived_not_hardcoded():
    """v2 修复历史 max_len=200 硬编码 bug：max_len 由 text_len * ratio 推导。

    target_len=3, max_token_text_ratio=20 → max_len=60。采样器恒产合法 token、
    永不发 EOS → max_len_reached（inference 非 eos 不向声学侧产出 token）。
    """
    model = _make_model()
    model.llm_decoder = _FixedDecoder(model.speech_token_size + 3, token=2)
    list(model.inference(**_inference_inputs(model, target_len=3)))
    # 非 eos（max_len_reached）→ last_decode_result 记录结构化结果
    assert model.last_decode_result.finish_reason == "max_len_reached"
    assert model.last_decode_result.max_len == 60  # int(3 * 20)，非历史 200
    assert model.last_decode_result.min_len == 6   # int(3 * 2)


def test_eos_is_resampled_before_min_len():
    """EOS-before-min 触发重采样（不提前终止）；重采样复用原始 scores。

    target_len=1 → min_len=2。采样器：count1=EOS（before min → 重采样），
    count2-3=合法 token（累积到 len=2 >= min_len），count4=EOS（after min → eos 完成）。
    """
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
        # count1: EOS before min（重采样）；count2-3: 合法 token；count4+: EOS after min
        if calls["count"] == 1 or calls["count"] >= 4:
            return eos
        return 2

    model.llm_decoder = _Decoder()
    model.sampling = sampling
    inputs = _inference_inputs(model, target_len=1, prompt_speech_len=0)
    outputs = list(model.inference(**inputs))

    assert outputs
    assert (outputs[0].item() if torch.is_tensor(outputs[0]) else outputs[0]) == 2
    assert len(scores_seen) >= 2
    assert torch.isfinite(scores_seen[0][eos])
    # 重采样复用同一 scores（同一 step 内 inner loop 不重新前向）
    torch.testing.assert_close(scores_seen[1], scores_seen[0])
    # eos 完成 → inference 产出 token
    assert model.last_decode_result.finish_reason == "eos"


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


# ---- Task 1: 包装层 tts 门控（非 eos 不得落 WAV）----

def _make_emo_model_for_tts():
    """构造一个跳过 __init__ 的 CosyVoice2Model_Emotion，仅满足 tts 所需属性。"""
    from cosyvoice.cli.model_emo import CosyVoice2Model_Emotion

    model = CosyVoice2Model_Emotion.__new__(CosyVoice2Model_Emotion)
    model.lock = threading.Lock()
    model.tts_speech_token_dict = {}
    model.llm_end_dict = {}
    model.hift_cache_dict = {}
    model.device = "cpu"
    model.fp16 = False
    model.llm_context = MagicMock()
    return model


def test_non_eos_finish_reason_does_not_token2wav():
    """max_len_reached 时 tts 不进 token2wav、不 yield 音频。

    合同（Task 1）：LLM ``inference`` 非 eos 不向 Flow/HiFT 产出 token，
    因此 ``last_decode_result.finish_reason`` 非 eos。``tts`` 必须据此跳过
    ``token2wav``，并 yield 一个 ``tts_speech=None`` 的标记结果，让下游
    （T4）能识别"非 eos 不得落 WAV"。
    """
    model = _make_emo_model_for_tts()

    bad = DecodeResult(
        tokens=[], finish_reason="max_len_reached",
        min_len=2, max_len=4, num_valid_speech_tokens=0,
        invalid_token_retries=0, text_len=1,
    )
    llm = MagicMock()
    llm.inference.return_value = iter([])  # 非 eos：inference 不 yield token
    llm.last_decode_result = bad
    model.llm = llm

    token2wav_called = []
    model.token2wav = lambda **kw: token2wav_called.append(kw) or torch.zeros(1)

    outputs = list(model.tts(
        text=torch.zeros(1, 1, dtype=torch.int32),
        emotion_ids=torch.zeros(1, 1, dtype=torch.long),
        intensity_ids=torch.zeros(1, 1, dtype=torch.long),
    ))

    assert token2wav_called == [], "非 eos 不得调用 token2wav"
    assert outputs, "tts 必须至少 yield 一个标记结果"
    assert outputs[0].get("finish_reason") == "max_len_reached"
    assert outputs[0].get("tts_speech") is None
    # 结构化 decode_result 透传，便于下游 T4 审计 / 写 manifest
    assert outputs[0].get("decode_result") is bad


def test_eos_finish_reason_still_token2wav():
    """eos 时正常 token2wav + yield 音频，并携带 finish_reason/decode_result。"""
    model = _make_emo_model_for_tts()

    good = DecodeResult(
        tokens=[2, 3], finish_reason="eos",
        min_len=2, max_len=4, num_valid_speech_tokens=2,
        invalid_token_retries=0, text_len=1,
    )
    llm = MagicMock()
    llm.inference.return_value = iter([torch.tensor(2), torch.tensor(3)])
    llm.last_decode_result = good
    model.llm = llm

    token2wav_called = []
    model.token2wav = lambda **kw: token2wav_called.append(kw) or torch.zeros(1, 8)

    outputs = list(model.tts(
        text=torch.zeros(1, 1, dtype=torch.int32),
        emotion_ids=torch.zeros(1, 1, dtype=torch.long),
        intensity_ids=torch.zeros(1, 1, dtype=torch.long),
    ))

    assert len(token2wav_called) == 1, "eos 应调用 token2wav 一次"
    assert outputs
    assert outputs[0]["finish_reason"] == "eos"
    assert outputs[0]["tts_speech"] is not None
    assert outputs[0]["decode_result"] is good


def test_tts_thread_llm_error_propagates_and_skips_token2wav():
    """llm_job 线程抛错时 tts 重抛且不进 token2wav（不掩盖线程错误）。"""
    model = _make_emo_model_for_tts()

    class _Boom(RuntimeError):
        pass

    def boom_inference(**kwargs):
        raise _Boom("decode failed")
        yield  # noqa: unreachable，使其成为 generator

    llm = MagicMock()
    llm.inference.side_effect = boom_inference
    llm.last_decode_result = None
    model.llm = llm

    token2wav_called = []
    model.token2wav = lambda **kw: token2wav_called.append(kw) or torch.zeros(1)

    gen = model.tts(
        text=torch.zeros(1, 1, dtype=torch.int32),
        emotion_ids=torch.zeros(1, 1, dtype=torch.long),
        intensity_ids=torch.zeros(1, 1, dtype=torch.long),
    )
    try:
        list(gen)
        raised = None
    except _Boom as exc:
        raised = exc

    assert isinstance(raised, _Boom), "线程错误必须重抛"
    assert token2wav_called == [], "出错时不得调 token2wav"


# ---- Task 2: decode_config 透传（yaml → inference_emo_film → tts → llm.inference）----

def test_decode_config_threaded_to_inference():
    """tts(decode_config=...) 的三项长度参数实际传到 llm.inference。

    合同（Task 2 / schema §2 decode_config）：yaml ``decode_config`` 必须能
    覆盖 ``Qwen2LM_Emotion.inference`` 的硬编码默认（20/2/2000），否则改 yaml
    不生效（历史 bug：``model_emo.py`` 不透传 decode_config，LLM 默认值恰好
    等于 yaml 但改 yaml 无效）。
    """
    model = _make_emo_model_for_tts()

    good = DecodeResult(
        tokens=[2, 3], finish_reason="eos",
        min_len=2, max_len=4, num_valid_speech_tokens=2,
        invalid_token_retries=0, text_len=1,
    )
    captured = {}

    def fake_inference(**kw):
        captured.update(kw)
        return iter([torch.tensor(2), torch.tensor(3)])

    llm = MagicMock()
    llm.inference.side_effect = fake_inference
    llm.last_decode_result = good
    model.llm = llm
    model.token2wav = lambda **kw: torch.zeros(1, 8)

    list(model.tts(
        text=torch.zeros(1, 1, dtype=torch.int32),
        emotion_ids=torch.zeros(1, 1, dtype=torch.long),
        intensity_ids=torch.zeros(1, 1, dtype=torch.long),
        decode_config={"min_token_text_ratio": 1,
                       "max_token_text_ratio": 5,
                       "max_len_hard_cap": 100},
    ))

    assert captured.get("min_token_text_ratio") == 1
    assert captured.get("max_token_text_ratio") == 5
    assert captured.get("max_len_hard_cap") == 100


def test_decode_config_none_uses_llm_defaults():
    """decode_config=None 时**不**传长度 kwargs，回退到 ``Qwen2LM_Emotion.inference`` 默认。

    防止"None 被当 dict 解包"导致 TypeError / 传 None 值破坏长度推导。
    """
    model = _make_emo_model_for_tts()

    good = DecodeResult(
        tokens=[2], finish_reason="eos",
        min_len=2, max_len=4, num_valid_speech_tokens=1,
        invalid_token_retries=0, text_len=1,
    )
    captured = {}

    def fake_inference(**kw):
        captured.update(kw)
        return iter([torch.tensor(2)])

    llm = MagicMock()
    llm.inference.side_effect = fake_inference
    llm.last_decode_result = good
    model.llm = llm
    model.token2wav = lambda **kw: torch.zeros(1, 8)

    list(model.tts(
        text=torch.zeros(1, 1, dtype=torch.int32),
        emotion_ids=torch.zeros(1, 1, dtype=torch.long),
        intensity_ids=torch.zeros(1, 1, dtype=torch.long),
        decode_config=None,
    ))

    # 长度 kwargs 不在 captured（交给 inference 默认值）
    assert "min_token_text_ratio" not in captured
    assert "max_token_text_ratio" not in captured
    assert "max_len_hard_cap" not in captured


def test_inference_emo_film_threads_decode_config():
    """``CosyVoice2_Emotion.inference_emo_film`` 把 decode_config 透传给 model.tts。

    合同（Task 2）：yaml ``decode_config`` 由 ``__init__`` 抽取到
    ``self.decode_config``（与 ``sample_rate`` 同模式；不保留全 configs 避免
    误用），``inference_emo_film`` 读取并作为 ``model.tts(decode_config=...)`` 入参。
    """
    from cosyvoice.cli.cosyvoice_emo import CosyVoice2_Emotion

    cv2 = CosyVoice2_Emotion.__new__(CosyVoice2_Emotion)
    cv2.sample_rate = 24000
    cv2.decode_config = {"min_token_text_ratio": 2,
                         "max_token_text_ratio": 20,
                         "max_len_hard_cap": 2000}

    frontend = MagicMock()
    frontend.frontend_emo_film.return_value = {
        "text": torch.zeros(1, 1, dtype=torch.int32),
        "emotion_ids": torch.zeros(1, 1, dtype=torch.long),
        "intensity_ids": torch.zeros(1, 1, dtype=torch.long),
    }
    cv2.frontend = frontend

    captured = {}

    def fake_tts(**kw):
        captured.update(kw)
        yield {"tts_speech": torch.zeros(1, 8), "finish_reason": "eos"}

    cv2.model = MagicMock()
    cv2.model.tts = fake_tts

    list(cv2.inference_emo_film(
        text_with_emo="<emotion type='hap' intensity='high'>hi</emotion>",
        prompt_text="ref",
        prompt_wav_path="/unused.wav",
    ))

    assert captured.get("decode_config") == cv2.decode_config
