"""基础 Emo-FiLM 运行时生命周期合同。"""
import threading
import importlib.util
from pathlib import Path

import pytest
import torch


def _make_model_with_fake_runtime():
    from cosyvoice.cli.model_emo import CosyVoice2Model_Emotion

    model = CosyVoice2Model_Emotion.__new__(CosyVoice2Model_Emotion)
    model.device = torch.device("cpu")
    model.fp16 = False
    model.lock = threading.Lock()
    model.tts_speech_token_dict = {}
    model.llm_end_dict = {}
    model.hift_cache_dict = {}

    def fake_llm_job(*args):
        uuid = args[-1]
        model.tts_speech_token_dict[uuid].append(2)
        model.llm_end_dict[uuid] = True

    model.llm_job = fake_llm_job
    return model


def _tts_kwargs():
    return {
        "text": torch.tensor([[2]], dtype=torch.int32),
        "emotion_ids": torch.ones(1, 1, dtype=torch.long),
        "intensity_ids": torch.ones(1, 1, dtype=torch.long),
    }


def test_uuid_state_is_cleaned_when_token2wav_fails():
    model = _make_model_with_fake_runtime()

    def failing_token2wav(*args, **kwargs):
        raise RuntimeError("synthetic token2wav failure")

    model.token2wav = failing_token2wav
    with pytest.raises(RuntimeError, match="synthetic token2wav failure"):
        list(model.tts(**_tts_kwargs()))

    assert model.tts_speech_token_dict == {}
    assert model.llm_end_dict == {}
    assert model.hift_cache_dict == {}


def test_uuid_state_is_cleaned_when_generator_is_closed():
    model = _make_model_with_fake_runtime()
    model.token2wav = lambda *args, **kwargs: torch.zeros(1, 4)

    generation = model.tts(**_tts_kwargs())
    next(generation)
    generation.close()

    assert model.tts_speech_token_dict == {}
    assert model.llm_end_dict == {}
    assert model.hift_cache_dict == {}


def test_llm_job_thread_errors_are_propagated_and_cleaned():
    model = _make_model_with_fake_runtime()

    def failing_llm_job(*args):
        raise RuntimeError("synthetic llm worker failure")

    model.llm_job = failing_llm_job
    model.token2wav = lambda *args, **kwargs: torch.zeros(1, 4)

    with pytest.raises(RuntimeError, match="synthetic llm worker failure"):
        list(model.tts(**_tts_kwargs()))

    assert model.tts_speech_token_dict == {}
    assert model.llm_end_dict == {}
    assert model.hift_cache_dict == {}


def test_bfloat16_model_state_hash_uses_stable_bytes():
    from cosyvoice.utils.emo_checkpoint import hash_model_state

    class _BFloat16Module(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("value", torch.tensor([value], dtype=torch.bfloat16))

    first = hash_model_state(_BFloat16Module(1.0))
    second = hash_model_state(_BFloat16Module(2.0))

    assert first != second


def test_model_state_hash_is_transparent_to_ddp_like_wrappers():
    from cosyvoice.utils.emo_checkpoint import hash_model_state

    bare = torch.nn.Linear(2, 2)

    class _Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    wrapped = _Wrapper(bare)
    assert hash_model_state(bare) == hash_model_state(wrapped)


def test_async_parquet_job_errors_are_propagated():
    module_path = Path(__file__).parents[1] / "tools" / "make_parquet_list.py"
    spec = importlib.util.spec_from_file_location("emofilm_make_parquet_list", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wait_for_async_jobs = module.wait_for_async_jobs

    class _FailedJob:
        def get(self):
            raise RuntimeError("synthetic parquet worker failure")

    with pytest.raises(RuntimeError, match="synthetic parquet worker failure"):
        wait_for_async_jobs([_FailedJob()])


def test_latest_checkpoint_is_atomic_and_finalized(tmp_path):
    from cosyvoice.utils.train_utils_emo import (
        finalize_latest_checkpoint,
        save_latest_checkpoint,
    )

    model = torch.nn.Linear(2, 2)
    save_latest_checkpoint(model, str(tmp_path), epoch=3, step=17)
    latest = tmp_path / "latest.pt"
    assert latest.is_file()
    assert not list(tmp_path.glob("*.tmp"))

    finalize_latest_checkpoint(str(tmp_path))
    assert not latest.exists()
    assert (tmp_path / "final.pt").is_file()
