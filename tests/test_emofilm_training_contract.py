"""基础 Emo-FiLM 训练合同。

这些测试只依赖小型 fake Qwen 模块，验证会改变训练行为的公开合同，
不加载生产 checkpoint。
"""
import re
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion
from cosyvoice.utils.train_utils_emo import freeze_all_except, init_optimizer_emo


EMOFILM_TRAINABLE_MODULES = (
    "emotion_encoder",
    "emotion_adapter",
    "llm_decoder",
)


def _checkpoint_helpers():
    try:
        from cosyvoice.utils.emo_checkpoint import load_base_state, load_trained_state
    except ModuleNotFoundError as exc:
        pytest.fail(f"checkpoint helper is not implemented: {exc}")
    return load_base_state, load_trained_state


class _FakeBackbone(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, model_dim)


class _FakeHF(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model = _FakeBackbone(model_dim)


class _FakeQwen(nn.Module):
    def __init__(self, model_dim=8):
        super().__init__()
        self.model = _FakeHF(model_dim)
        self.output_bias = nn.Parameter(torch.ones(model_dim))

    def forward(self, xs, xs_lens):
        mask = torch.ones(xs.shape[0], 1, xs.shape[1], dtype=torch.bool, device=xs.device)
        return xs + self.output_bias, mask

    def forward_one_step(self, xs, masks=None, cache=None):
        return xs + self.output_bias, cache


def _make_model(model_dim=8, speech_token_size=16):
    return Qwen2LM_Emotion(
        llm_input_size=model_dim,
        llm_output_size=model_dim,
        speech_token_size=speech_token_size,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(model_dim),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
    )


def _make_batch(batch_size=1, text_len=3, speech_len=6, speech_token_size=16):
    return {
        "text_token": torch.tensor([[2, 3, 4]] * batch_size),
        "text_token_len": torch.tensor([text_len] * batch_size, dtype=torch.int32),
        "speech_token": torch.tensor([list(range(1, speech_len + 1))] * batch_size),
        "speech_token_len": torch.tensor([speech_len] * batch_size, dtype=torch.int32),
        "emotion_ids": torch.ones(batch_size, text_len, dtype=torch.long),
        "intensity_ids": torch.ones(batch_size, text_len, dtype=torch.long),
    }


def test_emotion_classifier_reads_modulated_text():
    model = _make_model()
    adapter_outputs = []
    classifier_inputs = []
    model.emotion_adapter.register_forward_hook(
        lambda module, inputs, output: adapter_outputs.append(output.detach().clone())
    )
    model.emotion_classifier.register_forward_pre_hook(
        lambda module, inputs: classifier_inputs.append(inputs[0].detach().clone())
    )

    model(_make_batch(), torch.device("cpu"))

    assert adapter_outputs
    assert classifier_inputs
    torch.testing.assert_close(
        classifier_inputs[0], adapter_outputs[0], atol=1e-6, rtol=1e-6
    )


def test_effective_model_topology_and_shapes_match_emofilm_contract():
    model = _make_model(model_dim=896, speech_token_size=6561)

    assert model.llm_input_size == 896
    assert model.emotion_encoder.emotion_embedding.weight.shape == (6, 896)
    assert model.emotion_encoder.intensity_embedding.weight.shape == (4, 896)
    assert model.emotion_adapter.projection.weight.shape == (1792, 896)
    assert model.emotion_classifier.weight.shape == (6, 896)
    assert model.llm_decoder.out_features == 6564
    assert model.speech_embedding.num_embeddings == 6564


def test_exact_trainable_module_names():
    model = _make_model()
    freeze_all_except(model, EMOFILM_TRAINABLE_MODULES)

    trainable_prefixes = {
        name.split(".", 1)[0]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_prefixes == {
        "emotion_encoder",
        "emotion_adapter",
        "llm_decoder",
    }


def test_classifier_is_frozen_and_not_in_optimizer():
    model = _make_model()
    freeze_all_except(model, EMOFILM_TRAINABLE_MODULES)
    optimizer = init_optimizer_emo(
        model,
        {
            "train_conf": {
                "optim": "adam",
                "optim_conf": {"lr": 1e-5, "new_params_lr": 1e-5},
            }
        },
    )

    classifier_ids = {id(parameter) for parameter in model.emotion_classifier.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(not parameter.requires_grad for parameter in model.emotion_classifier.parameters())
    assert classifier_ids.isdisjoint(optimizer_ids)


def test_token_mel_ratio_two_trims_both_sequences():
    from cosyvoice.dataset.processor import trim_token_mel

    sample = {
        "speech_token": [1, 2, 3, 4, 5],
        "speech_feat": torch.zeros(8, 80),
    }
    trimmed = trim_token_mel(sample, token_mel_ratio=2)

    assert len(trimmed["speech_token"]) == 4
    assert trimmed["speech_feat"].shape[0] == 8


def test_dataset_compute_fbank_accepts_token_mel_ratio():
    from cosyvoice.dataset.processor import compute_fbank

    sample = {
        "utt": "u1",
        "sample_rate": 24000,
        "speech": torch.zeros(1, 2400),
        "text_token": [1, 2],
        "speech_token": [1, 2, 3, 4, 5],
    }

    def fake_feat_extractor(speech):
        return torch.zeros(1, 80, 8)

    output = list(
        compute_fbank(
            iter([sample]),
            feat_extractor=fake_feat_extractor,
            token_mel_ratio=2,
        )
    )
    assert len(output) == 1
    assert len(output[0]["speech_token"]) == 4
    assert output[0]["speech_feat"].shape[0] == 8


def test_single_process_dataloader_does_not_set_prefetch_factor(tmp_path):
    from cosyvoice.utils.train_utils import init_dataset_and_dataloader

    train_list = tmp_path / "train.list"
    cv_list = tmp_path / "cv.list"
    train_list.write_text("", encoding="utf-8")
    cv_list.write_text("", encoding="utf-8")
    args = SimpleNamespace(
        train_data=str(train_list),
        cv_data=str(cv_list),
        num_workers=0,
        prefetch=100,
        pin_memory=False,
    )

    _, _, train_loader, cv_loader = init_dataset_and_dataloader(
        args,
        {"data_pipeline": []},
        gan=False,
        dpo=False,
    )

    assert train_loader.prefetch_factor is None
    assert cv_loader.prefetch_factor is None


def test_config_uses_static_batch_four():
    config = Path(__file__).parents[1].joinpath("conf", "emo_film.yaml").read_text()
    batch_block = re.search(
        r"batch:\s*!name:cosyvoice\.dataset\.processor\.batch(?P<body>.*?)(?=\n\n|\n#|\ndata_pipeline:)",
        config,
        flags=re.DOTALL,
    )
    assert batch_block is not None
    assert re.search(r"batch_type:\s*['\"]static['\"]", batch_block.group("body"))
    assert re.search(r"batch_size:\s*4\b", batch_block.group("body"))


def test_train_entry_is_basic_sft_only():
    source = Path(__file__).parents[1].joinpath("cosyvoice", "bin", "train_emo.py").read_text()

    assert "--dpo" not in source
    assert "DPOLoss" not in source
    assert "train_one_epoc_gan" not in source
    assert "train_conf_gan" not in source


def test_train_entry_exposes_only_torch_ddp():
    source = Path(__file__).parents[1].joinpath("cosyvoice", "bin", "train_emo.py").read_text()

    assert 'choices=["torch_ddp"]' in source or "choices=['torch_ddp']" in source
    assert "deepspeed.add_config_arguments" not in source
    assert "deepspeed.save_states" not in source


class _CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.emotion_encoder = nn.Linear(3, 3)


def test_base_checkpoint_rejects_non_emotion_missing_key():
    load_base_state, _ = _checkpoint_helpers()
    model = _CheckpointModel()
    base_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("emotion_encoder.")
    }
    missing_backbone = dict(base_state)
    missing_backbone.pop("backbone.weight")

    load_base_state(model, base_state)
    with pytest.raises((KeyError, RuntimeError, ValueError), match="backbone.weight"):
        load_base_state(model, missing_backbone)


def test_trained_checkpoint_loads_strictly():
    _, load_trained_state = _checkpoint_helpers()
    model = _CheckpointModel()
    state = {key: value.clone() for key, value in model.state_dict().items()}

    load_trained_state(model, state)

    missing = dict(state)
    missing.pop("backbone.bias")
    with pytest.raises((KeyError, RuntimeError, ValueError), match="backbone.bias"):
        load_trained_state(model, missing)

    extra = dict(state)
    extra["unexpected.key"] = torch.zeros(1)
    with pytest.raises((KeyError, RuntimeError, ValueError), match="unexpected.key"):
        load_trained_state(model, extra)


def test_resume_checkpoint_preserves_epoch_and_step_metadata(tmp_path):
    from cosyvoice.bin.train_emo import load_resume_checkpoint

    model = _CheckpointModel()
    checkpoint = tmp_path / "latest.pt"
    payload = {key: value.clone() for key, value in model.state_dict().items()}
    payload.update({"epoch": 3, "step": 17})
    torch.save(payload, checkpoint)

    start_step, start_epoch = load_resume_checkpoint(model, str(checkpoint))

    assert start_step == 17
    assert start_epoch == 3


def test_training_checkpoint_loader_uses_base_policy_for_pretrained_state(tmp_path):
    from cosyvoice.bin.train_emo import load_training_checkpoint

    model = _CheckpointModel()
    checkpoint = tmp_path / "llm.pt"
    base_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("emotion_encoder.")
    }
    torch.save(base_state, checkpoint)

    start_step, start_epoch, checkpoint_role = load_training_checkpoint(
        model, str(checkpoint)
    )

    assert (start_step, start_epoch) == (0, -1)
    assert checkpoint_role == "base"


def test_training_checkpoint_loader_uses_strict_policy_for_resume_state(tmp_path):
    from cosyvoice.bin.train_emo import load_training_checkpoint

    model = _CheckpointModel()
    checkpoint = tmp_path / "latest.pt"
    payload = {key: value.clone() for key, value in model.state_dict().items()}
    payload.update({"epoch": 4, "step": 23})
    torch.save(payload, checkpoint)

    start_step, start_epoch, checkpoint_role = load_training_checkpoint(
        model, str(checkpoint)
    )

    assert (start_step, start_epoch) == (23, 4)
    assert checkpoint_role == "resume"

    payload.pop("backbone.weight")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="backbone.weight"):
        load_training_checkpoint(model, str(checkpoint))


def test_training_identity_records_resolved_config_and_parameter_hash(tmp_path):
    from cosyvoice.bin.train_emo import write_training_identity, write_resolved_config
    from cosyvoice.utils.emo_checkpoint import hash_model_state

    contract = tmp_path / "contract"
    provenance = contract / "provenance"
    provenance.mkdir(parents=True)
    for filename, value in (
        ("contract.json", {"contract_name": "emofilm_v1"}),
        ("sources.json", []),
        ("membership.json", {"train": [], "cv": []}),
    ):
        (provenance / filename).write_text(
            __import__("json").dumps(value), encoding="utf-8"
        )

    model = _CheckpointModel()
    base_checkpoint = tmp_path / "llm.pt"
    torch.save(
        {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not key.startswith("emotion_encoder.")
        },
        base_checkpoint,
    )
    args = SimpleNamespace(
        train_engine="torch_ddp",
        model="llm",
        config="conf/emo_film.yaml",
        train_data="data/contracts/emofilm_v1/splits/train/parquet/data.list",
        cv_data="data/contracts/emofilm_v1/splits/cv/parquet/data.list",
        qwen_pretrain_path="pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN",
        checkpoint=str(base_checkpoint),
        model_dir=str(tmp_path / "exp"),
        tensorboard_dir=str(tmp_path / "tb"),
        num_workers=0,
        prefetch=100,
        pin_memory=False,
        use_amp=False,
        timeout=60,
        contract_dir=str(contract),
        seed=1986,
    )
    resolved = tmp_path / "exp" / "resolved.yaml"
    write_resolved_config(
        resolved,
        config_path=args.config,
        args=args,
        train_conf={"max_epoch": 5, "batch_size": 4},
    )
    identity = write_training_identity(
        tmp_path / "exp" / "train_identity.json",
        model=model,
        code_root=Path(__file__).parents[1],
        contract_dir=contract,
        command="torchrun cosyvoice/bin/train_emo.py",
        seed=1986,
        base_checkpoint=base_checkpoint,
        resolved_config=resolved,
        checkpoint_role="base",
    )

    assert identity["contract_name"] == "emofilm_v1"
    assert identity["base_checkpoint"]["sha256"]
    assert identity["extra"]["parameter_hash"] == hash_model_state(model)
    assert identity["extra"]["resolved_config"] == str(resolved.resolve())
    assert yaml.safe_load(resolved.read_text(encoding="utf-8"))["train_conf"]["max_epoch"] == 5


def test_training_identity_records_final_checkpoint_and_parameter_hash(tmp_path):
    from cosyvoice.bin.train_emo import update_training_identity

    identity_path = tmp_path / "train_identity.json"
    identity_path.write_text(
        __import__("json").dumps({"extra": {"checkpoint_role": "base"}}),
        encoding="utf-8",
    )
    model = _CheckpointModel()
    final_checkpoint = tmp_path / "final.pt"
    torch.save(model.state_dict(), final_checkpoint)

    identity = update_training_identity(
        identity_path,
        model=model,
        final_checkpoint=final_checkpoint,
    )

    assert identity["extra"]["final_parameter_hash"]
    assert identity["extra"]["final_checkpoint"]["sha256"]


def test_resume_does_not_overwrite_init_checkpoint(tmp_path, monkeypatch):
    from cosyvoice.bin.train_emo import save_init_checkpoint_if_new
    import cosyvoice.bin.train_emo as train_module

    calls = []
    monkeypatch.setattr(
        train_module,
        "save_model",
        lambda model, name, info: calls.append((model, name, info)),
    )
    info = {"model_dir": str(tmp_path)}

    assert save_init_checkpoint_if_new(object(), info, resumed=True) is False
    assert calls == []

    (tmp_path / "init.pt").touch()
    with pytest.raises(FileExistsError, match="init.pt"):
        save_init_checkpoint_if_new(object(), info, resumed=False)


def test_new_run_writes_init_checkpoint_once(tmp_path, monkeypatch):
    from cosyvoice.bin.train_emo import save_init_checkpoint_if_new
    import cosyvoice.bin.train_emo as train_module

    calls = []
    monkeypatch.setattr(
        train_module,
        "save_model",
        lambda model, name, info: calls.append((model, name, info)),
    )
    info = {"model_dir": str(tmp_path)}

    assert save_init_checkpoint_if_new(object(), info, resumed=False) is True
    assert calls[0][1] == "init"


def test_base_checkpoint_starts_a_new_init_checkpoint_lifecycle():
    from cosyvoice.bin.train_emo import checkpoint_is_resume

    assert checkpoint_is_resume("base") is False
    assert checkpoint_is_resume("fresh") is False
    assert checkpoint_is_resume("resume") is True


def test_rank_zero_checkpoint_finalize_is_single_writer(tmp_path):
    from cosyvoice.bin.train_emo import finalize_checkpoint_on_rank_zero
    from cosyvoice.utils.train_utils_emo import save_latest_checkpoint

    model = nn.Linear(2, 2)
    save_latest_checkpoint(model, str(tmp_path), epoch=1, step=2)

    assert finalize_checkpoint_on_rank_zero(str(tmp_path), rank=1) is False
    assert (tmp_path / "latest.pt").is_file()
    assert not (tmp_path / "final.pt").exists()

    assert finalize_checkpoint_on_rank_zero(str(tmp_path), rank=0) is True
    assert not (tmp_path / "latest.pt").exists()
    assert (tmp_path / "final.pt").is_file()
