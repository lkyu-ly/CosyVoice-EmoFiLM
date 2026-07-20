"""唯一基础 Emo-FiLM SFT 训练入口。"""
from __future__ import print_function

import argparse
import datetime
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.emo_checkpoint import (
    hash_model_state,
    load_base_state,
    load_trained_state,
)
from cosyvoice.utils.train_utils import (
    check_modify_and_save_config,
    init_dataset_and_dataloader,
    init_distributed,
    init_summarywriter,
    save_model,
    wrap_cuda_model,
)
from cosyvoice.utils.train_utils_emo import (
    EMOFILM_TRAINABLE_MODULES,
    finalize_latest_checkpoint,
    freeze_all_except,
    init_optimizer_emo,
)
from cosyvoice.utils.scheduler import ConstantLR


def get_args():
    parser = argparse.ArgumentParser(description="train basic Emo-FiLM SFT")
    parser.add_argument("--train_engine", default="torch_ddp", choices=["torch_ddp"])
    parser.add_argument("--model", default="llm", choices=["llm"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--cv_data", required=True)
    parser.add_argument("--qwen_pretrain_path")
    parser.add_argument("--onnx_path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--contract_dir", default="data/contracts/emofilm_v1")
    parser.add_argument("--seed", default=1986, type=int)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tensorboard_dir", default="tensorboard")
    parser.add_argument("--ddp.dist_backend", dest="dist_backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prefetch", default=100, type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--timeout", default=60, type=int)
    return parser.parse_args()


def _load_checkpoint_payload(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint must contain a state dict, got {type(state_dict).__name__}")
    model_state = {
        key: value
        for key, value in state_dict.items()
        if key not in {"epoch", "step"}
    }
    has_training_metadata = "epoch" in state_dict or "step" in state_dict
    return state_dict, model_state, has_training_metadata


def load_resume_checkpoint(model, checkpoint_path):
    """严格加载训练中间 checkpoint，并读取恢复进度。"""
    state_dict, model_state, _ = _load_checkpoint_payload(checkpoint_path)
    load_trained_state(model, model_state)
    return int(state_dict.get("step", 0)), int(state_dict.get("epoch", -1))


def load_training_checkpoint(model, checkpoint_path):
    """按 checkpoint 形态区分基座初始化和训练恢复。

    预训练 `llm.pt` 不带训练元数据，只允许新版情感模块缺失；训练产生的
    `latest.pt`/`final.pt` 带有 epoch 或 step，必须按完整模型 strict load。
    """
    state_dict, model_state, is_resume = _load_checkpoint_payload(checkpoint_path)
    if is_resume:
        load_trained_state(model, model_state)
        return int(state_dict.get("step", 0)), int(state_dict.get("epoch", -1)), "resume"
    load_base_state(model, model_state)
    return 0, -1, "base"


def checkpoint_is_resume(checkpoint_role):
    """Return whether checkpoint loading should preserve an existing run."""
    return checkpoint_role == "resume"


def write_resolved_config(output_path, *, config_path, args, train_conf):
    """保存本次训练实际使用的 CLI 参数和训练配置。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_source": str(Path(config_path).resolve()),
        "arguments": vars(args).copy(),
        "train_conf": deepcopy(dict(train_conf)),
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output)
    return payload


def write_training_identity(
    output_path,
    *,
    model,
    code_root,
    contract_dir,
    command,
    seed,
    base_checkpoint=None,
    resolved_config,
    checkpoint_role,
):
    """记录训练身份、基座权重和当前模型参数哈希。"""
    from tools.write_emofilm_run_identity import write_run_identity

    return write_run_identity(
        output_path,
        run_kind="train",
        code_root=code_root,
        contract_dir=contract_dir,
        command=command,
        seed=seed,
        base_checkpoint=base_checkpoint,
        extra={
            "checkpoint_role": checkpoint_role,
            "parameter_hash": hash_model_state(model),
            "resolved_config": str(Path(resolved_config).resolve()),
        },
    )


def update_training_identity(identity_path, *, model, final_checkpoint):
    """将 final checkpoint 与最终模型参数哈希写回训练身份。"""
    import json

    from tools.write_emofilm_run_identity import sha256_file

    path = Path(identity_path)
    identity = json.loads(path.read_text(encoding="utf-8"))
    final_path = Path(final_checkpoint).resolve()
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    identity.setdefault("extra", {})["final_parameter_hash"] = hash_model_state(model)
    identity["extra"]["final_checkpoint"] = {
        "path": str(final_path),
        "sha256": sha256_file(final_path),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return identity


def finalize_checkpoint_on_rank_zero(model_dir, rank=None, model=None):
    """仅由 rank 0 收口 final.pt，并让其它 rank 等待完成。"""
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
    finalized = False
    if rank == 0:
        latest_path = os.path.join(model_dir, "latest.pt")
        final_path = os.path.join(model_dir, "final.pt")
        if os.path.isfile(latest_path):
            finalize_latest_checkpoint(model_dir)
            finalized = True
        elif not os.path.isfile(final_path):
            raise FileNotFoundError(latest_path)
        if model is not None:
            identity_path = os.path.join(model_dir, "train_identity.json")
            if not os.path.isfile(identity_path):
                raise FileNotFoundError(identity_path)
            update_training_identity(
                identity_path,
                model=model,
                final_checkpoint=final_path,
            )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return finalized


def save_init_checkpoint_if_new(model, info_dict, resumed):
    """只为新运行创建 init.pt，恢复运行不得覆盖初始权重。"""
    if resumed:
        return False
    init_path = os.path.join(info_dict["model_dir"], "init.pt")
    if os.path.exists(init_path):
        raise FileExistsError(init_path)
    save_model(model, "init", info_dict)
    return True


@record
def main():
    args = get_args()
    if args.onnx_path is not None:
        os.environ["onnx_path"] = args.onnx_path
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    override_dict = {name: None for name in ("llm", "flow", "hift", "hifigan") if name != args.model}
    if args.qwen_pretrain_path is not None:
        override_dict["qwen_pretrain_path"] = args.qwen_pretrain_path
    with open(args.config, "r", encoding="utf-8") as config_file:
        configs = load_hyperpyyaml(config_file, overrides=override_dict)
    configs["train_conf"].update(vars(args))

    init_distributed(args)
    train_dataset, cv_dataset, train_loader, cv_loader = init_dataset_and_dataloader(
        args, configs, gan=False, dpo=False
    )
    configs = check_modify_and_save_config(args, configs)
    writer = init_summarywriter(args)

    model = configs[args.model]
    freeze_all_except(model, EMOFILM_TRAINABLE_MODULES)
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        start_step, start_epoch, checkpoint_role = load_training_checkpoint(model, args.checkpoint)
    else:
        checkpoint_role = "fresh"

    model = wrap_cuda_model(args, model)
    optimizer = init_optimizer_emo(model, configs)
    scheduler = ConstantLR(optimizer)
    scheduler.set_step(start_step)

    info_dict = deepcopy(configs["train_conf"])
    info_dict["step"] = start_step
    info_dict["epoch"] = start_epoch
    info_dict["checkpoint_lifecycle"] = "latest_final"
    model_dir = Path(args.model_dir)
    resolved_config = model_dir / "resolved.yaml"
    identity_path = model_dir / "train_identity.json"
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
    if rank == 0:
        if not resolved_config.exists():
            write_resolved_config(
                resolved_config,
                config_path=args.config,
                args=args,
                train_conf=info_dict,
            )
        if not identity_path.exists():
            write_training_identity(
                identity_path,
                model=model,
                code_root=Path(__file__).resolve().parents[2],
                contract_dir=args.contract_dir,
                command=" ".join(["torchrun", *sys.argv]),
                seed=args.seed,
                base_checkpoint=args.checkpoint if checkpoint_role == "base" else None,
                resolved_config=resolved_config,
                checkpoint_role=checkpoint_role,
            )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    save_init_checkpoint_if_new(
        model,
        info_dict,
        resumed=checkpoint_is_resume(checkpoint_role),
    )

    executor = Executor(gan=False)
    executor.step = start_step
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    for epoch in range(start_epoch + 1, info_dict["max_epoch"]):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoc(
            model, optimizer, scheduler, train_loader, cv_loader, writer,
            info_dict, scaler, group_join,
        )
        dist.destroy_process_group(group_join)

    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
    finalize_checkpoint_on_rank_zero(args.model_dir, rank=rank, model=model)


if __name__ == "__main__":
    main()
