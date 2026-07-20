# EmoFiLM v1

EmoFiLM v1 是基于 CosyVoice2-0.5B 的情感语音合成基线。本仓库根目录是唯一受支持的运行入口；数据合同、模型与正式实验资产均使用下列相对路径。

## 目录约定

- `data/contracts/emofilm_v1/`：版本化数据合同；文本 manifest/meta 可跟踪，大型派生资产不进入 Git。
- `datasets/`：ESD 与 IEMOCAP 原始数据。
- `pretrained_models/`：CosyVoice2 与 emotion2vec 运行模型。
- `checkpoints/word_sequence_model/author_best_model.pth`：活跃 WordSequence checkpoint。
- `exp/emofilm_v1/`：完整正式训练、生成和评测产物。
- `artifacts/emofilm_v1/`：Git 跟踪的正式实验摘要。

## 环境

从仓库根目录执行：

```bash
source scripts/activate_env.sh
```

可通过 `CONDA_ROOT` 和 `CONDA_ENV` 覆盖默认 Conda 安装位置与环境名。

## 数据合同检查

```bash
python -m pytest tests/test_emofilm_data_contract.py -q
test -f data/contracts/emofilm_v1/splits/train/parquet/data.list
test -f data/contracts/emofilm_v1/splits/cv/parquet/data.list
```

## 训练

训练必须从 CosyVoice2 预训练语音 LM `llm.pt` 初始化。`CosyVoice-BlankEN` 仅提供 Qwen tokenizer/配置，不能替代 base checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:third_party/Matcha-TTS \
torchrun --standalone --nproc_per_node=1 \
  cosyvoice/bin/train_emo.py \
  --train_engine torch_ddp \
  --model llm \
  --config conf/emo_film.yaml \
  --train_data data/contracts/emofilm_v1/splits/train/parquet/data.list \
  --cv_data data/contracts/emofilm_v1/splits/cv/parquet/data.list \
  --qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --checkpoint pretrained_models/CosyVoice2-0.5B/llm.pt \
  --contract_dir data/contracts/emofilm_v1 \
  --model_dir exp/emofilm_v1 \
  --tensorboard_dir exp/emofilm_v1/tb \
  --num_workers 2 \
  --prefetch 2
```

## 三样本推理

以下命令分别生成 ESD、FEDD-A、FEDD-B 各一条，用于迁移后 smoke；不会覆盖正式 full 输出。

```bash
for split in esd fedd_a fedd_b; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:third_party/Matcha-TTS \
  python tools/inference_emo_film.py \
    --model_dir pretrained_models/CosyVoice2-0.5B \
    --llm_ckpt exp/emofilm_v1/final.pt \
    --test_manifest "data/contracts/emofilm_v1/eval/${split}/manifest.jsonl" \
    --esd_root datasets/ESD \
    --workspace_root . \
    --output_dir "exp/emofilm_v1/canonical_smoke/${split}" \
    --device cuda --fp16 --max_samples 1
done
```

## 评测入口

正式评测使用 `eval/eval_emo_film.py`。`exp/emofilm_v1/eval_refs/` 是不纳入迁移的可重建视图；运行前须按对应合同 manifest 的 `utt_id` 与 `reference_wav` 重建同名引用 WAV/符号链接。例如 ESD：

```bash
CUDA_VISIBLE_DEVICES=0 MODELSCOPE_OFFLINE=1 PYTHONPATH=.:third_party/Matcha-TTS \
python eval/eval_emo_film.py \
  --ref_dir exp/emofilm_v1/eval_refs/esd \
  --hyp_dir exp/emofilm_v1/full/esd \
  --ref_text_manifest data/contracts/emofilm_v1/eval/esd/manifest.jsonl \
  --output exp/emofilm_v1/eval/esd_metrics.json \
  --device cuda --expected_count 1500 --batch_size 16
```

已完成基线的命令、identity、合并 manifest、metrics、比较结果和实验报告见 `artifacts/emofilm_v1/`；完整 checkpoint、WAV 与日志见 `exp/emofilm_v1/`。
