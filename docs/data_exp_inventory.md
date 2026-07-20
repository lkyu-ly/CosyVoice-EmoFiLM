# EmoFiLM v1 canonical 资产清单

本清单只描述仓库根目录内的 canonical payload。统计日期为 2026-07-20；文件数与字节数来自 canonical checkout 当前落盘内容，不使用文件 hash。

## Git 跟踪资产

| 路径 | 角色 | 跟踪边界 |
|---|---|---|
| `README.md`、`CONTEXT.md` | 当前运行入口与领域语言 | 跟踪 |
| `docs/adr/` | ADR 0001–0018 | 跟踪 |
| `docs/superpowers/`、`docs/handoff/`、`docs/reports/` | 精选主线设计、计划、历史交接、审计与基线报告 | 跟踪 |
| `data/contracts/emofilm_v1/` | 数据合同 manifest、索引与 provenance | 文本 metadata 跟踪；大型二进制忽略 |
| `artifacts/emofilm_v1/` | 正式训练、生成、评测摘要 | 19 个文件：18 个历史原件加 1 个 canonical 路径映射 |

## 仓内非 Git 大型资产

| 路径 | 文件数 | 总字节数 | 角色 |
|---|---:|---:|---|
| `data/contracts/emofilm_v1/` | 108107 | 10671127580 | 完整数据合同；包含 23 个 parquet tar、派生特征、词块、对齐和文本 metadata |
| `datasets/` | 45083 | 4794580851 | ESD 与 IEMOCAP 原始数据 |
| `pretrained_models/` | 47 | 8690577028 | CosyVoice2 与 emotion2vec 运行模型 |
| `checkpoints/` | 4 | 91883736 | WordSequence checkpoint；活跃入口为 `checkpoints/word_sequence_model/author_best_model.pth` |
| `exp/emofilm_v1/` | 2568 | 5004380513 | 正式 5-epoch 训练、2500 条 full WAV、identity、metrics、日志及 3 条 canonical smoke 证据 |
| `emofilm_original/` | 36 | 10389129242 | 作者发布模型证据；非运行依赖；不含 Python 缓存 |
| `reference/` | 243 | 85233329 | 作者源码、论文与只读证据；非运行依赖；不含 Python 缓存 |

`exp/emofilm_v1/full/` 当前包含 ESD 1500、FEDD-A 500、FEDD-B 500，共 2500 条正式 WAV。

## 明确排除

- `.serena/`、`.scratch/`、`.superpowers/`、Python/test 缓存和编辑器临时文件。
- 中止的 `confirmation/`、可重建的 `eval_refs/` 视图，以及日志、分片 manifest、WAV、checkpoint 的 tracked 副本。
- 旧 stage、消融、debug、错误主线和训练故障排查文档；`docs/stage0_completion.md` 不属于白名单。

## Git 边界

`.gitignore` 整体忽略 `datasets/`、`pretrained_models/`、`checkpoints/`、`exp/`、`emofilm_original/`、`reference/`；在数据合同内仅忽略 PT/PTH/CKPT、WAV/FLAC、TextGrid、tar/parquet 等大型类型。合同 JSON、JSONL、list、scp、text、utt2spk 等文本 metadata 与 `artifacts/emofilm_v1/` 保持可跟踪。
