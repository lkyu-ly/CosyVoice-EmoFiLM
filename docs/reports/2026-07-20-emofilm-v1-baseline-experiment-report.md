# EmoFiLM v1 正式训练、生成与评测报告

日期：2026-07-19 至 2026-07-20

## 运行结论

- 正式训练从 CosyVoice2 预训练语音 LM `llm.pt` 启动，身份为 `checkpoint_role=base`，完成 5 epoch 后正常退出。
- `final.pt` strict load 成功；train/cv 各一批前向的全部 loss 为有限值。
- 单条人工听测：文本内容清晰且完全正确；情感偏中性，用户确认不视为错误并授权继续。
- 固定生成集全部完成：ESD 1500、FEDD-A 500、FEDD-B 500，共 2500 条；ID 集合与冻结 manifest 完全一致，全部 WAV 可解码且为 24 kHz。
- `emofilm-eval-v2` 三分区评测正常退出。该指标面不设置自动通过阈值，也不据此自动续训。

## 可追溯身份

- 训练命令：`exp/emofilm_v1/training_command.txt`
- 训练身份：`exp/emofilm_v1/train_identity.json`
- 最终 checkpoint：`exp/emofilm_v1/final.pt`
- 最终 checkpoint SHA-256：`ac213f4257c60fe6d96e7def80090af4f6d1abe316e7d89cec9032ff464c7be7`
- 基座 checkpoint：`pretrained_models/CosyVoice2-0.5B/llm.pt`
- 基座 SHA-256：`b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608`
- 生成身份：`exp/emofilm_v1/full/full_generation_identity.json`
- 评测身份：`exp/emofilm_v1/eval/evaluation_identity.json`
- 对照数据：`exp/emofilm_v1/eval/comparison.json`

## 正式评测结果

| 分区 | N | WER | Emo-SIM | cosine DTW | cosine DTW normalized |
|---|---:|---:|---:|---:|---:|
| ESD | 1500 | 9.4770% | 66.7451 | 48.1758 | 0.332368 |
| FEDD-A | 500 | 8.2975% | 81.9401 | 79.2508 | 0.178268 |
| FEDD-B | 500 | 14.4211% | 61.5963 | 53.5767 | 0.383877 |

## 与旧 v6 local final 的同口径比较

两次运行使用完全相同的 1500/500/500 样本 ID，指标合同均为 `emofilm-eval-v2`。正值表示当前值更高；对 WER 和 normalized DTW，负值表示改善。

| 分区 | Emo-SIM Δ | normalized DTW Δ | WER 百分点 Δ | 观察 |
|---|---:|---:|---:|---|
| ESD | +11.6150 | -0.115844 | -18.3404 | 三项均改善 |
| FEDD-A | -2.0820 | +0.020088 | -7.7634 | 可懂度改善，情感相似度/DTW 略降 |
| FEDD-B | +2.4216 | -0.023976 | -21.9741 | 三项均改善 |

## 与旧 v6 local identity 的同口径比较

| 分区 | Emo-SIM Δ | normalized DTW Δ | WER 百分点 Δ | 观察 |
|---|---:|---:|---:|---|
| ESD | +15.5508 | -0.155422 | +3.1301 | 情感指标改善，WER 较差 |
| FEDD-A | -1.1289 | +0.011205 | +6.8807 | 三项均较差 |
| FEDD-B | +8.1616 | -0.081552 | +2.1146 | 情感指标改善，WER 略差 |

## 事实边界

- 本次正式评测入口持久化三份分区 aggregate JSON；当前入口没有持久化逐样本 metric rows，因此本报告不声称存在新的逐样本指标表。
- 训练期间曾出现一次 PyTorch allocator OOM 警告，但训练继续完成全部 5 epoch、返回码为 0，最终 checkpoint strict load 和有限前向均通过。
- 生成从单卡切换到 GPU 0–3 四卡固定分片。ESD 已有的 134 条 WAV 通过 `--skip_existing` 纳入四卡合并 manifest，未重复生成；该恢复事实已记录在生成身份中。
- 旧错误训练目录已按授权删除，不保留错误 checkpoint、日志或错误 smoke 产物。

## 当前主线状态

Task 5（正式 5-epoch 训练）和 Task 6（固定生成与本地代理评测）已完成。后续属于 Task 7：文档/库存收口、提出第二阶段精确清理清单，以及分别取得 commit、merge、push 授权。
