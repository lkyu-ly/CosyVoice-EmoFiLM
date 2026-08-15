# EmoFiLM v3 全量中性基线（longepoch + sentlvl）

- 评测日期：2026-08-04
- 关联计划：`docs/superpowers/plans/2026-08-04-emofilm-eval-prereqs.md`（Task 4）
- 评测契约：`docs/contracts/emofilm_v3_eval.md`（`emofilm-eval-v3`）
- 评测脚本：`exp/emofilm_film_only_longepoch/run_eval.sh`、`exp/emofilm_sentlvl/run_eval.sh`
- 资源：GPU 0（longepoch）/ GPU 2（sentlvl）并行；whisper=cuda、emotion2vec=cuda
- hyp 来源：复用各模型 `full/{esd,fedd_a,fedd_b}` 既有生成 wav（与 v2 baseline 同批，未重新生成）
- 评测耗时：约 52 分钟（两模型并行；esd 1500 条占大头，whisper large-v3 串行转写是瓶颈）

## 0. 一句话结论

中性声学 prompt 协议下，两模型 ESD Emo-SIM 稳定在 **65.45–65.89**（与 v2 中性基线**完全一致到小数点后 2 位**，印证 v3 口径一致 + 复用同批 hyp）；判别指标经 `reference_wav` 合并后**在 eval 集可用**（n_scored=1422，acc≈43%，远超 chance≈22%）——判别增强（Task 1）生效。平台瓶颈仍在协议（声学钳制），与上轮 prompt_match 结论一致（情感匹配 prompt 下 Emo-SIM 跳到 84–87、5-way acc 73–83%）。

## 1. v3 主指标（emo_sim / dtw_normalized / wer / per_emotion_emo_sim）

### 1.1 ESD（1500 条）

| 模型 | emo_sim | dtw_norm | wer% | neu | ang | hap | sad | sur |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longepoch | 65.89 | 0.3409 | 7.43 | 93.72 | 61.40 | 58.58 | 69.63 | 46.12 |
| sentlvl | 65.45 | 0.3453 | 9.12 | 94.21 | 59.28 | 60.37 | 68.48 | 44.92 |

per_emotion 模式：**neu 最高（~94）**（生成用中性 prompt → 与 neu 参考最相似，声学钳制直接证据），sur 最低（~45）；情感强度排序两模型一致。sentlvl 的 WER（9.12）略高于 longepoch（7.43），与历史观察一致。

### 1.2 FEDD_A（500 条）/ FEDD_B（500 条）

| 模型 | 数据集 | emo_sim | dtw_norm | wer% |
| --- | --- | --- | --- | --- |
| longepoch | fedd_a | 82.46 | 0.1732 | 4.67 |
| longepoch | fedd_b | 64.31 | 0.3567 | 10.33 |
| sentlvl | fedd_a | 82.91 | 0.1687 | 6.52 |
| sentlvl | fedd_b | 63.19 | 0.3679 | 12.64 |

fedd_a 的 emo_sim（~82–83）显著高于 fedd_b（~63–64）与 ESD（~65）：fedd_a 参考为目标情感同说话人音频（同情感匹配度高），fedd_b 参考为中性音频（与 ESD 同结构）。两模型 fedd_b 的 WER（10.33/12.64）均高于 fedd_a，与 sentlvl 在 ESD 上 WER 更高的趋势一致。

## 2. 判别指标（reference_wav 合并的直接结果）

判别指标本次在 eval 集上**可算**（上轮因 train/eval 划分导致 target 缺失而 N/A）；Task 1 合并 `reference_wav`（目标情感真实音频）后，候选包含"正确答案"。same/cross/gap 为原始余弦，acc 为百分比。

### 2.1 ESD（1500）

| 模型 | n_valid | n_skipped | n_scored | n_way_avg | acc% | same | cross | gap | n_way 分布 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longepoch | 1422 | 78 | 1422 | 4.48 | 43.04 | 0.66 | 0.50 | 0.16 | 5-way×851 / 4-way×405 / 3-way×166 |
| sentlvl | 1422 | 78 | 1422 | 4.48 | 42.33 | 0.65 | 0.50 | 0.15 | 5-way×851 / 4-way×405 / 3-way×166 |

- chance ≈ 1/4.48 ≈ 22%；两模型 acc≈43% **显著高于 chance**，判别信号有效（中性 prompt 下模型仍能区分情感，只是 Emo-SIM 被声学钳制压在 65 平台）。
- n_scored=1422 与数据层独立验证（合并后候选≥3 的行数）**精确匹配**；78 条跳过（候选<3，诚实口径）。
- mean_sim_by_ref_emotion（参考情感→与所有 hyp 的均值余弦）：neu≈0.79 最高、sur≈0.31 最低 —— 与 per_emotion 模式一致（中性参考与中性 prompt 生成音频最相似）。

### 2.2 FEDD（500 ×2）

| 模型 | 数据集 | n_valid | n_scored | n_way_avg | acc% | same | cross | gap | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longepoch | fedd_a | 0 | 0 | — | — | — | — | — | no reference groups with >=3 emotions |
| longepoch | fedd_b | 427 | 427 | 3.62 | 50.12 | 0.64 | 0.50 | 0.15 | — |
| sentlvl | fedd_a | 0 | 0 | — | — | — | — | — | no reference groups with >=3 emotions |
| sentlvl | fedd_b | 427 | 427 | 3.62 | 46.60 | 0.63 | 0.50 | 0.13 | — |

- **fedd_a 判别 n=0**：Part A 无同文本跨情感参考组，候选恒<3，诚实返回 reason（不写 NaN、不报误导 0%）。
- **fedd_b 判别可算（n_scored=427）**：Part B 含同文本跨情感组（264 条 4-way + 163 条 3-way，73 条跳过），acc≈47–50%（chance≈1/3.62≈28%，显著高于 chance）。这是本次评测的额外收获——计划撰写时假设 FEDD 全集判别 n=0，实际 fedd_b 提供有效判别信号。

## 3. 关键观察

1. **判别增强生效**：Task 1 的 `reference_wav` 合并使 eval 集判别从 N/A 变为可算（ESD n_scored=1422、FEDD_B n_scored=427），`reason` 仅在真无参考组时出现（fedd_a）。这是本次收尾的核心交付。
2. **v3 口径一致**：ESD emo_sim（65.89/65.45）与 v2 中性基线（上轮报告 §1.2：longepoch 65.89 / sentlvl 65.45）完全一致到 2 位小数，确认 v3 评测 emo_sim 口径未漂移、复用同批 hyp。
3. **平台瓶颈定位不变**：中性 prompt 下 Emo-SIM 65 平台 + 判别 acc 显著高于 chance → 模型能区分情感但声学被中性 prompt 钳制。下一步主线（R2 监督改造）应把"情感匹配 prompt 能达成的 84–87 上限"内化到模型自身。
4. **两模型差异小**：longepoch 与 sentlvl 在所有数据集上 emo_sim/判别均接近（sentlvl WER 略高），与上轮"句级监督未突破平台"结论一致（参见 `docs/reports` 句级监督对照实验）。

## 4. 产物路径

- 指标 JSON：`exp/emofilm_film_only_longepoch/eval/v3/{esd,fedd_a,fedd_b}_metrics.json`、`exp/emofilm_sentlvl/eval/v3/{esd,fedd_a,fedd_b}_metrics.json`
- 评测日志：同目录下 `{esd,fedd_a,fedd_b}.log`
