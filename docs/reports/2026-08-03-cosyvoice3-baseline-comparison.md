# CosyVoice3 官方基线 vs EmoFiLM 四方对比（2026-08-03）

**问题**：CV3（Fun-CosyVoice3-0.5B-2512）官方基线在 ESD 上是否突破此前的 Emo-SIM ~66 平台？

- **CV3**：ESD 150 条子集（5 情感 × 30），instruct 情感指令 + 中性 prompt，v3 评测。
- **四方 EmoFiLM**：历史 v2 全量 1500 条 ESD，中性 prompt（中性 prompt 全量评测口径）。

> 注意：CV3 用 instruct 情感指令 + 中性声学 prompt；EmoFiLM 四方用中性 prompt（各自训练协议）。
> 两者 prompt 口径相近（均为中性声学 prompt），可比 Emo-SIM 量级。

## 1. Emo-SIM / WER 总览

| 模型 | n | Emo-SIM | WER% |
|---|---|---|---|
| **CV3 官方基线** | 150 | **70.76** | 5.17 |
| v1 (中性, v2) | 1500 | 66.75 | 9.48 |
| film_only (中性, v2) | 1500 | 66.11 | 8.18 |
| longepoch (中性, v2) | 1500 | 65.89 | 8.38 |
| sentlvl (中性, v2) | 1500 | 65.45 | 10.05 |

## 2. CV3 per-emotion Emo-SIM

| ang | hap | neu | sad | sur |
|---|---|---|---|---|
| 71.14 | 65.91 | 91.72 | 66.05 | 58.98 |

## 3. CV3 n-way 判别

- n_valid=128 / n_skipped=22，但 **nearest-ref 准确率结构上不可计算**
- 原因：target emotion absent from all reference groups (eval/train split excludes target; use full 5-emotion groups for discriminability)
- ESD train/eval 划分把每个 (speaker,text) 组的目标情感单独留到 eval，sources 参考恰好缺该情感 →
  判别准确率只在**完整 5 情感组**（如 prompt_match 验证集）上才有意义，eval 集上应视为 N/A。

## 4. 结论

- **CV3 Emo-SIM=70.76**（中性 prompt + instruct 情感指令）。相对 EmoFiLM 四方 ~66 平台
  仅高 ~5 点，**未质变**——即便 CV3 拿到 emotion instruct 这个额外信号，中性声学 prompt 仍把它压在平台附近。
- 对照 prompt_match 实验（同情感声学 prompt 下四模型跳到 84–87、5-way acc 73–83%）：
  **真正解锁情感差异的是「声学 prompt 匹配目标情感」，而非 instruct 指令或更强的基座**。
- 综合判据（报告 §6.4 决策树）：~66 平台是生成/评测协议（固定中性 prompt）的产物；
  问题不在基座（CV3 也未突破）或 EmoFiLM 改造，而在生成时未把目标情感注入声学条件。
