# EmoFiLM v1 对 v2 13/14 门禁问题的影响核查

日期：2026-07-24
范围：只读核验已冻结的 `emofilm_v1` 合同、训练/生成/评测身份、IEMOCAP 标注产物，以及 v2 门禁报告；未改动代码、模型、数据或既有实验产物。

## 结论

**此前已经跑通全流程的 EmoFiLM v1 不受 v2 第一项门禁问题影响，不需要因该问题重建 1024d word blocks、重标 v1 数据、重训 v1 或重跑其 2500 条生成。**

v2 的问题是其 IEMOCAP 小样本 manifest 被 `StubPredictor` 生成，且 ticket 02 的示例误把历史 `checkpoints/word_sequence_model/best.pt` 当成当前词级标注器。这个 `best.pt` 确实是 1024d/5 类/1D arousal 的历史本地路线；但 v1 的真实词级监督链路是 `emotion2vec-base` 768d 帧特征加作者的 5 类/3D VAD checkpoint `author_best_model.pth`。该作者 checkpoint 已存在，且与作者随包文件逐字节相同。

因此，原先“必须恢复 768d checkpoint 或重建 1024d word blocks”的二选一对 **v2 当前 stub 产物**不成立；对 v1 更不成立。正确的后续数据步骤是：若继续 v2，使用已经存在的作者 checkpoint 从 v1 的 768d word blocks 生成真实的 v2 span/manifest 产物。

## 对两个容易混淆路线的判定

| 路线 | 特征与下游模型 | 在 v1 中的职责 | 是否是 v1 IEMOCAP 词级训练监督 |
|---|---|---|---|
| 作者/canonical | `emotion2vec-base`，768d/50 Hz；作者 `WordSequenceModel`，5 类 + 3D VAD | 产生 IEMOCAP 词级伪标签 | 是 |
| 历史本地 | `emotion2vec-plus-large`，1024d；本地 `best.pt`，5 类 + 1D arousal | 历史证据；FEDD 全局标签/一致性检查等独立职责 | 否 |

`emotion2vec-plus-large` 确实在 v1 周边出现过：FEDD 的全局标签/一致性检查，以及 `emofilm-eval-v2` 的 embedding 型 Emo-SIM/DTW 特征提取都使用它。它与 `best.pt` 的 1024d WordSequence 标注器不是同一件事，也不表示 v1 的 IEMOCAP word blocks 曾按 1024d 路线重建。

## 可重复证据

1. **作者 checkpoint 身份与结构**

   - `checkpoints/word_sequence_model/author_best_model.pth` 与 `reference/Emo_PA_code_data/annotate_data/best_model.pth` 的 SHA-256 同为 `a4b373501632d8dad2685253e8a82aa6744e727b0ae28c5b1a5dc482a31afe13`。
   - v1 的默认合同测试 strict-load 该 checkpoint，并确认 `(T, 768)` 输入输出 `(1, 5)` 情感 logits 和 `(1, 3)` VAD：`2 passed in 3.35s`。
   - 将同一测试显式指向 `best.pt` 会稳定失败（`1 failed in 2.81s`）：attention、FFN、分类头均为 1024/768 mismatch，回归头为 1/3 mismatch。这正是 v2 示例命令选错 checkpoint 的故障，而非 v1 的运行方式。

2. **v1 真实数据而非 Stub 的回放**

   用 `author_best_model.pth` 从 v1 保存的真实 word blocks 重算三个分布在全量 tagged 文件开头、中部、尾部的样本，均与冻结的 `data/contracts/emofilm_v1/sources/iemocap/tagged.jsonl` 完全一致：

   | `utt_id` | word blocks | 重放结果 |
   |---|---:|---|
   | `Ses01F_impro01_F000` | 2 | 完全一致 |
   | `Ses04M_impro03_M002` | 5 | 完全一致 |
   | `Ses05M_script03_2_M000` | 14 | 完全一致 |

   这三条带 emotion tags 的文本也在 v1 实际训练所读取的 parquet shard 中各出现一次。它排除了“当前文档已改为 768d、但正式 v1 训练仍吃了另一条 1024d 标注数据”的解释。

3. **正式运行链条一致**

   `artifacts/emofilm_v1/train/train_identity.json`、`generation/full_generation_identity.json` 和 `evaluation/evaluation_identity.json` 都绑定：

   ```text
   contract_name = emofilm_v1
   contract_hash = b5aab1428acae55bba37dcac5b5c6720b6d4c09b67d4321d3b8ccdc1db5a3e9d
   ```

   保存的正式 `final.pt` 存在；历史 `final_verification.json` 记录 strict-load 通过、train/cv 有限前向通过。全量生成 manifest 仍在，规模为 ESD 1500、FEDD-A 500、FEDD-B 500。

## 对 v2 门禁逐项的影响

| v2 门禁 | 对 v1 的影响 | 结论 |
|---|---|---|
| 1. Stub IEMOCAP manifest / 1024d `best.pt` 不兼容 | v1 使用的是另一条、已验证的 768d 作者链路 | **无影响；不重建** |
| 2. 独立 frame/sliding-window emotion + arousal evaluator 缺失 | v1 的 WER、Emo-SIM、DTW 是已声明的整体质量/代理指标，不是独立校准的 span、boundary、intensity 验收 | **不推翻已报告指标；限制可作出的结论** |
| 3. v2 真实 checkpoint + MFA + ASR GPU smoke | 属于尚未训练的 v2 运行门禁；v1 已有自身正式训练、GPU 生成和三分区评测记录 | **不影响 v1；不能借此替 v2 放行** |
| 4. v2 patch bundle/monotonicity 记账问题 | v2 开发过程与新指标的追溯问题 | **与冻结 v1 产物无关** |

## 必须保留的 v1 限定

这不是“v1 已经独立证明细粒度控制”的结论。v1 的 IEMOCAP 词级标签仍是由句级弱监督训练出的伪标签；v1 正式评测也只持久化分区 aggregate，使用 plus-large embedding 相似度/DTW 与 WER，并未校准独立的局部情感类别、转场边界或 arousal 轨迹。因此：

- v1 继续是有效、可追溯的整体质量基线，适合与 v2 做同口径的 WER/Emo-SIM/DTW 比较；
- v1 的历史结果不得被表述为已由独立 evaluator 验证的词级情感、强度或边界控制；
- v2 ticket 08 的 evaluator 门禁仍必须在 v2 正式局部结论之前满足，不能用 v1 的整体评测替代。

## 决策

对 v1：保持只读冻结状态，**不采取数据或模型修复动作**。
对 v2：修正 checkpoint 选择并生成真实 IEMOCAP span 产物；独立 evaluator 门禁仍是单独、尚未满足的前置条件。
