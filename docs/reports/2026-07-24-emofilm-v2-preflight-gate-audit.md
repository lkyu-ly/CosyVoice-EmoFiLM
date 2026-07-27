# EmoFiLM v2 13/14 启动前门禁核查：标注器合同与声学 evaluator

日期：2026-07-24
范围：只读核查 v1 canonical 资产、v2 工作树、作者公开材料与已有票据；未修改业务代码、模型、数据或既有产物。

## 结论摘要

1. `checkpoints/word_sequence_model/best.pt` 与当前 768d `word_blocks` 不兼容这一观察是正确的，但它被错误地当成了“缺少兼容 checkpoint”。兼容的作者 checkpoint 已在本机，路径为 `checkpoints/word_sequence_model/author_best_model.pth`；它与 `reference/Emo_PA_code_data/annotate_data/best_model.pth` SHA-256 完全相同，且可对真实 768d 词块 strict-load 并推理。
2. v2 IEMOCAP 小样本确实是 `StubPredictor` 产物，不能作为训练真值。真正需要的后续数据运行是：用现有作者 768d/3-VAD checkpoint 重建 v2 span manifest；不是恢复 checkpoint，也不是重建 1024d word blocks。
3. 当前 canonical 主线的 IEMOCAP 词级链路不是 1024d/plus-large。作者公开链路和 v1 canonical 合同都是 `emotion2vec-base` 768d + 作者 `WordSequenceModel`（5 类、3D VAD）。1024d/plus-large 是历史本地词级路线，并保留为 FEDD 全局标签/一致性检查等不同职责。
4. 第二项门禁在“正式、独立的局部控制结论”这一含义下仍未满足；但“emotion2vec 是 utterance-level”这一表述不准确。emotion2vec 可提供帧特征，缺的是具有有效训练语义、校准和独立性的 emotion/arousal 判别器。当前真实评测 CLI 还缺少模块入口调用，不能作为正式运行入口。

## 1. 第一项争议：资产、合同与责任划分

| 断言 | 结论 | 证据 |
|---|---|---|
| `best.pt` 不能用于当前 768d word blocks | 正确 | `best.pt` 的 attention 输入为 1024，分类头为 `(5,1024)`，回归头为 `(1,1024)`；真实词块形状为 `(T,768)`。 |
| 因此必须“恢复 768d/3-VAD checkpoint” | 错误 | `author_best_model.pth` 已存在，SHA-256 为 `a4b373...31afe13`，与作者随包 `best_model.pth` 完全一致。 |
| 因此必须重建 1024d word blocks | 错误（在当前 canonical 合同下） | 只有主动切回历史 1024d/1-arousal 路线时才成立；当前 v1/v2 作者式合同不选择该路线。 |
| 当前 v2 IEMOCAP manifest 可用于训练 | 错误 | `data/contracts/emofilm_v2/sources/iemocap/tagged.jsonl` 的 18 行均标记 `predictor_class=StubPredictor`、`checkpoint=<stub>` 和全零 checkpoint hash。 |

### 1.1 当前的两套不可互换合同

| 路线 | 特征与模型 | 当前职责 |
|---|---|---|
| 作者/canonical 词级链路 | emotion2vec-base，768d/50 Hz；作者 WordSequenceModel，5 类 + 3D VAD | IEMOCAP 词级弱监督与 v1 canonical 数据合同 |
| 历史本地链路 | emotion2vec-plus-large，1024d；本地 `best.pt`，5 类 + 1D arousal | 历史证据；plus-large 还用于 FEDD 的整句全局标签/一致性检查 |

作者公开的 `annotate_data/run_ncssd_annotation.py` 传入 `emotion2vec_base.pt`；`model.py` 默认 `input_dim=768`，回归头为 3 维。`docs/reports/2026-07-17-emofilm-global-vs-word-annotation.md` 已明确区分这条词级链路与 plus-large 的 FEDD 整句用途。2026-07-20 的 canonical closeout 也把活跃入口固定为 `author_best_model.pth`；README 和 `tests/test_emofilm_data_contract.py` 使用同一路径。

因此，若“1024d/plus-large 被重新用于作者式 IEMOCAP 词级数据重建”是对当前主线的记忆，则该记忆与作者源码、canonical 计划、正式默认入口和实际资产均不一致。较可能的混淆是把历史本地 1024d 路线或 FEDD 全局检查职责，与作者回归后的 768d 词级链路混在了一起。

### 1.2 可重复的最小核验

对真实词块 `Ses01F_impro01_F000/0000_24_40.pt` 的只读推理结果：

- 词块：`frames.shape=(16,768)`；
- `author_best_model.pth`：strict-load 通过，输出 `(1,5)` emotion 与 `(1,3)` VAD；
- `best.pt`：strict-load 失败，报 1024/768、1/3 等 state-dict shape mismatch。

额外用作者 checkpoint 在内存中重建了 3 条已冻结 v1 tagged 文本，均与存储内容逐字节一致：

- `Ses01F_impro01_F000`（2 个词块）；
- `Ses04M_impro03_M002`（5 个词块）；
- `Ses05M_script03_2_M000`（14 个词块）。

`tools.generate_v2_tagged_jsonl.WordSequenceModelPredictor(author_best_model.pth)` 也已直接对真实词块返回 5 维 soft distribution 与 3 维 VAD。因此 v2 生成器的作者合同路径本身可用。

### 1.3 根因和影响

ticket 02 的报告和示例命令把 `--checkpoint` 指向了 `checkpoints/word_sequence_model/best.pt`。这正好选中了历史 1024d checkpoint；随后用 Stub 规避其失败。v2 CLI 本身要求显式 checkpoint，并没有强制该错误路径。

当前真实缺口是一个未执行的数据产物步骤：v2 的 IEMOCAP span 文件仍是 schema-demo 小样本，必须在进入 13 前用既有作者 checkpoint 重新生成完整、可追溯的 v2 产物。它不需要下载模型、恢复 checkpoint、重训标注器或重提取 1024d 特征。

## 2. 第二项争议：独立声学 evaluator 的真实状态

### 2.1 应校正的术语

`emotion2vec` 不能笼统称为“utterance-level”：

- 本机 `emotion2vec-base` provenance 的 loader smoke 已记录 `(204,768)` 帧特征；
- `tools/extract_emotion2vec_frame.py` 是 768d/50 Hz 帧特征提取器；
- `tests/smoke_test_emotion2vec.py` 也明确覆盖 plus-large 的 `granularity="frame"` 1024d 特征序列。

真正做时间池化的是 `WordSequenceModel`：attention/FFN 后对其输入序列求均值，再接分类与 VAD 头。作者流水线把它用于一个 MFA 词块；v2 的 `Emotion2Vec*Evaluator` wrapper 则把裸 `classification_head` / `regression_head` 直接施加到未经该模型前向的单帧特征上。这超出训练分布，不能当作经校准的逐帧判别器。

### 2.2 门禁结论

严格门禁仍应保持 **未满足**，理由是：

1. 当前没有独立、经校准的 emotion 分布轨迹或 arousal 轨迹模型。现有候选只有 emotion2vec 特征提取器及与 IEMOCAP 弱监督同源的 WordSequenceModel。
2. 没有针对真实参考音频完成类别映射、转场定位误差和 arousal 方向/校准的验收；ticket 08 报告也明确记录真实 smoke 未运行。
3. `eval/eval_emofilm_v2.py` 对“真实 emotion2vec”路径显式抛出 `gate NOT MET`。而该文件末尾没有 `if __name__ == "__main__": main()`，所以 `python -m eval.eval_emofilm_v2 ...` 会安静退出，并不会进入这条门禁或 fake 路径。接口/合成测试可用，不等于正式 CLI 已可运行。

“需要外部独立 frame-level 分类器 + arousal 回归器”方向正确，但措辞过强：ticket 08 接受可审计的“逐帧或滑窗”输出，而非只接受原生逐帧模型。能满足正式门禁的最小候选应当同时满足：与训练任务头隔离、来源与 IEMOCAP 弱监督可区分、标签映射已验证、转场定位已验证、arousal 至少通过方向性与适用范围校准，并把版本/窗口/限制写入 identity。仅下载另一个模型不足以解锁门禁。

作者 `eval/evaluate.py` 已表明可以按 ASR 对齐的词/段把 768d 特征送入完整 WordSequenceModel，得到局部段级诊断；但它与弱监督标签同源，且不是独立校准 evaluator。它最多可作为明确标记“自证风险”的作者兼容诊断，不能支撑 v2 的独立 span、边界和强度结论。

## 3. 对 13/14 的准确影响

| 阶段 | 当前状态 | 启动前需要解决的事实 |
|---|---|---|
| 13 正式训练 | v2 IEMOCAP 小样本为 Stub，不能用于真实训练 | 用已有 `author_best_model.pth` 生成真实 v2 IEMOCAP span/数据产物；不需重建 1024d blocks。 |
| 14 正式局部评测 | local-control evaluator 门禁未满足 | 接入并校准独立的逐帧或滑窗 emotion/arousal evaluator；同时补齐真实 evaluator 的可执行入口。 |
| 09/10/12 的代码合同测试 | 已可用 | Fake 证明合同、配对、统计和失败语义；不证明真实音频上的局部控制效果。 |

没有独立 evaluator 时，历史 WER/Emo-SIM/DTW 的整体质量观察仍可与 v1 分开讨论；但不得把它们或同源 WordSequenceModel 的结果升级为“v2 已验证词级控制/强度/边界”的证据。

## 4. 新鲜验证记录

本次只读核查运行：

```text
27 passed in 2.27s
```

覆盖 `test_emofilm_word_sequence_checkpoint_is_768_5_3`、`test_emofilm_word_sequence_state_dict_shapes_match_model_definition` 和 `tests/test_acoustic_evaluators.py`。另完成了上文所列真实词块 strict-load、v1 tagged 文本重现、v2 predictor 直调和 evaluator gate/CLI 入口检查。

## 5. 本次边界

本报告不改写 v2 manifest、不切换 checkpoint、不下载 evaluator、不运行训练或生成，也不修改任何业务代码。后续若要执行 13/14，应先按本报告分别处理数据产物运行和 evaluator 选择/校准两个独立决策。
