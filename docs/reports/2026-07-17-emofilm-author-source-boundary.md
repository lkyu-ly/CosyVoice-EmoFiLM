# 基础 Emo-FiLM 作者源码公开边界核验

日期：2026-07-17

范围：只讨论论文的基础 Emo-FiLM，不讨论 PA、GRPO、reward 或其他 RL 代码。

## 结论

`reference/Emo_PA_code_data` 不是“从原始数据到论文表格”的基础 Emo-FiLM 完整复现包。它的顶层 README 明确将自身描述为 **Emo-TTS GRPO + NCSSD** 工程；其中携带了基础 Emo-FiLM 的核心模型、通用 SFT 训练框架、词级标注推理流水线和推理运行时，但没有完整公开基础论文的数据构建、标注器训练、正式 ESD/FEDD 主评测集和论文主指标实现。

一句话概括作者公开边界：

> 作者公开了“核心算法代码和部分运行器”，没有公开“基础论文从原始数据、训练样本、正式推理清单到论文表格的完整闭环”。

## 逐环节边界

| 论文实现环节 | 状态 | 作者材料中实际存在的内容 | 关键缺口 |
|---|---|---|---|
| IEMOCAP、ESD 原始数据获取/清洗/索引 | 未公开 | 无基础数据准备入口 | 下载、过滤、标签映射、真实 manifest 均缺失 |
| ESD 训练/验证/测试切分 | 未公开 | 论文只说明测试集为 10 speaker × 5 emotion × 30，共 1500 条 | 具体 utterance ID、随机种子、训练和验证清单缺失 |
| 词级情感预测模型结构 | 已公开 | `annotate_data/model.py` 的 `WordSequenceModel`：768 维输入、5 类分类头、3 维回归头 | 无结构层面的主要缺口 |
| 词级情感预测模型权重 | 部分公开 | `annotate_data/best_model.pth` 存在且非空 | 其训练数据版本、运行配置和可追溯性缺失 |
| 词级情感预测模型训练 | 未公开 | 没有标注器训练入口 | IEMOCAP+ESD 样本构造、切分、loss 权重、3 epoch 训练命令和日志均缺失 |
| emotion2vec + MFA + SER 词级打标 | 部分公开 | `pipeline_word_emotion.py`、`generate_tagged_jsonl.py` 和 `run_ncssd_annotation.py` | 随包的 `emotion2vec_base.pt` 是 0 字节；还依赖外部 MFA 模型/词典和原始数据；公开编排入口针对 NCSSD，不是基础 IEMOCAP 数据包 |
| `annotated_IEMOCAP` 构建结果 | 未公开 | 只有可用于打标的通用/NCSSD 流水线 | 作者训练实际使用的 tagged 数据、span 分布和 manifest 缺失 |
| ESD Global Label 训练数据准备 | 未公开 | 无基础 ESD 打包入口 | 训练样本、标签编码、排除测试样本的清单、parquet/data.list 缺失 |
| FEDD 构建 | 未公开 | 论文描述 1000 条、500 mild + 500 strong，并说明 plus-large 用于全局标签/一致性检查 | 原音频、文本、生成 prompt、拼接脚本、标签检查脚本、manifest 和哈希均缺失 |
| CosyVoice 训练样本打包 | 未公开（基础链） | 仓库有 processor/parquet reader，另有后续 NCSSD parquet | 从 annotated_IEMOCAP+ESD 到基础 SFT parquet、speech token、speaker embedding、data.list 的脚本和产物缺失 |
| Emo-FiLM 模型定义 | 已公开，较完整 | `cosyvoice/llm_emo/llm_emo.py`、emotion tokenizer、FiLM、emotion encoder/classifier、训练前向和作者推理逻辑 | 属研究原型，含注释旧实现和硬编码，但核心结构可审计 |
| 基础 SFT 训练引擎 | 已公开 | `cosyvoice/bin/train.py`、`utils/train_utils.py`、`utils/executor.py`、dataset processor、YAML | 通用引擎存在，不等于论文训练运行可复现 |
| 论文实际 SFT 训练运行 | 部分公开 | 冻结/解冻代码、Adam、batch、模型前向和 loss 可见 | 缺真实 train/cv 数据、命令、日志、随机状态；YAML `max_epoch: 200` 与论文 5 epoch 不一致 |
| 基础 Emo-FiLM 权重 | 另行发布 | 本地另存 `emofilm_original/Emo_FiLM_hf/model.safetensors` 和 `emotion_modules.pt` | 不在源码包中自包含；情感模块还需显式加载/转换 |
| 单条或 JSON 驱动推理 | 已公开 | `cosyvoice/cli/*emo*`、`model_emo.py`、`eval/inference.py` | 依赖外部模型资产；示例入口主要面向后续 PA Task A/B schema，且默认只跑少量样本 |
| 基础 ESD 1500 / FEDD 1000 正式批量推理 | 未公开 | 没有对应正式 manifest/driver | PA 的 500 条 Task-A/B JSON 不能替代基础论文主实验 population |
| Emo-SIM、DTW、WER 主评测 | 未公开 | `eval/evaluate.py` 计算 ACC-E 与 speaker similarity；Whisper 在其中主要用于词时间戳对齐 | 没有论文第 2 章主指标的正式聚合实现和主表生成代码 |
| EMOS、NMOS 人工听测 | 未公开 | 论文给出定义和结果 | 听测样本、随机化协议、问卷、原始打分和统计脚本缺失 |
| baseline、消融和论文图表复现 | 未公开 | 论文给出最终数值 | baseline 运行链、消融权重、生成音频和表图脚本缺失 |

## 一级证据

### 1. 仓库身份与数据边界

作者包的 [README](../../reference/Emo_PA_code_data/README.md) 标题为 `Emo-TTS GRPO`，数据流写的是 `NCSSD → annotation → parquet → GRPO`。因此：

- `data/parquet_ncssd/*` 是后续 NCSSD/GRPO 资产，不是基础 Emo-FiLM 的 `annotated_IEMOCAP + ESD` SFT 数据；
- `eval/data/esd-taska.json`、`esd-taskb.json` 各为 500 条，字段含后续 Task A/B 的 `tagged_text` 和绝对 prompt 路径，不是论文基础 ESD 1500 条主测试清单；
- 不能仅因目录名含 `ESD` 就把它认定为基础论文评测闭环。

### 2. 论文明确要求但仓库没有的基础数据链

论文源码 [main.tex](../../reference/arXiv-2509.20378v1/main.tex) §3.1–3.2 明确写到：

- ESD 测试为 1500 条；
- FEDD 为 1000 条，其中 500 mild、500 strong；
- 使用 IEMOCAP 和 ESD 全局标签训练词级预测模型；
- 基础 Emo-FiLM 用 Adam、batch size 4、训练 5 epoch。

仓库没有这些基础数据的真实 manifest、split、FEDD 构建脚本或 SFT parquet 生成入口。学位论文还说明标注器训练 3 epoch、TTS 训练 5 epoch，但相应的标注器训练程序仍未出现。

### 3. 标注链只公开了“模型 + 推理”，没有公开训练

- [model.py](../../reference/Emo_PA_code_data/annotate_data/model.py) 定义 `WordSequenceModel`；
- [pipeline_word_emotion.py](../../reference/Emo_PA_code_data/annotate_data/pipeline_word_emotion.py) 完成 emotion2vec 特征、MFA 词边界和 SER 推理；
- [generate_tagged_jsonl.py](../../reference/Emo_PA_code_data/annotate_data/generate_tagged_jsonl.py) 合并词级预测；
- [run_ncssd_annotation.py](../../reference/Emo_PA_code_data/annotate_data/run_ncssd_annotation.py) 是 NCSSD 编排入口。

`annotate_data` 中没有 DataLoader、optimizer、backward、epoch 构成的 `WordSequenceModel` 训练循环。`best_model.pth` 非空，但三个随包出现的 `emotion2vec_base.pt` 均为 0 字节，因此代码不是离线自包含运行包。

### 4. 模型和训练引擎公开，不等于论文训练运行公开

- [llm_emo.py](../../reference/Emo_PA_code_data/cosyvoice/llm_emo/llm_emo.py) 给出基础 Emo-FiLM 的 emotion encoder、FiLM、emotion classifier、训练前向和 inference；
- [train.py](../../reference/Emo_PA_code_data/cosyvoice/bin/train.py) 给出 checkpoint 加载、冻结/解冻、训练循环；
- [train_utils.py](../../reference/Emo_PA_code_data/cosyvoice/utils/train_utils.py) 给出 DataLoader、optimizer、scheduler 和 backward/update；
- [cosyvoice2.yaml](../../reference/Emo_PA_code_data/configs/cosyvoice2.yaml) 给出模型和 processor 配置。

但是 `train.py` 强制要求外部 `--train_data` 和 `--cv_data`，作者没有给出基础 SFT 对应文件。并且公开 YAML 的 `max_epoch: 200` 与论文/学位论文的 5 epoch 不一致，所以它不能作为论文实际运行配置的完整快照。

需要特别区分同目录的两个模型文件：当前 YAML 明确引用 `cosyvoice.llm_emo.llm_emo.Qwen2LM_Emotion`，`train.py` 也从 `llm_emo.py` 导入该类；`llm_emo_source.py` 没有被当前 YAML 或正式入口引用，是另一套把 FiLM 插入 Qwen block 的历史/实验实现。审查作者实际运行路径时应以 `llm_emo.py` 为准，不能把两套代码拼接解释。

### 5. 当前 eval 不是基础论文 Emo-SIM/DTW/WER 实现

[eval/evaluate.py](../../reference/Emo_PA_code_data/eval/evaluate.py) 文件头和最终输出都明确是 `ACC-E + SS`。它加载 Whisper 是为了得到 word timestamps 并按词段执行情感分类，不会汇总论文主表的 WER；文件中也没有 Emo-SIM 或 DTW 实现。论文 [main.tex](../../reference/arXiv-2509.20378v1/main.tex) 只描述指标概念，没有给出正式计算程序。

## 对本地复现结论的约束

1. 可以严格回归作者已公开的模型结构、token/标签解析、loss、冻结策略、processor 和推理调用逻辑。
2. 可以使用作者另行发布权重验证公开推理路径，但必须注明权重不属于源码包的自包含资产。
3. 不能声称已严格复现作者的 IEMOCAP/ESD 训练 population、`annotated_IEMOCAP`、原始 FEDD 或论文绝对指标，因为决定这些结果的 manifest、数据和正式评测实现没有公开。
4. 本地 `FEDD-rebuilt` 和自建评测实现可以用于固定合同下的相对比较，但不能改称作者原始 FEDD 或论文官方评测代码。

## 最终判定

基础 Emo-FiLM 作者材料的公开程度可归纳为：

- **模型：公开较完整。**
- **训练代码：引擎和核心逻辑公开，论文训练运行不完整。**
- **标注代码：推理流水线和一个 checkpoint 公开，标注器训练未公开。**
- **数据：基础训练集、正式 split、annotated_IEMOCAP、FEDD 均未公开。**
- **推理：核心运行时公开，基础论文正式批量清单和入口未公开。**
- **评测：后续 PA 的部分指标公开，基础论文主指标和听测闭环未公开。**
