# 基础 Emo-FiLM 作者源码与本地实现对齐审计

**原始日期：** 2026-07-16

**重审日期：** 2026-07-18

**本地基线：** `CosyVoice-EmoFiLM/main@5ad481d`

**历史矩阵代码：** `feature/emofilm-author-model-control@e265281`
**范围：** 学位论文第 2 章基础 Emo-FiLM；排除 PA/RL、GRPO、reward 和 NCSSD 训练。

## 1. 结论

作者材料不是基础 Emo-FiLM 从原始数据到论文表格的一键复现包。公开包实际提供：

- 核心 Emo-FiLM 模型、tokenizer、训练前向和专用推理；
- 通用 CosyVoice SFT 训练引擎和一份公开 YAML 快照；
- WordSequenceModel 结构、作者 `best_model.pth` 和词级标注推理流水线；
- 后续 PA/GRPO 工程中的部分 JSON 推理和 ACC-E/SS 评测入口。

没有公开：

- 基础 IEMOCAP/ESD 获取、清洗、正式 train/cv/test manifest；
- WordSequenceModel 训练脚本和真实训练运行；
- `annotated_IEMOCAP`、基础 SFT parquet 和正式训练命令；
- 原始 FEDD、基础 ESD/FEDD 批量推理 population；
- 基础论文 Emo-SIM、DTW、WER、EMOS、NMOS 和表格生成实现。

因此本轮目标必须拆成三类：

1. **源码语义对齐：** 对齐作者公开且会改变训练或生成统计行为的逻辑。
2. **本地补全实现：** 保留作者未公开但本地已有合理、可追溯的实现。
3. **工程可靠性修复：** 修复本地资源清理、加载校验和运行身份问题，不冒充作者合同。

旧交付中“全链路已闭合”“10 任务 58 项证明覆盖完整”“唯一外部前置”等绝对表述不再成立；计数不能证明覆盖充分性。

## 2. 作者源码公开边界

| 论文环节 | 公开状态 | 实际边界 |
|---|---|---|
| IEMOCAP/ESD 原始数据和索引 | 未公开 | 无基础 manifest、过滤和标签映射代码 |
| ESD train/cv/test 切分 | 未公开 | 论文只给出 ESD test=1500 的统计定义 |
| WordSequenceModel 结构 | 已公开 | 768d 输入、5 类分类、3D VAD |
| WordSequenceModel checkpoint | 部分公开 | `best_model.pth` 非空，但训练来源不可追溯 |
| WordSequenceModel 训练 | 未公开 | 无 DataLoader、optimizer、loss 权重、split 或日志 |
| emotion2vec+MFA+SER 标注推理 | 部分公开 | 流水线存在；三个 base checkpoint 是 0 字节，MFA/模型/数据外置 |
| `annotated_IEMOCAP` | 未公开 | 无作者正式 tagged 数据和 span 分布 |
| 基础 SFT 数据打包 | 未公开 | 无 annotated_IEMOCAP+ESD 到 parquet 的入口和产物 |
| Emo-FiLM 模型 | 已公开，较完整 | 核心结构、loss、训练前向和专用推理可审计 |
| SFT 训练引擎 | 已公开 | 通用 train/executor/processor 存在 |
| 论文实际 SFT run | 部分公开 | 缺数据、命令、日志；公开 YAML 200 epoch 与论文 5 epoch 冲突 |
| 基础权重 | 另行发布 | `Emo_FiLM_hf` 情感模块需显式加载/转换 |
| 单条/JSON 推理 | 已公开 | 研究原型，依赖外部资产和后续 Task A/B schema |
| 基础 ESD/FEDD 正式批量推理 | 未公开 | PA Task A/B JSON 不能替代第 2 章 population |
| 基础主指标和听测 | 未公开 | 现有作者 eval 计算 ACC-E/SS，不是 Emo-SIM/DTW/WER 主表 |

另见独立边界报告：`docs/superpowers/reports/2026-07-17-emofilm-author-source-boundary.md`。

## 3. 必须对齐的源码语义

### 3.1 词级标注合同

作者公开链路是：

```text
emotion2vec-base frame features (768d, 50 Hz)
→ MFA word intervals
→ 作者 WordSequenceModel / best_model.pth
→ 5-class emotion + 3D VAD
→ arousal 分桶
→ emotion 与 intensity 均相同才合并
```

本地旧链路使用 plus-large 1024d、本地 5 类+1D checkpoint、三词平滑，并只按 emotion 合并，导致伪标签几乎退化为句级标签。新版必须：

- 外部取得官方有效 emotion2vec-base checkpoint，经作者 fairseq upstream 验证加载；作者随包 0 字节文件不可用；
- 使用作者 `annotate_data/model.py` 结构和 `best_model.pth`；
- 不训练 emotion2vec 或 WordSequenceModel；
- 不做三词多数投票或 arousal 滑动平均；
- 逐词先分桶 intensity，再按 `(emotion, intensity)` 合并；
- 只为 IEMOCAP 生成词级伪标签，ESD 保持数据集已知 Global Label。

作者没有公开标注器训练，因此不能把论文“训练 3 epoch”转换为本地重训任务。若新版标签出现可重复的空标签、文本丢失、类别异常塌缩或与明确数据标签大规模冲突，先分离 emotion2vec、MFA、WordSequence 推理和后处理；只有证据指向 WordSequenceModel 才另立修复设计。

### 3.2 模型前向与训练参数

高置信差异：

| 项目 | 作者公开行为 | 本地旧行为 | 裁决 |
|---|---|---|---|
| emotion loss 输入 | FiLM 后 `modulated_text_emb` | Qwen decoder text-position hidden | 对齐作者 |
| 可训练模块 | emotion encoder、FiLM adapter、LLM decoder | 额外训练 classifier | classifier 冻结 |
| token/mel | 按 1:2 联合裁剪 | 未联合裁剪 | 对齐作者 |
| batch | static batch 4 | dynamic frames | 对齐作者/论文共同口径 |
| filter | min speech=1、token max=9999 | min speech=100、token max=200 | 对齐公开配置快照 |
| epoch | 论文 5、公开 YAML 200 | 本地 5 | 保留 5，标为论文口径下的本地预算 |

作者 classifier 保持冻结却参与 loss，属于作者源码和论文直觉之间的冲突；作者 HF SFT 权重也支持 classifier 未更新。本轮按公开入口执行，不增加 classifier 消融。

### 3.3 神经网络结构与 checkpoint 形状

静态结构复核没有发现需要另起架构设计的隐藏差异，但确认了两类必须区分的问题：有效结构差异，以及只影响序列化兼容的无效参数差异。

| 子结构 | 作者有效结构 | 本地基线 | 裁决 |
|---|---|---|---|
| Qwen2 主干 | hidden=896、24 层、14 attention heads、2 KV heads、FFN=4864 | 同一 CosyVoice2-0.5B 主干规格 | 保留通用主干，不重建 Transformer |
| EmotionEncoder | emotion `Embedding(6,896)` + intensity `Embedding(4,896)`，逐位置相加 | 默认结构相同 | 保留相同张量拓扑 |
| FiLM | `Linear(896,1792)`，分为 gamma/beta，执行 `gamma*x+beta` | 默认结构相同；另有 AddFusion 消融类 | 删除 AddFusion 活跃路径 |
| emotion classifier | `Linear(896,6)`，读取 FiLM 后表示 | 张量形状可相同，但默认读取 decoder hidden | 修复前向连接关系；仅形状相同不能证明语义等价 |
| speech embedding/decoder | 6561 个 speech token + 3 个 special token | 父类提供相同规模 | 由严格 shape 测试固定 |
| FiLM `alpha` | 作者源码注册一个参数，但有效 forward 未使用 | 本地有效 forward 不需要 | 只作为旧 checkpoint 兼容项处理，不加入计算 |
| WordSequenceModel | 768d、8 heads、FFN 768→3072→768、分类头 768→5、VAD 头 768→3 | 旧本地入口固定 1024d，历史训练还使用 1D 强度头 | 主线直接采用作者结构和 checkpoint，旧模型退出活跃链 |

作者 `best_model.pth` 的序列化张量形状与作者 `model.py` 一致：attention 输入投影 `(2304,768)`、FFN `(3072,768)/(768,3072)`、分类头 `(5,768)`、回归头 `(3,768)`。因此 768d/5 类/3D 不是仅凭文件名推断。

本轮不追求作者参数逐值复刻；结构门禁只要求有效模块拓扑、关键张量形状、前向连接和加载边界正确。HF 转换后文本词表大小等封装差异按转换合同处理，不把序列化命名或未参与 forward 的 `alpha` 当成新模型结构。

### 3.4 Tokenizer 与推理条件

必须对齐：

- 正文片段无条件 lowercase；
- LLM 初始条件只包含 `[SOS, FiLM(target tagged text), task]`；
- prompt 音频仍供 speaker embedding、Flow prompt token 和 prompt feature 使用；
- `max_len=200`；
- 最短长度为 `2 × target text token count`；
- 最短长度前抽到 EOS 时在原分布重新采样；
- 只有真实 EOS 终止，另外两个辅助 special token 跳过且不加入前缀；
- RAS fallback 使用原始 scores，不预先排除第一次抽到的重复 token。

这些项目会改变模型条件、候选集合、停止规则或采样分布。

## 4. 不需要严格对齐的执行轨迹

以下差异不进入正式修复：

- 作者每步重算完整 prefix，本地可继续使用数学语义等价的 KV cache；
- 作者 speech tokenizer 固定 CPU provider，本地保留当前可用 provider；
- 作者整块 shuffle 与本地滑动 buffer shuffle；
- 多 GPU 分片导致的随机数消费顺序；
- 逐 utterance seed 和逐 token/逐字节输出一致性；
- 已确认等价的 sort 和实际 DDP AMP 计算路径。

它们可以改变某次随机轨迹或浮点累积，但现有证据不足以证明会产生目标统计量的系统偏差。本轮新规则是“语义对齐，不复刻执行轨迹”。

## 5. 工程可靠性问题

这些问题需要修复，但不称作者合同：

- 基座 checkpoint 的 `strict=False` 结果不能静默丢弃；仅允许明确的新增情感模块缺失；
- 训练后和续训 checkpoint 必须严格加载；
- 非流式 generator 被提前关闭时必须 `try/finally` 清理 UUID 会话状态；
- parquet 并行打包必须收集异步结果，子进程异常不得被忽略；
- checkpoint 不再同时保存五个 epoch 全量副本和相同内容的 final；
- 训练数据不再先打包全量 parquet，再重新编码 train/cv 形成字节重复；
- 每次数据、训练、推理和评测必须记录代码、输入、模型、随机性和命令身份。

## 6. 冻结的本地补全合同

作者未公开以下环节，本轮保留本地合理实现：

- ESD test 1500 条及训练排除关系；
- FEDD-rebuilt Part A MiMo 与 Part B ESD 词边界拼接；
- FEDD-A/FEDD-B 分区；
- prompt 选择、prompt_text 与 Part A neutral anchor；
- 数据索引、MFA、speech token、speaker embedding 和 parquet 打包；
- 多 GPU 批量推理；
- `emofilm-eval-v2` 的 WER、frame-mean Emo-SIM 和 cosine DTW。

这些内容必须称“本地补全实现”或“本地代理评测”，不能称作者官方实现。只有确定的工程缺陷、引用断裂或数据泄漏才触发单独修复。

现有 IEMOCAP MFA TextGrid 复用。仅允许跳过少量无有效词边界的 IEMOCAP 训练样本：总量不超过 manifest 的 1%，逐条记录原因，且不得明显集中于单个 speaker 或情感类别。ESD/FEDD test、reference 和 prompt 任何缺失均为硬失败。

## 7. 定向重建边界

### 7.1 重建或大幅精简

- `cosyvoice/llm/llm_emotion.py`：只保留作者有效训练前向和专用推理语义；
- `cosyvoice/llm/emo_film.py`：只保留 EmotionEncoder、FiLMLayer 和必要 checkpoint 兼容；
- `cosyvoice/cli/model_emo.py`：保留外部 TTS 接口，重建非流式生命周期和清理；
- `cosyvoice/cli/frontend_emo.py`：只负责目标文本三元组与 Flow/HiFT prompt；
- `cosyvoice/bin/train_emo.py`、`cosyvoice/utils/train_utils_emo.py`：去除 DPO/消融，固定基础 SFT 冻结和 optimizer；
- `conf/emo_film.yaml`：唯一基础配置。

### 7.2 局部修改

- `cosyvoice/dataset/processor.py`：token/mel 联合裁剪；
- `cosyvoice/tokenizer/emo_tokenizer.py`：lowercase 与闭合标签 fail-fast；
- `cosyvoice/utils/common.py` 或专用采样模块：作者 EOS/RAS 分布语义；
- 标注和打包工具：作者 768d/3D 合同、拒绝预算和异步失败传播。

### 7.3 保留

- 通用 CosyVoice、Flow、HiFT；
- 本地数据索引、FEDD、MFA、speech token、embedding；
- 批量并行推理与 `emofilm-eval-v2`；
- 仍服务正常链路的严格配对和运行身份原语。

## 8. 主线可达性清理

重构后 main 只保留唯一基础数据准备、训练、推理、评测及必要测试。以下内容在 Git tag/bundle 和证据包固化后移出活跃主线：

- 旧诊断矩阵编排和 tests；
- 消融 YAML、AddFusion adapter 和消融 tests；
- DPO/PA/RL 专用分支；
- 旧 1024d 标注器训练/评测路径；
- `per_emo_accuracy.py` 及其旧 checkpoint 依赖；
- 仅验证废弃兼容行为或内部实现细节的 tests。

历史恢复不再依靠把全部旧工具永久留在 main。

## 9. 数据与资产结论

新版派生产物统一进入：

```text
data/contracts/emofilm_author_v1/
├── manifests/
├── emotion2vec_base_frames/
├── word_blocks/
├── tagged/
├── src/
├── parquet/{train,cv}/
└── provenance/
```

train/cv utterance 成员关系沿用当前结果；只替换 IEMOCAP 标签，rejected 样本从原集合移除但不补位。

重构前采用证据优先分阶段清理。第一阶段候选包括旧 1024d 特征/词块、四组消融数据、v1-v5 重复转换权重和重复 ESD 归档；正式 v6 WAV、旧 `init/final` 和旧 full parquet 暂留到新版通过。任何删除必须另行列出精确路径并确认。

## 10. 最终验证边界

- 只训练一个论文口径 5-epoch Emo-FiLM；
- emotion2vec-base 与作者 WordSequenceModel 均冻结；
- 长期只保留 `init.pt + final.pt`，训练中使用可覆盖的 `latest.pt`；
- 执行 `3→30→本地固定 2500 条`；
- 只报告 WER、Emo-SIM、DTW 和运行完整性；
- 不运行 per-emotion accuracy、新消融或矩阵；
- 不设置论文或本地绝对数值失败阈值；
- 相对旧 local final 和 local identity 分区报告真实变化；混合结果不自动压成通过/失败。

单模型结果只回答“重构后的整体主线是否改善旧训练退化并保持合理质量”，不能证明某项修复的独立因果作用。

## 11. 已知不可闭合项

- 作者标注器训练数据、split、loss 权重和 3-epoch 运行；
- 作者基础 SFT manifest、数据顺序和实际运行配置；
- 作者原始 annotated_IEMOCAP span 分布；
- 原始 FEDD、prompt population 和正式批量 manifest；
- 作者基础 Emo-SIM/DTW/WER 实现与人工听测材料；
- 作者 checkpoint 的逐值训练轨迹。

这些是公开边界，不是继续搜索或增加实验分支可以消除的遗漏。
