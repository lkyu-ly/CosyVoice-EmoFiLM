# EmoFiLM 平台根因调研与训练方案（2026-08-02）

- 日期：2026-08-02
- 范围：四次可追溯实验（v1 / 5-epoch disabled / 27-epoch disabled / sentlvl 句级监督）的日志与产物；模型代码与数据合同；论文原文（`reference/arXiv-2509.20378v1/`）与作者源码（`reference/Emo_PA_code_data/`）；官方 CosyVoice3 配置（`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`）。
- 方法：日志/产物量化对比 + 4 个隔离实验（微型模型梯度分解、真实模型声学条件强度、随机扰动对照、评测指标动态范围）+ 官方第一手来源核对。
- 结论一句话：**四次实验共享同一条"弱条件通路"，Emo-SIM 的差异被非情感成分淹没；句级监督只训练了一个"探针可读但声学无效"的隐形方向，所以 WER 变差而情感不升。**

---

## 0. 置信度图例

- **高**：本报告直接测量/直接核对（代码、日志、隔离实验、官方文件）。
- **中**：机制推理有直接证据支撑但未做全量消融。
- **低**：推测，需要实验确认。

---

## 1. 四次实验为什么几乎没有差距（直接回答）

四个模型的差异可以被一张表完整解释：

| 实验 | 训练监督 | CV loss_tts（best） | ESD WER% | ESD Emo-SIM | 声学条件强度（本报告实测） |
|---|---|---|---|---|---|
| v1（init，有 loss_emotion，lr=1e-5 单组） | 句级 CE | 3.721（e4） | 9.48 | 66.75 | — |
| 5-epoch disabled | 仅 loss_tts | 3.695（e4） | 8.18 | 66.11 | — |
| 27-epoch disabled | 仅 loss_tts | 3.673（e21） | 8.38 | 65.89 | 情感对 KL 0.108、argmax 一致率 49.5% |
| sentlvl（重构版句级 CE，0.2） | loss_tts + 0.2·CE | 3.719（e14） | 10.05 | 65.45 | 情感对 KL 0.099、argmax 一致率 50.1% |

三个直接原因：

1. **训练变量只改变了"调制扰动的大小"，没有改变"调制的声学方向"**。实测（§2.2）：sentlvl 与 disabled 的"换情感 → 语音 token 分布变化"几乎完全相等（KL 0.099 vs 0.108，argmax 一致率 50.1% vs 49.5%），而 Emo-SIM 也几乎相等（65.45 vs 65.89）。句级监督让文本侧调制幅度变大（相对范数 0.91 vs 0.62），但多出来的部分恰好落在**语音预测器不敏感的方向**（§2.3），既不贡献情感，还抬高 CV loss_tts（3.719 vs 3.673）→ WER 变差。
2. **Emo-SIM 的平台不是"训练不够"而是"情感成分占比较低"**。§2.5 的指标动态范围实验显示：非中性情感下，生成音频与"同句真实情感参考"的相似度（45–68）经常**低于或接近**其与"同说话人中性 prompt"的相似度（63–80）。也就是说 ~66 这个分数里含有大量内容/说话人/中性基底成分，训练手段很难把它推高。
3. **四个实验都在同一方法平面上**：FiLM 句级常量仿射 + 冻结 Qwen2 骨干 + 只训练 1.48% 参数（§3.4）。换 epoch、换 LR、加句级 CE，都只在这个平面内滑动。

---

## 2. 证据链（本轮隔离实验）

### 2.1 梯度掩码 + 探针捷径（微型模型，CPU，200 步 × 3 变体）

用微型 Qwen2（hidden=32, 2 层）复刻完整 `Qwen2LM_Emotion` 训练路径（含 `freeze()` 三组、`emo_loss_weight=0.2`）：

```text
初始 loss_emotion 单独 backward 的梯度范数：
  emotion_encoder: 0.0        ← projection 零初始化完全屏蔽
  emotion_adapter (projection): 2.869

200 步 Adam 后 loss_emotion_input：
  both-trainable          0.074 → 0.0225
  encoder 冻结（仅 projection+decoder）  0.099 → 0.0358   ← 几乎一样低
  projection 冻结（仅 encoder+decoder）  1.689 → 1.689    ← 完全降不下去
```

结论（高置信度）：

- `FiLMLayer` 零初始化使 **loss_emotion 在初始时对 emotion_encoder 的梯度严格为 0**（真实训练日志也显示 loss_emotion_input 在前 500 步从 1.70 掉到 0.42——此时只有 projection 在学）。
- **探针 CE 可以只靠 projection 就降到接近最优**（0.036 vs 0.023），emotion_encoder 不是必要条件；相反，只训练 encoder 时探针 loss 纹丝不动。
- 这就是"标签回读捷径"的精确机制：**冻结随机线性探针 ↔ projection 的仿射重映射**构成一个闭环，根本不经过 emotion_encoder，更不经过声学。

### 2.2 真实模型：句级监督没有改变"有效声学条件强度"（GPU，20 条 ESD 样本，三模型对比）

加载真实 `CosyVoice2_Emotion` + 各实验 `final.pt`，固定文本与 teacher-forced speech token 续写，对比 hap/sad、ang/neu、sur/neu 三对条件下**全序列 120 个 speech-token 位置的分布**：

```text
                KL_avg（每位置）   argmax 一致率
base-untrained     0.000            100.0%   ← FiLM 零初始化，无条件
sentlvl            0.099             50.1%
longepoch-disabled 0.108             49.5%
```

另测前缀（`[SOS, FiLM(text), task]`）末位首个 speech-token 分布：

```text
                next-token KL（情感对均值）  同范数随机扰动的 KL
sentlvl                  0.488                    5.449
longepoch-disabled       0.478                    5.402
```

结论（高置信度）：

- **条件确实到达了声学链**：两个训练模型都让约一半 token 位置的 argmax 因情感改变；这不是"条件没进去"。
- **但句级监督没有改变这个强度**：sentlvl ≈ disabled（0.099 vs 0.108）。探针训练出的额外调制（文本侧相对范数 0.91 vs 0.62）在语音侧"不可见"。
- 随机扰动对照说明：同样范数的随机扰动会让 next-token 分布发生 11 倍大的变化（KL 5.4 vs 0.49）。**训练找到的是一个对语音预测器"低敏感"的隐形方向**——它能被随机探针读出（sentlvl 探针准确率 100%，disabled 0%），但不构成有效的声学情感信号。

### 2.3 情感表征几何：sentlvl 与 disabled 学到的 emotion embedding 几乎相同

```text
emotion_embedding 范数（5 类）：sentlvl 28.05–28.96；disabled 26.90–28.20
类间余弦矩阵：两者几乎逐位相同（近正交，0.02–0.07）
projection W 范数：sentlvl 11.84；disabled 8.66（bias 均 ~30.5）
```

结论（高置信度）：句级监督**没有**让情感表征变得更"有语义"——它只是把 projection 调大了一点。两个模型都把情感码当成纯控制码，几何几乎一致。

### 2.4 训练日志：探针 loss 首 epoch 即饱和，全程平坦

```text
sentlvl CV loss_emotion_input：e0=0.080 → e1=0.081 → ... → e14=0.074 → e17=0.074
（第一轮训练内从 1.70 掉到 ~0.1，此后 17 个 epoch 不动）
v1 loss_emotion：e0=0.346 → e4=0.114（lr=1e-5，更慢但同样趋近 ~0.1）
```

结论（高置信度）：探针在几百步内就饱和到 0.07–0.08（5 类 CE 的随机下界是 ln5≈1.61，饱和值约为其 1/20）。之后它对 emotion_encoder/backbone 几乎没有梯度压力。

### 2.5 评测指标动态范围（15 条 ESD 样本，emotion2vec_plus_large，与官方 eval 同函数）

```text
               hyp_sentlvl↔ref  hyp_disabled↔ref  neutral_prompt↔ref  hyp_sentlvl↔neutral  hyp_disabled↔neutral
ang (n=3)         61.7              79.8                38.6                 80.3                 62.7
hap (n=3)         47.2              35.3                18.3                 62.8                 76.9
sad (n=3)         67.7              68.2                40.6                 78.0                 78.9
sur (n=3)         45.0              15.5                12.7                 63.2                 77.1
ALL (n=15)        64.2              59.0                41.4                 —                    —
```

要点（中-高置信度，n 小仅作机制指示）：

- 全量官方口径（n=1500）下四模型 ESD Emo-SIM 都落在 65.45–66.75；本小样本的 ~64 与之一致。
- 非中性情感下，生成音频与"目标情感真实音频"的相似度经常**低于**其与"同说话人中性 prompt"的相似度——**可听情感成分整体偏弱**，Emo-SIM 高分主要由内容/说话人/中性基底贡献。
- 补充：本仓库评测用**同说话人 Neutral prompt**（`select_prompt_wav` 固定取 Neutral），模型必须纯靠标签把情感做出来。论文（§4）的 CosyVoice2 基线用**同情感参考音频**（近乎克隆），因此论文 ESD Emo-SIM 高达 98.73–99.32；两套数字**不可直接对比**。本仓库协议是更难的"标签→声学"测试。

### 2.6 数据事实（全量扫描 20774 条训练行）

- 训练集 20774 条，全部含 `<emotion>` 标签；**5484 条（26.4%）是词级多标签**（全部来自 IEMOCAP，逐词 emotion+intensity），其余为 ESD 句级单标签；FEDD 不在训练集。
- **没有任何训练 parquet 携带 span/对齐张量**（列仅 `utt,audio_data,wav,text,spk,utt_embedding,spk_embedding,speech_token`）；`downstream_supervision` 的 span 链路从未接线训练过。
- 评测集 2500 条（ESD 1500 + FEDD 1000）与训练集**零重叠**（无泄漏）。
- 因此"句级监督"的实际覆盖是：ESD 整句广播 + IEMOCAP 词级标签的逐 token CE；而仓库自 2026-07 就具备的 span 词级监督头（emotion_head/arousal_head + soft CE/MSE）**从未被训练过**。

---

## 3. 根因清单（按置信度从高到低）

### H1（高）探针-调制器捷径：loss_emotion 证明的是"文本可被随机投影分对"，不是"语音表达情感"

证据：§2.1（encoder 零梯度 + projection-only 即可饱和）、§2.4（探针首 epoch 饱和）、§2.2（sentlvl 探针 100% 准确但声学条件强度与 disabled 相同）。

这同时解释了：

- 为什么句级监督"看似在工作"（loss 降到 0.07）但 Emo-SIM 不动；
- 为什么它有害：为满足随机探针，projection 把 text embedding 推离预训练流形，`loss_tts` 收敛点抬高（3.719 vs 3.673），WER 恶化（10.05 vs 8.38）。

### H2（高）有效声学条件与句级监督无关：四模型的情感条件强度相同、方向非情感性

证据：§2.2（KL/argmax 全序列对比）、§2.5（指标动态范围）。训练时长（5 vs 27 epoch）与监督（有/无句级 CE）都不改变"换情感时 token 分布变化多少"；变化的只是扰动方向，而该方向不是声学情感方向。

### H3（高）FiLM 表达力上限：句级常量仿射 + 离散控制码 + 无时间变化

- `EmotionEncoder` = 两个离散 embedding 相加；`FiLMLayer` = 线性投影出 γ/β，对整句 text embedding 做逐 token 仿射（句内 emotion_ids 不变时 γ/β 恒定）。
- 情感韵律是时间变化的（词级重音、F0 轮廓、节奏），句级常量仿射只能做全局平移/缩放；即使有词级标签（IEMOCAP），也只在词边界处跳变，且 FiLM 仍然作用在 text embedding 上，不直接作用在 speech-token 隐状态。
- 论文消融（§4）也指出"去掉词级数据 DTW 49.6→134.0"——词级信息才是该方法的杠杆；本仓库训练里词级标签存在（26%），但没有走词级调制/词级监督的完整链路。

### H4（中-高）离散 speech-token 的韵律瓶颈 + 冻结的声学侧

- 论文相关文献（2510.05758）指出：LLM-TTS 依赖离散 speech token，难以直接建模连续情感强度/韵律凸显；本仓库实测 25 Hz token 流上情感条件只改变 ~50% 位置的 argmax，但这些变化没有转化为可听情感。
- CosyVoice2 的 speech token（speech_tokenizer_v2）以语义/内容为主；Flow（冻结的 masked diffusion）+ HiFT 的重建先验偏中性。
- CosyVoice3 换用 **S3Tokenizer v3**（监督语义 tokenizer），官方宣称在内容一致性、说话人相似度、韵律自然度上超过 CosyVoice2——这正是"离散 token 能否承载韵律/情感"层面的升级（中置信度：官方宣称，未在本仓库验证）。

### H5（中-高）评测协议与指标敏感度：Emo-SIM 平台部分是"指标假象"

- 2026 年论文《The False Resonance: A Critical Examination of Emotion Embedding Similarity for Speech Generation Evaluation》（arXiv 2604.26347）直接批评 emotion2vec 余弦相似度类指标会被语言/说话人因素主导。
- 本仓库 `emo_sim` = frame 特征 mean-pool 后余弦 ×100：平均池化抹掉时间结构，且对"情感成分"的敏感度有限（§2.5）。
- 论文原文自己也承认 "Emo SIM averages frame-level emotion vectors, which may partially obscure dynamic emotional information"。
- 结论：即使训练方案改进，单靠 Emo-SIM 也可能看不到收益，需要补强评测（§6 R6）。

### H6（高，事实）本仓库"v1 复现"与论文实现的差距

对照论文原文与作者代码（`reference/Emo_PA_code_data`）：

| 维度 | 论文/作者实现 | 本仓库实现 |
|---|---|---|
| 词级标注来源 | emotion2vec frame 特征 + MFA 对齐 → 训练词级预测模型（分类头 + 连续强度回归头） | IEMOCAP 逐词离散标签（词级伪标签/句级广播，provenance 有 `sentence_broadcast`） |
| 情感表征 | 离散类别 + **连续强度**（emotion2vec 派生） | 离散 emotion_ids + 离散 intensity_ids（0–3 档） |
| 辅助监督位置 | 冻结随机线性探针作用于 `modulated_text_emb`（作者代码与 v1 一致） | 同左（忠实复现） |
| span 监督头 | 论文的"Emotion Supervision Mechanism"对应词级预测模型的双头（分类+回归） | 仓库有 `emotion_head`/`arousal_head`（输出侧 span 池化），**从未训练** |
| 评测 | ESD 1500，CosyVoice2 基线用同情感参考音频（Emo-SIM 98–99） | ESD 1500，同说话人 Neutral prompt（Emo-SIM 65–66） |

因此：

- 本仓库的 `emotion_classifier` 机制与作者代码**一致**（作者训练脚本同样冻结 classifier，只解冻 EmotionEncoder/Adapter/Decoder，见 `train.py:167-198`）——"v1 原样复现"成立。
- 但论文效果的核心（词级 emotion2vec 连续特征、词级预测模型、FEDD 词级数据）在本仓库训练链路里**缺失或未接线**；仓库复现的只是"输入侧 CE"这一最弱部件。
- 论文 ESD Emo-SIM 98.78 与仓库 66 的差距主要来自评测协议（参考音频 vs neutral prompt）与 emotion2vec 口径，不能直接解读为"论文方法在仓库数据上一定有效"；但论文消融（无词级数据 DTW 49.6→134.0；无辅助 loss 49.6→73.9）仍是"词级+输出侧监督有效"的最强外部证据。

### H7（中）"CosyVoice2 太成熟、难微调"假说的局部成立

- 事实：499M/507M 参数冻结，只训练 1.48%；backbone 的输入嵌入流形高度锚定，小扰动被吸收（§2.2 的"隐形方向"）。
- 但实测 backbone **没有完全压制**条件（隐藏层相对变化 0.49–0.58）；瓶颈更多在"调制方向不被解码器映射为情感"。
- 所以"换更强的基座"能提高上限，但**不改变条件通路的设计缺陷**；必须与 H1–H3 的改造一起做。

---

## 4. 文献对照（本报告直接核实的主要来源）

- **Emo-FiLM / Emo-PA（arXiv 2509.20378）**：方法 = emotion2vec 词级特征 + 词级 FiLM 调制 + 辅助情感分类；消融证明词级数据与辅助 loss 关键（本仓库存有原文与作者代码，§3 H6）。
- **Emo-DPO（arXiv 2409.10157, ICASSP 2025）**：用直接偏好优化区分正负情感对，报告了比单纯 CE 更强的情感可控性——可作为"标签 CE 之外的监督"候选方向。
- **《The False Resonance》（arXiv 2604.26347, 2026）**：系统性批评 emotion2vec 余弦相似度作为情感生成评测指标（语言/说话人主导）——直接关系到 Emo-SIM 平台的可解释性。
- **TEMOTTS（arXiv 2405.11413）**：无标签情感风格建模——提示"连续风格空间"比离散标签更可能承载情感。
- **LLM-TTS 离散 token 局限（arXiv 2510.05758）**：离散 speech token 难以直接建模连续情感强度/韵律凸显——支撑 H4。
- **CosyVoice3 官方（HF `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` + 仓库 `cosyvoice3.yaml`）**：Apache-2.0、9 语言 + 18 方言、24 kHz；LLM 为 `CosyVoice3LM`（Qwen2.5-0.5B，hidden 896，`speech_token_size=6561`，输出头 `+200` 特殊 token）；Flow 换为 **DiT**（`CausalMaskedDiffWithDiT`，22 层）；HiFT 换为 `CausalHiFTGenerator`；speech tokenizer 换为 **S3Tokenizer v3**（`speech_tokenizer_v3.onnx`）。

---

## 5. CosyVoice3 切换评估

### 5.1 事实核查（本仓库代码 + 官方配置）

- **代码层面：本仓库已经是"CosyVoice3 时代"的上游代码**——`cosyvoice/cli/cosyvoice.py` 含 `CosyVoice3` 类（读 `cosyvoice3.yaml`、`speech_tokenizer_v3.onnx`），`cosyvoice/llm/llm.py` 含 `CosyVoice3LM`（`llm.py:664`），`cosyvoice/flow/flow.py` 含 `CausalMaskedDiffWithDiT`，`cosyvoice/hifigan/generator.py` 含 `CausalHiFTGenerator`，`cosyvoice/flow/DiT/` 存在。**用户"代码核心框架是 CosyVoice3"的说法基本成立**（代码支持 v3），但当前所有配置/权重/数据都锚定 CosyVoice2-0.5B。
- **配置层面**：官方 `cosyvoice3.yaml` 与当前 `conf/emo_film*.yaml` 的差异：
  - LLM 类 `Qwen2LM_Emotion`（继承 `Qwen2LM`，sos=0/task=1/eos=6561/+3 头）→ 需改为继承 `CosyVoice3LM` 的变体（sos=6561/eos=6562/task=6563/+200 头、无 bias、instruct token、`<|endofprompt|>` 语义）。
  - Flow：`CausalMaskedDiffWithXvec + UpsampleConformerEncoder` → `CausalMaskedDiffWithDiT + PreLookaheadLayer`（input_size 80 vs 512）。
  - HiFT：`HiFTGenerator + ConvRNNF0Predictor` → `CausalHiFTGenerator + CausalConvRNNF0Predictor`（`conv_pre_look_right: 4`）。
  - tokenizer：`speech_tokenizer_v2` → `speech_tokenizer_v3`；Qwen2 → Qwen2.5 tokenizer（`CosyVoice-BlankEN`）。
  - 数据：**现有 parquet 的 `speech_token` 是 v2 tokenizer 的 id，与 v3 不兼容，必须用 `speech_tokenizer_v3.onnx` 对 `datasets/` 下全部 wav 重新抽 token**（20774 训练 + 1092 CV + 2500 评测；这是最大的一项迁移工作）。
- **许可证**：CosyVoice3-0.5B-2512 为 Apache-2.0（与 CosyVoice2 相同/兼容），无商用障碍（社区镜像与模型卡均确认）。
- **微调支持**：官方仓库 train 脚本按 `cosyvoice3.yaml` 可训练 LLM/Flow；本仓库 `train_emo.py` 需要新增一个 `conf/emofilm_cosyvoice3.yaml` + `Qwen2LM_Emotion` 的 v3 变体 + flow/hift 配置替换。

### 5.2 建议（中置信度）

**换，但不要单独换。** 理由：

- 换 CosyVoice3 的确定收益：更好的内容一致性/自然度基线（官方宣称）、更可能承载韵律的 S3Tokenizer、Apache-2.0、代码已就绪（迁移工作量主要是配置 + 重抽 token，不是重写架构）。
- 换 CosyVoice3 不解决本报告 H1–H3：探针捷径、句级常量仿射、冻结骨干通路在新基座上原样存在；单独换基座很可能只是把 ~66 平台平移几个点。
- 正确顺序：**先（或同时）把监督与条件通路改造落地（§6 R1–R4），再迁移基座**；迁移后 EmoFiLM 的 FiLM/span 头/checkpoint 策略可以直接复用。
- 如果只想做一次大改：**R1（span 词级监督）+ R3（词级/时间可变条件）+ CosyVoice3 基座**三者一起做，方向一致且互相增强。

---

## 6. 推荐的训练框架调整方案（按优先级与置信度）

### R1（高优先级，中-高置信度）接通"输出侧 span 词级监督"——仓库已有代码，只差数据接线

- 现状：`emotion_head`（5 类 soft CE）+ `arousal_head`（连续 MSE）+ `_pool_span_features`（speech-token masked-mean 池化，反捷径契约）**已实现、已测试、从未训练**；`tools/generate_tagged_jsonl.py`、`cosyvoice/dataset/span_align.py` 已存在；训练数据里已有 5484 条 IEMOCAP 词级标签。
- 动作：
  1. 用现有 span 管线为训练 parquet 生成对齐 span 张量（词边界 + 每词 emotion soft 分布 + 连续 arousal + valid mask），写入新版本训练数据（不改冻结的 v1 合同，另建 v2 数据目录）。
  2. `conf` 新配置 `downstream_supervision: enabled`，`emotion_head_weight`/`intensity_head_weight` 调参起步 0.5–1.0。
  3. 训练时**关闭或删除 input-end 探针**（`emo_loss_weight=0`），避免两个捷径互相干扰。
- 预期：监督落在生成因果链下游（lm_output 的 speech-token 区段），目标与 FiLM 输入分离，不存在 H1 的回读捷径；这是 ADR-0019 与论文共同指向的正道。

### R2（高优先级，中置信度）句级监督改造：语义锚点或输出侧句级头

- 若保留句级监督，把冻结随机探针换成：
  - 输出侧句级头：对 `lm_output` 的 speech-token 区段做句级 mean-pool → 可训练分类器（与 R1 同一位置，只是粒度更粗）；
  - 或语义锚点监督：以 emotion2vec 的 utterance/frame embedding 为回归/对比目标（连续），而不是离散 ID 的 CE。
- 无论如何**不要再在 `modulated_text_emb` 上加线性读出器**（H1 已证伪）。

### R3（高优先级，中置信度）条件注入结构改造：让情感条件"时间可变 + 靠近声学侧"

三选一（可组合）：

1. **词级 FiLM**：`emotion_features` 按词边界展开成 token 级序列（IEMOCAP 已有词标签；FEDD-A 有构造的词级转移），γ/β 随词变化——这是论文的原始形态。
2. **speech-token 侧注入**：把情感 embedding 拼接到 `speech_token_emb`（或每步解码的 KV 前缀），让条件直接出现在生成因果链里，而不是只改 text embedding。
3. **learned emotion tokens**：在文本序列中插入可学习的情感 token（如 `<|emotion_hap|>`），让冻结骨干通过注意力自行"读取"条件（CosyVoice3 的 instruct 风格）；比 FiLM 仿射的表达空间大。

### R4（中优先级，中置信度）骨干适配：LoRA / 部分解冻

- 在 Qwen2 的 attention 投影上加 rank 8–32 LoRA（作者代码里保留过 LoRA 实验但被注释掉，见 `llm_emo.py:134-152`），或解冻最后 4–6 层。
- 目的：让"调制方向"能通过骨干的权重空间被映射为韵律变化，而不是只能走输入嵌入的隐形方向。
- 成本：0.5B 模型 LoRA 训练在 6×3090 上可行；`train_utils_emo.freeze()` 需要加一组 `lora` 参数组。

### R5（中优先级，中置信度）数据：词级伪标签管线 + 更大情感语料

- 论文的做法：emotion2vec 帧特征 + MFA → 词级预测模型（分类+强度回归）→ 词级标注。本仓库只有 IEMOCAP 词级伪标签（`sentence_broadcast` 弱监督），ESD 是句级。
- 建议：按论文流程为 ESD/FEDD 生成 emotion2vec 派生的词级 soft 标签 + 连续强度，让 74% 的句级数据也获得词级监督；若资源允许，引入带情感标注/伪标签的大规模语料（MSP-Podcast 等）。

### R6（中优先级，高置信度）评测补强：不能只看 Emo-SIM

- 增加：emotion2vec/分类头的**逐类准确率**（论文 ESD 报告了分类准确率，如 Happy 65.6%、Surprise 71.2%）、ABX 情感判别、主观 EMOS/NMOS（论文口径）、**neutral-prompt 对照**（当前协议保留）与**同情感参考对照**（论文口径）双轨报告。
- 关键：把"条件一致性"作为指标——同一文本换情感时，生成的韵律差异（F0/能量/时长）应可测且方向正确；Emo-SIM 平台下也要能看出方法是否在移动声学特征。

### R7（决策建议）CosyVoice3 基座：与 R1/R3 捆绑切换

- 单独换：预计平台平移，性价比低（迁移成本主要在重抽 token）。
- 捆绑换：S3Tokenizer v3 可能缓解 H4 的离散 token 瓶颈，配合 R1/R3 才是"有真正提升的训练方案"。
- 迁移清单见 §5.1；建议先做 1–2 天可行性 smoke（下载 v3 权重 + 重抽 100 条 token + 小规模训练冒烟）再决定。

---

## 7. 建议的下一步小型实验（可直接隔离验证）

1. **span 头接线冒烟**（1–2 GPU·小时）：取 1000 条 IEMOCAP 词级数据 → span_align → 训练 2–3 epoch（`downstream_supervision=enabled`、`emo_loss_weight=0`）→ 对比 CV loss_tts 与 WER/Emo-SIM。验证输出侧监督是否真的不损害质量。
2. **词级 FiLM 冒烟**：把 R3-1 的最小实现（emotion_features 按词展开）在现有 26% 词级数据上训练 5 epoch，对比 sentlvl/disabled 的 Emo-SIM 与 F0 差异。
3. **LoRA 消融**（中成本）：同配置 ± LoRA（rank 16），观察 next-token KL 是否变大且 Emo-SIM 是否移动——直接检验 H4/H7。
4. **指标对照**：在现有四个模型的 full wav 上补算分类准确率与 ABX，确认 Emo-SIM 平台是否也是指标盲区。
5. **CosyVoice3 token 兼容性**：对 100 条 ESD wav 用 `speech_tokenizer_v3.onnx` 重抽 token，对比 v2 token 的码本分布与重建质量，验证 H4 与迁移成本。

---

## 8. 可追溯性

- 实验产物：`exp/emofilm_v1/`（v1，log/CV 全）、`exp/emofilm_film_only/`（5-epoch）、`exp/emofilm_film_only_longepoch/`（27-epoch 早停）、`exp/emofilm_sentlvl/`（句级监督）；各目录含 `train.log`、`final.pt`、`full/`（2500 wav）、`eval/*_metrics.json`、`run_*.sh`。
- 官方指标文件：`artifacts/emofilm_v1/evaluation/*_metrics.json`。
- 本轮隔离实验脚本（未入库，命令在本次会话记录中）：微型梯度分解、真实模型 KL/argmax、随机扰动对照、评测动态范围、emotion embedding 几何。
- 主要来源：
  - 论文：`reference/arXiv-2509.20378v1/main.tex`；作者代码：`reference/Emo_PA_code_data/`
  - CosyVoice3：`https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`（模型卡与 `cosyvoice3.yaml`，经镜像 `gitserver.onethingai.com` 取得）
  - Emo-DPO：arXiv 2409.10157
  - 指标批评：arXiv 2604.26347
  - 离散 token 局限：arXiv 2510.05758
  - TEMOTTS：arXiv 2405.11413

> 注：两个联网调研子代理（CosyVoice3、文献）在本次会话中被派发但长时间未返回；本报告上述结论全部由主代理直接核对第一手来源得出，不依赖子代理。若子代理后续返回补充材料，可在本文件末尾以附录形式追加。
