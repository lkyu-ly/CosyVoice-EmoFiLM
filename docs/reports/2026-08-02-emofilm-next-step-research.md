# EmoFiLM 四次实验根因调研与下一步训练方案（最终合并报告）

- 日期：2026-08-02
- 范围：v1 / 5-epoch disabled / 27-epoch disabled（早停）/ sentlvl（句级监督）
  四次可追溯实验为何差距小、句级监督为何不及纯 loss_tts；规划"有真正提升"的
  训练方案；评估切换 Fun-CosyVoice3-0.5B-2512。
- 方法：日志/产物/ckpt 取证 + 隔离实验（copy-synthesis、token-carry、梯度
  分解、声学条件强度、调制病理、评测动态范围）+ 论文原文与作者代码核对 +
  CosyVoice3 官方配置核对 + 联网文献佐证（2024–2026）。
- 结论一句话：**四模型共享同一条"弱条件通路"（中性 prompt 声学钳制 + 冻结
  主干 + Emo-SIM 指标盲区），句级监督只训出了一个"随机探针可读、声学无效"
  的文本侧隐形方向，所以 WER 变差而情感不升。真正有提升的路是：修生成/评测
  协议 → 把监督从文本嵌入 CE 移到音频/生成链下游（音频奖励或 span 词级监督）
  → 修复 FiLM 数值病理 → 解冻/LoRA 主干 → 再评估 CosyVoice3 迁移。**

---

## 1. 方法与本报告可追溯性

本次调研由三个子代理并行取证，根代理对全部关键数值做了交叉核验：

1. **产物取证代理**：日志/评测 json/ckpt 漂移测量，报告：
   `docs/reports/2026-08-02-emofilm-why-plateau-investigation.md`。
2. **CosyVoice3 调研代理**：官方权重/配置/论文核对 + 隔离实验，报告：
   `docs/reports/2026-08-02-emofilm-rootcause-research.md`。
3. **文献调研代理**：因卡在孙代理未返回，其覆盖范围已由上面两个代理以一手
   来源（论文原文、作者代码、官方 yaml、HF 模型卡）直接核对，不依赖其返回。

根代理已复验：四份 `eval/*_metrics.json` 数字、`/tmp/grad_conflict_probe.json`
（读回 100%、unmodulated 16.7%、emotion_encoder 梯度余弦 −0.26、可训练后 CE
8.9e-5）、`/tmp/emo_discriminability.json`（5-way acc 41.7–50%、neutral 主导）、
`/tmp/emofilm_mod_analysis.log`（调制 4.2–9.2×、γ 偏离恒等 1.3–2.5）、
`/tmp/emofilm_sens2.log`（KL 0.20–0.29）、v1 单参数组 vs sentlvl 三参数组 +
WarmupLR 日志、copy-synthesis/token-carry 产物行。

---

## 2. 已核实的事实基线（四方对比）

| 数据集 | 指标 | v1 | 5ep disabled | 27ep disabled | sentlvl |
|---|---|---|---|---|---|
| ESD | WER% | 9.48 | 8.18 | 8.38 | **10.05** |
| ESD | Emo-SIM | 66.75 | 66.11 | 65.89 | **65.45** |
| FEDD-A | WER% | 8.30 | 4.70 | 4.69 | **6.54** |
| FEDD-A | Emo-SIM | 81.94 | 82.71 | 82.46 | **82.91** |
| FEDD-B | WER% | 14.42 | 12.30 | 11.57 | **14.04** |
| FEDD-B | Emo-SIM | 61.60 | 62.98 | 64.31 | **63.19** |

训练侧事实：

- v1：单参数组 `emotion_new`（7.5M 参数，lr=1e-5 constantlr），
  `emo_loss_weight=0.2` + 冻结分类器；loss_emotion 从 0.35 缓慢降到 ~0.1。
- sentlvl：三参数组（FiLM 1e-4 / heads 1e-4 / decoder 1e-5）+ WarmupLR，
  250 步后按 `1/sqrt(step)` 衰减，epoch 16 时实际 lr≈7.5e-6；loss_emotion_input
  从 1.70（batch 0）→ 0.42（batch 500）→ 0.15（batch 1000）→ 长期 ~0.07–0.08。
- sentlvl cv_loss_tts 收敛 3.719（best@14），27ep disabled 为 3.673（best@21）；
  ESD WER 劣化方向与 loss_tts 抬高方向一致。
- 训练数据 20774 条：5484 条（26.4%）IEMOCAP 词级多标签，其余 ESD 句级；
  **没有任何训练 parquet 携带 span/对齐张量**，输出侧 span 监督头
  （emotion_head/arousal_head，ADR-0019）从未被训练。

---

## 3. 问题清单（按置信度从高到低）

### H1【高】生成协议：中性 prompt 声学条件把情感差异"洗掉"，这是平台主因

决定性隔离实验（30 样本 = 5 情感 × 6，同说话人同文本）：

| 条件 | 5-way 判别 acc | 说明 |
|---|---|---|
| copy-synthesis（目标音频自身 token + 自身声学 prompt） | **93.3%** | 离散管线本身不丢情感 |
| 参考 token（原样提取）+ 中性 prompt（正式生成链路） | **53.3%** | 声学 prompt 一换，完美 token 也只剩随机水平 |
| v1 / 5ep / 27ep / sentlvl 模型生成 | 43.3 / 50.0 / 43.3 / 36.7 | 全部低于"中性 prompt"上限 |

机理：v2 单流协议下 prompt 不进 LLM 条件，只作为 Flow/HiFT 的声学条件；
Flow/HiFT 被中性参考的声学先验主导，token 层做得再好也到不了情感上限。
这同时解释了**为什么四次实验差距小**——四者共享同一条弱条件通路，训练变量
（epoch/LR/监督）只能在这个通路内滑动。

### H2【高】输入端句级 CE 是"文本可分捷径"，结构性无法提升声学情感

- 冻结随机分类器在 FiLM 后文本嵌入上的读回 CE=0.072 / acc=**100%**；同一分类器
  在未调制嵌入上 acc=16.7%（≈chance）。它证明的只是"FiLM 输出可读出条件 ID"。
- 微型模型梯度分解：FiLM 零初始化使初始时 emotion_encoder 的 loss_emotion
  梯度**严格为 0**；只训 projection 探针 CE 也能降到 0.036，只训 encoder 完全
  降不下去 → 标签回读闭环 = 冻结探针 ↔ projection 重映射，不经过情感编码器。
- 因此"句级监督没训到模型"不准确；准确说法是：**训到了文本侧错误目标**。

### H3【高】有效声学条件强度与句级监督无关，且方向非情感性

- 真实模型换情感 → 全序列 speech-token 分布变化：sentlvl KL=0.099 vs disabled
  KL=0.108，argmax 一致率 50.1% vs 49.5%——句级监督没有改变"条件进去了多少"。
- 同范数随机扰动会让 next-token KL 变化 5.4，而情感对只有 0.49——训练找到的
  是语音预测器低敏感的隐形方向（探针可读、声学无效）。
- 情感表征几何：sentlvl 与 disabled 的 emotion embedding 几乎逐位相同，句级
  监督只是把 projection 范数从 8.66 调大到 11.84。

### H4【高】FiLM 数值病理：调制幅度过大、方向不正，直接伤害 WER

| 模型 | projection W 范数 | γ 偏离恒等 | 调制幅度 ‖Δ‖/‖text‖ |
|---|---|---|---|
| v1 | 5.91 | 1.33 | 8.4–9.2× |
| sentlvl | 11.84 | 2.47 | 10.1–10.8× |
| 5ep disabled | 9.02 | 1.79 | 4.4–5.6× |
| 27ep disabled | 8.66 | 1.57 | 3.9–5.2× |

emotion/intensity embedding 初始范数 ~30，text embedding 范数 ~0.93；FiLM 用
~30 范数的特征乘出 γ/β，把文本嵌入扰动到自身 4–10 倍。带句级 CE 的两版
（v1/sentlvl）调制最大、WER 最差——这不是"FiLM 没动"，而是**动过头且方向不对**。

### H5【高】干预面过小 + 有效 LR 快速衰减，四模型同质化的放大因素

- 冻结 Qwen2 骨干，只训 1.48% 参数（FiLM 4 张量 + decoder 2 张量 + 头 4 张量）。
- WarmupLR 250 步后按 1/sqrt 衰减到 ~7e-6（FiLM）/ 7e-7（decoder）。
- 四模型从同一 llm.pt 出发，改动面极小，同 utt 跨模型生成音频余弦 0.72–0.99。

### H6【高】Emo-SIM 均值指标对情感差异不敏感，被中性/内容/说话人基底主导

- per-emotion 参考相似度：neutral 76–81，ang/hap/sad/sur 仅 32–54 → 全局 ~66
  主要是"中性参考相似度"贡献的。
- 模型输出同文本跨情感余弦 78–87 vs GT 39；5-way 判别 acc 41.7–50%（随机
  20%）。Emo-SIM 均值看不到这些差异。
- 2026 年文献《The False Resonance》(arXiv 2604.26347) 直接批评 emotion2vec
  余弦类指标被语言/说话人因素主导；论文作者也承认 frame 均值会模糊动态情感。
- 本仓库 neutral-prompt 协议比论文同情感参考（Emo-SIM 98–99）严格得多，两套
  数字不可直接对比。

### H7【高，事实】仓库"v1 复现"与论文实现的差距：只复现了最弱部件

| 维度 | 论文/作者实现 | 本仓库实现 |
|---|---|---|
| 词级标注 | emotion2vec 帧特征 + MFA → 词级预测模型 | IEMOCAP 离散词标签（26%），其余句级广播 |
| 情感表征 | 离散类 + 连续强度 | 离散 emotion + 4 档 intensity |
| span 监督头 | 词级预测模型双头（分类+回归） | 代码已有（ADR-0019），**从未训练** |
| 辅助监督位置 | 输入端冻结探针（与 v1 一致） | 同左 |
| 评测 | 同情感参考音频（Emo-SIM 98–99） | 中性 prompt（Emo-SIM 65–66） |

作者代码训练脚本同样冻结 classifier（train.py:167-198），所以"v1 原样复现"
成立；但论文效果的核心（词级连续特征、词级预测模型、词级数据）在本仓库训练
链路里缺失或未接线。

### H8【中高】FiLM 表达力上限：句级常量仿射 + 离散控制码 + 无时间变化

- EmotionEncoder 是两个离散 embedding 相加；FiLMLayer 对整句 text embedding
  做逐 token 常量仿射（γ/β 句内恒定）。
- 情感韵律本质是时间变化（词级重音、F0 轮廓、节奏）；即使有词级标签，也只在
  词边界跳变，且 FiLM 作用在文本嵌入而非 speech-token 隐状态。
- 论文消融：去掉词级数据 DTW 49.6→134.0，词级信息是方法的核心杠杆。

### H9【中高】离散 speech-token 的韵律瓶颈 + 冻结的声学侧

- LLM-TTS 依赖离散 speech token，难以直接建模连续情感强度/韵律凸显
  （arXiv 2510.05758）。
- 25Hz token 流上情感条件只改变 ~50% 位置的 argmax，但这些变化没有转化为
  可听情感；冻结的 Flow+HiFT 重建先验偏中性。

### H10【中高】早停/选点指标错位：cv loss_tts 是情感弱代理

- 27ep disabled best@21 在 ESD Emo-SIM 上反而略降（65.89 vs 5ep 66.11）；
  sentlvl best@14 的 loss_tts 更差（3.719）。
- 早停完全感知不到情感维度，选点与"情感最好"可能无关。

### H11【中】数据规模与标签质量

- 训练集 ~20.8k 句、10 个 ESD 说话人；74% 是句级离散单标签，词级标签仅
  IEMOCAP 且 provenance 含 `sentence_broadcast` 弱监督。
- FlexiVoice / EMORL 均引入更大或互补数据（Emilia/Expresso/NCSSD）。

### H12【中低，已验证】"可训练有语义锚点分类器"救不了输入端 CE

隔离验证：可训练分类器能把 CE 从 0.072 吸收到 8.9e-5，且 FiLM 上梯度范数从
0.66 缩到 0.0017——但它只是吸收残余梯度：init 时文本嵌入不含情感信息，分类器
只能通过推动 FiLM 制造可分性来降 CE；训练后它还会学文本-情感伪相关。**锚点
必须放在声学/生成链下游，而不是输入端文本嵌入上。**

### H13【中】"CosyVoice2 太成熟、难微调"假说局部成立

- 事实：只训 1.48% 参数，输入嵌入流形高度锚定，小扰动被吸收。
- 但隐藏层相对变化 0.49–0.58 说明 backbone 没有完全压制条件；瓶颈更多在
  "调制方向不被解码器映射为情感"。换基座能提高上限，但不改变条件通路缺陷。

---

## 4. 因果链总结（为什么四模型无差异、句级监督更差）

```
中性 prompt → Flow/HiFT 声学先验强 → 情感差异被洗掉（H1）
        + 冻结主干只训 1.48% + LR 快速衰减（H5）
        + Emo-SIM 均值被中性主导（H6）
→ 任何只改 LLM 侧 loss 的方案都只能在一个 ~66 平台上滑动

句级 CE（H2/H3/H4）：
  FiLM 零初始化 → 初始只推 projection → 冻结探针 ↔ projection 形成
  标签回读闭环 → 文本嵌入被推离预训练流形（调制 10×）
  → emotion_encoder 上与 loss_tts 梯度反方向（cos −0.26）
  → loss_tts 收敛点抬高（3.719 vs 3.673）→ WER +1.9pp
  → 声学条件强度不变（KL 0.099 vs 0.108）→ Emo-SIM 不升
```

---

## 5. 推荐的训练框架调整方案（按优先级；含设计 + 需要调整的模块）

### R1（最高优先级，先做）修正生成与评测协议——不训练也能立刻看清问题

**生成侧**：

- `tools/inference_emo_film.py::select_prompt_wav` 扩展为可选"情感匹配 prompt"
  （同说话人同情感参考），并把 prompt 情感写进 GenerationRow；先跑一次
  "四个现有模型 + 情感匹配 prompt" 小规模推理，验证 H1 并得到每个模型的
  实际上限。
- 可选：把 emotion feature 注入 Flow/DiT（类似 EmoCtrl-TTS 对 flow-matching
  的 Aro-Val 条件），让声学渲染不再只依赖 prompt。

**评测侧**（`eval/eval_emo_film.py`）：

- per-emotion Emo-SIM + 5-way nearest-ref 判别准确率（同文本跨情感参考）；
- 同文本跨情感余弦（模型输出 vs GT 38.8 的差距 = 情感区分度损失）；
- 保留 WER/DTW 作质量护栏；报告不再只看一个 Emo-SIM 均值。

### R2（最高优先级）把情感监督从"文本嵌入 CE"移到音频/生成链下游

**首选：emotion2vec 音频奖励 + DPO/GRPO**。仓库内已有
`reference/Emo_PA_code_data/reward_tts.py`（emotion2vec 分段情感分 + CAM++
说话人分）与 verl 入口，需把 token 格式改为本仓库 decoder 的裸 token 序列。

- DPO 偏好（FlexiVoice S1 配方，与现有数据完全同构）：同一说话人同一文本，
  目标情感为 preferred、另一情感为 dispreferred、中性参考作 prompt——ESD
  同文本跨情感对直接可用。
- GRPO 奖励（FlexiVoice S2 / EMORL / Emo-PA 配方）：`r_ser = P(emotion2vec
  分类=目标)` + `r_sv = CAM++ 说话人相似度` + 可选 intensity/VAD 距离。
- 规模参考：EMORL GRPO lr=1e-6、K=16、KL=0.1；Emo-PA lr=1e-6。

**备选/辅助：接通输出侧 span 词级监督**（ADR-0019 的 emotion_head/arousal_head，
代码已实现、测试已绿、从未训练）：

- 为训练 parquet 生成词边界对齐 span 张量（IEMOCAP 5484 条已有词级标签；
  `tools/generate_tagged_jsonl.py` + `cosyvoice/dataset/span_align.py` 已存在）；
- 新配置 `downstream_supervision: enabled`，`emotion_head_weight` /
  `intensity_head_weight` 起步 0.5–1.0；训练时 `emo_loss_weight=0`。
- 预期：监督落在生成因果链下游（lm_output 的 speech-token 区段），目标与
  FiLM 输入分离，不存在标签回读捷径。

**同时**：`emo_loss_weight` 输入 CE 路径默认关闭，仅作诊断开关保留
（观测"文本可分性"，不再作为训练目标）。

### R3（高优先级）FiLM 数值修复——工程最便宜、立刻降 WER

- `EmotionEncoder`：embedding 后加 LayerNorm 或固定缩放（把 ~30 范数降到与
  text embedding 同量级），或对 embedding 做 normalize 初始化。
- `FiLMLayer`：γ 限制在 [1−ε, 1+ε]（如 tanh 缩放），β 加幅度上限，或把
  projection 输出缩放 1/sqrt(dim)；目标调制幅度 ≤0.2–0.5× text 范数。
- 验证：5-epoch 小跑 + 对比 WER / modulation ratio（`/tmp/emofilm_mod_analysis.py`
  可复测）。

### R4（高优先级）条件注入结构改造：时间可变 + 靠近声学侧

三选一（可组合）：

1. **词级 FiLM**：emotion_features 按词边界展开成 token 级序列（IEMOCAP 已有
   词标签），γ/β 随词变化——论文原始形态。
2. **speech-token 侧注入**：情感 embedding 拼接到 speech_token_emb（或每步解码
   的 KV 前缀），让条件直接出现在生成因果链。
3. **learned emotion tokens**：文本序列中插入可学习情感 token
   （`<|emotion_hap|>` 风格，CosyVoice3 instruct 范式），让冻结骨干通过注意力
   自行读取条件，表达空间大于常量仿射。

### R5（中优先级）骨干适配：LoRA / 部分解冻

- Qwen2 attention 投影加 rank 8–32 LoRA（作者代码保留过 LoRA 实验但被注释），
  或解冻最后 4–8 层；配合音频奖励一起训。
- 0.5B 模型 LoRA 在 6×3090 可行；`train_utils_emo.freeze()` 加一组 `lora` 参数。
- 目的：让调制方向通过骨干权重映射为韵律变化，而不是只能走输入嵌入的隐形
  方向。

### R6（中优先级）早停/选点指标改情感感知

- CV 每 epoch（或每 2 epoch）在情感匹配 prompt 下计算 emotion2vec 判别准确率
  或 per-emotion Emo-SIM，用其选 best；loss_tts 仅作质量护栏。
- `EarlyStopTracker` 目前只读单 metric 键，扩展为 metric + aux_metric。

### R7（低优先级）LR/时长/数据

- 放弃 WarmupLR 的 1/sqrt 长期衰减，改 constant（1e-5~1e-4）或线性 warmup +
  余弦；预算以情感 CV 指标收敛为准。**协议与监督修正前不要加大预算**
  （已证明加 epoch 无收益）。
- 数据：先用 ESD 同文本跨情感对吃透 DPO/RL；有余力再并入 Expresso/NVSpeech/
  Emilia 子集。

### 需要调整的模块清单（对应 R1–R7）

| 模块 | 现状 | 调整 |
|---|---|---|
| `tools/inference_emo_film.py` | 固定中性 prompt | 情感匹配/可配置 prompt + 记录 prompt 情感（R1） |
| `eval/eval_emo_film.py` | 仅 Emo-SIM/WER/DTW 均值 | + 5-way 判别、per-emotion、跨情感余弦（R1） |
| `cosyvoice/llm/llm_emotion.py` | 输入侧冻结探针 CE | 默认关闭；保留诊断开关（R2） |
| `cosyvoice/llm/llm_emotion.py` + span 头 | 输出侧头已实现未接线 | span 数据接线 + 配置使能（R2） |
| `cosyvoice/dataset/span_align.py` | 零接线 | 为 parquet 生成 span 张量（R2） |
| `cosyvoice/llm/llm_emotion.py::EmotionEncoder/FiLMLayer` | 范数 ~30、调制 4–10× | 归一化 + γ/β 幅度约束（R3） |
| `cosyvoice/llm/llm_emotion.py` 条件注入 | 句级常量仿射 | 词级 FiLM / speech-token 注入 / learned tokens（R4） |
| `train_utils_emo.freeze()` | 三组冻结 | + LoRA 参数组或解冻尾部层（R5） |
| `train_utils_emo.py::EarlyStopTracker` | 单 metric | metric + aux_metric（R6） |
| `conf/emo_film_*.yaml` | WarmupLR、无情感选点 | LR 策略 + 情感感知选点配置（R7） |
| 训练数据 parquet | 无 span 列 | 新 v2 数据目录（不动 v1 冻结合同） |

---

## 6. CosyVoice3 切换评估（用户提出的方向）

### 6.1 代码与权重可行性（已核实）

- 本仓库代码已含 CosyVoice3 全套支持：`CosyVoice3LM`（`cosyvoice/llm/llm.py`）、
  `CosyVoice3Model`、`CosyVoice3Tokenizer`、`CausalMaskedDiffWithDiT`（DiT
  Flow）、`CausalHiFTGenerator`；`cosyvoice/cli/cosyvoice.py` 检测
  `cosyvoice3.yaml` 即加载。缺少的只是权重与配置。
- 官方 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`（Apache-2.0，含 base 与 `_RL`
  版）2025-12 发布，文件结构：llm.pt / flow.pt / hift.pt /
  speech_tokenizer_v3 / campplus / cosyvoice3.yaml；官方 README 含训练脚本。
- 关键差异：hidden 仍 896、speech_token_size 仍 6561，但特殊 token 布局不同
  （+200 头、sos/task 位置、instruct token、`<|endofprompt|>`）；Flow 换 DiT
  （input_size 80→512）；HiFT 换 CausalHiFT；tokenizer 换 S3Tokenizer v3。
- **数据成本**：现有 parquet 的 speech_token 是 v2 tokenizer 的 id，与 v3 不
  兼容，20774 训练 + 1092 CV + 2500 评测 wav 必须用 v3 重新抽 token——最大
  一项迁移工作。

### 6.2 换基座能带来什么

- CV3 的 speech tokenizer 是 MinMo 多任务监督训练（含 SER）的 FSQ tokenizer，
  token 本身承载情感/韵律信息，直接缓解 H9 的"CV2 token 无语义情感锚点"。
- 官方 RL 版（DiffRO + MTR，含 SER 奖励）在 CER/WER 上显著优于 base
  （zh CER 1.21→0.81），证明该架构上"情感/韵律后训练"有效。
- instruct 接口（情感文本指令）天然比"标签 ID + FiLM"更接近现代
  instruction-following 范式，可承载 R4-3 的 learned emotion tokens。

### 6.3 换基座不能解决什么（关键）

- CV3 仍是 "LLM → prompt 条件化 DiT Flow" 两阶段架构，**用中性 prompt + 只训
  LLM 的路径会重演声学洗刷**（H1 结论与主干无关）。
- CV3 的 LLM 输入协议（prompt text + content 拼接、FSQ prompt token 前缀、
  instruct 模板）与本仓库 target-only 协议不同，情感 FiLM 需要重新适配到
  `CosyVoice3LM`（或直接放弃 FiLM，改用 instruct/RL）。
- 评测口径不变的话，Emo-SIM 均值仍可能看不出差异。

### 6.4 建议动作（按成本递增）

1. **先跑官方 CV3 零样本/指令基线，不微调**：下载权重，用中性 prompt + 情感
   指令生成 ESD 子集，跑升级后的评测。成本 ~1 GPU 天；直接回答"基座本身是否
   已经超过 66 平台"。
2. 若 CV3 基线显著更好：优先走官方训练脚本做情感 SFT/RL，而不是在本 fork 上
   移植 FiLM；若需保留 FiLM 研究价值再移植到 `CosyVoice3LM`。
3. 若 CV3 基线仍 ~66：问题确在协议（prompt/Flow/评测），换基座无意义，回到
   R1/R2。

**结论：换，但不要单独换。** 单独换基座大概率只是把平台平移几个点；把它与
R1（协议）+ R2（音频奖励/span）+ R3（FiLM 修复）捆绑才是"有真正提升的训练
方案"。若只做一次大改，建议 R2-span + R4-词级条件 + CosyVoice3 基座三者捆绑。

---

## 7. 外部文献佐证（2024–2026，一手核对）

- **Emo-FiLM / Emo-PA（arXiv 2509.20378）**：方法核心 = emotion2vec 词级连续
  特征 + 词级 FiLM + 词级预测模型；消融证明词级数据与辅助 loss 关键。本仓库
  复现的只是其中"输入端冻结探针"这一最弱部件。
- **FlexiVoice（arXiv 2601.04656）**：S1 = ESD 同文本跨情感 DPO（目标情感
  preferred / 其他 dispreferred、中性 prompt）；S2 = GRPO（r_ser + r_sv）。
  与我们的数据与问题完全同构，是**最可直接复用**的配方。
- **EMORL-TTS（arXiv 2510.05758）**：SFT + GRPO（SER 奖励 + VAD 强度 + 重音）；
  其 CosyVoice2 基线"中性参考 + 文本提示"5 类平均准确率仅 0.63
  （neutral 0.99、ang 0.56、hap 0.70、sad 0.48、sur 0.44）——与本仓库
  "中性主导、非中性弱"现象完全一致。
- **Emo-DPO（arXiv 2409.10157, ICASSP 2025）**：直接偏好优化区分正负情感对，
  报告比单纯 CE 更强的情感可控性。
- **CosyVoice3（arXiv 2505.17589 + 官方 2025-12 发布）**：SER 监督 tokenizer +
  DiffRO/MTR——"情感奖励直接在 token 层优化"的官方实现。
- **GLM-TTS（2025-12）**：多奖励 GRPO（Similarity/CER/Emotion/Laughter）提升
  情感表达，RL 版 CER 1.03→0.89。
- **Llasa-GRPO / TEMOTTS / 《The False Resonance》(arXiv 2604.26347)**：
  同一趋势——LLM-TTS 情感控制的最优解是 preference/reward 后训练，而不是
  额外输入侧 CE；emotion2vec 余弦指标存在系统性盲区。

---

## 8. 建议的下一步隔离实验（可直接验证，按成本升序）

1. **协议验证（半天~1 GPU 天）**：四个现有模型 + 情感匹配 prompt 生成 ESD
   子集 + 升级评测 → 直接测量每个模型的声学情感实际上限（R1）。
2. **span 头接线冒烟（1–2 GPU·小时）**：1000 条 IEMOCAP 词级数据 → span_align
   → 训练 2–3 epoch（`emo_loss_weight=0`）→ 对比 CV loss_tts / WER / Emo-SIM。
3. **FiLM 数值修复冒烟（1–2 GPU·小时）**：归一化 + γ/β 约束，5 epoch，复测
   modulation ratio 与 WER（R3）。
4. **词级 FiLM 冒烟（2–3 GPU·天）**：现有 26% 词级数据上最小实现，对比
   sentlvl/disabled 的判别 acc 与 F0 差异（R4）。
5. **LoRA 消融（2–3 GPU·天）**：± LoRA rank 16，观察 next-token KL 与判别
   acc 是否移动（R5）。
6. **CV3 零样本基线（~1 GPU 天）**：官方权重 + 情感指令生成 ESD 子集 + 升级
   评测（§6.4）。

---

## 9. 局限说明

- copy-synthesis / token-carry 各 30 样本、5-way 判别 60 样本：样本量小，但
  方向与全量评测一致（模型生成部分与全量 Emo-SIM 吻合），且是结构性结论
  （93.3%→53.3%）。
- §3.3 的 KL/top-1 用单句 + 随机 speech token 的 teacher-forced 前向测量，
  只作机制佐证。
- 梯度探针用合成 batch，绝对值有批次噪声，但"读回 100%、unmodulated chance、
  emotion_encoder 梯度反方向"是结构性的。
- 未做"情感匹配 prompt + 现有四模型"的全量生成（建议作为第一个验证实验）。
- 文献代理未返回，文献结论均经两个主报告代理对一手来源直接核对。

---

## 10. 证据与产物索引

- 四次实验：`exp/emofilm_{v1,film_only,film_only_longepoch,sentlvl}/`
  （train.log / train_identity.json / final.pt / full/ / eval/）。
- 本次报告：`docs/reports/2026-08-02-emofilm-why-plateau-investigation.md`、
  `docs/reports/2026-08-02-emofilm-rootcause-research.md`（本文件为最终合并版）。
- 隔离实验（可复跑）：`/tmp/copy_synthesis.py`、`/tmp/copy_ceiling_eval.py`、
  `/tmp/token_carry_test.py`、`/tmp/token_carry_eval.py`、
  `/tmp/emo_discriminability.py`、`/tmp/emofilm_cross_emo.py`、
  `/tmp/emofilm_mod_analysis.py`、`/tmp/emofilm_llm_sens2.py`、
  `/tmp/grad_conflict_probe.py`（输出 `/tmp/grad_conflict_probe.json`）。
- 上游参考：`reference/Emo_PA_code_data/reward_tts.py`、
  `reference/arXiv-2509.20378v1`、`reference/2601.04656v1.pdf`。
