# EmoFiLM 四次实验平台根因调研与训练方案建议

- 日期：2026-08-02
- 范围：v1 / 5-epoch disabled / 27-epoch disabled（早停）/ sentlvl（句级监督）四次
  实验为何 Emo-SIM 均卡 ~66 平台、且句级监督不及纯 loss_tts；给出有真正提升的
  训练方案；评估切换 Fun-CosyVoice3-0.5B-2512 的可行性。
- 方法：源码与产物核对 + 复用小样本隔离实验（copy-synthesis、token-carry、5-way
  判别、梯度探针、参数诊断）+ 联网论文/开源项目调研（CosyVoice3、Emo-PA、
  FlexiVoice、EMORL-TTS、GLM-TTS、Llasa-GRPO）。
- 结论性质：除标注为"推断"的条目外，均有本仓库可复跑的产物/脚本/数值支撑。

---

## 0. 一句话结论

平台不是"训练不够"或"监督不够强"造成的，而是**生成协议（固定中性 prompt 声学
条件）与评测协议（Emo-SIM 均值余弦）在情感维度上共同钳制**的结果：即使把目标
音频自身的 speech token 完美地喂进声学管线，只要声学 prompt 仍是中性参考，
emotion2vec 五选一判别准确率也只有 ~53%（copy-synthesis 同条件下 93%）。
句级监督之所以更差，是因为它优化的是"FiLM 后文本嵌入可被随机冻结读出器分对类"
这一**文本侧捷径**（CE 0.07、读回准确率 100%），其梯度与 loss_tts 在 emotion
encoder 上方向相反（cos ≈ −0.26），把 FiLM 调制幅度从 disabled 的 ~4.5× 推到
~10.4×（文本嵌入范数的倍数），抬高 cv loss_tts（3.719 vs 3.673）与 WER
（ESD +1.9pp），对声学情感零收益。

真正有提升的路径是：**把监督从"文本嵌入 CE"移到"音频/生成链下游"（emotion2vec
音频奖励的 DPO/GRPO，或 CosyVoice3 的 DiffRO/MTR）+ 修正生成协议（情感匹配
prompt 或 Flow 侧情感条件）+ 修复 FiLM 数值病理 + 解冻/LoRA 主干 + 用情感感知
指标选点**。切换 CosyVoice3 可行且值得尝试（本仓库代码已具备全套 CV3 类、官方
权重与训练脚本已发布），但**单独换基座不能突破平台**——它必须与上述监督与协议
修正配套，且应先用官方 CV3 零样本/指令模型跑一个基线再决定。

---

## 1. 已核实的事实基线（全部可追溯）

### 1.1 四次实验的客观评测（baseline `eval/eval_emo_film.py` 同口径）

| 数据集 | 指标 | v1（有 loss_emotion） | 5-epoch disabled | 27-epoch disabled | sentlvl |
|---|---|---|---|---|---|
| ESD | WER% / Emo-SIM / DTW_norm | 9.48 / 66.75 / 0.332 | 8.18 / 66.11 / 0.339 | 8.38 / 65.89 / 0.341 | 10.05 / 65.45 / 0.345 |
| FEDD-A | WER% / Emo-SIM / DTW_norm | 8.30 / 81.94 / 0.178 | 4.70 / 82.71 / 0.171 | 4.69 / 82.46 / 0.173 | 6.54 / 82.91 / 0.169 |
| FEDD-B | WER% / Emo-SIM / DTW_norm | 14.42 / 61.60 / 0.384 | 12.30 / 62.98 / 0.370 | 11.57 / 64.31 / 0.357 | 14.04 / 63.19 / 0.368 |

产物：`exp/emofilm_{v1,film_only,film_only_longepoch,sentlvl}/eval/*_metrics.json`。

### 1.2 训练侧事实（日志核对）

- v1：`exp/emofilm_v1/train.log` 显示 optimizer 只有 **1 组** `emotion_new`
  （6 tensors / 7,504,292 params / lr=1e-5，即 emotion_encoder + emotion_adapter +
  llm_decoder 一体训练），scheduler 为 constantlr；`conf` 中 `emo_loss_weight=0.2`，
  冻结 `emotion_classifier`（`git show 9c6d84b:cosyvoice/llm/llm_emotion.py` L54-57）。
- sentlvl：3 组（FiLM 1e-4 / heads 1e-4 / decoder 1e-5）+ WarmupLR(warmup=250)。
  WarmupLR 在 step>250 后按 `1/sqrt(step)` 衰减：epoch 16 时实际 lr ≈ 7.5e-6
  （FiLM）/ 7.5e-7（decoder）——训练后期学习率极低。
- sentlvl 的 `loss_emotion_input`：batch 0 ≈ 1.70 → batch 500 ≈ 0.42 → batch
  1000 ≈ 0.15 → 之后长期 ≈ **0.07–0.08**（5 类 CE 的理论 ln5≈1.61，0.07 已接近
  完美可分）；CV 同值。即"冻结随机分类器读出情感"在几百步内就被打满。
- sentlvl cv_loss_tts 收敛于 **3.719**（best@14），27-epoch disabled 为 **3.673**
  （best@21）——句级监督使 TTS 损失收敛点明显抬高，与 WER 劣化方向一致。

### 1.3 评测/生成协议事实

- ESD / FEDD-B 推理的声学 prompt 固定为**同说话人 Neutral wav**
  （如 ESD speaker 0011 全部 1500 条都使用 `0011/Neutral/0011_000001.wav`；
  `tools/inference_emo_film.py::select_prompt_wav` + manifest `prompt_wav`）。
  FEDD-A 为构造数据自带的 `*_neutral_anchor.wav`。
- v2 单流协议下 prompt **不进 LLM 条件**，只作为 Flow/HiFT 的声学条件
  （`model_emo.py`：`flow_prompt_speech_token/prompt_speech_feat/flow_embedding`
  透传 Flow/HiFT；`Qwen2LM_Emotion.decode` 的 lm_input 只有
  `[SOS, FiLM(text), task_id]`）。
- 评测 ref = 数据集目标情感 WAV（`reference_wav == target_wav`）；Emo-SIM =
  emotion2vec_plus_large frame 均值余弦 ×100，DTW 为 frame cosine DTW。

---

## 2. 根因一：四模型"无差异"是声学协议 + 评测协议共同钳制

### 2.1 决定性隔离实验（30 样本 = 5 情感 × 6，同说话人同文本，`/tmp` 可复跑）

| 条件 | same 情感 Emo-SIM | gap(same−cross) | 5-way 判别 acc |
|---|---|---|---|
| copy-synthesis（目标音频自身 token + 自身声学 prompt） | 97.17 | 52.67 | **93.3%** |
| 参考 token + 真实中性 prompt 声学（同正式生成链路） | 71.22 | 21.02 | **53.3%** |
| sentlvl 模型生成 | 65.44 | 13.95 | 36.7% |
| 27-epoch disabled 模型生成 | 69.34 | 17.69 | 43.3% |
| 5-epoch disabled 模型生成 | 67.30 | 16.01 | 50.0% |
| v1 模型生成 | 70.05 | 18.55 | 43.3% |

脚本：`/tmp/copy_synthesis.py`、`/tmp/copy_ceiling_eval.py`、
`/tmp/token_carry_test.py`、`/tmp/token_carry_eval.py`。

含义：
1. 离散管线（LLM token → Flow → HiFT）**本身不丢情感**：token 和声学 prompt
   都匹配时 5-way 判别 93.3%。
2. **声学 prompt 是主导条件**：token 已经是目标音频原样提取（完美 token），只要
   声学 prompt 换成中性参考，判别 acc 立刻掉到 53.3%、gap 缩到 21。
3. 四个模型生成（token 由各自 LLM 产出）全部低于"参考 token + 中性 prompt"
   的上限——LLM 侧无论如何优化，声学渲染这一关把情感差异"洗掉"了大半。

### 2.2 音频层面的独立佐证

- 数据集参考音频（GT）同文本跨情感 emotion2vec 余弦均值 **38.83**（median 37.10）
  ——参考空间本身可分；
- 四个模型生成音频的同文本跨情感余弦：v1 79.96 / 5-epoch 86.92 / 27-epoch 77.67
  / sentlvl 84.27——**模型输出的不同情感彼此几乎不区分**；
- 同一 utt 跨模型生成音频余弦 0.72–0.99（多数 >0.85）——四模型输出高度同质；
- 5-way 判别（60 样本）：sentlvl 41.7% / 27-epoch 41.7% / 5-epoch 46.7% / v1 50.0%，
  mean Emo-SIM by ref emotion：neutral 76–81，ang/hap/sad/sur 仅 32–54——**全局
  Emo-SIM ~66 主要是"中性参考相似度"贡献的**，非情感控制能力强。
  （`/tmp/emo_discriminability.py` + `emo_discriminability.json`、
  `/tmp/emofilm_cross_emo.py` + `cross_emo_out.log`。）

### 2.3 模型侧同质化因素

- 冻结 LLM backbone，只训 1.48% 参数（FiLM 4 张量 + decoder 2 张量 + 头 4 张量）。
- 实际有效 LR 低且快速衰减（见 1.2）。四模型都从同一 `llm.pt` 出发，改动面极小，
  因此生成音频高度同质是预期内结果；"没有详细差距"首先应归因于**干预强度/协议**
  ，其次才是指标不敏感。
- 注意：LLM 的 speech-token **分布其实被情感明显改变**（见 §3.3），四模型在
  token 层并不完全相同；只是这些差异在声学渲染后被压缩，且 Emo-SIM 均值指标
  看不到。

### 2.4 小结（因果链）

`中性 prompt → Flow/HiFT 声学先验强 → 情感差异被洗掉` + `Emo-SIM 均值被中性主导`
→ 任何"只改 LLM 侧 loss"的方案（loss_tts 或输入 CE）都只能在这个 ~66 平台上
微调；token 层做得再好，也到不了"参考 token + 中性 prompt"的 53% acc 上限，
更到不了 copy-synthesis 的 93%。

---

## 3. 根因二：句级监督更差——文本可分捷径 + 梯度冲突 + 调制病理

### 3.1 捷径（隔离梯度探针，`/tmp/grad_conflict_probe.py`，sentlvl final 实测）

| 量 | 值 | 含义 |
|---|---|---|
| 冻结分类器在 FiLM 后 text emb 上的 CE / acc | 0.0718 / **100%** | 文本嵌入被推到随机子空间可完美分对 |
| 同一分类器在未调制 text emb 上的 CE / acc | 1.786 / 16.7%（chance） | 情感信息只由 FiLM 注入 |
| loss_emotion 梯度 vs loss_tts 梯度范数比（FiLM 参数） | 3.5% | 数值占比不大 |
| 二者余弦（emotion_adapter / emotion_encoder） | −0.022 / **−0.26** | emotion_encoder 上明显与 TTS 目标反方向 |

训练日志已显示 loss_emotion_input 数百步内打到 ~0.07；探针确认"文本可分"
100% 达成。**这个 CE 证明的只是"FiLM 输出可读出条件 ID"，与声学输出无关**——
这正是 ADR-0021 已知风险的实证。

### 3.2 FiLM 调制病理（`/tmp/emofilm_mod_analysis.py` 实测）

| 模型 | projection W 范数 | γ 偏离恒等 | 调制幅度 ‖Δ‖/‖text‖ | decoder 相对 base 漂移 |
|---|---|---|---|---|
| v1 | 5.91 | 1.33 | **8.4–9.2×** | 8.53% |
| sentlvl | 11.84 | 2.47 | **10.1–10.8×** | 5.92% |
| 5-epoch disabled | 9.02 | 1.79 | 4.4–5.6× | 4.20% |
| 27-epoch disabled | 8.66 | 1.57 | 3.9–5.2× | 6.43% |

补充事实：emotion/intensity embedding 初始化为标准正态（norm ≈ 29–30，text
embedding norm ≈ 0.93）；FiLM 把 ~30 范数的 emotion feature 乘出 γ/β，使 text
embedding 被 **4–10 倍于自身的扰动**覆盖。句级 CE 训练的两版（v1/sentlvl）调制
幅度最大、WER 最差（9.48 / 10.05 vs 8.18 / 8.38）。这不是"FiLM 没动"，而是
**动过头且方向不对**。

### 3.3 澄清一个易误读的点：LLM token 层其实"有反应"

`/tmp/emofilm_llm_sens2.py`（sentlvl final，单句）：
KL(neu‖hap)=0.25、KL(neu‖sad)=0.29、KL(neu‖ang)=0.21、KL(neu‖sur)=0.20，
top-1 一致率仅 26–39%；跨情感 token set Jaccard 0.02–0.15（四模型同量级）。
即 **LLM 输出分布确实随情感大幅改变**，但：a) 这些改变未被声学渲染保留（§2.1）；
b) 其中很大一部分来自"为让随机读出器可分"的文本侧扭曲而非声学上正确的韵律
模式（WER 劣化佐证）。因此"监督没训到模型"的说法不准确——准确说法是：
**监督把模型训到了错误的（文本侧）目标上，且声学侧根本没被监督到**。

### 3.4 对"可训练的有语义锚点分类器"建议的验证

在 sentlvl final 上把冻结分类器换成可训练（同 batch 训 300 步）：
CE 0.0718 → 8.9e-5，且 FiLM 上的 CE 梯度范数从 0.66 缩到 0.0017——**可训练分类
器确实能吸收残余梯度**。但这不能解决捷径：
- 在 init（FiLM=恒等）时，text emb 不含情感信息，分类器**只能**通过推动 FiLM
  制造可分性来降 CE（init 的 g_emo 范数 8.5 全部落在 emotion_adapter 上）——
  可训练与否都必须先动 FiLM；
- 训练后可训练分类器甚至会学文本-情感伪相关（unmodulated acc 由 16.7% 升到
  50%，2 样本 batch 上的过拟合信号）。

结论：**输入端分类器的"语义锚点"必须落在声学空间**（emotion2vec 对生成音频
打分，或对生成链下游的 speech-token hidden 监督），而不是把随机/可训练线性头
放在文本嵌入上。仓库已实现的 span 词级监督（ADR-0019）方向正确但仍在 LLM
hidden 层，**建议以音频奖励为主、span 头为辅**。

---

## 4. 发现清单（按置信度从高到低）

1. **[高] 声学 prompt 钳制是平台主因**：中性 prompt 下，即使参考 token 原样
   重建，emotion2vec 5-way acc 也只有 53.3%（copy-synthesis 93.3%）。
   证据：§2.1。影响：任何 LLM 侧训练在现有生成协议下都有 ~53% 的情感上限。
2. **[高] 输入端句级 CE 是文本可分捷径，结构性无法提升声学情感**：0.07 CE /
   100% 读回；与 loss_tts 梯度在 emotion_encoder 上反方向（−0.26）；v1/sentlvl
   无收益且 WER 劣化。证据：§1.2、§3.1、四方对比。
3. **[高] 当前 Emo-SIM 均值指标对情感差异不敏感且被中性主导**：neutral ref
   sim 76–81 vs 其他 32–54；四模型跨情感余弦 78–87 vs GT 39；同 utt 跨模型
   余弦 0.72–0.99。证据：§2.2。影响：需要补 per-emotion 判别/分类指标。
4. **[高] FiLM 数值病理**：emotion feature 范数 ~30、调制 4–10× text 范数、
   γ 偏离恒等 1.3–2.5，直接贡献 WER 劣化与训练不稳定。证据：§3.2。
5. **[高] 干预面过小 + 低有效 LR 使四模型同质**：冻结 backbone（1.48% 可训练）、
   WarmupLR 250 步后 1/sqrt 衰减到 ~7e-6（FiLM）。证据：§1.2、§2.3。
6. **[中高] 早停/选点指标错位**：cv loss_tts 是情感弱代理（27-epoch best@21
   在 ESD 上反而略降）。证据：2026-08-01 报告 + 四方对比。
7. **[中高] 换 CosyVoice3 可行但非充分**：本仓库已含 CV3 全套类（CosyVoice3LM/
   Tokenizer/Model + DiT Flow + CausalHiFTGenerator），官方 2025-12 发布
   base/RL 权重 + 训练脚本（Apache-2.0）；CV3 tokenizer 带 SER 多任务监督、
   RL 模型已验证情感/韵律后训练收益。但 CV3 仍是"LLM → prompt 条件化 Flow"架构，
   同样的声学洗刷风险存在；且其输入协议（instruct/prompt text + FSQ 前缀）与
   本仓库 target-only 协议不同。证据：§7、§8。
8. **[中] LR/时长/调度**：WarmupLR 快速衰减 + 5–27 epoch 的预算可能不足；
   upstream SFT 用 constant 1e-5、200 epoch；EMORL-TTS SFT 用 2e-4、50 epoch。
9. **[中] 数据规模**：ESD ~17.5k 句、说话人 10 个，四模型都过拟合/欠拟合到同一
   小分布；FlexiVoice/EMORL 均引入更大或互补数据（Emilia/Expresso/NCSSD）。
10. **[中低] 可训练分类器不解决输入端起捷径**（见 §3.4）；"语义锚点"应放在
    声学/生成链下游。

---

## 5. 推荐的训练框架调整（按优先级）

### 5.1 P0：修正生成/评测协议（不训练也能立刻看清问题）

- **推理 prompt 策略**：将 `tools/inference_emo_film.py::select_prompt_wav` 扩展为
  可选"情感匹配 prompt"（同说话人同情感参考，或情绪无关的说话人锚点 + 指令），
  并把 prompt 情感写进 GenerationRow。先跑一次"四个现有模型 + 情感匹配 prompt"
  的小规模推理，验证声学钳制假设并得到每模型的实际上限。
- **Flow 侧情感条件**（可选但推荐）：把 emotion feature 注入 Flow/DiT（类似
  EmoCtrl-TTS 对 flow-matching 的 Aro-Val 条件），让声学渲染不再只依赖 prompt。
- **评测指标**：`eval/eval_emo_film.py` 增加
  (a) per-emotion Emo-SIM 与 5-way nearest-ref 判别准确率（同文本跨情感参考，
  复用 `/tmp/emo_discriminability.py` 逻辑）；
  (b) 同文本跨情感余弦（模型输出 vs GT 38.8 的差距即"情感区分度损失"）；
  (c) 保留 WER/DTW 作质量护栏。正式报告不再只看一个 Emo-SIM 均值。

### 5.2 P0：把情感监督从"文本嵌入 CE"移到音频/生成链下游

- **首选：emotion2vec 音频奖励 + DPO/GRPO**。可直接复用仓库内
  `reference/Emo_PA_code_data/reward_tts.py`（emotion2vec 分段情感分 + CAM++
  说话人分，verlen GRPO 入口 `run_dual_gpu.sh`），但需把 token 格式从
  `<|s_i|>` 字符串改成我们 decoder 的裸 token 序列，并接入 `train_emo.py` 的
  rollout 循环（或直接跟 upstream verl 流程）。
  - DPO 偏好构造（FlexiVoice S1 配方，完全匹配我们现有数据）：
    同一说话人同一文本，目标情感为 preferred、另一情感为 dis-preferred，
    中性参考作 prompt——这正好复用 ESD 同文本跨情感对。
  - GRPO 奖励（FlexiVoice S2 / EMORL 配方）：`r_ser = P(emotion2vec 分类=目标)`
    + `r_sv = CAM++ 说话人相似度` + 可选 intensity/VAD 距离奖励；
    用"目标情感指令 vs 冲突情感参考"构造解耦场景。
  - 规模参考：EMORL GRPO lr=1e-6、K=16、KL 0.1，8×4090；Emo-PA lr=1e-6。
- **备选/辅助：现有 span 词级监督**（`emotion_head/arousal_head`，ADR-0019）接通
  数据链（`span_align.py` 目前零接线）。它至少把监督移到生成链下游、避开输入端
  标签回读；但它仍读 LLM hidden，不是音频，因此**只作为辅助损失**。
- **移除/默认关闭** `emo_loss_weight` 输入 CE 路径（保留代码作诊断开关即可，
  不再作为训练目标）；`loss_emotion_input` 只用于观测"文本可分性"。

### 5.3 P1：FiLM 数值修复（工程上最便宜、立刻降 WER）

- `EmotionEncoder`：embedding 后加 LayerNorm 或固定缩放（把 ~30 范数降到与 text
  emb 同量级），或对 emotion/intensity embedding 做 `normalize_` 初始化。
- `FiLMLayer`：把 γ 限制在 [1−ε, 1+ε]（如 tanh 缩放）、β 加幅度上限，或把
  projection 输出缩放 1/sqrt(dim)；目标是调制幅度 ≤0.2–0.5× text 范数。
- 验证：5-epoch 小跑 + 对比 WER / modulation ratio（`/tmp/emofilm_mod_analysis.py`
  可复测）。

### 5.4 P1：解冻/LoRA 主干

- 当前冻结 Qwen2 backbone，情感条件只能经 text embedding 和 decoder 影响输出；
  token 分布虽变，但被声学侧洗掉。建议对 backbone 用 LoRA（qwen2 各层
  q/k/v/o），或至少解冻最后 4–8 层；配合音频奖励一起训。
- 数据量小（~20k），LoRA rank 8–16 + 1e-4~1e-5 + KL/weight decay 防灾难遗忘。
- 这也直接回应"预训练模型太成熟、微调不加扰动难提升"的直觉：扰动应加在主干
  而非只在输入嵌入上，并且要由音频奖励定向。

### 5.5 P1：早停/选点指标换情感感知指标

- CV 每 epoch 计算（或每 2 epoch）小集 emotion2vec 判别准确率 / 情感匹配 prompt
  下的 Emo-SIM；用其选 best checkpoint；`loss_tts` 仅作质量护栏（阈值 + 早停
  下限）。
- 现有 `EarlyStopTracker` 只读单个 metric 键，扩展为 `metric=loss_tts` +
  `aux_metric=emotion_acc`（或直接换键），改动集中且可测。

### 5.6 P2：LR/训练时长

- 放弃 WarmupLR 的 1/sqrt 长期衰减，改 constant（FiLM/backbone 1e-5~1e-4）或
  线性 warmup + 余弦；确保情感阶段的有效 LR 不提前塌缩。
- 训练预算以情感 CV 指标收敛为准（可能 >30 epoch），但**在协议与监督修正前
  不要加大预算**（已证明加 epoch 无收益）。

### 5.7 P2：数据

- 优先用偏好/RL 构造吃透 ESD（同文本跨情感天然构成 DPO 对）；
- 有余力再并入 Expresso（FlexiVoice/EMORL 均使用）、NVSpeech/Emilia 子集。

---

## 6. 关于切换 Fun-CosyVoice3-0.5B-2512 的评估

### 6.1 可行性（已核实）

- 本仓库 `cosyvoice/cli/cosyvoice.py` 已支持 `CosyVoice3`（`AutoModel` 检测
  `cosyvoice3.yaml` 即加载），`CosyVoice3Model`/`CosyVoice3LM`/
  `CosyVoice3Tokenizer`/`CausalMaskedDiffWithDiT`/`DiT`/`CausalHiFTGenerator`
  全部存在；缺少的只是权重 + `conf/cosyvoice3.yaml`。
- 官方权重：ModelScope/HF `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`（含 base 与
  `_RL` 两版，Apache-2.0，文件结构与 CV2 类似：llm.pt / flow.pt / hift.pt /
  speech_tokenizer_v3 / campplus / cosyvoice3.yaml）。官方 README 明确
  2025-12 发布 base/RL 模型 + 训练/推理脚本。
- CV3 yaml（镜像核验）：`CosyVoice3LM`（speech_token_size 6561）、DiT Flow、
  CausalHiFT、dynamic batch、官方 SFT lr=1e-5、200 epoch。

### 6.2 换基座能带来什么

- CV3 的 speech tokenizer 是 MinMo 多任务监督训练（含 SER）的 FSQ tokenizer，
  **token 本身承载情感/韵律信息**——直接缓解本仓库"CV2 token 无语义情感锚点"
  的问题；
- 官方 RL 版（DiffRO + MTR，含 SER 任务奖励）在 CER/WER 上显著优于 base 版
  （zh CER 1.21→0.81，test-hard CER 6.71→5.44），证明该架构上"情感/韵律后训练"
  有效；
- CV3 的 instruct 接口（情感指令文本）天然比"标签 ID + FiLM"更接近现代
  instruction-following 范式。

### 6.3 换基座不能解决什么（关键）

- CV3 仍是 "LLM → prompt 条件化 Flow(DiT)" 两阶段架构，声学输出同样强依赖
  prompt；**用中性 prompt + 只训 LLM 的复现路径会重演声学洗刷**（§2.1 结论
  与主干无关）。
- CV3 的 LLM 输入协议（prompt text + content 拼接、FSQ prompt token 前缀、
  instruct 模板）与本仓库 target-only 协议不同，情感 FiLM 需要重新适配到
  `CosyVoice3LM`（或直接放弃 FiLM，改用 instruct/RL）。
- 换基座后评测口径若不变，Emo-SIM 均值仍可能看不出差异；必须同步升级指标。

### 6.4 建议动作（按成本递增）

1. **先跑官方 CV3 零样本/指令基线，不微调**：下载权重，用中性 prompt + 情感
   指令（如 "Use angry emotion to read it"）生成 ESD 2500 条，跑升级后的评测。
   成本 ~1 GPU 天；直接回答"基座本身是否已经超过 66 平台"。
2. 若 CV3 基线已显著更好：优先走**上游官方训练脚本**做情感 SFT/RL（Dec 2025
   已发布），而不是在本 fork 上移植 FiLM；若需要保留 FiLM 研究价值，再移植到
   `CosyVoice3LM`。
3. 若 CV3 基线仍 ~66：问题确在协议（prompt/Flow/评测），换基座无意义，回到
   §5.1/§5.2。

---

## 7. 外部论文/开源项目佐证（2024–2026）

- **Emo-PA（`reference/arXiv-2509.20378v1` + `Emo_PA_code_data`）**：输入端 CE
  只是其 SFT 部分；论文/代码里真正带来情感收益的是 **GRPO 多目标奖励**
  （`reward_tts.py`：emotion2vec 分段情感 + CAM++ 说话人；`run_dual_gpu.sh`
  verl 入口）。其 Emo-SIM 97–99 来自 RL 与不同评测口径，不能作为本仓库
  baseline。
- **FlexiVoice（arXiv 2601.04656）**：S1 用 ESD 同说话人同文本、目标情感为
  preferred / 其他情感为 dis-preferred、中性参考做 prompt 的 DPO；S2 用
  `r_ser`（emotion2vec 概率）+ `r_sv`（CAM++）在冲突场景做 GRPO。与我们数据
  与问题完全同构，是最可直接复用的配方。
- **EMORL-TTS（arXiv 2510.05758）**：SFT（ESD+Expresso，lr=2e-4，50 epoch）
  + GRPO（SER 奖励 + VAD 强度奖励 + 重音奖励）；其外部数字显示"中性参考 +
  文本情感提示"的 CosyVoice2 基线 5 类平均准确率仅 0.63（neutral 0.99、
  angry 0.56、happy 0.70、sad 0.48、surprise 0.44）——与本仓库"中性主导、
  非中性弱"现象完全一致。
- **CosyVoice3（arXiv 2505.17589 + 官方 2025-12 发布）**：SER 监督 tokenizer +
  DiffRO/MTR（Token2Text + SER 等可微 token 级奖励）——"情感奖励直接在 token
  层优化"的官方实现。
- **GLM-TTS（zai-org/GLM-TTS，2025-12）**：多奖励 GRPO（Similarity / CER /
  Emotion / Laughter）提升情感表达，RL 版 CER 1.03→0.89。
- **Llasa-GRPO / Emo-DPO 等**：同一趋势——LLM-TTS 情感控制的最优解是
  preference/reward 后训练，而不是额外的输入侧 CE。

---

## 8. 证据与产物索引

- 四次实验：`exp/emofilm_{v1,film_only,film_only_longepoch,sentlvl}/`
  （train.log / train_identity.json / full/ / eval/）。
- 报告：`docs/reports/2026-07-20-emofilm-v1-baseline-experiment-report.md`、
  `2026-07-30-emofilm-film-only-experiment-report.md`、
  `2026-08-01-emofilm-longepoch-convergence-comparison.md`、
  `2026-08-02-emofilm-sentlvl-experiment-report.md`。
- 隔离实验（可复跑）：`/tmp/copy_synthesis.py`、`/tmp/copy_ceiling_eval.py`、
  `/tmp/token_carry_test.py`、`/tmp/token_carry_eval.py`、
  `/tmp/emo_discriminability.py`、`/tmp/emofilm_cross_emo.py`、
  `/tmp/emofilm_mod_analysis.py`、`/tmp/emofilm_llm_sens2.py`、
  `/tmp/grad_conflict_probe.py`（输出 `/tmp/grad_conflict_probe.json`）。
- 上游参考：`reference/Emo_PA_code_data/reward_tts.py`、
  `reference/arXiv-2509.20378v1`、`reference/2601.04656v1.pdf`（FlexiVoice）。
- 在线资料：CosyVoice3 论文 arXiv 2505.17589；官方 README（ModelScope/HF，
  镜像核验 gitserver.onethingai.com）；EMORL-TTS arXiv 2510.05758；GLM-TTS
  GitHub README；FlexiVoice arXiv 2601.04656。

## 9. 局限说明

- copy-synthesis / token-carry 各 30 样本、5-way 判别 60 样本，样本量小但方向
  与全量评测一致（模型生成部分与全量 Emo-SIM 吻合）。
- §3.3 的 KL/top-1 用单句 + 随机 speech token 的 teacher-forced 前向测量，只作
  机制佐证，不作定量结论。
- 梯度探针用合成 batch（随机 text/speech token），绝对值有批次噪声，但
  "读回 acc 100%、unmodulated chance、emotion_encoder 梯度反方向"是结构性的。
- 未对"情感匹配 prompt + 现有四模型"做全量生成（未跑，建议作为第一个验证
  实验）；声学钳制结论由 token-carry 实验支撑，置信度已足够高。
