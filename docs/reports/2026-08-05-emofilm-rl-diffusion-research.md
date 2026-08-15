1. 没有 token 级 CV3 预解码入口。这是最大的未声明缺口：现网只有 wav 级推理，预解码必须新增 LLM-only 解码路径，且要和
   inference_instruct2 完全同语义（文本规范化分句、instruct 模板、RAS 采样、EOS/fill 处理）。报告把它写成"预解码符号即可"，低估了
   实现面。
2. "同文本跨情感组"与 GRPO 组语义冲突。标准 GRPO 的组 = 同一条件重复采样；"同文本跨情感"是不同条件（目标情感不同 → LLM 预解码符号
   不同），不能直接当作组内候选。数据清单必须分两层：condition_id（GRPO 组）和 contrast_key（跨情感对照/冲突数据），否则阶段 2 的
   奖励服务会按错键分组。这是设计歧义，不是代码问题，但会阻塞数据 schema。
3. 文本口径必须拆分。v1 train parquet 的 text 是带 <emotion></emotion> 标签的文本，而 CV3 instruct2 的 tts_text 应传纯文本、情感走 instruct
   指令、奖励又要用带标签文本作 ground_truth。阶段 1 需要显式派生三份文本（plain text / instruct / tagged ground_truth），报告没有
   提这一步；不处理会导致预解码质量劣化且奖励解析错位。
4. embedding 来源要按条件取。v1 parquet 的 spk_embedding 来自目标音频；RL 条件的 embedding 必须来自 prompt wav（flow 的说话人条
   件）。parse_embedding 默认会从行内音频现算，若直接复用 v1 列，flow 条件会拿到错误的说话人向量。需要条件视图显式绑定 prompt 音频
   的 embedding。
5. 对齐与过滤约束：v3 管线有 token_max_length=200、max_frames_in_batch=2000、token_mel_ratio=2 的裁剪；预解码的生成 token 长度可能
   突破过滤上限，且监督流需要 target token 与 mel 严格对齐。不是阻塞，但数据构建时要按同一阈值校验，避免训练时被 filter 静默丢弃或
   trim_token_mel 悄悄截断。
6. 冲突构造的数据可用性：ESD 同说话人 5 情感齐全，语义/音色冲突可行；IEMOCAP 词级 5484 条未必有同文本跨情感对，冲突构造需要按说话
   人建 prompt 池（作者脚本的 NCSSD 命名假设不适用）。属于可解决的适配工作，但要早做 manifest 验证。
7. CausalConditionalCFM 初始噪声固定（上一轮已确认）会直接影响"同条件多候选"的定义；阶段 1 的清单构建不应假定 flow 采样已有随机
   性，候选多样性要等阶段 2 的 ODE→SDE 落地后才成立。数据层只需按 condition_id 预留多候选槽位即可。

# EmoFiLM 强化学习 + 扩散策略：监督改造调研与方案设计报告

- 日期：2026-08-06（任务书日期 2026-08-05；本报告为任务书验收产物）
- 性质：调研综述 + 下一步执行方向方案设计（决策输出，非实施计划）
- 依据任务书：`docs/superpowers/plans/2026-08-05-emofilm-rl-diffusion-research-design.md`
- 结论一句话：**主方案 = 在 CosyVoice3 基座上，对扩散/流匹配声学模型（DiT 条件流匹配）做在线组相对策略优化（FlowTTS-GRPO 式），奖励 = 情感一致性 + 说话人一致性 + 可懂度护栏；备选方案 = 大语言模型语音符号层组相对策略优化（官方组相对策略优化配方 + 作者 Emo-PA 多目标奖励）。**

---

## 0. 摘要

在 08-02～08-04 已确认"情感相似度 ~66 平台由中性声学提示造成、输入端句级交叉熵是文本可分捷径"的基础上，本次调研回答下一步监督改造（R2）如何落地，并满足**必须以强化学习为主、方案必须带扩散策略**的硬性要求。

调查确认的核心事实：

1. 本地代码已是 CosyVoice3 时代代码：`CosyVoice3LM`、`CosyVoice3Tokenizer`、`CausalMaskedDiffWithDiT`（DiT 条件流匹配）、`CausalConditionalCFM`、`CausalHiFTGenerator` 全部存在；官方权重（含 `llm.rl.pt` 强化学习版）已本地可用，且本仓库命令行已跑通 CosyVoice3 情感指令推理。
2. 作者 Emo-PA 强化学习代码（组相对策略优化入口 + emotion2vec 情感奖励 + CAM++ 说话人奖励 + 情感原型 + 组内归一化 + 冲突数据构造）与本项目问题同构，是奖励函数与数据构造的算法级模板；其 语音符号格式、奖励服务器架构与官方配方一致。
3. 官方 CosyVoice 仓库已发布完整的"大语言模型层组相对策略优化"配方（veRL + vLLM + Triton 奖励服务器 + 模型转换），本仓库主线代码与该仓库同源；但本地环境未安装 veRL，官方配方目录不在本仓库内。
4. "扩散策略 + 强化学习"有直接先例且已在本项目同款架构上验证：**FlowTTS-GRPO（arXiv:2606.23190，阿里通义实验室）对 CosyVoice 3.0 的流匹配模型做在线组相对策略优化**，报告说话人相似度与感知质量提升；F5R-TTS（arXiv:2504.02407）对 F5 流匹配语音合成做组相对策略优化；Flow-GRPO（arXiv:2505.05470）给出图像流匹配模型在线强化学习的通用方法；EASPO（arXiv:2509.25416）对扩散语音合成做逐步偏好优化。
5. 专利差异论证：专利权利要求聚焦"更新语音离散表示生成网络、情感条件编码模块、条件调制模块"（即大语言模型侧模块），**未覆盖对扩散/流匹配生成器的强化学习后训练**；主方案以扩散/流匹配模型本身作为策略更新对象，形成清晰差异。

主方案一句话：**冻结 CosyVoice3 大语言模型并预解码语音符号，把 DiT 条件流匹配（扩散策略）作为强化学习策略本体，按 FlowTTS-GRPO 方式做在线组相对策略优化；奖励采用作者 Emo-PA 公式的情感一致性（emotion2vec 原型相似度）+ 说话人一致性（CAM++）+ 可懂度护栏（语音识别错误率），组内归一化后加权联合。**

备选方案一句话：**按官方大语言模型层组相对策略优化配方（veRL）对 语音符号生成策略做情感多目标组相对策略优化，奖励同上；扩散/流匹配模型作为候选解码与奖励链中的扩散策略成分（固定解码器）。**

推荐先执行主方案：它直接针对已确诊的声学瓶颈（H1：中性提示下声学渲染洗掉情感差异），扩散策略是强化学习策略本体，工程上不需要安装 veRL/vLLM，且与专利形成最强差异；备选方案作为工具链最完整、但扩散策略参与度较弱的兜底路线。

---

## 1. 背景与文件关系图景

### 1.1 时间线（2026-07-16 ~ 2026-08-06）

```text
07-16~07-30   v1 基线建立、架构审计、film-only 实验、preflight 全审、端到端 验证
08-01         longepoch 收敛对比（交叉验证 loss_tts 是情感弱代理）
08-02         句级监督修复计划 → 实现审查 → 重构复审 → 句级监督对照实验
              产物取证（why-plateau）→ 根因调研（rootcause）→ 最终合并报告
              （next-step-research：H1~H13、R1~R7）
08-03         评测 v3 重构计划 → 执行报告（情感匹配提示 84-87 / 五选一 73-83%；
              CosyVoice3 官方基线 70.76 未质变）
08-04         评测前置收尾计划 → 执行报告（判别增强 + v3 中性基线）；
              下一步方向规划（R2 监督改造待决策，D1~D7）
08-05         本调研任务书（writing-plans 产物）+ 并行子智能体派发
08-06         全部 08-02~08-04 文件重读核对、本地代码复核、方案决策、本报告
```

### 1.2 08-02 ~ 08-04 文件先后与调用关系

| 文件                                                                          | 角色与依赖                                                                                            |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `docs/superpowers/plans/2026-08-02-emofilm-sentlvl-fixes.md`                | 句级监督修复 9 任务实施计划；依赖实现审查的问题清单；交付后由重构复审验收                             |
| `docs/reports/2026-08-02-emofilm-sentlvl-implementation-review.md`          | 实现审查（P0×2 / P1×2），是修复计划的 Spec 来源                                                     |
| `docs/reports/2026-08-02-emofilm-sentlvl-refactor-review.md`                | 重构复审：9 任务交付全绿（477 通过 / 2 跳过）                                                         |
| `docs/reports/2026-08-02-emofilm-sentlvl-experiment-report.md`              | 句级监督对照实验：情感相似度不升（65.45）、词错误率劣化（10.05）                                      |
| `docs/reports/2026-08-02-emofilm-why-plateau-investigation.md`              | 产物取证 + 隔离实验（复制合成、符号搬运、五选一判别、梯度探针）                                       |
| `docs/reports/2026-08-02-emofilm-rootcause-research.md`                     | CosyVoice3 与文献核对 + 隔离实验（微型模型梯度分解、真实模型条件强度）                                |
| `docs/reports/2026-08-02-emofilm-next-step-research.md`                     | **最终合并报告**：H1~H13 问题清单、R1~R7 推荐方案、CosyVoice3 切换评估；后续一切规划的权威索引 |
| `docs/superpowers/plans/2026-08-03-emofilm-eval-v3-refactor.md`             | 评测 v3 重构 + 情感匹配提示验证 + CosyVoice3 基线实施计划；依赖 08-02 合并报告 R1/R6                  |
| `docs/reports/2026-08-03-emofilm-eval-v3-execution-report.md`               | 执行报告：情感匹配提示 84.3–87.4、五选一 73–83%；CosyVoice3 基线 70.76                              |
| `docs/reports/2026-08-03-cosyvoice3-baseline-comparison.md`                 | CosyVoice3 官方基线对比结论（未质变 → 问题在协议）                                                   |
| `docs/superpowers/plans/2026-08-04-emofilm-eval-prereqs.md`                 | 判别指标增强 + v3 中性基线 + v1 兼容精简实施计划；依赖 08-03 遗留项与下一步规划前置项                 |
| `docs/reports/2026-08-04-emofilm-eval-prereqs-execution-report.md`          | 执行报告：判别增强（ESD n_scored=1422）+ v3 中性基线                                                  |
| `docs/reports/2026-08-04-v3-neutral-baseline.md`                            | v3 全量中性基线：longepoch 65.89 / sentlvl 65.45，判别准确率 ~43%（随机 ~22%）                        |
| `docs/superpowers/plans/2026-08-04-emofilm-next-steps-planning.md`          | **下一步方向规划**：R2 监督改造为下一步主线，待决策 D1~D7                                       |
| `docs/superpowers/plans/2026-08-05-emofilm-rl-diffusion-research-design.md` | 本任务书：调查角度、硬性要求、方案决策标准、验收清单                                                  |

### 1.3 已确认的实验事实（08-02 ~ 08-04，本报告不重复调研）

- 四次实验（v1 / 5-epoch disabled / 27-epoch disabled / sentlvl）共享"弱条件通路"：ESD 情感相似度 65.45~66.75 平台、词错误率 8.18~10.05。
- 决定性隔离实验：复制合成（目标音频自身符号 + 自身声学提示）五选一判别 93.3%；**参考符号原样提取 + 中性声学提示 53.3%** → 声学提示钳制是平台主因（H1）。
- 输入端句级交叉熵是文本可分捷径：冻结随机读出器读回准确率 100%、情感编码器梯度与 TTS 损失反方向（余弦 −0.26）、调制幅度 4–10 倍文本嵌入范数（H2/H4）。
- 情感匹配提示（同说话人同情感不同句）下四模型情感相似度跳到 84.3–87.4、五选一判别 73–83% → 这是模型实际上限参照，正式主线评测保持中性提示（能力口径）。
- CosyVoice3 官方基线（instruct 情感指令 + 中性提示）情感相似度 70.76、词错误率 5.17 → 基座非瓶颈，协议仍是瓶颈。
- v3 全量中性基线：longepoch 65.89 / sentlvl 65.45；判别准确率 ~43%（随机 ~22%）→ 模型能区分情感，但声学渲染把差异压扁。
- 因此 R2 的正确方向：把情感监督从"文本嵌入交叉熵"移到"音频/生成链下游"（音频奖励强化学习或输出侧词级监督）；结合声学瓶颈 H1，**对声学渲染（扩散/流匹配）本身的强化学习是对症下药**。

### 1.4 专利现状与"扩散策略"界定

专利交底书（`一种基于强化学习的情感可控语音合成偏好对齐方法`）核心要素 E1~E8：细粒度情感标签（类别+强度+作用文本范围）、候选组采样、冲突条件数据、片段级情感原型相似度局部奖励、整句说话人一致性奖励、组内归一化联合优势、概率比截断 + 参考策略散度约束的组相对策略优化、推理直出。

专利新颖性调研报告结论（本任务只利用、不扩展检索）：

- FlexiVoice 已公开候选组 + 冲突数据 + 说话人奖励 + z 分数联合优势 + 组相对策略优化主干；联合优势公式与专利逐符号一致。
- 中国专利 CN122024692A、CN122157635A 覆盖组相对策略优化 + 多奖励 + 组内优势；EMORL-TTS 公开词级局部奖励与截断 + 散度约束。
- 唯一相对未公开的完整组合点是 E4（标签作用范围定位 + 片段情感原型相似度奖励）及其与 E5/E6/E7 的链路，但创造性强度存疑。
- 新方向要求：**必须引入扩散策略**，使强化学习后训练的对象/链路与专利权利要求形成差异。

本任务的"扩散策略"界定（与任务书一致）：扩散/流匹配生成模型作为可生成、可被优化的策略网络。本地两类载体：

- CosyVoice2：`CausalMaskedDiffWithXvec` + `ConditionalCFM`（掩码扩散 + 条件流匹配）；
- CosyVoice3：`CausalMaskedDiffWithDiT` + `CausalConditionalCFM`（DiT 22 层流匹配，官方权重与强化学习版权重已本地可用）。

---

## 2. 调查综述

### 2.1 本地代码：CosyVoice3 与扩散/流匹配（证据：本地源码 + 权重 + 实验产物）

组件盘点（全部实测）：

| 组件                                 | 位置                                                                | 状态                                                                                          |
| ------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `CosyVoice3LM`                     | `cosyvoice/llm/llm.py:664`                                        | 已实现（Qwen2LM 子类；语音符号表 6561、+200 特殊符号，sos=6561/eos=6562/task=6563/fill=6564） |
| `CosyVoice3Tokenizer`              | `cosyvoice/tokenizer/tokenizer.py:274`                            | 已实现（v3 分词器）                                                                           |
| `CausalMaskedDiffWithDiT`          | `cosyvoice/flow/flow.py:284`                                      | 已实现（输入 80 维、输出 80 维、DiT 估计器；`inference` 用 `n_timesteps=10`）             |
| DiT 估计器                           | `cosyvoice/flow/DiT/dit.py`                                       | 已实现（官方 yaml：维度 1024 / 深度 22 / 头数 16）                                            |
| `CausalConditionalCFM`             | `cosyvoice/flow/flow_matching.py:196`                             | 已实现（条件流匹配；`compute_loss` 在 :155，欧拉求解在 `solve_euler`）                    |
| `ConditionalCFM`                   | `cosyvoice/flow/flow_matching.py:21`                              | 已实现（训练/推理接口，含分类器自由引导）                                                     |
| `CausalHiFTGenerator`              | `cosyvoice/hifigan/generator.py:572`                              | 已实现（因果声码器）                                                                          |
| `CosyVoice3Model` / `CosyVoice3` | `cosyvoice/cli/model.py:397` / `cosyvoice/cli/cosyvoice.py:189` | 已实现；`AutoModel` 检测 `cosyvoice3.yaml` 加载                                           |

流匹配机制关键事实（对强化学习最重要）：

- 训练接口：`CausalConditionalCFM.compute_loss` 随机时间步 + 随机噪声，直线插值路径，速度场回归损失；训练时按 `training_cfg_rate=0.2` 随机丢弃条件。`cosyvoice/bin/train.py` 支持 `--model llm|flow|hifigan`，flow 训练可直接复用。
- 采样接口：`CausalConditionalCFM.forward` 每次重新采样初始噪声 `z ~ randn`，欧拉求解常微分方程，支持推理期分类器自由引导（`inference_cfg_rate=0.7`）。**同条件下多次生成天然随机，可作候选组采样。**
- **缺口**：本地没有"常微分方程 → 随机微分方程"路径转换、没有轨迹概率/似然计算、没有现成的流匹配层强化学习循环。这是"对扩散策略本体做强化学习"时唯一必须从论文落地的核心工程点（FlowTTS-GRPO / Flow-GRPO 提供配方）。
- 条件输入：语音符号（提示段 + 生成段）+ 提示梅尔谱（`conds` 前段）+ 说话人嵌入；`conds` 通道已预留（`flow_matching.py` 注释"Not used but kept for future purposes"），**可注入情感条件而不改接口**。

官方权重与配置（本地可用）：

- 缓存：`~/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/`（等价转义名 `Fun-CosyVoice3-0___5B-2512`，为符号链接）。
- 文件（实测）：`llm.pt`、`llm.rl.pt`（官方强化学习版）、`flow.pt`、`flow.decoder.estimator.fp32.onnx`、`hift.pt`、`speech_tokenizer_v3.onnx`、`speech_tokenizer_v3.batch.onnx`、`campplus.onnx`、`cosyvoice3.yaml`、`CosyVoice-BlankEN/`。
- `cosyvoice3.yaml` 关键项：`CosyVoice3LM`（语音符号表 6561）、`CausalMaskedDiffWithDiT`（DiT 1024/22）、`CausalConditionalCFM`（sigma_min 1e-6、余弦时间调度、训练引导率 0.2、推理引导率 0.7）、`CausalHiFTGenerator`、训练配置（学习率 1e-5、恒定调度、200 轮）。
- 本仓库 `conf/` 只有 `emo_film*.yaml`（v2 情感配置），**无 `conf/cosyvoice3.yaml`**；训练 v3 前需把官方 yaml 派生到 `conf/`。
- CosyVoice3 推理已跑通：`exp/cosyvoice3_baseline/smoke.md` + `gen_cv3.py`；150 条 ESD 基线已生成（371 秒/卡，情感相似度 70.76）。

现有 EmoFiLM 与 CosyVoice3 接入点差异：

| 维度       | 现有（v2 情感主线）                                                     | CosyVoice3                                                   |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| 大语言模型 | `Qwen2LM_Emotion`（继承 v2 Qwen2LM）                                  | `CosyVoice3LM`（特殊符号布局不同：instruct / endofprompt） |
| 流匹配     | `CausalMaskedDiffWithXvec` + `ConditionalCFM`                       | `CausalMaskedDiffWithDiT` + `CausalConditionalCFM`       |
| 声码器     | `HiFTGenerator`                                                       | `CausalHiFTGenerator`                                      |
| 语音分词器 | v2（语义符号）                                                          | v3（FSQ 监督分词器，含情感识别多任务监督）                   |
| 数据       | parquet 内 v2`speech_token`（训练 20774 + 交叉验证 1092 + 评测 2500） | 需 v3 重抽或由大语言模型预解码                               |

情感模块（`cosyvoice/llm/llm_emotion.py`：`emotion_encoder` / `emotion_adapter` / `emotion_classifier` / `emotion_head` / `arousal_head`）与主干解耦，结构上可平移到 `CosyVoice3LM` 子类；但这只在"大语言模型层强化学习"路线需要。

### 2.2 作者 Emo-PA 强化学习代码审计（证据：`reference/Emo_PA_code_data/` + 硕士论文第 3 章）

#### 组相对策略优化路径（veRL）

- `run_dual_gpu.sh`：`verl.trainer.main_ppo`，`algorithm.adv_estimator=grpo`；组大小 4（`GRPO_GROUP_SIZE=4`）、λ=1.0（`GRPO_LAMBDA=1.0`）；学习率 1e-6、`use_kl_loss=True`、`kl_coef=0.05`；采样 温度=0.8、top_p=0.95、top_k=25、n=1；2 GPU FSDP、`total_epochs=3`、训练批量 4、`max_prompt_length=512`、`max_response_length=128`；奖励注册 `reward_model.reward_manager=prime` + `custom_reward_function.path=reward_tts.py` + `name=compute_score`。
- `reward_tts.py`：`compute_score` 从 `solution_str` 解析 `<|s_i|>` 语音符号 → 调 `reward_server.compute_reward(token_ids, ground_truth, speaker_id)` 返回 `final_score`。
- `reward_server.py`：Flask 服务，预加载 CosyVoice2 解码器（大语言模型 strict=False，只取流匹配/声码器权重）、emotion2vec（funasr）、CAM++（onnx）、情感原型；符号→16kHz 音频走**流匹配 + 声码器 + 重采样**。
- `reward/emotion.py`：解析 `<emotion type='X' intensity='Y'>…</emotion>` 标签 → 强制对齐（代码中实际跳过，用字符比例估计时间边界）→ emotion2vec 对片段打分 → 与目标情感原型余弦相似度；`reward/speaker.py`：CAM++ 说话人相似度；`reward/normalize.py`：文件锁同步的组内 z 分数归一化 `A = (r_emo−μ)/σ + λ·(r_sv−μ)/σ`；`reward/prototypes.py`：`emotion_prototypes_ncssd/*.npy`。
- `convert_emotion_to_hf.py` / `huggingface_emotion_to_pretrained.py`：情感模型与 HuggingFace 格式双向转换；`prepare_ncssd_data.py` / `prepare_ncssd_conflict.py`：冲突条件数据构造；`compute_ncssd_prototypes.py`：情感原型计算。
- 数据：`data/parquet_ncssd/`（veRL 标准列 `data_source/prompt/ability/reward_model/extra_info`；prompt 字段为带情感标签文本；`extra_info` 含 `conflict_type/prompt_audio_path/speaker_id/text`）；`prompt_audio/` 为 ESD 说话人提示。

#### 直接偏好优化路径（CosyVoice 原生扩展）

- `cosyvoice/bin/train_dpo.py`（官方 `train.py` 的直接偏好优化扩展，`--dpo --beta`，deepspeed/torch_ddp）+ `llm/llm_dpo.py`、`dataset/processor_dpo.py`、`utils/{losses_dpo,executor_dpo,train_utils_dpo}.py` 完整存在。
- 与本仓库 `cosyvoice/bin/train_emo.py`（同为官方 train.py 扩展）结构同源；若选直接偏好优化路径可直接参考其数据管线与执行器改造。

#### 硕士论文第 3 章（`reference/23S136160-王思睿.pdf`，本地已提取 `/tmp/thesis_full.txt`）

- 数据：NCSSD 英文子集 9000 条（一致 4050 / 语义冲突 4050 / 音色冲突 450 / 双重冲突 450），约 16 小时。
- 训练设置：候选组大小 K=6（**与代码 `GROUP_SIZE=4` 不一致，实施时以代码为准**）、截断 ε=0.2、散度系数 β=0.05、批量 4、3 轮、Adam、2×RTX4090；LoRA 全层适配（大语言模型 + 情感编码器 + 特征线性调制）。
- 奖励：片段级情感一致性（emotion2vec 片段表征 vs 目标情感类别向量余弦）+ 全局音色一致性（CAM++ 余弦）；联合优势公式与专利 E6 一致。
- 结果（可引用）：NCSSD Task-B 情感指令遵循准确率 42.3（CosyVoice2）→ 56.8（Emo-FiLM）→ **82.1（Emo-FiLM-PA）**；ESD Task-A 65.5 → 79.2 → **85.8**，Task-B 48.7 → 61.2 → **81.2**；词错误率在一致场景从 7.1 降到 3.0。
- 消融（Task-B）：去掉片段级情感奖励 82.1 → **56.5**；去掉音色奖励时音色相似度 0.86 → 0.72；**片段级奖励换句级奖励 82.1 → 71.4** → 片段级局部奖励是核心杠杆。

#### 复刻性结论

- 作者流程 = 官方 CosyVoice 大语言模型层组相对策略优化配方的"情感版"；语音符号格式（`<|s_i|>` 字符串）、veRL 数据列、奖励服务器架构、HuggingFace 转换四件套与本项目完全兼容（本项目大语言模型输出同样是 CosyVoice 语音符号序列）。
- 直接复刻需改动：①数据准备（本仓库 ESD/IEMOCAP → veRL parquet：带情感标签文本 + speaker_id + 提示音频 + 同文本跨情感分组）；②奖励服务器换用本仓库解码链路（符号→音频可用作者 `reward/audio.py` 逻辑 + 本仓库 `pretrained_models/CosyVoice2-0.5B`）；③情感模型 HuggingFace 转换（本仓库 `Qwen2LM_Emotion` 与作者 `llm_emo.py` 同构，转换脚本可对照改写）；④veRL/vLLM 依赖安装（**本地环境未安装 veRL**，需按官方配方安装或使用官方 Docker）。

### 2.3 官方 CosyVoice 大语言模型层组相对策略优化配方（证据：GitHub API + 原始脚本，2026-08-06 核验）

官方仓库 `FunAudioLLM/CosyVoice` 的 `examples/grpo/cosyvoice2/`（**不在本仓库内，实施时需从官方仓库获取**）：

- `run.sh` 阶段：-2 安装 veRL（`yuekaizhang/verl -b thread`）+ vLLM；-1 下载并转换 CosyVoice2 大语言模型为 HuggingFace 格式；0 数据准备（JSONL → veRL parquet）；1 启动 符号→音频 + 语音识别服务器（Triton，SenseVoice）；2 组相对策略优化训练（8 GPU、训练批量 32、微批量 4、学习率 1e-6、n=4、温度 0.8、top_p 0.95、top_k 25、`reward_manager=prime`、自定义奖励函数）；3 FSDP 合并；4 评测；5 转回 CosyVoice 格式。
- 奖励：拼音级错误率映射 0~1（可懂度）；结果 CosyVoice3 `zero_shot_zh` 字符错误率 4.08% → **3.36%**。
- 该配方即作者组相对策略优化流程的"无情感版"原型；主方案若走大语言模型层路线，直接以它为训练骨架，把奖励替换为情感 + 说话人（作者奖励）并保留可懂度护栏。

### 2.4 扩散/流匹配策略 + 强化学习文献（证据：arXiv 原文 + 本地抓取全文）

#### 直接先例

| 方法                   | 出处                                        | 核心做法                                                                                                                                                                                                                                                                                                                                                                                                                   | 与本方案关系                                                                                                |
| ---------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **FlowTTS-GRPO** | arXiv:2606.23190（2026-06，阿里通义实验室） | **对 CosyVoice 3.0 与 F5-TTS 的流匹配模型做在线组相对策略优化**：常微分方程轨迹 → 随机微分方程路径；早期去噪步窗口采样 + 确定性欧拉其余步；奖励 = 说话人相似度（R_SS）+ 语音识别错误率（R_ASR，中文 Paraformer / 英文 Whisper-v3）+ DNSMOS 感知质量（R_MOS）；加权组合 `R = λ1·R_SS/标准差 + λ2·R_ASR/标准差 + λ3·R_MOS/标准差`；训练期省略分类器自由引导、硬样本增强、流匹配层强化学习主要提升音频细节指标 | **主方案的直接模板**：同架构（CosyVoice 3.0）、同范式（在线组相对策略优化）、奖励结构与本项目需求一致 |
| F5R-TTS                | arXiv:2504.02407（2025-04，腾讯）           | 把流匹配语音合成的确定性输出重表述为概率高斯分布，从而对流匹配模型做组相对策略优化；奖励 = 词错误率 + 说话人相似度                                                                                                                                                                                                                                                                                                         | "扩散/流匹配策略 + 组相对策略优化"首个语音先例；词错误率相对降 29.5%、说话人相似度 +4.6%                    |
| Flow-GRPO              | arXiv:2505.05470（2025-05）                 | 图像流匹配模型在线强化学习通用方法：常微分方程 → 随机微分方程路径 + 去噪步裁剪加速（GenEval 63→95）                                                                                                                                                                                                                                                                                                                      | 提供随机微分方程路径与训练加速的通用配方                                                                    |
| EASPO                  | arXiv:2509.25416（2025-09，ICASSP 2026）    | 扩散语音合成逐步偏好优化：时间条件评分模型构建中间去噪步偏好对                                                                                                                                                                                                                                                                                                                                                             | "扩散策略 + 情感偏好"先例；非组相对策略优化                                                                 |
| DiffRO                 | arXiv:2507.05911（2025-07）                 | CosyVoice3 官方可微奖励优化：直接在语音符号层计算可微奖励（Token2Text 语音识别、情感等）                                                                                                                                                                                                                                                                                                                                   | 官方 语音符号层强化学习思路；奖励设计参考                                                                   |

#### 结论（大语言模型层 vs 流匹配层）

- 先例中，流匹配/扩散层强化学习（F5R-TTS、FlowTTS-GRPO、Flow-GRPO、EASPO）已成熟；其中 FlowTTS-GRPO 直接在同款 CosyVoice 3.0 架构上验证。
- 流匹配层强化学习直接针对本项目确诊瓶颈（H1：声学渲染洗掉情感差异），且**不需要 veRL/vLLM**（大语言模型冻结、预解码符号即可），工程依赖更少。
- 大语言模型层强化学习（官方配方 + 作者 Emo-PA）工具链最完整、效果已有作者实验支撑，但扩散/流匹配模型仅作为固定解码器参与，专利差异较弱。
- 两者并存的先例尚无单一论文；本项目按"主方案 + 备选方案"分别落地，不要求同时做。

### 2.5 情感强化学习奖励配方启示（读论文找灵感，证据见 §7）

可复用配方：

| 来源                                                    | 奖励/训练要点                                                                                                                                                                                                                                        | 本项目落点                                                        |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| FlexiVoice（arXiv:2601.04656）                          | S1 直接偏好优化：同说话人同文本、目标情感为优选 / 其他情感为劣选、中性提示；S2 组相对策略优化：`r_ser`（emotion2vec 目标情感概率）+ `r_sv`（CAM++ 说话人验证），组内 z 分数联合；冲突数据：NCSSD 文本随机情感标签 + 参考音频 90% 中性 / 10% 情感 | ESD 同文本跨情感对天然构成偏好对；联合优势公式与作者实现一致      |
| EMORL-TTS（arXiv:2510.05758）                           | 组相对策略优化目标显式含截断 + 参考策略散度；词级局部奖励（强制对齐 + 基频/能量 z 分数）；情感识别 + 强度 + 重音奖励                                                                                                                                 | 片段级奖励与质量护栏设计参考；超参（学习率 1e-6、K=16、散度 0.1） |
| GLM-TTS（arXiv:2512.14291）                             | 多奖励组相对策略优化（相似度/词错误率/情感/笑声）+ LoRA；奖励分步归一化                                                                                                                                                                              | 多奖励尺度平衡方法；LoRA 可选                                     |
| Emo-PA 论文/硕士论文（arXiv:2509.20378 + 本地学位论文） | 片段级情感原型奖励 + 说话人奖励 + 组内归一化 + 截断/散度约束；消融证明片段级是关键                                                                                                                                                                   | 主方案奖励公式的直接来源                                          |
| HPRO（arXiv:2606.28249）                                | 帧/词/句三级情感奖励                                                                                                                                                                                                                                 | 词级增强参考                                                      |
| RLAIF-SPA（arXiv:2510.14628）                           | 组相对策略优化 + 四维韵律-情感标签对齐                                                                                                                                                                                                               | 奖励维度设计参考                                                  |
| Emo-DPO（arXiv:2409.10157）                             | 直接偏好优化区分正负情感对，比交叉熵更强                                                                                                                                                                                                             | 数据对构造参考                                                    |
| RRPO（arXiv:2512.04552）                                | 鲁棒奖励模型抗奖励欺骗                                                                                                                                                                                                                               | 长期风险对策                                                      |

失败模式：

- 《No Verifiable Reward for Prosody》（arXiv:2509.18531）：仅词错误率/负对数似然奖励会压制韵律；加说话人相似度可能破坏稳定性 → 多奖励必须归一化 + 约束（正是组内 z 分数 + 截断 + 散度约束的动机）。
- 《The False Resonance》（arXiv:2604.26347）：emotion2vec 余弦指标被语言/说话人因素主导 → 评测沿用 v3 判别 + 逐情感指标，不以均值相似度为准。
- FlowTTS-GRPO 报告：单一奖励会导致奖励欺骗，多奖励需按批量标准差归一化后加权；可懂度主要由大语言模型决定，流匹配层强化学习主要提升说话人相似度与感知质量 → **本项目主方案若只做流匹配层强化学习，情感提升需靠情感奖励直接驱动，可懂度靠护栏保住**。

### 2.6 专利差异论证（E1~E8 逐项）

主方案（流匹配/扩散策略层组相对策略优化）：

| 要素                               | 主方案                                                                              | 差异说明                                                                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| E1 细粒度标签                      | 复用现有标签（ESD 句级 + IEMOCAP 词级），不强依赖"作用文本范围"                     | 标签粒度可只到句级；不把结构化标签范围作为必要特征                                                                                    |
| E2 候选组                          | 流匹配采样天然随机，组内候选由扩散噪声路径产生                                      | 候选生成发生在扩散/流匹配声学层，不是专利所述的"语音离散表示生成网络"采样                                                             |
| E3 冲突条件                        | 可选构造（同文本跨情感对天然冲突）                                                  | 非必要特征                                                                                                                            |
| E4 片段级情感原型奖励              | 情感奖励可句级/片段级（片段级为增强项）                                             | 若采用片段级即与 E4 相同思想；主方案核心差异在策略对象                                                                                |
| E5 说话人一致性奖励                | 保留（CAM++）                                                                       | 已被 FlexiVoice 公开                                                                                                                  |
| E6 组内归一化联合优势              | 保留（按批量标准差归一化 + 加权）                                                   | 已被 FlexiVoice / FlowTTS-GRPO 公开                                                                                                   |
| E7 截断 + 散度约束的组相对策略优化 | 保留                                                                                | 标准算法                                                                                                                              |
| E8 推理直出                        | 保留                                                                                | 隐含/常见                                                                                                                             |
| **扩散/流匹配策略本体**      | **组相对策略优化的参数更新对象是 DiT 条件流匹配（扩散策略），大语言模型冻结** | **专利权利要求更新对象为"语音离散表示生成网络、情感条件编码模块、条件调制模块"（大语言模型侧），未覆盖扩散/流匹配生成器后训练** |
| 奖励组合                           | 情感 + 说话人 + 可懂度三路，组内归一化加权                                          | 专利只有情感 + 说话人两类                                                                                                             |

结论：主方案以"**对扩散/流匹配生成器做在线组相对策略优化**"为整体，与专利权利要求存在清晰差异；若再配合"情感条件注入流匹配条件通道"（`conds` 接口已预留），差异更显著。备选方案（大语言模型层组相对策略优化）与专利主干接近，专利差异论证较弱，仅作为工程兜底。本报告不扩展专利检索；正式申请前仍应委托专业机构做自由实施复核。

---

## 3. 方案决策

### 3.1 主方案：CosyVoice3 基座流匹配/扩散策略层在线组相对策略优化（FlowTTS-GRPO 式）

**一句话**：冻结 CosyVoice3 大语言模型并预解码语音符号；把 DiT 条件流匹配（`CausalMaskedDiffWithDiT` + `CausalConditionalCFM`）作为强化学习策略本体，按 FlowTTS-GRPO 方式做在线组相对策略优化；奖励 = 情感一致性（emotion2vec 原型相似度）+ 说话人一致性（CAM++）+ 可懂度护栏（语音识别错误率），组内归一化加权。

#### 技术要素

1. **策略对象**：`flow.pt`（CosyVoice3 DiT 流匹配）；大语言模型（`llm.pt`）冻结，只用于按情感条件采样/预解码语音符号。
2. **数据与符号**：现有 `data/contracts/emofilm_v1`（ESD 同文本跨情感 + IEMOCAP 词级 5484 条）→ 用 v3 分词器重抽（或直接用冻结大语言模型按情感指令预解码生成符号）；构造条件 = 文本（带情感标签或情感指令）+ 同说话人中性提示音频 + 目标情感；可选按作者 `prepare_ncssd_conflict.py` 思路构造语义/音色冲突。
3. **强化学习**：常微分方程 → 随机微分方程路径（FlowTTS-GRPO / Flow-GRPO 配方）；早期去噪步窗口采样，其余步确定性欧拉；组大小 4~8；组内优势 `A = (R−均值)/标准差`；组相对策略优化目标含概率比截断 + 参考策略散度约束（β≈0.05）。
4. **奖励**：
   - 情感一致性：emotion2vec 对生成音频（句级或按标签范围片段级）提取表征，与目标情感原型余弦相似度（作者 `reward/emotion.py` + `compute_ncssd_prototypes.py`）；
   - 说话人一致性：CAM++ 生成音频与提示音频说话人嵌入余弦（作者 `reward/speaker.py`）；
   - 可懂度护栏：语音识别（中文 Paraformer / 英文 Whisper）错误率映射 0~1（FlowTTS-GRPO 配方；本项目已有 Whisper 链路）；
   - 加权：`R = λ1·z(R_emo) + λ2·z(R_sv) + λ3·z(R_cer)`，按批量标准差归一化（作者/FlowTTS-GRPO 一致）。
5. **可选增强（推荐）**：情感条件注入流匹配条件通道（`conds` 接口已预留），让扩散策略直接接收目标情感；训练期省略分类器自由引导以加速收敛（FlowTTS-GRPO 实践）。
6. **训练骨架**：`cosyvoice/bin/train.py --model flow` + 派生 `conf/cosyvoice3.yaml`；新增流匹配层组相对策略优化循环（自研轻量，参照 FlowTTS-GRPO/Flow-GRPO 公式）；奖励服务复用作者 `reward_server.py` / `reward/audio.py` 逻辑（符号→音频改走本仓库 v3 流匹配 + 声码器）。
7. **评测**：沿用 v3 评测（情感相似度 + 判别 + 逐情感 + 词错误率），中性提示能力口径 + 情感匹配提示上限对照；强化学习前后同口径对比。

#### 接入点清单（代码/配置/数据）

- 新增：`conf/cosyvoice3-emofilm.yaml`（派生自官方 `cosyvoice3.yaml`）、流匹配层组相对策略优化训练脚本（`cosyvoice/bin/train_flow_grpo.py` 或等价入口）、情感奖励服务（`reference/Emo_PA_code_data/` 移植）、情感原型计算脚本（复用 `compute_ncssd_prototypes.py`）。
- 修改：`cosyvoice/flow/flow.py`（如需注入情感条件到 `conds`；接口已预留）、`cosyvoice/flow/flow_matching.py`（新增随机微分方程路径采样与轨迹概率/目标函数，参照 FlowTTS-GRPO）、数据管线（v3 符号重抽或预解码）。
- 复用：`exp/cosyvoice3_baseline/` 推理链路、`eval/eval_emo_film.py` + `eval/emotion_metrics.py`（v3 评测）、`tools/inference_emo_film.py`（生成）、Whisper 转写。

#### 成本估计

- v3 符号重抽/预解码：全量约 1~2 GPU·天（参考 150 条 371 秒/卡可并行分片）。
- 流匹配层强化学习循环实现：1~2 人周（含随机微分方程路径、窗口采样、组内归一化、奖励服务接线）。
- 训练：6×RTX3090 共享集群，组大小 4~8、批量 8~16，千步级冒烟 1~2 GPU·天，正式数 GPU·天~2 周量级（参照 FlowTTS-GRPO 9545 步收敛）。

#### 预期收益

- 直接作用已确诊瓶颈：声学渲染在中性提示下洗掉情感（H1）→ 流匹配层强化学习可学习保留/放大语音符号中的情感线索（FlowTTS-GRPO 同架构证明说话人/质量指标可提升）。
- 情感奖励直接驱动目标情感表达；可懂度护栏防止发音退化。
- 主方案满足"强化学习为主 + 扩散策略确实存在"：扩散/流匹配模型就是被优化的策略。

#### 风险

- 随机微分方程路径与轨迹概率实现是本方案最大工程点（本地无现成代码；FlowTTS-GRPO 论文与 Flow-GRPO 提供公式）。
- 中性提示下情感奖励动态范围可能偏窄（与情感相似度平台同源）→ 先用离线打分校准奖励（对既有四模型生成音频打分，检查区分度），必要时构造冲突条件放大差异。
- 大语言模型冻结意味着 语音符号层情感差异固定（现有模型分布散度 0.099~0.108、最大概率符号变化约 50%）→ 若流匹配层强化学习后情感收益不足，说明 语音符号差异确实不承载声学情感，需转备选方案（大语言模型层）或两者结合。

### 3.2 备选方案：大语言模型语音符号层多目标组相对策略优化（官方配方 + 作者 Emo-PA 奖励）

**一句话**：按官方 `examples/grpo/cosyvoice2` 配方（veRL + vLLM）对大语言模型 语音符号生成策略做情感多目标组相对策略优化；奖励 = 情感一致性 + 说话人一致性 + 可懂度护栏；扩散/流匹配模型作为候选解码与奖励链中的扩散策略成分（固定解码器）。

#### 技术要素

1. **策略对象**：大语言模型（`Qwen2LM_Emotion` 或 `CosyVoice3LM` 情感变体），语音符号自回归生成策略；扩散/流匹配 + 声码器冻结，仅用于候选音频解码与奖励计算。
2. **数据**：ESD 同文本跨情感对 + IEMOCAP 词级 → veRL parquet（带情感标签文本 + speaker_id + 提示音频 + 分组键）；可选冲突构造。
3. **训练骨架**：官方配方 `run.sh`（veRL + vLLM + Triton 奖励服务器 + HuggingFace 转换）；奖励函数改作者 `reward_tts.py` 逻辑（情感 + 说话人 + 可懂度）。
4. **奖励**：同主方案三路奖励；联合优势按组内 z 分数（作者公式）。
5. **评测**：同主方案。

#### 与主方案的本质区别

- 强化学习策略本体不同：大语言模型语音符号生成器 vs 扩散/流匹配声学模型；
- 工程依赖不同：需要 veRL/vLLM 环境与 HuggingFace 转换 vs 只需本地 `train.py --model flow` 与自研循环；
- 扩散策略参与度不同：备选方案中扩散/流匹配为固定解码器（仅作为生成链与奖励解码的扩散策略成分），主方案中扩散/流匹配为策略更新对象；
- 专利差异强度不同：主方案显著强于备选。

#### 为什么作为备选而非主选

- 作者实验已证明该路线在大语言模型层有效（ESD Task-B 情感指令遵循准确率 61.2 → 81.2），工具链最完整；
- 但它对已确诊的声学瓶颈（H1）没有直接作用（扩散/流匹配仍冻结），且与专利权利要求主干接近（强化学习更新对象 = 大语言模型侧模块 + 两类奖励），专利差异论证较弱；
- 若主方案（流匹配层）因 语音符号情感线索不足而收益有限，备选方案可作为第二阶段补强（大语言模型层先优化 语音符号情感差异，再流匹配层强化学习渲染），形成完整闭环。

### 3.3 决策理由与对比（逐条对照任务书决策标准）

| 维度                                 | 主方案（流匹配/扩散策略层组相对策略优化）                             | 备选方案（大语言模型层组相对策略优化）              |
| ------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------- |
| 强化学习为主                         | ✅ 组相对策略优化为主                                                 | ✅ 组相对策略优化为主                               |
| 必须带扩散策略                       | ✅ 扩散/流匹配模型即策略本体                                          | ⚠️ 扩散/流匹配为固定解码器（生成链与奖励链成分）  |
| 最简单可行                           | 中（需实现随机微分方程路径；不依赖 veRL/vLLM）                        | 中高（配方完整；需安装 veRL/vLLM + 转换）           |
| 效果最优                             | 直接对症 H1 声学瓶颈；同架构先例（FlowTTS-GRPO 应用于 CosyVoice 3.0） | 作者实验支撑（情感指令遵循准确率 +20pp）；不解决 H1 |
| 项目主线方向（R2 音频奖励/强化学习） | ✅ 音频奖励强化学习                                                   | ✅ 音频奖励强化学习                                 |
| 作者实现可算法级复刻                 | ✅ 奖励公式与数据构造复用作者代码                                     | ✅ 全流程可复刻作者代码                             |
| CosyVoice3 直接使用                  | ✅ 官方权重/配置/推理链路直接使用                                     | ⚠️ 需情感模块平移 + veRL 适配                     |
| 大面积冲突现有实现                   | 低（新增训练循环，不动现有推理/评测）                                 | 中（新增 veRL 环境 + 转换链）                       |
| 专利差异                             | 强（策略对象在专利范围外）                                            | 弱（接近专利主干）                                  |

---

## 4. 实施路线与验证

### 阶段 0：前置验证与奖励校准（0.5~1 周）

1. 把官方 `cosyvoice3.yaml` 派生到 `conf/cosyvoice3-emofilm.yaml`，用本仓库 `train.py --model flow` 跑通流匹配训练入口（1 步冒烟）。
2. 离线奖励校准：对既有四模型生成音频（`exp/emofilm_*/full/`）计算情感一致性、说话人一致性、可懂度三路奖励，检查动态范围与判别力（复用 `eval/emotion_metrics.py` 思路）。
3. 情感原型：在 ESD/IEMOCAP 语料上按情感类别计算 emotion2vec 原型（复用 `compute_ncssd_prototypes.py`）。

### 阶段 1：数据与符号准备（0.5~1 周）

1. v3 符号重抽或冻结大语言模型预解码：对训练/交叉验证/评测音频生成 v3 语音符号（或按情感条件采样 语音符号序列并缓存）。
2. 构造训练条件：文本（情感标签/指令）+ 同说话人中性提示 + 目标情感 + 同文本跨情感组；可选冲突构造。

```
review 发现的**可能**风险和阻塞：
  1. 没有 token 级 CV3 预解码入口。这是最大的未声明缺口：现网只有 wav 级推理，预解码必须新增 LLM-only 解码路径，且要和
     inference_instruct2 完全同语义（文本规范化分句、instruct 模板、RAS 采样、EOS/fill 处理）。报告把它写成"预解码符号即可"，低估了
     实现面。
  2. "同文本跨情感组"与 GRPO 组语义冲突。标准 GRPO 的组 = 同一条件重复采样；"同文本跨情感"是不同条件（目标情感不同 → LLM 预解码符号
     不同），不能直接当作组内候选。数据清单必须分两层：condition_id（GRPO 组）和 contrast_key（跨情感对照/冲突数据），否则阶段 2 的
     奖励服务会按错键分组。这是设计歧义，不是代码问题，但会阻塞数据 schema。
  3. 文本口径必须拆分。v1 train parquet 的 text 是带 <emotion> 标签的文本，而 CV3 instruct2 的 tts_text 应传纯文本、情感走 instruct
     指令、奖励又要用带标签文本作 ground_truth。阶段 1 需要显式派生三份文本（plain text / instruct / tagged ground_truth），报告没有
     提这一步；不处理会导致预解码质量劣化且奖励解析错位。
  4. embedding 来源要按条件取。v1 parquet 的 spk_embedding 来自目标音频；RL 条件的 embedding 必须来自 prompt wav（flow 的说话人条
     件）。parse_embedding 默认会从行内音频现算，若直接复用 v1 列，flow 条件会拿到错误的说话人向量。需要条件视图显式绑定 prompt 音频
     的 embedding。
  5. 对齐与过滤约束：v3 管线有 token_max_length=200、max_frames_in_batch=2000、token_mel_ratio=2 的裁剪；预解码的生成 token 长度可能
     突破过滤上限，且监督流需要 target token 与 mel 严格对齐。不是阻塞，但数据构建时要按同一阈值校验，避免训练时被 filter 静默丢弃或
     trim_token_mel 悄悄截断。
  7. CausalConditionalCFM 初始噪声固定（上一轮已确认）会直接影响"同条件多候选"的定义；阶段 1 的清单构建不应假定 flow 采样已有随机
     性，候选多样性要等阶段 2 的 ODE→SDE 落地后才成立。数据层只需按 condition_id 预留多候选槽位即可。
```

### 阶段 2：流匹配层组相对策略优化实现（1~2 周）

1. `flow_matching.py` 新增随机微分方程路径采样（参照 FlowTTS-GRPO 公式 6-8）与窗口训练；`flow.py` 可选注入情感条件到 `conds`。
2. 奖励服务：移植作者 `reward_server.py` + `reward/audio.py`，符号→音频走 v3 流匹配 + 声码器；奖励 = 情感 + 说话人 + 可懂度；组内归一化加权。
3. 组相对策略优化循环：组采样 → 奖励 → 优势 → 截断 + 参考策略散度约束 → 流匹配参数更新。
4. 冒烟：100~200 步，检查奖励均值/方差、音频不崩、词错误率护栏不劣化。

### 阶段 3：正式训练与评测（1~2 周）

1. 正式训练至奖励平台（参照 FlowTTS-GRPO 数千步量级；6×3090 可分片并行）。
2. 全量生成 2500 条（`tools/inference_emo_film.py` 或 v3 推理脚本），v3 评测 + 情感匹配提示上限对照。
3. 验收指标：中性提示下情感判别准确率与逐情感相似度相对强化学习前提升、词错误率不劣化、同文本跨情感余弦下降（区分度上升）、五选一判别接近情感匹配提示上限（73~83%）。

### 阶段 4（可选）：大语言模型层补强

若主方案 语音符号情感线索不足，按备选方案对 语音符号层做组相对策略优化后再回到流匹配层强化学习，形成两段式闭环。

---

## 5. 风险与开放问题

1. **随机微分方程路径实现**：主方案最大工程点；本地无现成代码，需按 FlowTTS-GRPO / Flow-GRPO 公式实现并做数值验证（轨迹分布、奖励方差）。
2. **奖励动态范围**：中性提示下情感奖励可能偏窄；用离线校准 + 冲突条件放大；不以情感相似度均值作为验收唯一指标。
3. **语音符号情感线索上限**：若冻结大语言模型预解码的符号差异不承载声学情感，主方案收益受限 → 转备选/两段式。
4. **环境依赖**：主方案不依赖 veRL/vLLM；备选方案需要安装 veRL（官方 Docker 优先）。
5. **评测口径**：中性提示 = 能力口径，情感匹配提示 = 上限口径，两者都保留；验收以判别/逐情感为主。
6. **专利合规**：主方案差异论证见 §2.6；正式申请/发表前委托专业机构复核。
7. **并行子智能体调查状态**：票据 01（本地代码）已收口；票据 02（作者代码）、票据 03（文献）仍在调查中，返回后将作为附录补充（见附录 A）。

---

## 6. 验收清单自查（对应任务书 §5）

- [X] 08-02 ~ 08-04 全部报告与计划已读全（本次会话逐文件从头重读），文件先后/调用关系已理清（§1.2）；
- [X] 专利现状与参考论文列表已纳入，未做扩展专利调研（§1.4、§2.6）；
- [X] 四个调查方向（本地代码、作者强化学习复刻、情感强化学习启示、扩散+强化学习并存）均有可追溯结论（§2）；
- [X] 主方案 + 最多一个备选方案，满足"强化学习为主 + 扩散策略"（§3）；
- [X] 所有结论有来源（本地路径/行号、论文编号/链接）（§7）；
- [X] 关键词使用无歧义中文（首次出现标注通用中文表达，模型/工具专名保留原名，见附录 B）；
- [X] 报告结构清晰易读、专业详实；
- [X] 方向性选择已由本报告决策，无需用户逐轮确认（若用户对主/备取舍有异议，可再按 batch-grill-me 技能逐轮对齐）。

---

## 7. 参考与证据索引

### 本地代码与产物

- `cosyvoice/llm/llm.py:664`（CosyVoice3LM）、`cosyvoice/tokenizer/tokenizer.py:274`、`cosyvoice/flow/flow.py:284`（CausalMaskedDiffWithDiT）、`cosyvoice/flow/flow_matching.py:21/196`（ConditionalCFM / CausalConditionalCFM）、`cosyvoice/flow/DiT/`、`cosyvoice/hifigan/generator.py:572`、`cosyvoice/cli/model.py:397`、`cosyvoice/cli/cosyvoice.py:189`、`cosyvoice/bin/train.py`（`--model llm|flow|hifigan`）。
- `cosyvoice/llm/llm_emotion.py`（Qwen2LM_Emotion + 特征线性调制 + 监督头）、`conf/emo_film*.yaml`、`cosyvoice/bin/train_emo.py`、`tools/inference_emo_film.py`、`eval/eval_emo_film.py` + `eval/emotion_metrics.py`（v3 评测）。
- CosyVoice3 权重：`~/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/`（含 `llm.pt`、`llm.rl.pt`、`flow.pt`、`cosyvoice3.yaml` 等）；冒烟：`exp/cosyvoice3_baseline/`（`smoke.md`、`gen_cv3.py`、`esd_150/`、`eval/`）。
- 作者代码：`reference/Emo_PA_code_data/`（`run_dual_gpu.sh`、`reward_tts.py`、`reward_server.py`、`reward/{config,prototypes,emotion,speaker,normalize,audio}.py`、`convert_emotion_to_hf.py`、`huggingface_emotion_to_pretrained.py`、`prepare_ncssd_{data,conflict}.py`、`compute_ncssd_prototypes.py`、`emotion_prototypes_ncssd/`、`data/parquet_ncssd/`、`cosyvoice/bin/train_dpo.py`、`cosyvoice/llm/llm_dpo.py`、`cosyvoice/utils/*_dpo.py`、`cosyvoice/dataset/processor_dpo.py`）。
- 本地论文：`reference/23S136160-王思睿.pdf`（硕士论文，全文已提取 `/tmp/thesis_full.txt`）、`reference/2509.20378v1.pdf`（Emo-FiLM）、`reference/2601.04656v1.pdf`（FlexiVoice，已提取 `/tmp/flexivoice.txt`）、`reference/arXiv-2509.20378v1/`（LaTeX 全文）。

### 项目文档

- `docs/reports/2026-08-02-emofilm-next-step-research.md`（R1~R7 权威）、`docs/reports/2026-08-02-emofilm-rootcause-research.md`、`docs/reports/2026-08-02-emofilm-why-plateau-investigation.md`、`docs/reports/2026-08-02-emofilm-sentlvl-experiment-report.md`、`docs/reports/2026-08-02-emofilm-sentlvl-implementation-review.md`、`docs/reports/2026-08-02-emofilm-sentlvl-refactor-review.md`。
- `docs/reports/2026-08-03-emofilm-eval-v3-execution-report.md`、`docs/reports/2026-08-03-cosyvoice3-baseline-comparison.md`、`docs/reports/2026-08-04-emofilm-eval-prereqs-execution-report.md`、`docs/reports/2026-08-04-v3-neutral-baseline.md`。
- `docs/superpowers/plans/2026-08-02-emofilm-sentlvl-fixes.md`、`2026-08-03-emofilm-eval-v3-refactor.md`、`2026-08-04-emofilm-eval-prereqs.md`、`2026-08-04-emofilm-next-steps-planning.md`、`2026-08-05-emofilm-rl-diffusion-research-design.md`。
- 票据/地图：`.scratch/rl-diffusion-research/map.md`、`issues/01~06-*.md`、`findings/01-local-code.md`。
- 专利：交底书《一种基于强化学习的情感可控语音合成偏好对齐方法》（附件 docx，已提取 `/tmp/patent_docx.txt`）；《专利新颖性调研报告》（附件 md）。

### 论文（专利调研报告已有列表 + 本次新读）

- FlowTTS-GRPO：arXiv:2606.23190（2026-06；已抓取全文 `/tmp/flowtts_grpo/paper.txt`）
- F5R-TTS：arXiv:2504.02407（2025-04；摘要已核验）
- Flow-GRPO：arXiv:2505.05470（2025-05；已抓取全文 `/tmp/flowgrpo_paper.txt`）
- EASPO：arXiv:2509.25416（2025-09 / v2 2026-02；摘要已核验）
- DiffRO：arXiv:2507.05911（2025-07）
- CosyVoice 3：arXiv:2505.17589（2025-05）
- FlexiVoice：arXiv:2601.04656（2026-01；全文已提取 `/tmp/flexivoice.txt`）
- EMORL-TTS：arXiv:2510.05758；GLM-TTS：arXiv:2512.14291；RLAIF-SPA：arXiv:2510.14628；HPRO：arXiv:2606.28249；Emo-LiPO：arXiv:2606.13006；Emo-DPO：arXiv:2409.10157；RRPO：arXiv:2512.04552；《No Verifiable Reward for Prosody》：arXiv:2509.18531；《The False Resonance》：arXiv:2604.26347
- 官方组相对策略优化配方：`github.com/FunAudioLLM/CosyVoice` → `examples/grpo/cosyvoice2/`（2026-08-06 GitHub API + 原始脚本核验）

---

## 附录 A：并行子智能体调查状态

- 票据 01（本地代码侦察）：已收口 → `findings/01-local-code.md`；结论已并入 §2.1，含两处修正（权重缓存路径等价；本仓库无官方 `examples/grpo` 目录，实施时需从官方仓库获取）。
- 票据 02（作者强化学习代码复刻审计）：调查中；其结论若返回将并入 §2.2 并标注来源。
- 票据 03（扩散/流匹配强化学习文献）：调查中；已抓取 FlowTTS-GRPO / Flow-GRPO / FlexiVoice / Emo-DPO 全文，核心结论已由主代理直接核验并并入 §2.4。
- 票据 04（情感强化学习奖励配方）、票据 05（专利差异论证）、票据 06（可行性核算）：由主代理基于已收口材料直接完成（§2.5、§2.6、§3、§4），后续可按需派发细化。

## 附录 B：中文术语对照（本报告用词规范）

| 英文/缩写                                                                               | 本报告中文表达                   |
| --------------------------------------------------------------------------------------- | -------------------------------- |
| GRPO                                                                                    | 组相对策略优化                   |
| DPO                                                                                     | 直接偏好优化                     |
| RL                                                                                      | 强化学习                         |
| WER / CER                                                                               | 词错误率 / 字符错误率            |
| SIM / SS                                                                                | 说话人相似度                     |
| KL                                                                                      | 散度（散度约束）                 |
| MFA                                                                                     | 强制对齐                         |
| FiLM                                                                                    | 特征线性调制                     |
| CFM / Flow Matching                                                                     | 条件流匹配 / 流匹配              |
| DiT                                                                                     | DiT 扩散 Transformer（保留专名） |
| LLM                                                                                     | 大语言模型                       |
| SER                                                                                     | 情感识别                         |
| ASR                                                                                     | 语音识别                         |
| prompt                                                                                  | 声学提示 / 提示                  |
| hyp audio                                                                               | 待测音频                         |
| emotion2vec / CAM++ / veRL / vLLM / Triton / SenseVoice / Whisper / Paraformer / DNSMOS | 模型/工具专名，保留原名          |
