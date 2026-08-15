# EmoFiLM RL + 扩散策略监督改造：调研与方案设计任务书

- 日期：2026-08-05
- 性质：调研与设计工作细则（不是可执行的实施计划）
- 用途：作为本会话所有并行子智能体调查的统一依据，以及最终调查报告的验收清单

---

## 0. 本会话执行记录（主代理，动态更新）

### 0.1 文件全读核对（2026-08-05，已完成）

已完整读取（并按时间关系理清）：

| 日期 | 文件 | 角色 |
| --- | --- | --- |
| 08-02 | `docs/reports/2026-08-02-emofilm-next-step-research.md` | 四次实验根因 + R1–R7 方案（最终合并版） |
| 08-02 | `docs/reports/2026-08-02-emofilm-rootcause-research.md` | 同主题独立调研版 |
| 08-02 | `docs/reports/2026-08-02-emofilm-why-plateau-investigation.md` | 隔离实验证据链 |
| 08-02 | `docs/reports/2026-08-02-emofilm-sentlvl-experiment-report.md` | 句级监督对照实验 |
| 08-02 | `docs/reports/2026-08-02-emofilm-sentlvl-implementation-review.md` | 实现审查（P0/P1 问题） |
| 08-02 | `docs/reports/2026-08-02-emofilm-sentlvl-refactor-review.md` | 重构审查（修复确认） |
| 08-02 | `docs/superpowers/plans/2026-08-02-emofilm-sentlvl-fixes.md` | 句级监督修复实施计划 |
| 08-03 | `docs/reports/2026-08-03-cosyvoice3-baseline-comparison.md` | CV3 官方基线对比 |
| 08-03 | `docs/reports/2026-08-03-emofilm-eval-v3-execution-report.md` | 评测 v3 重构 + 协议验证执行报告 |
| 08-03 | `docs/superpowers/plans/2026-08-03-emofilm-eval-v3-refactor.md` | 评测 v3 重构实施计划 |
| 08-04 | `docs/reports/2026-08-04-emofilm-eval-prereqs-execution-report.md` | 判别增强 + v3 基线执行报告 |
| 08-04 | `docs/reports/2026-08-04-v3-neutral-baseline.md` | v3 全量中性基线 |
| 08-04 | `docs/superpowers/plans/2026-08-04-emofilm-eval-prereqs.md` | 收尾实施计划 |
| 08-04 | `docs/superpowers/plans/2026-08-04-emofilm-next-steps-planning.md` | 下一步方向规划（D1–D7 待决策） |
| 08-05 | 本文件 | 调研与设计任务书 |

外部材料（已完整读取）：

- 专利交底书 `.docx`（`/home/hanlvyuan/.codex/attachments/f1aad9f0-.../一种基于强化学习的情感可控语音合成偏好对齐方法.docx`）：E1–E8 要素全文。
- 专利新颖性调研报告 `.md`（`/home/hanlvyuan/.codex/attachments/6c1648ad-.../专利新颖性调研报告.md`）：24 篇论文、13 件中国专利、国际专利、时间线、要素对比。

时间线（已理清）：
`v1/5ep/27ep/sentlvl 四次实验（07-20~08-02）→ 根因调研（08-02）→ 句级监督修复与对照（08-02）→ 评测 v3 重构 + 情感匹配 prompt 验证 + CV3 基线（08-03）→ 判别增强 + v3 全量基线 + v1 精简（08-04）→ R2 监督改造待决策（08-04 规划）→ 本任务：RL 为主 + 扩散策略的 R2 方案调研（08-05）`。

### 0.2 子智能体派发计划（动态更新）

阶段 1（并行基础调查，第一批 3 个）：

- [x] 派发 `sub_rl_diffusion_local_code`（本地代码侦察：CosyVoice3 与扩散/流匹配）
- [x] 派发 `sub_rl_diffusion_author_code`（作者 RL 代码可复刻性审计）
- [x] 派发 `sub_rl_diffusion_lit`（扩散/流匹配 RL 文献 + 官方 CV3 RL）

阶段 2（按阶段 1 结论追加，待派发）：

- [x] 情感 RL 奖励配方深读（由主代理直接精读论文一手来源完成，见最终报告 §2.5）
- [x] 专利差异论证（由主代理依据专利调研报告与交底书完成，见最终报告 §2.6）
- [x] 代码级落点与成本核算（由主代理完成，见最终报告 §3–§4）

### 0.5 最终交付（2026-08-06，已完成）

- 最终调查报告：
  `docs/reports/2026-08-05-emofilm-rl-diffusion-research.md`（438 行，含调查综述、
  主方案 + 备选方案、实施路线、风险与证据索引）。
- 主方案（修订版）：CosyVoice3 基座 ＋ 对 DiT 流匹配（扩散策略本体）做在线
  强化学习（FlowTTS-GRPO 范式，常微分方程→随机微分方程路径、组相对策略优化）
  ＋ 作者 Emo-PA 情感/说话人奖励与官方可懂度护栏三路联合；大语言模型冻结，
  仅做数据预解码。依据：08-06 补入的 FlowTTS-GRPO（arXiv:2606.23190）一手
  证据——其在 CosyVoice 3.0 上只优化流匹配组件并验证有效。
- 备选方案：现有 CosyVoice2-EmoFiLM 基座 ＋ verl 组相对策略优化（作者 Emo-PA
  配方）作用于大语言模型 token 策略；流匹配作为固定扩散解码器。与主方案本质
  区别 = 训练范式（大语言模型 token 层 vs 扩散/流匹配声学层）。
- 阶段一并行子智能体（本地代码、作者 RL、扩散+RL 文献）在交付时仍未返回最终
  结果；其下载/核验产物（论文文本、官方仓库树）已被主代理直接采用，报告结论
  全部由主代理一手核实。子智能体后续若返回新材料，以附录形式追加。

### 0.4 主代理二次全读核对（2026-08-06，本次会话）

本次会话已**从头完整重读** 08-02～08-04 全部报告与计划文件（不再依赖 0.1
的先前记录），并补读专利交底书全文（.docx 提取至 `/tmp/patent_docx.txt`）与
专利新颖性调研报告全文（含 4.3/4.4/5.1 被截断中段）。核对结论：

- 时间线与调用关系确认无误：四次实验（07-20~08-02）→ 根因调研（08-02）→
  句级监督修复与对照（08-02）→ 评测 v3 重构 + 情感匹配 prompt 验证 + CV3
  基线（08-03）→ 判别增强 + v3 全量基线 + v1 精简（08-04）→ R2 监督改造
  待决策（08-04 next-steps-planning，D1–D7）→ 本任务（RL 为主 + 扩散策略）。
- 计划-执行-审查调用链：每个实验计划都指向其执行报告与审查报告；最终合并版
  根因报告（next-step-research）是 R1–R7 方案的权威索引，next-steps-planning
  是待决策清单的权威索引。
- 本地代码现状复核（2026-08-06，全部实测）：
  - `cosyvoice/flow/flow.py:284 CausalMaskedDiffWithDiT`、
    `cosyvoice/flow/flow_matching.py:21/196 ConditionalCFM/CausalConditionalCFM`、
    `cosyvoice/llm/llm.py:664 CosyVoice3LM`、
    `cosyvoice/tokenizer/tokenizer.py:274 CosyVoice3Tokenizer`、
    `cosyvoice/hifigan/generator.py:572 CausalHiFTGenerator`、
    `cosyvoice/cli/model.py:397 CosyVoice3Model`、
    `cosyvoice/cli/cosyvoice.py:189 CosyVoice3` 均存在；
  - `conf/` 下仍只有 emo_film / emo_film_earlystop / emo_film_sentlvl 三个配置，
    **尚无 `conf/cosyvoice3.yaml`**；
  - CV3 官方权重在 ModelScope 缓存
    `~/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，且**含
    `llm.rl.pt`（官方 RL 权重）**；
  - `exp/cosyvoice3_baseline/` 已有 gen_cv3.py / esd_150 / eval / smoke.md /
    run_infer.sh / infer.log；
  - 作者 RL 代码 `reference/Emo_PA_code_data/` 完整：reward_tts.py、
    reward_server.py、reward/{config,prototypes,emotion,speaker,normalize,audio}.py、
    run_dual_gpu.sh、prepare_ncssd_{data,conflict}.py、compute_ncssd_prototypes.py、
    convert_emotion_to_hf.py、huggingface_emotion_to_pretrained.py、
    configs/cosyvoice2.yaml、annotate_data/（含 best_model.pth、emotion2vec_base.pt）、
    emotion_prototypes_ncssd/*.npy；
- 本地论文资源：`reference/23S136160-王思睿.pdf`（硕士论文）、
    `reference/2509.20378v1.pdf`（Emo-FiLM）、`reference/2601.04656v1.pdf`
    （FlexiVoice）、`reference/arXiv-2509.20378v1/`（LaTeX 全文）。

### 0.6 会话交接记录（2026-08-06）

- 已从零完整重读 08-02~08-04 全部报告与计划（含三份大计划全文：
  sentlvl-fixes / eval-v3-refactor / eval-prereqs），专利交底书全文与专利
  新颖性调研报告全文（含此前被截断的 4.3/4.4/5.1 中段）均已补齐。
- 阶段 1 子智能体状态：票据 01（本地代码侦察）已收口
  （`findings/01-local-code.md`）；票据 02（作者 RL 代码审计）与票据 03
  （扩散/流匹配强化学习文献）仍在运行；其已抓取材料（FlowTTS-GRPO /
  Flow-GRPO / FlexiVoice 全文等）已由主代理直接核验并入最终报告。
- 阶段 2 三个方向（情感奖励配方、专利差异、可行性落点）由主代理依据已收口
  材料直接完成，见最终报告 §2.5/§2.6/§3–§4。
- 最终调查报告文件名为 `docs/reports/2026-08-05-emofilm-rl-diffusion-research.md`
  （沿用任务书建议命名；完成日期为 2026-08-06）。
  （日期按实际完成日，任务书原建议 08-05 可作历史引用）。

### 0.3 主代理本地复核（已完成）

- `cosyvoice/flow/flow.py:284` `CausalMaskedDiffWithDiT` 存在；`cosyvoice/flow/flow_matching.py` 含 `ConditionalCFM`/`CausalConditionalCFM`（Matcha 条件流匹配）。
- `cosyvoice/llm/llm.py:664` `CosyVoice3LM`、`cosyvoice/tokenizer/tokenizer.py:274` `CosyVoice3Tokenizer`、`cosyvoice/hifigan/generator.py:572` `CausalHiFTGenerator`、`cosyvoice/cli/model.py:397` `CosyVoice3Model`、`cosyvoice/cli/cosyvoice.py:189` `CosyVoice3` 均存在。
- `conf/` 尚无 `cosyvoice3.yaml`；CV3 权重位于 ModelScope 缓存（`~/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`）。
- 作者代码 `reference/Emo_PA_code_data/` 含 `reward/`、`reward_tts.py`、`reward_server.py`、`run_dual_gpu.sh`、`prepare_ncssd_*.py`、`compute_ncssd_prototypes.py`、`convert_emotion_to_hf.py`、`huggingface_emotion_to_pretrained.py`、`configs/`、`cosyvoice/`（含 DPO 路径）、`Matcha-TTS/`。
- 论文/PDF 本地资源：`reference/23S136160-王思睿.pdf`（硕士论文）、`reference/2509.20378v1.pdf`（Emo-FiLM）、`reference/2601.04656v1.pdf`（FlexiVoice）、`reference/arXiv-2509.20378v1/`。

---

## 1. 任务背景（已确认事实，无需重复调研）

### 1.0 已确认的本地资源补充（2026-08-05 主代理复核）

- 哈工大硕士论文全文已本地存在：`reference/23S136160-王思睿.pdf`
  （已用 pdftotext 提取至 `/tmp/thesis_full.txt`，2816 行；第 3 章即
  "基于强化学习的情感可控语音合成偏好对齐方法"，与专利/任务高度对应）。
- 作者 RL 代码（`reference/Emo_PA_code_data/`）除 GRPO（verl 入口
  `run_dual_gpu.sh`）外，还包含完整 DPO 路径：`cosyvoice/bin/train_dpo.py`、
  `cosyvoice/llm/llm_dpo.py`、`cosyvoice/utils/{losses_dpo,executor_dpo,
  train_utils_dpo}.py`、`cosyvoice/dataset/processor_dpo.py`。
- 作者 GRPO 关键实现（已复核）：奖励 = 片段级 emotion2vec-原型余弦
  （`reward/emotion.py`；MFA 默认跳过、用字符比例估计时间边界）+ CAM++
  整句说话人余弦（`reward/speaker.py`）；`reward/normalize.py` 实现
  A=(r_emo−μ)/σ + λ·(r_sv−μ)/σ（λ=1.0），verl 奖励函数入口
  `reward_tts.py::compute_score`；token 格式 `<|s_i|>` 字符串解析。
  GRPO 超参：lr=1e-6、KL=0.05、组大小 G=4、温度 0.8、top_p=0.95、
  top_k=25、FSDP 2 GPU、总 3 epoch。
- 本仓库 `conf/cosyvoice3.yaml` 尚不存在；CV3 官方权重经 ModelScope 下载，
  推理冒烟与脚本在 `exp/cosyvoice3_baseline/`（`smoke.md`、
  `gen_cv3.py`、`run_infer.sh`）；`cosyvoice/flow/flow_matching.py` 存在。

### 1.1 项目进度（08-02 ~ 08-04，全部已实现）

- 四次实验（v1 / 5-epoch disabled / 27-epoch disabled / sentlvl）共享"弱条件通路"：
  Emo-SIM 卡在 ~66 平台；句级输入 CE 是"文本可分捷径"（探针读回 100%、梯度与
  loss_tts 反方向、WER 劣化）。
- 生成/评测协议是平台主因：中性声学 prompt 下即使 token 完美，5-way 判别也只有
  ~53%；情感匹配 prompt 下四模型 Emo-SIM 跳到 84.3–87.4、5-way 73–83%。
- CosyVoice3 官方基线（instruct + 中性 prompt）Emo-SIM=70.76，未质变；基座非瓶颈。
- 下一步主线：R2 监督改造——把情感监督从"文本嵌入 CE"移到"音频/生成链下游"。
  待决策方向（D1）：DPO/GRPO 音频奖励 vs span 词级监督 vs 两者分阶段。
- 本仓库已具备 CosyVoice3 全套代码类：`CosyVoice3LM`、`CausalMaskedDiffWithDiT`
  （DiT 流匹配/扩散）、`CausalHiFTGenerator`、`CosyVoice3Tokenizer`；
  官方权重 Fun-CosyVoice3-0.5B-2512 已本地可用。
- 作者 RL 代码在 `reference/Emo_PA_code_data/`：verl 入口 `run_dual_gpu.sh`、
  奖励 `reward_tts.py` / `reward_server.py` / `reward/*`（emotion2vec 情感一致性 +
  CAM++ 说话人一致性 + 情感原型相似度 + 组内归一化）、NCSSD 冲突数据准备脚本、
  `emotion_prototypes_ncssd/*.npy`。

### 1.2 专利现状（来自专利新颖性调研报告，不做扩展调研）

- 拟申请专利：《一种基于强化学习的情感可控语音合成偏好对齐方法》，核心组合：
  E1 细粒度情感控制标签（类别+强度+作用文本范围）；E2 同一条件候选组采样；
  E3 语义/音色/双重冲突条件数据；E4 片段级情感原型相似度局部奖励；
  E5 整句说话人一致性奖励；E6 组内 z-score 归一化联合优势；E7 概率比截断 +
  参考策略 KL 约束的 GRPO；E8 推理直出。
- 风险：FlexiVoice（arXiv:2601.04656）公开了 E2+E3+E5+E6+E7 主干；
  CN122024692A、CN122157635A 覆盖 GRPO+多奖励+组内优势；EMORL-TTS 公开了
  词级局部奖励与 clip+KL；E4 的创造性强度存疑（局部奖励思想先例充分）。
- 唯一相对未公开的完整组合点：E4（标签作用范围定位 + 片段情感原型相似度奖励）
  与 E5/E6/E7 的完整链路。
- 本任务要求：新方案必须引入"扩散策略"，以 RL 为主且带扩散策略，从而在方向上
  与专利权利要求形成差异（扩散/流匹配生成器作为策略网络的后训练，不在专利
  权利要求明确保护范围内）。

### 1.3 "扩散策略"在本任务中的界定（待调查确认，初步假设）

初步理解为：扩散/流匹配（Diffusion / Flow Matching）生成模型作为可优化的策略
网络，并对其做强化学习/偏好对齐后训练；CosyVoice3 的 DiT Flow
（`CausalMaskedDiffWithDiT`）即为本地可用的扩散/流匹配载体。调查时需覆盖：

- 扩散/流匹配生成器的 RL 训练范式（如 F5R-TTS 的流匹配 + GRPO、EASPO 的
  扩散 TTS + 偏好优化、DDPO/DIPO/DITTO 等扩散模型 RL 方法）；
- CosyVoice3 官方 RL 后训练（DiffRO/MTR 的可微奖励、`_RL` 权重）与本仓库代码的
  兼容性；
- 把 RL 监督作用于 LLM（token 生成）与/或 Flow（声学渲染）两种接入点的利弊；
- 与专利 E1–E8 的差异论证（RL 为主 + 扩散策略）。

---

## 2. 硬性要求（全程必须遵守）

1. 必须引入"扩散策略"：新方案必须包含扩散/流匹配策略成分，但以 RL 为主；
   不要求扩散策略与 RL 深度耦合，只要求 RL 为主、扩散策略确实存在且可解释。
2. 方案选择原则（重要）：找出一条最可行的路，而不是罗列多个方向。优先级：
   - 在我们项目已有实现基础上改造最小、且效果明显；
   - 作者（Emo-PA）实现可算法级复刻的直接优先；
   - CosyVoice3 自带扩散方案/RL 训推实现可直接调用的优先；
   - 大面积冲突于现有项目实现、工程量极大的方案不考虑。
3. 调查要全面深入：分细粒度调查任务交给子智能体并行探索，不限制调查时间，
   可动态添加子智能体；每个结论必须可追溯（文件路径、论文编号、代码行）。
4. 专利范围：不扩展专利新颖性调研；直接利用现有调研报告中的参考论文列表，
   范围外确需新读的论文用检索工具获取；取不到时由子智能体使用终端浏览器控制
   工具直接获取，不得跳过或假装完成。
5. 报告语言：所有关键词有公认中文翻译的用中文（如"待测音频"而非"hyp audio"、
   "扩散策略/流匹配"而非"diffusion policy/flow matching"缩写）；报告结束后必须
   重新审查是否符合本任务书一切要求。
6. 最终交付：一份非常详细、可追溯的调查报告（建议
   `docs/reports/2026-08-05-emofilm-rl-diffusion-research.md`），包含：
   - 调查综述（按角度分节，含证据与来源）；
   - 后续实验前进方向：一个主方案 + 最多一个备选方案；
   - 主方案要有明确的技术路线、接入点（代码文件/配置/数据）、成本估计、
     预期收益、与专利的差异说明、风险与验证步骤。

---

## 3. 调查角度与任务分解

### 阶段 1（并行基础调查，4 个子智能体）

1. **本地代码侦察：CosyVoice3 与扩散/流匹配的关系**
   - `cosyvoice/flow/flow.py` 的 `CausalMaskedDiffWithDiT`、`cosyvoice/flow/DiT/*`、
     `cosyvoice/flow/flow_matching.py`（条件流匹配采样/训练接口）；
   - `CosyVoice3LM`、`CosyVoice3Tokenizer`、`CausalHiFTGenerator`、`CosyVoice3Model`
     的接口与现配置（是否已有 `conf/cosyvoice3.yaml`）；
   - 现有 EmoFiLM（`Qwen2LM_Emotion` + `CausalMaskedDiffWithXvec`）与 CV3 的接入点
     差异；把情感 FiLM/监督改造迁移到 CV3 的最小改动面；
   - 官方 CV3 权重目录（`pretrained_models/`）与推理入口现状。
2. **作者 RL 代码可复刻性审计（Emo-PA）**
   - 精读 `reference/Emo_PA_code_data/` 的 `reward_tts.py`、`reward_server.py`、
     `reward/*`、`run_dual_gpu.sh`、`prepare_ncssd_conflict.py`、
     `prepare_ncssd_data.py`、`compute_ncssd_prototypes.py`、
     `convert_emotion_to_hf.py` / `huggingface_emotion_to_pretrained.py`、
     `configs/cosyvoice2.yaml`；
   - 明确 token 格式（`<|s_i|>` 字符串 vs 本仓库裸 token）、verl 数据列、
     奖励计算的数据流、模型转换流程；
   - 评估：直接在作者流程上换成本仓库数据/权重的最小改动；或把作者奖励函数
     移植到本仓库训练循环的最小改动；GPU/显存/时间成本估计。
3. **扩散/流匹配策略 + RL 文献调查**
   - 精读/核验：F5R-TTS（arXiv:2504.02407，流匹配 + GRPO）、EASPO
     （arXiv:2509.25416，扩散 TTS + 偏好优化）、DDPO/DIPO/DITTO（扩散模型 RL
     通用范式）、DiffSinger RL 或类似语音扩散 RL；CosyVoice3 官方 RL
     （DiffRO/MTR，arXiv:2507.05911 + 官方 README）；
   - 回答：把 RL 监督放在"LLM token 生成"还是"Flow/DiT 声学生成"上更符合
     "扩散策略"要求、更简单可行、更有效？两者并存的先例？
   - 与本项目 CosyVoice3 代码的兼容性。
4. **官方 CosyVoice3 RL/后训练与本地可复刻性**
   - 官方 Fun-CosyVoice3-0.5B-2512（base / `_RL`）的差异、官方训练/RL 脚本
     （`train.py`、diffusion/flow 训练、DiffRO/MTR 奖励）；
   - 官方 RL 是否基于 verl、是否需要官方仓库（`gitserver.onethingai.com` 镜像或
     ModelScope/HF），本地 CLI 能否直接训练/推理；
   - 本地数据（20774 训练 + 1092 CV + 2500 评测）迁到 CV3 需要重抽 token 的
     工具与成本。

### 阶段 2（方案合成后按需派发）

5. **情感 RL 论文深读（奖励配方与失败模式）**：FlexiVoice S1/S2、EMORL-TTS、
   GLM-TTS、HPRO、RLAIF-SPA、Emo-LiPO、RRPO、DiffRO、Emo-PA 论文
   （arXiv:2509.20378）——提炼可直接复用的奖励公式、超参、数据结构、评测口径。
6. **专利差异论证**：把候选主方案与专利 E1–E8 逐项对比，确认新颖性差异点与
   规避点；结合扩散/流匹配策略的引入论证创造性。
7. **可行性验证**：对选定的主方案做代码级落点确认（训练入口、数据管线、
   推理链路、评测脚本），给出分步实施路径与冒烟实验设计。

---

## 4. 主方案与备选方案的决策标准

1. 最简单可行且效果最优；既不要极其复杂但可能效果更好的，也不要明显效果不行的。
2. 优先：项目已有主线方向（R2 音频奖励/RL）、作者 Emo-PA 实现、CosyVoice3 直接
   使用。
3. 必须满足：RL 为主 + 扩散策略确实存在。
4. 备选方案最多一个，且与主方案有本质区别（如监督接入点或训练范式不同）。

---

## 5. 最终报告验收清单（完成后逐项自查）

- [x] 08-02 ~ 08-04 全部报告与计划已读全，文件先后/调用关系已理清；
- [x] 专利现状与参考论文列表已纳入，未做扩展专利调研；
- [x] 四个调查方向（本地代码、作者 RL 复刻、情感 RL 启示、扩散+RL 并存）都有
      可追溯结论；
- [x] 主方案 + 最多一个备选方案，满足"RL 为主 + 扩散策略"；
- [x] 所有结论有来源（本地路径/论文编号/代码行）；
- [x] 关键词使用无歧义中文；
- [x] 报告结构清晰易读、专业详实；
- [x] 若存在必须由用户拍板的方向性选择，按 batch-grill-me 技能与用户逐轮确认。
      （本任务由主代理直接给出推荐主/备方案，无必须由用户拍板的阻塞项；如用户
      对主方案有异议，可基于本报告进行单轮确认。）
