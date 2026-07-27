---
status: accepted
supersedes: [0003, 0004, 0005, 0007, 0013, 0014]
---

# EmoFiLM v2 细粒度控制修复合同与替代决策

静态架构审计（`2026-07-22-emofilm-architecture-static-audit.md`）与模型设计静态分析（`2026-07-24-emofilm-model-design-static-analysis.md`，前者见仓库内 `docs/reports/`，后者位于仓库父目录）确认：EmoFiLM v1 优化基线在整体质量上已验证，但其细粒度情感控制链路存在一组互相耦合的合同级缺陷——输入端 classifier 与双流训练混淆了控制语义、固定 `max_len=200` 与 EOS-only 静默成功掩盖了长度/结束失败、词级伪标签被当作可用词级监督、评测只判整句 aggregate 无法分辨 span/强度/转移。为在不破坏 v1 权威基线的前提下修复这些缺陷，项目新增版本化合同 `emofilm_v2`（`CONTRACT_NAME = "emofilm_v2"`，`SCHEMA_VERSION = 2`），定义监督 span、生成 row、评测 row、aggregate 与运行身份的必需字段、版本字段与跨文件引用关系，并以本文逐条点名被新证据替代的局部历史决策。v1 基线锚定 git commit `9c6d84b`；活跃代码可随主线演化（重构），v1 实验产物磁盘冻结只读。详见 ADR-0020（原 "一律只读 / 不得原地覆盖" 措辞据此重定义）。

## 被新证据替代的局部决策

- **0003 的「输入端 emotion loss / 固定 max_len=200 / EOS-only 静默成功」被推翻**：下游 speech-token hidden state 监督取代输入端 classifier CE（控制信号应作用并被观测于输出端，而非在 text-token embedding 上再加一个线性头）；`min_len/max_len` 改由文本长度 + resolved ratio + hard cap 推导并在解码前保证 `max_len > min_len`；成功/失败以结构化 `finish_reason ∈ {eos, max_len_reached, invalid_token_retry_exhausted, sampler_error, input_rejected}` 显式记录，循环自然结束不再等于静默成功。
- **0004 的「仅整句 aggregate 评测判胜负」被推翻**：评测扩展为逐样本 / 逐 span / 强度 / 转移评测，aggregate 改为从已持久化 rows 确定性派生；v1 整体质量指标（WER / Emo-SIM / DTW）保留为历史兼容与整体退化守门，但不再是唯一判据。
- **0005 的「词级伪标签当作可用词级监督」被降级**：IEMOCAP 词级 span 的 `supervision_granularity = word` / `label_source = word_annotator_pseudo_label`（词级标注器伪标签），但其句级广播弱监督本质由 `provenance.weak_supervision = sentence_broadcast` 显式标记（句级情感广播到词级 span，非词级真值）。每条 span 携带 soft distribution、完整 VAD、连续 arousal、raw score 与校准状态；词边界信息从 `word_blocks/*.pt` 与 `provenance/iemocap_word_boundaries.jsonl` 显式进入 span，不再以 hard label 形式声称词级真值。
- **0007 的「pre-min-len EOS resample」保留为采样语义，不冲突**：作者的 EOS 重采样与 RAS fallback 的原始-score 语义继续保留；v2 仅在长度与 finish 上新增合同（结构化 `finish_reason`、解码前 `max_len > min_len`、非 EOS 不落正式 WAV），不改变采样分布语义本身。
- **0013 的「冻结 local-completion 合同」保留**：v1 的 ESD 测试清单 / FEDD A-B 分区 / parquet / 批量推理 / `emofilm-eval-v2` 合同仍冻结只读；v2 是后继版本化合同，新增 span / transition / intensity / identity 字段与校验，不覆盖、不改写 v1 产物。FEDD-B 的真实 MFA 词边界拼接标为 `boundary_evidence_tier = exact`，FEDD-A 的 MiMo 词数中点近似标为 `approximate`，且 approximate 不进 exact aggregate。
- **0014 的「五类测试面」扩展**：v2 测试面在 v1 五类外部合同基础上扩展为含 span / transition / intensity / identity 的合同测试，仍遵循“只锁外部行为与 schema，不锁私有函数名 / 层数 / 内部张量顺序”的原则。

## 明确保留（不替代、不重开）

- **canonical checkout** 仍是唯一运行入口；命令从仓库根执行，env 为 `source scripts/activate_env.sh`。
- **冻结 train/cv 成员关系**沿用 v1 既有 parquet 的 `prepare_emofilm_v1_data.frozen_split_ids`（train 20774 / cv 1092），v2 不重切。
- **统一清洁规则**与 v1 一致（manifest 路径规整、文本规范化、reject 预算、parquet 打包）。
- **Flow / HiFT prompt 职责**不变：speaker / Flow / HiFT prompt 继续作为声学侧条件；被删除的只是进 LLM 条件的 `prompt_text / prompt_speech_token / prompt_emotion / prompt_intensity` 死字段（v1 `inference` 签名接受却全部 `del`）。
- **EmoFiLM v1 基线资产边界**：v1 基线锚定 git commit `9c6d84b`，活跃代码可随主线演化（重构；ADR-0020）；v1 实验产物（`data/contracts/emofilm_v1/`、`exp/emofilm_v1/`、`artifacts/emofilm_v1/`）磁盘冻结只读。现有 ADR 0001-0018 为已发生的历史决策记录，字节冻结不动。
- **训练 base** 必须是 CosyVoice2 `llm.pt`（`CosyVoice-BlankEN` 仅作 tokenizer）。
- **死配置字段**：v2 resolved 配置不得含 `mix_ratio`（双流）、输入端 `emo_loss_weight`（输入端 classifier CE）、顶层 `alpha`（v1 `conf/emo_film.yaml:45` 标注 `# 配置占位` 的未消费占位；采样超参归属逐生成 `decode_config`），由 `assert_no_dead_config` 强制。

## 合同不变量（摘要，详见 `docs/contracts/emofilm_v2_schema.md` 与 MAP.md §3）

监督 span 每条携带控制值、soft distribution、完整 VAD、连续 arousal、离散控制标签、raw score、校准状态（未校准不得命名 `confidence`）、per-target 有效 mask、监督权重与可溯源 provenance；ESD fixed-medium 的 `intensity_mask` 恒为 False。生成 row 每条绑定 source / checkpoint / control / prompt 四族身份引用、解码配置、`finish_reason` 与输出 WAV；仅 `eos` 进声学与正式 WAV。评测 row 每条绑定 generation row + 控制 span + evaluator 版本 + 逐样本/逐 span 指标；任一缺失 / 重复 / 非 EOS / 身份不一致 → 携 `utt_id` hard-fail。运行 identity 记录每个 optimizer 参数组的张量数 / 参数数 / 初始 LR / weight decay / scheduler 类型，并绑定数据合同 / base ckpt / resolved config / seed / 输出 ckpt / 源码身份。

## 证据来源

- `docs/reports/2026-07-22-emofilm-architecture-static-audit.md`（仓库内）
- `2026-07-24-emofilm-model-design-static-analysis.md`（仓库父目录）
- 本仓库 `docs/contracts/emofilm_v2_schema.md`（人类可读单一来源）与 `tools/build_emofilm_v2_contract.py`（校验权威实现）。
