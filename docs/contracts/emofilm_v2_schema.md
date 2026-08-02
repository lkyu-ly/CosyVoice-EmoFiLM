# EmoFiLM v2 数据合同 Schema

本文是 EmoFiLM v2 细粒度情感控制修复的**单一可读来源**。所有下游票据
（02-12）只读本文即可对齐字段、类型、必需性与跨文件引用关系。校验权威实现
在 `tools/build_emofilm_v2_contract.py`（`CONTRACT_NAME = "emofilm_v2"`，
`SCHEMA_VERSION = 2`）；本文与其保持一致，冲突以代码校验器为准。

v1 合同（`data/contracts/emofilm_v1/`、`tools/build_emofilm_contract.py`）全程
只读，v2 是**后继版本化合同**，不覆盖、不就地改写 v1 产物（ADR-0019）。

## 顶层不变量（MAP.md §3 摘要）

- **协议**：唯一 LLM 协议为非流式 target-only 单流。训练
  `SOS + FiLM(target text) + task + teacher-forced target speech`，目标只含
  target speech + EOS；推理 `SOS + FiLM(target text) + task`。无 fill token、
  无交错文本、无双流。`prompt_text / prompt_speech_token / prompt_emotion /
  prompt_intensity` 不进 LLM 条件；speaker / Flow / HiFT prompt 条件保留。
- **长度**：`min_len / max_len` 由文本长度 + resolved ratio
  （`min_token_text_ratio`, `max_token_text_ratio`）+ hard cap 推导；每个请求
  解码前必须 `max_len > min_len`；hard cap 不足时解码前结构化拒绝
  （`finish_reason = input_rejected`）。
- **生成**：仅 `finish_reason == eos` 进声学与正式 WAV；其余 finish reason
  不落正式 WAV。
- **评测**：任一缺失 / 重复 / 非 EOS / 身份不一致 → 携 `utt_id` hard-fail，
  禁止跳过算部分均值。aggregate 只从已持久化 rows 确定性派生。FEDD-A
  （approximate）不进 FEDD-B（exact）aggregate。
- **死配置**：v2 resolved 配置不得含 `mix_ratio`（双流）、顶层 `alpha`（v1
  配置占位；采样超参归属 `decode_config`）。`emo_loss_weight` 是可选 input-end
  句级监督权重（默认 0=关闭），不属于死字段。

## 标签 id 空间（与 `cosyvoice/tokenizer/emo_tokenizer.py` 一致）

| 维度 | 取值 | 说明 |
|---|---|---|
| `control_emotion_id` | `1..5` | `{1:ang, 2:hap, 3:neu, 4:sad, 5:sur}`；`0=pad` 不作为控制值。注意 WordSequenceModel 内部 `{0:ang..4:sur}` 与 tokenizer **不同源**，v2 控制标签一律走 tokenizer id 空间。 |
| `control_intensity_id` | `1..3` | `{1:low, 2:medium, 3:high}`；`0=pad`。 |

---

## 1. SupervisionSpan

一条监督 span = 控制值 + 监督属性 + 可溯源来源。每条携带完整 soft
distribution、完整 VAD、连续 arousal、离散控制标签、原始 raw score、校准
状态、per-target 有效 mask、监督权重与来源。IEMOCAP 词级标签必须标记为句级
广播弱监督（非词级真值）。ESD fixed-medium：`intensity_mask` 恒为 False。

校验器：`validate_span(row)`。

| 字段 | 类型 | 必需 | 语义 |
|---|---|---|---|
| `utt_id` | str | 是 | 非空；归属样本。 |
| `label_source` | str | 是 | 来源语义。典型值：`word_weak_sentence_broadcast`（IEMOCAP）、`construction_known_transition`（FEDD）、`esd_fixed_medium_control`（ESD）。 |
| `supervision_granularity` | str | 是 | `utterance` / `span` / `word`。 |
| `start_sec` | float | 是 | `>= 0`，且 `< end_sec`。 |
| `end_sec` | float | 是 | span 结束秒。 |
| `emotion_soft_distribution` | list[5] float | 条件必需 | `emotion_mask=True` 时必需；`emotion_mask=False` 时可缺省。5 类情感概率，每项 `[0,1]`，和为 1（容差 1e-6）。**one-hot 是硬标签（ESD 数据集全局标签 / FEDD 构造标签）的诚实表示**，允许。 |
| `vad` | list[3] float | 否 | `[valence, arousal, dominance]`。始终可选；存在时必须长度 3 数值。IEMOCAP 标注器有完整 VAD；ESD/FEDD 无则缺省（不得伪造）。 |
| `arousal` | float | 条件必需 | `intensity_mask=True` 时必需（连续强度目标）；`intensity_mask=False` 时**必须缺省**（无连续强度目标）。 |
| `control_emotion_id` | int | 是 | `[1,5]`；喂给 FiLM 的离散控制值（控制输入，与是否有真值无关）。 |
| `control_intensity_id` | int | 是 | `[1,3]`；喂给强度嵌入的离散控制值（即使无强度真值也填控制输入）。 |
| `raw_score` | float | 条件必需 | `calibrated=True` 时必需（被校准的那个值）；`calibrated=False` 时可选（IEMOCAP 未校准标注器分数可保留；ESD/FEDD 无模型分数则缺省）。 |
| `calibrated` | bool | 是 | 是否经过校准（始终必需）。 |
| `calibration` | `{method, version}` 或 null | 条件必需 | `calibrated=True` 时必需且 `method`/`version` 非空；`calibrated=False` 时必须为 null/缺省。 |
| `emotion_mask` | bool | 是 | 情感监督对本目标是否有效（门控 `emotion_soft_distribution`）。 |
| `intensity_mask` | bool | 是 | 强度监督对本目标是否有效（门控 `arousal`）。`intensity_policy` 为 `fixed_*` 时**必须 False**。 |
| `supervision_weight` | float | 是 | `[0,1]`；样本/span 级监督权重。 |
| `provenance` | str \| dict | 是 | 非空；来源溯源（模型版本 + 数据集 + 边界来源）。 |
| `intensity_policy` | str | 否 | 典型值 `fixed_low/medium/high`、`ground_truth`。`fixed_*` 触发 `intensity_mask=False`。 |
| `confidence` | float | 禁 | **永远不是合法字段**（无论是否校准）。诚实字段是 `raw_score` + `calibration`（MAP §3）。 |

**条件必需小结（按数据来源诚实表达，不伪造监督）：**
- IEMOCAP（标注器）：`emotion_mask=True` + soft dist；`intensity_mask=True` + arousal；`vad` 有；`calibrated` 视校准而定。
- ESD（数据集全局标签）：`emotion_mask=True` + one-hot soft dist；`intensity_mask=False` + 无 arousal；无 vad/raw_score；`calibrated=False`。
- FEDD（构造 emo_from/emo_to）：同 ESD（硬标签 one-hot；无 VAD/arousal/model score）；`intensity_policy=fixed_medium`。

合并规则：相邻 span 仅控制值（`control_emotion_id`, `control_intensity_id`）
与监督属性（`emotion_mask`, `supervision_weight`, `calibration`, `provenance`
兼容时）可合并，合并必须可溯源（保留被合并 span 的 `provenance`）。

> **值域可扩展：** `supervision_granularity`（当前 `{utterance, span, word}`）
> 与 `boundary_evidence_tier`（当前 `{exact, approximate}`）的合法值集合由
> `tools/build_emofilm_v2_contract.py` 的 `SPAN_GRANULARITIES` /
> `BOUNDARY_EVIDENCE_TIERS` 常量定义；下游若引入新粒度/证据等级，扩展该常量
> 并同步更新本文，无需改动校验逻辑。

---

## 2. GenerationRow

一条生成结果 = 身份 + 控制/prompt 引用 + 解码配置 + 结构化 finish reason
+ 输出 WAV。`finish_reason ∈ {eos, max_len_reached,
invalid_token_retry_exhausted, sampler_error, input_rejected}`。
仅 `eos` 落正式 WAV。`skip-existing` 仅在完整逐条身份一致时复用。

校验器：`validate_generation_row(row)`；常量：`FINISH_REASONS`。

| 字段 | 类型 | 必需 | 语义 |
|---|---|---|---|
| `utt_id` | str | 是 | 非空。 |
| `finish_reason` | str | 是 | 见上枚举。 |
| `source_revision` | str | 身份族（≥1） | 干净 git revision sha。 |
| `source_patch_bundle` | dict | 身份族 | 不可变 patch bundle（dirty worktree 重建用）。 |
| `source_patch_sha256` | str | 身份族 | patch bundle 的 sha256。 |
| `checkpoint_sha256` | str | 身份族（≥1） | 64-char hex；checkpoint 内容 sha256。 |
| `checkpoint_ref` | dict | 身份族 | 结构化 checkpoint 引用（path + sha256 + 训练 identity）。 |
| `control_row_ref` | str | 身份族（≥1） | 指向 SupervisionSpan/control row（control_row 字典亦可）。 |
| `prompt_row_ref` | str | 身份族（≥1） | 指向 speaker/Flow/HiFT prompt row（prompt_row 字典亦可）。 |
| `decode_config` | dict | 是 | 解码配置：`min_token_text_ratio`、`max_token_text_ratio`、`max_len_hard_cap`，以及采样超参（top_p / top_k / RAS alpha 等）。 |
| `seed` | int | 是 | per-request 固定随机种子（默认 1986）；per-utt 生成前重置 torch+cuda RNG。seed 变化→不同指纹→不复用。 |
| `wav_path` | str | 条件必需 | `finish_reason=eos` 时必需（workspace-relative POSIX）；非 eos 时**不得**出现。 |

身份引用约束：source / checkpoint / control / prompt 四族各至少满足一个键，
缺一即 hard-fail。`checkpoint_sha256` 若出现必须是 64-char hex。

> ADR-0020：`wav_sha256` 字段已**移除**——禁止用 WAV 内容哈希标定产物；
> 产物身份用 `wav_path` + 结构化身份字段。safe-resume 仅校验文件存在性 + 逐条身份指纹。

---

## 3. EvaluationRow

一条逐样本 / 逐 span 评测结果 = generation row 引用 + 控制 span + evaluator
版本 + 指标。任一缺失 / 重复 / 非 EOS / 身份不一致 → 携 `utt_id` hard-fail。

校验器：`validate_eval_row(row)`。

| 字段 | 类型 | 必需 | 语义 |
|---|---|---|---|
| `utt_id` | str | 是 | hard-fail 归属键。 |
| `generation_row_ref` | str | 是（或 dict） | 指向 GenerationRow（`generation_row` 字典亦可）。 |
| `control_span_ref` | str | 是（或 dict） | 指向 SupervisionSpan（`control_span` 字典亦可）。 |
| `evaluator` | dict | 是 | `{name, version, ...}`；`name`/`version` 非空。 |
| `evaluator.name` | str | 是 | 评测器名称（如 `emotion2vec-v2`）。 |
| `evaluator.version` | str | 是 | 评测器版本（冻结快照标签）。 |
| `evaluator.label_space` | list[str] | 否 | 标签空间（如 5 类情感）。 |
| `evaluator.sample_rate_hz` | int | 否 | 采样率。 |
| `evaluator.frame_rate_hz` | float | 否 | 帧率（emotion2vec-base=50Hz）。 |
| `evaluator.self_evidence_risk` | bool | 否 | 与 IEMOCAP 弱监督生成器共享模型时标自证风险。 |
| `boundary_evidence_tier` | str | 是 | `exact` / `approximate`；用于 aggregate 分离。 |
| `metrics` | dict | 是 | 逐样本 / 逐 span 指标（可为空占位，但键必需）。 |

---

## 4. Aggregate

聚合指标，从已持久化 EvaluationRow 确定性派生。携带 `evidence_tier` 以支持
exact / approximate 分离（FEDD-B exact 不混入 FEDD-A approximate）。

校验器：`validate_aggregate(row)`。

| 字段 | 类型 | 必需 | 语义 |
|---|---|---|---|
| `evidence_tier` | str | 是 | `exact` / `approximate`。 |
| `metric_contract_version` | str | 否 | 指标合同版本标签（如 `emofilm_v2_eval`）。 |
| `metrics` | dict | 是 | 聚合指标（emo_sim / dtw / wer / 强度 / 转移等）。 |
| `n_samples` | int | 是 | 参与聚合的样本数，`>= 0`。 |

---

## 5. Identity（v2，由 `tools/write_emofilm_run_identity.py` 扩展绑定）

- **训练 identity**：记录每个 optimizer 参数组的张量数 / 参数数 / 初始 LR /
  weight decay / scheduler 类型 + 关键参数；绑定数据合同、base checkpoint、
  resolved config、seed、输出 checkpoint、源码身份（干净 revision 或保存
  patch bundle）。
- **聚合/生成 identity**：绑定确定的 rows 集合（generation manifest）+
  evaluator 身份 + 解码配置集合。
- v1 历史运行身份原样保留为“运行发生时”的事实；v2 身份新增逐样本绑定，
  不改写 v1 入口签名。

---

## 6. 跨文件引用关系

```
SupervisionSpan (control row)
        │  control_row_ref / control_span_ref
        ▼
GenerationRow ──source_revision/checkpoint_sha256──▶ Identity / Checkpoint
        │  generation_row_ref                         ▲
        ▼                                             │
EvaluationRow ──evaluator──▶ Evaluator (frozen)      │
        │                                               │
        ▼ (确定性派生，按 evidence_tier 分离)           │
Aggregate ────────────────────────────────────────────Identity (rows 集合)
```

- GenerationRow 通过 `control_row_ref` 绑定 SupervisionSpan（控制输入）。
- GenerationRow 通过 `prompt_row_ref` 绑定 speaker / Flow / HiFT prompt
  （不是 LLM 条件；仅声学侧）。
- EvaluationRow 通过 `generation_row_ref` + `control_span_ref` 双向绑定生成
  结果与控制 span；缺一不可。
- Aggregate 从已持久化 EvaluationRow 派生；`evidence_tier=approximate` 的
  rows 不进 `evidence_tier=exact` 的 aggregate（由评测侧 `eval/eval_emofilm_v2.py`
  实现分离，本合同只定义 tier 值）。
- 所有身份引用缺失 / 重复 / 非 EOS 落正式 WAV / sha 不符 → hard-fail 并携带
  `utt_id`，禁止静默跳过。

## 7. 路径规整

所有路径字段（`wav_path`、各 `*_ref` 若为路径、`source_patch_bundle` 路径等）
统一为 **workspace-relative POSIX**（`tools/build_emofilm_v2_contract.py ::
normalize_workspace_path`），禁止绝对路径泄漏到合同产物。仓库外路径 →
ValueError。语义与 v1 `normalize_workspace_path` 等价，v2 独立重写以避免
引入 v1 的 torch / pyarrow 重依赖。
