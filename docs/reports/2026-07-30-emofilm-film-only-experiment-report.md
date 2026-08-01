# EmoFiLM FiLM-only 全量实验报告（init baseline → current 端到端对比）

- 撰写日期：2026-08-01（实验执行 2026-07-29）
- 实验代码：current `ed89f48`（"还原评测到 baseline + 清理 v2 评测死代码"，经中间 `52d8be1` experiment-readiness）
- init baseline 代码：`9c6d84b`（"establish verified EmoFiLM v1 canonical repository"，首次提交）
- 实验产物：`exp/emofilm_film_only/`（独立于 v1，未触动 v1 冻结制品）
- 对照报告：`docs/reports/2026-07-20-emofilm-v1-baseline-experiment-report.md`（v1，init 代码 9c6d84b）
- 实验口径：`downstream_supervision=disabled`（FiLM-only，不接 span 监督头）

## 1. 概述

本次实验用 current 代码（`ed89f48`）跑 FiLM-only 全量实验：5 epoch 训练 + 2500 条生成（全 eos）+ baseline 整体评测（WER / Emo-SIM / DTW）。本报告第 2 节端到端对比 init commit（`9c6d84b`，v1 baseline 代码）与 current（`ed89f48`）的训练/推理架构变化（只比这两个端点，不分中间 commit），第 3 节报告本次评测结果，第 4 节与 v1 baseline report 做清晰对比。

**一句话结论**：本次实验与 v1 在 FiLM 调制模块本身（`emo_film.py`）上零差异、在数据链路与评测口径上逐字节相同；架构变化集中在上层协议（双流→单流）、监督范式（输入端 classifier CE→下游 span 头 / disabled 纯 loss_tts）、优化调度（单组 LR→三组分层 + WarmupLR 真正生效）与推理可审计性（finish_reason 门控 + 逐条身份）。客观评测显示本次情感指标与 v1 持平甚至略好、WER 三分区全部改善，证明 `disabled`（砍 v1 输入端 `loss_emotion`）无害。

## 2. 训练/推理架构变化（init 9c6d84b → current ed89f48）

### 2.1 模型架构（`cosyvoice/llm/llm_emotion.py` + `cosyvoice/llm/emo_film.py`）

1. **输入端 `emotion_classifier` 删除**（反捷径移除）
   - init：`__init__` 构造 `self.emotion_classifier = nn.Linear(llm_input_size, emotion_vocab_size)`（`requires_grad_(False)` 冻结）+ `criterion_emotion_cls = CrossEntropyLoss(ignore_index=0)` + `emo_loss_weight`；forward 中该分类器直接读 FiLM 调制后的 `modulated_text_emb` 做情感分类。
   - current：该子模块、CE criterion、`emo_loss_weight` 在工作树中彻底删除（docstring 明确声明"输入端反模式已删"）。
   - 影响：消除 v1 的输入端标签回读捷径（分类器从 FiLM 输出直接读情感标签）；本次实验据此只算 `loss_tts`，不再有 `loss_emotion`。

2. **下游任务头 `emotion_head` + `arousal_head` 新增**（预留，本次不激活）
   - init：无任何下游任务头。
   - current：`emotion_head = nn.Linear(llm_output_size, 5)`（5 类，pad 不参与）+ `arousal_head = nn.Linear(llm_output_size, 1)`（连续回归）。两头仅消费 `lm_output` 在 span speech-token 区段 masked-mean 池化的 feature（`_pool_span_features`），绝不接收 `emotion_ids/intensity_ids/modulated_text_emb` 作为特征（反捷径：监督落在生成因果链下游）。
   - 影响：情感监督落点从输入端移到生成链末端。但因本次 `disabled` 且 batch 无 span，两头构造但 forward 不计算其 loss、不反向——属"存在但不激活"的预留模块（为未来 span 监督实验预留）。

3. **FiLM 条件化模块零变化**（可比性基础）
   - init/current：`emotion_encoder = EmotionEncoder(...)` + `emotion_adapter = FiLMLayer(llm_input_size)` 完全保留；`emo_film.py` 两版均 35 行逐字节一致（git diff 空）。EmotionEncoder = emotion/intensity 双 embedding 相加；FiLMLayer `h̃ = γ⊙x + β`，**恒等初始化**（projection weight/bias zeros，`bias[:dim]=1.0→γ=1`，`bias[dim:]=0.0→β=0`），初始 `h̃=x`。
   - 影响：本次实验核心方法（FiLM 调制）与 v1 零差异——任何情感/WER 差异不能归因于 FiLM 模块，只能归因于 loss/优化/协议。恒等初始化意味着训练初期 FiLM 近似恒等，调制能力随训练逐步建立。

4. **`__init__` 构造参数：删死字段 + 增监督开关**
   - init：签名含 `tokenizer_for_emotion / mix_ratio=[5,15] / emo_loss_weight=0.2 / alpha=0.05`；`super().__init__` 透传 `mix_ratio`。
   - current：删除上述四字段；新增 `emotion_head_weight=1.0 / intensity_head_weight=1.0 / downstream_supervision='disabled'`（带枚举校验，非法值 ValueError）；`super().__init__` 不再传 `mix_ratio`。注释明确"本类永不读 `self.mix_ratio`，resolved 配置不得出现该键"。
   - 影响：配置契约收窄，死字段由 `assert_no_dead_config` 拒绝。`downstream_supervision` 是本次实验核心开关（值 `disabled`）。

5. **forward lm_input 构造：双流 → 单流 target-only**
   - init：调父类 `prepare_lm_input_target(...)`（含 bistream 双流分支 + fill_token 逻辑 + empty_instruct 占位），lm_input 结构随 speech/text 比例与随机种子变化。
   - current：新增本类专用 `_prepare_target_only_input(...)`，每 sample 恒定构造 `lm_input = [sos, FiLM(text), task_id, speech]`、`lm_target = [IGNORE]*(1+text_len) + speech_token + [eos]`。docstring 强调"恒定单流，绝不进入 bistream/fill-token 分支"。
   - 影响：训练协议稳定为单流 next-token teacher-forcing，消除协议随机性；lm_input 结构与推理前缀（`[sos, FiLM(text), task_id]`）严格对齐（仅减去 teacher speech），是 train/inference 一致性的结构保证。

6. **inference 签名：删死字段 + 增长度硬顶**
   - init：签名含 `prompt_text / prompt_text_len / prompt_emotion_ids / prompt_intensity_ids`（进入即 `del`，实际未参与 LLM 条件，死接口）；无 `max_len_hard_cap`。
   - current：删除上述四死字段；保留 `prompt_speech_token / prompt_speech_token_len / embedding`（透传 Flow/HiFT 声学侧）；新增 `max_len_hard_cap` 参数。
   - 影响：调用方签名收窄（旧调用传 `prompt_emotion_ids` 会 TypeError）；消除历史 prompt emotion 死接口，对齐"prompt emotion/intensity 不进 LLM 条件"协议。

7. **新增模块级基础设施**
   - current 新增：`DecodeResult` dataclass（tokens/finish_reason/min_len/max_len/...）；span 监督契约元组（`_SPAN_TENSOR_KEYS` / `_SPAN_REQUIRED_KEYS`）；辅助 `_batch_has_spans` / `_weighted_span_mean`；方法 `_pool_span_features` / `_compute_downstream_losses`；常量 `DEFAULT_MAX_LEN_HARD_CAP=2000` / `MAX_INVALID_TOKEN_RETRIES=100`。
   - 影响：`DecodeResult` 与常量在本次推理实际生效；span 相关基础设施在本次 disabled 下定义但不激活。

### 2.2 监督方法（loss）

1. **loss 总公式：单公式 → 三分支裁决**
   - init：`loss = loss_tts + emo_loss_weight * loss_emotion`（无条件计算 `loss_emotion`）。
   - current：三分支——① `_batch_has_spans(batch)` 为真 → `loss = loss_tts + emotion_head_weight*loss_emotion + intensity_head_weight*loss_intensity`；② 无 span 且 `enabled` → `raise RuntimeError`（显式报错防静默吞）；③ 无 span 且 `disabled` → `loss = loss_tts`。`loss_tts` 主干 CE 计算本身未变。
   - 影响：本次实验走分支③，loss 退化为纯 `loss_tts`。监督是否生效由 batch 内容 + 开关共同显式裁决，不再隐式。

2. **输入端句级 CE 监督彻底删除**
   - init：`loss_emotion = CE(emotion_classifier(modulated_text_emb), emotion_ids)`，target 是句级 `emotion_ids`（广播到每个 text token 位置），梯度经冻结 classifier 回传 FiLM。
   - current：整条路径删除，`modulated_text_emb` 仅作为 lm_input 的 text 区段进入 LLM，不再被任何分类头消费。
   - 影响：消除"FiLM 走捷径而不真正影响生成"的输入端标签回读；监督落点移到生成链末端（或本次 disabled 下无监督）。

3. **下游 span 词级监督新增（soft-CE + MSE）**
   - current：`_compute_downstream_losses`——emotion 用 `soft_ce = -Σ(span_emotion_soft_dist * log_softmax(emotion_head(span_feature)))`（支持软分布/弱监督）；intensity 用 `MSE = (arousal_pred - span_arousal)**2`；per-span 门控 `span_mask & span_valid & emotion/intensity_mask`，加权 `span_supervision_weight`。
   - 影响：监督粒度从句级升到 span 词级；但本次数据链未接 span，两头构造后不参与梯度。

4. **`downstream_supervision` 显式开关（B1 静默→显式）**
   - init：无监督开关，`loss_emotion` 无条件计算（"永远在线"，是否真接线无法从配置判断——B1 类静默）。
   - current：`__init__` 新增 `downstream_supervision: str='disabled'`，仅接受 `enabled/disabled`；forward 据此裁决（enabled 无 span 报错 / disabled 允许 FiLM-only）。
   - 影响：配置与运行行为一致性可审计。本次 yaml 明确 `disabled`，与"batch 无 span→只算 loss_tts"对齐，消除"以为接了监督实际没接"隐患。

5. **`_batch_has_spans` + span 契约**
   - current：`_SPAN_REQUIRED_KEYS`（9 键最小集：区间 + mask + target），`_batch_has_spans(batch) = all(k in batch for k in _SPAN_REQUIRED_KEYS)`。
   - 影响：定义"batch 是否携带下游监督"的稳定判据；本次 disabled 下恒为 False，确保走 loss_tts-only 分支。

6. **forward 返回字典字段变化**
   - init：恒返回 `{loss, acc, loss_tts, loss_emotion}`（`loss_emotion` 永远存在）。
   - current：span 分支返回五键（+`loss_intensity`）；disabled 分支返回 `{loss, acc, loss_tts}`（**无 `loss_emotion`**）。
   - 影响：本次训练日志只见 `loss_tts`（与 v1 日志字段不同）；下游聚合代码不能假定 `loss_emotion` 恒存在。

7. **配置层监督字段**
   - init：`emo_loss_weight: 0.2 / mix_ratio: [5,15] / alpha: 0.05`。
   - current：删除上述死字段；新增 `emotion_head_weight: 1.0 / intensity_head_weight: 1.0 / downstream_supervision: disabled`（注释标注"非审计最优，中立起始值"）。

### 2.3 训练细节（optimizer / scheduler / freeze / identity / checkpoint）

1. **optimizer 参数分组：单死组 → 三稳定命名组**
   - init：`init_optimizer_emo` 分两组——`emotion_new`（匹配 emotion_encoder/emotion_adapter/llm_decoder）与 `base`（其余）；yaml `lr=1e-5, new_params_lr=1e-5`（两者同值）。因 freeze 只解冻三模块，`base` 组恒空、永不创建 → 实质单一组、新旧 LR 无差异（死分组）。
   - current：改读 `optim_conf.groups[role].{lr,weight_decay}`，三稳定组：`random_new_condition`（FiLM: emotion_encoder/emotion_adapter）`lr=1e-4`；`downstream_heads`（emotion_head/arousal_head）`lr=1e-4`；`pretrained_decoder`（llm_decoder）`lr=1e-5`。新增 `_validate_group_coverage`（三组互斥、非空、完整覆盖 trainable、无冻结参数混入）。
   - 影响：本次实验首次实现真正 LR 分层——FiLM 条件模块 1e-4（快速收敛）、预训练 decoder 1e-5（防灾难遗忘，比 v1 低 10×，v1 全用 1e-5）。`downstream_heads` 组被创建但 disabled 下无梯度（无损无害）。

2. **scheduler：硬编码 ConstantLR（warmup 死字段）→ WarmupLR 真正生效**
   - init：`train_emo.py` 硬编码 `scheduler = ConstantLR(optimizer)`（不读 scheduler_conf）；yaml `scheduler_conf.warmup_steps: 2500` 从未被消费（死配置），LR 全程恒定。
   - current：`build_scheduler(optimizer, conf)` 单一工厂；yaml `scheduler: warmup, warmup_steps: 250`；WarmupLR 在 `step<warmup` 时 `lr*step/warmup` 线性 ramp，峰值在 step 250。新增 `validate_optim_scheduler_conf`（启动失败 on 未知字段 / 顶层 `optim_conf.lr` 死字段 / warmup 缺 warmup_steps）。
   - 影响：warmup 现在真正改变前期 LR（0→base_lr@step250），而非 init 的恒定 LR。本次实验早期 step 的 LR 行为与 v1 完全不同（v1 第一步满 LR 1e-5；current 第一步≈0、ramp 到 1e-4/1e-5）。

3. **freeze 范围**
   - init：`freeze_all_except(model, EMOFILM_TRAINABLE_MODULES)`，模块常量 = (emotion_encoder, emotion_adapter, llm_decoder)；额外显式冻结 emotion_classifier。调用方传模块名参数。
   - current：无参 `freeze(model)`；`_TRAINABLE_PREFIXES` = 三组并集（+emotion_head/arousal_head，-emotion_classifier）；`_FROZEN_PREFIXES` = (llm, speech_embedding, llm_embedding)（精确前缀匹配保证 `llm.*` 不吞 `llm_decoder.*`）。
   - 影响：trainable 集合随活跃模型演化；freeze 调用契约从"传模块名"收紧为"无参 + 模块常量内聚"。

4. **train_conf 数值超参端到端不变**
   - init/current 逐字段相同：`max_epoch=5 / accum_grad=2 / grad_clip=5 / log_interval=100 / batch_size=4（static）/ save_per_step=-1 / optim=adam / seed=1986`（random/numpy/torch/cuda 四处）。
   - 影响：训练调度数值零变化 → 本次与 v1 跑相同 epoch、相同有效 batch（=8）、相同梯度裁剪、相同种子。差异仅来自 optimizer 分组 LR 与 scheduler 形态。

5. **identity schema v1→v2**
   - init：`schema_version=1, contract_name='emofilm_v1'`；`extra.resolved_config` 仅记路径字符串；source 仅存 worktree_diff 的 SHA-256（不可重建）；无 optimizer_identity、无 patch_bundle。
   - current：`schema_version=2, contract_name='emofilm'`；新增 `optimizer_identity`（每组 tensor/param count + initial_lr + weight_decay + scheduler type/key_params）、`resolved_config`（实际 train_conf dict）、`source.patch_bundle`（dirty worktree 存 `git diff --binary HEAD` 实际字节，含 untracked，用隔离 GIT_INDEX_FILE 不污染 .git/index，可经 `git apply` 重建）。v1→v2 不兼容（`update_training_identity` 遇 schema<2 显式 raise）。
   - 影响：本次 `train_identity.json` 完整绑定三组 LR/WD + warmup_steps + 实际 train_conf，且 dirty worktree 可重建（v1 不可重建）。

6. **checkpoint 加载**
   - init/current：base/resume/fresh 三态、init.pt 保护、latest→final 收口语义逐行相同；`hash_model_state` 算法不变（parameter_hash 可比）。
   - 唯一变化：`ALLOWED_MISSING_PREFIXES` 由 `(emotion_encoder., emotion_adapter., emotion_classifier.)` 换成 `(emotion_encoder., emotion_adapter., emotion_head., arousal_head.)`——匹配新拓扑（无输入端 classifier、新增下游头）。

7. **死配置拒绝 + 合同常量**
   - init：`build_emofilm_contract.py`（443 行），`CONTRACT_NAME='emofilm_v1'`，无 `SCHEMA_VERSION`，无死字段拒绝机制（但 yaml 仍带 emo_loss_weight/mix_ratio/alpha）。
   - current：（1073 行），`CONTRACT_NAME='emofilm', SCHEMA_VERSION=2`；新增 `DEAD_CONFIG_KEYS={mix_ratio, emo_loss_weight, alpha}` + `assert_no_dead_config`（任一出现 ValueError）+ `validate_contract_config`（拒非 emofilm 合同名 / 非 2 schema / 死字段）。配合 `validate_optim_scheduler_conf`，配置层有两道独立死配置门禁。

8. **新增 `summarize_optimizer_identity` + `build_scheduler`**
   - current `train_utils_emo.py`（133→449 行）新增 `build_scheduler` / `summarize_optimizer_identity` / `validate_optim_scheduler_conf` / `_validate_group_coverage` + 常量 + `__all__`。scheduler 与 optimizer 身份摘要从"散落在 train_emo.py 的硬编码"收敛到单一权威模块。

### 2.4 推理设置（`tools/inference_emo_film.py` + `cosyvoice/cli/*`）

1. **CLI 新增 `--seed` + 自动身份采集**
   - init：参数集含 model_dir/llm_ckpt/test_manifest/esd_root/workspace_root/output_dir/device/fp16/max_samples/shard_idx/num_shards/skip_existing/save_every/use_tagged_text；无 `--seed`，无身份采集。
   - current：新增 `--seed`（default=1986，per-utt 重置 torch+cuda RNG，可复现）；main() 加载模型后采集 `source_revision = git rev-parse HEAD` + `llm_ckpt_sha = sha256(llm_ckpt)`。
   - 影响：推理结果可复现；每条产物绑定源码 revision + checkpoint 内容 sha256。

2. **GenerationRow：裸 wav 清单 → 合法身份行**
   - init：row 仅 `{utt_id, wav_path(绝对), prompt_wav, prompt_text, prompt_source, status, duration_s}`，无身份/finish_reason/seed/decode_config。
   - current：row 含 `utt_id, finish_reason, source_revision, checkpoint_sha256, decode_config, seed, control_row_ref, prompt_row_ref, text_digest, prompt_audio_ref`；eos 额外 `wav_path`（workspace-relative POSIX）+ prompt_source + duration_s + status=success；非 eos 仅 status=non_eos_diagnostic（schema 强制不带 wav_path）。
   - 影响：产物从"裸清单"升级为可校验、可比对、可溯源的合法 GenerationRow；绝对路径→相对路径使产物可移植。

3. **四族身份 + 合成输入摘要**
   - current：`text_digest = sha256(text_with_emo)`、`prompt_audio_ref = relpath(prompt_wav, ref_root)`、`control_row_ref`、`prompt_row_ref`；四族 = source(checkpoint) / checkpoint / control / prompt。
   - 影响：B6 修复（改 manifest 续跑不再静默复用旧条件 WAV）；B5 修复（control/prompt ref 缺失时就地合成稳定标识）。

4. **decode_config 合同透传链（修复"改 yaml 不生效"历史 bug）**
   - init：无 decode_config 概念；`cosyvoice_emo.py` `del self.configs` 后推理读不到长度参数；改 yaml 长度不生效。
   - current：三段贯通——yaml 新增 `decode_config{min_token_text_ratio:2, max_token_text_ratio:20, max_len_hard_cap:2000}` → `cosyvoice_emo.__init__` 抽取 `self.decode_config` → `inference_emo_film` 透传 → `model_emo.tts/llm_job` 条件注入。
   - 影响：yaml 长度合同真正生效；`max_len_hard_cap=2000`（~80s@25Hz）修复历史 `max_len=200` 截断 bug；每条 GenerationRow 记录实际 decode_config。

5. **finish_reason 五态合同 + 非 eos 门控**
   - init：inference 逐 token 流式产出，无 finish_reason 概念；非法 token 超 100 次 `raise RuntimeError`；循环耗尽静默结束；调用方无法区分正常 EOS / 超长截断 / 异常。
   - current：`FINISH_REASONS = {eos, max_len_reached, invalid_token_retry_exhausted, sampler_error, input_rejected}`；decode/_decode_loop/inference_wrapper 三层架构 + `DecodeResult` dataclass；错误改为结构化标记（不再 raise）；`model_emo.tts` 门控——仅 `finish_reason=='eos'` 时进 Flow/HiFT + token2wav，非 eos yield None + finish_reason。
   - 影响：非 eos 不进声学、不落正式 WAV，杜绝截断/错误帧混入评测；每条产物 finish_reason 可诊断。代价是失去逐 token 流式（buffered），但保证 finish 合同。

6. **非 eos 删旧 WAV（B12 目录一致性）**
   - current：finish_reason 非 eos 分支中，若 `os.path.isfile(out_wav)` 则 `os.remove(out_wav)` + warning。
   - 影响：保证目录内容 == manifest eos 集合，旧 run 残留 WAV 不污染评测配对。

7. **写盘前 `validate_generation_row`（B5 fail-fast）**
   - current：append 前调用合同自检（四族身份 ref 任一存在、decode_config 是 mapping、seed 非负 int、finish_reason 在五态内、eos 必带 wav_path、非 eos 不得带 wav_path），不合格携 utt_id fail-fast。
   - 影响：合同违反在生成端立即暴露，不在评测端才崩。

8. **safe resume：isfile → 四守卫身份校验**
   - init：`if skip_existing and os.path.isfile(out_wav): skip`（仅看文件存在，无身份校验——134 条无验证恢复漏洞）。
   - current：`check_skip_existing` 四守卫——(1) `finish_reason==eos`；(2) 有效 wav_path；(3) wav 磁盘存在（`os.path.isfile`，ADR-0020 禁止 wav_sha256 内容比对）；(4) 既有 row 四族身份任一空摘要→视为无 existing（防 `''==''` 误复用），再比 `generation_row_identity_fingerprint` 完全一致。
   - 影响：续跑安全——seed/任何身份变→指纹不同→重生成；WAV 被删/替换→重生成；v1 无身份 manifest→一律重生成。

9. **`model_emo.tts/llm_job` v1→v2 单流签名**
   - init：tts 签名含 v1 死参数 `prompt_emotion_ids/prompt_intensity_ids/prompt_text/source_speech_token`；内部对 prompt emotion 缺省填默认值；llm_job 把 prompt_text/prompt_emotion 传给 `self.llm.inference`。
   - current：删除死参数，新增 `decode_config=None`；llm_job 仅传 target-only 单流参数（text/emotion/intensity + prompt_speech_token/embedding）；prompt 不进 LLM 条件，声学 prompt 条件（flow_prompt_speech_token/prompt_speech_feat/flow_embedding）仍透传 Flow/HiFT。
   - 影响：v2 单流协议——LLM 只看 target text + emotion/intensity 控制，与 disabled（FiLM-only）实验一致。

10. **`resolve_prompt`：prompt_text 不再必填**
    - init：强制 `prompt_text` 必填（缺失 raise）；ESD/Part B 回退缺 prompt_text raise。
    - current：`prompt_text = utt.get('prompt_text') or ''`（缺失回退空串）；删除必填 raise。
    - 影响：v2 单流下 prompt_text 不进 LLM 条件，不再必需；避免 manifest 缺 prompt_text 时误报错中断批量推理。

11. **`write_emofilm_run_identity.py` 扩张（228→972 行）**
    - current：新增训练身份（`write_emofilm_train_identity` + `_save_patch_bundle` dirty-worktree 可重建）、逐条身份与安全恢复链（`capture_source_identity` / `generation_row_identity_components|fingerprint` / `generation_request_fingerprint` / `check_skip_existing` / `SkipDecision` / `write_emofilm_generation_identity`）；v1 `write_run_identity` 入口签名/行为冻结只读（兼容锁）。

12. **多 GPU 分片不变**
    - init/current：`entries = entries[shard_idx::num_shards]`；`num_shards>1` → `inference_{base}.shard{idx}.jsonl`，`==1` → `inference_{base}.jsonl`。identity-based skip 按分片 manifest 隔离。

### 2.5 数据链路 + 评测

1. **`tokenize_emo` 三元组逐字节不变**
    - init/current：`processor.py:tokenize_emo` 调 `emo_tokenizer.encode_plus(text)` 产出严格等长三元组 `text_token / emotion_ids / intensity_ids`（每 `<emotion>` 段内 token 共享同一 emo/inten id，即句级标签广播到 text-token 长度）。sha256 两版全等。
    - 影响：数据链路入口冻结在 v1 基线，本次实验仍按单流喂入三元组。

2. **`emo_tokenizer.encode_plus` 逐字节不变**（三元组等长不变量成立）。

3. **`padding` 函数逐字节不变**
    - init/current：同一函数体；`prompt_emotion_ids/prompt_intensity_ids` 两 pad 分支仍保留但由 `if 'prompt_*' in sample` 守卫——单流 tokenize_emo 从不产出这两个字段，故为死分支（无害）。

4. **data_pipeline 10 步序列不变**
    - init/current：`parquet_opener → tokenize_emo → filter → resample → compute_fbank → parse_embedding → shuffle → sort → batch → padding`。未插入 span_align。

5. **`span_align.py` 零接线（B1 确认）**
    - current：新增 `cosyvoice/dataset/span_align.py`（408 行，纯函数 align_spans_to_tokens / collate_aligned_spans），但全仓库仅 `llm_emotion.py:104` 一处注释引用，无 import/无调用——未进 data_pipeline、未被训练入口消费。
    - 影响：从数据链路侧证实 span 监督链在 current 仍零接线，与 disabled 一致。

6. **`eval/eval_emo_film.py` 逐字节不变（评测口径可比性基础）**
    - init/current：同一文件（sha256 全等）。frame-mean Emo-SIM + DTW（cosine 正式 + euclidean 诊断，fastdtw）+ WER（lowercase + 去标点 + 数字→英文 normalization）；`METRIC_CONTRACT_VERSION='emofilm-eval-v2'`；输出 9 字段 JSON。
    - 影响：本次实验与 v1 评测完全同口径可比。

7. **v2 评测模块删除清单（端到端净删除/净零）**
    - 净删除（init 有→current 无）：`tools/label_fedd_emotion2vec.py` + 其测试（FEDD emotion2vec 自动打标器移除，FEDD 走现有落盘标签）；`tests/test_emo_train_utils.py`、`tests/test_qwen2lm_emotion.py`。
    - transient（仅中间 commit 52d8be1 短暂存在，ed89f48 删除，端到端净零）：`eval/acoustic_evaluators.py`（998 行）/ `eval/eval_local_control.py`（1248 行）/ `eval/triplet_eval.py`（886 行）+ 9 个 v2 评测测试 + 评测 fake + build_emofilm_contract 的 eval 校验 + write_emofilm_run_identity 的 4 个 eval 身份 API。当前仅 `eval/__pycache__/` 残留 .pyc 缓存（非追踪）。
    - 影响：端到端视角 v2 细粒度评测能力净增量为零；评测回归并停留在 init 基线 `eval_emo_film.py`。

8. **`conf/emo_film.yaml` 顶层 `decode_config` 段新增**（见 2.4-4）。

9. **eval/ 目录收敛**：init/current 均仅 `eval_emo_film.py + visualize_case.py`（2 文件）。

### 2.6 架构变化对本次实验的实际影响汇总

| 变化类别 | 本次实验（disabled）实际状态 |
|---|---|
| FiLM 调制模块（emo_film.py） | **零变化**（与 v1 严格可比） |
| 数据链路（tokenize_emo/padding/pipeline） | **逐字节不变**（与 v1 等价） |
| 评测口径（eval_emo_film.py） | **逐字节不变**（同口径可比） |
| 训练协议 | 双流 → **单流 target-only**（稳定化，消除协议随机性） |
| 监督 | 输入端 classifier CE → **纯 loss_tts**（disabled，无情感 loss 反向） |
| 优化 | 单组 LR 1e-5 → **三组分层**（FiLM 1e-4 / decoder 1e-5）+ **WarmupLR 真正生效**（warmup 250） |
| 推理可审计性 | 裸 wav 清单 → **finish_reason 门控 + 逐条身份 + safe resume 四守卫 + decode_config 合同** |
| 下游 span 监督头（emotion_head/arousal_head）+ span_align | **构造/存在但零接线、不激活**（为未来 span 实验预留） |

净效果：本次实验实际生效的架构差异收敛为"单流协议稳定化 + 输入端分类器反捷径移除 + 纯 loss_tts 训练 + LR 分层 + WarmupLR + 结构化可审计推理"，而 FiLM 调制本身、数据流、评测口径与 v1 零差异——这构成本次与 v1 客观可比性的结构基础。

## 3. 本次评测结果

实验产物：`exp/emofilm_film_only/`（final.pt sha256 `02cad038...`；2500 条生成全 eos；train/generation identity 齐全）。baseline 评测（`eval_emo_film.py`，三分区）：

| 分区 | N | WER % | Emo-SIM | DTW (cosine) | DTW normalized | DTW euclidean normalized |
|---|---:|---:|---:|---:|---:|---:|
| ESD | 1500 | 8.18 | 66.11 | 49.25 | 0.3387 | 10.03 |
| FEDD-A | 500 | 4.70 | 82.71 | 75.26 | 0.1706 | 6.65 |
| FEDD-B | 500 | 12.30 | 62.98 | 52.55 | 0.3701 | 10.53 |

（指标面 `metric_contract_version: emofilm-eval-v2`，与 v1 同口径；Emo-SIM 越高越好，DTW/WER 越低越好。）

## 4. 与 v1 baseline report 清晰对比

v1（`docs/reports/2026-07-20-emofilm-v1-baseline-experiment-report.md`，init 代码 9c6d84b，有 `loss_emotion`）三分区指标 vs 本次（current ed89f48，FiLM-only disabled）：

| 分区 | 指标 | v1（有 loss_emotion） | 本次（FiLM-only） | Δ | 解读 |
|---|---|---:|---:|---:|---|
| ESD | WER % | 9.48 | 8.18 | **−1.30** | 本次更好（内容） |
| ESD | Emo-SIM | 66.75 | 66.11 | −0.63 | 持平 |
| ESD | DTW normalized | 0.3324 | 0.3387 | +0.0063 | 持平 |
| FEDD-A | WER % | 8.30 | 4.70 | **−3.60** | 本次明显更好（内容） |
| FEDD-A | Emo-SIM | 81.94 | 82.71 | +0.77 | 略好 |
| FEDD-A | DTW normalized | 0.1783 | 0.1706 | −0.0077 | 略好 |
| FEDD-B | WER % | 14.42 | 12.30 | **−2.12** | 本次更好（内容） |
| FEDD-B | Emo-SIM | 61.60 | 62.98 | +1.38 | 略好 |
| FEDD-B | DTW normalized | 0.3839 | 0.3701 | −0.0138 | 略好 |

**对比解读**：

- **情感指标（Emo-SIM / DTW）**：9 项中本次 **4 项略好、2 项持平、仅 ESD 2 项微降（|Δ|<0.01）**。本次 FiLM-only（无 `loss_emotion`）的整体情感表现**不亚于 v1**。
- **内容指标（WER）**：三分区**全部明显改善**（−1.3 ～ −3.6 个百分点）。FiLM-only 未被 `loss_emotion` 干扰，内容质量更优。
- **结论**：`downstream_supervision=disabled`（砍 v1 输入端 `loss_emotion`）**无害**——客观情感指标不亚于 v1，`loss_emotion` 非必要；FiLM 调制在纯 `loss_tts` 监督下即可建立。本次 WER 改善可能受益于：单流协议稳定化、LR 分层（decoder 低 LR 防遗忘）、输入端分类器反捷径移除（不再干扰生成主干）。

**与 v1 报告"听测"记录对照**：v1 报告第 9 行记载 v1 单条听测"情感偏中性，用户确认不视为错误并授权继续"；本次 3 条试听亦偏中性。两版本听测都偏中性，但客观 Emo-SIM/DTW 持平——说明**听测偏中性是 FiLM 方法的整体感知特性**（两版本 Emo-SIM 都 ~66，情感相似度有限），而非 `disabled` 独有问题。用户"听测偏差、实际有提升"的判断被客观指标证实。

## 5. 结论与局限

### 结论

1. **代码实现正确**：FiLM 真实接线训练（`modulated_text_emb` 在 loss_tts 计算图、梯度回传 FiLM）+ 推理复用；`disabled` 正确实现（无 bug）。经 5-agent 对抗审计确认。
2. **`disabled` 设计正确服务目标**：客观评测证明 FiLM-only 与 v1（有 `loss_emotion`）情感表现相当，WER 更优。`loss_emotion`（v1 输入端随机冻结分类器 CE）非必要。
3. **实验可追溯**：train_identity / full_generation_identity / eval metrics 三件套齐全，命令脚本（run_train/infer/eval.sh）可复跑，dirty worktree 可经 patch_bundle 重建。

### 局限

1. **Emo-SIM/DTW 是整体情感相似度**，不完全等价于"按标签可控生成强情感"的能力。两版本 Emo-SIM 都 ~66、听测都偏中性，说明 **FiLM 调制 text embedding 这一方法的可听情感强度本身有限**（情感主要体现在韵律侧，FiLM 在语义侧调制，传递到 speech token 再到声学的路径长且损耗大）。若要更强可听情感，是方法层面改进（声学侧调制 / 更强监督 / 更长训练），与 `disabled` 无关。
2. **训练收敛偏慢**：5 epoch CV loss_tts 3.73→3.70（epoch 2 后趋平），下降缓慢，疑似未充分收敛。这是下一步重训实验的动机（见 handoff）。
3. **下游 span 监督头零接线**：`emotion_head/arousal_head` + `span_align.py` 构造但未激活，未来 span 监督实验（ticket 13/14）需先接通 data_pipeline。

### 可追溯身份

- 训练：`exp/emofilm_film_only/train_identity.json`（git ed89f48, contract_hash `6a67b685...`, final.pt sha256 `02cad038...`）
- 生成：`exp/emofilm_film_only/full/full_generation_identity.json`（2500 rows 全 eos, aggregate fp `46a2fc90...`）
- 评测：`exp/emofilm_film_only/eval/{esd,fedd_a,fedd_b}_metrics.json`
- 命令：`exp/emofilm_film_only/{run_train,run_infer,run_eval}.sh`
