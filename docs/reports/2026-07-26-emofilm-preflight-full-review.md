# EmoFiLM 实验前全量四轴审查报告（2026-07-26）

> 目的：13（正式训练）/ 14（固定生成 + 局部评测 + 报告）真实实验前的最后一次全量代码审查 + 训练/推理/评测逻辑校验 + 数据处理逻辑与流向真实性审查。
> 方法：按 handoff（`/tmp/emofilm-preflight-review-handoff-2026-07-26.md` §4.2）基架执行——五链路审查子代理（训练/推理/评测/数据身份/真实性横切专项）+ 13/14 票据预检并行；每条 blocking 候选发现经两个独立对抗核查者（存在性反驳 lens + 实验影响面 lens）双重验证，should-fix 经一个；completeness critic 识别 6 个覆盖缺口后第二轮定向补审。共 118 个子代理、约 1884 次工具调用。全程只读，未修改任何仓库文件。
> 对照合同：`docs/contracts/emofilm_v2_schema.md`、`docs/contracts/emofilm_v2_evaluators.md`、ADR-0019/0020、`.scratch/emofilm-mainline-remediation/spec.md` §9。
> 原始发现全文（82 条含逐条核查 verdict）：workflow journal `~/.claude/projects/-home-hanlvyuan-LLM-Audio/f769e6a6-bb98-465c-a688-51bf511610ca/subagents/workflows/wf_cf56b521-308/journal.jsonl`。

---

## 0. 总裁决：**实验未就绪（NOT READY）**

82 条发现（对抗核查零 REFUTED），合并跨链重复后裁决为 **13 个独立 blocking 根因**、24 项 should-fix、9 项 deferred。其中最根本的一条（B1）意味着：**若现在直接跑 13，v2 的核心科学修复（下游 speech-token 监督）将静默不生效**——训练全程绿灯、loss 正常下降，但 emotion/arousal 头零梯度，14 评测的是一个从未受过 span 监督的模型。

> **增补注记**：同日另一次独立审查运行（115 子代理、83 条发现、同方法学）交叉验证了本报告全部 blocking，并补充 **2 条本文缺失的 blocking（B14 FEDD-B 测床污染、B15 eval 运行身份零接线）+ 5 条 should-fix（S25-S29）+ 1 条 deferred（D10）**，详见文末《增补：第二次独立审查运行的交叉验证》。合并后总计 **15 blocking / 29 should-fix / 10 deferred**。

**方法论教训（handoff §4.1 预警被证实）**：全套 pytest 651 passed 与上述结论并存——测试全部在手工注入的 fixture batch 上验证了 span 监督的*逻辑*，但真实数据路径（tagged.jsonl → parquet → batch）从未接通。"数据不可达但测试还绿"正是最危险的信号。

**正面结论**：三轮修复（ADR-0019 合同、扁平化、remediation 12 项）本身的实现质量高——协议不变量、finish_reason 门控、seed 全链、hard-fail 语义、exact/approximate 分离、哈希边界等 60+ 个关键合同点全部逐行核查通过（见 §5）。问题集中在**链路接缝**（模块单体正确但没接上）与**数据/资产门禁**（ADR-0020 §6 本就延后的部分暴露出比预期更深的缺口）。

---

## 1. 四轴总体结论

| 轴               | 结论                                                                                                                                                                                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **可行性** | ❌ 端到端不可行。13 可以启动但训练错的东西（B1）；14 的生成会批量产出非法 GenerationRow 在评测侧才崩（B5），强度轴无入口可跑（B9）。                                                                                                                                                        |
| **正确性** | ✅ 单模块层面高度忠实于合同：schema §1/§2 校验器逐字段一致，remediation 12 项全部确认落地（见 §5）。❌ 但若干校验器**没有被生产入口调用**（validate_span、validate_generation_row 在产出端缺位）。                                                                                 |
| **合理性** | ⚠️ 架构设计（FiLM 逐 token 注入、下游监督、triplet 设计、tier 分离）对目标 make sense；但**数据层**存在两个科学性问题：强度三档在训练数据上不可分（B4，low 档实测 ≈1%），词级情感伪标签与句级一致率仅 24%（S19，5 类随机 =20%）。不修/不显式降级口径，强度与词级情感主张无法成立。 |
| **真实性** | ❌ 发现 8 处静默降级/假数据路径（B2/B3/B6/B7/B10/B11/B12/B13），其中 B2+B3 是 StubPredictor 事故的同构复现且**实跑复现**（Stub 占位数据通过 provenance 审计 PASS）。                                                                                                                  |

---

## 2. Blocking 根因（13 项，实验前必须修或经显式决策降级）

每条均经双 lens 对抗核查（存在性 + 影响面），标注跨链独立印证次数。

### B1 — span 监督零接线：v2 核心修复未接入训练数据管线【5 链独立印证】

- **轴**：authenticity/feasibility；**位置**：`conf/emo_film.yaml:217-228`（data_pipeline）、`cosyvoice/llm/llm_emotion.py:364`（`_batch_has_spans` 门）、`cosyvoice/dataset/processor.py:399-462`（padding 无 span_* 键）、`tools/jsonl_to_cosyvoice_src.py:35-62`
- **证据**：`llm_emotion.py` 进入 head loss 的唯一分支要求 batch 携带 9 个 `span_*` 键；而 data_pipeline 末端 `padding()` 产出的 batch 无任何 span 键。全仓 grep 证实 `span_align.py` 的 `align_spans_to_tokens`/`collate_aligned_spans` 生产链**零调用方**（仅测试手工注入假张量）。断链共六层：jsonl→src 转换丢弃 span 字段、splits builder 对 per-span 行直接 raise、parquet 无 spans 列、processor 无解析、DDP `find_unused_parameters=True` 使零梯度头不报错、无任何"期望有监督却没有"的断言。
- **实验影响**：13 训练静默退化为 loss_tts-only，emotion/arousal 头以随机初始化进入 final.pt；issue 13 DoD（分别记录三类 loss、监督覆盖率）不可能满足；论文核心主张失去训练侧支撑，且全程无一处报错。
- **修法**：贯通五层接线（src 侧按 utt_id 聚合 span sidecar → splits 聚合 → parquet spans 列 → processor 调 `align_spans_to_tokens`(25Hz)+`collate_aligned_spans` → batch）；同时在 `llm_emotion`/训练入口加显式 `expect_spans` 开关：开启时 batch 无 span 即 raise、首个 epoch 内监督覆盖率为零即 raise；若有意跑 FiLM-only 基线必须在 resolved config + train_identity 显式写 `downstream_supervision=disabled`。禁止把 `_batch_has_spans` 静默 fallback 当事实开关。

### B2 — 标注器 Flex fallback 默认开启 + provenance 硬编码假合同身份【4 处印证 + 主循环仲裁】

- **轴**：authenticity；**位置**：`tools/generate_tagged_jsonl.py:617`（`allow_flexible_fallback: bool = True`）、`:624-646`、`:790-796`
- **证据**（仲裁记录：核查者对可达性分歧，主循环读码裁定 blocking）：`load_default_predictor` 默认 fallback 开启，合同 strict-load 抛任何异常仅 stderr WARNING 即回退 `FlexWordSequenceModelPredictor`（`load_state_dict(strict=False)`，缺失键保持随机初始化仍继续产标签）；且 `generate_iemocap_v2_spans` 对**任意** predictor 硬编码 `annotator_provenance = {model_class: WordSequenceModel, contract: 768d/5emo/3VAD}`——现存 StubPredictor fixture 行实测就带着这个假合同声明。这是 ticket-02 StubPredictor 事故的同构复现：加载失败→不停→换非合同模型继续产"真"数据，且身份记录撒谎。
- **实验影响**：13 数据重生时若 checkpoint 路径错/被换/维度不符，CLI 不停，静默用形状自适应模型对 20774 句产出全量 spans，provenance 仍声称合同身份→训练真值源不可信且事后不可辨。
- **修法**：正式数据运行反转默认 `allow_flexible_fallback=False`（Flex 显式 opt-in 且每条 span provenance 记 `fallback=True` + 实际 input_dim/reg_dim）；`annotator_provenance` 从 predictor 实例真实属性派生，禁止硬编码；Flex load missing/unexpected 键非空时 raise。

### B3 — provenance 门禁双向失效：放假拒真【4 处印证，实跑复现】

- **轴**：authenticity；**位置**：`tools/audit_label_provenance.py:68-73`（iemocap profile）、`:42-65`（esd profile）
- **证据**：`audit_iemocap_supervision` 唯一检查 `label_source == "word_annotator_pseudo_label"`，而 `generate_tagged_jsonl.py:63` 对一切 predictor（含 Stub/Flex）统一写该常量。**实跑复现**：对现存 18 行 StubPredictor fixture（predictor_class=StubPredictor、checkpoint_sha256=64 个 0）跑审计 → `PASS` exit 0。唯一真实身份标记 `provenance.annotator.predictor_class` 全仓无任何门禁消费。反向：`audit_esd_control` 检查的 `granularity` 字段与 v2 字段名 `supervision_granularity` 不符，对合法 v2 ESD span 行必 FAIL（诱导操作者跳过门禁）。
- **实验影响**：门禁 1 重生若部分失败/残留旧 stub 行/误用 Flex 产物，审计照常 PASS，占位分布带着"审计通过"标签进入正式训练。
- **修法**：iemocap profile 逐行强校验 `predictor_class == "WordSequenceModelPredictor"` 且 `checkpoint_sha256` 非全零 64-hex（可选：等于 author checkpoint 实测 sha `a4b373…`）；esd profile 改按 v2 字段（supervision_granularity/one-hot/intensity_mask=False）校验。

### B4 — arousal 三档阈值与标注器输出分布失配：low 档覆盖实测 ≈1%【核查者实测复现】

- **轴**：soundness；**位置**：`tools/generate_tagged_jsonl.py:59-60,95-104`（写死 `2.5/3.5` 切点）
- **证据**：核查者用 `author_best_model.pth`（strict load 成功的 canonical 路径）对 v1 word_blocks 做两套独立 CPU 抽样推理（合计 ~1050 词块，另一核查者用不同种子独立复现）：arousal 实测范围 2.2-4.4、mean≈3.33、std≈0.35，三档占比 **low=1.06%~1.25% / medium=58%~66% / high=32%~41%**。2.5 切点位于均值下 ~2.4σ。等分三档的实测分位切点应 ≈3.23/3.50。该阈值在任何 doc/ADR/spec 中无出处，2026-07-22 静态审计已点名此限制，v2 原样保留。
- **实验影响**：ESD 69% 训练数据本就 fixed_medium；IEMOCAP 词块中 low 仅约 700-900 个——`intensity_embedding` 的 low=1 嵌入从 <0.5% 训练 token 获得梯度，13 训完接近随机初始化；14 的 `arousal_strict_monotonic = a_low<a_med<a_high` 判定依赖这个几乎未训练的嵌入，**单调性结论无论正负都无法归因于控制设计**。
- **修法（需用户决策，二选一并显式落盘）**：(a) 改为基于标注器输出分布的分位数校准切点（≈3.23/3.50 或对校准集重新拟合），三档覆盖统计写进数据 provenance；(b) 保留写死阈值，则 14 报告必须显式降级强度主张（low 档无训练支持）并记录实测覆盖。禁止不做记录直接重生。
- **正面佐证**：标注器 arousal 方向性信号真实存在（按句级情感分组 ang 3.60 > hap 3.41 > neu 3.32 > sad 3.16），分位校准路线可行。

### B5 — eval manifest 全部缺 control/prompt 身份族 + 生成端不自校验【7 处印证】

- **轴**：feasibility；**位置**：`tools/inference_emo_film.py:291-292`；数据现状 `data/contracts/emofilm_v1/eval/{esd,fedd_a,fedd_b}/manifest.jsonl`
- **证据**：三个 eval manifest 程序化全量扫描 **0/2500 行**含 `control_row_ref`/`prompt_row_ref`；`run_inference` 用 `utt.get()` 直取（恒 None）且全文件从不调用 `validate_generation_row`；评测侧 `eval_local_control._strict_pair:815` 与 `triplet_eval:755` 对称校验必 hard-fail。修复约束已核实：历史 builder 输入（`data/tagged_jsonl` 等）已删除不可重跑；现行 manifest 属 ADR-0020 磁盘冻结；fill 脚本是原地覆写。
- **实验影响**：14 烧完 2500 条 GPU 生成后在评测第一个样本崩溃，全部产物作废。
- **修法**：新增只读派生层（如 `tools/derive_v2_eval_manifests.py`）：读冻结 v1 manifest，为每行附加含版本判别符的 ref（如 `emofilm_v2/eval/fedd_b@v2#<utt_id>`，行内容任何重生必然改变 ref→指纹失配→强制重生成），写到 `data/contracts/emofilm_v2/eval/` 不触碰冻结文件；同时 `run_inference` 写盘前逐条 `validate_generation_row` fail-fast。

### B6 — skip-existing 指纹不含合成内容，真实 manifest 下退化为 run 级常量【CPU 实测复现】

- **轴**：authenticity；**位置**：`tools/write_emofilm_run_identity.py:696-703`（指纹 payload）、`:585`（缺失族归 ""）、`:770-843`（无"缺身份族→拒绝"守卫）
- **证据**：指纹仅含 source/checkpoint/control_ref/prompt_ref/decode_config/seed 六项，不含 tagged_text/plain_text/prompt_wav/use_tagged_text 任何实际合成输入；control/prompt 引用双缺时摘要归 "" 照常相等比较。核查者 CPU 实测：两条不同 tagged_text 的 row 指纹相等；tagged_text 已变更 + WAV 在盘 → `check_skip_existing` 返回 `skip=True`。`inference_emo_film.py:186` docstring"无身份→当作无 existing"在实现中不存在（仲裁：docstring 错误）。
- **实验影响**：14 中途修 manifest（改标签/换 prompt/修 boundary）后带 `--skip_existing` 续跑，旧条件 WAV 被静默复用，局部控制指标被污染且事后不可发现；tagged vs plain 消融同 output_dir 续跑则整组交叉复用，消融差值归零假象。
- **修法**：指纹纳入 `text_digest = sha256(text_with_emo)`（`_pick_text` 之后取值）与 resolved prompt_wav 相对路径（输入数据摘要不在 ADR-0020 禁区——禁区是源码哈希锁与 WAV 产物哈希）；`check_skip_existing` 加显式守卫：existing row 四族任一摘要为 "" → `skip=False, reason="missing identity family"`。B5 的版本化 ref 落地后与本修法互补。

### B7 — source_revision 身份谎报：dirty worktree 下写 v1 基线 commit【4 链印证】

- **轴**：authenticity；**位置**：`tools/inference_emo_film.py:366-368`
- **证据**：`main()` 无条件 `git rev-parse HEAD` 无任何 dirty 检查；当前实测 22 modified/deleted + 30 untracked（v2 实现全部未提交），HEAD=`9c6d84b`（v1 基线）。同仓库为此场景而建的 `capture_source_identity`/`build_source_revision_or_patch`（`write_emofilm_run_identity.py:593-664`）未被推理入口调用。schema §2 明确 source_revision="干净 git revision sha"。
- **实验影响**：每条 GenerationRow 谎称源码=v1 基线而实际跑的是 v2 代码——身份链系统性失真；且指纹 source 分量恒等于 HEAD，改代码/yaml 后 `--skip_existing` 重跑会被误判身份一致而复用旧代码产物。
- **修法**：推理入口改用 `capture_source_identity` + `build_source_revision_or_patch`（clean→source_revision；dirty→patch bundle+sha，schema 已支持）；或最低限度 dirty 即 hard-fail 提示先提交。训练入口已正确走 patch bundle 路径（见 §5），仅推理入口漏接。

### B8 — 门禁 2 确认未满足：独立局部 evaluator 缺位，唯一可跑路径产虚构分数【2 链印证】

- **轴**：soundness/authenticity；**位置**：`eval/eval_local_control.py:1088-1098`、`eval/acoustic_evaluators.py:575-579`
- **证据**：CLI 对非 fake evaluator 显式 `raise RuntimeError("real emotion2vec evaluator gate NOT MET…")`——这是**诚实的硬拒绝**（正面确认：无静默降级路径，且崩溃发生在生成之后评测之前，GPU 产物可重用）。但唯一可跑通的 `--evaluator fake` 路径收到 wav 路径时构造 `duration_sec=2.0` 的默认 clip **完全不读音频**，输出均匀分布（argmax 恒 'ang'）；`calibrate_eval_contract.py` 真实可执行但只覆盖 v1 整句指标自检，与 frame-level evaluator 校准无关。同源 `Emotion2Vec*Evaluator` 的 `self_evidence_risk=True`/`gate_status=not_met_frame_level` 已如实记录。
- **实验影响**：14 的局部 emotion/transition/强度正式结论在接入独立 evaluator 前**不可能真实产出**；任何以同源 wrapper 出的数字构成自证循环。
- **修法（需用户决策）**：(a) 按 `emofilm_v2_evaluators.md` §4 接入外部独立 frame/滑窗 emotion 分类器 + arousal 回归器，过三项校准（validate_emotion_label_mapping/transition_localization/arousal_direction）并把版本/窗口写入 identity；(b) 资源不允许则显式决策把 14 局部结论降级为"同源 evaluator + self_evidence_risk 标记的诊断性观察"并写入 ADR。同时给 fake 路径输出加 `is_synthetic_evaluator=True` 顶层标记。

### B9 — 强度轴无生产入口：triplet_eval 无 CLI、无三元集合构建工具【2 链印证】

- **轴**：feasibility；**位置**：`eval/triplet_eval.py:864-886`（以 `__all__` 结尾，无 main/argparse）
- **证据**：`evaluate_triplet_dataset` 全仓消费方仅测试；其要求的输入（triplet_specs 三档 + generation_rows 携 group_id/intensity_tier 元数据）没有任何生产工具产出；对照 `eval_local_control.py` 已按 ADR-0020 §6 补 CLI，证明这是缺口而非风格。
- **实验影响**：14 的"low/medium/high 单调率、有效跨度、emotion 保持、分档 WER"整节无法执行，或被临时脚本以不可审计方式补齐。
- **修法**：补两件套——三元 manifest 构建工具（对每个 base case 产出仅 intensity 不同的三条 tagged_text，共享 group_id/seed/prompt_ref/control_ref，独立 utt_id）+ triplet 评测 CLI（与 `eval_local_control.main` 同模式）。

### B10 — MFA aligner 静默缺位 → 空 exact aggregate 正常退出【补审发现】

- **轴**：authenticity；**位置**：`eval/eval_local_control.py:1100`
- **证据**：`aligner = MfaForcedAligner(...) if args.mfa_bin else None`——`resolve_mfa_bin` 的 MFA_BIN/PATH 三级 fallback 在 CLI 路径不可达（MFA 装好也不用）。aligner=None 时 500 条 exact 样本全部 `not_attempted` 与 `failed` 无区分计入 `n_exact_alignment_failed`，构造出 `n_samples=0` 的**合法** aggregate，打印 "Wrote 500 rows" exit 0。全文件无失败率阈值/warning/hard-fail。
- **实验影响**：运行脚本漏传 `--mfa_bin`（最易犯的默认错误）→ 评测"成功"完成但 exact tier 结论整体消失，唯一线索埋在输出 JSON 计数字段。
- **修法**：控制 manifest 含 exact 记录而 aligner 为 None 时 hard-fail（或要求显式 `--no_aligner`）；区分 `not_attempted` 与 `failed` 两个计数；exact n_samples==0 或失败率超阈值时非零退出。

### B11 — MFA 词序同构零校验：boundary 静默错位仍标 aligned【补审发现，词典+对齐产物实证】

- **轴**：correctness/authenticity；**位置**：`eval/eval_local_control.py:481-506`
- **证据**：`resolve_aligned_boundary_sec` 仅做 k 范围检查即取 `words[k-1].end_sec`，从不比对 MFA 词序列与 `text.split()` 的词数/词面。实证：english_mfa.dict 不含 "you'll"/"Peter's" 但含 clitic "'ll"/"'s"；仓库自有 MFA 产物中实测 25 个独立 clitic 区间（recombine 在停顿处失效）；fedd_b 500 行中 4 行 boundary 前有必拆词、6 行有连字符/逗号 token。拆分发生在 boundary 前时 `words[k-1]` 指错词，boundary 秒数偏差约一个词长（0.2-0.5s），status 仍 "aligned" 进 exact aggregate。
- **实验影响**：headline 指标 `mean_abs_boundary_error_sec` 被静默污染（间歇性、依赖生成音频的停顿），事后从落盘 row 无法识别。裁决 blocking：静默假数据进 exact 层指标，且修法便宜。
- **修法**：对齐返回前做词序同构校验（normalize 后逐位比对），不匹配 → 显式 `status="failed", reason="word_sequence_mismatch"` 落入既有排除路径；可选 clitic 合并重试并记录 reconciliation。

### B12 — 陈旧 WAV 混源进 v1 整体回归门【补审发现】

- **轴**：authenticity；**位置**：`tools/inference_emo_film.py:301-310` + `eval/eval_emo_film.py:168-198`
- **证据**：非 eos 分支只写诊断 row + warning，**不删除**上一轮同名 WAV（全文件无 os.remove）；而 v1 回归门 `pair_wavs_strict` 按目录 glob 配对，从不读 generation manifest；wav_sha256 已按设计移除，目录 WAV 与 GenerationRow 零绑定。复用同一 output_dir 续跑是 v1 实跑先例（`full_generation_4gpu_resume_command.txt`）。
- **实验影响**：正式轮某条 utt 非 eos → 旧 checkpoint 的同名 WAV 仍在目录 → 集合完全匹配、expected_count 通过，v1 回归门把旧模型音频当 v2 正式输出打分，回归数字无法归因到唯一 checkpoint。
- **修法**：非 eos 分支显式删除既有 `{utt_id}.wav`（目录内容==manifest eos 集合的硬一致）；且 14 编排从 manifest eos rows 物化 hyp 评测视图（symlink 平铺）而非直接用生成目录，eos 计数/非 eos 名单写进 evaluation identity。

### B13 — WER 参考文本逐条静默回退 Whisper-vs-Whisper【补审发现】

- **轴**：authenticity；**位置**：`eval/eval_emo_film.py:372-400,274-277,430-433`
- **证据**：utt_id 不在 text_map 时零日志回退为"参考音频转写 vs 合成转写"；`load_text_manifest` 对空 text 行静默跳过；WARN 仅在 text_map 整体为空时打印（部分缺失/传错 manifest 零重叠时连 WARN 都没有）；输出 JSON 无 gt/fallback 覆盖率字段（`used_gt` 标志在生产路径是死代码）。
- **实验影响**：回退行的 WER 系统性偏低（转写一致性错误互相抵消），混入标称"论文口径可懂度 WER"的均值与 v1 基线（全 1500 条 ground-truth）比较，可能伪造"WER 无退化"结论且事后不可发现。
- **修法**：`--ref_text_manifest` 提供时任一配对 utt_id 缺 ground-truth → hard-fail 列出缺失 id；输出 JSON 增加 `wer_ref_source: {gt_count, fallback_count}`；空 text 行报错而非跳过。

---

## 3. Should-fix（24 项，实验可容忍但必须记录/尽量随手修）

| #   | 位置                                                      | 问题                                                                                                                                                                                                                                                          | 印证                           |
| --- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| S1  | `generate_tagged_jsonl.py:801`                          | 产出端从不调用`validate_span`，schema 门禁在生成时刻缺位（docstring 与实现不符）                                                                                                                                                                            | data-6, authenticity-6         |
| S2  | `train_utils.py:307-317`, `executor.py`               | 13 DoD 训练观测缺失：无 per-group LR 日志、无目标有效率/监督覆盖统计、NaN 梯度仅 warning+跳步（无 hard-fail）                                                                                                                                                 | train-5, precheck-9            |
| S3  | `train_emo.py:50`, `write_emofilm_run_identity.py:42` | v2 合同目录无 provenance 基础设施：训练 contract_hash 要么绑 v1（重生的 v2 span 不进身份）、要么 FileNotFoundError；refresh 工具硬编码 v1                                                                                                                     | data-7, precheck-10, train-6   |
| S4  | `train_emo.py:266-341`                                  | `assert_no_dead_config` 未接训练入口（仅静态测试覆盖当前 yaml），运行时改配置不被拒                                                                                                                                                                         | train-2                        |
| S5  | `train_emo.py:51,324`                                   | `--seed` 只写进 identity/resolved.yaml，从不设置 RNG；identity 记录的 seed 可与真实 seed 不符                                                                                                                                                               | train-3                        |
| S6  | `generate_tagged_jsonl.py:833-839`                      | ESD TextGrid 缺失时 end_sec 静默写伪造占位 1.0，provenance 不区分真实 xmax 与 fallback                                                                                                                                                                        | data-9, authenticity-11        |
| S7  | `write_emofilm_run_identity.py:262`                     | patch bundle`git add -A` 把 untracked 的 data/contracts/emofilm_v2 数据卷进源码 patch（身份混同 + bundle 膨胀）                                                                                                                                             | data-8                         |
| S8  | `conf/emo_film.yaml:62-66`                              | followup A 核实：decode_config 指纹确不含采样超参（top_p/top_k/win_size/tau_r 固定于模型构造）。**B7 修复后**（source 身份真实反映代码/yaml 变化）风险可容忍，需在实验纪律中显式"不改采样参数"                                                          | inference-3                    |
| S9  | `model_emo.py:37`                                       | `tts` 仍保留 `**kwargs` 吞咽，spec #2 DoD 不变量未字面达成                                                                                                                                                                                                | inference-6                    |
| S10 | `triplet_eval.py:739`                                   | `wer_eval=None` 时 hypothesis 恒空串 → 全样本 WER=1.0 进 mean 分母，docstring 却称"记 0"，aggregate 无 no-wer 标记                                                                                                                                         | eval-10, authenticity-7        |
| S11 | `triplet_eval.py:392-397,447-464,752-762`               | 三处身份/一致性弱点：mapping 形态身份三档全 None 恒通过；重复 utt_id/(group,tier) 冲突静默 last-wins；组一致性 set 比对对真实 span（无 text/emotion/speaker 字段）全 None 恒通过                                                                              | eval-4, eval-5, authenticity-9 |
| S12 | `eval_local_control.py:403` + followup C 两处           | `detect_transition_from_frames` 无 NaN 门禁；followup C（validate_transition_localization/validate_arousal_direction 相邻 argmax 全 NaN 脆弱）核实确未修，后者空输出静默注入 0.0                                                                            | eval-6                         |
| S13 | `eval_local_control.py:948-953`                         | FEDD aggregate 不携带 evaluator 身份与 self_evidence_risk，同源/异源结论无法在 aggregate 层区分（与 B8 联动）                                                                                                                                                 | eval-7                         |
| S14 | `triplet_eval.py:285`                                   | triplet 分档 WER 无文本 normalization，与 v1 口径（lowercase+去标点+数字转词）不一致，绝对值被系统性抬高                                                                                                                                                      | eval-8                         |
| S15 | `acoustic_evaluators.py:751-757`                        | Emotion2Vec 逐帧头调用在`torch.no_grad` 之外 + 首次完整前向是死计算（性能/内存，非正确性）                                                                                                                                                                  | eval-9                         |
| S16 | `processor.py:45-52`                                    | `parquet_opener` 损坏 shard 仅 warning 后继续，冻结训练集可静默缩水                                                                                                                                                                                         | authenticity-10                |
| S17 | `label_fedd_emotion2vec.py:65-85`                       | FEDD-A 参考音频 227/500（45.4%）未过 emotion2vec 一致性校验（other=128）却无过滤/caveat，14 对 FEDD-A 报 Emo-SIM/DTW 需显式 caveat                                                                                                                            | gap-2-2                        |
| S18 | `eval_local_control.py:116-164,1016-1022,631-647`       | MFA 工程三项：per-sample 全量子进程（500 条×30-60s/条，单条件数小时）+`/tmp/mfa_eval_v2` 固定路径并行多 checkpoint 竞态；aligner 一切异常吞成 per-sample failed（基础设施错误伪装成对齐失败）；EvaluationRow 不记录 aligner 身份（MFA 版本/词典/声学模型） | gap-1-3/4/5                    |
| S19 | `generate_tagged_jsonl.py:185-189`                      | 词级情感 pseudo-label 与句级标签一致率实测仅 24%（5 类随机=20%），argmax 分布畸形（sur 45% vs 句级先验 1.6%）——词级情感控制训练信号近噪声，14 报告口径必须显式此限制（sentence_broadcast 弱监督的定量面目）                                                 | gap-4-2                        |
| S20 | `generate_tagged_jsonl.py:467-497`                      | 校准链恒空：canonical predictor 从不输出 calibrated 键、全仓无 calibrated=True 生产实现——schema 的校准机制是"制度在、从未行使"，训练强度目标=未校准压缩量程 raw arousal，需显式记录                                                                         | gap-4-3                        |
| S21 | `eval_emo_film.py:181-198`                              | v1 整体门的 ref_dir（平铺参考 wav 视图）重建无工具、无记录（v1 的 eval_refs 是刻意不迁移的 symlink 视图，现为空），14 编排缺这一环                                                                                                                            | gap-5-3                        |
| S22 | `run_inference_parallel.py:78-94`                       | 并行包装不透传`--seed/--fp16/--max_samples`，并行路径 seed 恒默认 1986 不可配（当前实验恰用 1986，故非 blocking）                                                                                                                                           | inference-4                    |
| S23 | `emo_tokenizer.py:91-92`                                | 未知 emotion/intensity 值与格式变异标签双重静默回退 neu/low，控制身份可被无声替换（现有 manifest 全合法，休眠路径）                                                                                                                                           | inference-5                    |
| S24 | `build_emofilm_contract.py:1158`                        | CLI 假入口：`__main__` 只 parse_args 即 exit 0（**裁决自 blocking 降级**：全仓+handoff 无任何流程把它当门禁命令；真实校验以库形式在 splits/tests 中执行。但作为诱饵入口应补真实 main 或删除）                                                         | precheck-6                     |

## 4. Deferred（9 项，并入 followup 记录）

| #  | 位置                                      | 问题                                                                                                              |
| -- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| D1 | `train_emo.py:254-262`                  | 多卡 DDP init.pt 写入竞态（非 0 rank 可能 FileExistsError 崩溃；单卡实验不触发）                                  |
| D2 | `span_align.py:179-201`                 | 乱序/重叠输入 span 被单调 clamp 静默截断，不校验"按时间序"前置假设                                                |
| D3 | `inference_emo_film.py:82-117`          | ESD 缺 Neutral 时静默 fallback 首个情感目录但 prompt_source 仍记 neutral（休眠路径：ESD 数据集恒有 Neutral）      |
| D4 | `triplet_eval.py:571-574`               | 强度单调性基于每组单 seed 单次抽样，无方差估计                                                                    |
| D5 | `triplet_eval.py:67`                    | triplet 整句行挪用`boundary_evidence_tier='exact'` 仅为过校验，跨管线按 tier 汇总会虚增 exact 计数              |
| D6 | `eval_local_control.py:369-430`         | `detect_transition_from_frames` 的 frame_rate_hz 死参数且 docstring 描述与实现不符                              |
| D7 | `data/contracts/emofilm_v1/eval/fedd_b` | fedd_b_0013_sur2neu_0293 的 prompt 与拼接源为同一录音（1/500，prompt 复读捷径 + Emo-SIM 参考污染，记录项）        |
| D8 | `write_emofilm_run_identity.py:702`     | decode_config 为 None 的 row 在 resume 指纹时抛 dict(None) TypeError（上游`.get` 可产 None）                    |
| D9 | `emofilm_v2_schema.md:5` 等             | 扁平化后旧名（`build_emofilm_v2_contract.py`/`eval_emofilm_v2.py` 等）残留于合同/报告文档，按文执行找不到文件 |

---

## 5. 正面证据：核查通过的关键合同点（摘要）

以下逐行核查确认**正确落地**（详见 journal 各链 verified_ok，共 60+ 条）：

- **协议不变量**：训练序列恒定单流 target-only（SOS+FiLM(text)+task+teacher speech，目标 IGNORE×(1+text_len)+speech+EOS），无 bistream/fill token；FiLM 逐 token 作用于目标文本嵌入（γ=1/β=0 恒等初始化）；prompt_* 不进 LLM 条件。训练/推理共用 `encode_plus`，id 空间 {1..5}/{1..3} 与 schema 一致，**训练/推理 span 语义同源**。
- **finish_reason 门控全链闭合无旁路**：`max_len<=min_len` 解码前 input_rejected；五值枚举结构化互斥；仅 eos yield token；`model_emo.tts` 消费 `last_decode_result` 非 eos 不 token2wav；入口仅 eos 写 WAV+wav_path；全仓无第二条生成侧 WAV 写入路径。ADR-0007 采样语义（原始 scores 重采样/RAS fallback）保留。
- **seed 全链**（Grilling 决策）：per-utt torch+cuda RNG 重置、默认 1986/cli 可配、独立 row.seed、进指纹、validator 拒 bool、triplet 比实际 seed。
- **remediation 12 项全部确认落地**：#1 非 EOS 门控、#3 decode_config 长度三项全链透传、#5 训练 identity（optimizer 分组一致）、#6/#7 patch bundle GIT_INDEX_FILE 隔离正确覆盖 untracked、#8 exact 排除对齐失败、#9 `_strict_pair` hard-fail 无残留、#10 triplet 入口逐行 validate、#11 空/NaN→valid=False 不进分母、#12 非有限分布门禁。
- **数据资产实测**：`author_best_model.pth` strict load 成功（16 键形状全匹配 768d/5emo/3VAD，sha 前缀 `a4b373…` 与 ADR-0020 一致）并在真实 word_block 上 CPU 推理产出合法输出——**门禁 1 的真实重生路径静态可行**；word_blocks 6746 utt/75619 词块全量在盘、结构与消费假设逐键一致；label_map 与 tokenizer 同序无错位；冻结切分 20774/1092 实数核验（train=ESD 14362+IEMOCAP 6412）；三个 eval manifest 2500 行 wav 路径全部在盘、fedd_b 500 行 boundary_word_index 与 tagged_text 全量自洽、eval 集与 train/cv 零重叠。
- **正式口径确认**：`conf/emo_film.yaml` max_epoch=5（论文对齐）、warmup 250、三组 LR 1e-4/1e-4/1e-5——历史"epoch200"质疑不成立（gan 段不被 `--model llm` 消费）。checkpoint 生命周期 13→14 闭合（latest→final→filter→strict load 同拓扑）。
- **StubPredictor 本体**：仅经显式 `--stub_predictor` flag 可达，产物携显式身份标记（predictor_class/全零 sha/`<stub>`）——占位路径自身诚实，缺的是下游门禁消费（B3）。
- **MFA 环境在位**：mfa 3.3.9 + english_mfa 声学/词典已装，TextGrid 解析对真实输出正确，帧→秒换算无硬编码错误。

---

## 6. 13/14 实验执行 checklist（预检产出 + 审查修订）

按序执行；标注当前状态与阻塞项（修复 §2 后逐项转 ready）：

| 步 | 内容                                                                                                          | 状态       | 阻塞/说明                                                                               |
| -- | ------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------- |
| 0  | 环境与代码基线固定（用户决定是否 commit；不 commit 则 B7 修复必须先行）                                       | unverified | B7                                                                                      |
| 1  | 门禁 1：`generate_tagged_jsonl` 用 `author_best_model.pth` 重生真实 IEMOCAP v2 span（替换 18 行 fixture） | ready*     | *前置 B2（关 Flex fallback）+B4（阈值决策）+S1（产出端 validate_span）；数据/模型已在盘 |
| 2  | 重生 ESD v2 span（替换 5 行样本）                                                                             | unverified | 同上 + S6                                                                               |
| 3  | provenance/contract 门禁（加强版审计）                                                                        | blocked    | B3（门禁修复后才有意义）+S24                                                            |
| 4  | **接通 span→训练数据管线（票据未列的隐性前置工程）**                                                   | blocked    | B1                                                                                      |
| 5  | 短周期训练 smoke + 监督覆盖率断言 + 真实三样本 GPU smoke                                                      | blocked    | B1+S2                                                                                   |
| 6  | 正式训练（13）                                                                                                | blocked    | 上游全部                                                                                |
| 7  | 训练 identity 落盘核验                                                                                        | ready      | S3 建议随手修                                                                           |
| 8  | strict-load + 固定小 batch 前向 + 三分区各一条生成 smoke                                                      | unverified | 依赖 6                                                                                  |
| 9  | v2 派生 eval manifest（注入版本化 control/prompt ref）                                                        | blocked    | B5（新工具）                                                                            |
| 10 | 固定生成（14）：小规模确认 → 全量分片                                                                        | blocked    | B5+B6+B7+B12（非 eos 清理）                                                             |
| 11 | 整体质量评测（v1 同口径 WER/Emo-SIM/DTW）                                                                     | blocked    | B12+B13+S21（ref_dir 重建工具）                                                         |
| 12 | FEDD 局部 span/transition/boundary 评测                                                                       | blocked    | B8（evaluator 决策）+B10+B11                                                            |
| 13 | intensity 三元单调性评测                                                                                      | blocked    | B9（CLI+构建工具）+B4                                                                   |
| 14 | 联合报告（含 S17/S19/S20 的显式口径 caveat）                                                                  | blocked    | 上游全部                                                                                |

**干跑必须覆盖的"真实数据上从未走过的路径"**（handoff §4.2 要求，审查后更新）：① `generate_tagged_jsonl` 真实重生（静态可行已证，端到端未跑）；② span→batch 新接线（B1 修复件）；③ v2 派生 manifest → 生成 → 评测的三跳接缝；④ MFA 真实对齐一条样本（环境已证在位，`eval_local_control` 调用路径未实跑）。

---

## 7. 修复路线建议

按依赖排序为四个工作流（blocking 修复走小步 TDD + 聚焦测试，完毕跑全套 pytest 对 651 基准回归；静默降级的修法一律=显式 hard-fail 或显式记录身份，不是把 fallback 藏更深）：

- **W1 数据重生与门禁加固**（先行）：B2（Flex 默认关+真实 provenance）→ B3（审计双向修复）→ B4（**用户决策**：分位校准 vs 显式降级）→ S1/S6/S20 随手 → 步骤 1-3 重生+门禁。
- **W2 训练接线**（核心工程量）：B1 五层接线 + expect_spans 断言 + S2 监督覆盖日志/NaN hard-fail + S3/S4/S5 随手 → 步骤 4-5 smoke。
- **W3 生成身份闭合**：B5（v2 派生 manifest 层 + 写前自校验）→ B6（指纹纳入内容摘要 + 缺身份守卫）→ B7（capture_source_identity 接入）→ B12（非 eos 清理）→ S8 纪律显式化/S9/S22/S23。
- **W4 评测入口与真实性**：B8（**用户决策**：接独立 evaluator vs 显式降级口径）→ B9（triplet CLI+构建工具）→ B10/B11（MFA 门禁+同构校验）→ B13（WER 回退 hard-fail）→ S10-S18/S21。

**三个必须由用户拍板的决策点**：① B4 强度阈值（推荐分位校准，标注器方向性信号已证可行）；② B8 门禁 2（接独立 evaluator or 显式降级为诊断性观察写入 ADR）；③ B7 的形态（推荐 patch bundle 路径而非强制 commit，与 ADR-0020 §7"提交由用户决定"一致）。

---

## 增补：第二次独立审查运行的交叉验证（run `wf_cf56b521-308`，115 子代理 / 83 发现 / 5 补审缺口）

同日以相同方法学（五链路 + 票据预检 + 双 lens 对抗核查 + critic 补审）独立执行的第二次审查运行，交叉验证结果：**本报告 §2 的 13 条 blocking 全部被独立印证，无一被反驳**；第二次运行另有以下本文缺失的发现（均经其自身双 lens 对抗核查 CONFIRMED），合并如下。

### B14 — FEDD-B 测床 51/500（10.2%）emo2vec 一致性失败行无行级标记、无消费端过滤，将全额进入 exact 层分母【4 核查 CONFIRMED/blocking】

- **轴**：soundness；**位置**：`tools/label_fedd_emotion2vec.py:65-84` + `data/contracts/emofilm_v1/eval/fedd_b/manifest.jsonl`
- **证据**：一致性判定 `emo2vec_label in (emo_from, emo_to)` 只累计进聚合 report（该 report 文件已在冗余清理中丢失——89.8% 通过率的证据链已断），行级仅回写 emo2vec_label/emo2vec_score，**无 pass/fail 字段**。实测 51/500 行 emo2vec_label ∉ {emo_from, emo_to}（失败行 score 均值 0.887，属高置信不一致；41/51 涉 sur，分布非均匀，如 `fedd_b_0011_ang2hap_0000` 标 sur@0.974）。全链 grep 确认无任何代码读 emo2vec_label 做过滤；`_strict_pair` 的 utt_id 集合相等约束使事后剔除必须同时过滤 control 与 generation 两侧，目前无工具支持。
- **实验影响**：独立校验器都无法在参考构造音频上识别出任一端情感的 51 行，原样进入 **exact 层**（旗舰证据层）front/back 命中率、transition direction、boundary error 分母，系统性压低命中率且偏差非均匀（sur 类集中）；『局部控制被独立证明』的头条数字被未验证测床污染，v1/v2 对比同样继承该噪声；审稿人可据 10.2% 未处置失败率直接质疑测床有效性。与 S17（FEDD-A 45.4%，approximate 层）同机制，但 FEDD-B 在 exact 层，严重度更高。
- **修法（三选一并记录在案，不是静默过滤；与 B5 的 v2 派生 manifest 层同票实现）**：① 最小修——扩展 label 工具写行级 `emo2vec_consistency_pass`，report 持久化进 v2 合同目录；② `derive_v2_eval_manifests` 显式排除 51 行并记录排除计数与理由；③ 论证保留（混合情感整句本就可能不属任一端）+ aggregate 增 with/without 两套敏感性数字。更诚实的替代校验：对 src_utt_from/src_utt_to 两段源 ESD wav 分别做 utterance 级校验（源段有数据集真值）。

### B15 — eval CLI 输出无任何运行身份：ticket-11 的 eval 身份 API 生产调用链零接线【4 核查 CONFIRMED/blocking】

- **轴**：authenticity；**位置**：`eval/eval_local_control.py:1106-1115`
- **证据**：main() 落盘的 output 仅 5 键（metric_contract_version/rows/aggregate_exact/aggregate_approximate/n_samples）——无输入 manifest 路径与内容哈希、无 argv/command、无代码身份、无 `aggregate_identity`、无 evaluator 运行级身份、无时间戳。而完整的 eval 身份 API 已存在且设计正确（`write_emofilm_run_identity.py:851` eval_row_identity_fingerprint、`:877` compute_aggregate_identity——有序集合 hash 可检测 rows 被替换/遗漏/混入、`:1009` write_emofilm_evaluation_identity），全仓调用者仅 tests/——**生产入口一行都没接**；`write_emofilm_generation_identity` 同样生产零调用。直接违反 schema §5『聚合 identity 绑定 rows 集合 + evaluator 身份』与 §6 引用图。与 remediation #5/#6 同根因（能力存在但未接入生产链）。
- **实验影响**：14 对不同 generation manifest（smoke vs 正式、v1 vs v2 模型）各跑一次得到两份结构完全同构的 JSON，事后无法从文件本身回答『这份 aggregate 的命中率来自哪个输入、哪个代码版本、哪个 evaluator』；rows 数组被事后编辑/删行无任何锚可检测；**与 B8 复合**：fake evaluator 跑出的 aggregate 与未来真实评测完全同构（evaluator 身份只藏在 rows 深处），合成数字可被当真实结果引用。
- **修法**（接线既有 API，**样板已在 `tests/test_emofilm_local_control_e2e_smoke.py:460-479` 完整演示，成本低**）：output 增 `aggregate_identity: compute_aggregate_identity(rows)`；落盘后写 `write_emofilm_evaluation_identity` sidecar（command/输入 manifest 路径+行数+内容哈希/evaluator_info 全量不截断/aggregate_identity/建议补时间戳）；B9 的 triplet CLI 票据必须包含同规格身份出口；顺带接线 write_emofilm_generation_identity。输入 manifest 内容哈希不触 ADR-0020 哈希边界（禁区是源码哈希锁与 WAV 产物哈希）。

### 新增 should-fix（S25-S29）

| #   | 位置                                                                                    | 问题                                                                                                                                                                                                                                                                                    | 核查               |
| --- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| S25 | `cosyvoice/utils/train_utils.py:201`                                                  | resume 不恢复 optimizer/scheduler 状态（checkpoint 只存权重+epoch/step），13 DoD『resume 后优化状态一致』无法满足；不改则须在 train_identity 与报告显式记录 "resume=weights-only" 限制                                                                                                  | CONFIRMED          |
| S26 | `cosyvoice/bin/train_emo.py:50`                                                       | `--contract_dir` 默认仍指 `data/contracts/emofilm_v1`：漏传参数时 v2 训练身份绑定 v1 合同哈希且无一致性校验——改 required 或默认 v2 并校验目录内容（与 S3 合并实现）                                                                                                               | CONFIRMED          |
| S27 | `cosyvoice/dataset/processor.py:465-484`                                              | FiLM 输入标签（tagged_text`<emotion>` 标签）与 span 监督标签**双路径异源**，无逐词一致性校验——现存数据已实证同一 utt 两路径矛盾。B1 接线票据必须显式规定标签同源（同一 predictor 同时导出双产物或由 spans 反导 tagged_text）+ 开训前逐词比对 audit，不一致携 utt_id hard-fail | PARTIAL/should-fix |
| S28 | `tools/jsonl_to_cosyvoice_src.py:10-11`                                               | 桥接文件合同漂移：docstring 指向已被 ADR-0020 改名的生产者，且 v1（per-utt tagged_text）与 v2（per-span）的 tagged.jsonl**一名两义**——建议 v2 span 文件改名 `spans.jsonl`，schema 文档同步（配合 D9）                                                                         | CONFIRMED          |
| S29 | `tests/test_eval_smoke.py:31-35`、`tests/test_extract_emotion2vec_frame.py:172-175` | 2 个环境门控测试缺前置时以 FAIL 而非 SKIP 呈现，651 基线在干净环境需人工归因才能与真实回归区分——改`pytest.mark.skipif` + reason，真实资产冒烟加 marker                                                                                                                              | CONFIRMED          |

另：S18（MFA 工程）的量化补充——第二次运行实测单条全流程 **35.1s** → FEDD-B 500 行 ≈ **4.9 小时/每 checkpoint**（仅对齐一项）；修法建议批量对齐 pass（一次建 corpus 单次 `mfa align` 查表，ForcedAligner 协议加可选 align_batch() 默认逐条回退）。S23（emo_tokenizer 静默回退）的量化补充——2500 行真实 tagged_text 实跑触发面 = 0，维持 should-fix；但应加 strict raise/计数器 + 数据重生 auditor 增『残段=0 且值域合法』断言。

### 新增 deferred（D10）

| #   | 位置                                      | 问题                                                                                                                                                                                                             |
| --- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D10 | `tools/build_fedd_tagged_text.py:37-41` | `word_level_tagged_text` 的静默 clamp 与单词句退化会生成与 `method='exact_concatenation_boundary'` 标记矛盾的行（当前数据 0 触发，潜伏缺陷）——clamp 改 raise，与既有 emo_from==emo_to raise 的严格风格对齐 |

### 对 §6/§7 的合并影响

- checklist 步骤 9（v2 派生 eval manifest）增加 B14 处置为其 DoD 一部分；步骤 12/13/14 增加 B15（eval 身份 sidecar）为落盘门禁。
- 修复路线：**B14 并入 W3**（与 B5 的派生 manifest 层同票）；**B15 并入 W4**（与 B9 的 triplet CLI 身份出口同规格，e2e 测试有现成接线样板）。
- 第二次运行的正面核验与本报告 §5 一致，另补充：653 collected / 651 passed / 2 env-fail 基线**实跑复现**；ESD TextGrid 命中率实测 15120/15127（99.95%，S6 的 fallback 触发面仅 7 行）；三个 eval manifest 的 tagged_text 用构建工具全量重生成比对 0 mismatch；MFA 端到端单条实跑成功。
- 第二次运行工件：run `wf_cf56b521-308`（journal 同目录），逐发现全文与核查 verdict 见该 journal。

---

## 附录：审查覆盖与仲裁记录

- **覆盖**：五链 44 个生产文件逐行 + 数据现状实测（manifest 全量程序化扫描、checkpoint state_dict 实测、MFA 环境实测、标注器 CPU 抽样推理 ~1050 词块两套独立复现）+ 13/14 票据逐 DoD + 6 个 critic 缺口补审（MFA 生产路径、14 评测集构建链、resume 语义、强度可分性、v1 回归门接缝、覆盖清点收尾）。critic 确认 `tools/diagnostics/` 为空目录、`cosyvoice/flow|hifigan|transformer` 属 v1 冻结共享路径、`visualize_case.py` 可跑，无进一步缺口。
- **仲裁记录**（核查者分歧 → 主循环读码裁决）：① B2 Flex fallback：existence 核查 PARTIAL（分歧在 CLI 是否显式传参），主循环读码确认默认 True + provenance 硬编码假身份 → 维持 blocking；② B6 gap-3-2 docstring 与实现不符：裁定 docstring 错误、实现缺守卫 → 并入 B6 blocking；③ precheck-6 假 CLI：双 lens PARTIAL/should-fix（无任何流程消费该入口）→ 降级 S24；④ gap-1-1 boundary 错位：existence 核查建议 should-fix、impact 核查 blocking → 按"静默假数据进 exact 指标默认 blocking"判据维持 blocking。
- **一致性说明**：wav_sha256 移除、safe-resume 仅 isfile+指纹、源码哈希锁禁止均为 ADR-0020 设计——所有核查者已预置该边界，本报告无一条发现依据这些设计。iemocap 18 行/esd 5 行 fixture 为已知占位，本报告的相关发现（B3）针对的是"门禁放行它们"而非"它们存在"。
