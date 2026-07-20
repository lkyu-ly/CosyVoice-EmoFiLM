# 基础 Emo-FiLM 作者语义回归与干净重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. All filesystem deletion, bulk movement, `git commit`, `git push`, and training require the authorization gates stated below.

**Goal:** 建立唯一、干净、可追溯的基础 Emo-FiLM 数据准备、训练、推理和本地代理评测主线。

**Architecture:** 对被旧消融、DPO 和诊断分支污染的 Emo-FiLM 专用模块做定向重建，保留通用 CosyVoice/Flow/HiFT 和作者未公开环节的本地补全实现。新版派生产物集中在 `data/contracts/emofilm_v1/`，只训练一个 5-epoch 模型并执行 `3→30→本地固定2500条`。

**Tech Stack:** Python 3.10、PyTorch、fairseq emotion2vec upstream、MFA、HyperPyYAML、PyArrow、ONNX Runtime、pytest、torchrun、`emofilm-eval-v2`。

---

## 执行约束

- 代码仓库：仓库根目录。
- 当前基线：`main@5ad481d`；矩阵代码：`feature/emofilm-author-model-control@e265281`。
- 实施前使用隔离 worktree；不得在当前共享 main 直接开始大改。
- 本计划不授权删除、批量移动、commit、push、下载大型模型或训练。
- Task 1 只生成备份与清理清单；Task 2 必须停下取得危险操作确认。
- 新版通过前不得删除正式 v6 WAV、旧 `init.pt/final.pt` 或旧 full parquet。
- 任何执行轨迹差异若不改变统计行为，不再为其增加实现或测试。
- 训练起点硬约束：`CosyVoice-BlankEN/` 仅用于 Qwen2 文本骨干/tokenizer；CosyVoice2 语音 LM 必须从 `pretrained_models/CosyVoice2-0.5B/llm.pt` 加载。
- 新建生产训练不得走 `fresh` 分支；训练命令必须显式传入上述 `llm.pt` 的 `--checkpoint`。缺失、路径不符或 identity 未记录该 checkpoint 时，GPU 训练和后续生成均停止。

## 目标文件映射

### 重建/精简

| 文件 | 最终职责 |
|---|---|
| `cosyvoice/llm/emo_film.py` | EmotionEncoder 与 FiLMLayer |
| `cosyvoice/llm/llm_emotion.py` | 作者语义训练前向与 Emo-FiLM 专用推理 |
| `cosyvoice/cli/frontend_emo.py` | target 三元组与 Flow/HiFT prompt 输入 |
| `cosyvoice/cli/model_emo.py` | 非流式生成和会话生命周期 |
| `cosyvoice/bin/train_emo.py` | 唯一基础 SFT 入口 |
| `cosyvoice/utils/train_utils_emo.py` | 冻结、optimizer、checkpoint 生命周期 |
| `conf/emo_film.yaml` | 唯一基础配置 |

### 局部修改

| 文件 | 修改 |
|---|---|
| `tools/extract_emotion2vec_frame.py` | 作者 fairseq base 768d/50Hz |
| `tools/build_word_emo_dataset.py` | 固定帧率切词与拒绝原因 |
| `tools/generate_tagged_jsonl.py` | 作者 WordSequence checkpoint、无平滑、双标签合并 |
| `cosyvoice/dataset/processor.py` | token/mel 1:2 联合裁剪 |
| `cosyvoice/tokenizer/emo_tokenizer.py` | lowercase 与闭合标签检查 |
| `cosyvoice/utils/common.py` | EOS 重采样可复用原语、RAS fallback 分布 |
| `tools/make_parquet_list.py` | 异步异常传播、train/cv 直接打包 |
| `tools/inference_emo_film.py` | strict trained load、generator close |
| `tools/run_inference_parallel.py` | 唯一正式分片入口 |
| `eval/eval_emo_film.py` | 仅保留 v2 正式指标面 |

### 新增

| 文件 | 职责 |
|---|---|
| `cosyvoice/utils/emo_checkpoint.py` | base whitelist、trained strict、参数哈希 |
| `tools/build_emofilm_contract.py` | 冻结成员关系并组织单一数据合同 |
| `tools/write_emofilm_run_identity.py` | 训练/生成/评测运行身份 |
| `tests/test_emofilm_data_contract.py` | 数据合同外部测试 |
| `tests/test_emofilm_training_contract.py` | 训练语义测试 |
| `tests/test_emofilm_inference_contract.py` | 推理语义测试 |
| `tests/test_emofilm_runtime_contract.py` | 清理、批量、配对和指标测试 |
| `tests/test_emofilm_e2e_smoke.py` | 三样本端到端 smoke |

## Task 1: 固化代码与历史证据，生成危险清理清单

**Files:**
- Create outside repo: `../backups/emofilm-rebuild-20260718/`
- Modify: `docs/data_exp_inventory.md`
- No deletion in this task.

- [ ] **Step 1: Verify immutable code identities**

```bash
git rev-parse main
git rev-parse feature/emofilm-author-model-control
git status --short --branch
```

Expected: main 解析为 `5ad481dadd6b9b8516890335325558539dbae410`，feature 分支解析为 `e265281ab7927a4f9aa4793b4d66974db5f24b76`；除已知 `.serena/` 外无用户未说明改动。若状态不同，停止并更新基线。

- [ ] **Step 2: Prepare exact backup commands without executing Git mutations**

生成待确认命令清单，内容包括：

```bash
git tag archive/emofilm-pre-rebuild-main-20260718 5ad481d
git tag archive/emofilm-author-control-20260718 e265281
git bundle create "../backups/emofilm-rebuild-20260718/code.bundle" \
  archive/emofilm-pre-rebuild-main-20260718 \
  archive/emofilm-author-control-20260718
```

本 Step 只写入命令清单。创建 tag 属 Git 修改，执行前单独确认。

- [ ] **Step 3: Build read-only asset inventory**

为所有第一阶段候选输出 JSONL；每行必须包含 `path`、`bytes`、`kind`、`reason`、`sha256` 和 `rebuild_source`。`kind` 只允许 `regenerable`、`duplicate`、`evidence`。

候选必须覆盖：

- `data/emo_feats/`、`data/word_blocks/`；
- 四组消融 parquet/src；
- author-control v1-v5 重复 `author_sft_converted.pt`；
- `datasets/ESD.BAK/` 与重复 zip；
- 重构后拟删除的 tools/tests/config 文件。

- [ ] **Step 4: Build compact evidence package specification**

证据包清单至少包含：

- 旧 WordSequence checkpoint/log/label map、目录统计与代表样本；
- v1-v5 非 WAV JSON/JSONL/manifest/log；
- v6 aggregate、per-sample metrics、technical gates、matrix manifest、equivalence、conversion manifest；
- 两个代码 commit 和现有报告路径。

- [ ] **Step 5: Stop for two dangerous-operation confirmations**

分别请求：

1. 创建 Git tag/bundle、源码快照，以及从已验证 main 基线创建隔离实施分支/worktree；
2. 执行第一阶段精确删除清单。

未经两项明确确认不得进入 Task 2。

## Task 2: 执行已批准备份与第一阶段清理

**Prerequisite:** 用户已对 Task 1 两份精确清单明确确认。

- [ ] **Step 1: Create tags, bundle and source snapshot**

执行批准命令；源码快照排除：

```text
.git data exp checkpoints pretrained_models __pycache__ .pytest_cache .ruff_cache .serena
*.pt *.pth *.onnx *.safetensors *.wav *.parquet *.tar
```

- [ ] **Step 2: Verify backup recovery**

```bash
git bundle verify "../backups/emofilm-rebuild-20260718/code.bundle"
git clone "../backups/emofilm-rebuild-20260718/code.bundle" "../backups/emofilm-bundle-check"
git -C "../backups/emofilm-bundle-check" cat-file -e 5ad481d^{commit}
git -C "../backups/emofilm-bundle-check" cat-file -e e265281^{commit}
```

Expected: bundle valid；两个 commit 可读取。

- [ ] **Step 3: Create and verify compact evidence archives**

每个压缩包同时写 `.sha256`，解包到 `/tmp` 后检查必需 manifest 可读取。

- [ ] **Step 4: Execute only approved deletions**

只删除 Task 1 清单中逐路径批准的项目；不得使用宽泛 glob。删除脚本逐条核对 `resolved_path` 位于批准根目录，并追加删除审计 JSONL；每行固定包含已批准的绝对路径、ISO-8601 删除时间、删除前 SHA-256 和 `status=deleted`。

- [ ] **Step 5: Reconcile inventory and free space**

更新 `docs/data_exp_inventory.md` 的实际状态；比较删除前后 `df -h` 与 `du`。不存在于批准清单的文件变化视为失败。

- [ ] **Step 6: Create the isolated implementation worktree**

按 `superpowers:using-git-worktrees` 检查项目既有 worktree 约定和忽略规则，然后从已验证的 `main@5ad481dadd6b9b8516890335325558539dbae410` 创建唯一实施分支 `rebuild/emofilm-v1`。创建前确认当前没有同名分支或 worktree；创建后再次验证目标 worktree 的 `HEAD` 正是该 commit，且除项目约定的本地元数据外状态为空。Task 3–7 全部在该 worktree 执行，不在共享 main 修改代码。

## Task 3: 建立最小测试面并定向重建核心代码

**Files:** 使用上文目标文件映射；删除旧 tests 仅限已批准清理清单。

- [ ] **Step 1: Write failing model/training contract tests**

`tests/test_emofilm_training_contract.py` 固定包含以下外部断言：

| 测试 | 输入与断言 |
|---|---|
| `test_emotion_classifier_reads_modulated_text` | 用 forward hook 捕获 classifier 输入；断言它与 FiLM 输出相同，不是 Qwen decoder hidden |
| `test_effective_model_topology_and_shapes_match_author_contract` | 断言 hidden=896、EmotionEncoder embedding 为 `(6,896)/(4,896)`、FiLM 为 `(1792,896)`、classifier 为 `(6,896)`、speech head 输出为 6564 |
| `test_exact_trainable_module_names` | 调用冻结 helper 后收集 `requires_grad=True` 参数；前缀集合只能是 emotion encoder、adapter、decoder |
| `test_classifier_is_frozen_and_not_in_optimizer` | 断言 classifier 全部冻结，且 optimizer 的参数 ID 集合不包含 classifier |
| `test_token_mel_ratio_two_trims_both_sequences` | 构造 5 个 speech token 和 8 帧 mel；ratio=2 后断言二者长度为 4 和 8 |
| `test_dataset_compute_fbank_accepts_token_mel_ratio` | 从 YAML 构造 data pipeline；断言 `compute_fbank(token_mel_ratio=2)` 可调用且无 `TypeError` |
| `test_config_uses_static_batch_four` | 解析 resolved YAML；断言 batch type 为 static、size 为 4 |
| `test_base_checkpoint_rejects_non_emotion_missing_key` | 删除一个 Qwen 主干键后加载 base；断言抛出包含该键名的异常 |
| `test_trained_checkpoint_loads_strictly` | 分别删除和增加一个键；两次 trained load 都必须失败 |

- [ ] **Step 2: Run training contract tests and verify RED**

```bash
pytest tests/test_emofilm_training_contract.py -q
```

Expected: failure on old `emo_loss_on=llm_output`、classifier trainable、dynamic batch，以及 `compute_fbank` 不接受配置中 `token_mel_ratio` 的接口缺口。

- [ ] **Step 3: Replace Emo-FiLM training modules**

实现固定前向：

```python
text_emb = self.llm.model.model.embed_tokens(text_token)
emotion_features = self.emotion_encoder(emotion_ids, intensity_ids)
modulated_text_emb = self.emotion_adapter(text_emb, emotion_features)
emotion_logits = self.emotion_classifier(modulated_text_emb)
loss_emotion = self.criterion_emotion_cls(
    emotion_logits.reshape(-1, emotion_logits.size(-1)),
    emotion_ids.reshape(-1),
)
```

冻结集合固定为：

```python
AUTHOR_TRAINABLE_MODULES = (
    "emotion_encoder",
    "emotion_adapter",
    "llm_decoder",
)
```

- [ ] **Step 4: Write failing inference contract tests**

`tests/test_emofilm_inference_contract.py` 固定包含：

| 测试 | 输入与断言 |
|---|---|
| `test_text_is_lowercased` | 输入含大写正文和标签；捕获 tokenizer plain-text 调用并断言正文已 lowercase、标签语义不变 |
| `test_llm_condition_excludes_prompt_text_and_prompt_speech` | target/prompt 使用不同哨兵 embedding；捕获首轮 LLM 输入，只允许 SOS、target FiLM、task |
| `test_prompt_is_retained_for_flow_and_hift` | 同一哨兵 prompt 必须仍出现在 speaker embedding、Flow token 和 prompt feature 输入 |
| `test_max_len_is_200` | 让采样器永不返回 EOS；断言最多执行 200 个解码步 |
| `test_eos_is_resampled_before_min_len` | 采样序列先返回 EOS 再返回普通 token；min 前断言普通 token 被产出且 EOS 未终止 |
| `test_auxiliary_special_tokens_do_not_stop_or_extend_prefix` | 依次返回两个 aux token、普通 token、EOS；断言仅普通 token 产出并进入后续前缀 |
| `test_ras_fallback_uses_unmodified_scores` | 首次抽到重复 token 后触发 fallback；断言第二次采样收到的 logits 与原始 logits 相同 |

测试不得断言使用完整 prefix 或 CPU provider。

- [ ] **Step 5: Run inference contract tests and verify RED**

```bash
pytest tests/test_emofilm_inference_contract.py -q
```

- [ ] **Step 6: Replace Emo-FiLM inference modules**

初始 LLM 条件固定：

```python
initial = torch.cat((sos, modulated_target, task), dim=1)
min_len = int(text_len.item()) * 2
max_len = 200
```

允许内部继续调用 KV-cache wrapper，但 wrapper 必须支持：

- EOS-before-min 原分布重采样；
- `token == speech_token_size` 才停止；
- 其他 `token >= speech_token_size` 跳过；
- RAS fallback 不修改原始 scores。

- [ ] **Step 7: Rebuild runtime lifecycle and checkpoint helper**

`model_emo.py` 以 `try/finally` 清理所有 UUID 字典；`emo_checkpoint.py` 只暴露三个接口：`load_base_state(model, state)` 仅允许新版情感模块缺失；`load_trained_state(model, state)` 对缺失键和多余键均失败；`hash_model_state(model)` 按排序后的 state-dict 键、dtype、shape 和连续 tensor bytes 计算 SHA-256。

- [ ] **Step 8: Run focused contracts**

```bash
pytest \
  tests/test_emofilm_training_contract.py \
  tests/test_emofilm_inference_contract.py \
  tests/test_emofilm_runtime_contract.py -q
```

Expected: all pass。

## Task 4: 构建 `emofilm_v1` 单一数据合同

**Files:** 标注、合同、parquet 工具和 `tests/test_emofilm_data_contract.py`。

- [ ] **Step 1: Write failing data contract tests**

必须覆盖：

| 测试 | 输入与断言 |
|---|---|
| `test_author_frame_contract_is_768d_50hz` | 读取产物 schema 和 provenance；每帧 768 维，帧步长记录为 20ms |
| `test_author_word_sequence_checkpoint_is_768_5_3` | strict load 作者 checkpoint；分类输出 5、回归输出 3 |
| `test_author_word_sequence_state_dict_shapes_match_model_definition` | 断言 attention `(2304,768)`、FFN `(3072,768)/(768,3072)`、heads `(5,768)/(3,768)` |
| `test_merge_key_is_emotion_and_intensity` | 相邻词同 emotion 不同 intensity 时不得合并；两者都相同才合并 |
| `test_no_smoothing_option_exists_in_production_cli` | 生产 CLI parser 不暴露 majority/smoothing 参数，主函数也不调用历史平滑函数 |
| `test_train_cv_membership_matches_frozen_ids` | rejected 集合扣除后，train/cv ID 分别等于冻结集合，且交集为空 |
| `test_rejected_iemocap_is_within_one_percent_and_not_concentrated` | 根据 manifest 计算总体、speaker、emotion 分布；超过已批准数据容差或出现单组异常集中时失败 |
| `test_esd_and_fedd_eval_assets_are_complete` | 固定 2500 个 ID 的 target/reference/prompt/文本/标签必须逐项存在 |
| `test_train_and_cv_parquet_are_directly_loadable` | 分别遍历 train/cv `data.list` 的全部 shard，schema 可解码且无共享 shard |

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest tests/test_emofilm_data_contract.py -q
```

- [ ] **Step 3: Obtain and record official emotion2vec-base asset**

下载属于网络/大型资产操作，执行前按项目规则确认。provenance 必须记录：model ID、revision、checkpoint SHA-256、fairseq/upstream commit 或目录 hash、环境版本。作者包内三个 0 字节 `emotion2vec_base.pt` 只记录为不可用占位，不得作为来源。

禁止回退到 plus-large 或 FunASR wrapper。

- [ ] **Step 4: Generate only IEMOCAP base frames and word blocks**

输出：

```text
data/contracts/emofilm_v1/emotion2vec_base_frames/iemocap/
data/contracts/emofilm_v1/word_blocks/iemocap/
```

复用现有 IEMOCAP TextGrid。每个 rejected 写 `utt_id/reason/speaker/emotion/original_split`。

- [ ] **Step 5: Generate author WordSequence pseudo-labels**

使用作者 `best_model.pth` strict load；输出逐词预测和最终 tagged manifest。验证 plain text 归一化后完整覆盖，不允许静默丢词。

- [ ] **Step 6: Build ESD Global Label and combined manifests**

ESD 不经过 WordSequence 覆盖。按冻结的 train/cv ID 集合分别生成 `tagged/train.jsonl` 和 `tagged/cv.jsonl`。

- [ ] **Step 7: Build train/cv src and parquet directly**

先创建独立 `src/train`、`src/cv`，再分别调用 parquet packer。异步结果必须 `.get()`；任一 shard 失败则不写最终 `data.list`。

- [ ] **Step 8: Run data contract tests and inventory report**

```bash
pytest tests/test_emofilm_data_contract.py -q
```

输出样本数、数据源、标签来源、rejected 分布、train/cv ID hash、parquet SHA-256 和字节数。

## Task 5: 训练唯一 5-epoch 模型

**Authorization gate:** GPU 训练和大型 checkpoint 写入前再次确认运行命令、GPU、预计时间和输出目录。

**参考正式训练命令（仅替换已授权的 GPU ID）：**

```bash
cd "$(git rev-parse --show-toplevel)"
CUDA_VISIBLE_DEVICES=<authorized_gpu> torchrun --standalone --nproc_per_node=1 \
  cosyvoice/bin/train_emo.py \
  --train_engine torch_ddp \
  --model llm \
  --config conf/emo_film.yaml \
  --train_data data/contracts/emofilm_v1/splits/train/parquet/data.list \
  --cv_data data/contracts/emofilm_v1/splits/cv/parquet/data.list \
  --qwen_pretrain_path pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN \
  --checkpoint pretrained_models/CosyVoice2-0.5B/llm.pt \
  --contract_dir data/contracts/emofilm_v1 \
  --model_dir exp/emofilm_v1 \
  --tensorboard_dir exp/emofilm_v1/tb \
  --num_workers 2 \
  --prefetch 2
```

- [ ] **Step 1: Run training preflight without optimizer steps**

检查一批 train/cv：张量 shape、loss finite、trainable 参数集合、optimizer 参数 ID、token/mel ratio 和 static batch 4；同时确认 `llm.pt` 存在、可按 base policy 加载，且不会进入 `fresh` 分支。

- [ ] **Step 2: Create immutable training identity**

目录建议：

```text
exp/emofilm_v1/
├── init.pt
├── latest.pt
├── resolved.yaml
├── train_identity.json
├── train.log
└── tb/
```

identity 包含代码 commit/worktree hash、数据合同 hash、base checkpoint hash、seed、GPU/库版本和命令。
其中 `base_checkpoint` 必须非空，解析后必须指向 `pretrained_models/CosyVoice2-0.5B/llm.pt`，并记录其 SHA-256；`checkpoint_role` 必须为 `base`。

- [ ] **Step 3: Train 5 epochs**

每 epoch 只原子覆盖 `latest.pt`。不得生成永久 `epoch_N_whole.pt`。

- [ ] **Step 4: Finalize checkpoint**

训练成功后原子重命名 `latest.pt → final.pt`，计算参数 SHA-256；确认目录只有 `init.pt` 和 `final.pt` 两份长期模型权重。

- [ ] **Step 5: Verify trained load and minimal forward**

strict load final；对 train/cv 各一个 batch 前向，所有 loss finite。

## Task 6: 完成 `3→30→本地固定2500条` 代理验收

**Authorization gate:** 全量生成前再次确认 6 GPU 命令、输出目录和预计占用。

生成前硬门：读取 `train_identity.json` 和 `resolved.yaml`，确认 base checkpoint 非空且 SHA-256 与本地 `llm.pt` 一致、`checkpoint_role=base`、`checkpoint` 非空；任一不满足则禁止 smoke、confirmation 和全量评测。

- [ ] **Step 1: Run mainline tests**

```bash
pytest \
  tests/test_emofilm_data_contract.py \
  tests/test_emofilm_training_contract.py \
  tests/test_emofilm_inference_contract.py \
  tests/test_emofilm_runtime_contract.py \
  tests/test_emofilm_e2e_smoke.py -q
```

- [ ] **Step 2: Run fixed three-sample smoke**

ESD、FEDD-A、FEDD-B 各一条。要求音频、manifest、运行身份完整；不设置质量数值阈值。

- [ ] **Step 3: Run fixed 30-sample confirmation**

每分区 10 条。检查空输出、缺文件、文本明显遗漏和配对失败；结果只作为进入全量的技术确认。

- [ ] **Step 4: Run local fixed 2500 samples**

使用冻结的 ESD 1500、FEDD-A 500、FEDD-B 500。多 GPU 分片布局写入 identity；不要求与小样本逐 token 一致。

- [ ] **Step 5: Evaluate fixed WAVs**

按分区运行 `emofilm-eval-v2`，输出 technical completeness、WER ratio/percent、Emo-SIM、cosine DTW 和逐样本 rows。

- [ ] **Step 6: Compare without absolute thresholds**

与旧 local final 和 local identity 做相同分区对照。报告改善、恶化和混合变化，不自动产生 pass/fail、不续训、不建矩阵。

## Task 7: 收口主线、更新文档并提出第二阶段清理

- [ ] **Step 1: Run reachability review**

从以下入口建立导入/调用图：

```text
build_emofilm_contract.py
train_emo.py
inference_emo_film.py / run_inference_parallel.py
eval_emo_film.py
mainline tests
```

列出仍不可达的 tools/tests/config；若未在第一阶段删除，形成新的精确危险清单。

- [ ] **Step 2: Run verification**

```bash
pytest tests/test_emofilm_*.py -q
git diff --check
rg -n "TBD|TODO|10.*58|完整闭合|作者官方评测|per_emo_accuracy" \
  docs/superpowers/specs/2026-07-16-emofilm-author-source-regression-design.md \
  docs/superpowers/plans/2026-07-16-emofilm-author-source-regression.md \
  docs/superpowers/reports/2026-07-16-emofilm-author-local-alignment-audit.md \
  .scratch/emofilm-author-source-regression/spec.md
```

- [ ] **Step 3: Write final report and update inventory**

最终报告必须分开陈述：作者源码语义、本地补全合同、工程 hardening、不可闭合项和相对评测结果。

- [ ] **Step 4: Propose second-stage cleanup**

只在新版成功后列出旧 full parquet、v6 全量 WAV、旧 init/final 等候选。不得在同一步自动删除。

- [ ] **Step 5: Stop for user decision**

向用户提供：保留、压缩离线或删除三种选项，以及每项空间收益和失去的重算能力。

## 计划自检

- 作者源码事实、本地补全和工程 hardening 已分开；
- 不再把 CPU provider、完整 prefix、整块 shuffle、逐样本 seed 列为必修；
- EOS 重采样和 RAS fallback 保留，因为它们改变采样分布；
- 不训练 WordSequenceModel，不运行 per-emotion accuracy；
- 无消融、矩阵、自动延长和绝对数值失败阈值；
- 所有删除、Git 修改、下载、训练和全量生成均有显式授权门禁。
