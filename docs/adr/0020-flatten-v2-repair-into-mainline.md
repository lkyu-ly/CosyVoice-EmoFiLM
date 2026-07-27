/

---
status: accepted
amends: [0019]
---
# 扁平化整合 EmoFiLM v2 修复到 main 主线（取代 v2 并行副本策略）

EmoFiLM v2 细粒度控制修复（ADR-0019）此前采用"新增 `*_v2.py` 版本化副本、v1 只读"的策略交付 12 个代码票据。该策略在实践中不可持续：全部 v2 源码（27 个路径）停留在 git 之外（untracked），公开推理 cli 仍固定 v1 配置导致 v2 无法端到端推理（审查 Critical），运行身份的 patch bundle 因 `git diff --binary HEAD` 漏未跟踪文件而不可重建。本 ADR 决定：把 v2 修复**原地整合进 main 主线代码**，删除所有 v2 并行副本，以 git 首次 commit `9c6d84b` 作为 v1 基线锚点——git 本身即 v1 备份，不再需要第二套文件副本。

## 背景

- 仓库唯一 commit `9c6d84b`（"baseline: establish verified EmoFiLM v1 canonical repository"，无父根）即 v1 基线锚点。任一文件可 `git checkout 9c6d84b -- <path>` 单文件回退，或整树回到 v1 干净基线。
- 这是论文代码而非长期演化的产品：已跑通的 v1 实验代码已固定在首次 commit，扁平化管理优于维护 v1/v2 双套代码。
- 本轮工作的审查（`docs/reports/2026-07-24-emofilm-v1-impact-audit.md`）确认 v1 基线**不受影响、仍有效、科学**，不需要任何 v1 数据/模型重建。

## 决策

1. **核心文件原地替换**。v2 修复逻辑合并进对应 v1 原文件并删除 `*_v2` 副本；全新模块去掉 `v2`/`_v2` 后缀作为正式模块；v1 `eval/eval_emo_film.py`（整句整体质量评测）保留为基线，v2 局部 span/triplet 评测作为互补新增模块（命名在重构时定，避免与 v1 eval 冲突）。主要映射：
   - `llm_emotion_v2.py` → `llm_emotion.py`；`train_utils_emo_v2.py` → `train_utils_emo.py`；`build_emofilm_v2_contract.py` → `build_emofilm_contract.py`；`generate_v2_tagged_jsonl.py` → `generate_tagged_jsonl.py`；`emo_film_v2.yaml` → `emo_film.yaml`；`write_emofilm_v2_identity.py` → `write_emofilm_run_identity.py`（已部分合并）。
   - `span_align_v2.py` → `span_align.py`；`acoustic_evaluators.py`、`triplet_eval.py`（无后缀保留）。
   - 类名 `Qwen2LM_EmotionV2` → `Qwen2LM_Emotion`（替换 v1 同名类）。
2. **重构不留旧遗留（普适原则）**。每次修复/改进都是重构任务，旧代码仅存于 git 历史。合同原语库重构为**单一活跃权威**（schema_version=2，合同名 `emofilm`），删除 v1 合同代码遗留；v1 冻结产物的身份（`contract_name=emofilm_v1` + contract_hash）靠冻结数据 + git 锚点解析，活跃代码不维护 v1 校验路径。
3. **禁止用文件内容哈希标定文件**。删除所有 `_V1_BASELINES` 类源码 md5/sha256 断言与"v1 文件未修改"测试段。文本子串锁**反转语义**（从"v1 保留反模式"改为"v1 反模式已删除"，断言 `emotion_classifier`/`emo_loss_weight`/`mix_ratio` 不在活跃代码中）。签名兼容锁保留（v1 入口签名仍兼容）。三类哈希用途的边界：
   - 源码文件内容哈希锁 → **禁止**。
   - 产物完整性哈希（wav_sha256 比对）→ **禁止用于 skip/safe-resume 决策**；safe-resume 仅加 `os.path.isfile` 存在性检查 + 既有逐条身份指纹（checkpoint/control/prompt/decode_config/source）。
   - `GenerationRow.wav_sha256` 字段 → **移除**（本质是给 WAV 贴哈希标签，违反原则）；产物身份改用 `wav_path` + 结构化身份字段。v1 冻结产物不受影响（其身份用 emofilm_v1 contract_hash，无 wav_sha256 依赖）。
   - git commit SHA（`9c6d84b`）作为基线锚点版本引用 → **允许**（git 原生版本句柄，非文件内容哈希）。
4. **v1 只读语义重定义**。"v1 只读"从"工作树文件字节不变"重定义为"v1 基线锚定 git commit `9c6d84b`，活跃代码可演化，v1 实验产物磁盘冻结只读"。修订 ADR-0019 第 8/25 行的"一律只读/不得原地覆盖"措辞。ADR 0001-0018 是已发生的历史决策记录，**仍冻结不动**。
5. **执行结构：重构先行 + 逻辑修复并行**。第一步扁平化重构（合并/删副本/去后缀/重定位测试锁/修 cli 接线/修 ADR），顺带解决审查 Critical（cli 接线）与 #4（patch bundle——重构后 v2 文件不再 untracked，提交后走干净 source_revision 路径）。第二步在单一代码基上并行修各模块内部缺陷（#2 合同校准字段、#3 optimizer 顶层 lr 死字段、#5 FEDD 配对内嵌身份、#6 三元组 prompt_row_ref + 注释代码不符、#7 safe-resume isfile）。
6. **门禁分层处理**。本次顺带修**代码层门禁**：`eval/eval_emofilm_v2.py`（→ 重命名后）补 `if __name__ == "__main__": main()` 入口。**数据/资产层门禁延后**（仍 Out of Scope，属 13/14 GPU 实验运行）：用已有 `author_best_model.pth`（SHA-256 `a4b373…afe13`，与作者随包逐字节相同）重生真实 IEMOCAP span 产物；接入并校准独立 emotion/arousal evaluator。两门禁的真实状态见下"13/14 门禁澄清"。
7. **git 提交由用户在完成后决定**。本轮不主动 commit。扁平化重构 + 逻辑修复全部完成、测试全过后，由用户决定是否提交。#4（patch bundle）的彻底解决依赖提交：提交后 worktree clean，走 source_revision 路径，不再需要 patch bundle；未提交时 patch bundle 仍漏未跟踪文件（但重构后 v2 文件已合并/删除，untracked 问题大幅减弱）。

## 13/14 门禁澄清（据 2026-07-24 两份核查报告）

- **门禁 1（IEMOCAP）实为 checkpoint 选错，不是"缺 checkpoint"**：v2 ticket 02 的示例误用历史 `checkpoints/word_sequence_model/best.pt`（1024d/5类/1D arousal），strict-load 失败。正确的作者/canonical 词级链路是 `emotion2vec-base` 768d/50Hz + 作者 `WordSequenceModel`（5 类/3D VAD），checkpoint 为已存在的 `author_best_model.pth`。当前 v2 IEMOCAP manifest 18 行为 StubPredictor 产物，不能作训练真值；后续数据运行是用已有作者 checkpoint 从 v1 768d word blocks 重生成真实 v2 span——不需下载/恢复/重建/重训。
- **门禁 2（evaluator）措辞校正但仍未满足**：emotion2vec 非纯 utterance-level（有 768d/1024d 帧特征），缺的是独立、经校准、与 IEMOCAP 弱监督可区分的 emotion/arousal 判别器；当前 `Emotion2Vec*Evaluator` 把裸 head 施加到未经 WordSequenceModel 前向的单帧特征，超训练分布。门禁在"正式独立局部控制结论"含义下仍**未满足**，不能用 v1 整体评测替代。
- **v1 基线有效性边界**：v1 是有效的整体质量基线（WER/Emo-SIM/DTW），适合与 v2 同口径比较；但不得表述为"v1 已独立证明细粒度词级控制"。

## 后果

- 活跃代码库变为单一权威，无 v1/v2 并行副本；v1 旧逻辑仅在 `git checkout 9c6d84b` 可达。
- 10 个 v2 测试文件的只读锁需重定位（哈希锁删除、文本锁反转、签名锁保留）；测试文件本身去 v2 后缀（命名在重构时定，避免与现有 v1 测试冲突）。
- 公开推理 cli（`model_emo.llm_job`、`cosyvoice_emo`、`inference_emo_film`）同步改调用契约，删 v1 死参数（`prompt_text`/`prompt_emotion_ids`/`prompt_intensity_ids` 等），接合并后的 `Qwen2LM_Emotion` 单流协议。
- 合同原语库合并后需同时被生成/评测/身份工具引用；重命名/合并需同步所有 import。
- 提交前 `tools/write_emofilm_run_identity.py` 的 `_save_patch_bundle` 仍是临时脏工作树回退手段；提交后该路径退居二线（source_revision 优先）。
