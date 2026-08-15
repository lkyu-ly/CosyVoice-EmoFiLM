# EmoFiLM 调整算法前收尾（判别增强 + v3 基线 + v1 精简）— 执行完成报告

- 执行日期：2026-08-04
- 执行依据（唯一）：`docs/superpowers/plans/2026-08-04-emofilm-eval-prereqs.md`（5 Task）
- 交接文件：`/tmp/emofilm-eval-prereqs-execution-handoff.md`
- 全部 5 Task 完成；全量 pytest **462 passed / 4 skipped / 0 failed**。

> 路径约定：本报告以 `$WR`（仓库根）或相对路径表示，不出现字面宿主目录（受 `tests/test_canonical_paths.py` 守卫）。

---

## 0. 一句话结论

三件收尾全部落地：① 主线评测 eval 集判别指标从 N/A 变为**可算**（ESD n_scored=1422、FEDD_B n_scored=427）；② longepoch/sentlvl 两模型 v3 全量中性基线已跑（ESD Emo-SIM 65.89/65.45，与 v2 完全一致）；③ v1 兼容残留精简（删 padded ckpt + robust 脚本、收窄注释、清理 docstring）。下一步主线（R2 监督改造）的前置清零。

---

## 1. 每 Task 执行详述

### Task 1：判别指标增强（核心 TDD）✅

- **做了什么**：
  - 先写 2 个失败测试（`tests/test_emotion_metrics.py`：`test_discriminability_merges_reference_wav`、`test_discriminability_skips_rows_without_enough_refs`）+ `_MapEmoModel` fake；导入列表补 `compute_discriminability`。
  - 确认 RED：测试 1 `assert 0==1`（旧实现不合并 reference_wav → target 不在候选 → n_scored=0）；测试 2 `KeyError: 'n_skipped'`（旧退化分支缺键）。
  - 整体重构 `eval/emotion_metrics.py::compute_discriminability`（非补丁）：新增模块级 `EMOTIONS` 常量 + `_merged_refs` 辅助（合并 sources 索引与 eval 行 `reference_wav`）；单遍预处理（合并→过滤候选≥3→批量提嵌入→单遍判别→聚合）；所有返回分支统一 schema（退化时 `same/cross/gap=None` + `reason`，不写 NaN、不缺键）。
  - `eval/eval_emo_film.py::load_manifest` 增加 `reference_wav` 字段（`os.path.abspath`，绝对路径或 None）；docstring 同步更新。
  - 更新 `docs/contracts/emofilm_v3_eval.md`：表格补 `n_skipped/n_scored`，"判别指标口径"小节改写（删除"结构上不可计算/N/A"旧表述，改为 reference_wav 合并口径 + ESD 1422/1500 可算、FEDD_A n=0 / FEDD_B n=427 诚实口径、原始余弦 vs 百分比说明）。
- **验证**：`pytest tests/test_emotion_metrics.py tests/test_eval_emo_film.py -q` → **16 passed**（原 14 + 新增 2）。
- **偏离**：见 §3.1（测试断言口径修正——计划自身缺陷）。
- **真实有效性确认**（数据层 + 真实评测双证）：
  - 数据层：ESD eval 1500 条合并后 1422 条候选≥3（5-way×851 / 4-way×405 / 3-way×166），reference_wav 全部可解析（缺文件=0）。
  - 调用链：`main()`→`eval_rows=[{utt_id,**v}...]`（`**v` 自动携带 reference_wav）→`run_evaluation` 按 utt_ids 重排 → `compute_discriminability`。

### Task 2：v1 兼容精简 ✅

- **做了什么**（顺序调整：先脚本重构再删文件，遵循交接坑位 7 避免悬空引用）：
  - 重构 `exp/prompt_match/run_infer.sh`：删 v1 分支与写死 `GPU=2`，改 `GPU` 环境变量；保留 film_only/longepoch/sentlvl 三模型。
  - 重构 `exp/prompt_match/run_eval.sh`：删 v1 循环、固定 GPU、OOM 重试逻辑；改 `GPU`/`WHISPER_DEV` 环境变量。
  - 删除 `exp/prompt_match/run_eval_robust.sh`（OOM 重试补丁，资源决策改为现场传参，无保留价值）。
  - 删除 `exp/prompt_match/v1_legacy_padded.pt`（2.4GB 临时产物；v1 不再做代码级支持）。
  - 收窄 `cosyvoice/utils/emo_checkpoint.py` 的 `TRAINED_ALLOWED_MISSING_PREFIXES` 注释（明确仅 `emotion_classifier.`、longepoch 依赖、sentlvl 应含；常量本体不变）。
  - 清理 `tools/inference_emo_film.py` docstring 示例（`exp/emofilm_v1/...` → `exp/emofilm_film_only_longepoch/...`）。
  - 修复 `docs/reports/2026-08-03-emofilm-eval-v3-execution-report.md:132` 字面宿主路径（消除 test_canonical_paths 失败）。
- **验证**：`pytest tests/ -q` → 462 passed（test_canonical_paths 含在内）；`rg run_eval_robust` → CLEAN；`v1_legacy_padded` 仅 2026-08-03 历史报告 3 处（历史记录，可接受）。

### Task 3：run_eval.sh 精简更新 ✅

- **做了什么**：`exp/emofilm_film_only_longepoch/run_eval.sh`、`exp/emofilm_sentlvl/run_eval.sh` 替换为简洁模板（`GPU`/`WHISPER_DEV` 环境变量、串行 3 数据集、不写自动选卡代码；仅 `EXP` 不同，复用 `exp/emofilm_film_only/eval_refs` 数据集级参考视图）。
- **验证**：`bash -n` 两个脚本均 OK（顺带验证 prompt_match 两脚本亦 OK）。

### Task 4：跑 v3 全量中性基线 ✅

- **做了什么**：`nvidia-smi` 确认 GPU 0/2/3/4 空闲 → longepoch@GPU0、sentlvl@GPU2 **并行**（计划允许"另选空闲卡"；两模型独立无共享状态）→ 各跑 esd(1500)/fedd_a(500)/fedd_b(500)；写 `docs/reports/2026-08-04-v3-neutral-baseline.md`。
- **验证**：6 份 metrics.json 全部生成（n_samples 匹配 expected_count）；ESD 判别 `n_scored=1422`、`reason=None`（不再 N/A）。
- **耗时**：约 52 分钟（两模型并行；whisper large-v3 串行转写 1500 条是瓶颈，esd 占 ~29 分钟）。
- **偏离**：见 §3.3（FEDD_B 判别意外可算，比计划假设更乐观）、§3.5（并行而非串行）。

### Task 5：全量验证 ✅

- **验证**：`pytest tests/ -q` → **462 passed / 4 skipped / 0 failed**；`rg v1_legacy_padded`（活跃）→ 仅历史报告；新建/修改 docs 宿主路径核查 → 0。

---

## 2. 关键实验结果（v3 全量中性基线）

### 2.1 ESD（1500）

| 模型 | emo_sim | dtw_norm | wer% | 判别 n_scored | acc% | same−cross gap |
| --- | --- | --- | --- | --- | --- | --- |
| longepoch | 65.89 | 0.3409 | 7.43 | 1422 | 43.04 | 0.16 |
| sentlvl | 65.45 | 0.3453 | 9.12 | 1422 | 42.33 | 0.15 |

判别 chance≈22%，acc≈43% 显著高于 chance；n_scored=1422 精确匹配数据层验证。

### 2.2 FEDD（500×2）

- fedd_a：emo_sim 82.46/82.91，判别 n=0（reason，Part A 无同文本跨情感组）。
- fedd_b：emo_sim 64.31/63.19，**判别 n_scored=427 acc≈47–50%**（Part B 有 4-way×264 + 3-way×163 候选）。

完整指标（per_emotion、mean_sim_by_ref_emotion、n_way 分布）见 `docs/reports/2026-08-04-v3-neutral-baseline.md`。

---

## 3. 偏离计划之处（共 5 处，均有据）

### 3.1 Task 1 测试断言口径修正（计划自身缺陷）

计划测试 `same_emotion_mean` 期望 `100.0`（误用 `emo_sim` 的 ×100 口径），但计划实现用 `np.dot`（原始余弦）。核对既有 prompt_match 四模型实验数据（`exp/prompt_match/*/eval/esd_v3_metrics.json`：`same`≈0.85、`gap`≈0.38、`mean_sim_by_ref_emotion`≈0.6），**实现口径与既有实验完全一致**，是测试断言值算错。修正测试断言到原始余弦（1.0/0.0/1.0）+ 加注释防回归；`nearest_ref_acc_pct` 保持百分比。实现不变（与既有口径一致 + 余弦标准定义）。类比上轮 v3 refactor 的计划自身缺陷（§4.1/§4.3）。

### 3.2 审查报告文件缺失

交接 §4 列 `docs/superpowers/plans/2026-08-04-emofilm-eval-prereqs-review.md` 为必读，但该文件实际不存在（`plans/` 目录下仅计划本体与 next-steps-planning）。不阻塞：计划 Self-Review 部分已完整覆盖 spec 覆盖/无占位符/类型一致性/风险对策，且交接 §6.1 明确"实施计划是唯一执行依据"。

### 3.3 FEDD_B 判别意外可算（数据比假设乐观）

交接 §5 坑位 6 与计划假设"FEDD 无同文本跨情感组，判别 n=0"。实际：fedd_a 确实 n=0（符合），但 **fedd_b n_scored=427**（4-way×264 + 3-way×163）。这是数据事实比假设更乐观，非 bug；如实记入基线报告，判别信号覆盖范围比预期更广。

### 3.4 Task 2 步骤顺序调整

按交接坑位 7，先重构 `run_infer.sh`（移除 v1 引用）再删 `v1_legacy_padded.pt`，避免中间态悬空引用（计划 Step 1→1b 顺序会短暂留下对已删文件的引用）。

### 3.5 Task 4 两模型并行

计划 Step 1/2 分开写（longepoch 后 sentlvl，串行）。当前 4 张卡空闲，两模型评测独立（不同 exp/ckpt/hyp，无共享状态），用 GPU 0/2 并行缩短总时长（计划注释明确允许"另选空闲卡"）。指标口径不受影响（各进程独立加载模型到各自 GPU）。

---

## 4. 产物清单

**代码/测试**：
- `eval/emotion_metrics.py`（重构 `compute_discriminability` + `EMOTIONS` + `_merged_refs`）
- `eval/eval_emo_film.py`（`load_manifest` + `reference_wav`）
- `tests/test_emotion_metrics.py`（+2 判别测试 + `_MapEmoModel`）

**脚本**：
- `exp/prompt_match/run_infer.sh`、`run_eval.sh`（重构）；**删除** `run_eval_robust.sh`
- `exp/emofilm_film_only_longepoch/run_eval.sh`、`exp/emofilm_sentlvl/run_eval.sh`（简洁模板）

**删除**：`exp/prompt_match/v1_legacy_padded.pt`（2.4GB）

**注释/docstring**：`cosyvoice/utils/emo_checkpoint.py`（注释收窄）、`tools/inference_emo_film.py`（docstring 示例）

**文档**：`docs/contracts/emofilm_v3_eval.md`（判别口径）、`docs/reports/2026-08-03-emofilm-eval-v3-execution-report.md`（行 132 修复）、`docs/reports/2026-08-04-v3-neutral-baseline.md`（新建基线）、本报告

**评测产物**：`exp/emofilm_film_only_longepoch/eval/v3/{esd,fedd_a,fedd_b}_metrics.json` + `.log`、`exp/emofilm_sentlvl/eval/v3/{esd,fedd_a,fedd_b}_metrics.json` + `.log`（共 6 份）

---

## 5. 遗留问题与后续建议

1. **whisper 串行转写是评测瓶颈**：`transcribe_parallel` 因 whisper 线程不安全加锁串行化，ESD 1500 条转写约 29 分钟（占评测大部分时长）。本次不改（超计划范围）。若后续频繁全量评测，可考虑多进程转写或更快 whisper 后端。
2. **v1 legacy 历史提及**：`docs/reports/2026-08-03-...md` 仍含 3 处 `v1_legacy_padded.pt` 描述（上一轮历史操作的记录）。保留不篡改（历史报告完整性）；活跃代码与脚本已完全清理（rg 仅历史命中）。
3. **未 git commit**：遵循用户约束（工作树即交付物，本次未要求提交）。全部改动在工作区，可随时检阅或提交。
4. **后续主线（本任务范围外）**：判别增强既已让 eval 集判别可算，且中性平台 65 + 判别 acc 显著高于 chance 已再次确认声学钳制假设；下一块是 R2 监督改造（span 级情感监督/可训练分类器，把 prompt_match 的 84–87 上限内化到模型）。

---

## 6. 与计划预期一致性

- **Task 1（判别增强）**：与预期一致——合并 reference_wav 后 eval 集判别可算（n_scored=1422），统一 schema 消除退化分支缺键/NaN。修了 1 处计划自身测试断言缺陷（§3.1）。
- **Task 2（v1 精简）**：与预期一致——padded ckpt + robust 脚本删除、注释收窄、docstring 清理；test_canonical_paths 修复。
- **Task 3（run_eval.sh 简洁模板）**：与预期一致——环境变量 + 串行 + 不自动选卡。
- **Task 4（v3 基线）**：结论与预期一致——ESD Emo-SIM 65.89/65.45 与 v2 完全一致（口径未漂移）；判别可算且 acc 显著高于 chance。FEDD_B 判别可算是额外收获（§3.3）。
- **Task 5（全量验证）**：462 passed / 4 skipped / 0 failed，无 v1 活跃残留。
- 全部用户已确认决策（v3 仅跑 longepoch+sentlvl、v1 不做 legacy、不写自动选卡、仅判别合并加 2 测试、不主动 commit、所有新代码中文）均落实，无变更。
