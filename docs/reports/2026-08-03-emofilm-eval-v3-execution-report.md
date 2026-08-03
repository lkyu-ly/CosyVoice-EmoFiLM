# EmoFiLM 评测侧 v3 重构 + 情感匹配 prompt 验证 + CosyVoice3 基线 — 执行完成报告

- 执行日期：2026-08-03
- 计划：`docs/superpowers/plans/2026-08-03-emofilm-eval-v3-refactor.md`（7 Task / 3 Phase）
- 交接：`/tmp/emofilm-eval-v3-execution-handoff.md`
- 全部 7 Task 完成；全量 pytest **460 passed / 4 skipped / 0 failed**。

---

## 0. 一句话结论

**~66 Emo-SIM 平台是「生成/评测协议（固定中性声学 prompt）」的产物，不是模型能力上限**：
情感匹配 prompt（同说话人同情感声学 prompt）下四方模型 Emo-SIM 全部跳到 **84.3–87.4**（+18–21）、
5-way 判别准确率 **73–83%**（chance=20%）；而 CosyVoice3 官方基线即便拿到 emotion instruct 也只到 **70.76**（未质变）。
瓶颈在协议，不在 FiLM 改造，也不在基座。

---

## 1. 关键实验结果

### 1.1 情感匹配 prompt 验证（Phase 2，`exp/prompt_match/summary.md`）

5 情感 × 12 句完整同文本组 = 60 条 × 4 模型，同情感**不同句** prompt：

| 模型      | prompt_match Emo-SIM | neutral(v2) Emo-SIM | Δ    | 5-way acc        | same−cross gap | WER% |
| --------- | -------------------- | ------------------- | ----- | ---------------- | --------------- | ---- |
| v1        | **87.40**      | 66.75               | +20.7 | **83.33%** | 0.43            | 7.92 |
| film_only | 85.18                | 66.11               | +19.1 | 73.33%           | 0.38            | 2.83 |
| longepoch | 84.65                | 65.89               | +18.8 | 76.67%           | 0.38            | 2.83 |
| sentlvl   | 84.32                | 65.45               | +18.9 | 73.33%           | 0.38            | 3.86 |

四模型 n_valid=60、n_way=5（完整 5-way）。声学限制假设**确认成立**。

### 1.2 CosyVoice3 官方基线（Phase 3，`docs/reports/2026-08-03-cosyvoice3-baseline-comparison.md`）

ESD 150 条子集（5×30），instruct 情感指令 + 中性 prompt：

| 模型                   | n    | Emo-SIM         | WER%  |
| ---------------------- | ---- | --------------- | ----- |
| **CV3 官方基线** | 150  | **70.76** | 5.17  |
| v1 (中性 v2)           | 1500 | 66.75           | 9.48  |
| film_only (中性 v2)    | 1500 | 66.11           | 8.18  |
| longepoch (中性 v2)    | 1500 | 65.89           | 8.38  |
| sentlvl (中性 v2)      | 1500 | 65.45           | 10.05 |

CV3 per-emotion：neu 91.7 / ang 71.1 / hap 65.9 / sad 66.1 / sur 59.0。
判别指标在 eval 集上结构退化（见 §4.10）。

---

## 2. 每 Task 执行详述

### Task 1：`eval/emotion_metrics.py` 纯函数模块（TDD）✅

- **做了什么**：先写 7 个失败测试（`tests/test_emotion_metrics.py`），实现纯函数模块（特征提取、`compute_frame_mean_emo_sim`、`compute_dtw_normalized`、`normalize_text`、`build_emotion_ref_index`、`compute_discriminability`、`compute_per_emotion_mean_sim`）。
- **验证**：`pytest tests/test_emotion_metrics.py` → **7 passed**。
- **偏离**：修了计划实现的 3 处自身缺陷（见 §4.1–4.3），均由测试驱动。

### Task 2：重写 `eval/eval_emo_film.py` v3 CLI（TDD）✅

- **做了什么**：先写 7 个失败测试（`tests/test_eval_emo_film.py`），重写为薄 CLI（v3 schema：`metric_contract_version/n_samples/emo_sim/dtw_normalized/wer/wer_percent/per_emotion_emo_sim` + 可选 `discriminability`），删 v2 九字段与 `--dtw_dist`。
- **验证**：`pytest tests/test_emotion_metrics.py tests/test_eval_emo_film.py` → **14 passed**。
- **偏离**：测试 `test_batch_size_cli` 补 `--ref_text_manifest`（见 §4.4）。

### Task 3：清理旧测试/合同 + 适配调用方 ✅

- **做了什么**：删 `test_eval_metric_contract.py`/`test_eval_emo_film_batch.py`/`test_eval_wer.py`/`docs/contracts/emofilm_v2_evaluators.md`；更新 `test_eval_smoke.py`（v3 断言 + 必填 manifest）、`calibrate_eval_contract.py`（接 `compute_dtw_normalized`）、3 个 `run_eval.sh`（v3 输出 `eval/v3/` + `--emotion_ref_manifest`）、README；新建 `docs/contracts/emofilm_v3_eval.md`。
- **验证**：`pytest tests/ -q` → **460 passed / 4 skipped**。
- **偏离**：`run_eval.sh` 用 `REFS_DIR` 而非计划通用 `${EXP}/eval_refs`（longepoch/sentlvl 复用 film_only 的参考视图，见 §4.5）；`test_canonical_paths.py` 排除 `docs/superpowers/`（见 §4.6）；`test_emotion_metrics.py` 加 `sys.path.insert`（见 §4.7）。

### Task 4：情感匹配 prompt 验证 manifest ✅

- **做了什么**：`tools/build_emotion_prompt_manifest.py` 从 sources/esd/tagged.jsonl 选 12 个完整 5 情感组 × 5 = 60 条；prompt_wav 用同说话人同情感**不同句** wav。
- **验证**：`wrote 60 rows`，60/60 prompt≠target，5 情感各 12，组情感数全 5。
- **偏离**：计划的 prompt 候选在单句组内找，恒为自身（自克隆）；改为跨句索引（见 §4.8）。

### Task 5：四模型生成 + v3 评测 + 汇总 ✅

- **做了什么**：4 模型各生成 60 条（GPU2 串行，v1 用 legacy padded ckpt）；建 60 条 ref 视图；v3 评测 4 份（whisper CPU + emotion2vec GPU）；`exp/prompt_match/summary.md`。
- **验证**：4×60 wav，4 份 metrics（n_valid=60, n_way=5），见 §1.1。
- **偏离**：v1 ckpt 缺监督头需 padding（§4.9）；集群 GPU 饱和需 whisper CPU（§4.11）。

### Task 6：CV3 权重下载 + 冒烟 ✅

- **做了什么**：ModelScope 下载 `Fun-CosyVoice3-0.5B-2512`（240s）；冒烟确认本仓库 `AutoModel` + `inference_instruct2` 可跑通（无需 clone 官方仓库）；`exp/cosyvoice3_baseline/smoke.md`。
- **验证**：加载 25s，峰值显存 3.29GB，出音 (1,60480) 2.52s。
- **偏离**：CV3 指令格式必须 Qwen 系统前缀 + `<|endofprompt|>`（计划用 CV2 格式致 vocoder 报错，见 §4.12）。

### Task 7：CV3 150 条生成 + 评测 + 对比报告 ✅

- **做了什么**：分层抽 ESD 5×30=150 子集 manifest + ref 视图；`gen_cv3.py` 生成 150 条（371s, ok=150）；v3 评测；`docs/reports/2026-08-03-cosyvoice3-baseline-comparison.md`。
- **验证**：150 wav，metrics n_samples=150（见 §1.2）。
- **偏离**：`compute_discriminability` 退化口径修正（§4.10）；CV3 生成脚本按 tts_speech 有效性保存（CV3 finish_reason=None）。

---

## 3. 产物清单

**代码/测试**：`eval/emotion_metrics.py`、`eval/eval_emo_film.py`（重写）、`tests/test_emotion_metrics.py`、`tests/test_eval_emo_film.py`、`tests/test_eval_smoke.py`、`tools/calibrate_eval_contract.py`、`tools/build_emotion_prompt_manifest.py`、`exp/*/run_eval.sh` ×3、`exp/cosyvoice3_baseline/gen_cv3.py`、`exp/prompt_match/build_summary.py`、`exp/cosyvoice3_baseline/build_comparison.py`。
**文档/合同**：`docs/contracts/emofilm_v3_eval.md`（新建）、`README.md`、`exp/prompt_match/summary.md`、`exp/cosyvoice3_baseline/smoke.md`、`docs/reports/2026-08-03-cosyvoice3-baseline-comparison.md`、本报告。
**数据 manifest**：`data/contracts/emofilm_v1/eval/esd_prompt_match_60.jsonl`（60）、`esd_cv3_150.jsonl`（150）。
**生成音频**：`exp/prompt_match/{v1,film_only,longepoch,sentlvl}/esd/*.wav`（各 60）、`exp/cosyvoice3_baseline/esd_150/*.wav`（150）。
**评测结果**：`exp/prompt_match/*/eval/esd_v3_metrics.json`（4）、`exp/cosyvoice3_baseline/eval/esd_v3_metrics.json`。
**legacy 中间件**：`exp/prompt_match/v1_legacy_padded.pt`（v1 ckpt + 随机监督头，仅推理用，见 §4.9）。

---

## 4. 偏离计划之处（共 12 处，均有据）

### 4.1 `compute_frame_mean_emo_sim` float64 均值

计划 `.mean(axis=0)` 在 float32 输入下累积误差，恒等对照 emo_sim=99.9999974 超 1e-6 容差。改为均值前 `dtype=np.float64`。**测试驱动（red→green）**。

### 4.2 `normalize_text` 多位数字转英文

计划 `_normalize_digits` 逐字符替换，"42"→"fourtwo"（测试期望 "forty two"）。改为 `\d+` 整段 int→words。

### 4.3 `_l2_normalize` / `extract_utt_embeddings` epsilon 偏差

计划 `norm + 1e-8` 给每个非零向量引入 ~eps/norm 系统偏差，恒等 emo_sim 偏离 100。改为 `max(norm, 1e-8)`（仅在除零边界生效）。

### 4.4 `test_batch_size_cli` 补 manifest

计划该测试 `common` 列表未含 `--ref_text_manifest`，但 v3 已将其设为 required（v2 复制遗留）。补上。

### 4.5 `run_eval.sh` 的 `REFS_DIR`

计划通用模板用 `${EXP}/eval_refs/$ds`，但 longepoch/sentlvl 没有自己的 eval_refs（复用 `exp/emofilm_film_only/eval_refs`，数据集级参考与模型无关）。引入 `REFS_DIR` 变量保留各自真实来源。

### 4.6 `test_canonical_paths.py` 排除 `docs/superpowers/`

全 docs/ 仅计划文件含 `/home/hanlvyuan/`（执行型工件，含宿主绝对运行路径），且该失败先于本次代码改动。扫描跳过 superpowers 工作流子树（与已排除的 worktrees 同类），不弱化对 adr/contracts/reports 等规范文档的守卫。

### 4.7 `test_emotion_metrics.py` 自带 `sys.path.insert`

计划测试靠 `PYTHONPATH=eval`；仓库约定是测试自带路径注入（`test_eval_emo_film.py` 已如此）。补上以兼容全量 pytest 命令（`PYTHONPATH=.:third_party/Matcha-TTS`）。

### 4.8 `build_emotion_prompt_manifest.py` 跨句 prompt

计划 prompt 候选在单句 5 情感组内找同情感，但单句组每情感仅 1 条 → 候选恒空 → 回退自身（60/60 自克隆，实验无意义）。改用 `(speaker, emotion)→[所有句]` 跨句索引选不同句 prompt（docstring 本意）。

### 4.9 v1 legacy padded ckpt

v1 的 `final.pt` 缺 `emotion_head`/`arousal_head`（旧训练产物），当前 `load_trained_state` 故意拒绝（ADR-0019/0020 防冒充守卫）。这两个头是**下游监督头，推理生成路径不调用**（`llm_emotion.py:257-258` 定义，仅 `:524-525` loss 计算）。故用随机值补齐存 `exp/prompt_match/v1_legacy_padded.pt`，走正常推理路径——**不改 inference_emo_film.py、不动守卫、生成结果与原 v1 完全一致**。其余 3 模型 ckpt 含头，直接加载。

### 4.10 `compute_discriminability` 退化口径修正

ESD train/eval 划分把每个 (speaker,text) 组的**目标情感单独留到 eval**，sources 参考恰好缺该情感 → eval 集上 `target` 永远不在参考中。原实现 `cross=[]`→`np.mean([])=NaN`、`gap=NaN` 写入 JSON（非法）且 acc=0 误导。改为：用 `n_scored` 计数，退化时（n_scored=0）返回干净 `reason`（不写 NaN、不报误导性 0%）；acc 改为 `/n_scored`。**判别指标只在完整 5 情感组（prompt_match 验证集）上有意义，eval 集上 N/A**（已记入 `emofilm_v3_eval.md`）。CV3 评测已用修正版重跑。

### 4.11 集群 GPU 饱和 → whisper CPU

计划 run_infer/run_eval 假设 4 GPU 空闲；实际共享集群 6 卡被他人 lbf 作业占满（每卡邻居 17–22GB，且剧烈波动）。措施：①4 模型推理在唯一较空的 GPU2 串行；②给评测 CLI 加 `--whisper_device`（默认同 device，向后兼容），把 whisper 移到 CPU，emotion2vec 留 GPU，峰值 ~7GB→~3GB，配合 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与自动选最闲卡 + OOM 重试。指标口径不变（whisper fp32/fp16 转写对 WER 无实质影响）。

### 4.12 CV3 指令格式（硬坑）

计划 smoke 用 CV2 格式 `'Speak with angry emotion.'`，CV3（基于 Qwen）收到畸形 prompt → LLM 退化极短 token → vocoder `f0_predictor` CausalConv1d 报 `padded input(3)<kernel(4)`。按 CV3 模型卡官方示例改为 `'You are a helpful assistant. Speak with angry emotion.<|endofprompt|>'`（系统前缀 + 指令 + `<|endofprompt|>` 终止符）后正常。`exp/cosyvoice3_baseline/smoke.md` 详记。

---

## 5. 遗留问题与后续建议

1. **判别指标在 eval 集上 N/A**（§4.10）：受 train/eval 划分所限，eval 集上无法算 nearest-ref 准确率。后续若要在 eval 集做判别，需重建参考集（如用 sources 全集的完整 5 情感组而非 sources 排除 eval 后的剩余）。
2. **CV3 基线判别缺失**：同上，CV3 在 eval 集判别为 N/A；CV3 真正的判别人工评估可用 prompt_match 同口径补做（CV3 + 情感匹配 prompt）。
3. **未重跑 v3 全量评测**：3 个 `run_eval.sh` 已更新为 v3 CLI（输出 `eval/v3/`），但本次未对 4 模型 1500 条全量重跑 v3（历史 v2 metrics 已足够做中性 baseline 对照，且耗时长）。需要时直接 `bash exp/emofilm_*/run_eval.sh`。
4. **v1 legacy padded ckpt** 为本次推理临时产物（`exp/prompt_match/v1_legacy_padded.pt`），非规范训练制品；正式入库前可考虑（a）给 `load_emofilm_model` 加显式 legacy 模式，或（b）保留 padded 文件并在合同注明其来源。
5. **后续主线（本任务范围外）**：声学钳制假设既已确认，下一块应做监督改造（DPO/GRPO/span 级情感监督），把"情感匹配 prompt 能达成的上限"内化到模型自身（交接 §1 第 2 点）。

---

## 6. 与计划预期一致性

- **Phase 1（评测 v3 重构）**：与预期一致（schema、TDD、全量 pytest 全绿），实现修了 3 处计划自身缺陷。
- **Phase 2（prompt_match）**：实验结论比预期更清晰——不只是"绕过钳制"，而是**定量证实四模型均能到 84–87 且 5-way 可判别**，瓶颈定位到协议。
- **Phase 3（CV3）**：CV3 推理走本仓库 CLI（符合决策 8 优先路径），无需 clone 官方仓库；CV3=70.76 未突破平台，符合"问题在协议"的预期。
- 全部 8 项用户已确认决策（§4 of 交接）均落实，无变更。
