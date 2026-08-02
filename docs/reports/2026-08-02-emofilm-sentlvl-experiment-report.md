# EmoFiLM 句级情感监督对照实验报告（sentlvl）

- 日期：2026-08-02
- 实验对象：加回 init 版句级 `loss_emotion` 监督（恒构造冻结探针，`emo_loss_weight=0.2`），在早停下训推评
- 基线：HEAD `a89afcd`（句级监督重构落地后）；Spec：`/tmp/emofilm-sentlvl-handoff-2026-08-02.md`
- 对照组：v1（init `9c6d84b`，有 loss_emotion）/ 5-epoch disabled / 27-epoch disabled（早停 best@21）
- 评测口径：baseline `eval/eval_emo_film.py`（WER / Emo-SIM / DTW），与三个对照同口径
- 实现依据：审查报告 `docs/reports/2026-08-02-emofilm-sentlvl-implementation-review.md` + 重构计划 `docs/superpowers/plans/2026-08-02-emofilm-sentlvl-fixes.md` + `docs/adr/0021-emofilm-input-end-sentence-supervision.md`

## 摘要

加回句级监督（原样恢复 init 版机制：恒构造冻结 `emotion_classifier` + `emo_loss_weight=0.2`），早停训练（best@14, cv_loss_tts=3.7193）→ 2500 全量推理 → baseline 评测 → 四方对比。

**核心结论：句级监督（原样恢复）未突破 Emo-SIM 平台，且损害语音质量（WER 升高）。** 这印证了交接文档 §5 先验（v1 原样句级监督未突破平台）与 ADR-0021 的已知风险（冻结随机读出器梯度无语义锚点；"文本可分"≠"声学遵从情感"）。

## 1. 背景与动机

`docs/reports/2026-08-01-emofilm-longepoch-convergence-comparison.md` 的核心发现：CV `loss_tts` 是情感/WER 的弱代理——27-epoch（更收敛）不带来 Emo-SIM 统一收益；ESD Emo-SIM ~66 跨 v1/5-epoch/27-epoch 三模型稳固 → FiLM 情感上限是**方法层面**，非训练时长可解。

用户决策（2026-08-02 handoff）：换杠杆——加回 init 版句级 `loss_emotion` 监督，做对照实验看能否突破平台。先验（handoff §5）：v1（有 loss_emotion）Emo-SIM 并不比 disabled 高（ESD 66.75 vs 66.11/65.89）。本轮以重构后的干净实现复现并量化这一对照。

## 2. 方法

### 2.1 实现（审查重构后）

第一轮实现因两个 P0（训练启动 `load_base_state` 拒绝分类器 missing、推理 `load_trained_state` strict 拒绝分类器 unexpected）+ 两个 P1（span 分支短路丢弃 input-end loss、`loss_emotion` 键双语义）被审查否定。重构采用"把可选性从拓扑层移到损失层"的干净设计：

- **恒构造冻结探针**：`emotion_classifier = Linear(llm_input_size, emotion_vocab_size)`，`requires_grad_(False)`，**不再条件创建**。`emo_loss_weight` 恒为 float 属性。模型拓扑与 checkpoint schema 不随配置漂移（训练/推理/基线同一 state_dict 键集）。
- **统一 loss 组合**：`loss = loss_tts [+ w_e·loss_emotion_span + w_i·loss_intensity] [+ emo_w·loss_emotion_input]`，两条路径可叠加，键名分离（`loss_emotion_span` / `loss_intensity` / `loss_emotion_input`）。
- **checkpoint 双向容忍**：`emotion_classifier.` 进 `ALLOWED_MISSING_PREFIXES`（base 加载）+ `TRAINED_ALLOWED_MISSING_PREFIXES`（trained 加载，兼容旧 disabled ckpt）；`emotion_head/arousal_head` trained 仍严格必填（v1 防冒充守卫）。
- **死字段**：`emo_loss_weight` 移出 `DEAD_CONFIG_KEYS`（现 `{mix_ratio, alpha}`）。

### 2.2 训练配置（`conf/emo_film_sentlvl.yaml`）

派生自 `conf/emo_film_earlystop.yaml`，**唯一差异**：`llm` 块加 `emo_loss_weight: 0.2`。其余逐字段一致：
- `max_epoch=30` + 早停（`early_stop_metric=loss_tts`, `patience=5`, `min_delta=0.001`, `min_epoch=5`）。
- base `pretrained_models/CosyVoice2-0.5B/llm.pt` 起步；emofilm_v1 数据合同；optimizer 三组（FiLM/heads/decoder）LR；`downstream_supervision=disabled`（句级 input-end 路径，不接 span）。

### 2.3 推理 / 评测口径

- **推理**用 `conf/emo_film.yaml`（disabled，`emo_loss_weight` 默认 0）构造模型——恒构造后分类器仍存在，与训练 ckpt 拓扑一致 → `load_trained_state` 严格加载通过（P0-2 修复验证）。
- **评测**：`eval/eval_emo_film.py`，hyp = sentlvl full 生成 wav，ref = `exp/emofilm_film_only/eval_refs/`（数据集级参考音频，与模型无关）。WER（ASR）/ Emo-SIM（emotion2vec_plus_large 余弦）/ DTW（归一化）。

## 3. 训练结果

- 启动 11:16，单 GPU（nproc=1），freeze 1.48% trainable，optimizer 三组，早停 enabled。
- **best@14**：`cv_loss_tts=3.719299`（全局最优）。用户于 epoch 17 手动停（cv 在 min_delta 边缘反复波动突破、重置 bad，预计跑满 max_epoch=30）；`best.pt → final.pt` 收口，清理 epoch 17 半成品 `latest.pt`。
- **cv_loss_tts 趋势**（每 epoch）：3.768(e0) → 3.730(e5) → 3.723(e9) → 3.747(e12, 过拟合峰) → 3.722(e13) → **3.719(e14, best)** → 3.720(e15-17, plateau)。对比 27-epoch disabled best@21 `cv_loss_tts=3.673`——**sentlvl 的 cv_loss_tts 收敛点明显更高**（+0.046），预示 WER 会变差。
- **cv_loss_emotion_input 全程 ~0.07–0.08**（ln5≈1.61 的 1/20），从 epoch 0 就低——冻结随机读出器下的**捷径效应**：模型几乎立即学会让 `modulated_text_emb` 在随机投影下情感可分。

## 4. 推理与评测

- **全量推理**：2500 样本（ESD 1500 + FEDD-A 500 + FEDD-B 500），5 GPU 并行（esd 3 shard + fedd_a + fedd_b），0 失败，全部 `finish_reason=eos` 落正式 wav。
- **试听 sanity**：3 样本（`exp/emofilm_sentlvl/listen/`）验证 final.pt 推理链路（load_trained_state + LLM + Flow + HiFT）正常。
- **baseline 评测**：3 数据集并行（GPU 3/0/1，避开并发用户占用），产出 `exp/emofilm_sentlvl/eval/{esd,fedd_a,fedd_b}_metrics.json`。

## 5. 四方对比

| 数据集 | 指标 | v1（有loss_emotion） | 5-epoch（disabled） | 27-epoch（disabled,早停） | **sentlvl（句级监督）** |
|---|---|---|---|---|---|
| ESD    | WER%    | 9.48  | 8.18  | 8.38  | **10.05** |
| ESD    | Emo-SIM | 66.75 | 66.11 | 65.89 | **65.45** |
| ESD    | DTW     | 0.332 | 0.339 | 0.341 | 0.345 |
| FEDD-A | WER%    | 8.30  | 4.70  | 4.69  | **6.54** |
| FEDD-A | Emo-SIM | 81.94 | 82.71 | 82.46 | **82.91** |
| FEDD-A | DTW     | 0.178 | 0.171 | 0.173 | **0.169** |
| FEDD-B | WER%    | 14.42 | 12.30 | 11.57 | **14.04** |
| FEDD-B | Emo-SIM | 61.60 | 62.98 | 64.31 | **63.19** |
| FEDD-B | DTW     | 0.384 | 0.370 | 0.357 | 0.368 |

## 6. 分析

### 6.1 Emo-SIM 平台未突破

- **ESD** 65.45（vs 65.89–66.75）→ 略降。ESD ~66 平台不动。
- **FEDD-A** 82.91（vs 81.94–82.71）→ 微升 +0.2~1.0，但本就在 ~82 高位，非平台突破。
- **FEDD-B** 63.19（vs 61.60–64.31）→ 居中，**低于 27-epoch 64.31**。

**确认交接文档先验**：原样恢复的句级监督不改善情感可分性。

### 6.2 WER 升高（句级监督的代价）

- **ESD** 10.05（vs 8.18–9.48），**+1.9pp**；**FEDD-A** 6.54（vs 4.69–8.30），+1.8pp vs 27-epoch。
- **机制证据**：sentlvl `cv_loss_tts` 收敛于 **3.719**，显著高于 27-epoch disabled 的 **3.673**。`loss_emotion_input` 的梯度经冻结分类器回流 FiLM，推动 `emotion_adapter`（projection）偏离纯 TTS 最优 → `loss_tts` 收敛点抬高 → WER 变差。即句级监督**以语音质量为代价换取"文本情感可分"**，但后者未转化为声学情感收益。

### 6.3 机制闭环：印证 ADR-0021 风险预警

训练日志 `cv_loss_emotion_input ≈ 0.07–0.08`（全程稳定低位）揭示了捷径：

1. 冻结的 `emotion_classifier` 是随机投影。要让其输出正确 emotion，只需 `modulated_text_emb` 在该随机子空间可分——模型几个 step 内即可达成（`loss_emotion_input` 从 1.70 降到 0.42 在前 500 step）。
2. 但"FiLM 后文本表示可被随机读出器分对类"**不蕴含**"生成语音承载该情感"。声学情感（Emo-SIM 度量）平台不动。
3. 同时，为达成这个"文本可分"，FiLM 调制被推向偏离 TTS 最优的方向，损害 WER。

这与 ADR-0021「已知风险」三条完全吻合：(a) 目标标签同时作 FiLM 输入，存在标签回读捷径；(b) 冻结随机读出器给上游的梯度无语义锚点；(c) 该 CE 证明"文本可读出条件 ID"，不证明"声学输出遵从情感"。

## 7. 方向启示

原样恢复的冻结随机探针是**无效杠杆**（情感不升、质量反降）。要真正突破 Emo-SIM 平台，候选方向：

1. **span 词级监督**（ADR-0019 长期主路径）：监督落在生成因果链下游的 speech-token hidden，细粒度、反捷径，是细粒度情感控制的正道。
2. **可训练 / 句级池化分类器**：给读出器梯度语义锚点（可训练）或用句级聚合表征（池化）代替逐 token 随机投影，避免捷径。
3. **其他句级表征增强**：更强的 emotion encoder / 对比学习等，使情感条件真正渗入声学。

## 8. 产物清单

- 模型：`exp/emofilm_sentlvl/{init,best,final}.pt`（best@14 收口）
- 训练：`exp/emofilm_sentlvl/train.log` + `tb/` + `train_identity.json`（注：手动收口未更新 final_parameter_hash，推理不依赖）
- 推理：`exp/emofilm_sentlvl/listen/`（试听 3）+ `full/{esd,fedd_a,fedd_b}/`（2500 wav）
- 评测：`exp/emofilm_sentlvl/eval/{esd,fedd_a,fedd_b}_metrics.json`
- 配置/脚本：`conf/emo_film_sentlvl.yaml` + `exp/emofilm_sentlvl/run_{train,infer,eval}.sh`

## 9. 引用

- 动机报告：`docs/reports/2026-08-01-emofilm-longepoch-convergence-comparison.md`
- 实现审查：`docs/reports/2026-08-02-emofilm-sentlvl-implementation-review.md`
- 重构计划：`docs/superpowers/plans/2026-08-02-emofilm-sentlvl-fixes.md`
- 决策记录：`docs/adr/0021-emofilm-input-end-sentence-supervision.md`
- baseline 报告：`docs/reports/2026-07-20-emofilm-v1-baseline-experiment-report.md`、`docs/reports/2026-07-30-emofilm-film-only-experiment-report.md`
- init 参考实现：`git show 9c6d84b:cosyvoice/llm/llm_emotion.py`
