# EmoFiLM FiLM-only 收敛实验：长 epoch + 早停 vs 5-epoch 基线对比（2026-08-01）

> 回答一个问题：**5-epoch 基线（CV loss_tts 3.6951@epoch4 后趋平）是「欠训」还是「已收敛」？**
> 方法：加 CV 早停 + 容忍度 + restore-best，把 max_epoch 从 5 提到 30，让模型自己决定何时停，再与 5-epoch + v1 公平对比。

---

## 1. 实验设置

| 维度 | 5-epoch 基线（参照） | longepoch（本次） |
|---|---|---|
| 配置 | `conf/emo_film.yaml` | `conf/emo_film_earlystop.yaml`（派生） |
| max_epoch | 5 | 30（上限） |
| 早停 | 无 | `early_stop: true`（patience=5 / min_delta=0.001 / min_epoch=5） |
| restore-best | 无 | 有（收口 final.pt = CV 最优 epoch） |
| 其余 | — | **lr / bs / 数据 / seed / disabled 全不变**（单一变量：epoch 预算） |

代码：`train_emo.py` 主循环 + `train_utils_emo.py::EarlyStopTracker`（详见 `docs/reports` 同目录前序与 memory `project_emofilm_early_stop`）。4-lens 对抗式审查无 critical/important。

---

## 2. 训练 CV 曲线与早停决策

| epoch | CV loss_tts | 备注 | epoch | CV loss_tts | 备注 |
|---|---|---|---|---|---|
| 0 | 3.7336 | | 14 | 3.6804 | |
| 1 | 3.7168 | | 16 | 3.6776 | |
| 2 | 3.7016 | | 18 | 3.6746 | |
| 3 | 3.7039 | bad | **21** | **3.6729** | **⭐ best（restore 收口）** |
| 4 | 3.6970 | 5-epoch 基线在此停 | 22-26 | 3.677~3.679 | 平台，bad 累积 |
| 5 | 3.6942 | | 26 | — | **patience 耗尽 → 早停** |
| 9 | 3.6867 | | | | |
| 13 | 3.6805 | | | | |

- **27 epoch 训练后早停**（best@epoch21，CV 3.6729）；restore-best 把 epoch21 权重收口为 final.pt。
- **CV 确实在 epoch 4 后继续下降**：3.6951 → 3.6729（Δ≈0.022，17 个 epoch）→ **5-epoch 是「轻度欠训」**，但下降渐进且夹杂噪声（epoch 11-12 反弹到 3.70/3.71）。
- 容忍度 `min_delta=0.001` 正确过滤噪声：epoch24 CV=3.6726 比 best 3.6729 仅低 0.0003，不计为改善（避免追噪声刷新 best）。

---

## 3. 评测结果（baseline eval_emo_film，同口径）

### longepoch(收敛) vs 5-epoch（主对比，同配置仅 epoch 不同）

| 数据集 | 指标 | 5-epoch | longepoch | Δ | 方向 |
|---|---|---|---|---|---|
| ESD | WER% | 8.18 | 8.38 | +0.20 | 略差 |
| ESD | Emo-SIM | 66.11 | 65.89 | −0.22 | 略差 |
| ESD | DTW_norm | 0.3387 | 0.3409 | +0.0022 | 略差 |
| FEDD-A | WER% | 4.70 | 4.69 | −0.01 | 持平 |
| FEDD-A | Emo-SIM | 82.71 | 82.46 | −0.25 | 持平 |
| FEDD-A | DTW_norm | 0.1706 | 0.1732 | +0.0026 | 持平 |
| FEDD-B | WER% | 12.30 | 11.57 | **−0.73** | **改善** |
| FEDD-B | Emo-SIM | 62.98 | 64.31 | **+1.33** | **改善** |
| FEDD-B | DTW_norm | 0.3701 | 0.3567 | **−0.0134** | **改善** |

### longepoch vs v1（disabled vs loss_emotion）

| 数据集 | 指标 | v1 | longepoch | Δ |
|---|---|---|---|---|
| ESD | WER% / Emo-SIM / DTW | 9.48 / 66.75 / 0.332 | 8.38 / 65.89 / 0.341 | WER −1.10 / SIM −0.86 |
| FEDD-A | WER% / Emo-SIM / DTW | 8.30 / 81.94 / 0.178 | 4.69 / 82.46 / 0.173 | WER −3.61 / SIM +0.52 |
| FEDD-B | WER% / Emo-SIM / DTW | 14.42 / 61.60 / 0.384 | 11.57 / 64.31 / 0.357 | WER −2.85 / SIM +2.71 |

---

## 4. 结论

1. **早停机制工作正确**：27 epoch 后在 plateau（best@21 后 5 个 bad epoch）触发，restore-best 把 CV 最优点收口。容忍度有效过滤噪声。**机制本身达成设计目标。**

2. **但「更收敛」没有带来指标上的统一收益**：
   - **FEDD-B 三项全部改善**（WER −0.73、Emo-SIM +1.33、DTW −0.013）——真实提升（评测在 eval 模式下确定性，差异反映模型权重）。
   - **FEDD-A 基本持平**（差异在 ±0.25 量级）。
   - **ESD 三项略降**（WER +0.20、Emo-SIM −0.22）。
   - 净效果：**混合、小幅、非统一**。

3. **核心洞察：CV loss_tts 是情感/WER 的弱代理**。epoch21（CV 最优 3.6729）并不统一优于 epoch4（CV 3.6951）；在 ESD 上反而略差。**继续最小化 CV loss ≠ 更好情感**——CV 主要反映 speech-token 重建质量，与情感表达（韵律侧）相关但不等价。

4. **Emo-SIM ~66（ESD）平台跨 v1 / 5-epoch / longepoch 三模型稳固**（66.75 / 66.11 / 65.89）→ **FiLM 方法的情感上限是方法层面**（调制语义侧 text embedding，情感主要在韵律侧 speech-token，传递链长），**非训练时长可解**。

5. **disabled（砍 loss_emotion）经三模型持续验证无害**：longepoch 在 WER 三数据集全优于 v1（−1.1~−3.6），FEDD-B 情感也更好。

**实践建议**：5-epoch 是合理的工程操作点（训练成本 5×，指标与 27-epoch 相当）。若要进一步提升可听情感，应从**方法层面**入手（声学侧调制 / 更强/更直接的监督 / 更丰富的情感数据），而非单纯加 epoch。早停机制本身有价值——它**用证据排除了「再训就好」的假设**，避免在递减收益上浪费算力。

---

## 5. 可追溯性

产物目录 `exp/emofilm_film_only_longepoch/`：
- 训练：`final.pt`（=best.pt@epoch21）、`train_identity.json`（final.pt sha256 `2730ec17...` 校验一致）、`resolved.yaml`（early_stop=true / max_epoch=30）、`patch_bundle.patch`（92KB，dirty diff）、`train.log`（完整 CV 曲线 + 早停日志）。
- 生成：`full/`（2500 wav，全 eos，0 报错）、`full/full_generation_identity.json`（aggregate `ef15d1ab...`，2500 唯一行指纹，checkpoint sha256 一致）。
- 评测：`eval/{esd,fedd_a,fedd_b}_metrics.json`。
- 试听：`listen/`（3 条 eos 样本）。
- 脚本：`run_train.sh` / `run_infer.sh` / `run_eval.sh`（可复跑）。

对比参照：`docs/reports/2026-08-01-emofilm-film-only-experiment-report.md`（5-epoch）、`docs/reports/2026-07-20-emofilm-v1-baseline-experiment-report.md`（v1）。
