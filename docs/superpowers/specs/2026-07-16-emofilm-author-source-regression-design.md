# 基础 Emo-FiLM 作者语义回归与干净重构设计

**原始日期：** 2026-07-16

**重构日期：** 2026-07-18

**状态：** Ready for agent
**代码基线：** `CosyVoice-EmoFiLM/main@5ad481d`

## 1. 目标

将本地基础 Emo-FiLM 重构为一条干净、单一、可追溯的正常实验链：

- 在作者公开边界内对齐会改变统计行为的标注、训练和推理语义；
- 保留作者未公开但本地已有合理实现的数据与评测合同；
- 删除不再服务主线的消融、诊断矩阵、PA/RL、DPO、旧标注器和兼容代码；
- 用一个新版数据合同、一个 5-epoch 模型和一次本地代理评测得出结论。

目标不是作者参数逐值复刻，也不是论文绝对指标承诺。

## 2. 设计原则

1. **语义优先：** 只对齐影响数据人口、条件、目标函数、参数更新或生成分布的差异。
2. **公开边界分层：** 作者源码、本地补全、工程 hardening 分开命名和验收。
3. **定向重建：** 重建被旧分支污染的 Emo-FiLM 专用模块；保留通用深模块。
4. **主线可达性：** main 中无法从唯一正常链路到达的历史代码和测试删除。
5. **单一数据合同：** 新版派生产物集中在一个版本目录。
6. **证据优先清理：** 先 tag/bundle/hash/证据包，再申请危险删除。
7. **单模型验收：** 不再建立消融、矩阵或自动延长训练。

## 3. 目标架构

```text
datasets/IEMOCAP
  + existing MFA TextGrid
  + external official emotion2vec-base
  + author WordSequenceModel/best_model.pth
          │
          ▼
data/contracts/emofilm_v1
  manifests → frames → word_blocks → tagged → src → train/cv parquet
          │
          ▼
clean Emo-FiLM SFT (5 epoch local budget)
  EmotionEncoder + FiLM + decoder trainable
          │
          ▼
clean production inference
  tagged target → target-only LLM condition
  prompt → speaker / Flow / HiFT only
          │
          ▼
local fixed ESD 1500 + FEDD-A 500 + FEDD-B 500
  emofilm-eval-v2: WER / Emo-SIM / DTW
```

## 4. 模块边界

### 4.1 定向重建模块

| 模块 | 目标接口 | 实现要求 |
|---|---|---|
| `llm_emotion.py` | `forward(batch, device)`、`inference(...)` | 作者有效前向；target-only LLM condition；允许 KV cache |
| `emo_film.py` | EmotionEncoder、FiLMLayer | 删除 AddFusion 和 loss seam 变体；alpha 仅可作 checkpoint 兼容 |
| `model_emo.py` | `tts(**model_input)` | 非流式唯一主线；任何关闭路径都清理状态 |
| `frontend_emo.py` | `frontend_emo_film(...)` | 目标 text/emotion/intensity；prompt 只供 speaker/Flow/HiFT |
| `train_emo.py` | 基础 SFT CLI | 删除 DPO/GAN 特殊拼接；精确冻结、最小 checkpoint 生命周期 |
| `train_utils_emo.py` | 冻结与 optimizer helper | 只收集 requires_grad；无消融模块表 |

不直接复制作者研究原型的绝对路径、调试输出、`pass` 接口或未使用旧代码。

### 4.2 局部修改模块

- `processor.py`：增加 `token_mel_ratio=2` 联合裁剪；保留现有 shuffle/sort 实现。
- `dataset.py`：验证 processor 参数接口，避免 GAN 路径潜在 TypeError；基础主线不引入 GAN。
- `emo_tokenizer.py`：正文固定 lowercase；保留闭合标签 fail-fast。
- `common.py` 或 Emo-FiLM 专用采样模块：EOS 重采样和作者 RAS fallback 分布。
- 标注工具：作者 fairseq base、768d/50Hz、作者 checkpoint strict load、无平滑、双标签合并。
- parquet 工具：先固定 train/cv manifest，再分别直接打包；异步错误必须传播。

### 4.3 保留模块

- 通用 CosyVoice LLM、Flow、HiFT；
- ESD/IEMOCAP 索引、MFA、FEDD-rebuilt、prompt anchor；
- speech token、speaker embedding；
- 正式批量并行推理；
- `eval_emo_film.py` 中仍服务 `emofilm-eval-v2` 的原语；
- 严格样本配对和运行身份记录。

## 5. 作者语义合同

### 5.1 标注

- `emotion2vec-base`，帧特征 `(T, 768)`，50 Hz provenance；
- 作者 WordSequenceModel，5 类+3D VAD，加载 `best_model.pth`；
- 每个词独立处理；
- arousal 映射 low/medium/high；
- 不平滑；
- 相邻词只有 emotion 和 intensity 均相同才合并；
- ESD 使用已知 Global Label，IEMOCAP 使用伪标签；
- 两个模型均冻结，不重训。

外部取得的 checkpoint 必须是官方 emotion2vec-base 资产，并验证可由作者随包 fairseq upstream 严格加载；作者包内三个同名文件为 0 字节。不得把任意同名 FunASR/ModelScope wrapper 产物直接视为逐帧等价。

### 5.2 训练

- emotion classifier 读取 FiLM 后文本表示；
- trainable 精确为 emotion encoder、FiLM adapter、LLM decoder；
- classifier 冻结但参与 loss；
- token/mel 1:2 联合裁剪；
- static batch 4；
- 公开 filter 值；
- Adam、`lr=1e-5`；
- 5 epoch 是论文口径的本地预算，不称作者源码合同。

### 5.3 推理

- lowercase 后 target tagged text 产生等长三元组；
- LLM condition 为 `[SOS, FiLM(target), task]`；
- prompt 不进入 LLM，但继续进入 speaker/Flow/HiFT；
- `min_len=2×text_tokens`、`max_len=200`；
- min_len 前 EOS 在原分布重采样；
- 真正 EOS 终止，其他辅助 special token 跳过；
- RAS fallback 不排除首次重复 token；
- KV cache 和当前 ONNX provider 保留。

## 6. 本地补全合同

以下版本冻结，不随核心重构改变：

- ESD 1500 测试集；
- FEDD-rebuilt 1000 条及 A/B 分区；
- prompt、prompt_text、neutral anchor；
- train/test 排除规则；
- 批量并行推理；
- `emofilm-eval-v2` WER、Emo-SIM、DTW。

文件和报告统一标注为本地补全/代理实现。

## 7. 数据设计

### 7.1 目录

```text
data/contracts/emofilm_v1/
├── manifests/{train,cv,rejected}.jsonl
├── emotion2vec_base_frames/iemocap/
├── word_blocks/iemocap/
├── tagged/{iemocap,esd,train,cv}.jsonl
├── src/{train,cv}/
├── parquet/{train,cv}/
└── provenance/
```

### 7.2 成员关系

- 沿用当前 TTS train/cv utterance 归属；
- 继续排除 ESD test 与既有 FEDD-B 训练源；
- rejected IEMOCAP 从原集合移除，不补位；
- 不重新随机切分，不增加 speaker-disjoint。

### 7.3 IEMOCAP 监督配对清洁

- 冻结 train/cv 使用同一质量规则，从原 split 扣除且不补位；
- 完全空词区间或音频内容未被 tagged text 完整覆盖时 rejected；
- 精确覆盖和仅撇号切分等价的配对保留；
- 尾部量化裁剪若仍有有效词块则保留，并逐条记录边界事件；
- 不设掩盖已确认异常的固定百分比预算，不做逐样本人工补词或裁音频；
- rejected 逐条记录原因、原 split、speaker 和 emotion，并报告分布；
- ESD/FEDD test、reference、prompt 任何失败都 hard-fail。

## 8. Checkpoint 与运行身份

- 保存一次 `init.pt`；
- 每 epoch 原子覆盖 `latest.pt`；
- 成功时将 latest 原子转为 `final.pt`；
- 不永久保留 epoch 全量副本；
- final 旁保存 resolved config、数据合同 hash、Git commit、seed、参数 hash 和训练日志；
- 基座只允许新增情感模块缺失；trained/resume strict load；
- 这些属于本地 hardening。

关键结构合同同时固定为：Qwen2 hidden 896/24 层/14 heads/2 KV heads，EmotionEncoder 为 `6×896 + 4×896` 两个 embedding，FiLM projection 为 `896→1792`，emotion classifier 为 `896→6`，speech embedding/decoder 覆盖 `6561+3` token。作者 WordSequenceModel 固定为 768d、8 heads、FFN 768→3072→768、5 类和 3D VAD。只参与旧 checkpoint 序列化而不参与有效 forward 的 FiLM `alpha` 不视为架构差异。

## 9. 活跃测试面

只保留：

1. 数据合同：隔离、标签来源、作者 WordSequence state-dict 形状、768d/3D、覆盖、拒绝预算、parquet 加载；
2. 训练语义：有效模块拓扑与关键张量形状、loss seam、trainable 集合、token/mel、static batch、checkpoint；
3. 推理语义：lowercase、target-only condition、长度、EOS、aux、RAS、Flow prompt；
4. 可靠性：generator 清理、分片合并、严格配对、指标聚合；
5. 三样本端到端 smoke。

删除旧消融、矩阵、PA/RL、旧 1024d、per-emotion accuracy 和墓碑测试。

## 10. 资产生命周期

### 10.1 重构前

- 为 main 和 feature 分支创建 tag/bundle；
- 生成排除 data/exp/model 的源码快照；
- 生成清理 manifest 和 SHA-256；
- 建立旧标注器、v1-v5、v6 的精简证据包。
- 从已验证 main commit 创建隔离实施分支/worktree；共享 main 只作为基线和恢复点。

### 10.2 第一阶段清理候选

- 旧 plus-large `emo_feats/word_blocks`；
- 四组消融 parquet/src；
- v1-v5 五份重复转换 checkpoint；
- `datasets/ESD.BAK` 与重复 zip；
- 重构后不可达的工具、tests、配置和适配器。

实际删除前必须另行确认。

### 10.3 新版通过后

再决定旧 full parquet、v6 全量 WAV 和其他历史大资产是否压缩离线或删除。

## 11. 最终验证

```text
focused tests
→ 3 samples smoke
→ 30 samples confirmation
→ local fixed 2500 samples
→ partitioned evaluation and report
```

只报告 ESD、FEDD-A、FEDD-B 的：

- technical completeness；
- WER ratio/percent；
- Emo-SIM；
- cosine DTW。

与旧 local final 和 local identity 做分区相对比较。不设置论文或本地绝对失败阈值；不自动延长、不增加矩阵。混合变化如实解释。

## 12. 完成条件

- 备份和证据 manifest 完成；
- 危险清理经单独确认并留下记录；
- main 只含可达主线；
- 新数据合同完整、可加载、无测试泄漏；
- WordSequence 作者 checkpoint 与标签产物检查通过；
- 唯一 5-epoch 模型完成并具有完整运行身份；
- 3→30→2500 技术链完整；
- 最终报告明确作者公开、本地补全和不可闭合边界；
- inventory 更新为实际状态。
