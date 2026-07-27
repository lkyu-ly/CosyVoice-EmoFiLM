# EmoFiLM v1：模型架构与可追溯评测静态审查报告（核验合并版）

原始日期：2026-07-22；核验合并：2026-07-24  
范围：当前 canonical checkout 的 EmoFiLM v1 训练、推理、数据合同、运行身份与正式评测入口。  
方法：以本报告和 `2026-07-24-emofilm-model-design-static-analysis.md` 的既有论断为索引，快速复核其直接代码、配置、manifest、checkpoint 元数据和运行记录；重复发现直接确认，独有发现定点核验；未重新开展全项目审查，未运行训练、推理或评测。  
立场：当前实现是被审查对象。本文不将任何作者代码、历史实现或历史决策视为正确参照，也不以“与其不同”作为问题成立的理由。

## 执行结论

EmoFiLM v1 已完成一次 5-epoch 训练、2500 条生成和三分区 aggregate 评测。这些产物记录了当前基线的输入、checkpoint、输出集合和 aggregate 数值，但不能证明系统已经实现词级 emotion/intensity 控制；P2-3 说明其源码与逐条生成身份仍不具备严格重建条件。静态审查只保留下列高确定性问题，按下一轮架构决策的优先级排序。

| 优先级 | 确定性结论 | 为什么优先 |
|---|---|---|
| P0 | 辅助 emotion CE 在条件刚注入后的文本表示上回读同一个 emotion ID，存在不经过 speech token、声学模型或波形的标签捷径。 | 当前辅助目标不能作为“情感已进入生成语音”的监督。 |
| P0 | 该 CE 的 classifier 是基座中不存在的新建线性层，且从随机初始化起永久冻结。 | 它仍向 FiLM 支路传梯度，但梯度几何没有已验证的情感或声学语义来源。 |
| P1 | 训练启用了带 teacher-forced speech history 与 fill token 的双流因子化；部署只支持 target-only 单流解码，并拒绝 fill 等辅助 token。 | 部分训练样本的概率分解与唯一产品协议不同。 |
| P1 | 推理同时使用 max_len=200 和 min_len=2×text_len；当 text_len 大于或等于 100 时，正常 EOS 完成不可能发生。 | 这是可由代码直接推出的确定性长度错误。 |
| P1 | 评测只使用整句 Emo-SIM、全序列 DTW 和整句 WER；不消费 span、边界、transition 或 intensity，且不持久化逐样本行。 | 当前数值不能验证词级动态控制，也不能为架构改动提供可证伪反馈。 |
| P1（强度分支） | 所有 2500 条正式评测均固定为 medium，且唯一辅助 CE 没有 intensity target。 | 目前结果不能证明 low/high 强度控制；不能据此判断强度设计优劣。 |
| P1 | IEMOCAP 的 6,412/20,774 条训练样本使用逐词伪标签，但标注器训练时把句级 emotion/VAD 广播到每个 word block，且生成合同只保留 hard label。 | 这些逐词变化没有独立词级真值或置信度约束，直接削弱局部控制监督的可解释性。 |
| P1 | optimizer 声明区分新模块与 base 参数，实际所有可训练参数都进入同一组且两档 LR 相同；配置的 warmup_steps 未被调度器读取。 | 当前优化合同与生效行为不一致，无法归因新模块与预训练输出头的学习动态。 |
| P2 | FEDD-A 的 500 条局部转场标签全部是“按词数中点”的显式近似，而不是声学边界。 | FEDD-A 不能支撑精确词级边界结论，应与有已知边界的 FEDD-B 区分。 |
| P2 | 正式运行来自 dirty worktree；记录了旧 commit 与 diff hash，却未保存可重建 patch；134 条 ESD WAV 以 skip-existing 恢复。 | 当前结果可追溯到输入/输出集合，但不能严格重建执行代码或逐条验证恢复 WAV 的生成身份。 |

关于“classifier 应冻结还是训练”的直接回答是：**当前随机、输入端的 classifier 不应被冻结后继续作为情感监督；但仅解冻它也不是修复。应直接把辅助监督迁移到生成因果链下游，并让分类/回归头具有可验证的语义来源。** 可行方向是从与目标 span 对齐的 speech-token hidden state 或生成声学片段预测 emotion/intensity，由联合训练任务头或标签空间对齐的声学教师提供监督；现有输入端 CE 不应继续承担“输出情感监督”的职责。

## 审查准则与已核验背景

本项目要解决的是：对文本局部片段施加 emotion/intensity 条件，并让生成语音在相应位置实现可控变化。故只将同时满足下列条件的问题纳入结论：

- 代码、配置、数据合同或运行记录能直接定位；
- 存在明确的失效机制，不以“可能效果更差”替代因果说明；
- 与局部可控情感语音合成或下一轮实验归因直接相关。

未纳入“应加 L2 / weight decay”“5 epoch 一定不足”“LLM、Flow 或 HiFT 冻结一定错误”“FiLM 只在输入层一定不足”“emotion 与 intensity 共享投影必然有害”等常见猜测：这些都需要训练曲线、消融或声学证据，静态审查不能把通用经验写成已发生的缺陷。

已核验背景如下。

- 正式基线从 CosyVoice2-0.5B 的 llm.pt 以 checkpoint_role=base 启动，完成 5 epoch；正式报告的样本数为 ESD 1500、FEDD-A 500、FEDD-B 500。见 [基线实验报告](2026-07-20-emofilm-v1-baseline-experiment-report.md) 和 [训练身份](../../artifacts/emofilm_v1/train/train_identity.json)。
- base checkpoint 缺失所有 emotion_ 前缀参数；加载器明确只允许这些新增模块缺失。final.pt 则包含新增 emotion 模块参数。证据位置：cosyvoice/utils/emo_checkpoint.py 第 8–49 行。
- 当前训练/验证合同分别为 20,774 / 1,092 条。训练 manifest 的一个样本只计一个 active tagged_text：61,578 个 tag segment 中，low 为 309（0.502%）、medium 为 38,665（62.790%）、high 为 22,604（36.708%）。该统计说明支持度不均衡，但不单独证明模型无法学习某一档。
- 训练来源并非全部是词级标注器伪标签：训练集 14,362 条 ESD 样本使用 `dataset_global_label`，6,412 条 IEMOCAP 样本使用 `word_annotator_pseudo_label`；验证集对应为 758 与 334 条。故“全部 emotion/intensity 条件都来自同一标注器”的原论断不成立，本文只对 IEMOCAP 伪标签子集作粒度结论。

现有正式 aggregate 记录如下。它们是当前实现的固定基线，而不是词级控制有效性的证明。

| 分区 | N | WER | Emo-SIM | cosine DTW normalized |
|---|---:|---:|---:|---:|
| ESD | 1500 | 9.4770% | 66.7451 | 0.332368 |
| FEDD-A | 500 | 8.2975% | 81.9401 | 0.178268 |
| FEDD-B | 500 | 14.4211% | 61.5963 | 0.383877 |

基线报告还保留了与旧 v6 local final / local identity 的同口径 aggregate 比较：ESD 与 FEDD-B 的部分或全部全局指标改善，而 FEDD-A 的变化并不一致。本文将这些数值只作为历史背景，不把它们作为任何架构优劣或后续设计选择的依据：它们同样没有 span/boundary/intensity 指标，而且历史决定不参与本轮决策。

| 历史比较对象 | 分区 | Emo-SIM Δ | normalized DTW Δ | WER 百分点 Δ |
|---|---|---:|---:|---:|
| old v6 local final | ESD | +11.6150 | -0.115844 | -18.3404 |
| old v6 local final | FEDD-A | -2.0820 | +0.020088 | -7.7634 |
| old v6 local final | FEDD-B | +2.4216 | -0.023976 | -21.9741 |
| old v6 local identity | ESD | +15.5508 | -0.155422 | +3.1301 |
| old v6 local identity | FEDD-A | -1.1289 | +0.011205 | +6.8807 |
| old v6 local identity | FEDD-B | +8.1616 | -0.081552 | +2.1146 |

其中 Emo-SIM 的正值表示更高；normalized DTW 与 WER 的负值表示改善。该表只转录既有正式报告，并不暗示旧实现是参考实现，也不用于本报告的严重度排序。

## P0-1：辅助 emotion CE 是条件标签自重构，不是生成语音情感监督

### 实现事实

训练前向的有效依赖图为：

    emotion_ids + intensity_ids
            │
            ▼
    EmotionEncoder ──► FiLM(text_token_emb) ──► modulated_text_emb
                                                 │
                                                 ├──► LLM ─► speech-token CE
                                                 │
                                                 └──► emotion_classifier ─► CE(target = emotion_ids)

- emotion_ids 与 intensity_ids 先进入 EmotionEncoder，再用于 FiLM 调制文本 token 表示：cosyvoice/llm/llm_emotion.py 第 65–72 行。
- 同一份 emotion_ids 随后又作为 emotion_classifier(modulated_text_emb) 的 CE target：cosyvoice/llm/llm_emotion.py 第 96–102 行。
- 该辅助分支不读取 speech_token、LLM 输出、Flow 输出、HiFT 输出、参考音频或任意声学 emotion target。EmotionEncoder 仅把 emotion/intensity embedding 相加：cosyvoice/llm/emo_film.py 第 9–17 行。

### 为什么这是结构问题

目标标签已经作为显式输入到达 classifier 的上游，因此存在低损失路径：FiLM 将 ID 写入 modulated_text_emb，classifier 再读出该 ID。此路径无需标签影响 LLM 的 speech-token 分布，也无需最终 mel 或波形表达目标情感。

该 CE 最多能证明“FiLM 后文本表示携带了可读出的条件 ID”，不能证明“生成语音在相应 span 遵从了该条件”。这个标签捷径独立于 classifier 是否冻结：把 classifier 解冻，只会使输入标签复制器更容易拟合，并不会让该目标自动迁移到声学输出。

### 设计方向

1. 移除或停用该输入端 CE，并以生成因果链下游的情感目标直接替代，而不是把“只保留 speech-token likelihood”作为终点。
2. 让预测对象位于生成因果链下游，且不能直接读取控制标签。例如从与目标 span 对齐的 speech-token hidden state 预测 emotion/intensity，或采用经验证、标签空间对齐的声学 SER 教师约束生成片段。
3. emotion 与 intensity 应定义各自的 target、mask、验收指标；文本侧 probe 如保留，应明确只用作条件可读性诊断，不能计作声学情感监督。

## P0-2：随机初始化且永久冻结的 classifier 仍在向上游施加无语义锚点的梯度

### 实现事实

- 模型构造时新建 nn.Linear(llm_input_size, emotion_vocab_size)，随后立即 requires_grad_(False)：cosyvoice/llm/llm_emotion.py 第 52–57 行。
- base loader 允许 emotion_encoder、emotion_adapter 与 emotion_classifier 整体缺失：cosyvoice/utils/emo_checkpoint.py 第 8–49 行；正式训练身份也确认采用普通 base llm.pt。
- 训练冻结策略只允许 emotion_encoder、emotion_adapter、llm_decoder 可训练，并再次强制 classifier 冻结；optimizer 同样排除它：cosyvoice/utils/train_utils_emo.py 第 15–19、59–107 行。
- CE 仍按 emo_loss_weight=0.2 加入总 loss：cosyvoice/llm/llm_emotion.py 第 96–105 行；活跃配置见 conf/emo_film.yaml 第 32–45 行。

### 澄清：冻结不等于没有梯度

令 h=modulated_text_emb，z=Wh+b，L=CE(z,y)。即使 W,b 不更新，仍有：

    ∂L/∂h = Wᵀ(softmax(Wh+b) - onehot(y))

因此梯度会回传到 FiLM、emotion embedding 和 intensity embedding。错误不在于辅助 loss “失效”，而在于该正式 base 初始化路径上的 W,b 没有预训练来源、也未由当前任务校准。它可以迫使上游形成随机可分代码，却没有可追溯理由把这种几何与情感语义或生成语音的声学情感对应起来。

### 设计方向

- 不要仅把该头从 frozen 改为 trainable；P0-1 的输入标签捷径仍然存在。
- 只有当 classifier 是独立训练、输入表征分布匹配、标签语义已验证的教师/probe 时，冻结才有合理的语义基础。
- 若是项目内新任务头，应移动到其所读的下游生成表征并联合训练，用独立的 span/control 评测检验它约束了输出而非标签复制。

## P1-1：双流训练协议与唯一部署协议不一致

### 实现事实

- 活跃配置设置 mix_ratio: [5, 15]：conf/emo_film.yaml 第 36–37 行。
- 当 speech_token_len / text_token_len 大于 15/5 且随机分支命中时，训练构造交错的 text(5)、teacher-forced speech(15)、fill 序列，并把 fill_token 放入 target：cosyvoice/llm/llm.py 第 302–349 行。分支概率是 0.5，fill token 位于 speech vocab 之外：cosyvoice/llm/llm.py 第 274–297 行。
- 唯一产品接口显式拒绝 stream：cosyvoice/cli/model_emo.py 第 25–36 行。推理固定以 SOS、FiLM(target text)、task 开始，然后只自回归 speech token：cosyvoice/llm/llm_emotion.py 第 129–193 行。
- 推理遇到 EOS 之前的 EOS 或任意大于等于 speech_token_size 的 token 都会重采样；这包括训练双流分支使用的 fill token：cosyvoice/llm/llm_emotion.py 第 171–185 行。

### 失效机制

命中双流分支的训练样本学习的是“给一段文本、消费真值 speech history、输出 fill、再接收下一段文本”的因子化；产品推理从不提供真值 speech history、交错文本块或 fill 的状态转移。二者的前缀状态和下一个 token 分布不是同一协议。

这里不声称静态分析可以量化指标损失；可确定的是，部分优化步骤被分配给部署端不会执行的语法，且部署端会把其中的 fill token 当成非法 token 重采样。

### 设计方向

若产品确定为非流式 target-only TTS，训练也应只使用与其一致的单流 SOS、FiLM(target text)、task、speech 分解。若确实需要双流，则必须实现同一解码协议，并分别固定两种模式的输入、停止逻辑和评测合同；不能训练一套、部署另一套。

同一类协议不一致还出现在 zero-shot prompt 条件：基座 `Qwen2LM.inference` 会拼接 `prompt_text` 与 `prompt_speech_token_emb`，而 EmoFiLM 的训练前向不包含 prompt，推理又显式删除 prompt_text、prompt_speech_token 与 embedding（cosyvoice/llm/llm.py 第 458–515 行；cosyvoice/llm/llm_emotion.py 第 60–88、109–145 行）。这能确认 EmoFiLM 改写后的 LLM 条件协议与基座 zero-shot 协议不同，但不能仅凭静态图量化其对声纹或情感的损失。修正时不应只在推理端“拼回” prompt；应选定一个训练与推理共同使用的 prompt/target 条件序列，并在 prompt 段与目标段上明确 emotion mask，避免再次制造新的 train/inference mismatch。

## P1-2：推理长度合同自相矛盾，长文本存在确定性停止错误

### 实现事实与推导

- inference() 暴露 max_token_text_ratio，却忽略它并硬编码 max_len=200；同时默认 min_len=int(text_len×2)：cosyvoice/llm/llm_emotion.py 第 109–145 行。
- 解码循环仅执行 range(max_len)；如果未生成接受的 EOS，函数自然结束，没有截断状态、警告或异常：cosyvoice/llm/llm_emotion.py 第 148–193 行。
- 训练 filter 允许至多 9,999 个 text token，未施加“输出 speech token 至多 200”的对应合同：conf/emo_film.yaml 第 150–154 行。

当 text_len 大于或等于 100 时，min_len 大于或等于 200。循环最多产生 200 个有效 speech token，而 EOS 仅在当前已产生 token 数不小于 min_len 时才可接受。因此正常 EOS 完成不可能发生：通常会在 200 个 token 后无 EOS 退出；若采样持续只给 EOS 或辅助 token，则会在重采样上限处报错。无论哪种结果，都不符合一个可完成的长度合同。

### 设计方向

将训练、服务与评测统一到单一长度合同：max_len 应从文本长度与明确 ratio 推导，且始终满足 max_len 大于 min_len。若全局安全上限不足以容纳输入，应明确拒绝、分段，或返回可审计的截断状态；不能把部分生成伪装成正常完成。

## P1-3：正式评测不验证细粒度控制，也不提供可归因的逐样本证据

### 实现事实

- 评测从 WAV 目录按 ID 严格配对，计算整段帧均值 Emo-SIM、全序列 DTW 与整句 WER：eval/eval_emo_film.py 第 107–129、181–220、343–415 行。
- run_evaluation() 的输入只有 ref/hyp WAV 路径和 WER 文本 map；它不读取 tagged_text、emotion_transition、boundary_word_index、emo_from、emo_to 或 intensity_policy。
- FEDD-B manifest 含明确的 boundary_word_index，而 FEDD-A/B 都含 tagged_text 与 transition 元数据；这些字段并不进入指标计算。标签构造见 tools/build_fedd_tagged_text.py 第 24–75 行。
- 评测在内存建立 rows，但只返回 aggregate；CLI 仅写 aggregate JSON：eval/eval_emo_film.py 第 386–441 行。正式报告也明确未持久化逐样本 metric rows。

### 失效机制

整句均值 embedding 与无边界 DTW 能描述总体接近程度，却不能判断：

- emotion A 是否落在指定前段、emotion B 是否落在指定后段；
- 转场是否发生在给定词边界附近；
- strong/mild 转场是否按目标方向发生；
- low/medium/high 是否有可分、单调的响应。

相同 aggregate 分数可来自正确转场、整句保持单一 emotion，或转场时序错误。因此当前 WER/Emo-SIM/DTW 的变化不能用于证明“词级控制更好”，也无法裁决 classifier、FiLM 注入位置或标签设计的架构优劣。

### 设计方向

- 评测入口应消费 manifest 的 span、边界、emotion/intensity target，输出逐样本、逐 span 行；aggregate 只能作为派生产物。
- 对 FEDD-B 至少报告前/后段情感命中、转场方向和边界时间误差；FEDD-A 与其近似边界应单独分层。
- 建立同文本、同 prompt、只替换控制标签的成对生成评测，以检验输出对控制变量的敏感性，而不是只与单一参考音频比较。

## P1-4：强度分支缺乏验证闭环；当前结果无法支持三档控制结论

### 实现事实

- tokenizer 定义 low、medium、high 三档；EmotionEncoder 将 intensity embedding 与 emotion embedding 相加：cosyvoice/tokenizer/emo_tokenizer.py 第 11–16、45–46 行，cosyvoice/llm/emo_film.py 第 9–17 行。
- 唯一 auxiliary CE 的 target 是 emotion_ids，没有 intensity CE、回归、排序或一致性 target：cosyvoice/llm/llm_emotion.py 第 96–105 行。
- 所有正式 eval manifest 均是 intensity_policy=fixed_medium：ESD 1500 个 span、FEDD-A 1000 个 span、FEDD-B 1000 个 span，合计 3500 个控制 span 全为 medium。构建器默认强度亦为 medium：tools/build_esd_tagged_text.py 第 21–59 行、tools/build_fedd_tagged_text.py 第 24–75 行。

### 严格可得的结论

loss_emotion 对 intensity 没有直接的语义目标；它的梯度至多鼓励 intensity 分支帮助预测 emotion。speech-token likelihood 仍可能间接学习强度，因此不能将“没有 intensity head”表述为“强度必然学不会”。

但因为正式 2500 条评测从未输入 low/high，当前结果无法区分强度 embedding 有效、失效、反向响应或塌缩。故它们不能支持有关三档强度可控性、强度编码设计或强度损失的架构结论。

### 设计方向

在比较强度架构前，先定义有来源和校准方式的强度 target，并加入固定文本、固定 emotion、固定 prompt、仅改变 low/medium/high 的成对或三元评测。报告应同时检验强度可分性/单调性、emotion 保持和可懂度。

当前强度合同还有两项已确认的表达能力限制：IEMOCAP 标注器虽然训练 3 维 VAD 回归，但伪标签生成只读取 arousal 并按固定阈值离散成三档，valence/dominance 不进入下游；模型构造参数 `alpha=0.05` 也未参与任何有效计算（tools/train_annotator.py 第 71–81、121–148 行；tools/generate_tagged_jsonl.py 第 28–63 行；cosyvoice/llm/llm_emotion.py 第 23–59 行）。这不证明三档强度必然失效，但证明当前接口无法表达连续强度，也没有利用完整 VAD 监督。直接改善方向是保留校准后的连续 arousal 或 VAD 条件，并为 intensity 建立独立的回归/排序目标；不要只扩大离散档位而继续缺少验收信号。

## P1-5：IEMOCAP 逐词伪标签超出标注器的监督粒度

### 实现事实

- `WordEmoDataset` 读取每条 utterance 的 `sentence_emotion` 与 `sentence_vad`，然后把同一 label/VAD 复制给该 utterance 的每个 word block：tools/train_annotator.py 第 38–91 行。
- `WordSequenceModel` 对一个 word block 的帧序列做 attention、FFN 与 masked mean pooling，输出一个 5 类 emotion 和 3 维 VAD：cosyvoice_emo/emo_annotator.py 第 7–79 行。
- 标签生成阶段逐个 word block 独立调用模型，使用 emotion argmax 与 arousal 三档阈值生成词级标签；输出合同没有保存 class probability、VAD 全向量或置信度：tools/generate_tagged_jsonl.py 第 25–65、97–142 行。
- 该问题只覆盖 IEMOCAP 伪标签子集：训练 6,412 条、验证 334 条。ESD 的 14,362/758 条使用 `dataset_global_label`，不能把两者概括为“全量训练条件均来自词级标注器”。

### 失效机制

标注器只见过句级标签广播后的 word block；因此模型可以学习“某个词块对所属句级 emotion/VAD 的代理预测”，但训练目标没有告诉它同一句内哪些词真的发生情感变化。逐词独立推理产生的局部波动没有词级真值校准，而 hard argmax/分桶又丢弃了不确定性。由此可确认：这些标签是弱监督伪标签，不能当作已验证的词级 ground truth，也不能让当前逐词控制结论自证成立。

这里不推导“每个伪标签都错误”，也不把 label smoothing 当作根治手段；平滑 CE 不能补回缺失的词级监督。

### 设计方向

- 直接提升创新点所需的局部监督：优先使用有边界/局部标签的数据，或以声学教师生成带置信度与时间边界的 soft target，并把低置信度 token 从 loss 中 mask/降权。
- 将标签来源、置信度、时间边界和软分布保留到训练合同，避免在离线生成阶段永久压成 hard ID。
- 若只能使用句级标签，应把它明确作为句级条件或多实例弱监督目标，而不是宣称为词级真值；局部控制能力必须由独立的 span/boundary 评测验证。

## P1-6：optimizer 分组与 warmup 配置未形成有效优化合同

### 实现事实

- `freeze_all_except` 只解冻 `emotion_encoder`、`emotion_adapter`、`llm_decoder`；`init_optimizer_emo` 又把同一集合全部归入 `emotion_new`，所以当前 `base_params` 必为空，optimizer 只创建一个参数组：cosyvoice/utils/train_utils_emo.py 第 15–19、42–128 行。
- 活跃配置中 `lr` 与 `new_params_lr` 都是 `1e-5`，即使未来出现 base 组也没有学习率差异：conf/emo_film.yaml 第 215–223 行；正式 resolved.yaml 保留同样配置。
- yaml 声明 `scheduler_conf.warmup_steps=2500`，训练入口却无条件构造 `ConstantLR(optimizer)`；该 scheduler 没有 warmup 参数且始终返回 base LR：cosyvoice/bin/train_emo.py 第 233–246 行；cosyvoice/utils/scheduler.py 第 719–738 行。

### 失效机制

确定性问题不是“1e-5 必然太低”或“无 warmup 必然导致效果差”，而是配置表达的两个优化意图没有生效：新模块与预训练输出头无法使用不同优化策略，warmup 字段是死配置。这使训练身份看似记录了分组/warmup，实际运行却不能验证这些策略，也阻断了对新随机模块学习动态的归因。

### 设计方向

- 先按参数角色建立真实分组：随机初始化的 `emotion_encoder`/`emotion_adapter`、预训练继承的 `llm_decoder` 分开配置，并在运行身份中持久化每组参数数、初始 LR、weight decay 与 scheduler。
- 若需要 warmup，训练入口应从配置实例化支持 warmup 的 scheduler；若采用常量 LR，则删除 `warmup_steps`，保持配置与行为单一真实来源。
- 具体 LR、warmup 和正则数值应由短周期训练曲线与局部控制验证选择，不能从静态代码直接指定为确定最优值。

## P2-1：prompt 情感字段是未消费的接口，掩盖了真实条件协议

### 实现事实

- 推理前端为 prompt 文本创建固定 neu/low 的 `prompt_emotion_ids` / `prompt_intensity_ids`，模型 API 和 batch collate 也保留这些字段：cosyvoice/cli/frontend_emo.py 第 56–83 行；cosyvoice/dataset/processor.py 第 440–445 行；cosyvoice/cli/model_emo.py 第 25–59、89–105 行。
- 训练 tokenizer 不生成 prompt emotion/intensity，`Qwen2LM_Emotion.forward` 只读取目标端 `emotion_ids/intensity_ids`，`prepare_lm_input_target` 也没有 prompt 情感参数：cosyvoice/dataset/processor.py 第 465–484 行；cosyvoice/llm/llm_emotion.py 第 60–88 行；cosyvoice/llm/llm.py 第 302–349 行。
- `Qwen2LM_Emotion.inference` 接收后立即删除 prompt emotion/intensity，因此这些值从未进入模型计算：cosyvoice/llm/llm_emotion.py 第 109–145 行。

### 影响与设计方向

这是高置信度的接口/合同缺陷，但没有证据表明 prompt emotion 本来就应成为独立控制变量，所以严重度低于训练监督与评测问题。应先定义统一的 prompt/target 条件协议：若 prompt 情感用于上下文建模，就在训练和推理同一位置接入并用 mask 区分 prompt 与 target；若产品只控制目标文本，则删除这些字段，避免接口伪装成已实现能力。

## P2-2：FEDD-A 的局部边界是显式近似，不能与有已知边界的 FEDD-B 混为同一证据等级

### 实现事实

- FEDD 标签构建器在缺少 boundary_word_index 时，按文本词数中点切成两个 span，并写入 method=midpoint_two_span_approximation：tools/build_fedd_tagged_text.py 第 24–75 行。
- FEDD-A 的 500/500 条均使用该方法；FEDD-B 的 500/500 条使用 method=exact_concatenation_boundary，且具有 boundary_word_index。这是当前 manifest 的只读静态计数。

### 影响与设计方向

FEDD-A 所标的“在第几个词前后切换”没有参考音频对应的、可审计的声学边界记录。它适合作为带弱局部先验的总体相似度分区，但不应与 FEDD-B 一样支撑精确边界能力结论，更不应把两者混合为单一“词级动态准确性”数字。

下一轮评测应将它们分层：FEDD-B 用于可检查的边界/方向度量；FEDD-A 若继续保留，应标记为近似边界分区，或补充独立的声学时间标注后再用于局部结论。

## P2-3：现有基线具有部分身份记录，但不能严格重建或逐条归因

### 实现事实

- 训练、生成、评测 identity 都记录 dirty=true、旧 git_head=5ad481… 与 worktree_diff_sha256；当前 canonical 对象库不存在该 commit。见 artifacts/emofilm_v1/train/train_identity.json、artifacts/emofilm_v1/generation/full_generation_identity.json、artifacts/emofilm_v1/evaluation/evaluation_identity.json。
- identity 写入器只计算 dirty diff 的 SHA-256，不保存 patch 或源码 bundle：tools/write_emofilm_run_identity.py 第 77–100 行。
- ESD 有 134 条恢复生成使用 skip-existing；输出 manifest 对这些条目只记录 utt_id、status=skipped_existing 与 WAV 路径：tools/inference_emo_film.py 第 156–207 行，正式 generation identity 也记录该计数。
- 评测只读取 ref/hyp WAV 目录，不消费 generation manifest：eval/eval_emo_film.py 第 343–415 行。

### 影响与设计方向

这不表示正式结果错误。可确定的是：输入合同、checkpoint hash、输出集合和 aggregate 指标已有记录，但无法从当前 identity 文件重建“实际的脏代码”，也无法机器验证被恢复的 134 条 WAV 与同一 checkpoint、控制文本、prompt 和生成参数绑定。

未来架构比较应以干净可访问 commit，或随运行保存不可变 patch/source bundle 为前提；generation manifest 需逐条绑定 checkpoint、控制 manifest、prompt、参数与输出身份；评测需消费并保存该逐条身份和逐样本指标。

## 未纳入结论的候选项

- **L2 / weight decay、训练 epoch、过拟合。** 没有本轮训练曲线或新实验，不作结论。
- **冻结 LLM 主干、Flow 或 HiFT。** 参数高效微调可以是合理设计；静态图不足以证明下游模块是当前瓶颈。
- **FiLM 仅在文本 embedding 注入必然不足。** 代码确认条件只显式注入文本 embedding，后续 speech token 依靠同一 decoder 的因果状态间接承接条件；但静态图不能证明信号在深层消失，也不能把多层 AdaLN/逐步 speech FiLM 写成已验证修复。它们是直接增强创新点的候选设计，应在局部控制评测下比较。
- **句级恒定条件使调制“退化”为全局仿射。** 对整句同标签，逐 token FiLM 得到相同 scale/shift 是输入条件本身恒定的自然结果；词级标签变化时实现仍会逐 token 变化，不能据此定罪。
- **emotion 与 intensity embedding 相加、共享投影必然造成有害纠缠。** 代码能确认两者当前不可独立分解，但加性条件编码本身是有效架构选择；是否需要独立投影、门控或正交约束须由可控性/组合泛化实验判定。
- **FiLM 零初始化导致冷启动失败。** projection 的零初始化使初始调制为恒等映射，是保护预训练模型的合理 warm start；没有参数轨迹或消融，不能断言 5 epoch 后仍接近零。
- **词级伪标签必然错误。** P1-5 已确认其监督粒度与不确定性限制；但没有逐词真值，不能断言每一条预测错误。
- **同源伪标签 + hard CE 无 label smoothing 单独构成“确认偏置失效”。** 同一 emotion ID 同时作为 FiLM 条件和输入端 CE target 的核心问题已由 P0-1 覆盖；label smoothing 不能修复标签回读路径，也不能补回词级真值。
- **ESD 平行文本重叠。** 是否构成泄漏取决于目标是未见文本泛化，还是同文本不同 emotion rendition 的控制；本轮不扩大为确定缺陷。

## 下一轮探索的推荐顺序（不构成实施授权）

1. **把无效输入端 CE 直接替换为下游情感监督。** 从与目标 span 对齐的 speech-token hidden state 或生成声学片段预测 emotion/intensity，任务头联合训练或使用已校准的声学教师；不要把“只移除模块的降级基线”作为最终修正。
2. **锁定单一训练—部署条件协议。** 对当前非流式产品，统一单流 target-only、prompt/target 条件序列、emotion mask 和停止合同；prompt 是否参与 LLM 条件必须在训练与推理同时成立。
3. **修复监督合同与优化合同。** IEMOCAP 局部标签保留 soft target、置信度和边界；随机 emotion 模块与预训练 llm_decoder 使用真实参数分组，scheduler 配置必须与运行行为一致。
4. **建立局部、强度分层的评测闭环。** 在没有 span/boundary/intensity 指标前，不应把 aggregate 变化解释为可控性提升。
5. **在上述闭环下直接增强 EmoFiLM 注入。** 比较输入层 FiLM、decoder 多层残差/AdaLN 条件化、speech-token 生成步条件化，以及 emotion/intensity 独立参数化；以控制敏感性、边界准确性、强度单调性和可懂度共同验收，而不是先排除模块回退后再决定是否改善。

## 外部资料

以下资料仅用于解释通用机制；本文对当前实现的结论均由上述仓库事实独立成立：

- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, 2018：<https://arxiv.org/abs/1709.07871>
- Bengio et al., *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*, 2015：<https://arxiv.org/abs/1506.03099>
- PyTorch, *CrossEntropyLoss*：<https://docs.pytorch.org/docs/2.13/generated/torch.nn.CrossEntropyLoss.html>

## 核验合并说明

- 两份报告共同命中的输入端标签回读捷径和随机冻结 classifier 已直接确认，并以原 P0-1/P0-2 为主合并证据。
- `2026-07-24-emofilm-model-design-static-analysis.md` 中经确认的新增事实已分别并入 P1-1、P1-4、P1-5、P1-6 与 P2-1。
- 已纠正原报告中“全部标签都来自词级标注器”的范围错误：只有 IEMOCAP 子集使用 `word_annotator_pseudo_label`，ESD 使用 dataset-global label。
- 未保留为确定问题的论断包括：单点 FiLM 必然不足、恒定标签导致实现退化、共享投影必然有害、零初始化导致冷启动失败、无 weight decay/dropout 已造成过拟合。它们均缺少当前运行的消融、曲线或声学证据。
- 本轮所有遗留论断均可由静态证据确认或降级，没有发现必须通过隔离 worktree 动态测试才能判定的项，因此未创建 worktree、未运行模型测试。
