# EmoFiLM v2 冻结声学 Evaluator 合同

本文是 EmoFiLM v2 ticket 08 交付的 evaluator 接口、候选、校准状态、适用限制
与**自证风险**的单一可读来源。权威实现：``eval/acoustic_evaluators.py``。

MAP §3 evaluator 不变量要求：emotion/arousal evaluator 冻结、与训练任务头隔离；
记录标签空间 / 采样率 / 帧率 / 窗口 / 语义 / 限制 / 校准；与 IEMOCAP 弱监督
生成器共享模型则标自证风险。

---

## 1. 接口契约

### EmotionEvaluator / ArousalEvaluator

两个 ``runtime_checkable Protocol``（``eval/acoustic_evaluators.py``）：

| 方法 / 属性                          | 返回     | 语义                                                                              |
| ------------------------------------ | -------- | --------------------------------------------------------------------------------- |
| ``is_frozen``                        | ``bool`` | 恒 True。包装对象``requires_grad=False``，不复用训练下游任务头。                  |
| ``identity()``                       | ``dict`` | 完整身份记录（见下方字段表）。                                                    |
| ``predict_frames(wav_path_or_clip)`` | ``dict`` | 逐帧输出：emotion →``(T, K)`` 分布（每帧和为 1）；arousal → ``(T,)`` 标量轨迹。 |

### ``identity()`` 必需字段

| 字段                                            | 类型        | 语义                                                                    |
| ----------------------------------------------- | ----------- | ----------------------------------------------------------------------- |
| ``model_id``                                    | str         | 模型标识（如``emotion2vec-base``、``fake-acoustic-evaluator``）。       |
| ``revision``                                    | str         | 冻结快照标签。                                                          |
| ``name``                                        | str         | 合同 Evaluator TypedDict 字段；评测器显示名。                           |
| ``version``                                     | str         | 合同 Evaluator TypedDict 字段；冻结版本。                               |
| ``label_space``                                 | list[str]   | 标签空间（emotion: 5 类；arousal: 空）。                                |
| ``label_mapping``                               | dict / None | 标签名→数组索引映射（emotion 有；arousal None）。                      |
| ``sample_rate_hz``                              | int         | 输入采样率（emotion2vec: 16000）。                                      |
| ``frame_rate_hz``                               | float       | 输出帧率（emotion2vec: 50Hz）。                                         |
| ``window_strategy``                             | str         | 窗口/步幅策略描述。                                                     |
| ``output_semantics``                            | str         | 输出含义说明（含训练分布合规性）。                                      |
| ``known_limitations``                           | list[str]   | 已知限制列表。                                                          |
| ``calibration``                                 | dict / None | ``{method, version, set}`` 或 None（未校准）。                          |
| ``self_evidence_risk``                          | bool        | 合同 Evaluator TypedDict 字段；与 IEMOCAP 弱监督生成器共享模型时 True。 |
| ``shares_source_with_iemocap_weak_supervision`` | bool        | 与 IEMOCAP 弱监督标注器共享上游模型/特征来源时 True。                   |

### ``predict_frames()`` 输出格式

**emotion**（``EmotionEvaluator``）:

```
{
    "frames": np.ndarray (T, 5),      # 每帧 5 类情感分布，和为 1
    "frame_rate_hz": float,
    "times_sec": np.ndarray (T,),
    "label_space": ["ang", "hap", "neu", "sad", "sur"],
}
```

**arousal**（``ArousalEvaluator``）:

```
{
    "frames": np.ndarray (T,),        # 每帧 arousal 标量
    "frame_rate_hz": float,
    "times_sec": np.ndarray (T,),
}
```

**禁止键**：``"confidence"`` 永远不得出现（MAP §3：诚实字段是
``raw_score`` + ``calibration``）。``assert_output_honest()`` 强制检查。

---

## 2. 自证风险（关键）

### 风险链

```
emotion2vec-base (768d/50Hz 帧特征)
        │
        ▼
WordSequenceModel (cosyvoice_emo/emo_annotator.py)
    ├── classification_head → 5 类情感
    └── regression_head → 3D VAD (含 arousal)
        │
        ▼
IEMOCAP 词级弱监督标签 (句级广播, tools/generate_tagged_jsonl.py)
        │
        ▼
EmoFiLM v1/v2 训练目标
```

**emotion2vec-base 768d 特征是 IEMOCAP 弱监督标注器（WordSequenceModel）
的输入**。因此用 emotion2vec+WordSequenceModel 做 evaluator 验收 IEMOCAP
训练结果，**评测器与训练标签共享上游模型/特征来源**，构成自证
（self-evidence）。

### 影响

- **emotion 验收**带同源偏差：evaluator 与训练标签来源相同，对 emotion
  分类的验收结论不能视为完全独立的第三方验证。
- **arousal 验收**同理：arousal 来自 WordSequenceModel 的 VAD[1]，是标注器
  的副产物。
- **强度真值**：未经独立校准的 arousal score **不得命名为强度真值**
  （MAP §3）。evaluator 未校准时 ``calibration=None``，只能用于**相对控制
  响应**验证（如「高 rank → 高均值」方向性），不能做绝对验收。

### 缓解

1. evaluator 的 ``shares_source_with_iemocap_weak_supervision=True`` 和
   ``self_evidence_risk=True`` 在 ``identity()`` 中如实记录，下游
   EvaluationRow.evaluator 继承该标记。
2. 验收结论区分**相对控制响应**（如 emo_sim 高于基线、transition 可定位）
   与**绝对真值**（如「arousal=0.7」）。前者可用同源 evaluator；后者需要
   独立来源。
3. 强度结论限定为方向性（``validate_arousal_direction``）而非绝对值。
4. 长期：引入外部独立 evaluator（见 §4 所需外部资产）。

---

## 3. Evaluator 候选清单与门禁状态

### 3.1 FakeAcousticEvaluator — 门禁 PASS（合成数据）

| 属性                                            | 值                                                       |
| ----------------------------------------------- | -------------------------------------------------------- |
| ``model_id``                                    | ``fake-acoustic-evaluator``                              |
| ``kind``                                        | ``emotion`` / ``arousal``                                |
| ``is_frozen``                                   | True                                                     |
| ``calibration``                                 | None                                                     |
| ``shares_source_with_iemocap_weak_supervision`` | False                                                    |
| ``self_evidence_risk``                          | False                                                    |
| CPU                                             | 是（不加载任何模型）                                     |
| 门禁状态                                        | **PASS on synthetic data**（供 09/10/12 测试消费） |

用途：接口契约测试、方向性/transition/arousal 校验逻辑的行为测试、
09/10 评测管线集成测试。**不用于真实音频验收。**

### 3.2 Emotion2VecEmotionEvaluator — 门禁 NOT MET

| 属性                                            | 值                                                           |
| ----------------------------------------------- | ------------------------------------------------------------ |
| ``model_id``                                    | ``emotion2vec-base``                                         |
| 上游                                            | emotion2vec-base 768d/50Hz                                   |
| 下游头                                          | WordSequenceModel.classification_head                        |
| ``is_frozen``                                   | True（注入的 model 需``.eval()`` + ``requires_grad=False``） |
| ``calibration``                                 | None（未校准）                                               |
| ``shares_source_with_iemocap_weak_supervision`` | **True**                                               |
| ``self_evidence_risk``                          | **True**                                               |
| 门禁状态                                        | **NOT MET**                                            |

**不满足门禁的原因**：

1. **架构不匹配**：WordSequenceModel 是 utterance-level 池化架构
   （``forward()`` 对时间维做 mean pooling 后再过分类头）。逐帧应用
   ``classification_head`` 于未池化的 768d 特征**超出训练分布**，输出
   不具备校准语义。
2. **自证风险**：classification_head 与 IEMOCAP 弱监督标注器共享同一
   WordSequenceModel checkpoint，emotion 验收结论带同源偏差。

### 3.3 Emotion2VecArousalEvaluator — 门禁 NOT MET

| 属性                                            | 值                                         |
| ----------------------------------------------- | ------------------------------------------ |
| ``model_id``                                    | ``emotion2vec-base``                       |
| 上游                                            | emotion2vec-base 768d/50Hz                 |
| 下游头                                          | WordSequenceModel.regression_head (VAD[1]) |
| ``is_frozen``                                   | True                                       |
| ``calibration``                                 | None                                       |
| ``shares_source_with_iemocap_weak_supervision`` | **True**                             |
| ``self_evidence_risk``                          | **True**                             |
| 门禁状态                                        | **NOT MET**                          |

**不满足门禁的原因**：同 §3.2（池化架构 + 自证），且 arousal 仅为 VAD[1]
副产物，无独立校准。

### 3.4 emotion2vec_plus_large（未包装，仅记录）

`tools/label_fedd_emotion2vec.py` 使用 ``iic/emotion2vec_plus_large``
做 FEDD utterance-level 标签一致性校验（9→5 类映射）。该模型是不同
checkpoint，**不直接用于 IEMOCAP 弱监督链**，可论证
``shares_source_with_iemocap_weak_supervision=False``。但：

- 仅支持 utterance-level 分类（``granularity="utterance"``），**不支持
  frame-level**；
- 同属 emotion2vec 模型族，存在方法论同源风险；
- 输出是离散标签 + score，不是 frame-level 分布/arousal 轨迹。

结论：**不满足 frame-level emotion/arousal 门禁**，但可作为 utterance-level
标签一致性检查的独立来源（用于 FEDD 标签验证，已有先例）。

---

## 4. 所需外部资产（门禁未满足时的缺口）

要满足独立、经校准的 frame-level emotion/arousal 评测门禁，需要以下外部
资产之一：

### 4.1 独立 frame-level emotion 分类器

- **要求**：在 IEMOCAP 之外训练的、支持 frame-level 或滑窗 emotion 分类的
  模型，输出 (T, 5) 分布。
- **候选方向**：wav2vec2-based emotion recognition（如 HuggingFace
  ``ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition``）、
  SpeechBrain emotion recognition、或独立训练的 emotion2vec 下游头。
- **校准**：需要在带已知 emotion 的参考集上完成 label mapping 验证
  （``validate_emotion_label_mapping``）和 transition 定位验证
  （``validate_transition_localization``）。

### 4.2 独立 arousal / VAD 回归器

- **要求**：在 IEMOCAP 之外训练的、支持 frame-level 或滑窗 arousal 回归的
  模型，输出 (T,) 标量轨迹。
- **候选方向**：基于 wav2vec2 的 VAD 回归、或专用的 voice activity/arousal
  模型。
- **校准**：需要在可排序参考集上完成方向性验证
  （``validate_arousal_direction``）和绝对值校准（如与人类评分对齐）。

### 4.3 下载/新增模型规则

任何下载或新增模型必须另行遵守项目网络与资产确认规则（MAP §0；
不在本票范围内）。本票不下载任何新模型。

---

## 5. 方向性校验逻辑（纯函数）

以下纯函数可在 CPU 上对任何实现了 ``EmotionEvaluator`` / ``ArousalEvaluator``
接口的对象运行（含 Fake），用于在接入真实模型前验证其行为：

| 函数                                                        | 验证内容                                                      | 输入                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------ |
| ``validate_emotion_label_mapping(evaluator, clips)``        | 已知 emotion 参考片段上 argmax 与已知标签一致（不止凭模型名） | ``[SyntheticReferenceClip(known_emotion=...)]``        |
| ``validate_transition_localization(evaluator, clips, tol)`` | 逐帧输出能定位已知 transition，时间偏差 ≤ 容差               | ``[SyntheticReferenceClip(known_transition_sec=...)]`` |
| ``validate_arousal_direction(evaluator, clips)``            | arousal 在可排序参考上单调递增                                | ``[SyntheticReferenceClip(known_arousal_rank=...)]``   |

所有函数返回 ``{passed: bool, details: ...}``，不修改 evaluator 状态。

---

## 6. 下游消费指南（09/10/12）

- **09（span 评测）**：消费 ``EmotionEvaluator`` 接口 + ``FakeAcousticEvaluator``
  做集成测试。真实音频验收需要外部 evaluator（见 §4）；当前用 emotion2vec
  wrapper 仅做 smoke，不作为正式门禁。
- **10（边界评测）**：消费 ``validate_transition_localization`` 做边界定位
  验收；Fake 用于测试管线，真实 transition 验收需要满足门禁的 evaluator。
- **12（强度评测）**：消费 ``validate_arousal_direction`` 做方向性验收；
  绝对强度真值需要校准后的独立 arousal evaluator（见 §4.2）。

EvaluationRow.evaluator 字段从 ``identity()`` 派生，``self_evidence_risk``
继承到 EvaluationRow，供 aggregate 分离同源/异源验收结论。
