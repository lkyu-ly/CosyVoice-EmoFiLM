# Input-end 句级情感监督恢复为可选路径（ADR-0021）

- status: accepted
- supersedes: ADR-0019 中"输入端 classifier CE 被取代 / 死配置字段含
  emo_loss_weight"相关条款（仅该部分）

## 背景

句级 loss_emotion 为 v1 设计（git 锚点 `9c6d84b`）。ADR-0019 因输入端标签回读
捷径将其删除并以下游 span 监督取代。为检验"加回句级监督能否突破 Emo-SIM~66
平台"（用户决策，2026-08-02 handoff），恢复该路径作为可选监督做对照实验。

## 决策

- `emotion_classifier`（`Linear(llm_input_size, emotion_vocab_size)`）**恒构造、
  冻结**（训练期探针），`emo_loss_weight>0` 时才计入 loss；disabled 基线
  loss_dict 不含 `loss_emotion_input`。
- input-end 句级监督与 downstream span 词级监督**可叠加**：loss =
  loss_tts + w_e·loss_emotion_span + w_i·loss_intensity + emo_w·loss_emotion_input；
  loss 键名按路径分离，避免日志/聚合歧义。
- checkpoint 策略：`emotion_classifier` 属"训练期专用模块"——base 加载允许缺失、
  trained 加载允许缺失（旧 disabled ckpt）；trained 加载仍拒绝
  `emotion_head` / `arousal_head` 缺失（v1 制品防冒充守卫保留）。
- 死配置字段集合：移除 `emo_loss_weight`（`DEAD_CONFIG_KEYS` 已同步为
  `{mix_ratio, alpha}`）。

## 已知风险（沿用 2026-07-22 审计 P0-1/P0-2，不视为本次实验的阻断项）

- 目标标签同时作为 FiLM 输入，存在标签回读捷径；该 CE 证明"FiLM 后文本表示可读
  出条件 ID"，不直接证明声学输出遵从情感。
- 冻结随机读出器给上游的梯度无语义锚点。
- 本路径定位为对照实验；span 监督仍是细粒度控制的长期主路径（ADR-0019 方向
  未变）。
