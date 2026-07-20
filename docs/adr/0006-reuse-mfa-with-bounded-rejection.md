---
status: superseded by ADR-0017
---

# 复用现有 MFA 并限制训练样本跳过

作者没有公开基础 IEMOCAP 实验使用的 MFA 词典、声学模型和版本，因此本地继续使用现有 IEMOCAP TextGrid，不因后续 NCSSD 编排中的 `english_us_arpa` 全量重跑。只允许跳过无法产生有效词边界的 IEMOCAP 训练样本，总量上限为 manifest 的 1%，每条必须记录原因；失败若超过上限或集中于某个 speaker/情感类别，则停止并修复 MFA。ESD/FEDD 测试、reference 和 prompt 的任何缺失均为硬失败。
