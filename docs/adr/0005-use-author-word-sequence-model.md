# 基础主线使用作者 WordSequenceModel

词级伪标签主线固定使用作者 `annotate_data/model.py` 的 WordSequenceModel 与作者 `annotate_data/best_model.pth`，其合同为 emotion2vec-base 768d 输入、5 类情感和 3 维 VAD；emotion2vec 与该模型均冻结，不重新训练。若新标签出现可重复的空标签、文本丢失、类别异常塌缩或与明确数据标签大规模冲突，先分离检查 emotion2vec、MFA、WordSequence 推理和后处理，只有根因证据指向 WordSequenceModel 时才另开修复或重训设计。本地旧 1024d/1D 标注器只进入历史证据包。
