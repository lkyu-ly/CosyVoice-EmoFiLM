# 新版数据沿用现有 train/cv 成员关系

`emofilm_author_v1` 沿用当前基础 TTS 数据中每个 ESD/IEMOCAP utterance 的 train 或 cv 归属，继续排除 ESD 1500 条测试样本及既有 FEDD-B 训练源，只替换 IEMOCAP 词级标签并按新版 processor 重新打包。允许拒绝的少量 IEMOCAP 样本从原集合移除但不补位，不重新随机切分，也不新增论文未要求的 speaker-disjoint 约束；合同记录两集合的 utterance ID 和哈希。
