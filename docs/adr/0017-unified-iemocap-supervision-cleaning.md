---
status: accepted
---

# 对 IEMOCAP train/cv 使用统一监督配对清洁规则

全量机器普查和提示后人工听测确认，2,681 条仍有有效末词帧的尾部量化裁剪不是坏数据，应保留并显式记录；真正需要排除的是完全空词区间，以及整条音频内容未被词级 tagged text 完整覆盖的样本。冻结 train/cv 使用同一规则，从原 split 扣除且不补位：精确覆盖和仅撇号切分等价的配对保留，其他文本遗漏统一 rejected；不再使用仅限 train、总量不超过 1% 的预算，也不做逐样本补词、裁音频或人工赋情感标签。原始 WAV、TextGrid 和 frame 来源资产保留，派生 word blocks、tagged、split 缓存和 parquet 不得引用 rejected 成员。
