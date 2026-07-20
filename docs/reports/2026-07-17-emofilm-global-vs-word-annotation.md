# Emo-FiLM：全局标签与词级标注模型核验

**日期：** 2026-07-17
**核验范围：** 作者论文、学位论文、作者公开源码、本地旧数据流程和现有 checkpoint。
**结论置信度：** 高；但作者仓库中的 `emotion2vec_base.pt` 是 0 字节占位文件，因此无法核验作者实际训练时使用 checkpoint 的逐字节身份。

## 1. 结论

论文中的“Global emotion labels are automatically assigned using emotion2vec-plus-large”与作者公开词级标注源码使用 `emotion2vec_base` **不构成直接矛盾**，因为它们描述的是两条不同链路：

| 链路 | 输入/输出粒度 | emotion2vec 的用途 | 公开材料指定的模型 |
|---|---|---|---|
| FEDD 全局标签与一致性检查 | 一条语音 → 一个整句标签/检查结果 | 直接做 utterance 级标签或特征一致性检查 | 论文明确写 `emotion2vec-plus-large` |
| Emo-FiLM 词级伪标注 | 语音 → 帧特征 → MFA 切成单词 → 每词 emotion/intensity | 只作为冻结的帧特征提取器，后接单独的 WordSequenceModel | 论文只写 `emotion2vec`；作者源码明确指向 `emotion2vec_base`，下游 checkpoint 也要求 768 维输入 |

因此，当前“作者源码回归”计划使用 `emotion2vec-base + 作者 best_model.pth`，是为了复现作者公开的**词级标注链路**；它不要求删除或替代 FEDD 上用于**全局打标/一致性校验**的 plus-large。

另一个容易混淆的点是：本地此前并没有重新训练 `emotion2vec-plus-large` 本体。本地代码用冻结的预训练 plus-large 提取 1024 维帧特征，真正训练的是后面的 `WordSequenceModel`。本地 checkpoint 与作者 checkpoint 的实测结构分别为：

```text
本地 checkpoints/word_sequence_model/best.pt:
  attention input = 1024, classification = 5×1024, regression = 1×1024

作者 annotate_data/best_model.pth:
  attention input = 768, classification = 5×768, regression = 3×768
```

这两个 checkpoint 不是可互换的同一个标注器。

## 2. 论文证据：plus-large 明确用于 FEDD 全局标签/检查

arXiv 论文 §3.1 在介绍 FEDD 的 1000 条测试样本、500 条平滑过渡和 500 条突变之后写道：

> “Global emotion labels are automatically assigned using emotion2vec-plus-large.”

来源：[2509.20378v1.pdf](../../reference/2509.20378v1.pdf)，PDF 第 3 页，§3.1。

这里的主语是前文的 FEDD 样本，输出被明确称为 **Global emotion labels**。它没有说 plus-large 用作词级预测模型的帧特征 backbone。

学位论文给出了更不含糊的说明：FEDD 的所有样本使用 emotion2vec-plus-large 做“特征提取与标签一致性校验”，并且这些样本与训练数据隔离、只用于测试。来源：[23S136160-王思睿.pdf](../../reference/23S136160-王思睿.pdf)，PDF 第 29 页（印刷页 20）。

本地旧流程对此的实现也是 utterance 级用途：`label_fedd_emotion2vec.py` 的说明明确写“FEDD 全局情感打标 + 一致性校验”，调用 `granularity="utterance"`，默认模型为 `iic/emotion2vec_plus_large`，并检查预测的整句标签是否属于目标过渡情感对。来源：`tools/label_fedd_emotion2vec.py` 第 1、42、88 行。

## 3. 论文证据：词级链路没有指定 plus-large

arXiv §2.1 对词级标注链路的描述是：

1. 用 `emotion2vec` 提取帧级情感特征；
2. 用 MFA 将帧与单词对齐；
3. 聚合每个词对应的帧序列；
4. 用分类头预测情感类别、回归头预测强度。

来源：[2509.20378v1.pdf](../../reference/2509.20378v1.pdf)，PDF 第 2 页，§2.1。

arXiv §3.2 又说明，IEMOCAP 和 ESD 的**句级全局标签**被用来弱监督一个词级预测模型，再由该模型产生细粒度语音—文本对；但该段仍只写 frame-level features，没有指定 base 或 plus-large。来源：同一 PDF 第 3 页，§3.2。

学位论文也采用相同口径：§2.3.1 只写用 `emotion2vec` 提取帧级特征；实验设置则写 IEMOCAP 和 ESD 的句级标签训练词级预测模型，训练后只自动标注 IEMOCAP。来源：[23S136160-王思睿.pdf](../../reference/23S136160-王思睿.pdf)，PDF 第 23 页和第 28 页。

所以论文在词级链路上存在的是**模型变体细节省略**，而不是明确指定 plus-large 后又被源码反驳。

## 4. 作者源码证据：公开的词级链路是 base + 768d SER

作者公开入口把词级标注流水线参数固定为：

- `emotion2vec_base.pt`；
- 作者随仓库附带的 fairseq upstream；
- `best_model.pth` 词级 SER checkpoint。

来源：`reference/Emo_PA_code_data/annotate_data/run_ncssd_annotation.py` 第 84 行。

具体流水线通过 fairseq 加载该 checkpoint，调用 `extract_features(...)["x"]` 得到帧特征；SER 默认 `input_dim=768`，对每个 MFA 单词区间独立取帧块，并输出 5 类情感及 valence/arousal/dominance 三维值。来源：`reference/Emo_PA_code_data/annotate_data/pipeline_word_emotion.py` 第 56、177、213、233 行。

作者模型定义也明确是默认 768 维输入、5 类分类头和 3 维回归头。来源：`reference/Emo_PA_code_data/annotate_data/model.py` 第 5 行。作者的 `best_model.pth` 参数形状与该定义完全一致。

作者仓库 README 把这个 base 资产指向官方 `iic/emotion2vec_base`，并说明 base 是“未经微调的基座模型”，支持 50 Hz 帧级特征。来源：`reference/Emo_PA_code_data/models/emotion2vec_base/README.md` 第 24、33、42 行。

限制：作者仓库内的 `annotate_data/emotion2vec_base.pt` 是 0 字节，因此只能确认作者公开代码的预期模型族、接口和维度，不能证明公开占位文件就是作者训练时的原始 checkpoint。

## 5. 本地旧流程与作者流程的实际区别

本地旧帧特征脚本默认加载 `iic/emotion2vec_plus_large`，以 `granularity="frame"` 提取特征，并根据音频实际时长反推 fps。来源：`tools/extract_emotion2vec_frame.py` 第 1、17、45 行。

随后本地训练脚本新建一个 `input_dim=1024` 的 WordSequenceModel；真正被 Adam 更新的是这个下游模型，而不是 plus-large。来源：`tools/train_annotator.py` 第 157、206、211 行。

两条词级链路的主要差异是：

| 项目 | 作者公开链路 | 本地旧链路 |
|---|---|---|
| 帧特征 extractor | emotion2vec-base | emotion2vec-plus-large |
| 帧特征维度 | 768 | 1024 |
| downstream checkpoint | 作者 `best_model.pth` | 本地 `checkpoints/word_sequence_model/best.pt` |
| intensity 输出 | 3D VAD，取 arousal | 当前 checkpoint 为 1D arousal |
| checkpoint 是否兼容 | 只接收 768d | 只接收 1024d |

plus-large 更大或更新，并不自动意味着“接在作者 768d SER checkpoint 前面会更好”。二者的接口维度不同，且下游 SER 的训练分布随 extractor 一起变化。若继续使用 plus-large，就必须继续使用或重新训练本地 1024d SER；这会保留一条由本地论文解释形成的实现分支，而不是回归作者公开链路。

## 6. 为什么当前修复计划选择作者标注器

选择依据不是“base 一定比 plus-large 准”，目前没有公开证据支持这种质量排序。依据是：

1. **任务目标是定位本地复现偏差。** 作者模型在作者训练/推理路径上的已有对照表现更好，因此最短的可证伪路线是先把可见接口回归作者源码。
2. **作者 checkpoint 与 base 是一个不可拆的合同。** `best_model.pth` 明确接收 768 维并输出 3D VAD，不能直接接本地 1024 维 plus-large 特征。
3. **避免再次训练一个未公开细节的替代标注器。** 保留本地 plus-large 路线意味着还要判断训练标签、回归维度、随机划分、epoch、平滑等多个选择；这会把“源码回归”重新变成新的实验设计。
4. **不会破坏论文中 plus-large 的正确用途。** FEDD 的 utterance 级全局打标/一致性校验可以继续保留 plus-large；计划替换的是 IEMOCAP 训练监督所用的词级标注路径。

本地旧 checkpoint 并非被判定为无价值或错误；它只是与作者公开词级标注器不是同一系统。它应作为历史产物保留，不应混入本次“作者源码回归”主线。

## 7. 最准确的一句话表述

> 论文明确要求 emotion2vec-plus-large 用于 FEDD 测试集的整句标签/一致性检查；词级伪标注部分只泛称 emotion2vec。作者公开源码进一步表明，其词级帧特征和已发布 SER checkpoint 的合同是 emotion2vec-base、768d、50 Hz、5 类 + 3D VAD。因此两种模型承担不同职责，不矛盾；当前计划选择 base 是为了回归作者公开词级标注实现，而不是断言 base 的一般性能优于 plus-large。
