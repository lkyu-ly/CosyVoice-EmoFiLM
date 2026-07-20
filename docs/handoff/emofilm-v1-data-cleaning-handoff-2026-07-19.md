# EmoFiLM v1 数据清理与继续实验交接

日期：2026-07-19

## 下一会话目标

数据合同已经完成一次性清理和重建。下一会话不要继续扩大数据审计或人工听测；先完成必要代码/训练安全核验，取得用户明确授权后执行唯一一次 5-epoch `emofilm_v1` GPU 训练、checkpoint 验证，再按单独授权进行生成与评测。

## 数据清理结论

- 冻结基线不变：train 20,870、cv 1,098、union 21,968。
- 实际训练入口：train 20,774、cv 1,092、union 21,866。
- IEMOCAP 冻结成员 6,848；active tagged 6,746；rejected 102（train 96、cv 6）。
- rejected 构成：101 条整条音频内容未被词级 tagged text 完整覆盖；1 条 `Ses03F_impro05_F015` 的末词 `it` 映射为 0 帧。
- 32 条仅撇号分词等价的样本保留。
- 2,681 个 `tail_clipped` 和 1 个 `tail_empty` 已完整记录。`tail_clipped` 不是独立拒绝原因；有效词块保留。
- rejected 已从 tagged、word blocks、train/cv manifest、src 缓存和所有 parquet 中移除且不补位；各活跃入口残留引用均为 0。
- 原始 WAV、TextGrid、emotion2vec frame 作为来源证据保留，不属于训练残留；没有单样本补词、裁音频或人工标签补丁。
- staging 和中间拒绝文件均已清除。
- 数据门禁最终结果：42 passed；parquet 行数 train 20,774、cv 1,092；ESD/FEDD 评测集仍为 1500/500/500。

## 必须读取的参考资料

设计与执行主线：

1. `docs/handoff/emofilm-v1-decisions-handoff-2026-07-19.md`
2. `docs/superpowers/specs/2026-07-16-emofilm-author-source-regression-design.md`
3. `docs/superpowers/plans/2026-07-16-emofilm-author-source-regression.md`
4. `CONTEXT.md`

本轮数据清洁决策与事实：

1. `docs/adr/0017-unified-iemocap-supervision-cleaning.md`
2. 历史人工审计协议（临时工作文件，未纳入 canonical 白名单）
3. `data/contracts/emofilm_v1/provenance/contract.json`
4. `data/contracts/emofilm_v1/provenance/membership.json`
5. `data/contracts/emofilm_v1/provenance/split_build.json`
6. `data/contracts/emofilm_v1/provenance/iemocap_word_boundaries.jsonl`
7. `data/contracts/emofilm_v1/sources/iemocap/rejected.jsonl`

## 当前代码与产物状态

- worktree：历史执行位置，仅作背景；当前命令从仓库根目录执行
- branch：`rebuild/emofilm-v1`
- 基线 HEAD：`5ad481dadd6b9b8516890335325558539dbae410`
- 存在大量本轮未提交修改和删除；不得 reset、checkout 或覆盖用户改动。
- 尚未 commit、merge、push。
- 尚未启动正式 GPU 训练；`exp/emofilm_v1/` 尚无 checkpoint。
- 数据合同位于 `data/contracts/emofilm_v1/`。

## 继续实验的最短执行顺序

1. 读取上述主线文档和真实 provenance，不重新讨论已收敛的数据质量问题。
2. 运行与训练安全**直接**相关的主线测试和真实 CPU preflight；不要增加低价值数据审查。不进行已做过的测试和已确定完成内容的过度审查，避免浪费时间。
3. 向用户明确列出 GPU、预计时间、命令和输出目录，单独取得 GPU 训练及 checkpoint 写入授权。
4. 执行唯一 5-epoch 训练，输出到 `exp/emofilm_v1/`；长期只保留 `init.pt`、`final.pt`。
5. strict load `final.pt`，对 train/cv 各做最小 finite-loss 前向，并核验可训练参数、optimizer 和 checkpoint 元数据。
6. 再单独取得生成/评测授权，依次执行 3 样本、30 样本和固定 2500 条 ESD/FEDD 生成评测。
7. commit、merge、push 仍分别需要用户明确确认。

## 直接可用的训练输入

- train list：`data/contracts/emofilm_v1/splits/train/parquet/data.list`
- cv list：`data/contracts/emofilm_v1/splits/cv/parquet/data.list`
- 配置：`conf/emo_film.yaml`
- 训练入口：`cosyvoice/bin/train_emo.py`

## Suggested skills

- `$implement`：继续执行既有 spec/plan，不重新设计数据合同。
- `$superpowers:verification-before-completion`：训练、checkpoint 和评测结论前使用新鲜证据核验。
- `$superpowers:systematic-debugging`：仅在测试、训练或推理出现真实异常时使用。
- `$code-review`：训练前只针对未完成代码和训练合同做必要审查，避免重复数据审计。
- `$superpowers:finishing-a-development-branch`：所有实验验证完成并获授权后讨论 commit/merge。
