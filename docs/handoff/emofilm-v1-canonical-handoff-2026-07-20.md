# EmoFiLM v1 canonical 迁移交接

日期：2026-07-20

## 当前状态

`CosyVoice-EmoFiLM/` 已切换为唯一 canonical checkout，并以新的无父 `main` 根提交承载已验证代码、精选文档与 tracked 实验摘要。权威数据、模型、checkpoint 和完整实验产物均已位于 canonical 仓库目录内。旧 checkout、linked worktree、外层重复资产和共享遗留状态已按授权清理；`LLM-Audio/` 现在只保留 canonical 仓库。

当前仍需单独授权的后置工作：

1. 后续独立任务统一清除代码、测试、数据、文档和 provenance 中既有 hash/SHA-256 留存。

切换前验收证据：主线 CPU/合同集合 `109 passed`，当时唯一未通过项仅因无父根提交尚未创建而缺少 `git_head`；production `final.pt` strict-load 成功；ESD/FEDD-A/FEDD-B 三条 GPU smoke 均成功；Git 边界仅包含代码、80 个合同文本 metadata、19 个正式实验摘要和 31 个白名单文档。切换后按用户要求未重复运行测试或推理，仅静态确认单根提交、干净工作树、关键资产、三条 smoke 产物和 linked worktree 指针。

远端 `origin/main` 已使用显式 `force-with-lease` 替换为当前单根 `main`；未删除或修改其他远端 branches/tags。

## canonical 执行入口

- 快速开始：`README.md`
- 当前领域语言：`CONTEXT.md`
- 迁移设计：`docs/superpowers/specs/2026-07-20-emofilm-v1-canonical-closeout-design.md`
- 迁移计划：`docs/superpowers/plans/2026-07-20-emofilm-v1-canonical-closeout.md`
- 资产清单：`docs/data_exp_inventory.md`
- 基线实验报告：`docs/reports/2026-07-20-emofilm-v1-baseline-experiment-report.md`
- tracked 实验摘要：`artifacts/emofilm_v1/`

## 历史背景

以下三份 handoff 只用于恢复已发生的决策和数据清理背景，不再作为当前路径或执行命令来源：

- `docs/handoff/emofilm-author-source-regression-handoff-2026-07-18.md`
- `docs/handoff/emofilm-v1-data-cleaning-handoff-2026-07-19.md`
- `docs/handoff/emofilm-v1-decisions-handoff-2026-07-19.md`

历史 command/identity 在 `artifacts/emofilm_v1/` 中按原始字节保存，其中的旧绝对路径只描述运行发生时的现场。未来运行必须使用 `README.md` 和当前脚本中的仓库相对路径。
