---
status: accepted
---

# 从已验证的 emofilm_v1 基线重建 canonical checkout

当前 `rebuild/emofilm-v1` 工作树中的代码、`emofilm_v1` 数据合同、正式模型和完整实验已经通过训练、生成、听测与评测验证；现有 `main` 实现错误且不再作为恢复来源。项目将先创建同级 candidate checkout，完整承接已验证基线，再将其替换为唯一 canonical checkout，并创建不继承旧 Git 历史的无父 `main` 根提交。

## 已确认边界

- canonical checkout 承载活跃代码、原始 ESD/IEMOCAP、完整数据预处理合同、运行所需预训练模型、WordSequence checkpoint，以及训练、生成和评测正式产物；大型资产保留在仓库目录中但不纳入 Git。
- Git 跟踪代码、精选主线文档、manifest，以及复制到 `artifacts/emofilm_v1/` 的小型 provenance、metrics、identity 和实验报告；大型模型、WAV、parquet 和其他二进制资产保持忽略。
- `emofilm_original/` 与 `reference/` 作为作者发布模型、源码和论文的只读证据迁入 canonical checkout，但不纳入 Git，也不作为活跃代码或测试的运行依赖。
- `artifacts/emofilm_v1/` 精确跟踪正式训练、生成、评测命令与 identity、resolved config、三个合并生成 manifest、metrics、comparison、final verification 和实验报告；日志、分片 manifest、WAV、checkpoint 与临时 confirmation/smoke 不进入该 tracked 摘要。
- README 重写为 EmoFiLM v1 quickstart。新的 `CONTEXT.md` 从零生成于 canonical 仓库根，不继承共享根中错误 `main`、debug 或旧实验内容。
- 文档采用当前主线白名单：保留 ADR 0001–0018、2026-07-16 主线 spec/plan、三份主线 handoff、新 canonical handoff、inventory、实验报告，以及 2026-07-16/17 三份作者语义与来源审计；不保留错误 `main` 实现、问题排查、debug、旧 stage、消融或已被替代的过程文档。
- 活跃代码、测试、脚本和新文档不得依赖用户目录、旧主目录或 worktree 绝对路径。迁移后命令必须从 canonical checkout 可直接执行。
- 不为迁移计算、记录或对比文件 hash/SHA-256。验收采用复制命令成功状态、目录文件数和总大小、关键资产存在性、模型 strict-load、主线测试、路径扫描及三样本 smoke。
- 历史运行身份原样保留其“运行发生时”的事实，包括旧绝对路径；它不作为迁移后的可执行配置。迁移另写不含 hash 的轻量路径映射，当前可执行命令则单独以相对 canonical checkout 的 quickstart/脚本维护。
- candidate 验收完成后，旧目录、旧 worktree、bundle、branch、tag 和共享遗留文件仍须按精确清单另行授权清理；取消 hash 验收不改变这一门禁。

后续另有一项独立清理任务：移除数据、测试、文档和 provenance 中既有的 hash/SHA-256 留存。该工作不属于本次 canonical 替换，当前不改写既有实验记录。
