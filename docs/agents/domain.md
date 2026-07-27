# Domain Docs

本文件定义工程技能探索本仓库时读取和使用领域文档的规则。

## Before exploring, read these

- 根目录 `CONTEXT.md`。
- `docs/adr/` 中与当前工作相关的 ADR。
- 如果将来出现根目录 `CONTEXT-MAP.md`，则按其映射读取与当前工作相关的上下文文档。

缺失的文档不构成阻塞；继续使用已有代码、测试和报告证据。

## File structure

本仓库采用单上下文布局：

```text
/
├── CONTEXT.md
├── docs/adr/
└── source and test modules
```

## Use the glossary's vocabulary

规格、票据、测试名称与设计说明必须使用 `CONTEXT.md` 定义的领域术语，并避免使用其中明确列出的同义词。

若所需概念尚未进入领域词汇，先判断是否正在引入项目不使用的语言；确属领域缺口时，再通过领域建模流程补充。

## Flag ADR conflicts

任何与现有 ADR 冲突的规划必须显式指出冲突及重开该决策的理由，不得静默覆盖。
