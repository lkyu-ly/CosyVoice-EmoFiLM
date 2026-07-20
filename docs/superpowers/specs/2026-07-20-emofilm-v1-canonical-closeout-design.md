# EmoFiLM v1 Canonical 收尾设计

## 目标

将已经完成训练、生成、听测和评测的 `rebuild/emofilm-v1` 工作树重建为唯一 canonical checkout：代码与现有错误 `main` 解耦，全部权威数据、模型和实验产物落在 `CosyVoice-EmoFiLM/` 内，并以新的无父 `main` 根提交作为后续优化基线。

## 最终布局

```text
CosyVoice-EmoFiLM/
├── CONTEXT.md
├── README.md
├── artifacts/emofilm_v1/       # Git 跟踪的小型正式实验摘要
├── data/contracts/emofilm_v1/  # 完整合同；文本 manifest/meta 跟踪，大型派生资产忽略
├── datasets/{ESD,IEMOCAP}/      # 原始数据，非 Git
├── pretrained_models/           # 运行模型，非 Git
├── checkpoints/                 # WordSequence checkpoint，非 Git
├── exp/emofilm_v1/              # 完整正式实验，非 Git
├── emofilm_original/            # 作者发布模型证据，非 Git、非运行依赖
├── reference/                   # 作者源码与论文证据，非 Git、非运行依赖
└── docs/                        # 精选当前主线文档
```

## 资产边界

### Git 跟踪

- 当前 worktree 中除 `.serena/`、缓存和实验目录外的有效代码、测试与配置；删除状态也按当前 worktree 生效。
- 数据合同中的 JSON/JSONL/list/scp/text/utt2spk 等 manifest、索引与 provenance；不跟踪 PT、WAV、TextGrid、tar/parquet 和对齐分析缓存。
- `artifacts/emofilm_v1/`：训练/生成/评测命令与 identity、`init.yaml`、`resolved.yaml`、三个合并生成 manifest、三组 metrics、comparison、final verification 和实验报告。
- 新 README、CONTEXT、ADR 0001–0018、2026-07-16 主线 spec/plan、三份主线 handoff、新 canonical handoff、inventory、实验报告及三份作者语义/来源审计。

### 仓内但不跟踪

- `datasets/`、完整 `data/contracts/emofilm_v1/` 二进制资产、`pretrained_models/`、`checkpoints/`、完整 `exp/emofilm_v1/`、`emofilm_original/`、`reference/`。
- `exp/emofilm_v1/` 保留正式 checkpoint、完整 2500 条 WAV、日志、正式 manifest、identity、metrics 和 smoke；排除中止的 `confirmation/` 与可重建的 `eval_refs/` 符号链接视图。
- 活跃 WordSequence 默认入口使用 `checkpoints/word_sequence_model/author_best_model.pth`；该文件由作者 `best_model.pth` 原样复制，避免运行代码依赖 `reference/`。

### 不进入 canonical

- `.serena/`、`.scratch/`、`.superpowers/`、`__pycache__/`、`.pytest_cache/`、`test/tmp.txt`。
- 旧 stage、消融、错误 `main`、训练故障排查和 debug 文档/代码。
- 旧 `CONTEXT.md`、旧 `CLAUDE.md`、仓内 `docs/stage0_completion.md`。

## 路径与追溯规则

- 活跃代码、测试、脚本、README 和保留文档只使用仓库相对路径、CLI 参数或环境变量，不包含用户目录、旧主目录或 worktree 绝对路径。
- 历史 command/identity 原样保留旧绝对路径，因为它们描述已经发生的运行，不是未来命令模板。
- 新增 `artifacts/emofilm_v1/canonical_path_mapping.json`，仅记录旧资产角色与新 canonical 相对位置，不写 hash。
- 本次不计算、记录或比较文件 hash/SHA-256；既有 identity/provenance 中的 hash 字段保持原样，后续另立任务统一清理。

## 构建与验收

先在同级 `CosyVoice-EmoFiLM.canonical-next/` 构建 candidate，绝不直接覆盖现目录。验收只使用：复制命令成功状态、源/目标文件数与总大小、关键资产存在性、测试、production checkpoint strict-load 和 ESD/FEDD-A/FEDD-B 三样本 GPU smoke。不得重训或重跑 2500 条。

candidate 通过后，依次设置独立危险操作门禁：创建新的无父 `main` 根提交；将旧目录改名隔离并把 candidate 切换为 canonical；清理旧目录/worktree/共享遗留资产；最后才决定是否 force-push 远端 `main`。任何后一阶段失败都不得触发前一份有效资产的删除。
