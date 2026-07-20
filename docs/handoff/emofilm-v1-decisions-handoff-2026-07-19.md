# Emo-FiLM v1 重构决策交接

日期：2026-07-19（Asia/Shanghai）

## 目的

本文件记录本轮已经由用户明确确认、会影响后续实现和数据生产的决策。它是跨会话恢复用的决策摘要；详细模型语义和原始审计事实仍以以下仓库文档为规范来源：

- `docs/handoff/emofilm-author-source-regression-handoff-2026-07-18.md`
- `docs/superpowers/specs/2026-07-16-emofilm-author-source-regression-design.md`
- `docs/superpowers/plans/2026-07-16-emofilm-author-source-regression.md`
- `docs/data_exp_inventory.md`
- `CONTEXT.md`

## 当前代码状态

- 代码仓库：仓库根目录
- 当时隔离 worktree：历史执行位置，仅作背景，不再是当前运行入口
- 当前分支：`rebuild/emofilm-author-v1`
- 当前 HEAD：`5ad481dadd6b9b8516890335325558539dbae410`
- 计划中的新分支/worktree 名称：`rebuild/emofilm-v1`；截至本交接尚未执行改名。
- 之前的代码重构改动和删除标记均属于本轮工作，不得 `reset --hard`、`checkout` 或覆盖。
- 尚未 commit、merge 或 push。
- shared `main` 目前未承载本轮代码修改；main 只作为基线和恢复点。

## 1. 主线命名和身份

用户确认：本轮成果正式命名为项目自身的 `emofilm_v1`，不再把 `author` 放进活跃代码、配置、测试、数据或实验目录命名。

约定：

- 数据：`data/contracts/emofilm_v1/`
- 实验：`exp/emofilm_v1/`
- 构建入口：`tools/build_emofilm_contract.py`
- 活跃测试使用 `test_emofilm_*` 命名。
- `author` 只出现在 provenance、审计报告、设计/交接文档中，用于说明语义来源和可追溯性。
- 该实现虽然基于作者公开实现的有效语义，但从本轮验收后作为本项目原生主线；后续以本项目需求和稳定数据合同为准，不把作者未来代码或论文更新作为自动约束。

## 2. 主线合入策略

用户确认：本轮隔离分支完成代码、数据合同、训练 preflight、smoke 和最终验证后，作为后续开发基线合入 `main`。

- `main` 后续直接代表 `emofilm_v1` 主线。
- 旧基线保留 tag/bundle 作为回退依据。
- `git commit`、合入 `main`、`git push` 是不同危险操作，分别在执行前确认。
- 在最终验证前不删除旧实验回退资产。

## 3. 旧数据生命周期

用户确认：新版生成期间旧数据不覆盖；新版本通过后再按精确路径生成清理/归档清单。

- 原始 ESD/IEMOCAP WAV 暂留原位。
- 旧派生数据、旧 full parquet、旧训练 checkpoint、v6 WAV、旧实验结果目前不自动删除。
- 旧派生数据在新版主线验证成功后可列为清理候选，但必须逐路径列出并单独确认。
- 历史报告、备份、作者源码证据、旧实验结果单独列清单，用户重新核验后决定；当前不自动删除。
- 文档若列入删除候选，必须先制作备份，再执行删除。
- 已确认可以删除的精简测试和相关文件，也必须先形成精确清单并备份后再删除。
- 已删除的 MFA 日志是本轮唯一已执行的额外删除：见第 8 节。

## 4. 精简生产校验

用户确认：生产流程不再执行低概率、重复或人为构造的复杂负向测试；代码级合同测试可继续精简，确认属于废弃范围的测试/文件可以删除。

生产硬门禁只保留：

1. emotion2vec 抽样输出为 768d、50Hz，且 provenance/checkpoint/upstream 信息正确；
2. WordSequence checkpoint strict load，768d 输入、5 类分类、3D VAD；
3. 全量 train/cv 成员关系正确且互斥；
4. IEMOCAP rejected 输出原 split、speaker/emotion 和原因分布；旧的 train-only/1% 预算已由 ADR-0017 取代；
5. train/cv parquet 的 `data.list` 可直接加载，且不共享 shard；
6. 三样本 smoke：ESD、FEDD-A、FEDD-B 各一条。

不作为数据生产前置门禁：

- 人为删除/增加 checkpoint key 的负向测试；
- EOS、RAS、aux token 每个边界情形的重复生产验证；
- 对全部 2500 条评测资产重复静态检查；
- 为低概率异常增加复杂 fallback 或隐式跳过。

运行异常必须显式失败，不得静默回退到旧数据。

## 5. 唯一数据入口和目录

用户确认：彻底切换为单一 `emofilm_v1` 数据树，不保留旧路径 fallback、软链接、双 schema 或兼容适配层。

```text
data/contracts/emofilm_v1/
├── sources/
│   ├── iemocap/
│   │   ├── manifest.jsonl
│   │   ├── rejected.jsonl
│   │   ├── mfa/train/
│   │   ├── emotion2vec_base/
│   │   ├── word_blocks/
│   │   └── tagged.jsonl
│   ├── esd/
│   │   ├── manifest.jsonl
│   │   ├── mfa/{train,test}/
│   │   └── tagged.jsonl
│   └── fedd/
│       ├── manifest/{part_a,part_b}.jsonl
│       ├── target_wav/{part_a,part_b}/
│       └── prompts/
├── splits/
│   ├── train/
│   │   ├── manifest.jsonl
│   │   ├── src/
│   │   └── parquet/
│   └── cv/
│       ├── manifest.jsonl
│       ├── src/
│       └── parquet/
├── eval/{esd,fedd_a,fedd_b}/
└── provenance/
```

分类原则是“来源优先，最终 split 打包”：

- `sources/<dataset>` 保存来源级 manifest 和该来源的派生产物；
- `splits/train`、`splits/cv` 是唯一训练入口，允许跨来源合并；
- `eval/*` 保存规范化评测清单，不复制已有 target/reference/prompt 音频；
- provenance 保存来源、生成方式、复用/重生成状态、成员和产物摘要。

## 6. 原始输入和路径规则

用户确认：

- `datasets/ESD/` 和 `datasets/IEMOCAP/` 的原始 WAV 暂留原位；不复制、不移动、不建软链接。
- 模型留在 `pretrained_models/` 或 `reference/`，不放进 `data/`。
- `data/mfa_alignments/` 是前期数据处理产物，迁移到新版来源目录；不再作为运行入口。
- 新版 manifest 使用工作区相对路径，例如 `datasets/ESD/0011/Angry/0011_000671.wav`。
- provenance 记录实际工作区根目录、绝对路径、来源统计和快速校验。
- 代码只解析新版 manifest；旧 `data/raw_manifests`、`data/tagged_jsonl`、`data/src`、`data/parquets` 不再被主线读取。

## 7. MFA 迁移和已执行删除

确认的迁移映射：

- `data/mfa_alignments/iemocap_train/*` → `data/contracts/emofilm_v1/sources/iemocap/mfa/train/`
- `data/mfa_alignments/esd_train/*` → `data/contracts/emofilm_v1/sources/esd/mfa/train/`
- `data/mfa_alignments/esd_test/*` → `data/contracts/emofilm_v1/sources/esd/mfa/test/`
- 迁移完成后删除空的旧 `data/mfa_alignments/` 目录；不删除任何 TextGrid。

用户明确要求删除 MFA 日志。已于本轮精确删除并核验不存在：

```text
data/mfa_alignments/logs/esd_test.log
data/mfa_alignments/logs/esd_train.log
data/mfa_alignments/logs/iemocap_train.log
```

删除前 SHA-256：

- `esd_test.log`: `22fc142eed867bfac6d6963d91a3841d5945237d390aa3f2bf37567a7434dd93`
- `esd_train.log`: `15570bd7a45ad4b6f86a0625a7dd5aa6b4b18ca78340b379c2d9e508ce33fb45`
- `iemocap_train.log`: `5eb5261f084bee95adab735a7db3d201c98d4fc4f83516abe54f561de2a8fa1c`

## 8. 复用、重建和数据集边界

### IEMOCAP

必须重建：

- emotion2vec-base frames：旧 `data/emo_feats` 在当前工作区不存在；新版需要 `(T,768)`、50Hz/20ms；
- word blocks：旧 `data/word_blocks` 在当前工作区不存在；
- 词级 tagged：旧 `data/tagged_jsonl/iemocap_train.jsonl` 只有 6848 条，来自旧 1024d 标注器链，旧代码默认 smoothing，且缺少新版 provenance，不可复用。

来源级 manifest 使用当前 raw manifest 的 6987 条并规范化路径。旧实际 split 只覆盖其中 6848 条；额外 139 条保留在 source manifest，不进入 train/cv。

### ESD

可以复用并规范化：

- `data/raw_manifests/esd_train.jsonl` + `esd_test.jsonl` 合并为 `sources/esd/manifest.jsonl`；
- `data/tagged_jsonl/esd_train.jsonl` + `esd_test.jsonl` 合并为 `sources/esd/tagged.jsonl`；
- ESD Global Label 已符合新版语义，不重新推理，只改为相对路径和规范目录；
- `eval/esd/manifest.jsonl` 从规范化 test 条目生成，固定 1500 条。

当前 ESD source raw train 15127 条，但冻结 split 只含 15120 条；额外 7 条 source-only，不进入 train/cv。

### FEDD-rebuilt

- 当前本地重建评测集为 1000 条：Part A 500、Part B 500；不称为原始 FEDD。
- 现有 manifest、tagged manifest、Part A prompt、target WAV、neutral anchor 作为本地评测资产归档到 `sources/fedd/`；不重新生成，除非验证发现确切错误。
- `eval/fedd_a`、`eval/fedd_b` 只存规范化清单和 provenance，不复制 target/prompt WAV。

### 可复用缓存

旧 `data/src/src_full` 的 speech token、utterance embedding、speaker embedding 覆盖恰好冻结 union 的 21,968 条：

- speech token：每 utterance；
- utterance embedding：192d；
- speaker embedding：20 个 speaker，192d。

这些缓存可以逐 ID、shape、有限值和覆盖率校验后复用并按新版 `splits/train`、`splits/cv` 规范组织。旧 parquet 的文本、成员组织和 shard 不复用；新版 parquet 必须重新打包。

## 9. 冻结成员关系

用户确认严格冻结当前实际训练 split，不重新随机切分、不增加 speaker-disjoint：

- train：20,870 条；
- cv：1,098 条；
- union：21,968 条；
- train/cv 互斥。

当前 raw source manifest 总数为 22,114 条：IEMOCAP 6,987 + ESD train 15,127。与冻结 union 的差异为 146 条：IEMOCAP 139、ESD 7。差异样本保留在 source manifest，但不静默加入 split。

2026-07-19 数据质量清洁后，冻结基线不变，实际训练入口为：

- active train：20,774 条；
- active cv：1,092 条；
- active union：21,866 条；
- IEMOCAP rejected：102 条，其中 train 96、cv 6；不补位。

IEMOCAP rejected 规则（ADR-0017）：

- 冻结 train/cv 使用相同监督配对清洁规则；
- 完全空词区间或音频内容未被 tagged text 完整覆盖时 rejected；
- 精确覆盖和仅撇号切分等价的配对保留；
- `tail_clipped` 若仍有有效词块则保留并显式记录；
- 每条有 `utt_id/reason/speaker_id/sentence_emotion/original_split`；
- 输出 speaker/emotion 分布，不得明显集中；
- ESD/FEDD test/reference/prompt 任何缺失都 hard-fail。

当前冻结成员的 MFA 已完整：IEMOCAP 6848/6848、ESD train 15120/15120；额外 source-only IEMOCAP 中有 94 条缺 TextGrid，额外 ESD 7 条缺 TextGrid，但它们不属于冻结 split。

## 10. 分层完整性校验

用户确认大型数据不做全量内容 SHA-256，采用分层快速校验：

必须全量 SHA-256：

- 模型 checkpoint；
- 作者 fairseq upstream 目录 hash；
- 核心 manifest；
- `contract.json`、`sources.json`、`membership`；
- train/cv `data.list`。

大型派生目录只记录：

- 文件数；
- 总字节数；
- ID 集合 hash；
- schema/shape 统计；
- 固定样本和边界文件抽样 hash。

生产时立即验证：文件存在、可读取、shape/schema、数量、有限值、ID 覆盖和 split 关系。不要使用 mtime 作为完整性依据，不引入旧数据 fallback。

## 11. 版本规则

- 当前 `emofilm_v1` 生产、验证和修复期间允许原地修复；只重建受影响数据及其下游产物。
- 本轮验收完成后，改变数据集、切分、标签、特征模型、MFA、过滤规则或数据语义时新建 `emofilm_v2`。
- 只改 metadata/provenance 可在 v1 内更新。
- 训练/评测运行身份绑定 contract hash，不只绑定目录名。

## 12. 后续执行顺序和授权边界

尚未执行：

1. 更新规范文档、plan、inventory 中的旧 `author_v1` 命名；
2. 将 worktree/branch 改名为 `rebuild/emofilm-v1`；
3. 移动 MFA TextGrid 到新版来源目录；
4. 规范化 ESD/FEDD/IEMOCAP source manifests；
5. 生成 IEMOCAP frames、word blocks、词级标签；
6. 复用并组织 speech token/embedding；
7. 生成冻结 split manifests、src 和 train/cv parquet；
8. 执行精简正向数据合同验证。

仍需单独门禁：

- GPU 训练和大型 checkpoint 写入；
- 3→30→2500 全量生成和评测；
- 精简测试/旧数据/旧文档的逐路径删除；
- Git commit、合入 main、push。

## Suggested skills

- `implement`：按现有 plan 继续执行。
- `superpowers:executing-plans`：读取更新后的 plan，按任务推进。
- `superpowers:using-git-worktrees`：完成 `rebuild/emofilm-v1` 隔离工作区改名/核验。
- `superpowers:test-driven-development`：修改数据合同和入口前先写最小正向合同测试。
- `superpowers:systematic-debugging`：真实数据、训练、推理出现异常时先定位根因。
- `superpowers:verification-before-completion`：任何完成/通过结论前运行新鲜验证。
- `code-review` 或 `superpowers:requesting-code-review`：训练前做 spec/standards 双轴审查。
- `superpowers:finishing-a-development-branch`：所有实现和验证完成后决定如何合入。

## 下一会话第一步

1. 读取本文件和四份规范文档，不重新讨论已确认决策。
2. 先核验 worktree/branch 当前实际名称和未提交改动。
3. 更新规范文档中的命名和数据树，再执行 MFA 迁移。
4. 不删除旧派生数据，直到新版数据合同和 smoke 通过且用户另行确认。
