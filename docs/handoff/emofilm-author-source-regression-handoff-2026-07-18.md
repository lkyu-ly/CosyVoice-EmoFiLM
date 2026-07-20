# Emo-FiLM 作者语义回归与干净重构交接

日期：2026-07-18

下一会话目标：按照已批准的设计和实施计划开始执行；先完成只读的 Task 1，生成精确备份、证据和清理清单，在任何 Git 写操作、删除、下载或训练前重新取得明确授权。

## 当前状态

- 调研、作者源码边界审计、设计、ADR 和详细实施计划已经完成并通过文档自检。
- 设计状态为 `Ready for agent`，代码实施尚未开始。
- 代码仓库当前为 `main@5ad481dadd6b9b8516890335325558539dbae410`。
- 历史作者控制矩阵代码位于 `feature/emofilm-author-model-control@e265281ab7927a4f9aa4793b4d66974db5f24b76`，未合入 main。
- 当前代码仓库状态为 `main...origin/main [ahead 1]`，仅有未跟踪 `.serena/`；不要删除或吸收该目录，除非用户另行说明。
- 项目根目录本身不是该代码仓库；Git 操作只能明确指向 `CosyVoice-EmoFiLM/`。外层 `docs/`、`CONTEXT.md` 和 ADR 不受该仓库的 `git diff` 覆盖。
- 尚未执行代码重构、数据删除、Git tag/bundle、模型下载、训练、全量生成、commit 或 push。

## 必读交付材料

以下路径相对于项目根目录：

1. 短规格：`.scratch/emofilm-author-source-regression/spec.md`
2. 详细设计：`docs/superpowers/specs/2026-07-16-emofilm-author-source-regression-design.md`
3. 实施计划：`docs/superpowers/plans/2026-07-16-emofilm-author-source-regression.md`
4. 本地/作者完整对齐审计：`docs/superpowers/reports/2026-07-16-emofilm-author-local-alignment-audit.md`
5. 作者源码公开边界：`docs/superpowers/reports/2026-07-17-emofilm-author-source-boundary.md`
6. 历史矩阵最终报告：`docs/superpowers/reports/2026-07-15-emofilm-author-control-final.md`
7. 数据与实验资产清单：`docs/data_exp_inventory.md`
8. 领域术语：`CONTEXT.md`
9. 决策记录：`docs/adr/0001-*.md` 至 `docs/adr/0016-*.md`

不要在本交接中推断未写出的实现细节；以上文档是规范来源，实施计划中的精确文件、断言、门禁和命令优先。

## 调研结果总述

### 作者公开边界

作者仓库不是基础 Emo-FiLM 从原始数据到论文表格的一键复现包。它公开了核心模型、训练前向、冻结逻辑、通用 SFT 引擎、专用推理、WordSequenceModel 结构与 checkpoint，以及部分词级标注推理流水线；没有公开基础论文的真实数据 manifest、WordSequenceModel 训练、正式 SFT parquet、实际训练运行、原始 FEDD、正式批量推理 population，以及 WER/Emo-SIM/DTW 主表实现。

因此交付明确分为三层：作者公开源码语义、本地合理补全、工程可靠性 hardening。不得把本地 FEDD、MFA、批量推理或代理指标称为作者官方实现。

### 必须回归的作者语义

- 词级标注使用外部官方 emotion2vec-base 的 768d/50Hz 特征，并通过作者随包 fairseq upstream 验证加载。
- 使用作者 `annotate_data/model.py` 与 `best_model.pth`：5 类情感、3D VAD；不训练 emotion2vec 或 WordSequenceModel。
- 不做旧三词平滑；相邻词仅在 emotion 和 intensity 都相同的情况下合并。
- emotion loss 必须读取 FiLM 后文本表示。
- 仅 emotion encoder、FiLM adapter、LLM decoder 可训练；classifier 冻结但参与 loss。
- token/mel 按 1:2 联合裁剪；static batch 4；采用公开 filter 和 Adam 1e-5。
- 推理使用 lowercase target、target-only LLM 条件、`min_len=2×text token`、`max_len=200`、EOS-before-min 原分布重采样、EOS-only 停止、辅助 special token 跳过、作者 RAS fallback 分布。

### 不需要复刻的执行轨迹

保留 KV cache、当前可用 ONNX provider、现有 shuffle/sort 等数学或统计语义等价实现；不要求 CPU provider、完整前缀重算、逐样本 seed、相同随机数消费顺序或逐 token/逐字节一致。EOS 重采样和 RAS fallback 会改变候选概率分布，不能归入这一类，仍需修复。

### 模型结构证据

- Qwen2 主干规格为 hidden 896、24 层、14 attention heads、2 KV heads、FFN 4864。
- EmotionEncoder 为 emotion `(6,896)` 与 intensity `(4,896)` 两个 embedding 相加。
- FiLM projection 为 `896→1792`；classifier 为 `896→6`；speech token head 为 `6561+3`。
- 作者 WordSequence checkpoint 的静态 tensor 形状与 768d、8 heads、FFN `768→3072→768`、分类 `768→5`、VAD `768→3` 一致。
- 作者有效 FiLM forward 不使用序列化 `alpha`；它只应作为旧 checkpoint 兼容项，不应产生新架构。
- 本地旧 WordSequence 路径固定为 1024d，历史训练还出现 1D 强度头，与作者 checkpoint 不兼容，应退出活跃主线。

### 已确认的本地实现问题

- 本地默认 emotion loss 接入 decoder hidden，而作者接入 FiLM 后表示。
- 本地额外训练 emotion classifier，作者公开训练入口未解冻 classifier。
- 本地 `compute_fbank` 不接受 YAML 已传入的 `token_mel_ratio`，存在真实接口缺口。
- 本地推理混入 prompt text/prompt speech、使用长度比例上限和不同停止语义。
- 本地 EOS mask 与作者原分布重采样不同；本地 RAS fallback 会预先排除首次重复 token。
- checkpoint 加载过于宽松、generator 可能遗留 UUID 状态、parquet 异步异常可能丢失、训练和数据打包存在重复大文件。
- 旧 `per_emo_accuracy.py` 依赖 1024d plus-large/旧 WordSequence，不进入本轮正式验收。

### 数据和评测裁决

- 沿用当前 train/cv utterance 归属，不重新随机切分。
- 复用现有 IEMOCAP MFA；只允许少量 IEMOCAP 训练样本进入 rejected manifest，并遵守设计文档中的分布约束。ESD/FEDD test、reference、prompt 不允许缺失。
- 冻结本地 ESD 1500、FEDD-A 500、FEDD-B 500、prompt/anchor、批量推理和 `emofilm-eval-v2`。
- 最终只报告技术完整性、WER、Emo-SIM、DTW，并与旧 local final/local identity 做分区相对比较。
- 不设任何绝对数值失败阈值；不自动续训、不建新矩阵、不增加消融。
- 只训练一个论文口径的 5-epoch 模型；5 epoch 是本地代理预算，不是作者公开 YAML 合同。作者 YAML 写 200 epoch，论文/学位论文写 5 epoch，这一冲突无法从公开材料闭合。

## 实施计划任务总述

详细步骤以实施计划为准。计划共有七个任务：

### Task 1：固化身份并生成精确清单

只读核验 main/feature commit 和工作区状态；建立第一阶段资产 JSONL 清单、SHA-256、可重建来源及精简证据包规格；准备而不执行 tag/bundle、源码快照和隔离 worktree 命令。完成后必须分别申请 Git 写操作/worktree 与精确删除清单的危险确认。

### Task 2：执行已批准的备份和第一阶段清理

仅在明确授权后创建 tag、bundle、源码快照、证据归档并验证可恢复；只删除清单逐路径批准的资产；随后创建 `rebuild/emofilm-author-v1` 隔离 worktree。不得使用宽泛 glob，不得在共享 main 上实施代码改动。

### Task 3：以最小测试面定向重建核心代码

先写训练、推理、运行时合同测试并验证旧实现失败，再重建/精简 Emo-FiLM model、frontend、train、checkpoint 与生命周期代码；局部修复 processor、tokenizer 和采样。测试固定有效拓扑、shape、loss seam、冻结集合、token/mel 接口、target-only condition、长度、EOS/aux/RAS 和资源清理，不保留旧消融及墓碑测试。

### Task 4：构建单一 `emofilm_author_v1` 数据合同

在授权后取得并记录官方 emotion2vec-base；生成 IEMOCAP 768d frames、word blocks、作者 WordSequence 伪标签、ESD Global Label、冻结 train/cv manifest，并分别直接打包 train/cv parquet。所有 provenance、rejected、成员 hash、shard hash 和加载检查都写入版本目录。

### Task 5：训练唯一 5-epoch 模型

先做不更新参数的 preflight；记录不可变运行身份；经 GPU 训练授权后只训练一个模型。长期只保留 `init.pt` 和 `final.pt`，训练时单一覆盖 `latest.pt`；final 必须 strict load 并通过 train/cv 最小前向。

### Task 6：完成 `3→30→2500` 本地代理验收

运行主线测试；ESD/FEDD-A/FEDD-B 各一条 smoke；每分区 10 条 confirmation；经全量授权后用 6 GPU 完成固定 2500 条生成和三项指标评测。只做相对分区报告，不设绝对门槛。

### Task 7：收口主线并提出第二阶段清理

从唯一数据、训练、推理、评测入口做可达性审查；删除仍不可达代码仍须另列精确危险清单；运行最终验证，更新报告和 inventory。新版成功后只提出旧 full parquet、v6 WAV、旧 init/final 的保留、离线或删除选项，不在同一步自动处置。

## 资产与安全门禁

- 第一阶段候选和预计空间收益已经记录在 `docs/data_exp_inventory.md`，但这不是实际删除授权。
- 正式 v6 WAV、旧 `init.pt/final.pt` 和旧 full parquet 在新版成功前必须保留。
- 作者随包三个 `emotion2vec_base.pt` 均为 0 字节，不得误用；下载外部官方资产需要单独确认并记录 revision/SHA。
- 文件删除、批量移动、Git tag/commit/push、创建隔离分支/worktree、下载大型模型、GPU 训练和全量生成都应按项目危险操作格式请求确认。
- 用户要求主线 KISS/YAGNI：不要增加新矩阵、消融、后备模型、OOF、per-emotion accuracy、自动延长、per-sample seed 或复杂 trace 体系。
- 用户要求失败样本若数量少且不影响测试结果可跳过；具体只落在已批准的 IEMOCAP 训练 rejected 合同内，不能扩展到测试/reference/prompt。

## 下一代理的第一步

1. 完整阅读短规格、详细设计、实施计划、审计报告、作者边界报告和 inventory。
2. 调用计划执行与 worktree 技能，但本次先只执行 Task 1 的只读部分。
3. 重新核验两个 commit 和当前状态，不信任交接中的陈旧状态。
4. 生成精确备份命令文件、资产 manifest 和证据包清单；不要创建 tag/bundle、worktree 或删除文件。
5. 向用户展示两个清晰的危险操作确认：Git 备份/隔离 worktree，以及第一阶段逐路径删除。

## Suggested skills

- `superpowers:using-superpowers`：先确定必须遵循的技能流程。
- `superpowers:executing-plans`：按现有书面实施计划逐任务执行并设置检查点。
- `superpowers:using-git-worktrees`：在取得授权后创建隔离实施 worktree。
- `superpowers:test-driven-development`：Task 3 和 Task 4 必须先验证失败测试，再做最小实现。
- `superpowers:systematic-debugging`：任何测试、数据、训练或推理异常先定位根因，不做猜测性补丁。
- `superpowers:verification-before-completion`：每个任务和最终结论必须提供新鲜验证证据。
- `superpowers:requesting-code-review`：核心重建完成后、训练前进行一次规范与规格双轴审查。
- `superpowers:subagent-driven-development`：只有在实施计划允许且任务文件互不冲突时并行；共享工作区下必须避免并发编辑同一文件。

## 已完成验证

本轮文档验证已确认：交付文件非空、16 个 ADR 从 0001 连续到 0016、本地 Markdown 链接有效、代码块外无未解决占位符、无尾随空白；作者源码中的 768d/5 类/3D、FiLM 后 classifier、`max_len=200`、static batch 4 和 YAML 200 epoch 等事实已重新核验。下一会话仍应对即将依赖的源码事实做快速、针对性复核，避免把本交接当作运行时事实缓存。
