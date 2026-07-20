# EmoFiLM v1 Canonical Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从已验证 worktree 构建、验收并切换唯一 canonical checkout，同时完整保留非 Git 数据、模型和正式实验资产。

**Architecture:** 在同级 candidate 目录完成代码清理、资产复制、文档收口和验证；candidate 通过后才创建新 Git 根并切换目录。删除旧资产和替换远端历史分别使用后置授权门禁。

**Tech Stack:** Bash、rsync、Git、Python 3.10、PyTorch、pytest、CUDA、CosyVoice/EmoFiLM 本地主线工具。

---

## 固定路径

```bash
WORKSPACE="$(cd .. && pwd)"
SOURCE_CODE="${SOURCE_CODE:?set SOURCE_CODE to the verified emofilm-v1 worktree}"
SOURCE_ASSETS="${SOURCE_ASSETS:-${WORKSPACE}/CosyVoice-EmoFiLM}"
CANDIDATE="${CANDIDATE:-${WORKSPACE}/CosyVoice-EmoFiLM.canonical-next}"
CANONICAL="${CANONICAL:-${WORKSPACE}/CosyVoice-EmoFiLM}"
PYTHON="${PYTHON:-python}"
```

### Task 1: 建立无破坏 candidate

**Files:**
- Create: `CosyVoice-EmoFiLM.canonical-next/`
- Exclude: `.git/`, `.serena/`, `exp/`, caches

- [ ] **Step 1: 执行危险操作确认**

确认范围为：创建约 40GB candidate 并批量复制；不移动、不删除、不提交。

- [ ] **Step 2: 检查空间与来源状态**

```bash
test ! -e "${CANDIDATE}"
df -BG "${WORKSPACE}"
git -C "${SOURCE_CODE}" status --short
```

Expected: candidate 不存在；可用空间大于 80GB；worktree 改动与已验证基线一致。

- [ ] **Step 3: 复制当前代码状态**

```bash
mkdir -p "${CANDIDATE}"
rsync -a --exclude='.git' --exclude='.serena' --exclude='exp' \
  --exclude='__pycache__' --exclude='.pytest_cache' \
  "${SOURCE_CODE}/" "${CANDIDATE}/"
```

Expected: 已删除的旧诊断/消融文件不会出现；新文件和未提交代码全部出现。

### Task 2: 消除活跃路径耦合

**Files:**
- Modify: `scripts/activate_env.sh`
- Modify: `tests/test_emo_cli_regression.py`
- Modify: `tests/test_emo_inference_fake.py`
- Modify: `tests/test_emo_processor.py`
- Modify: `tests/test_emo_tokenizer.py`
- Modify: `tests/test_emofilm_data_contract.py`
- Modify: `tests/test_generate_tagged_jsonl.py`
- Modify: `tests/test_index_esd_iemocap.py`
- Modify: `tests/test_mfa_align_batch.py`
- Modify: `tools/download_iemocap.py`
- Modify examples only: `tools/build_fedd_part_b_v2.py`, `tools/inference_emo_film.py`, `tools/run_inference_parallel.py`

- [ ] **Step 1: 为仓库相对路径写失败测试**

新增 `tests/test_canonical_paths.py`，扫描活跃 Python、shell、README 和精选 docs，禁止出现用户目录绝对路径或旧 worktree 字符串。明确排除 `artifacts/emofilm_v1/`、`exp/emofilm_v1/` 和 `data/contracts/emofilm_v1/provenance/` 中的历史记录。

- [ ] **Step 2: 确认测试先失败**

```bash
cd "${CANDIDATE}"
"${PYTHON}" -m pytest \
  tests/test_canonical_paths.py -q
```

Expected: FAIL，并准确列出当前活跃硬编码文件。

- [ ] **Step 3: 最小修复路径解析**

统一采用 `Path(__file__).resolve().parents[...]` 获取 repo root；数据默认位置使用 `repo_root / "datasets"`；Python/MFA 可执行文件优先使用 `sys.executable`、`shutil.which("mfa")` 或显式环境变量。`activate_env.sh` 仅从脚本位置推导根目录，并允许 `CONDA_ROOT`/`CONDA_ENV` 覆盖。

WordSequence checkpoint 默认改为：

```text
checkpoints/word_sequence_model/author_best_model.pth
```

- [ ] **Step 4: 验证路径测试通过**

```bash
"${PYTHON}" -m pytest \
  tests/test_canonical_paths.py \
  tests/test_generate_tagged_jsonl.py \
  tests/test_emofilm_data_contract.py -q
```

Expected: PASS。

### Task 3: 复制权威非 Git 资产

**Files:**
- Create: `data/contracts/emofilm_v1/`
- Create: `datasets/`
- Create: `pretrained_models/`
- Create: `checkpoints/`
- Create: `exp/emofilm_v1/`
- Create: `emofilm_original/`
- Create: `reference/`

- [ ] **Step 1: 复制运行与数据资产**

```bash
rsync -a "${SOURCE_ASSETS}/data/contracts/emofilm_v1/" \
  "${CANDIDATE}/data/contracts/emofilm_v1/"
rsync -a "${SOURCE_ASSETS}/pretrained_models/" "${CANDIDATE}/pretrained_models/"
rsync -a "${SOURCE_ASSETS}/checkpoints/" "${CANDIDATE}/checkpoints/"
rsync -a "${WORKSPACE}/datasets/" "${CANDIDATE}/datasets/"
rsync -a "${WORKSPACE}/emofilm_original/" "${CANDIDATE}/emofilm_original/"
rsync -a "${WORKSPACE}/reference/" "${CANDIDATE}/reference/"
rsync -a --exclude='confirmation/' --exclude='eval_refs/' \
  "${SOURCE_CODE}/exp/emofilm_v1/" "${CANDIDATE}/exp/emofilm_v1/"
```

- [ ] **Step 2: 固化活跃 WordSequence checkpoint**

```bash
cp -p \
  "${CANDIDATE}/reference/Emo_PA_code_data/annotate_data/best_model.pth" \
  "${CANDIDATE}/checkpoints/word_sequence_model/author_best_model.pth"
```

- [ ] **Step 3: 不使用 hash 核对规模**

对每个来源和目标分别执行 `find ... -type f | wc -l` 与 `du -sb`。除有意排除的 `confirmation/`、`eval_refs/` 以及新增的 author checkpoint 副本外，数量和总大小必须对应；将结果写入最终 inventory，不写文件 hash。

### Task 4: 收口文档与正式实验摘要

**Files:**
- Create: `CONTEXT.md`
- Rewrite: `README.md`
- Create: `docs/adr/0001-0018-*.md`
- Create: `docs/superpowers/specs/2026-07-16-emofilm-author-source-regression-design.md`
- Create: `docs/superpowers/plans/2026-07-16-emofilm-author-source-regression.md`
- Create: `docs/superpowers/specs/2026-07-20-emofilm-v1-canonical-closeout-design.md`
- Create: `docs/superpowers/plans/2026-07-20-emofilm-v1-canonical-closeout.md`
- Create: `docs/handoff/` selected handoffs and final canonical handoff
- Create: `docs/reports/` selected author audits and baseline experiment report
- Rewrite: `docs/data_exp_inventory.md`
- Create: `artifacts/emofilm_v1/`

- [ ] **Step 1: 仅复制文档白名单**

复制 ADR 0001–0018、2026-07-16 主线 spec/plan、三份主线 handoff、三份指定作者审计、本 design/plan。把 `docs/canonical-emofilm-v1/CONTEXT.md` 安装为根 `CONTEXT.md`；不复制外层旧 CONTEXT、CLAUDE、agents 文档及其他 stage/debug 文档。

- [ ] **Step 2: 重写运行入口文档**

README 只包含：目录约定、环境激活、数据合同检查、正确 base checkpoint 训练命令、三样本推理、评测入口和正式资产位置。所有命令从 repo root 使用相对路径；明确训练 `--checkpoint pretrained_models/CosyVoice2-0.5B/llm.pt`，不得从 BlankEN 新建模型。

- [ ] **Step 3: 生成 tracked 实验摘要**

精确复制以下内容，不复制日志、分片 manifest、WAV 或 checkpoint：

```text
artifacts/emofilm_v1/train/{training_command.txt,train_identity.json,init.yaml,resolved.yaml}
artifacts/emofilm_v1/generation/{full_generation_command.txt,full_generation_4gpu_resume_command.txt,full_generation_identity.json,inference_esd.jsonl,inference_fedd_a.jsonl,inference_fedd_b.jsonl}
artifacts/emofilm_v1/evaluation/{evaluation_command.txt,evaluation_identity.json,esd_metrics.json,fedd_a_metrics.json,fedd_b_metrics.json,comparison.json}
artifacts/emofilm_v1/{final_verification.json,experiment_report.md,canonical_path_mapping.json}
```

`canonical_path_mapping.json` 只记录资产角色、历史位置和 canonical 相对位置；不含 hash。历史文件内容原样保留。

- [ ] **Step 4: 重写 inventory 与最终 handoff**

inventory 只描述 canonical 内的 tracked/non-Git/排除资产、文件数、总大小和角色。最终 handoff 将旧 handoff 标为历史背景，提供新的唯一执行入口；保留文档中的旧绝对路径全部改为 canonical 相对路径。

### Task 5: 配置新的 Git 边界

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 精确忽略大型资产**

保留数据合同文本 manifest/index/provenance 可跟踪；忽略 `datasets/`、`pretrained_models/`、`checkpoints/`、`exp/`、`emofilm_original/`、`reference/`，以及合同内 PT、WAV、TextGrid、tar/parquet 等大型文件。`artifacts/emofilm_v1/` 不得被忽略。

- [ ] **Step 2: 初始化但暂不提交新历史**

```bash
cd "${CANDIDATE}"
git init -b main
git remote add origin https://github.com/lkyu-ly/CosyVoice-EmoFiLM
git add .
git status --short
```

Expected: staged 内容只有代码、测试、精选文档、合同文本 metadata 和 `artifacts/emofilm_v1/`；无模型、WAV、PT、tar、TextGrid、缓存或 reference 作者资产。

### Task 6: 验收 candidate

- [ ] **Step 1: 运行主线测试，不重复旧 debug 审查**

```bash
cd "${CANDIDATE}"
"${PYTHON}" -m pytest \
  tests/test_emofilm_*.py \
  tests/test_emo_*.py \
  tests/test_generate_tagged_jsonl.py \
  tests/test_index_esd_iemocap.py \
  tests/test_mfa_align_batch.py -q
```

Expected: PASS；允许环境相关测试按既有显式 skip 规则跳过，不接受新失败。

- [ ] **Step 2: 验证关键资产与生产 strict-load**

确认 `llm.pt`、`final.pt`、WordSequence checkpoint、三个 eval manifest、23 个 train/cv parquet tar、2500 个 full WAV 和三组 metrics 存在。使用 `tools/inference_emo_film.py` 的生产加载路径加载 `exp/emofilm_v1/final.pt`；该入口内部必须执行 trained checkpoint strict-load。

- [ ] **Step 3: 执行三样本 GPU smoke**

在一张空闲 GPU 上分别对 ESD、FEDD-A、FEDD-B manifest 执行 `--max_samples 1`，输出到新的 `exp/emofilm_v1/canonical_smoke/`。确认三条 WAV、三个 manifest 行和进程退出码正确；无需重跑 30/2500 条。

- [ ] **Step 4: 执行最终路径与 Git 边界扫描**

```bash
rg -n '/home/|/Users/|superpowers/worktrees' \
  scripts tests tools cosyvoice cosyvoice_emo eval README.md CONTEXT.md docs
git status --short
git diff --cached --stat
```

Expected: 活跃代码和文档零旧路径；仅历史 artifacts/exp/provenance 保留历史位置。Git staged 集合符合 Task 5。

### Task 7: 创建新根并切换 canonical

- [ ] **Step 1: 单独请求 Git commit 与目录移动确认**

该确认必须同时列出：创建无父根提交；把现 `CosyVoice-EmoFiLM` 改名为隔离目录；把 candidate 改名为 canonical。此阶段仍不删除任何旧资产，不 push。

- [ ] **Step 2: 创建唯一根提交**

```bash
cd "${CANDIDATE}"
git commit -m "baseline: establish verified EmoFiLM v1 canonical repository"
test "$(git rev-list --max-parents=0 HEAD | wc -l)" -eq 1
test "$(git rev-list --count HEAD)" -eq 1
```

- [ ] **Step 3: 原子切换目录**

```bash
mv "${CANONICAL}" "${WORKSPACE}/CosyVoice-EmoFiLM.legacy-main-20260720"
mv "${CANDIDATE}" "${CANONICAL}"
```

- [ ] **Step 4: 在最终路径做最小复验**

从 canonical 路径重新运行路径测试、关键资产存在性、production strict-load 和三样本 smoke。全部通过后生成最终 canonical handoff；不重训、不重评 2500 条。

### Task 8: 后置清理与远端替换

- [ ] **Step 1: 输出精确删除清单并再次确认**

清单包括但不限于：旧 main 隔离目录、旧 worktree、外层 `datasets/`、`emofilm_original/`、`reference/`、旧共享 docs/CONTEXT/CLAUDE、`.scratch/`、`.superpowers/`、旧 backup/bundle/branch/tag。只有确认 canonical 已完整承接对应资产后才能删除。

- [ ] **Step 2: 删除后复验 canonical**

重复文件数/总大小库存、主线测试、strict-load 和三样本 smoke；不得依赖已删除路径。

- [ ] **Step 3: 单独请求远端历史替换确认**

确认后才使用带 lease 的 force-push 将新单提交 `main` 替换远端 `main`。远端 branch/tag 清理按精确列表执行；不在本地切换或资产迁移确认中隐含授权。

## 明确延期

- 清除代码、测试、数据、文档和 provenance 中现有 hash/SHA-256 字段与逻辑。
- 进一步整理不优雅但不阻碍 canonical 独立运行的工具链细节。
- 任何新训练、2500 条重新生成或全量重新评测。
