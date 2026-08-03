# 调整算法前收尾（判别指标增强 + v3 中性基线 + v1 兼容精简）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让主线评测具备可用的情感判别维度（合并 reference_wav），跑出 longepoch/sentlvl 两个模型的 v3 全量中性基线，并精简 v1 相关兼容残留。

**Architecture:** 判别指标在 `eval/emotion_metrics.py::compute_discriminability` 内部合并 eval manifest 的 `reference_wav`（目标情感参考）与 sources 索引（其他情感参考）；评测资源用共享探测脚本动态决策 GPU/whisper 设备；v1 兼容只做删除与注释收窄，不改加载守卫。

**Tech Stack:** Python 3.10 / torch / funasr / whisper / pytest / bash（nvidia-smi 探测）。

**执行约定（用户约束，替代 writing-plans 模板的 commit 步骤）：**
- 不主动 git commit；工作树即交付物。每个 Task 收尾 = 跑验证命令确认通过。
- 敏捷 + 精简：严禁过度设计与过度保护；仅判别合并逻辑（核心算法）加 2 个测试；旧/冗余测试冲突直接删除。
- 所有新代码/注释使用中文。
- 环境：`/home/hanlvyuan/miniconda3/envs/emofilm/bin/python`，`PYTHONPATH=.:third_party/Matcha-TTS`，6×RTX3090（共享集群，可能被占）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `eval/emotion_metrics.py`（修改） | `compute_discriminability` 合并 reference_wav |
| `eval/eval_emo_film.py`（修改） | `load_manifest` 增加 reference_wav 字段（绝对路径） |
| `tests/test_emotion_metrics.py`（修改） | 新增 2 个判别合并核心测试 |
| `exp/emofilm_film_only_longepoch/run_eval.sh`（修改） | 动态资源 + v3 评测 |
| `exp/emofilm_sentlvl/run_eval.sh`（修改） | 同上 |
| `cosyvoice/utils/emo_checkpoint.py`（修改） | 注释收窄（TRAINED_ALLOWED_MISSING 语义） |
| `tools/inference_emo_film.py`（修改） | docstring 示例从 v1 改为当前模型 |
| `docs/contracts/emofilm_v3_eval.md`（修改） | 判别指标口径更新（合并 reference_wav 后 eval 集可算） |
| `exp/prompt_match/run_infer.sh`、`run_eval.sh`（修改） | 移除 v1 分支 + GPU 环境变量（现场决定） |
| `exp/prompt_match/run_eval_robust.sh`（删除） | 上一轮 OOM 重试补丁，动态/现场决策取代，删除 |
| `docs/reports/2026-08-03-emofilm-eval-v3-execution-report.md`（修改） | 第 132 行字面宿主路径改写（消除 test_canonical_paths 失败） |
| `docs/reports/2026-08-04-v3-neutral-baseline.md`（新建） | longepoch/sentlvl v3 基线结果 |

---

## Task 1: 判别指标增强（核心 TDD）

**Files:**
- Modify: `eval/emotion_metrics.py`（`compute_discriminability`）
- Modify: `eval/eval_emo_film.py`（`load_manifest`）
- Test: `tests/test_emotion_metrics.py`

- [ ] **Step 1: 写失败测试（2 个核心测试）**

在 `tests/test_emotion_metrics.py` 末尾追加：

```python
class _MapEmoModel:
    """按路径映射返回 utterance/frame 向量的 fake（判别测试用，不读音频）。"""

    def __init__(self, vec_by_path):
        self.vec_by_path = vec_by_path

    def generate(self, inp, **kw):
        lst = [inp] if isinstance(inp, str) else list(inp)
        if kw.get("granularity") == "utterance":
            return [{"feats": self.vec_by_path[p]} for p in lst]
        return [{"feats": self.vec_by_path[p][None, :]} for p in lst]


def test_discriminability_merges_reference_wav():
    """目标情感参考从 eval_rows.reference_wav 合并进候选，判别可算且方向正确。"""
    import numpy as np
    e0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    e3 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    model = _MapEmoModel({
        "/refs/a_ang.wav": e0,  # 目标情感参考（reference_wav）
        "/refs/a_hap.wav": e1,
        "/refs/a_sad.wav": e2,
        "/refs/a_sur.wav": e3,
        "/hyp/a.wav": e0,       # 生成音频与目标参考同向 → 判别应选 ang
    })
    ref_index = {("s1", "hi"): {
        "hap": "/refs/a_hap.wav",
        "sad": "/refs/a_sad.wav",
        "sur": "/refs/a_sur.wav",
    }}
    eval_rows = [{
        "utt_id": "a", "speaker_id": "s1", "text": "hi",
        "emotion": "ang", "reference_wav": "/refs/a_ang.wav",
    }]
    out = compute_discriminability(["/hyp/a.wav"], eval_rows, ref_index, model)
    assert out["n_valid"] == 1
    assert out["n_scored"] == 1
    assert out["nearest_ref_acc_pct"] == 100.0
    assert out["same_emotion_mean"] == pytest.approx(100.0, abs=1e-6)
    assert out["cross_emotion_mean"] == pytest.approx(0.0, abs=1e-6)
    assert out["gap_same_minus_cross"] == pytest.approx(100.0, abs=1e-6)
    assert out["n_way_distribution"] == {"4": 1}


def test_discriminability_skips_rows_without_enough_refs():
    """参考不足（含 target 后仍 <3）→ 进 n_skipped，不写 NaN、不误导。"""
    import numpy as np
    model = _MapEmoModel({
        "/refs/a_hap.wav": np.array([0.0, 1.0], dtype=np.float32),
        "/hyp/a.wav": np.array([1.0, 0.0], dtype=np.float32),
    })
    ref_index = {("s1", "hi"): {"hap": "/refs/a_hap.wav"}}
    eval_rows = [{
        "utt_id": "a", "speaker_id": "s1", "text": "hi",
        "emotion": "ang",  # 无 reference_wav → 合并后仅 1 个候选
    }]
    out = compute_discriminability(["/hyp/a.wav"], eval_rows, ref_index, model)
    assert out["n_valid"] == 0
    assert out["n_skipped"] == 1
    assert "reason" in out
```

注意：测试文件中需已有 `from emotion_metrics import (...)` 的导入；若无 `compute_discriminability`，在导入列表补上（现有导入为 `build_emotion_ref_index, compute_dtw_normalized, compute_frame_mean_emo_sim, compute_per_emotion_mean_sim, normalize_text`）。

- [ ] **Step 2: 运行确认失败**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=.:third_party/Matcha-TTS /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/test_emotion_metrics.py -k discriminability -v
```

Expected: FAIL。测试 1 因 target 不在候选返回退化 reason（n_scored=0、acc 断言失败）；测试 2 因提前返回分支缺 `n_skipped` 键而 KeyError（见 Step 3 修复）。

- [ ] **Step 3: 实现合并逻辑**

重构 `eval/emotion_metrics.py::compute_discriminability`（**整体替换旧函数体**，不是补丁）：

1. 新增模块级常量 `EMOTIONS = ["ang", "hap", "neu", "sad", "sur"]`（替换函数内硬编码列表）。
2. 在 `compute_discriminability` 上方新增私有辅助 `_merged_refs`。
3. 重写 `compute_discriminability`：单遍预处理（合并参考 → 过滤候选 >=3 → 批量提嵌入 → 单遍判别 → 聚合），所有返回分支统一 schema（退化时 `same_emotion_mean`/`cross_emotion_mean`/`gap` 为 `None` + `reason`，不写 NaN、不缺键）。

完整替换代码：

```python
EMOTIONS = ["ang", "hap", "neu", "sad", "sur"]


def _merged_refs(ref_index, row):
    """合并 sources 索引与 eval 行自带 reference_wav（目标情感参考）。

    ESD train/eval 划分会把 (speaker, text) 组的目标情感单独留到 eval，sources
    索引缺目标情感；reference_wav 是目标情感真实音频，合并后判别候选才包含
    "正确答案"。返回 {emotion: wav_path}；reference_wav 存在则覆盖目标情感。
    """
    target = (row.get("emotion") or row.get("sentence_emotion")
              or row.get("label"))
    refs = dict(ref_index.get(
        (row.get("speaker_id"), (row.get("text") or "").lower()), {}))
    rw = row.get("reference_wav")
    if target and rw:
        refs[target] = rw
    return refs


def compute_discriminability(hyp_paths, eval_rows, ref_index, emo_model,
                             batch_size=16):
    """对每条 hyp 计算与可用情感参考的余弦，输出 n-way 判别指标。

    前置契约：eval_rows 必须与 hyp_paths 按 utt_id 对齐（由调用方保证）。
    流程：合并参考 → 过滤候选>=3 → 批量提取嵌入 → 单遍判别 → 聚合。
    所有返回分支使用同一 schema；不可算时 reason 说明原因，不写 NaN。
    """
    # 1) 预处理：每行合并参考并过滤候选 >=3
    refs_by_idx = {}
    for i, row in enumerate(eval_rows):
        refs = _merged_refs(ref_index, row)
        if len(refs) >= 3:
            refs_by_idx[i] = refs
    if not refs_by_idx:
        return {
            "n_valid": 0, "n_skipped": len(eval_rows), "n_scored": 0,
            "n_way_avg": 0.0, "n_way_distribution": {},
            "nearest_ref_acc_pct": 0.0, "same_emotion_mean": None,
            "cross_emotion_mean": None, "gap_same_minus_cross": None,
            "mean_sim_by_ref_emotion": {},
            "reason": "no reference groups with >=3 emotions",
        }

    # 2) 批量提取参考嵌入（路径去重后一次提取）
    ref_paths = sorted({p for refs in refs_by_idx.values() for p in refs.values()})
    ref_embs = extract_utt_embeddings(emo_model, ref_paths, batch_size)
    ref_emb_by_path = dict(zip(ref_paths, ref_embs))
    hyp_embs = extract_utt_embeddings(emo_model, hyp_paths, batch_size)

    # 3) 单遍判别聚合
    n_way_counts = {}
    acc = n_scored = 0
    same_vals, cross_vals = [], []
    for i, refs in refs_by_idx.items():
        target = (eval_rows[i].get("emotion") or eval_rows[i].get("sentence_emotion")
                  or eval_rows[i].get("label"))
        if target not in refs:
            continue
        sims = {e: float(np.dot(hyp_embs[i], ref_emb_by_path[p]))
                for e, p in refs.items()}
        n_scored += 1
        n_way = len(sims)
        n_way_counts[str(n_way)] = n_way_counts.get(str(n_way), 0) + 1
        acc += int(max(sims, key=sims.get) == target)
        same_vals.append(sims[target])
        cross_vals += [s for e, s in sims.items() if e != target]
    if n_scored == 0:
        return {
            "n_valid": len(refs_by_idx),
            "n_skipped": len(eval_rows) - len(refs_by_idx),
            "n_scored": 0, "n_way_avg": 0.0, "n_way_distribution": {},
            "nearest_ref_acc_pct": 0.0, "same_emotion_mean": None,
            "cross_emotion_mean": None, "gap_same_minus_cross": None,
            "mean_sim_by_ref_emotion": {},
            "reason": "target emotion absent from merged references",
        }

    present = {e for refs in refs_by_idx.values() for e in refs}
    mean_sim_by_emo = {
        e: float(np.mean([np.dot(hyp_embs[i], ref_emb_by_path[refs[e]])
                          for i, refs in refs_by_idx.items() if e in refs]))
        for e in EMOTIONS if e in present
    }
    return {
        "n_valid": len(refs_by_idx),
        "n_skipped": len(eval_rows) - len(refs_by_idx),
        "n_scored": n_scored,
        "n_way_avg": float(sum(int(k) * v for k, v in n_way_counts.items())
                           / sum(n_way_counts.values())),
        "n_way_distribution": n_way_counts,
        "nearest_ref_acc_pct": round(acc / n_scored * 100.0, 2),
        "same_emotion_mean": round(float(np.mean(same_vals)), 2),
        "cross_emotion_mean": round(float(np.mean(cross_vals)), 2),
        "gap_same_minus_cross": round(
            float(np.mean(same_vals) - np.mean(cross_vals)), 2),
        "mean_sim_by_ref_emotion": mean_sim_by_emo,
    }
```

（旧函数体的 `key` 局部变量、`path_set` 循环、退化 reason 分支均被上述结构替代；`mean_sim_by_ref_emotion` 现含实际均值而非 nanmean，因缺失情感已被 `present` 过滤。）

修改 `eval/eval_emo_film.py::load_manifest`，在 `mapping[uid]` 中增加：

```python
                "reference_wav": (os.path.abspath(rec["reference_wav"])
                                  if rec.get("reference_wav") else None),
```

（放在 `"speaker_id": rec.get("speaker_id"),` 之后；同步把 `load_manifest` docstring 改为 `返回 {utt_id: {"text","speaker_id","emotion","reference_wav"}}`。）

- [ ] **Step 4: 运行确认通过**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=.:third_party/Matcha-TTS /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/test_emotion_metrics.py tests/test_eval_emo_film.py -q
```

Expected: 全部通过（原 14 + 新增 2 = 16）。

- [ ] **Step 5: 更新契约文档 `docs/contracts/emofilm_v3_eval.md`**

把 `discriminability` 字段说明中"eval 集结构上不可计算 / N/A"的口径更新为：

```markdown
| `discriminability`（可选） | n-way 判别：n_valid / n_skipped / n_scored / n_way_avg / nearest_ref_acc_pct / same_emotion_mean / cross_emotion_mean / gap_same_minus_cross / n_way_distribution / mean_sim_by_ref_emotion。目标情感参考 = eval manifest 的 reference_wav，其他情感参考 = sources 索引；候选 >=3 才计入（ESD eval 集 1422/1500 可算，78 条跳过）。 |
```

同时删除/改写该文档中"结构上不可计算 / N/A"的旧口径表述。

## Task 2: v1 兼容精简

**Files:**
- Delete: `exp/prompt_match/v1_legacy_padded.pt`
- Modify: `cosyvoice/utils/emo_checkpoint.py`（注释收窄）
- Modify: `tools/inference_emo_film.py`（docstring 示例）

- [ ] **Step 1: 删除临时 padded 产物**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && rm exp/prompt_match/v1_legacy_padded.pt
```

说明：该文件是上一轮为 v1 推理临时生成的（v1 权重 + 随机 emotion_head/arousal_head）；用户已确认 v1 不再需要代码支持，且 v3 基线不跑 v1。删除后可重建（随机头 + v1 权重），非不可恢复。删除后必须同步清理引用（见 Step 1b）：

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && rg -n "v1_legacy_padded" --glob '!docs/superpowers/**' .
```

Expected: 唯一活跃引用为 `exp/prompt_match/run_infer.sh:28`（`run_model v1 exp/prompt_match/v1_legacy_padded.pt`），必须移除。

- [ ] **Step 1b: 移除 prompt_match 脚本中的 v1 分支**

按"启动前看卡、现场传参、脚本简洁"原则重构（不写自动选卡代码）：

`exp/prompt_match/run_infer.sh` 整体替换为（删除 v1 分支与写死 `GPU=2`，GPU 由环境变量决定）：

```bash
#!/bin/bash
# 情感匹配 prompt 推理：3 模型 × 60 条（同情感不同句 prompt）。
# 运行前先 nvidia-smi 查看显卡，现场决定：GPU=<index> bash exp/prompt_match/run_infer.sh
set -o pipefail
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
WR=/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
PY=/home/hanlvyuan/miniconda3/envs/emofilm/bin/python
export PYTHONPATH=.:third_party/Matcha-TTS
export MODELSCOPE_OFFLINE=1
MANIFEST=$WR/data/contracts/emofilm_v1/eval/esd_prompt_match_60.jsonl
GPU="${GPU:-0}"

run_model() {
  local name=$1 ckpt=$2
  mkdir -p exp/prompt_match/$name/esd
  echo "=== [$name] start $(date) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY tools/inference_emo_film.py \
    --test_manifest $MANIFEST \
    --output_dir exp/prompt_match/$name/esd \
    --model_dir $WR/pretrained_models/CosyVoice2-0.5B --llm_ckpt $ckpt \
    --esd_root $WR/datasets/ESD --workspace_root $WR \
    --device cuda --fp16 --seed 1986 --save_every 20 --skip_existing \
    > exp/prompt_match/$name/infer.log 2>&1
  echo "=== [$name] done $(date) ==="
}

run_model film_only     exp/emofilm_film_only/final.pt
run_model longepoch     exp/emofilm_film_only_longepoch/final.pt
run_model sentlvl       exp/emofilm_sentlvl/final.pt
echo "ALL PROMPT-MATCH INFERENCE DONE $(date)"
```

`exp/prompt_match/run_eval.sh` 整体替换为（删除 v1 循环、固定 GPU=2 与 `run_eval_robust.sh` 的 OOM 重试逻辑；GPU/whisper 设备由环境变量现场决定）：

```bash
#!/bin/bash
# 情感匹配 prompt 评测：3 模型 × 60 条 v3 口径。
# 运行前先 nvidia-smi 查看显卡：GPU=<index> WHISPER_DEV=<cuda|cpu> bash exp/prompt_match/run_eval.sh
set -o pipefail
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
WR=/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
PY=/home/hanlvyuan/miniconda3/envs/emofilm/bin/python
export PYTHONPATH=.:third_party/Matcha-TTS
export MODELSCOPE_OFFLINE=1
MANIFEST=$WR/data/contracts/emofilm_v1/eval/esd_prompt_match_60.jsonl
GPU="${GPU:-0}"
WHISPER_DEV="${WHISPER_DEV:-cuda}"

for name in film_only longepoch sentlvl; do
  mkdir -p exp/prompt_match/$name/eval
  CUDA_VISIBLE_DEVICES=$GPU $PY eval/eval_emo_film.py \
    --ref_dir $WR/exp/prompt_match/refs/esd \
    --hyp_dir exp/prompt_match/$name/esd \
    --ref_text_manifest $MANIFEST \
    --emotion_ref_manifest $WR/data/contracts/emofilm_v1/sources/esd/manifest.jsonl \
    --output exp/prompt_match/$name/eval/esd_v3_metrics.json \
    --device cuda --whisper_device $WHISPER_DEV --expected_count 60 --batch_size 16 \
    > exp/prompt_match/$name/eval/esd_v3.log 2>&1
done
echo "ALL PROMPT-MATCH EVAL DONE $(date)"
```

`exp/prompt_match/run_eval_robust.sh`：**删除**（上一轮 GPU 饱和时的 OOM 重试补丁；资源决策已改为启动前人工查看 + 现场传参，该脚本无保留价值）。

说明：`exp/prompt_match/v1/esd/` 等历史生成产物目录**保留不删**；v1 不再参与任何新生成/新评测。

- [ ] **Step 2: 收窄 `emo_checkpoint.py` 注释语义**

把 `TRAINED_ALLOWED_MISSING_PREFIXES` 上方注释改为：

```python
#: trained checkpoint 加载时允许缺失的顶层模块前缀（模型有、旧 ckpt 无）。
#: 仅 ``emotion_classifier.``：27-epoch disabled 基线（film_only_longepoch）的
#: final.pt 在冻结探针恒构造重构之前训练，不含该键；加载时随机初始化即可
#: （冻结随机权重对推理零影响）。sentlvl 及未来 ckpt 均应含该键。
#: 刻意不含 ``emotion_head.`` / ``arousal_head.`` —— v1 旧制品缺任务头
#: 必须在 trained 加载时失败（防冒充当前训练产物，ADR-0019/0020）。
```

常量本体不变（longepoch 依赖，删除会导致其推理崩溃）。

- [ ] **Step 3: 清理 `tools/inference_emo_film.py` docstring 示例**

把文件头用法示例中 `--llm_ckpt exp/emofilm_v1/final.pt` 与 `--output_dir exp/emofilm_v1/wav_esd` 改为当前活跃模型：

```text
  CUDA_VISIBLE_DEVICES=0 python tools/inference_emo_film.py \\
    --model_dir pretrained_models/CosyVoice2-0.5B \\
    --llm_ckpt exp/emofilm_film_only_longepoch/final.pt \\
    --test_manifest data/contracts/emofilm_v1/eval/esd/manifest.jsonl \\
    --esd_root datasets/ESD \\
    --output_dir exp/emofilm_film_only_longepoch/wav_esd \\
    --device cuda
```

（`data/contracts/emofilm_v1/` 路径是数据合同目录名，不是 v1 模型制品，保留。）

- [ ] **Step 4: 验证**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=.:third_party/Matcha-TTS /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/ -q
```

Expected: 全绿（需先完成 Step 4b 的路径修复，否则 test_canonical_paths 仍失败）。

- [ ] **Step 4b: 修复 test_canonical_paths 的既有失败**

`docs/reports/2026-08-03-emofilm-eval-v3-execution-report.md` 第 132 行（§4.6 中"全 docs/ 仅计划文件含 `/home/hanlvyuan/`"）含字面宿主路径，触发 `test_canonical_paths`。改写该句为抽象表述（不出现字面 `/home/hanlvyuan/`）：

```text
全 docs/ 仅计划文件含宿主绝对路径（执行型工件，含运行路径），且该失败先于本次代码改动。
```

另：Task 4 新建的 `docs/reports/2026-08-04-v3-neutral-baseline.md` 内容**不得出现**字面 `/home/hanlvyuan/`（run_eval.sh 的 `WR=` 在 exp/ 下不受此测试扫描）。

## Task 3: run_eval.sh 精简更新（启动前现场决定资源）

**Files:**
- Modify: `exp/emofilm_film_only_longepoch/run_eval.sh`
- Modify: `exp/emofilm_sentlvl/run_eval.sh`

- [ ] **Step 1: 更新两个 run_eval.sh（简洁模板，不写自动选卡代码）**

两个脚本替换为同一模板（仅 `EXP` 不同）：

```bash
#!/bin/bash
# EmoFiLM v3 全量中性基线评测。
# 运行前先 nvidia-smi 查看显卡，现场决定资源：
#   GPU=<index> WHISPER_DEV=<cuda|cpu> bash exp/<exp>/run_eval.sh
set -o pipefail
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
WR=/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM
PY=/home/hanlvyuan/miniconda3/envs/emofilm/bin/python
export PYTHONPATH=.:third_party/Matcha-TTS
export MODELSCOPE_OFFLINE=1

EXP=exp/emofilm_film_only_longepoch
# 数据集级参考视图与模型无关，复用 film_only 的 eval_refs
REFS_DIR=exp/emofilm_film_only/eval_refs
GPU="${GPU:-0}"
WHISPER_DEV="${WHISPER_DEV:-cuda}"
mkdir -p ${EXP}/eval/v3

run_eval() {
  local ds=$1 cnt=$2
  echo "[$(date +%H:%M:%S)] $ds: gpu=$GPU whisper=$WHISPER_DEV"
  CUDA_VISIBLE_DEVICES=$GPU $PY eval/eval_emo_film.py \
    --ref_dir ${REFS_DIR}/$ds \
    --hyp_dir ${EXP}/full/$ds \
    --ref_text_manifest $WR/data/contracts/emofilm_v1/eval/$ds/manifest.jsonl \
    --emotion_ref_manifest $WR/data/contracts/emofilm_v1/sources/esd/manifest.jsonl \
    --output ${EXP}/eval/v3/${ds}_metrics.json \
    --device cuda --whisper_device "$WHISPER_DEV" \
    --expected_count $cnt --batch_size 16 \
    > ${EXP}/eval/v3/${ds}.log 2>&1
}

run_eval esd    1500
run_eval fedd_a 500
run_eval fedd_b 500
echo "ALL EVAL DONE $(date)"
```

`exp/emofilm_sentlvl/run_eval.sh` 仅把 `EXP=exp/emofilm_sentlvl`，其余逐字段相同。

说明：脚本串行跑 3 个数据集（每实验目录自包含惯例）；若现场想并行，可开多个终端分别指定不同 GPU 与数据集（不写入脚本，保持简洁）。

- [ ] **Step 2: 语法验证**

```bash
bash -n exp/emofilm_film_only_longepoch/run_eval.sh exp/emofilm_sentlvl/run_eval.sh && echo OK
```

Expected: `OK`。

## Task 4: 跑 v3 全量中性基线（longepoch + sentlvl）

**Files:**
- Create: `docs/reports/2026-08-04-v3-neutral-baseline.md`

- [ ] **Step 1: 跑 longepoch 基线**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && nvidia-smi
# 查看显卡状态后现场决定 GPU 与 whisper 设备（GPU 余量充足 → cuda，紧张 → cpu）：
GPU=<index> WHISPER_DEV=<cuda|cpu> bash exp/emofilm_film_only_longepoch/run_eval.sh
```

Expected: 3 个数据集日志无报错；`eval/v3/{esd,fedd_a,fedd_b}_metrics.json` 生成；`n_samples` 分别 1500/500/500。

- [ ] **Step 2: 跑 sentlvl 基线**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && nvidia-smi
# 现场决定资源（可与 longepoch 共用一张卡串行，或另选空闲卡）：
GPU=<index> WHISPER_DEV=<cuda|cpu> bash exp/emofilm_sentlvl/run_eval.sh
```

Expected: 同上。

- [ ] **Step 3: 确认判别指标不再是 N/A**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && \
  /home/hanlvyuan/miniconda3/envs/emofilm/bin/python - <<'EOF'
import json
for exp in ['exp/emofilm_film_only_longepoch', 'exp/emofilm_sentlvl']:
    d = json.load(open(f'{exp}/eval/v3/esd_metrics.json'))
    disc = d.get('discriminability', {})
    print(exp, 'n_scored=', disc.get('n_scored'),
          'acc=', disc.get('nearest_ref_acc_pct'),
          'n_way=', disc.get('n_way_avg'),
          'gap=', disc.get('gap_same_minus_cross'),
          'reason=', disc.get('reason'))
EOF
```

Expected: `n_scored > 0`、`acc` 为数值、无 `reason` 退化键（ESD）。FEDD 判别可为小 n 或 0（无同文本跨情感组，诚实口径，不失败）。

参考（review 子代理数据验证）：ESD eval 1500 条合并 reference_wav 后，1422 条候选 >=3（851 条 5-way、405 条 4-way、166 条 3-way），78 条跳过——`n_scored` 应约 1422。

- [ ] **Step 4: 写基线汇总**

在 `docs/reports/2026-08-04-v3-neutral-baseline.md` 写两张表：

1. longepoch / sentlvl × 3 数据集的 v3 指标（emo_sim / dtw_normalized / wer / per_emotion_emo_sim）。
2. ESD 判别维度（n_scored / n_way_avg / acc / same / cross / gap），并注明"判别在 eval 集可用，是本次 reference_wav 合并的直接结果"。

## Task 5: 全量验证

- [ ] **Step 1: 全量 pytest**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && PYTHONPATH=.:third_party/Matcha-TTS /home/hanlvyuan/miniconda3/envs/emofilm/bin/python -m pytest tests/ -q
```

Expected: 全绿（460+ 通过，仅环境门控 skip）。

- [ ] **Step 2: 确认工作区无 v1 legacy 残留**

```bash
cd /home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM && rg -n "v1_legacy_padded" --glob '!docs/superpowers/**' . || echo CLEAN
```

Expected: `CLEAN`（或仅历史报告提及，可接受并注明）。

---

## Self-Review

**Spec 覆盖：**
- 判别指标增强（§3 前置项 1）→ Task 1（合并 reference_wav + 2 核心测试 + load_manifest）。
- v3 全量中性基线（§3 前置项 2，用户裁剪为 longepoch+sentlvl）→ Task 3-4（动态资源 + 跑 + 汇总）。
- v1 兼容精简（§3 前置项 3，用户否定 legacy 模式）→ Task 2（删 padded + 注释收窄 + docstring 清理）。
- 双口径文档：用户已确认"留在那即可"，本计划不做任何双口径工作（无 Task）。
- 用户资源决策修正：不写自动选卡代码；启动前 nvidia-smi 查看、现场传参（GPU/WHISPER_DEV 环境变量）→ Task 2 Step 1b（prompt_match 脚本精简）+ Task 3（run_eval.sh 简洁模板）+ Task 4（先 nvidia-smi 再运行）。
- 用户重构倾向：Task 1 为整体重构（消除两遍循环重复、统一返回 schema、EMOTIONS 常量），非补丁式替换。

**无占位符检查：** 所有代码块完整；`_merged_refs` 在 Task 1 Step 3 定义并与调用处一致；测试中的导入补充明确说明。

**类型一致性：** `load_manifest` 返回新增 `reference_wav` 字段（绝对路径或 None）；`compute_discriminability` 的 eval_rows 由 main() 从 text_map 构造（`{"utt_id", **v}`）→ 自动携带 reference_wav；`_merged_refs` 读取该字段。现有 `test_run_evaluation_with_emotion_ref_manifest` 的 eval_rows 无 reference_wav → 行为不变。

**已知风险与对策：**
- longepoch 依赖 `TRAINED_ALLOWED_MISSING_PREFIXES`：Task 2 只改注释不改常量，验证步骤保证 pytest 与推理不受影响。
- 共享集群 GPU 波动：由人工启动前 nvidia-smi 判断（脚本不自动选卡）；whisper 设备同理现场决定。
- FEDD 判别可能 n=0：诚实口径（reason/n_skipped），不硬性失败，计划 Step 3 已注明。
