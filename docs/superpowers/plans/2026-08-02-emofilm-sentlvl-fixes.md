# EmoFiLM 句级监督修复（clean refactor）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `docs/reports/2026-08-02-emofilm-sentlvl-implementation-review.md` 逐项修复，采用"恒构造冻结探针 + 统一 loss 组合"的干净重构，打通训练→推理→评测全链路，并同步修正文档/ADR 漂移。

**Architecture:** 把"可选性"从模块拓扑层移到损失组合层：`emotion_classifier`（冻结探针）恒构造，`emo_loss_weight>0` 才计入 loss；checkpoint 加载器把 `emotion_classifier` 作为"训练期专用模块"在 base/trained 两个方向统一处理；span 与 input-end 两条监督路径改为可叠加、loss 键名分离（`loss_emotion_span` / `loss_emotion_input`）。

**Tech Stack:** Python 3.10 / PyTorch / hyperpyyaml / pytest。

> 执行约定：所有命令在 `emofilm` conda 环境激活后执行
> （`source scripts/activate_env.sh`，下文的 `python` 即该环境解释器），
> 并设置 `PYTHONPATH=.:third_party/Matcha-TTS`。

---

## 计划前核对（对报告逐条复核结论）

| 报告问题 | 复核结果 | 本计划修复 |
| --- | --- | --- |
| P0：训练启动 `load_base_state` 拒绝 `emotion_classifier.*` 缺失 | 已复现：`missing keys: ['emotion_classifier.bias', 'emotion_classifier.weight']`；base `llm.pt` 实测 295 键、无任何 emotion 键 | Task 3：`ALLOWED_MISSING_PREFIXES` 增加 `emotion_classifier.` |
| P0：推理 `load_trained_state`（strict）拒绝多余分类器键 | 已复现：`unexpected keys: ['emotion_classifier.bias', 'emotion_classifier.weight']`；真实入口是 `tools/inference_emo_film.py::load_emofilm_model` | Task 2：恒构造分类器使拓扑一致，推理 strict 加载天然通过；Task 3 删 `ALLOWED_UNEXPECTED_PREFIXES` |
| P1：span 分支提前 return，input-end loss 静默丢弃；`loss_emotion` 键双语义 | 已核对 forward：span 分支无条件提前 return，input-end 块只在 disabled 分支 | Task 4：统一 loss 组合点，两条路径可叠加；键名分离 |
| P1：`emo_film_sentlvl.yaml` 头部注释自相矛盾 | 已核对：L15-20 宣称"不接受 emo_loss_weight / 死字段"，L86 却 `emo_loss_weight: 0.2` | Task 7：重写头部注释 |
| P2：条件建模块 + `getattr` 守卫 + 白名单散落五处、方向选错 | 已核对：`getattr(self, "emo_loss_weight", 0.0)`、`if emo_loss_weight > 0` 条件构造、`ALLOWED_UNEXPECTED_PREFIXES` 无真实调用点 | Task 2/3/4：恒构造 + 删守卫 + 删白名单 |
| Standards：`load_base_state` 双遍 unexpected 过滤死代码 | 已核对：第二遍在首遍 raise 后恒为空 | Task 3：删除第二遍过滤 |
| Standards：schema 文档/ADR-0019/测试 docstring 漂移 | 已核对 `docs/contracts/emofilm_v2_schema.md:27-28`、`docs/adr/0019:12,27`、两个测试文件头部 | Task 6/7/8 |
| 测试覆盖缺口：两个真实加载入口零测试 | 已核对：新增测试只覆盖 forward 图与 `load_base_state` 的无人调用方向 | Task 1：补两个方向的集成回归测试 |

**执行约束（用户持久约束，来自 handoff §0）：**
- 中文响应 + 中文代码注释；不主动 git（每任务以测试/验证收尾，不写 commit 步骤）。
- 采用**适量核心 TDD**：Task 1 先写核心回归测试（红），Task 2-4 实现转绿；旧测试冲突的直接删除/重构精简，不为测试覆盖率而堆测试。
- 拒绝打补丁：删除 `ALLOWED_UNEXPECTED_PREFIXES` 与 `getattr` 守卫，不做"加一个白名单再修一个白名单"。

## 文件结构

- `cosyvoice/llm/llm_emotion.py` — 恒构造冻结探针 + 统一 loss 组合 + 键名分离（核心）。
- `cosyvoice/utils/emo_checkpoint.py` — 可选模块策略收敛（base/trained 两个加载器）。
- `tests/test_emofilm_downstream_heads.py` — 新增核心回归测试；重构旧测试（键名、删除反转锁）。
- `tests/test_emofilm_downstream_supervision.py` — 键名同步（`loss_emotion` → `loss_emotion_span`）。
- `tests/test_emofilm_protocol.py` — docstring 更新 + 死字段循环精简。
- `tests/test_emofilm_contract.py` — docstring 死字段清单更新。
- `conf/emo_film_sentlvl.yaml` / `conf/emo_film.yaml` / `conf/emo_film_earlystop.yaml` — 注释修正。
- `docs/contracts/emofilm_v2_schema.md` — 死配置节更新。
- `exp/emofilm_sentlvl/run_infer.sh` — 注释与真实机制一致。
- `docs/adr/0021-emofilm-input-end-sentence-supervision.md` — 新建，记录决策重开。

---

### Task 1: 核心回归测试（先红）

**Files:**
- Modify: `tests/test_emofilm_downstream_heads.py`（在文件末尾新增两节）

- [ ] **Step 1: 在 `tests/test_emofilm_downstream_heads.py` 末尾追加以下测试代码**

```python
# ============================================================
# I. 恒定拓扑 + checkpoint 双向兼容（P0 回归）
# ============================================================


def test_constant_topology_always_has_frozen_classifier():
    """恒构造冻结探针：disabled 模型也含 emotion_classifier（训练/推理/基线
    同一 state_dict 键集）；emo_loss_weight 恒为 float 属性。"""
    model = _make_model()
    assert model.emo_loss_weight == 0.0
    assert isinstance(model.emotion_classifier, nn.Linear)
    assert model.emotion_classifier.in_features == model.llm_input_size
    assert model.emotion_classifier.out_features == 6
    assert all(not p.requires_grad for p in model.emotion_classifier.parameters())
    keys = set(model.state_dict())
    assert "emotion_classifier.weight" in keys
    assert "emotion_classifier.bias" in keys


def test_base_load_allows_missing_classifier():
    """训练启动路径：sentlvl 模型（emo_loss_weight>0）加载 base llm.pt
    （无分类器键）必须成功（P0 回归）。"""
    model = Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=10,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
        downstream_supervision="disabled",
    )
    base = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith(ALLOWED_MISSING_PREFIXES)
    }
    load_base_state(model, base)


def test_trained_load_accepts_with_and_without_classifier():
    """推理路径：disabled 模型 strict 加载 sentlvl final.pt（含分类器）成功；
    旧 disabled ckpt（无分类器）也成功（训练期专用模块缺失容忍）。"""
    sentlvl = Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=10,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
        downstream_supervision="disabled",
    )
    infer = _make_model()  # disabled 推理拓扑
    load_trained_state(infer, dict(sentlvl.state_dict()))
    old_ckpt = {
        key: value.clone()
        for key, value in infer.state_dict().items()
        if not key.startswith("emotion_classifier.")
    }
    load_trained_state(infer, old_ckpt)


def test_trained_load_still_rejects_v1_missing_heads():
    """v1 防冒充守卫：trained 加载仍拒绝 emotion_head/arousal_head 缺失。"""
    model = _make_model()
    v1_like = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith(("emotion_head.", "arousal_head."))
    }
    with pytest.raises(RuntimeError, match="emotion_head"):
        load_trained_state(_make_model(), v1_like)


# ============================================================
# J. 两条监督路径可叠加 + loss 键名分离（P1 回归）
# ============================================================


def test_input_end_and_span_losses_coexist_with_distinct_keys():
    """span 分支不再提前 return：input-end loss 不被静默丢弃；键名分离
    （loss_emotion_span / loss_intensity / loss_emotion_input）。"""
    model = Qwen2LM_Emotion(
        llm_input_size=4,
        llm_output_size=4,
        speech_token_size=10,
        emotion_vocab_size=6,
        intensity_vocab_size=4,
        llm=_FakeQwen(4),
        sampling=lambda scores, decoded, sampling: 2,
        emo_loss_weight=0.2,
        downstream_supervision="disabled",
    )
    batch = _add_one_span(
        _base_batch(text_len=2, speech_len=4), tok_start=0, tok_end=2
    )
    out = model.forward(batch, torch.device("cpu"))
    assert "loss_emotion_span" in out
    assert "loss_intensity" in out
    assert "loss_emotion_input" in out
    expected = (
        out["loss_tts"]
        + 1.0 * out["loss_emotion_span"]
        + 1.0 * out["loss_intensity"]
        + 0.2 * out["loss_emotion_input"]
    )
    torch.testing.assert_close(out["loss"].detach(), expected)


def test_zero_weight_gates_input_end_loss():
    """emo_loss_weight=0：分类器存在但不计入 loss，loss_dict 键集与基线一致。"""
    model = _make_model()
    out = model.forward(_base_batch(), torch.device("cpu"))
    assert set(out.keys()) == {"loss", "acc", "loss_tts"}
    torch.testing.assert_close(out["loss"].detach(), out["loss_tts"])
```

> 说明：`test_zero_weight_gates_input_end_loss` 将**替换**旧测试
> `test_disabled_model_has_no_input_end_classifier`（Task 6 删除旧测试）。

- [ ] **Step 2: 运行新测试，确认按预期失败**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest \
  tests/test_emofilm_downstream_heads.py \
  -k "constant_topology or base_load_allows_missing_classifier or trained_load_accepts \
      or trained_load_still_rejects_v1 or input_end_and_span_losses_coexist \
      or zero_weight_gates_input_end_loss" -q
```

Expected: 6 个测试全部 FAIL——
- `constant_topology`：`AttributeError: 'Qwen2LM_Emotion' object has no attribute 'emo_loss_weight'`（disabled 模型无属性）。
- `base_load_allows_missing_classifier`：`RuntimeError ... missing keys: ['emotion_classifier.bias', 'emotion_classifier.weight']`。
- `trained_load_accepts_with_and_without_classifier`：第一段 `RuntimeError ... unexpected keys: ['emotion_classifier.bias', ...]` 或第二段 missing 失败。
- `input_end_and_span_losses_coexist`：`KeyError: 'loss_emotion_span'`（当前键名是 `loss_emotion`，且 span 分支不计算 input-end loss）。
- `zero_weight_gates_input_end_loss`：`AttributeError: ... no attribute 'emo_loss_weight'`。

---

### Task 2: 恒构造冻结探针（`cosyvoice/llm/llm_emotion.py`）

**Files:**
- Modify: `cosyvoice/llm/llm_emotion.py`（模块 docstring、类 docstring、`__init__`）

- [ ] **Step 1: 替换模块 docstring 的"监督路径"段（当前 L18-37）**

删除现有"input-end 句级监督（可选...默认关闭...load_base_state 容忍 unexpected）"叙述，替换为：

```python
监督路径（两条，可叠加，loss 键名分离）：

- **input-end 句级监督**（``emo_loss_weight>0`` 计入 loss，默认 ``0.0``=不计）：
  ``emotion_classifier``（随机初始化 ``Linear(llm_input_size,
  emotion_vocab_size)``，``requires_grad_(False)`` 恒冻结）**恒构造**，作用在
  FiLM 输出 ``modulated_text_emb`` 上，``CrossEntropyLoss(ignore_index=0)`` 对
  token 级 ``emotion_ids``（同句 token 共享句级标签，padding=0 自动忽略）。
  梯度经冻结探针回流 ``emotion_encoder``/``emotion_adapter``（FiLM）。恒构造使
  模型拓扑与 checkpoint schema 不随配置变化（训练/推理/基线同一 state_dict
  键集）；推理不调用该探针。loss 键：``loss_emotion_input``。
- **downstream span 词级监督**（``downstream_supervision='enabled'`` + span 数据）：
  ``emotion_head``/``arousal_head`` 可训练，仅消费 ``lm_output`` 在 speech-token
  span 区段上 masked-mean 池化的 feature（反捷径：辅助监督落在生成因果链下游，
  不接收 control ID/loss target 作为特征）。loss 键：``loss_emotion_span`` /
  ``loss_intensity``。

无 span 且 ``emo_loss_weight==0`` 时 loss 仅 ``loss_tts``。
```

- [ ] **Step 2: 替换类 docstring 的 input-end 段（当前 L189-194）**

```python
    **input-end 句级监督**：``emotion_classifier``（随机初始化、冻结）恒构造，
    ``emo_loss_weight>0`` 时对 FiLM 输出 ``modulated_text_emb`` 做句级情感 CE
    （token 级 ``emotion_ids``，padding=0 由 ``ignore_index=0`` 忽略），梯度经
    冻结探针回流 FiLM（``emotion_encoder``/``emotion_adapter``）。
    ``emo_loss_weight==0``（默认）时不计入 loss（disabled 路径 loss 仅
    ``loss_tts``，loss_dict 无 ``loss_emotion_input``）。
```

- [ ] **Step 3: 替换 `__init__` 的条件构造块（当前 L233-249）**

删除整个 `if emo_loss_weight > 0:` 块及其注释，替换为：

```python
        # input-end 句级情感监督探针（恒构造、冻结；emo_loss_weight>0 才计入
        # loss）：emotion_classifier 随机初始化并 requires_grad_(False)，
        # loss_emotion 梯度经固定读出器回流 FiLM（emotion_encoder /
        # emotion_adapter）。恒构造使模型拓扑与 state_dict 键集不随配置变化
        # （训练/推理/基线同一 schema），推理不调用该探针；emo_loss_weight==0
        # 时仅不计入 loss。
        self.emo_loss_weight = float(emo_loss_weight)
        self.emotion_classifier = nn.Linear(
            llm_input_size, emotion_vocab_size
        )
        self.emotion_classifier.requires_grad_(False)
        self.criterion_emotion_cls = nn.CrossEntropyLoss(ignore_index=0)
```

- [ ] **Step 4: 运行恒拓扑测试（其余仍红，属预期）**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest \
  tests/test_emofilm_downstream_heads.py -k "constant_topology or zero_weight_gates_input_end_loss" -q
```

Expected: `constant_topology` PASS；`zero_weight_gates_input_end_loss` PASS
（loss 键集：`{"loss","acc","loss_tts"}`，forward 尚无 input-end 块但 getattr 删除后
`self.emo_loss_weight` 恒为 0.0，`if self.emo_loss_weight > 0` 为 False，行为正确）。

---

### Task 3: checkpoint 可选模块策略收敛（`cosyvoice/utils/emo_checkpoint.py`）

**Files:**
- Modify: `cosyvoice/utils/emo_checkpoint.py`

- [ ] **Step 1: 替换模块 docstring（L4-11）与常量区（L18-35）**

模块 docstring 替换为：

```python
"""Emo-FiLM checkpoint 边界与参数身份（活跃主线权威）。

本模块是 Emo-FiLM 的单一活跃 checkpoint 加载器（ADR-0020 扁平化）。允许缺失的
前缀与活跃 ``Qwen2LM_Emotion`` 拓扑一致：FiLM（``emotion_encoder`` /
``emotion_adapter``）+ 下游监督任务头（``emotion_head`` / ``arousal_head``）+
input-end 句级监督探针（``emotion_classifier``）均为随机新增模块，base
CosyVoice2 ``llm.pt`` 不含，允许在 base 加载时缺失。

``emotion_classifier`` 是训练期专用模块（恒构造、冻结、推理不调用）：trained
加载时同样允许其缺失（旧 disabled 基线 ckpt 不含该键，随机初始化即可）；但
``emotion_head`` / ``arousal_head`` 在 trained 加载时**不允许**缺失（v1 旧制品
防冒充守卫，ADR-0019/0020）。
"""
```

常量区替换为（删除 `ALLOWED_UNEXPECTED_PREFIXES`）：

```python
#: base checkpoint 加载时允许缺失的顶层模块前缀（活跃 ``Qwen2LM_Emotion`` 拓扑）。
#: FiLM + 下游 emotion/arousal 任务头 + input-end 探针允许缺失；backbone /
#: decoder / embedding 缺失或任何多余键 → 失败。
ALLOWED_MISSING_PREFIXES = (
    "emotion_encoder.",
    "emotion_adapter.",
    "emotion_head.",
    "arousal_head.",
    "emotion_classifier.",
)

#: trained checkpoint 加载时允许缺失的顶层模块前缀（模型有、旧 ckpt 无）。
#: ``emotion_classifier.`` 为 input-end 句级监督探针（恒构造、冻结、推理不调用）；
#: 旧 disabled 基线 ckpt 不包含它，加载时随机初始化即可（冻结随机权重对推理零
#: 影响）。刻意不含 ``emotion_head.`` / ``arousal_head.`` —— v1 旧制品缺任务头
#: 必须在 trained 加载时失败（防冒充当前训练产物）。
TRAINED_ALLOWED_MISSING_PREFIXES = (
    "emotion_classifier.",
)
```

- [ ] **Step 2: 重写 `load_base_state`（删除双遍过滤与 unexpected 白名单）**

```python
def load_base_state(model, state: Mapping[str, torch.Tensor]):
    """加载基础 checkpoint，只允许新增情感模块缺失、不允许任何多余键。

    - missing（模型有、ckpt 无）：仅允许 ``ALLOWED_MISSING_PREFIXES`` 前缀
      （FiLM / 下游任务头 / input-end 探针，base ``llm.pt`` 不含的新增模块）。
    - 其余 missing 或**任何** unexpected → schema mismatch 失败（base 必须是
      CosyVoice2 ``llm.pt``，不应携带超出模型拓扑的键）。
    """
    expected = set(model.state_dict().keys())
    actual = _state_keys(state)
    missing = expected - actual
    unexpected = actual - expected
    disallowed_missing = {
        key for key in missing
        if not key.startswith(ALLOWED_MISSING_PREFIXES)
    }
    if disallowed_missing or unexpected:
        _raise_mismatch("base", disallowed_missing, unexpected)
    return model.load_state_dict(dict(state), strict=False)
```

- [ ] **Step 3: 重写 `load_trained_state`（容忍训练期专用模块缺失，仍拒绝多余键）**

```python
def load_trained_state(model, state: Mapping[str, torch.Tensor]):
    """严格加载训练后 checkpoint；仅容忍训练期专用模块缺失。

    - missing：仅允许 ``TRAINED_ALLOWED_MISSING_PREFIXES``
      （``emotion_classifier.``，旧 disabled ckpt 不含的冻结探针）。
    - unexpected：任何多余键失败。
    """
    expected = set(model.state_dict().keys())
    actual = _state_keys(state)
    missing = expected - actual
    unexpected = actual - expected
    disallowed_missing = {
        key for key in missing
        if not key.startswith(TRAINED_ALLOWED_MISSING_PREFIXES)
    }
    if disallowed_missing or unexpected:
        _raise_mismatch("trained", disallowed_missing, unexpected)
    return model.load_state_dict(dict(state), strict=False)
```

- [ ] **Step 4: 运行 checkpoint 回归测试**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest \
  tests/test_emofilm_downstream_heads.py \
  -k "base_load_allows_missing_classifier or trained_load_accepts or trained_load_still_rejects_v1" -q
```

Expected: 3 个测试全部 PASS。

---

### Task 4: 统一 loss 组合 + 键名分离（`cosyvoice/llm/llm_emotion.py` forward）

**Files:**
- Modify: `cosyvoice/llm/llm_emotion.py`（`forward` 的 L410-465 区域）

- [ ] **Step 1: 重写 forward 的监督组合段**

删除现有"span 分支提前 return"与"disabled 分支 getattr 守卫"两个互斥块，替换为：

```python
        # ------------------------------------------------------------
        # 监督组合点（两条路径可叠加；loss 键名分离）
        # ------------------------------------------------------------
        # ``lm_output`` 为最后一层 hidden (B, T, llm_output_size)；
        # speech-token 区段 = ``lm_target != IGNORE_ID`` 的列。
        # ``speech_token_mask`` 既标识 supervised 列，也作为池化的安全网
        # （排除 IGNORE/padding 列）。
        speech_token_mask = lm_target != IGNORE_ID  # (B, T) bool, True = supervised
        loss = loss_tts
        loss_dict = {"loss": loss, "acc": acc, "loss_tts": loss_tts.detach()}

        # 1) 下游 span 词级监督（batch 携带 span 张量时计算；无 span 时由
        #    downstream_supervision 显式裁决，禁止静默降级）。
        if _batch_has_spans(batch):
            spans = {
                k: batch[k].to(device)
                for k in _SPAN_TENSOR_KEYS
                if k in batch
            }
            # feature 仅由 ``lm_output`` + span 几何区间决定（反捷径核心）。
            span_feature = self._pool_span_features(
                lm_output,
                speech_token_mask,
                text_token_len,
                spans["span_tok_start"],
                spans["span_tok_end"],
                spans["span_mask"],
                spans["span_valid"],
            )
            loss_emotion_span, loss_intensity = self._compute_downstream_losses(
                span_feature, spans
            )
            loss = (
                loss
                + self.emotion_head_weight * loss_emotion_span
                + self.intensity_head_weight * loss_intensity
            )
            loss_dict["loss_emotion_span"] = loss_emotion_span.detach()
            loss_dict["loss_intensity"] = loss_intensity.detach()
        elif self.downstream_supervision == "enabled":
            raise RuntimeError(
                "downstream_supervision='enabled' 但 batch 未携带 span 张量——"
                "下游监督头未接入数据管线（span→parquet→batch 链断）。"
                "若本次为 FiLM-only 实验，请在配置设 downstream_supervision='disabled'。"
            )

        # 2) input-end 句级监督（emo_loss_weight>0 时计入；与 span 路径可叠加，
        #    不再被 span 分支短路丢弃）。
        if self.emo_loss_weight > 0:
            emotion_logits = self.emotion_classifier(modulated_text_emb)
            loss_emotion_input = self.criterion_emotion_cls(
                emotion_logits.reshape(-1, emotion_logits.size(-1)),
                emotion_ids.reshape(-1),
            )
            loss = loss + self.emo_loss_weight * loss_emotion_input
            loss_dict["loss_emotion_input"] = loss_emotion_input.detach()

        loss_dict["loss"] = loss
        return loss_dict
```

- [ ] **Step 2: 运行组合测试（新键名）**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest \
  tests/test_emofilm_downstream_heads.py -k "input_end_and_span_losses_coexist" -q
```

Expected: PASS。

---

### Task 5: 核心测试全绿确认

- [ ] **Step 1: 运行 Task 1 的全部 6 个新测试**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest \
  tests/test_emofilm_downstream_heads.py \
  -k "constant_topology or base_load_allows_missing_classifier or trained_load_accepts \
      or trained_load_still_rejects_v1 or input_end_and_span_losses_coexist \
      or zero_weight_gates_input_end_loss" -q
```

Expected: 6 passed。

- [ ] **Step 2: 记录此刻已知的红测（旧测试尚未重构，预期失败）**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest \
  tests/test_emofilm_downstream_heads.py tests/test_emofilm_downstream_supervision.py -q 2>&1 | tail -20
```

Expected: 失败集中在 `test_disabled_model_has_no_input_end_classifier`（已过时，Task 6 删除）与
`loss_emotion` 键名相关断言（Task 6 同步为 `loss_emotion_span`）。

---

### Task 6: 精简/重构旧测试

**Files:**
- Modify: `tests/test_emofilm_downstream_heads.py`
- Modify: `tests/test_emofilm_downstream_supervision.py`
- Modify: `tests/test_emofilm_protocol.py`
- Modify: `tests/test_emofilm_contract.py`

- [ ] **Step 1: 删除过时测试 `test_disabled_model_has_no_input_end_classifier`**

`tests/test_emofilm_downstream_heads.py` L720-728 整个函数删除（已被
`test_zero_weight_gates_input_end_loss` 取代；"disabled 无分类器"已是过时拓扑断言）。

- [ ] **Step 2: 重命名并精简 `test_model_has_downstream_heads_and_no_input_classifier`**

改名为 `test_model_has_downstream_heads`，注释头从"A. 模型结构：heads 存在 / 无输入端
classifier"改为"A. 模型结构：可训练下游 heads + 冻结 input-end 探针"，函数体只保留
heads 断言（现有 body 已无 classifier 断言，仅名称与注释过时）。

- [ ] **Step 3: 同步 span loss 键名（`tests/test_emofilm_downstream_heads.py`）**

把以下 span 路径断言里的 `out["loss_emotion"]` 全部改为 `out["loss_emotion_span"]`：
- `test_anti_shortcut_target_change_does_not_change_head_input`（约 L345）
- `test_emotion_mask_false_contributes_no_emotion_loss`（约 L455）
- `test_intensity_mask_false_esd_contributes_no_intensity_loss`（约 L470）
- `test_emotion_mask_independent_from_intensity_mask`（约 L482）
- `test_invalid_span_contributes_no_loss`（约 L495）
- `test_soft_distribution_loss_matches_manual_soft_ce`（约 L520）
- `test_hard_ce_one_hot_is_special_case_of_soft_ce`（约 L540）
- `test_total_loss_is_tts_plus_weighted_emotion_plus_weighted_intensity`（约 L576）

`test_input_end_loss_emotion_gradient_flows_to_film`（L652-717）内的
`out["loss_emotion"]` 改为 `out["loss_emotion_input"]`，并把 L656/L679/L682 注释与
断言文案同步（"loss_emotion" → "loss_emotion_input"）；`model.emo_loss_weight == 0.2`
断言保留。

- [ ] **Step 4: 同步 `tests/test_emofilm_downstream_supervision.py`**

`test_disabled_no_span_returns_tts_only`（L41）与
`test_disabled_with_span_still_computes_heads`（L57）里的
`"loss_emotion"` 改为 `"loss_emotion_span"`。

- [ ] **Step 5: 更新 `tests/test_emofilm_protocol.py`**

模块 docstring L4-18 改为（删除"无输入端 emotion_classifier / 反转语义锁"叙述）：

```python
覆盖（brief 04 §D / issue 04 checklist）：
- 活跃模型恒构造冻结的 input-end 探针 ``emotion_classifier``（``emo_loss_weight>0``
  才计入 loss；默认 0 时 loss_dict 无 ``loss_emotion_input``）；
- forward 恒定单流 ``[sos, FiLM(text), task, speech]``，
  target = ``[IGNORE] * (1 + text_len) + speech + [eos]``；
- 无论 speech/text 比例如何均不产生 fill_token / 交错文本 / 双流状态
  （强制覆盖 speech/text > 3 的原双流触发比例）；
- disabled（emo_loss_weight=0、无 span）forward 仅产出 ``loss_tts``；
- inference 前缀 = ``[sos, FiLM(target text), task]``（训练前缀减去 teacher speech），
  LLM 条件只含 target text + emotion/intensity 控制；
- inference 签名不接受 ``prompt_emotion_ids`` / ``prompt_intensity_ids`` /
  ``prompt_text``（死字段已删），保留 ``prompt_speech_token`` / ``embedding``
  透传给 Flow/HiFT（不进 LLM lm_input）；
- ``conf/emo_film.yaml`` 无死配置字段（mix_ratio / alpha），
  实例化 ``Qwen2LM_Emotion``，base 仍指 CosyVoice2 llm.pt；
- 反转语义锁仅保留残余反模式（``mix_ratio`` / ``alpha`` 死字段）已删断言
  （ADR-0020 禁源码哈希标定；``emotion_classifier`` / ``emo_loss_weight`` 现为
  可选 input-end 句级监督，不再是反模式）。
```

`test_active_config_does_not_pass_dead_kwargs_to_llm`（L339-352）：循环
`for key in ("mix_ratio", "emo_loss_weight", "alpha")` 改为
`for key in ("mix_ratio", "alpha")`（测试名保留），函数 docstring 注明
"emo_loss_weight 是可选活参数，基线配置默认不传"。

- [ ] **Step 6: 更新 `tests/test_emofilm_contract.py` 模块 docstring**

L11 的"死配置字段（mix_ratio / emo_loss_weight / alpha）"改为
"死配置字段（mix_ratio / alpha）"。

- [ ] **Step 7: 更新 `tests/test_emofilm_downstream_heads.py` 模块 docstring**

把 L22-27 的"v2 模型仍无 ``emotion_classifier``...断言 ``emotion_classifier`` 等
反模式已删"改为：

```python
  - 活跃模型恒构造冻结 ``emotion_classifier``（input-end 句级监督探针，
    ``emo_loss_weight>0`` 才计入 loss）+ 可训练 ``emotion_head``/``arousal_head``
    （随机初始化 Linear，5 类 / 标量回归）；
  - base loader 允许 emotion_head/arousal_head（+ FiLM + emotion_classifier）
    缺失于 base ckpt；trained loader 仅容忍 emotion_classifier 缺失
    （旧 disabled ckpt），仍严格拒绝任务头缺失；
  - v1 基线锚 git ``9c6d84b``（ADR-0020 扁平化后不再用源码 sha256 锁）。
```

- [ ] **Step 8: 运行两个测试文件确认全绿**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest \
  tests/test_emofilm_downstream_heads.py tests/test_emofilm_downstream_supervision.py \
  tests/test_emofilm_protocol.py tests/test_emofilm_contract.py \
  tests/test_emofilm_training_contract.py tests/smoke_test_emo_config.py -q
```

Expected: 全部 PASS（数量以实际为准，预计 130+）。

---

### Task 7: 配置/文档/脚本注释修正

**Files:**
- Modify: `conf/emo_film_sentlvl.yaml`
- Modify: `conf/emo_film.yaml`
- Modify: `conf/emo_film_earlystop.yaml`
- Modify: `docs/contracts/emofilm_v2_schema.md`
- Modify: `exp/emofilm_sentlvl/run_infer.sh`

- [ ] **Step 1: 修正 `conf/emo_film_sentlvl.yaml` 头部矛盾注释（L15-21）**

把继承自基线配置的"关键点"块替换为：

```yaml
#   - llm: ``Qwen2LM_Emotion``。FiLM (emotion_encoder + emotion_adapter) +
#     下游 emotion/arousal 监督任务头 + input-end 句级监督探针 emotion_classifier
#     （恒构造、冻结；emo_loss_weight>0 才计入 loss）。
#   - 死配置字段仅剩：mix_ratio（双流）、顶层 alpha（配置占位；采样超参归属逐生成
#     decode_config）。emo_loss_weight 为可选 input-end 句级监督权重（默认 0=关闭）。
#     → tools/build_emofilm_contract.py :: assert_no_dead_config 必须通过。
```

- [ ] **Step 2: 修正 `conf/emo_film.yaml` 头部（L5-8）与 llm 块后注释（L76）**

头部替换为：

```yaml
#   - llm: ``Qwen2LM_Emotion``。FiLM (emotion_encoder + emotion_adapter) +
#     下游 emotion/arousal 监督任务头 + input-end 句级监督探针 emotion_classifier
#     （恒构造、冻结；emo_loss_weight>0 才计入 loss，本基线配置默认 0=关闭）。
#   - 死配置字段仅剩：mix_ratio（双流）、顶层 alpha（配置占位；采样超参归属逐生成
#     decode_config）。emo_loss_weight 是可选 input-end 句级监督权重，非死字段。
#     → tools/build_emofilm_contract.py :: assert_no_dead_config 必须通过。
```

L76 的 `# 不传 mix_ratio / emo_loss_weight / alpha（死字段，assert_no_dead_config 拒绝）。`
改为 `# 不传 mix_ratio / alpha（死字段，assert_no_dead_config 拒绝）；emo_loss_weight
# 默认 0=关闭（可选 input-end 句级监督）。`

- [ ] **Step 3: 对 `conf/emo_film_earlystop.yaml` 做与 Step 2 相同的两处替换**

（该文件头部与 L84 注释与 `emo_film.yaml` 逐字相同。）

- [ ] **Step 4: 更新 `docs/contracts/emofilm_v2_schema.md` 死配置节（L27-28）**

```markdown
- **死配置**：v2 resolved 配置不得含 `mix_ratio`（双流）、顶层 `alpha`（v1
  配置占位；采样超参归属 `decode_config`）。`emo_loss_weight` 是可选 input-end
  句级监督权重（默认 0=关闭），不属于死字段。
```

- [ ] **Step 5: 修正 `exp/emofilm_sentlvl/run_infer.sh` 头部注释（L5-8）**

把"load_base_state 容忍 ckpt 的 emotion_classifier.* unexpected（冻结随机权重丢弃）"
改为：

```bash
# 推理用 conf/emo_film.yaml 构造模型（emo_loss_weight 默认 0）；emotion_classifier
# 恒构造（冻结探针），与训练 ckpt 拓扑一致 → load_trained_state 严格加载通过。
# 旧 disabled 基线 ckpt（无分类器键）由 TRAINED_ALLOWED_MISSING_PREFIXES 容忍。
```

---

### Task 8: 新建 ADR-0021 记录决策重开

**Files:**
- Create: `docs/adr/0021-emofilm-input-end-sentence-supervision.md`

- [ ] **Step 1: 创建 ADR 文件，内容如下**

```markdown
# Input-end 句级情感监督恢复为可选路径（ADR-0021）

- status: accepted
- supersedes: ADR-0019 中"输入端 classifier CE 被取代 / 死配置字段含
  emo_loss_weight"相关条款（仅该部分）

## 背景

句级 loss_emotion 为 v1 设计（git 锚点 `9c6d84b`）。ADR-0019 因输入端标签回读
捷径将其删除并以下游 span 监督取代。为检验"加回句级监督能否突破 Emo-SIM~66
平台"（用户决策，2026-08-02 handoff），恢复该路径作为可选监督做对照实验。

## 决策

- `emotion_classifier`（`Linear(llm_input_size, emotion_vocab_size)`）**恒构造、
  冻结**（训练期探针），`emo_loss_weight>0` 时才计入 loss；disabled 基线
  loss_dict 不含 `loss_emotion_input`。
- input-end 句级监督与 downstream span 词级监督**可叠加**：loss =
  loss_tts + w_e·loss_emotion_span + w_i·loss_intensity + emo_w·loss_emotion_input；
  loss 键名按路径分离，避免日志/聚合歧义。
- checkpoint 策略：`emotion_classifier` 属"训练期专用模块"——base 加载允许缺失、
  trained 加载允许缺失（旧 disabled ckpt）；trained 加载仍拒绝
  `emotion_head` / `arousal_head` 缺失（v1 制品防冒充守卫保留）。
- 死配置字段集合：移除 `emo_loss_weight`（`DEAD_CONFIG_KEYS` 已同步为
  `{mix_ratio, alpha}`）。

## 已知风险（沿用 2026-07-22 审计 P0-1/P0-2，不视为本次实验的阻断项）

- 目标标签同时作为 FiLM 输入，存在标签回读捷径；该 CE 证明"FiLM 后文本表示可读
  出条件 ID"，不直接证明声学输出遵从情感。
- 冻结随机读出器给上游的梯度无语义锚点。
- 本路径定位为对照实验；span 监督仍是细粒度控制的长期主路径（ADR-0019 方向
  未变）。
```

---

### Task 9: 全量验证

- [ ] **Step 1: 全量 pytest**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python -m pytest tests -q 2>&1 | tail -12
```

Expected: 全部 PASS，除两个已知环境门控失败（与本改动无关）：
`test_eval_smoke.py::test_eval_runs_and_outputs_valid_json`（缺 `/tmp/smoke_*.wav` +
modelscope 认证过期）与 `test_extract_emotion2vec_frame.py::test_real_emofilm_emotion2vec_loader_smoke`
（未设 `EMOFILM_PROJECT_ROOT` / `EMOFILM_UPSTREAM`）。

- [ ] **Step 2: 端到端 smoke 复现真实入口（可选，GPU 空闲时）**

```bash
PYTHONPATH=.:third_party/Matcha-TTS python - <<'PY'
import torch
from cosyvoice.llm.llm_emotion import Qwen2LM_Emotion
from cosyvoice.utils.emo_checkpoint import load_base_state, load_trained_state
from tests._emofilm_fakes import _FakeQwen

def make(emo_w):
    return Qwen2LM_Emotion(
        llm_input_size=4, llm_output_size=4, speech_token_size=10,
        emotion_vocab_size=6, intensity_vocab_size=4,
        llm=_FakeQwen(4), sampling=lambda s, d, smp: 2,
        emo_loss_weight=emo_w, downstream_supervision="disabled",
    )

train = make(0.2)
base = {k: v for k, v in train.state_dict().items()
        if not k.startswith(("emotion_encoder.", "emotion_adapter.",
                             "emotion_head.", "arousal_head.", "emotion_classifier."))}
load_base_state(train, base)              # 训练启动路径
infer = make(0.0)
load_trained_state(infer, dict(train.state_dict()))  # 推理加载路径
print("TRAIN-START OK; INFER-LOAD OK")
PY
```

Expected: 打印 `TRAIN-START OK; INFER-LOAD OK`。

- [ ] **Step 3: 更新计划状态（全部 checkbox 勾选后，把结果回报给用户，不主动 git）**

汇报模板：每条报告问题的修复落点 + 测试结果 + 未解决的边界（如有）。

---

## 自审记录（计划作者）

1. **Spec 覆盖**：报告 P0×2 → Task 2/3；P1 静默丢弃与键名歧义 → Task 4；
   注释/文档漂移 → Task 6/7；ADR 冲突 → Task 8；测试覆盖缺口 → Task 1；
   P2 冗余 → Task 2/3/4 的删除动作。无遗漏项。
2. **占位符扫描**：所有步骤含具体代码/命令/预期输出，无 TBD。
3. **类型/命名一致性**：`emo_loss_weight`、`emotion_classifier`、
   `loss_emotion_input`、`loss_emotion_span`、`TRAINED_ALLOWED_MISSING_PREFIXES`
   在测试、实现、文档三侧一致；旧 `ALLOWED_UNEXPECTED_PREFIXES` 全库删除
   （已核对 tests 无引用）。
4. **TDD 边界**：只对核心行为（恒定拓扑、双向 checkpoint、可叠加 loss、键门控）
   写测试；旧测试按冲突删改，不为覆盖而堆测试。
