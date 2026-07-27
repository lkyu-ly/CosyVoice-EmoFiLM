# EmoFiLM 主线补救修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 `superpowers:subagent-driven-development`（每 Task 一个 fresh implementer + review）或 `superpowers:executing-plans` 执行。Steps 用 `- [ ]` 复选框跟踪。

**Goal:** 修复扁平化后二次审查发现的 1 Critical + 11 Important（公开训推入口/身份链/评测科学有效性），使主线"逻辑闭合、科学可用"。

**Architecture:** 按 spec §7 的 6 组依赖结构，组间并行、组内串行。推理链（#1/#3/#2/#5）是 Critical 路径，先做。所有修复原位修改主线代码，TDD（先失败测试→实现→通过），不提交 git（checkpoint = 聚焦测试绿 + progress ledger）。

**Tech Stack:** PyTorch / HyperPyYAML / numpy / pytest；conda env `emofilm`（`<emofilm-env-python> -m pytest`，即 conda env `emofilm` 的 python）。

## 冲突校验结论（2026-07-25，执行前已做）

读完主线 spec（`fine-grained-control-repair/spec.md`）+ issues 04/05/07/11 + schema + ADR 0001-0020：**12 项修复与主线设计无宏观冲突**——它们是主线 spec 早已要求、扁平化实现（ADR-0020）遗漏或未接入生产调用链的 DoD 项。ADR-0010/0013/0014 不阻塞（#8 属 ADR-0013「工程缺陷修复」例外；`eval_local_control.py` 是 v2 新增局部评测，非 v1 整句冻结范围）。本 remediation 是 13/14 GPU 实验的**前置**（代码门禁先于数据/资产门禁，ADR-0020 §6）。

## Global Constraints

- **不主动 git 提交/分支**（用户全局指令 + ADR-0020 §7）。每个 Task 末尾的 "checkpoint" 步骤 = 运行该 Task 的聚焦测试确认绿 + 在 `.scratch/emofilm-mainline-remediation/sdd/progress.md` 追加一行。**禁止** `git commit`。
- **哈希边界**：禁源码文件内容哈希锁；safe-resume 仅 `os.path.isfile` + 逐条身份指纹；不 reintroduce `wav_sha256`；git commit SHA 作锚点允许。
- **扁平化**：原位改主线，旧代码仅存 git `9c6d84b`。
- **不动**：`tools/build_fedd_part_b_v2.py`+其测试；v1 冻结产物（`exp/emofilm_v1/` 等）；ADR 0001-0018；v1 `write_run_identity` 8 参数签名兼容锁（Task 5 只新增调用，不改该函数签名）。
- **注释中文**，术语英文（argmax/softmax/match/sub/del/ins 等）。
- **spec 权威**：`.scratch/emofilm-mainline-remediation/spec.md`（根因/证据/验收 + §9 Grilling 决策）。本 plan 不重复根因，只给实现步骤。
- conda env pytest 路径：`PY=<emofilm-env-python>`（conda env `emofilm` 的 python 绝对路径）；`$PY -m pytest <path> -v`。

## Grilling 决策（2026-07-25，已与用户确认）

1. **seed 策略 = per-request 固定**（非全局 RNG 漂移）。`ras_sampling` 全用 `torch.multinomial`，`torch.manual_seed`+`torch.cuda.manual_seed_all` 可完全控制采样随机性。
2. **seed 来源 = 固定默认 1986**（cli `--seed` 可配）。`run_inference` 循环内每个 utt 生成前重置 RNG。不透传 seed 到 `model.tts`/`llm.inference`（重置全局即可）。per-utt hash(utt_id) 对 triplet 有害（三档 utt_id 不同→三 seed），已否决。
3. **seed 载体 = 独立 `GenerationRow.seed` 字段**（int，必需）。进 schema §2 + `validate_generation_row` + `generation_row_identity_fingerprint` payload。triplet 比 `seed`（替代读未定义的 `seed_policy`——schema 从无此字段，删除其读取）。
4. **#9 形态**：control 身份任一缺失→hard-fail；**删除 per-pair prompt 校验死代码**（`_extract_ctrl_prompt_core` + `_strict_pair` 内 prompt 段）——schema §1 SupervisionSpan 无 `prompt_row_ref`，该校验恒跳过。gen 的 prompt 一致性靠 schema prompt 族≥1 + #10 三档 `prompt_row_ref` 一致。
5. **#10 身份全缺**：由入口 `validate_generation_row` 保证（schema 四族≥1），不另发明校验。
6. **#4 calibrated 成员不一致**：raise（spec L246 合并兼容键含 `calibrated`）。
7. **decode_config 采样超参**（top_p/top_k/tau_r）与 schema §2 L108 脱节——**非本次 12 项**，记 `issues/10` follow-up（seed 已独立字段，不强求补进 decode_config）。

---

## File Structure

| 文件 | 改动 Task | 职责 |
|---|---|---|
| `cosyvoice/cli/model_emo.py` | T1, T2 | tts/llm_job 消费 finish_reason + 透传 decode_config |
| `cosyvoice/cli/cosyvoice_emo.py` | T2 | 读 yaml decode_config 传入 |
| `cosyvoice/cli/frontend_emo.py` | T3 | 删 prompt_* 死字段 |
| `tools/inference_emo_film.py` | T4 | 写 GenerationRow（含 seed）+ per-utt seed 重置 + 接入 check_skip_existing |
| `docs/contracts/emofilm_v2_schema.md` | T4 | GenerationRow 加 seed 字段 |
| `tools/build_emofilm_contract.py` | T4, T10 | validate_generation_row 校验 seed |
| `tools/write_emofilm_run_identity.py` | T4, T5, T6 | 指纹加 seed；patch_bundle 覆盖 untracked |
| `cosyvoice/bin/train_emo.py` | T5 | 训练入口改调 v2 identity |
| `tools/generate_tagged_jsonl.py` | T7 | 校准贯穿 |
| `eval/eval_local_control.py` | T8, T9, T11 | exact 聚合 / control 身份 hard-fail + 删 prompt 死代码 / NaN |
| `eval/triplet_eval.py` | T10, T11 | seed 比对 / NaN |
| `eval/acoustic_evaluators.py` | T12 | NaN 门禁 |
| 测试（新增/扩展） | 各 T | `tests/test_emofilm_*.py` |

---

## Task 1: [#1 Critical] 非 EOS 结果不得落 WAV — 模型包装层门控

**Files:**
- Modify: `cosyvoice/cli/model_emo.py:31-104`（`tts` / `llm_job`）
- Test: `tests/test_emofilm_inference_contract.py`（新建或扩展）

**Interfaces:**
- Consumes: `Qwen2LM_Emotion.last_decode_result`（`DecodeResult`，`llm_emotion.py:779` 已写入）。
- Produces: `tts()` 在非 eos 时**不 yield** `tts_speech`；改为 yield `{"tts_speech": None, "finish_reason": <fr>, "decode_result": <DecodeResult>}`（EOS 时仍 yield `{"tts_speech": tensor}`）。下游 T4 据此决定是否存 wav。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_emofilm_inference_contract.py
import torch
from unittest.mock import MagicMock
from cosyvoice.cli.model_emo import CosyVoice2Model_Emotion

def test_non_eos_finish_reason_does_not_token2wav(monkeypatch):
    """max_len_reached 时 tts 不进 token2wav、不 yield 音频。"""
    model = CosyVoice2Model_Emotion.__new__(CosyVoice2Model_Emotion)  # 跳过 __init__
    model.lock = __import__("threading").Lock()
    model.tts_speech_token_dict = {}
    model.llm_end_dict = {}
    model.hift_cache_dict = {}
    model.device = "cpu"; model.fp16 = False
    model.llm_context = MagicMock()

    # mock LLM：inference 不 yield（模拟非 eos），last_decode_result = max_len_reached
    from cosyvoice.llm.llm_emotion import DecodeResult
    bad = DecodeResult(tokens=[], finish_reason="max_len_reached",
                       min_len=2, max_len=4, num_valid_speech_tokens=0,
                       invalid_token_retries=0, text_len=1)
    llm = MagicMock()
    llm.inference.return_value = iter([])  # 非 eos 不 yield token
    llm.last_decode_result = bad
    model.llm = llm

    token2wav_called = []
    model.token2wav = lambda **kw: token2wav_called.append(kw) or torch.zeros(1)

    outputs = list(model.tts(text=torch.zeros(1, 1, dtype=torch.int32),
                             emotion_ids=torch.zeros(1, 1, dtype=torch.long),
                             intensity_ids=torch.zeros(1, 1, dtype=torch.long)))
    assert token2wav_called == [], "非 eos 不得调用 token2wav"
    assert outputs and outputs[0].get("finish_reason") == "max_len_reached"
    assert outputs[0].get("tts_speech") is None

def test_eos_finish_reason_still_token2wav(monkeypatch):
    """eos 时正常 token2wav + yield 音频。"""
    # 同上构造，但 llm.inference yield [token1, token2]，last_decode_result.finish_reason="eos"
    # 断言 token2wav 被调用、outputs[0]["tts_speech"] 非 None、finish_reason=="eos"
    ...
```

- [ ] **Step 2: 跑红**

Run: `$PY -m pytest tests/test_emofilm_inference_contract.py -v`
Expected: FAIL（当前 `tts` 无条件 token2wav，`token2wav_called` 非空）。

- [ ] **Step 3: 实现 — `llm_job` 暴露 finish_reason，`tts` 门控**

修改 `cosyvoice/cli/model_emo.py`。在 `llm_job` 结束后保留 `self._last_decode_result`（线程内写、主线程读，已有 `thread_errors` 机制保证线程完成）：

```python
# llm_job 内：inference 循环结束后
self.llm_end_dict[uuid] = True
# 暴露结构化 finish_reason（供 tts 门控；inference 非 eos 不 yield token）
self._last_decode_result = getattr(self.llm, "last_decode_result", None)
```

`tts` 在 `p.join()` + `thread_errors` 检查后、token2wav 前加门控：

```python
# tts 内，p.join() / thread_errors 检查之后：
decode_result = getattr(self, "_last_decode_result", None)
finish_reason = getattr(decode_result, "finish_reason", None)
if finish_reason != "eos":
    # 非 eos 不进声学、不落正式 WAV（schema：非 eos 不得携 wav_path）
    yield {"tts_speech": None, "finish_reason": finish_reason,
           "decode_result": decode_result}
    return
this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
this_tts_speech = self.token2wav(...)
yield {"tts_speech": this_tts_speech.cpu(), "finish_reason": "eos",
       "decode_result": decode_result}
```

- [ ] **Step 4: 跑绿**

Run: `$PY -m pytest tests/test_emofilm_inference_contract.py -v`
Expected: PASS。

- [ ] **Step 5: checkpoint**

`$PY -m pytest tests/test_emofilm_inference_contract.py -v 2>&1 | tail -5`。确认绿。progress.md 追加 `Task 1: done (#1 non-eos gating)`。

---

## Task 2: [#3] decode_config 接入公开推理

**Depends on:** T1（同调用链）。
**Files:**
- Modify: `cosyvoice/cli/model_emo.py:31-104`（`tts`/`llm_job` 增 decode_config 形参）；`cosyvoice/cli/cosyvoice_emo.py:45-89`
- Test: `tests/test_emofilm_inference_contract.py`

**Interfaces:**
- Consumes: yaml `decode_config`（`emo_film.yaml:27-30`）。
- Produces: `tts(decode_config=...)` / `llm_job(..., decode_config)`；`inference_emo_film` 从 `configs["decode_config"]` 读并传入。

> 注：decode_config 仍只含长度参数（min/max ratio + hard cap）。采样超参（top_p/top_k/tau_r）与 schema §2 L108 的脱节是既存问题，记 `issues/10`，本 Task 不强修。

- [ ] **Step 1: 写失败测试**

```python
def test_decode_config_threaded_to_inference(monkeypatch):
    """tts 的 decode_config 实际传到 llm.inference。"""
    captured = {}
    model = CosyVoice2Model_Emotion.__new__(CosyVoice2Model_Emotion)
    # ... 构造同 T1 ...
    llm = MagicMock()
    def fake_inference(**kw):
        captured.update(kw)
        llm.last_decode_result = <eos DecodeResult>
        return iter([1, 2])
    llm.inference.side_effect = fake_inference
    model.llm = llm
    model.token2wav = lambda **kw: torch.zeros(1)

    list(model.tts(text=..., emotion_ids=..., intensity_ids=...,
                   decode_config={"min_token_text_ratio": 1, "max_token_text_ratio": 5,
                                  "max_len_hard_cap": 100}))
    assert captured["min_token_text_ratio"] == 1
    assert captured["max_token_text_ratio"] == 5
    assert captured["max_len_hard_cap"] == 100
```

- [ ] **Step 2: 跑红** — `$PY -m pytest tests/test_emofilm_inference_contract.py::test_decode_config_threaded_to_inference -v` → FAIL。

- [ ] **Step 3: 实现**

`model_emo.py` `tts` 签名加 `decode_config: dict | None = None`；`llm_job` 签名加 `decode_config`，传给 `self.llm.inference(..., min_token_text_ratio=..., max_token_text_ratio=..., max_len_hard_cap=...)`。`tts` 调 `llm_job` 时透传。`decode_config is None` 时用 `Qwen2LM_Emotion.decode` 的默认（不传 kwargs）。

`cosyvoice_emo.py` `inference_emo_film` 读 `self.configs["decode_config"]`（`__init__` 保留 configs 或提前抽出），作为 `model.tts(decode_config=...)` 入参。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — contract 测试 + `tests/test_inference_emo_film*.py`。progress.md `Task 2: done (#3 decode_config)`。

---

## Task 3: [#2] 公开前端闭合 target-only 协议（删 prompt_* 死字段）

**Files:**
- Modify: `cosyvoice/cli/frontend_emo.py:11-84`
- Test: `tests/test_frontend_emo.py`（扩展）

**Interfaces:**
- Produces: `frontend_emo_film` 返回 dict 只含 `tts()` 消费的键：`text`/`emotion_ids`/`intensity_ids` + 声学 prompt 键（`flow_embedding`/`llm_embedding`/`llm_prompt_speech_token`/`flow_prompt_speech_token`/`prompt_speech_feat`）。

- [ ] **Step 1: 写失败测试**

```python
def test_frontend_emo_film_no_dead_prompt_fields(fake_frontend):
    out = fake_frontend.frontend_emo_film("<emotion type='hap'>hi</emotion>",
                                          "ref text", "/fake/prompt.wav")
    dead = {"prompt_text", "prompt_text_len", "prompt_emotion_ids",
            "prompt_intensity_ids", "prompt_text_token", "prompt_emo_ids",
            "prompt_inten_ids"}
    assert dead.isdisjoint(out.keys()), f"死字段仍在: {dead & out.keys()}"
    for k in ("flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"):
        assert k in out
```

- [ ] **Step 2: 跑红** — FAIL（当前 out 含 prompt_emotion_ids 等）。

- [ ] **Step 3: 实现**

`_PROMPT_CONDITIONING_KEYS` 删 `"prompt_text"`、`"prompt_text_len"`。`frontend_emo_film` 删除：line 70-73（`prompt_text_token`/`prompt_emo_ids`/`prompt_inten_ids` 构造）、line 81-82（`model_input["prompt_emotion_ids"]/prompt_intensity_ids`）。保留 line 75-80 的 `text`/`emotion_ids`/`intensity_ids` 注入。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_frontend_emo.py -v`。progress.md `Task 3: done (#2 frontend)`。

---

## Task 4: [#5] 生产入口写 GenerationRow（含 seed）+ per-utt seed 重置 + 安全 skip

**Depends on:** T1（finish_reason 可得）、T2（decode_config 可得）。
**Files:**
- Modify: `tools/inference_emo_film.py:154-222`（`run_inference`）；`docs/contracts/emofilm_v2_schema.md`（GenerationRow 加 seed）；`tools/build_emofilm_contract.py`（`validate_generation_row` 校验 seed）；`tools/write_emofilm_run_identity.py`（`generation_row_identity_fingerprint` payload 加 seed）
- Test: `tests/test_inference_emo_film.py`、`tests/test_emofilm_contract.py`、`tests/test_emofilm_run_identity.py`

**Interfaces:**
- Consumes: T1 的 `finish_reason`/`decode_result`；T2 的 decode_config；`check_skip_existing` + `generation_row_identity_fingerprint`。
- Produces: manifest row = 合法 GenerationRow（utt_id/finish_reason/source_revision/checkpoint_sha256/decode_config/**seed**/wav_path/control_row_ref/prompt_row_ref）；per-utt RNG 重置；schema/validator/指纹含 seed。

**Grilling 决策**：seed per-request 固定（默认 1986，cli `--seed`），重置全局 torch+cuda RNG（不透传到 model.tts/llm）；`GenerationRow.seed` 独立字段进身份指纹。

- [ ] **Step 1: 写失败测试**

```python
def test_run_inference_writes_generation_row_with_seed(tmp_path, fake_cv2):
    """产物 row 含 seed + 过 validate_generation_row；非 eos 不写 wav。"""
    out_dir = tmp_path / "wav"; manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"utt_id":"u1","tagged_text":"<emotion>hi</emotion>",
                                    "control_row_ref":"control/u1","prompt_row_ref":"prompt/spk1"})+"\n")
    rows = run_inference(fake_cv2, str(manifest), str(tmp_path), str(out_dir),
                         llm_ckpt_sha="a"*64, source_revision="9c6d84b", seed=1986,
                         decode_config={"min_token_text_ratio":2,"max_token_text_ratio":20,"max_len_hard_cap":2000})
    r = rows[0]
    assert r["finish_reason"] == "eos"
    assert r["seed"] == 1986
    assert r["control_row_ref"] == "control/u1" and r["prompt_row_ref"] == "prompt/spk1"
    from tools.build_emofilm_contract import validate_generation_row
    validate_generation_row(r)  # 不 raise（含 seed + 四族身份）

def test_per_utt_seed_reset_reproducible(tmp_path, fake_cv2):
    """同 utt + 同 seed + 同 config → 两次 token2wav 输出一致（per-utt 重置生效）。"""
    # mock 模型捕获 token2wav 输入 token 序列；两次同 seed 调用断言 token 序列 bit-identical
    ...

def test_non_eos_skips_wav(tmp_path, fake_cv2_non_eos):
    """LLM 返 max_len_reached → 不写 wav、row 无 wav_path、finish_reason 正确。"""
    rows = run_inference(...)
    assert rows[0]["finish_reason"] == "max_len_reached"
    assert "wav_path" not in rows[0]
    assert not (tmp_path / "wav" / "u1.wav").exists()

def test_skip_existing_seed_change_not_reused(tmp_path):
    """既有 row seed=1986，请求 seed=42 → 指纹不同 → 不 skip（重生成）。"""
    # 既有 manifest row（seed=1986）+ 请求 seed=42 → check_skip_existing 返回 skip=False
    ...
```

- [ ] **Step 2: 跑红** — FAIL（当前无 seed 字段、无 per-utt 重置、非 eos 写 wav、skip 只 isfile）。

- [ ] **Step 3: 实现**

(a) `emofilm_v2_schema.md` GenerationRow 表加：
```
| `seed` | int | 是 | per-request 固定随机种子（默认 1986）；per-utt 生成前重置 torch+cuda RNG。 |
```
身份约束补：seed 变化→不同指纹→不复用。

(b) `validate_generation_row`（`build_emofilm_contract.py`）：加 `seed` 必需（非负 int）。

(c) `generation_row_identity_fingerprint`（`write_emofilm_run_identity.py:640`）：payload 加 `"seed": row.get("seed")`。

(d) `run_inference`（`inference_emo_film.py`）：签名加 `seed: int = 1986`；循环体每 utt 生成前：
```python
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```
row 构造（eos 与非 eos）都加 `"seed": seed`。`main()` 加 `--seed`（默认 1986）。

(e) #1 非 eos 不写 wav：取首 chunk `finish_reason`（T1 暴露），eos+audio 才 save+wav_path；非 eos 仅诊断 row（无 wav_path）：
```python
chunks = list(cv2.inference_emo_film(text_with_emo=..., prompt_text=..., prompt_wav_path=...,
                                     decode_config=decode_config))
first = chunks[0] if chunks else {}
finish_reason = first.get("finish_reason", "sampler_error")
if finish_reason == "eos" and first.get("tts_speech") is not None:
    torchaudio.save(out_wav, first["tts_speech"].cpu(), cv2.sample_rate)
    row = {"utt_id": utt_id, "finish_reason": "eos", "source_revision": source_revision,
           "checkpoint_sha256": llm_ckpt_sha, "decode_config": decode_config, "seed": seed,
           "wav_path": os.path.relpath(out_wav, workspace_root),
           "control_row_ref": utt.get("control_row_ref"), "prompt_row_ref": utt.get("prompt_row_ref")}
else:
    row = {"utt_id": utt_id, "finish_reason": finish_reason, "source_revision": source_revision,
           "checkpoint_sha256": llm_ckpt_sha, "decode_config": decode_config, "seed": seed}
    non_eos_summary.append(row)
```

(f) skip 改 `check_skip_existing`：既有 row（含 seed）算指纹 vs 请求指纹（含 seed）；seed 变→指纹不同→不 skip。

- [ ] **Step 4: 跑绿** — 四个测试 PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_inference_emo_film.py tests/test_emofilm_contract.py tests/test_emofilm_run_identity.py -v`。progress.md `Task 4: done (#5 GenerationRow+seed+skip)`。

---

## Task 5: [#6] 训练入口写 v2 identity

**Depends on:** 无（可与推理组并行；与 T6 同文件 `write_emofilm_run_identity.py` 需串行或协调）。
**Files:**
- Modify: `cosyvoice/bin/train_emo.py:117-170`（`write_training_identity` / `update_training_identity`）
- Test: `tests/test_train_emo_identity.py`

**Interfaces:**
- Consumes: `write_emofilm_run_identity.write_emofilm_train_identity`；`train_utils_emo.summarize_optimizer_identity`。
- 不动: `write_run_identity`（v1 8 参数签名兼容锁）。

- [ ] **Step 1: 写失败测试**

```python
def test_training_identity_is_v2_schema(tmp_path, fake_model_optim_sched):
    identity_path = tmp_path / "train_identity.json"
    write_training_identity(identity_path, model=..., code_root=repo_root,
                            contract_dir=None, command="torchrun ...", seed=1986,
                            base_checkpoint=None, resolved_config=tmp_path/"resolved.yaml",
                            checkpoint_role="fresh", optimizer=..., scheduler=...)
    data = json.loads(identity_path.read_text())
    assert data["contract_name"] == "emofilm"
    assert data["schema_version"] == 2
    assert "optimizer_identity" in data["extra"] or "param_groups" in str(data)
    assert "resolved_config" in str(data)
```

- [ ] **Step 2: 跑红** — FAIL（当前写 emofilm_v1，无 optimizer_identity）。

- [ ] **Step 3: 实现**

`write_training_identity` 改调 `write_emofilm_train_identity`，补 `optimizer_identity=summarize_optimizer_identity(model, optimizer, scheduler, info_dict_conf)` + `resolved_config=payload`（读 resolved.yaml 解析为 dict 或传 path）+ `patch_bundle_path`（交由 `write_emofilm_train_identity` 内部据 dirty 决定）。当前签名无 optimizer/scheduler——**扩展签名**增 `optimizer=None, scheduler=None`（main 调用处 line 264 传入）。`update_training_identity` 改写 v2 extra（final_checkpoint + sha256）；若读到旧 v1 identity，raise 明确提示重训。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_train_emo_identity.py tests/test_emofilm_run_identity.py -v`。progress.md `Task 5: done (#6 train v2 identity)`。

---

## Task 6: [#7] patch_bundle 覆盖 untracked

**Depends on:** 与 T5 同文件，串行（T5 后）。
**Files:**
- Modify: `tools/write_emofilm_run_identity.py:233-251`（`_save_patch_bundle`）
- Test: `tests/test_emofilm_run_identity.py`

- [ ] **Step 1: 写失败测试**

```python
def test_patch_bundle_includes_untracked(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git","init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git","config","user.email","t@t"], cwd=repo, check=True)
    subprocess.run(["git","config","user.name","t"], cwd=repo, check=True)
    (repo/"tracked.py").write_text("a=1\n")
    subprocess.run(["git","add","-A"], cwd=repo, check=True)
    subprocess.run(["git","commit","-m","init"], cwd=repo, check=True, capture_output=True)
    (repo/"tracked.py").write_text("a=2\n")
    (repo/"untracked_new.py").write_text("b=1\n")   # 关键：未 git add
    patch_out = repo / "patch.bundle"
    info = _save_patch_bundle(repo, patch_out)
    bundle = patch_out.read_bytes()
    assert b"untracked_new.py" in bundle, "patch_bundle 必须含 untracked"
    # worktree 未被改动（GIT_INDEX_FILE 隔离）
    assert (repo/"untracked_new.py").read_text() == "b=1\n"
    assert subprocess.run(["git","status","--porcelain"], cwd=repo, capture_output=True).stdout.decode().count("\n") >= 2  # worktree 仍 dirty，未被 reset
```

- [ ] **Step 2: 跑红** — FAIL（当前 `git diff --binary HEAD` 不含 untracked_new.py）。

- [ ] **Step 3: 实现 — 隔离 index 方案**

```python
def _save_patch_bundle(code_root, output_patch):
    # 用临时 index 暂存全部（含 untracked），diff 后不污染真实 index/worktree
    tmp_index = output_patch.parent / f".git_index_{os.getpid()}.tmp"
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp_index)}
    subprocess.run(["git", "read-tree", "HEAD"], cwd=code_root, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=code_root, check=True, env=env)
    diff_bytes = subprocess.run(
        ["git", "diff", "--binary", "--cached", "HEAD", "--"],
        cwd=code_root, check=True, capture_output=True, env=env).stdout
    try: tmp_index.unlink(missing_ok=True)
    except OSError: pass
    output_patch.parent.mkdir(parents=True, exist_ok=True)
    output_patch.write_bytes(diff_bytes)
    return {"path": str(output_patch), "sha256": hashlib.sha256(diff_bytes).hexdigest(),
            "size_bytes": len(diff_bytes)}
```

> `GIT_INDEX_FILE` 指向临时文件，`read-tree HEAD` 初始化它为 HEAD 状态，`git add -A` 把 worktree 全部（含 untracked）写入临时 index，`diff --cached HEAD` 得到完整 patch。真实 `.git/index` 与 worktree 均不动。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_emofilm_run_identity.py -v`。progress.md `Task 6: done (#7 patch_bundle untracked)`。

---

## Task 7: [#4] 校准信息贯穿生成链

**Files:**
- Modify: `tools/generate_tagged_jsonl.py:134-151`（`_build_member_record`）、`:154-233`（`_span_from_members`）、`:236-303`（合并）、Predictor 协议 docstring
- Test: `tests/test_generate_tagged_jsonl.py`

**Interfaces:**
- Consumes: predictor `predict_word` 返回的 `calibrated`（bool）/`calibration`（`{method, version}` 或 None）。
- Produces: member/span 透传 calibrated/calibration；`_span_from_members` 不再硬编码 False。

- [ ] **Step 1: 写失败测试**

```python
def test_calibrated_predictor_propagates_to_span():
    pred = {"word":"hi","start_sec":0.0,"end_sec":0.5,"start_frame":0,"end_frame":12,
            "frame_rate_hz":50.0,"emotion_soft_distribution":[.1,.8,.05,.03,.02],
            "arousal":3.0,"raw_score":0.8,
            "calibrated": True, "calibration": {"method":"temperature", "version":"v1"}}
    spans = merge_word_predictions_to_v2_spans(
        utt_id="u1", word_preds=[pred], sentence_emotion="hap",
        sentence_vad=None, annotator_provenance={"model":"x"})
    assert spans[0]["calibrated"] is True
    assert spans[0]["calibration"] == {"method":"temperature", "version":"v1"}

def test_mixed_calibrated_members_raise():
    # 相邻成员 calibrated 不一致 → raise 携 utt_id（spec L246 合并兼容键含 calibrated）
    ...

def test_default_calibrated_false_when_predictor_omits():
    pred = {... 无 calibrated/calibration ...}
    spans = merge_word_predictions_to_v2_spans(...)
    assert spans[0]["calibrated"] is False
    assert spans[0].get("calibration") is None
```

- [ ] **Step 2: 跑红** — FAIL（当前硬编码 False，True 不透传）。

- [ ] **Step 3: 实现**

`_build_member_record` 增：
```python
"calibrated": bool(pred.get("calibrated", False)),
"calibration": pred.get("calibration"),  # None 或 {method, version}
```
合并键（`merge_word_predictions_to_v2_spans`）增 `calibrated`/`calibration` 一致性检查（不一致 → raise ValueError 携 utt_id）。`_span_from_members`：
```python
"calibrated": bool(members[0]["calibrated"]),  # 合并已保证一致
"calibration": members[0].get("calibration"),
```
Predictor Protocol docstring 声明可选输出 `calibrated`/`calibration`。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_generate_tagged_jsonl.py tests/test_emofilm_contract.py -v`。progress.md `Task 7: done (#4 calibration)`。

---

## Task 8: [#8] exact 聚合排除对齐失败样本

**Files:**
- Modify: `eval/eval_local_control.py:507-532`（回退逻辑）、`:600-610`（tier 标记）、聚合 `:825-895`
- Test: `tests/test_eval_local_control.py`

- [ ] **Step 1: 写失败测试**

```python
def test_alignment_failed_not_in_exact_aggregate():
    # exact tier + 对齐失败（align_status="failed"）→ aggregate exact 分母不含它
    rows = [make_eval_row(evidence_tier="exact", alignment_status="failed"),
            make_eval_row(evidence_tier="exact", alignment_status="aligned")]
    agg = derive_aggregate(rows, "exact")
    assert agg["n_total"] == 1  # 只含 aligned 那条
    assert agg.get("n_exact_alignment_failed") == 1  # 单独计数
```

- [ ] **Step 2: 跑红** — FAIL（当前 exact aggregate 含对齐失败样本，n_total==2）。

- [ ] **Step 3: 实现**

对齐失败时（exact tier 且 `boundary_sec is None` 回退后）样本 `alignment_status != "aligned"`；聚合时 `tier_rows` 过滤为 `evidence_tier=="exact" and alignment_status=="aligned"`，对齐失败的 exact 样本单独计入 `n_exact_alignment_failed`（不进精确 boundary_error/命中分母）。approximate tier 不变。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_eval_local_control.py -v`。progress.md `Task 8: done (#8 exact aggregate)`。

---

## Task 9: [#9] control 身份严格校验 + 删 per-pair prompt 死代码

**Depends on:** T8（同文件，串行）。
**Files:**
- Modify: `eval/eval_local_control.py:691-720`（删 `_extract_ctrl_prompt_core`）、`:785-813`（`_strict_pair` control 段改 hard-fail + 删 prompt 段）
- Test: `tests/test_eval_local_control.py`

**Grilling 决策**：schema §1 SupervisionSpan **无** `prompt_row_ref` → `_extract_ctrl_prompt_core` 恒 "" → per-pair prompt 校验从不执行（死代码，删）。gen 的 prompt 一致性靠 schema prompt 族≥1 + #10 三档 `prompt_row_ref` 一致。control 身份：任一缺失 hard-fail（schema/spec L133）。

- [ ] **Step 1: 写失败测试**

```python
def test_ctrl_declares_but_gen_missing_control_raises():
    ctrl = [{"utt_id":"u1","control_row_ref":"control/u1"}]
    gen  = [{"utt_id":"u1","finish_reason":"eos","wav_path":"x.wav",
             "checkpoint_sha256":"a"*64, "seed":1986}]  # gen 无 control_row_ref
    with pytest.raises(ValueError, match="control"):
        _strict_pair(ctrl, gen)

def test_gen_embeds_but_ctrl_missing_raises():
    # gen 有 control_row_ref 但 ctrl 无 utt_id/control_row_ref → hard-fail
    ...

def test_both_present_and_match_passes():
    # 正例：双方声明且一致 → 通过
    ...

def test_prompt_pair_check_removed():
    # _extract_ctrl_prompt_core 已删；_strict_pair 不再读 ctrl prompt
    import eval.eval_local_control as m
    assert not hasattr(m, "_extract_ctrl_prompt_core")
```

- [ ] **Step 2: 跑红** — FAIL（当前缺失静默跳过；prompt 校验仍在）。

- [ ] **Step 3: 实现**

- 删除 `_extract_ctrl_prompt_core`（line 704-719）+ `_strict_pair` 内 prompt 校验段（line 799-811）。
- `_strict_pair` control 段（line 785-797）改为对称强制：
```python
expected_ctrl_core = _extract_ctrl_control_core(ctrl)
gen_ctrl_core = _extract_gen_identity_core(gen, str_key="control_row_ref", mapping_key="control_row")
if not expected_ctrl_core or not gen_ctrl_core:
    raise ValueError(f"sample '{uid}' hard-fail: control identity missing "
                     f"(expected={expected_ctrl_core!r}, gen={gen_ctrl_core!r})")
if expected_ctrl_core != gen_ctrl_core:
    raise ValueError(f"sample '{uid}' hard-fail: control_row_ref mismatch — "
                     f"gen 内嵌 '{gen_ctrl_core}' 与配对 ctrl '{expected_ctrl_core}' 不一致")
```
- gen 的 prompt 族存在性由 `validate_generation_row` 保证（不在 per-pair 校）。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_eval_local_control.py -v`。progress.md `Task 9: done (#9 control hard-fail + del prompt deadcode)`。

---

## Task 10: [#10] triplet 比 seed + 身份不可全缺（validate 保证）

**Depends on:** T4（seed 字段/指纹/schema 已加）。
**Files:**
- Modify: `eval/triplet_eval.py:357-395`（`_check_group_membership`）；schema 已在 T4 加 seed
- Test: `tests/test_triplet_eval.py`

**Grilling 决策**：比 `seed` 字段（替代读未定义的 `seed_policy`，schema 从无此字段→恒 None 通过的根因）。身份全缺由入口 `validate_generation_row` 保证（schema 四族≥1）。

- [ ] **Step 1: 写失败测试**

```python
def test_different_seed_invalid():
    member_rows = {t: {"generation_row": {"finish_reason":"eos","wav_path":"x",
        "checkpoint_sha256":"a"*64, "source_revision":"9c6d84b", "seed": s}}
        for t, s in zip(INTENSITY_TIERS, [1,2,3])}
    ok, msg = _check_group_membership(member_rows)
    assert not ok and "seed" in msg

def test_all_missing_identity_invalid():
    member_rows = {t: {"generation_row": {"finish_reason":"eos","wav_path":"x",
        "checkpoint_sha256": None, "source_revision": None, "seed": 1986}}
        for t in INTENSITY_TIERS}
    ok, msg = _check_group_membership(member_rows)
    assert not ok and "identity" in msg
```

- [ ] **Step 2: 跑红** — FAIL（当前不比 seed；全 None 通过）。

- [ ] **Step 3: 实现**

- 删 `seed_policies` 比对（line 379-384，schema 无 seed_policy），改为：
```python
seeds = {member_rows[t].get("generation_row", {}).get("seed") for t in INTENSITY_TIERS}
if None in seeds or len(seeds) > 1:
    return False, f"seed_mismatch: {sorted(s for s in seeds if s is not None)}"
```
- checkpoint/source 全 None → `False, "missing_identity"`（schema 四族≥1 由 validate 保证，此处防御性 double-check）。
- triplet 入口对每条 generation_row 调 `validate_generation_row`（保证身份族非空 + seed 存在）。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_triplet_eval.py tests/test_emofilm_contract.py -v`。progress.md `Task 10: done (#10 triplet seed+identity)`。

---

## Task 11: [#11] 空/NaN evaluator 输出标记无效并剔除

**Depends on:** T8/T9（`eval_local_control.py` 协调）、T10（`triplet_eval.py` 协调）。建议放 E1/E2 之后。
**Files:**
- Modify: `eval/triplet_eval.py:103-130`（`compute_arousal_score`/`compute_emotion_prediction` 返 sentinel）、`:740-790`（非 EOS 行 invalid）；`eval/eval_local_control.py:259-289`（`evaluate_spans_from_frames` NaN 检查）
- Test: `tests/test_triplet_eval.py`、`tests/test_eval_local_control.py`

- [ ] **Step 1: 写失败测试**

```python
def test_empty_arousal_returns_none_not_zero():
    assert compute_arousal_score({"frames": np.array([])}) is None  # 非 0.0

def test_empty_emotion_returns_none_not_first_label():
    assert compute_emotion_prediction({"frames": np.array([])}, ["ang","hap"]) is None

def test_nan_frames_span_invalid():
    frames = np.full((4,5), np.nan)
    metrics = evaluate_spans_from_frames(frames, np.array([0,0.5,1,1.5]), 0.5,
                                         "ang","hap", ["ang","hap","neu","sad","sur"])
    assert metrics.get("valid") is False  # NaN → span invalid，不计分母
```

- [ ] **Step 2: 跑红** — FAIL（当前空→0/首类，NaN→argmax 首类）。

- [ ] **Step 3: 实现**

`compute_arousal_score`：空或全非有限 → `return None`（调用方据 None 标 `valid=False`，不进分母）。`compute_emotion_prediction`：空/单列/全非有限 → `return None`。`evaluate_spans_from_frames`：`frames`/`mean_dist` 用 `np.isfinite` 检查，全非有限或全空 → metrics 加 `"valid": False`，聚合剔除。triplet 非 EOS 行的 metrics 标 `valid=False` 不进 aggregate 分母（`build_triplet_aggregate` 跳过 valid=False）。

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_triplet_eval.py tests/test_eval_local_control.py -v`。progress.md `Task 11: done (#11 NaN/empty)`。

---

## Task 12: [#12] 校准门禁拒全 NaN 分布

**Files:**
- Modify: `eval/acoustic_evaluators.py:367-394`（`validate_emotion_label_mapping`）
- Test: `tests/test_acoustic_evaluators.py`

- [ ] **Step 1: 写失败测试**

```python
def test_all_nan_distribution_fails_calibration(monkeypatch):
    evaluator = FakeEmotionEvaluator(label_space=["ang","hap","neu","sad","sur"])
    monkeypatch.setattr(evaluator, "predict_frames",
                        lambda _: {"frames": np.full((4,5), np.nan)})
    clip = SyntheticReferenceClip(wav_path="x", known_emotion="ang")
    result = validate_emotion_label_mapping(evaluator, [clip])
    assert result["passed"] is False
    assert "non-finite" in str(result["details"][0].get("error","")).lower()
```

- [ ] **Step 2: 跑红** — FAIL（当前全 NaN argmax→首类→passed=True）。

- [ ] **Step 3: 实现**

`validate_emotion_label_mapping` 在 `mean_dist = frames.mean(axis=0)` 后加：
```python
if mean_dist.size == 0 or not np.isfinite(mean_dist).all():
    details.append({"wav_path": str(clip.wav_path), "known_emotion": clip.known_emotion,
                    "predicted": None, "passed": False,
                    "error": "non-finite or empty distribution"})
    continue
```

- [ ] **Step 4: 跑绿** — PASS。

- [ ] **Step 5: checkpoint** — `$PY -m pytest tests/test_acoustic_evaluators.py -v`。progress.md `Task 12: done (#12 NaN gate)`。

---

## Final Whole-Branch Review & 整套测试

- [ ] **Step 1: 整套 pytest**

`$PY -m pytest -q 2>&1 | tail -15`
Expected: ≥ 581 + 新增 passed；仅 2 个预存环境失败（`test_eval_smoke`/`test_extract_emotion2vec_frame`）。

- [ ] **Step 2: 最终 code-review**

`/code-review`（fixed point = git `9c6d84b`），双轴（Standards + Spec）。Dispatch 最终 reviewer subagent（用 review-package 脚本生成 diff 文件传入）。

- [ ] **Step 3: 收尾**

汇总 progress.md；更新 `/tmp/emofilm-flatten-handoff-2026-07-25.md` 的"未完成事项"（移除已修复项）；把 decode_config 采样超参脱节记入 `.scratch/emofilm-mainline-remediation/issues/10-followup.md`。**不提交 git**（等用户授权）。

## Self-Review

- **Spec 覆盖**：12 项发现 → Task 1-12 一一对应，无遗漏。Grilling 决策（seed×3 + #9 形态）已并入 T4/T9/T10 + schema/validator/指纹。
- **冲突校验**：已对照主线 spec/issues/ADR，无宏观冲突（见"冲突校验结论"段）。
- **Placeholder 扫描**：T1 的 eos 正例测试、T4 的 per-utt 复现测试用 `...` 标注显式待补全测试体（模式与已给反例镜像），implementer 按 failing-test 模板补；其余步骤代码完整。
- **类型一致**：`DecodeResult.finish_reason`（T1/T4）、`decode_config` dict 键名（T2/T4）、`GenerationRow.seed`（T4/T10 跨 schema+指纹+triplet 一致）、`valid` flag（T11 跨 triplet+FEDD）跨任务一致。
- **约束一致**：所有 Task checkpoint 用 pytest + progress.md，无 `git commit`；哈希边界遵守（T4 skip 用 seed 进指纹而非 wav_sha256，T6 patch_bundle 用 git 句柄）。
