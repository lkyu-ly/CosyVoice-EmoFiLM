# 重构后审查报告（2026-08-02）：句级监督修复 9 任务 TDD 交付

- 审查对象：工作树未提交改动（HEAD = `a89afcd` 之上，无 commit），核心为
  `cosyvoice/llm/llm_emotion.py`、`cosyvoice/utils/emo_checkpoint.py`、测试/配置/文档/ADR-0021。
- Spec 来源：`docs/superpowers/plans/2026-08-02-emofilm-sentlvl-fixes.md`（9 任务计划）+
  `docs/reports/2026-08-02-emofilm-sentlvl-implementation-review.md`（问题清单）。
- 验证事实（全部实测）：全量 pytest = **477 passed / 2 skipped / 0 failed**；
  端到端 smoke = `TRAIN-START OK; INFER-LOAD OK`；6 个新核心测试全绿；
  `test_canonical_paths` 通过（计划文件已无用户主目录硬编码）。

## 结论总览

两个 P0 与两个 P1 的核心修复**全部正确落地，且是干净重构而非补丁**：

| 计划项 | 实现核对 | 结论 |
| --- | --- | --- |
| Task 2 恒构造冻结探针 | `__init__` 无条件建 `emotion_classifier` + `requires_grad_(False)`，`emo_loss_weight` 恒 float；`getattr` 守卫已删（全库无残留） | ✅ |
| Task 3 checkpoint 双向策略 | base 缺分类器容忍、trained 双向兼容、unexpected 恒拒；`ALLOWED_UNEXPECTED_PREFIXES` 全库删除；双遍过滤死代码已删 | ✅ |
| Task 4 统一 loss 组合 | span 分支不再提前 return，两条路径可叠加；键名 `loss_emotion_span` / `loss_intensity` / `loss_emotion_input` 分离；数值组合与计划一致 | ✅ |
| Task 1/6 核心 TDD + 旧测试精简 | 6 个新回归测试先红后绿；过时反转锁删除/改名/键名同步 | ✅ |
| Task 7/8 文档 + ADR | 三配置头部、schema 死配置节、run_infer.sh 注释、ADR-0021 均落地 | ⚠️ 见 Standards |
| Task 9 验证 | 477 passed / 2 skipped（环境门控）；e2e 双向加载 OK | ✅ |

## Spec 轴（实现 vs 计划/问题清单）

1. **核心需求全部满足，未发现"实现错了"的项**：两个真实加载入口
   （`load_base_state` 训练启动、`tools/inference_emo_film.py` 的 `load_trained_state`）
   方向均正确；loss 数值组合、disabled 键集、v1 防冒充守卫均与计划一致。
2. **局部缺失（文档同步不完整，3 处）**：
   - 计划 Task 7 的注释修正漏掉 `conf/emo_film_sentlvl.yaml:85` 内部注释
     "=0（如 emo_film.yaml）则不创建分类器"——与同一计划 Task 2 的恒构造设计
     直接矛盾（计划自身也未点名该行）。
   - 计划 Task 2 Step 2 只替换了类 docstring 的 input-end 段，`llm_emotion.py:182`
     下游段总 loss 公式仍写旧键 `loss_emotion`（应为 `loss_emotion_span`）。
   - 计划 Task 6 未覆盖 `tests/test_emofilm_protocol.py:143` 的禁止键循环，
     仍用旧名 `loss_emotion`；按 dict 精确键语义，该断言**抓不到**
     `loss_emotion_span` / `loss_emotion_input`（比注释宣称的弱）。
3. **范围扩张**：仅两处——`test_eval_smoke.py` / `test_extract_emotion2vec_frame.py`
   由"环境失败"改为"环境缺失即 skip"。系用户本轮明确授权（与主线逻辑无关），
   且是合理改进，不算越界。

## Standards 轴

### 硬问题（文档/实现矛盾，3 处）

- `conf/emo_film_sentlvl.yaml:85`：注释宣称 `emo_loss_weight=0` 时"不创建分类器"，
  与恒构造实现矛盾（该文件正是本次重构的主题文件）。
- `cosyvoice/llm/llm_emotion.py:182`：类 docstring 总 loss 公式用旧键
  `loss_emotion`，与同文件新键名体系不一致。
- `tests/test_emofilm_downstream_heads.py:56-57`：遗留"反转语义锁"注释声称
  "v1 输入端 classifier 反模式已从活跃代码删除"——重构后为假，且与本文件
  刚更新的模块 docstring（L22-27）自相矛盾。

### 坏味道（judgement calls）

- **死代码**：`ACTIVE_LLM_EMOTION_PATH` 在 `tests/test_emofilm_downstream_heads.py:54`
  与 `tests/test_emofilm_protocol.py:41` 均只剩定义无使用（原使用者是本次删除/
  重写的测试），属本次重构遗留。`hashlib`（heads L35）与 `ACTIVE_MODEL_EMO_PATH`
  （protocol L43）在 HEAD 上本就未使用，非本次引入。
- **Mysterious/陈旧命名**：`tests/test_emofilm_protocol.py:143` 禁止键循环用旧名，
  且断言强度弱于注释意图。

### 反向确认（无问题）

- 未引入 Duplicated Code / Shotgun Surgery / Speculative Generality——本次重构
  反而删除了白名单、getattr 守卫、双遍过滤等冗余。
- 执行约束（中文注释、不主动 git）均遵守。

## 失败测试处置结论

- `test_eval_smoke`：已改为缺 `/tmp/smoke_*.wav` 时 skip——保留（环境门控合理）。
- `test_extract_emotion2vec_frame`：已改为缺 `EMOFILM_PROJECT_ROOT` / `EMOFILM_UPSTREAM`
  或资产不完整时 skip——保留（环境门控合理）。
- `test_canonical_paths`：现已通过（计划文件硬编码路径已清除），无需改动。
- 当前工作区无需再调整/删除任何测试。

## 建议的收尾动作（均文档级，不改行为）

1. `conf/emo_film_sentlvl.yaml:85` 改为"=0（如 emo_film.yaml）则不计入 loss
   （分类器恒构造）"。
2. `cosyvoice/llm/llm_emotion.py:182` 旧键 `loss_emotion` → `loss_emotion_span`。
3. `tests/test_emofilm_downstream_heads.py:56-57` 删除/改写过时反转锁注释。
4. `tests/test_emofilm_protocol.py:143` 禁止键改为
   `("loss_emotion_span", "loss_intensity", "loss_emotion_input")`。
5. 删除两个文件的死常量 `ACTIVE_LLM_EMOTION_PATH`。
