# EmoFiLM 句级监督（sentlvl）实现审查报告

- 日期：2026-08-02
- 审查对象：工作树未提交改动（`git diff HEAD`）+ 未跟踪文件
  `conf/emo_film_sentlvl.yaml`、`exp/emofilm_sentlvl/{run_train,run_infer,run_eval}.sh`
- 基线：HEAD `a89afcd`；Spec：`/tmp/emofilm-sentlvl-handoff-2026-08-02.md`；
  v1 参考实现：`9c6d84b:cosyvoice/llm/llm_emotion.py`
- 审查口径：**只审实现逻辑**（句级监督是否正确工作、是否接到正确位置），
  不评宏观方法论（句级 vs 词级监督的选择）。高视角审视：是否有更统一的设计、
  历史原因造成的冗余/绕远路。
- 方法：Standards / Spec 双轴（两个并行子代理）+ 主代理架构深化（codebase-design
  词汇）。证据来自源码阅读、配置/脚本核对、pytest 复跑与两个加载路径的独立复现
  （子代理与主代理分别复现，结论一致）。

## 1. 结论速览

| 级别 | 问题                                                                                                   | 影响                                                                   |
| ---- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| P0   | 训练启动`load_base_state` 拒绝 `emotion_classifier.*` 缺失键                                       | sentlvl 训练在加载 base`llm.pt` 时必然崩溃，无法开始                 |
| P0   | 推理`load_trained_state` 严格拒绝 `emotion_classifier.*` 多余键                                    | 训练产物`final.pt` 无法加载，推理/评测不可达                         |
| P1   | 两条监督路径并非"并列"：span 分支提前`return`，input-end 损失静默丢失；`loss_emotion` 键双语义冲突 | 未来组合配置会静默丢损失；日志/聚合口径歧义                            |
| P1   | `emo_film_sentlvl.yaml` 头部注释自相矛盾                                                             | 配置文档误导（头部宣称`emo_loss_weight` 是死字段，llm 块却启用 0.2） |
| P2   | 条件建模块 +`getattr` 守卫 + checkpoint 白名单散落五处，且白名单方向选错                             | 架构冗余，是 P0/P0 的温床；删除测试通过后复杂度应收敛                  |

## 2. 句级监督实现逻辑核查（核心问题：是否正确工作）

### 2.1 正确项（与 agent 汇报一致）

- **损失位置与标签对齐**：`emotion_logits = emotion_classifier(modulated_text_emb)`，
  CE 对 `emotion_ids.reshape(-1)`。数据管线（`processor.py` `tokenize_emo` /
  `padding`）产出严格等长的 token 级三元组，`emotion_to_id = {emo: i+1}`，
  `padding_value=0`；`CrossEntropyLoss(ignore_index=0)` 正确忽略 padding。
  `modulated_text_emb` 与 `emotion_ids` 形状一致（B, T_text, D）↔ (B, T_text)。
- **冻结与梯度回流**：`requires_grad_(False)` 不阻挡激活梯度，
  `autograd` 路径 loss_emotion → classifier（固定权重转置）→ modulated_text_emb →
  emotion_adapter / emotion_encoder 完整无断点；新测试 `test_input_end_loss_emotion_ gradient_flows_to_film` 验证了这一点（并正确打破 FiLM 恒等初始化以暴露路径）。
- **loss 组合**：`loss = loss_tts + 0.2·loss_emotion`，数值正确；loss_dict 顺序
  与 Executor 的 `loss_dict.items()` 聚合兼容。
- **早停不受影响**：`early_stop_metric: loss_tts`，`EarlyStopTracker` 只读该键。
- **优化器分组**：`freeze` 只解冻三组前缀；分类器已冻结，不进入任何参数组。
- **disabled 零回归**：`emo_loss_weight=0` 时不设置任何属性，state_dict 与基线一致，
  forward 经 `getattr` 守卫跳过。
- **配置门禁**：`DEAD_CONFIG_KEYS` 移除 `emo_loss_weight`、协议测试的文本扫描同步
  更新，两处门禁一致放行（handoff 决策点 2 落实）。
- **开关语义**：`emo_loss_weight` 独立于 `downstream_supervision`，两路径未用同一
  开关（handoff 决策点 3 落实）。
- **训练动力学说明**：FiLMLayer 恒等初始化（`projection.weight=0`）使初始
  `modulated = text_emb`，此时 `∂loss_emotion/∂emotion_features = 0`，loss_emotion
  初期只推动 projection；agent 的动力学分析成立。

### 2.2 错误项（P0，复现确认）

**训练启动必失败。** `train_emo.py` 对 base `llm.pt` 走 `load_base_state`；
`emo_loss_weight=0.2` 的模型 state_dict 含 `emotion_classifier.weight/bias`，而 base
`llm.pt` 没有 → missing 键不在 `ALLOWED_MISSING_PREFIXES`（只有 FiLM + 两个任务头）
→ `_raise_mismatch`。复现：

```
enabled has classifier: True
1) training-start base load: FAIL -> base checkpoint schema mismatch;
   missing keys: ['emotion_classifier.bias', 'emotion_classifier.weight']
```

本次改动只加了 `ALLOWED_UNEXPECTED_PREFIXES`（ckpt 有、模型无），没有加 missing
方向（模型有、ckpt 无）。两个方向正好相反，真实需要的方向被漏掉了。

**推理必失败。** `tools/inference_emo_film.py::load_emofilm_model` 用
`CosyVoice2_Emotion(model_dir)`（固定读 `conf/emo_film.yaml`，disabled、无分类器）
构造模型，然后 `load_trained_state`（**严格**加载，不是 `load_base_state`）。
训练产物带 `emotion_classifier.*` → unexpected 键 → 抛错。复现：

```
2) inference trained load: FAIL -> trained checkpoint schema mismatch;
   unexpected keys: ['emotion_classifier.bias', 'emotion_classifier.weight']
```

agent 汇报表中"推理可加载 ✅ load_base_state 容忍分类器 unexpected"引错了入口：
真实推理入口从未调用 `load_base_state`；`run_infer.sh` 的注释同样把这个错误机制
写进了脚本。当前 `ALLOWED_UNEXPECTED_PREFIXES` 没有任何真实调用路径需要它
（base 加载的语义是"模型比 ckpt 多模块"= missing 方向；模型比 ckpt 少模块且 ckpt
带分类器的场景只出现在推理，而推理走严格加载器）。

因此 handoff §3 的执行路径"训练 → 试听 → 全量推理 → 评测"在第一步（训练启动）
就断，即便绕过也会断在推理。

### 2.3 测试覆盖缺口

新增测试验证了 forward 图与梯度，但未覆盖两个真实入口（训练启动 base 加载、
推理 strict 加载）。"接口即测试面"：`load_base_state` / `load_trained_state` 与
配置拓扑的组合才是本功能真正的集成 seam，缺了它，261 个单元测试全绿也拦不住
P0。本次复跑：emofilm 相关测试全绿（384 passed；1 个无关的 emotion2vec 环境测试
因未设 `EMOFILM_PROJECT_ROOT` 失败，与本改动无关）。

## 3. Spec 轴

### 3.1 缺失 / 未完成

- handoff §3 执行路径要求"训练（可复用早停）→ 试听 3 → 全量推理（2500）→
  baseline 评测"，实际训练与推理入口均不可达 → 任务核心交付未成立。
- handoff §3 检查清单要求"推理可加载"（§0 约束 + §3 第 3 步对抗式审查），
  未满足；汇报表该行结论与代码事实不符。

### 3.2 实现错误

- `emo_checkpoint.py` 的 `ALLOWED_UNEXPECTED_PREFIXES` 解决了不存在于真实调用路径
  的方向，却漏掉两个真实方向（训练 missing、推理 unexpected-strict）。spec 并未
  要求该白名单机制；它是实现缺陷的产物。

### 3.3 合规项

- 决策点 2（配置门禁更新）：✅ `DEAD_CONFIG_KEYS` + 文本扫描测试同步。
- 决策点 3（独立开关、不混淆两路径）：✅ 独立 `emo_loss_weight` 参数。
- 早停保留、disabled 基线零回归、不触动 v1 冻结制品、实验目录独立：✅。
- 中文注释、未主动 git 提交：✅。
- 无实质 scope creep（脚本/配置改动均在实验目录内）。

## 4. Standards 轴

### 4.1 硬问题（文档化标准 / 注释与事实一致性）

- `run_infer.sh` 注释声称"推理用 disabled 配置构造模型…`load_base_state` 容忍 ckpt
  的 `emotion_classifier.*` unexpected"——代码事实是推理走 `load_trained_state`
  严格加载，注释与实现矛盾（AGENTS.md：注释与代码一致性）。
- `conf/emo_film_sentlvl.yaml` 头部整段复制自基线配置，宣称"不接受 `emo_loss_weight` /
  `alpha`""删除死配置字段：…`emo_loss_weight`…"，而同一文件 llm 块设置
  `emo_loss_weight: 0.2`——文件内自相矛盾（DRY：配置权威声明随复制失真；
  KISS：读者无法从文件头部获得真实语义）。
- `docs/contracts/emofilm_v2_schema.md`"死配置"一节仍把 `emo_loss_weight` 列为
  v2 resolved 配置禁止字段，与 `DEAD_CONFIG_KEYS` 的同步修改不一致——契约文档是
  `build_emofilm_contract.py` 声称的"人类可读 schema 单一来源"，本次未同步。
- 恢复 input-end 监督与 `docs/adr/0019` 的明确决策（"下游 speech-token hidden
  state 监督取代输入端 classifier CE""死配置字段含输入端 `emo_loss_weight`，由
  `assert_no_dead_config` 强制"）冲突，改动未更新 ADR、`.scratch/` 亦无
  emofilm-sentlvl 票据目录记录理由。按 `docs/agents/domain.md` 属应显式标注的
  ADR 冲突；即使科学决策合理，也应补记，否则历史反转重演。
- 测试文件内部残留反转语义叙述：`tests/test_emofilm_protocol.py` 模块 docstring
  仍称"活跃模型无输入端 `emotion_classifier`…v1 反模式已从活跃代码删除"；
  `tests/test_emofilm_downstream_heads.py` 模块 docstring 与
  `test_model_has_downstream_heads_and_no_input_classifier` 测试名同样过时
  （该测试已不再断言无分类器）。

### 4.2 基线坏味道（判断性）

- **Speculative Generality**：`ALLOWED_UNEXPECTED_PREFIXES` 无真实调用者，属于
  为不存在场景预埋的钩子；且恰好放错了方向。
- **Shotgun Surgery**：一个"句级监督是否启用"的逻辑改动散落在 `__init__` 条件分支、
  forward `getattr` 守卫、checkpoint 白名单、配置门禁、五处测试里；下一次翻转
  （删/改开关）仍需扫五处。
- **Mysterious Name**：`loss_emotion` 在 span 分支是词级下游头损失、在 disabled
  分支是 input-end 句级 CE，同一键两种语义，TensorBoard/日志无法区分。
- **Duplicated Code / 注释复制**：`emo_film.yaml` 的 300 行头部被整份复制进
  earlystop、sentlvl 两个派生配置，"单一活跃配置权威"的声明随之被复制失效。
- **Repeated Switches**：`emo_loss_weight>0` 的判定在 `__init__` 与 forward 各出现
  一次，且语义必须保持一致（属性存在性 = 开关），靠 `getattr` 默认值兜底；
  类结构不稳定使调用方（加载器、测试）必须知道拓扑随权重变化。
- **Duplicated Code（`emo_checkpoint.py` 内部）**：`load_base_state` 对 unexpected
  键做了两遍前缀过滤（第一遍在调用 `load_state_dict` 前，第二遍在
  `result.unexpected_keys` 上），第二遍在第一遍 raise 后恒为空，属死代码。
- **边界注记**：`ALLOWED_UNEXPECTED_PREFIXES` 同时削弱了"base 加载器拒绝 v1
  旧制品（含 `emotion_classifier`）"的既有防线——若有人把无训练元数据的 v1 ckpt
  当 base checkpoint 传入，现在会被静默接受。真实推理入口仍走 strict
  `load_trained_state`（v1 制品缺任务头会失败），故影响仅限 base 加载误用场景，
  但属于本次改动引入的未声明行为变化。

## 5. 高视角架构（更统一的设计？）

用 codebase-design 词汇看，当前设计把**可选性表达在模块拓扑层**而不是损失组合层：

- `Qwen2LM_Emotion` 的 interface（state_dict 键集、属性集、loss_dict 键集）随
  `emo_loss_weight` 变化 → 一个类呈现两种拓扑，加载器、测试、配置门禁都要知道这个
  条件。分类器只有 ~896×6 参数（冻结随机），"disabled 零差异"换来的 state_dict
  稳定，代价是 P0×2 + 白名单机制 + `getattr` 守卫 + 双拓扑测试。
- **删除测试**：删除"条件建分类器 + ALLOWED_UNEXPECTED + getattr 守卫"后，复杂度
  不会消失（训练 missing 方向仍在），但会**集中**到两个单点：损失组合
  （`emo_loss_weight>0` 才把 `loss_emotion` 加进 loss）与一行 missing 白名单。
  这是"加深模块"的信号。
- **统一拓扑方案**（建议，未实现）：恒建冻结 `emotion_classifier`，仅用
  `emo_loss_weight>0` 门控损失计算。效果：
  1. 训练 base 加载只需把 `emotion_classifier.` 加入 `ALLOWED_MISSING_PREFIXES`
     （与 FiLM/任务头同类：base ckpt 不含的新增模块）；
  2. 推理 strict 加载天然通过（模型有该键）；
  3. `ALLOWED_UNEXPECTED_PREFIXES`、`getattr` 守卫、`hasattr` 断言全部可删；
  4. 代价：disabled 模型 state_dict 多两个冻结随机键（~7KB），未来 disabled 运行
     的 identity 哈希随之变化（历史 v1 产物不受影响）。若"disabled 零差异"是硬
     合同，则退而求其次：分类器仍条件创建，但**补上 missing 白名单**并在推理
     加载器显式容忍/过滤该前缀（最小修复，见 §6）。
- **损失组合单点**：span 分支与 disabled 分支应共享同一个 loss 组合点（
  `loss = loss_tts + Σ w_i·loss_i`），而不是两个互斥分支各写一份；两条路径
  同时启用时要么都算、要么显式报错，不能靠分支顺序静默丢 input-end 损失。
- **键名分离**：`loss_emotion_input`（句级 CE）与 `loss_emotion_span`（词级头）
  分开，避免日志/聚合歧义。
- **配置 delta 化**：实验配置不应整份复制权威头部；要么用 hyperpyyaml override
  派生，要么头部按真实内容改写，避免"权威声明随复制失真"的历史绕远路。

历史绕远路的现象是真实的：v1 删分类器 → 本轮恢复，语义翻转散落为 docstring 反转、
测试锁反转、死字段门禁删除三处表达，每处都是单独改动。统一后应只有一个开关字段 +
一个损失函数表达"input-end 句级监督"，一处翻转全库一致。

## 6. 修复建议

### 最小修复（不重构，先跑通实验）

1. `ALLOWED_MISSING_PREFIXES` 增加 `"emotion_classifier."`（训练启动）；
2. 推理入口显式容忍分类器键：`tools/inference_emo_film.py` 的
   `filter_state_dict` 过滤 `emotion_classifier.*`（注释说明冻结随机权重、推理零
   影响），或给 `load_trained_state` 加允许前缀参数；
3. 修正 `run_infer.sh` 注释（描述真实机制）；
4. 修正 `emo_film_sentlvl.yaml` 头部矛盾注释。

### 深化修复（推荐，纳入下次重构）

5. 恒建冻结分类器 + 损失门控，删除 `ALLOWED_UNEXPECTED_PREFIXES` 与 `getattr`
   守卫（或保持最小修复后补做）；
6. 两路径共享 loss 组合点，`loss_emotion` 键按路径分离；
7. 新增两个集成测试：带分类器模型加载 base ckpt、disabled 模型加载带分类器
   的训练 ckpt（覆盖真实入口）；
8. 实验配置改 delta 派生或至少更新头部。

## 7. 验证记录

- 复现脚本：`/tmp/repro_emofilm_load.py`（只读，未改仓库）；
  输出见 §2.2 两段 FAIL。
- `emo_film_sentlvl.yaml` 可被 hyperpyyaml 正确解析，llm 带分类器、权重 0.2。
- pytest：`tests/ -k "emofilm or emo"` → 384 passed / 1 failed（无关：
  `test_extract_emotion2vec_frame.py` 未设 `EMOFILM_PROJECT_ROOT`）/ 89 deselected。
- 主代理全量复跑 `pytest tests -q`：470 passed / 2 failed / 2 skipped。两个失败均
  与环境相关、与本次改动无关：`test_eval_smoke`（`/tmp/smoke_*.wav` 缺失 +
  modelscope 认证过期，wav pair 为空）、`test_extract_emotion2vec_frame`
  （未设 `EMOFILM_PROJECT_ROOT` / `EMOFILM_UPSTREAM`）。
