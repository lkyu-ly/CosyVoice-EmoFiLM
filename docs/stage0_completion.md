# Emo-FiLM 阶段 0 完成报告

## 环境快照

- Python: 3.10.20（conda env: `/home/lkyu/miniconda3/envs/emofilm`）
- 解释器绝对路径: `/home/lkyu/miniconda3/envs/emofilm/bin/python`
- CUDA: 可用（torch 2.3.1+cu121）
- GPU: NVIDIA GeForce RTX 4060 Ti, 16380 MiB（**注意**：与论文 RTX 4090 不同，后续训练 batch_size 可能需调小）
- 项目根: `/home/lkyu/LLM-Audio/CosyVoice-EmoFiLM`

## 关键依赖版本（最终落地）

| 包                                                   | 版本        | 备注                                                                                                                       |
| ---------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| torch                                                | 2.3.1+cu121 | 原始 requirements 不变                                                                                                     |
| numpy                                                | 1.26.4      | uv 固化：曾漂移到 2.2.6 触发 onnxruntime 报错；按 requirements.txt 用 uv pip install 锁回 1.26.4                           |
| onnxruntime-gpu                                      | 1.18.0      | requirements.txt 原版本，与 numpy 1.26.4 配对原生工作                                                                      |
| funasr                                               | 1.3.11      | 加载 emotion2vec_plus_large 的入口                                                                                         |
| Montreal_Forced_Aligner                              | 3.3.9       |                                                                                                                            |
| modelscope                                           | 1.20.0      |                                                                                                                            |
| openai-whisper                                       | 20231117    |                                                                                                                            |
| fastdtw / praatio / librosa / matplotlib / soundfile | 已装        |                                                                                                                            |
| matcha-tts                                           | 0.0.5.1     | **新增**：CosyVoice 的 `third_party/Matcha-TTS` submodule，需 `git submodule update --init` + `pip install -e . --no-deps` |

## 模型清单

| 模型                       | 路径                                                                        | 大小                      |
| -------------------------- | --------------------------------------------------------------------------- | ------------------------- |
| CosyVoice2-0.5B            | `pretrained_models/CosyVoice2-0.5B/` → `iic/CosyVoice2-0___5B/`（符号链接） | 5.3 GB                    |
| emotion2vec_plus_large     | `~/.cache/modelscope/hub/iic/emotion2vec_plus_large/`                       | 1.9 GB（model.pt 1.94GB） |
| MFA acoustic english_mfa   | `/home/lkyu/Documents/MFA/pretrained_models/acoustic/english_mfa.zip`       | v3.1.0                    |
| MFA dictionary english_mfa | `/home/lkyu/Documents/MFA/pretrained_models/dictionary/english_mfa.dict`    |                           |

## 与设计规范的偏差

1. **emotion2vec-plus 不是 PyPI 包**：设计规范 `pip install emotion2vec-plus` 是错的。正确做法是 `funasr.AutoModel(model='iic/emotion2vec_plus_large')`，funasr 通过 modelscope 自动下载权重。本阶段已在 `tests/smoke_test_emotion2vec.py` 验证。

2. **依赖固化用 uv pip install 而非升级 onnxruntime**：设计规范默认走 CosyVoice 的 requirements.txt，但 numpy 一度被 funasr/whisper 间接升级到 2.x，触发 onnxruntime 1.18 报 `_ARRAY_API not found`。修复方式是用 `uv pip install -r requirements.txt --index-strategy unsafe-best-match --no-build-isolation` 把 numpy 锁回 1.26.4（与 requirements.txt 一致），保留 onnxruntime 1.18 原版本。**不**修改 requirements.txt。

3. **CosyVoice2 路径规整**：modelscope 缓存用 `___` 替代 `.`，实际目录名是 `CosyVoice2-0___5B`。本阶段创建符号链接 `CosyVoice2-0.5B` 让代码不必关心 modelscope 命名约定。

4. **Matcha-TTS 需显式 init + pip install**：CosyVoice 的 `third_party/Matcha-TTS` 是 git submodule，初始为空目录，需要 `git submodule update --init --recursive third_party/Matcha-TTS` + `pip install -e third_party/Matcha-TTS --no-deps` 才能 `import matcha`。`scripts/activate_env.sh` 仅注入 PYTHONPATH 不够，必须装包。

5. **setup.py + 激活脚本**：让 cosyvoice 可在任意 CWD 导入；Matcha-TTS 已 pip install，激活脚本中 PYTHONPATH 仅作为冗余保险。

6. **emotion2vec_plus_large 返回字段是 `feats` 而非 `last_hidden_state`**：设计规范与原计划假设 funasr 返回字典含 `last_hidden_state` / `tser_emb`，但 funasr 1.3.11 的 `AutoModel.generate(..., extract_embedding=True)` 实际返回 `[{'key', 'labels', 'scores', 'feats'}]`，utterance 与 frame 两种 granularity 都用 `feats` 字段（分别为 `(1024,)` 与 `(T, 1024)` 的 ndarray）。`tests/smoke_test_emotion2vec.py` 已改为优先读 `feats`，原字段名作为兜底。**阶段 1 的 `extract_emotion2vec_frame.py` 与阶段 3 的 `eval_emo_film.py` 都必须用 `feats`**。另注：加载阶段会打印约 10 条 `Warning, miss key in ckpt: modality_encoders.AUDIO.decoder.*`，属于 emotion2vec_plus_large 与 funasr 内置 data2vec config 的 decoder 层不匹配（decoder 不参与 audio-only embedding 提取），不影响功能。

7. **MFA 通过 subprocess 调用时需要 PATH 注入**：MFA 内部依赖 `fstcompile` 等 openfst 二进制（emofilm env 的 `bin/` 下）与 `sqlite3` CLI（系统未装，用 `/home/lkyu/miniconda3/bin/sqlite3`）。直接 `mfa align` 命令行 OK，但通过 `subprocess.run` 调用必须显式 `PATH=emofilm/bin:conda_base/bin:$PATH`，否则在 `check_third_party()` 阶段报 `ThirdpartyError: Could not find 'fstcompile'`。`tests/smoke_test_mfa.py` 已封装 PATH 注入，阶段 1 批量对齐脚本需复用同样模式。

## 冒烟产物

| 脚本                              | 产物                                                                           | 验收                                                   |
| --------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------ |
| `tests/smoke_test_cosyvoice2.py`  | `/tmp/smoke_zh.wav`（8.88s @ 24000Hz）、`/tmp/smoke_en.wav`（5.60s @ 24000Hz） | 加载 + zh 推理 17.2s + en 推理 7.4s                    |
| `tests/smoke_test_emotion2vec.py` | `/tmp/emo_frame.npy`（1.77 MB, (443, 1024), float32）                          | 首次加载 1-2 min（下载），再次 5-6s；frame seq 49.9 Hz |
| `tests/smoke_test_mfa.py`         | `/tmp/mfa_smoke/aligned/smoke_en.TextGrid`（含 alignment_analysis.csv）        | align 耗时 22.24s，tier=[words, phones]，对齐 8 词     |
| `tests/env_check.py`              | stdout 报告                                                                    | 12 项体检全 PASS                                       |

## 进入阶段 1 的前置条件

- [x] `tests/env_check.py` 12 项全 PASS
- [x] CosyVoice2 加载与推理冒烟通过（zh + en）
- [x] emotion2vec_plus_large 可加载、特征维度正确（1024d，~50Hz）
- [x] MFA 单条对齐通过（words + phones tier）
- [x] 模型路径已规整到 `pretrained_models/CosyVoice2-0.5B/`
- [x] cosyvoice 可在任意 CWD 导入（setup.py + Matcha-TTS 已装）
- [ ] ESD / IEMOCAP 数据集已下载并索引（阶段 1 Task 4 负责）
- [ ] `data/ESD/` 与 `data/IEMOCAP/` 目录就绪（阶段 1 Task 4 负责）

## Changelog

- 2026-06-21: 用 `uv pip install -r requirements.txt --index-strategy unsafe-best-match --no-build-isolation` 固化依赖，numpy 锁定 1.26.4，onnxruntime 1.18 在 numpy<2 下原生工作，无需升级 onnxruntime。
- 2026-06-21: 添加 setup.py + scripts/activate_env.sh，cosyvoice 可在任意 CWD 导入。
- 2026-06-21: git submodule update --init 第三方 Matcha-TTS + pip install -e --no-deps；解决 `No module named 'matcha'`。
- 2026-06-21: 创建符号链接 pretrained_models/CosyVoice2-0.5B → iic/CosyVoice2-0**\_5B，清理 .\_\_\_**temp/ 残留（20K）。
- 2026-06-21: tests/smoke_test_cosyvoice2.py 跑通，zh 推理 17.2s 出 8.88s wav，en 推理 7.4s 出 5.60s wav。
- 2026-06-21: tests/smoke_test_emotion2vec.py 跑通，首次加载 1-2 min，cache 1.9GB；frame (443, 1024) @ 49.9Hz；产物 /tmp/emo_frame.npy。**关键**：funasr 1.3.11 返回字段是 `feats`，不是 `last_hidden_state`。
- 2026-06-21: tests/smoke_test_mfa.py 跑通，align 22.24s，对齐 8 词。**关键**：subprocess 调用 mfa 需注入 PATH（emofilm/bin + conda_base/bin/sqlite3）。
- 2026-06-21: tests/env_check.py 上线，12 项体检全通过。
