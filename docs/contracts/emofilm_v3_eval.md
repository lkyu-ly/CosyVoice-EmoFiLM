# EmoFiLM v3 评测契约（emofilm-eval-v3）

权威实现：`eval/eval_emo_film.py`（CLI）+ `eval/emotion_metrics.py`（纯函数）。

## CLI

`python eval/eval_emo_film.py --ref_dir REF --hyp_dir HYP --output OUT
--expected_count N --ref_text_manifest EVAL.jsonl
[--emotion_ref_manifest SOURCES.jsonl] [--batch_size 16] [--device cuda]`

## 输出 schema

| 字段 | 语义 |
|---|---|
| `metric_contract_version` | 恒 `emofilm-eval-v3` |
| `n_samples` | 参与聚合样本数 |
| `emo_sim` | frame 均值池化余弦 ×100 |
| `dtw_normalized` | cosine fastdtw 按路径长度归一化 |
| `wer` / `wer_percent` | WER 比例 / 展示百分比（GT 文本 vs hyp 转写） |
| `per_emotion_emo_sim` | 按 emotion 分组的 emo_sim 均值 |
| `discriminability`（可选） | n-way 判别：n_valid / n_skipped / n_scored / n_way_avg / nearest_ref_acc_pct / same_emotion_mean / cross_emotion_mean / gap_same_minus_cross / n_way_distribution / mean_sim_by_ref_emotion。目标情感参考 = eval manifest 的 reference_wav，其他情感参考 = sources 索引；候选 >=3 才计入（ESD eval 集 1422/1500 可算，78 条跳过）。 |

## 破坏性变更（相对 v2）

- 删除 dtw / dtw_euclidean / dtw_euclidean_normalized / `--dtw_dist`。
- `--ref_text_manifest` 必填；不再支持转写 ref 回退 WER。
- 新增 `--emotion_ref_manifest` 与判别指标。

## 判别指标口径

- `--emotion_ref_manifest` 指向 sources 级 jsonl（需含 speaker_id / text / sentence_emotion / wav_path），提供"其他情感"参考。
- 目标情感参考来自 eval manifest 的 `reference_wav` 字段（绝对路径）；`compute_discriminability` 内部按 (speaker_id, text) 合并 sources 索引与 reference_wav，使候选包含"正确答案"。
- 候选情感数 ≥ 3 才计入 `n_valid`，否则计入 `n_skipped`（诚实口径，不失败）；所有返回分支统一 schema，不可算时返回 `reason`，不写 NaN / 不报误导性 0%。
- same_emotion_mean / cross_emotion_mean / gap / mean_sim_by_ref_emotion 为**原始余弦**（与 emo_sim ×100 不同）；nearest_ref_acc_pct 为百分比。
- ESD eval 集（1500）合并 reference_wav 后 1422 条候选 ≥3（5/4/3-way 混合），78 条跳过；FEDD_A 无同文本跨情感参考 → `n_scored=0`（诚实口径，返回 reason），FEDD_B 部分组可判别 → `n_scored=427`（4-way×264 / 3-way×163，73 条跳过）。
