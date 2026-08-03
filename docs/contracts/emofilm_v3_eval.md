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
| `discriminability`（可选） | n-way 判别：n_valid / n_way_avg / nearest_ref_acc_pct / same_emotion_mean / cross_emotion_mean / gap_same_minus_cross / n_way_distribution / mean_sim_by_ref_emotion |

## 破坏性变更（相对 v2）

- 删除 dtw / dtw_euclidean / dtw_euclidean_normalized / `--dtw_dist`。
- `--ref_text_manifest` 必填；不再支持转写 ref 回退 WER。
- 新增 `--emotion_ref_manifest` 与判别指标。

## 判别指标口径

- `--emotion_ref_manifest` 指向 sources 级 jsonl（需含 speaker_id / text / sentence_emotion / wav_path）。
- 对每条 hyp，用同 (speaker_id, text) 的其他情感参考做嵌入余弦；参考情感数 ≥ 3 才计入 `n_valid`，否则计入 `n_skipped`（诚实口径，不失败）。
- ESD 全集有完整 5 情感组 → 5-way；ESD eval 集（1500）最多 4 情感 → n-way 为 3/4；FEDD 无同文本跨情感参考 → `n_valid=0`。
- **重要退化**：ESD 的 train/eval 划分把每个 (speaker, text) 组的**目标情感单独留到 eval**，sources 参考恰好缺该情感 →
  eval 集上每行的目标情感都不在参考中，`n_scored=0`，nearest-ref 准确率 / gap **结构上不可计算**（返回 `reason`，不写 NaN / 不报误导性 0%）。
  因此判别指标只在**完整 5 情感组集合**（如 `esd_prompt_match_60` 验证集）上有意义；eval 集上应视为 N/A。
