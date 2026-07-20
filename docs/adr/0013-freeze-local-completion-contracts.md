# 冻结作者未公开环节的本地补全合同

核心代码重构不改写现有 ESD 1500 条测试清单及训练排除关系、FEDD-rebuilt Part A MiMo/Part B ESD 拼接、A/B 分区、prompt 与 neutral anchor、本地数据索引、speech token、speaker embedding、parquet 打包、批量并行推理和 `emofilm-eval-v2` 的 WER/Emo-SIM/DTW。上述内容统一标为本地补全实现或本地代理评测，不称作者官方实现；只有确定的工程缺陷、引用断裂或数据泄漏才触发单独修复，原始数据与最终 manifest 保留，中间缓存可按数据合同重建。
