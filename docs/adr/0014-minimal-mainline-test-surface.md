# 测试仅覆盖唯一基础主线

重构后的测试只保留五类外部合同：数据 manifest/标签来源/隔离/作者 WordSequence state-dict 形状/768d+3D 标注/拒绝预算/parquet 加载，有效模型拓扑/FiLM 与 emotion loss/冻结集合/token-mel/static batch/checkpoint，lowercase/目标文本 LLM 条件/max_len/EOS 重采样/辅助 token/RAS/prompt Flow 条件，generator 清理/批量合并/严格配对/三项指标聚合，以及三样本端到端 smoke。旧消融、诊断矩阵、PA/RL、1024d 本地标注器、per-emotion accuracy、废弃兼容行为和只锁内部实现细节的测试从活跃主线删除。
