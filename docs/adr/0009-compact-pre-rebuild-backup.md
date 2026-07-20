# 使用代码快照与精简证据完成重构前备份

重构前为 `main@5ad481d` 和 `feature/emofilm-author-model-control@e265281` 创建 Git tag 与完整 bundle，并生成排除 `data/`、`exp/`、模型权重和缓存的源码快照。author-control v1–v5 只保留 JSON、manifest、日志和哈希，不保留五份重复转换权重；正式 v6 保留聚合、逐样本指标、技术门、运行身份、哈希和少量代表音频，全量 WAV 暂留到新版主线通过。旧 `init.pt/final.pt` 同样暂留用于新旧对照，不制作整个 76GB 工作目录镜像。
