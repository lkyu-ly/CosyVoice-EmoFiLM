# 将新版派生产物集中到单一数据合同目录

新版基础 Emo-FiLM 数据统一写入 `data/contracts/emofilm_author_v1/`，目录内共同保存 manifests、emotion2vec-base frames、word blocks、tagged 数据、src、分别直接打包的 train/cv parquet 及 provenance。原始 `datasets/`、现有 MFA 和模型资产只引用而不复制；不再先打包全量 parquet 再重新编码 train/cv，也不保留 `full/no_word/no_global` 等旧消融命名。新版通过完整性与训练加载检查后，旧 plus-large 特征、词块和消融数据可按已确认清理策略删除。
