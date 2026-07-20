# 对齐语义而不复刻执行轨迹

基础 Emo-FiLM 只对齐会改变数据人口、模型条件、训练目标、参数更新或生成分布的作者公开语义。必须对齐 768d/3D 标注合同、标签后处理、emotion loss、冻结集合、token/mel 裁剪、static batch、lowercase、目标文本 LLM 上下文、`max_len=200`、EOS-only 停止、最短长度前 EOS 重采样和 RAS fallback；不再强制 CPU ONNX provider、完整前缀重算、整块 shuffle、逐样本 seed 或逐 token 相同输出。EOS 屏蔽会重新形成 top-p/top-k 候选集合，并不严格等价于作者在原分布上抽到 EOS 后重采样；本地 RAS 预先排除首次重复 token 也会改变 fallback 分布，所以二者属于语义差异。训练 5 epoch 仍属于论文口径下的本地代理实验协议，不称作者源码合同。
