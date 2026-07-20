# 训练长期只保留起点与终点 checkpoint

新版训练保存一次 `init.pt`，训练期间以单一 `latest.pt` 支持中断续训，成功后原子转为 `final.pt`；不永久保存每个 epoch 的完整 checkpoint，也不额外序列化一份与最后 epoch 相同的 final。日志和 TensorBoard记录逐 epoch 指标，final 旁保存配置、数据合同 hash、Git commit、随机种子和参数 hash。基座加载只允许情感新增模块缺失、训练后和续训权重严格加载，这属于本地工程可靠性修复而非作者源码合同。
