# 对齐作者的采样分布语义

最短长度前采用作者的 EOS 重采样，而不是把 EOS 设为负无穷后重新执行 top-p/top-k；RAS fallback 也使用原始 scores，不预先排除第一次抽到的重复 token。两项差异都会改变实际候选集合或概率分布，因此属于源码语义对齐；仍不要求相同随机种子、随机数消费顺序或逐 token 输出，KV cache 和当前 ONNX provider继续保留。
