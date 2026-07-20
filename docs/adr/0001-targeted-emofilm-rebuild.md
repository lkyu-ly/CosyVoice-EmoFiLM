# 定向重建基础 Emo-FiLM 专用模块

基础 Emo-FiLM 修复采用定向重建：重建被旧消融、DPO 和兼容分支污染的 `llm_emotion.py`、`model_emo.py`、`frontend_emo.py`，原地精简训练入口与 optimizer，并只局部修改 processor、tokenizer 和采样语义；保留通用 CosyVoice、Flow/HiFT，以及作者未公开但本地已有合理实现的数据构建、批量推理和评测链。这样既能获得单一、可审计的基础主线，也避免复制作者研究原型中的硬编码、调试代码和不完整接口。
