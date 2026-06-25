#!/usr/bin/env bash
# Emo-FiLM 通用激活脚本：激活 conda env + 设置 Matcha-TTS 路径
source /home/lkyu/miniconda3/etc/profile.d/conda.sh
conda activate emofilm
export PYTHONPATH="/home/lkyu/LLM-Audio/CosyVoice-EmoFiLM/third_party/Matcha-TTS:${PYTHONPATH}"
export EMOFILM_ROOT="/home/lkyu/LLM-Audio/CosyVoice-EmoFiLM"
cd "${EMOFILM_ROOT}"
