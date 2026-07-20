#!/usr/bin/env bash
# Emo-FiLM 通用激活脚本：激活 conda env + 设置项目根 + Matcha-TTS 路径
# 项目根入 PYTHONPATH：torchrun 子进程需能 import cosyvoice 包（脚本目录入 path 但项目根未入）
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-emofilm}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export EMOFILM_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${EMOFILM_ROOT}:${EMOFILM_ROOT}/third_party/Matcha-TTS:${PYTHONPATH}"
cd "${EMOFILM_ROOT}"
