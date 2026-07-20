"""Emo-FiLM 阶段 0 环境一键体检。

用法：python tests/env_check.py
退出码 0 = 全部通过；非 0 = 存在失败项。
"""

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "pretrained_models", "CosyVoice2-0.5B")
MFA_BIN = os.environ.get("MFA_BIN") or shutil.which("mfa")

# 项目以源码树方式运行（未 pip 安装 cosyvoice / matcha），与 smoke 脚本一致地补 sys.path。
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn

    return deco


@check("python version")
def _py():
    assert sys.version_info[:2] == (3, 10), f"expected 3.10, got {sys.version_info[:2]}"


@check("torch + cuda")
def _torch():
    import torch

    # 环境已升级到 torch 2.13（Stage 3 起）；放宽到 2.x，保留 cuda 检查。
    assert torch.__version__.startswith("2."), torch.__version__
    assert torch.cuda.is_available(), "cuda not available"


@check("numpy pinned to 1.26.4")
def _numpy():
    import numpy

    assert numpy.__version__ == "1.26.4", numpy.__version__


@check("onnxruntime 1.18")
def _ort():
    import onnxruntime as ort

    assert ort.__version__.startswith("1.18"), ort.__version__


@check("funasr")
def _funasr():
    import funasr

    assert funasr.__version__.startswith("1.3"), funasr.__version__


@check("mfa binary")
def _mfa_bin():
    assert MFA_BIN and os.path.isfile(MFA_BIN), "set MFA_BIN or add mfa to PATH"
    import subprocess

    r = subprocess.run([MFA_BIN, "version"], capture_output=True, text=True)
    assert r.returncode == 0 and "3.3" in r.stdout, r.stdout


@check("mfa acoustic english_mfa")
def _mfa_acoustic():
    import subprocess

    assert MFA_BIN, "set MFA_BIN or add mfa to PATH"
    r = subprocess.run(
        [MFA_BIN, "model", "inspect", "acoustic", "english_mfa"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and "Acoustic model" in r.stdout, r.stdout


@check("mfa dictionary english_mfa")
def _mfa_dict():
    import subprocess

    assert MFA_BIN, "set MFA_BIN or add mfa to PATH"
    r = subprocess.run(
        [MFA_BIN, "model", "inspect", "dictionary", "english_mfa"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and "Dictionary" in r.stdout, r.stdout


@check("cosyvoice2 model files")
def _cv2_model():
    for f in [
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "cosyvoice2.yaml",
        "campplus.onnx",
        "speech_tokenizer_v2.onnx",
    ]:
        p = os.path.join(MODEL_DIR, f)
        assert os.path.isfile(p), f"missing {p}"


@check("emotion2vec_plus_large cache")
def _emo_cache():
    cache = os.path.expanduser("~/.cache/modelscope/hub/iic/emotion2vec_plus_large")
    assert os.path.isdir(cache), f"missing {cache}; run tests/smoke_test_emotion2vec.py"
    pt_files = [
        f
        for f in os.listdir(cache)
        if f.endswith(".pt") or f.endswith(".bin") or f.endswith(".safetensors")
    ]
    assert pt_files, f"no weight file in {cache}: {os.listdir(cache)}"


@check("cosyvoice import (from source tree)")
def _cv2_import():
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.llm.llm import Qwen2LM


@check("matcha-tts importable (vendored in third_party)")
def _matcha():
    import matcha

    assert matcha.__file__ is not None


def main():
    npass = nfail = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
            npass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            nfail += 1
    print(f"\nResult: {npass} pass, {nfail} fail, {len(CHECKS)} total")
    sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
