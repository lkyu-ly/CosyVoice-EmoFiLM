"""原 CosyVoice2 zero-shot smoke 回归 + 硬编码路径检查。"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
ASSET_ROOT = ROOT
EMOFILM_PY = sys.executable


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "asset", "zero_shot_prompt.wav")),
    reason="asset/zero_shot_prompt.wav 不存在（Stage 0 产物），跳过 CosyVoice2 smoke 回归。"
           "本测试不涉及 Emo 代码修改，仅验证原 CosyVoice2 不受影响。",
)
def test_cosyvoice2_smoke_still_passes():
    """Stage 0 已验证的 CosyVoice2 冒烟仍然通过。"""
    model_dir = os.path.join(ASSET_ROOT, "pretrained_models", "CosyVoice2-0.5B")
    zh_text = "收到好友从远方寄来的生日礼物。"
    zh_prompt = os.path.join(ASSET_ROOT, "asset", "zero_shot_prompt.wav")
    prompt_text = "希望你以后能够做的比我还好呦。"

    script = """
import sys
sys.path.insert(0, "{root}")
sys.path.insert(0, "{root}/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice2
import torchaudio
cv2 = CosyVoice2("{model_dir}", load_jit=False, load_trt=False, fp16=False)
for i, chunk in enumerate(cv2.inference_zero_shot("{zh_text}", "{prompt_text}", "{zh_prompt}", stream=False)):
    torchaudio.save("/tmp/smoke_regression.wav", chunk["tts_speech"].cpu(), cv2.sample_rate)
    break
print("SMOKE_REGRESSION_OK")
""".format(root=ROOT, model_dir=model_dir, zh_text=zh_text, prompt_text=prompt_text, zh_prompt=zh_prompt)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    r = subprocess.run([EMOFILM_PY, script_path], capture_output=True, text=True, timeout=180)
    os.unlink(script_path)
    assert "SMOKE_REGRESSION_OK" in r.stdout, r.stderr


def test_no_hardcoded_autodl_paths():
    """新推理文件不含 /root/autodl-tmp 硬编码路径。"""
    for fname in ["cosyvoice_emo.py", "model_emo.py", "frontend_emo.py"]:
        fpath = os.path.join(ROOT, "cosyvoice", "cli", fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            content = f.read()
        assert "/root/autodl-tmp" not in content, f"{fname} contains hardcoded path"
        assert "autodl-tmp" not in content, f"{fname} contains hardcoded path"


def test_inference_emo_api_exists():
    """CosyVoice2_Emotion 包含 inference_emo_film 方法。"""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice_emo import CosyVoice2_Emotion
    assert hasattr(CosyVoice2_Emotion, "inference_emo_film")
