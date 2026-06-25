"""CosyVoice2 加载与推理冒烟测试。

用法：python tests/smoke_test_cosyvoice2.py
成功标志：生成 /tmp/smoke_zh.wav 与 /tmp/smoke_en.wav，stdout 打印 OK。
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))

import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2

MODEL_DIR = os.path.join(ROOT, "pretrained_models", "CosyVoice2-0.5B")
ZH_PROMPT = os.path.join(ROOT, "asset", "zero_shot_prompt.wav")
EN_PROMPT = os.path.join(ROOT, "asset", "cross_lingual_prompt.wav")

ZH_TEXT = (
    "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。"
)
ZH_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
EN_TEXT = (
    "And then later on, fully acquiring that company. So keeping management in line."
)


def main():
    assert os.path.isfile(
        os.path.join(MODEL_DIR, "cosyvoice2.yaml")
    ), f"missing cosyvoice2.yaml in {MODEL_DIR}"
    assert os.path.isfile(ZH_PROMPT), f"missing {ZH_PROMPT}"
    assert os.path.isfile(EN_PROMPT), f"missing {EN_PROMPT}"

    t0 = time.perf_counter()
    cv = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
    t_load = time.perf_counter() - t0
    print(f"[load] CosyVoice2 ready in {t_load:.1f}s")

    t0 = time.perf_counter()
    for i, chunk in enumerate(
        cv.inference_zero_shot(ZH_TEXT, ZH_PROMPT_TEXT, ZH_PROMPT, stream=False)
    ):
        torchaudio.save("/tmp/smoke_zh.wav", chunk["tts_speech"].cpu(), cv.sample_rate)
        break
    t_zh = time.perf_counter() - t0
    print(f"[infer-zh] {t_zh:.1f}s -> /tmp/smoke_zh.wav")

    t0 = time.perf_counter()
    for i, chunk in enumerate(
        cv.inference_cross_lingual(EN_TEXT, EN_PROMPT, stream=False)
    ):
        torchaudio.save("/tmp/smoke_en.wav", chunk["tts_speech"].cpu(), cv.sample_rate)
        break
    t_en = time.perf_counter() - t0
    print(f"[infer-en] {t_en:.1f}s -> /tmp/smoke_en.wav")

    info_zh = torchaudio.info("/tmp/smoke_zh.wav")
    info_en = torchaudio.info("/tmp/smoke_en.wav")
    assert (
        info_zh.num_frames > cv.sample_rate
    ), f"zh wav too short: {info_zh.num_frames} frames"
    assert (
        info_en.num_frames > cv.sample_rate
    ), f"en wav too short: {info_en.num_frames} frames"
    print(
        f"[wav-zh] {info_zh.num_frames} frames @ {info_zh.sample_rate}Hz = {info_zh.num_frames/info_zh.sample_rate:.2f}s"
    )
    print(
        f"[wav-en] {info_en.num_frames} frames @ {info_en.sample_rate}Hz = {info_en.num_frames/info_en.sample_rate:.2f}s"
    )
    print("OK")


if __name__ == "__main__":
    main()
