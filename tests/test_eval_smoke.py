"""eval 冒烟测试：用 CosyVoice2 合成2条 wav 做 ref→hyp 指标计算。"""
import json
import os
import shutil
import subprocess
import tempfile

EMOFILM_PY = "/home/hanlvyuan/miniconda3/envs/emofilm/bin/python"
ROOT = "/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM"
EVAL_SCRIPT = f"{ROOT}/eval/eval_emo_film.py"
PER_EMO_SCRIPT = f"{ROOT}/eval/per_emo_accuracy.py"


def _offline_env():
    """返回带 MODELSCOPE_OFFLINE=true 的 env，跳过 modelscope 更新检查（避免 2.88G 重复下载）。"""
    env = os.environ.copy()
    env["MODELSCOPE_OFFLINE"] = "true"
    return env


def setup_module():
    tmp = tempfile.mkdtemp(prefix="eval_smoke_")
    ref_dir = os.path.join(tmp, "ref")
    hyp_dir = os.path.join(tmp, "hyp")
    os.makedirs(ref_dir)
    os.makedirs(hyp_dir)
    # 复用 Stage 0 冒烟产物
    for name in ["smoke_zh.wav", "smoke_en.wav"]:
        src = f"/tmp/{name}"
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(ref_dir, name))
            shutil.copy(src, os.path.join(hyp_dir, name))
    return tmp


def test_eval_runs_and_outputs_valid_json():
    tmp = setup_module()
    ref = os.path.join(tmp, "ref")
    hyp = os.path.join(tmp, "hyp")
    out = os.path.join(tmp, "result.json")

    cmd = [
        EMOFILM_PY, EVAL_SCRIPT,
        f"--ref_dir={ref}", f"--hyp_dir={hyp}",
        f"--output={out}", "--device=cpu",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=_offline_env())
    assert r.returncode == 0, r.stderr
    assert os.path.isfile(out)
    with open(out) as f:
        data = json.load(f)
    assert "emo_sim" in data
    assert "dtw" in data
    assert "wer" in data
    assert "n_samples" in data
    assert 0 <= data["emo_sim"] <= 100 + 1e-2  # cos sim 浮点误差容差（emo_sim=dot*100，归一化向量 dot 理论 ≤1.0）
    assert data["dtw"] >= 0
    assert data["n_samples"] >= 1


def test_per_emo_accuracy_runs():
    """per_emo_accuracy 基本可运行（需要 WordSequenceModel 存在时）。"""
    tmp = setup_module()
    hyp = os.path.join(tmp, "hyp")
    out = os.path.join(tmp, "per_emo.json")
    ckpt = f"{ROOT}/checkpoints/word_sequence_model_best.pt"
    if not os.path.isfile(ckpt):
        print("SKIP: no WordSequenceModel checkpoint (run Stage 1 first)")
        return
    cmd = [
        EMOFILM_PY, PER_EMO_SCRIPT,
        f"--hyp_dir={hyp}", f"--annotator_ckpt={ckpt}",
        f"--output={out}", "--device=cpu",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=_offline_env())
    assert r.returncode == 0, r.stderr
    with open(out) as f:
        data = json.load(f)
    assert "accuracy" in data
    assert 0 <= data["accuracy"] <= 100
