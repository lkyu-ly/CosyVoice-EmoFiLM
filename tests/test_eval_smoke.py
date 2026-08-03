"""eval 冒烟测试：用 CosyVoice2 合成2条 wav 做 ref→hyp 指标计算。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

EMOFILM_PY = sys.executable
ROOT = str(Path(__file__).resolve().parents[1])
EVAL_SCRIPT = f"{ROOT}/eval/eval_emo_film.py"


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
    # v3：--ref_text_manifest 必填，写一份最小 manifest 供冒烟使用
    manifest = os.path.join(tmp, "manifest.jsonl")
    with open(manifest, "w", encoding="utf-8") as f:
        for name in ["smoke_zh.wav", "smoke_en.wav"]:
            uid = os.path.splitext(name)[0]
            f.write(json.dumps({"utt_id": uid, "text": "smoke text",
                                "sentence_emotion": "neu"}) + "\n")
    return tmp, manifest


def test_eval_runs_and_outputs_valid_json():
    tmp, manifest = setup_module()
    for name in ("smoke_zh.wav", "smoke_en.wav"):
        if not os.path.isfile(f"/tmp/{name}"):
            pytest.skip(
                f"环境门控：缺少 /tmp/{name} 冒烟音频（外部资产，与主线逻辑无关）"
            )
    ref = os.path.join(tmp, "ref")
    hyp = os.path.join(tmp, "hyp")
    out = os.path.join(tmp, "result.json")

    cmd = [
        EMOFILM_PY, EVAL_SCRIPT,
        f"--ref_dir={ref}", f"--hyp_dir={hyp}",
        f"--ref_text_manifest={manifest}",
        f"--output={out}", "--device=cpu",
        "--expected_count=2",  # 复制了 smoke_zh.wav + smoke_en.wav
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=_offline_env())
    assert r.returncode == 0, r.stderr
    assert os.path.isfile(out)
    with open(out) as f:
        data = json.load(f)
    # v3 schema
    assert data["metric_contract_version"] == "emofilm-eval-v3"
    for key in ("emo_sim", "dtw_normalized", "wer", "wer_percent",
                "n_samples", "per_emotion_emo_sim"):
        assert key in data, f"missing v3 field: {key}"
    assert 0 <= data["emo_sim"] <= 100 + 1e-2
    assert data["dtw_normalized"] >= 0
    assert 0 <= data["wer"] <= 1.0
    assert data["wer_percent"] == pytest.approx(data["wer"] * 100.0)
    assert data["n_samples"] >= 1
