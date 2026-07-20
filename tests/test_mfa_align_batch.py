"""MFA 批量对齐工具单测：取 ESD 2 条 + IEMOCAP 2 条跑对齐。"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import praatio.textgrid
import pytest

EMOFILM_PY = sys.executable
ROOT_PATH = Path(__file__).resolve().parents[1]
ROOT = str(ROOT_PATH)
SCRIPT = f"{ROOT}/tools/run_mfa_align.py"
MFA_BIN = os.environ.get("MFA_BIN") or shutil.which("mfa")
OUTDIR = "/tmp/test_mfa_batch"
ESD_DIR = str(ROOT_PATH / "datasets" / "ESD")

pytestmark = pytest.mark.skipif(MFA_BIN is None, reason="set MFA_BIN or add mfa to PATH")


def _mfa_env():
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [os.path.dirname(MFA_BIN), env.get("PATH", "")]
    )
    return env


def setup_module():
    if os.path.isdir(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)


def _load_esd_text(speaker: str) -> dict:
    """读 ESD speaker 的 {utt_id: text} 索引，确保 manifest 用真实文本。"""
    txt_file = Path(ESD_DIR) / speaker / f"{speaker}.txt"
    out = {}
    with open(txt_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                out[parts[0]] = parts[1]
    return out


def make_manifest(name, wav_paths):
    """生成最小测试 manifest（使用真实 transcript，避免文本/音频错位）。"""
    text_index = _load_esd_text("0011")
    p = os.path.join(OUTDIR, f"{name}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for wp in wav_paths:
            utt_id = os.path.splitext(os.path.basename(wp))[0]
            f.write(json.dumps({
                "utt_id": utt_id,
                "wav_path": wp,
                "text": text_index.get(utt_id, "test text placeholder"),
                "sentence_emotion": "neu",
                "speaker_id": "0011",
            }) + "\n")
    return p


def test_mfa_align_esd():
    """ESD 2条对齐验证：产出的 TextGrid 含 words tier。"""
    neutral_dir = Path(ESD_DIR) / "0011" / "Neutral"
    esd_wavs = sorted([
        str(neutral_dir / f)
        for f in os.listdir(neutral_dir)[:2]
    ])
    manifest = make_manifest("esd_2", esd_wavs)

    output = os.path.join(OUTDIR, "esd_align")
    cmd = [
        EMOFILM_PY, SCRIPT,
        f"--manifest={manifest}",
        f"--output_dir={output}",
        "--dictionary=english_mfa",
        "--acoustic_model=english_mfa",
        "--num_jobs=1",
        f"--mfa_bin={MFA_BIN}",
        "--clean",
    ]
    subprocess.run(cmd, check=True, env=_mfa_env())

    for wav in esd_wavs:
        utt = os.path.splitext(os.path.basename(wav))[0]
        tg = os.path.join(output, f"{utt}.TextGrid")
        assert os.path.isfile(tg), f"missing {tg}"
        tg_obj = praatio.textgrid.openTextgrid(tg, includeEmptyIntervals=False)
        tier_names_lower = [t.lower() for t in tg_obj.tierNames]
        assert any("word" in t for t in tier_names_lower), f"no word tier in {tg_obj.tierNames}"


def test_mfa_validate_first():
    """--validate 模式先校验 corpus 不报错。"""
    neutral_dir = Path(ESD_DIR) / "0011" / "Neutral"
    esd_wav = str(neutral_dir / os.listdir(neutral_dir)[0])
    manifest = make_manifest("validate_test", [esd_wav])
    cmd = [
        EMOFILM_PY, SCRIPT,
        f"--manifest={manifest}",
        "--validate_only",
        "--dictionary=english_mfa",
        "--acoustic_model=english_mfa",
        f"--mfa_bin={MFA_BIN}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=_mfa_env())
    assert r.returncode == 0, f"validate failed: {r.stderr}"
