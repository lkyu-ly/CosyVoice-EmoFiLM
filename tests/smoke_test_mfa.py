"""MFA 强制对齐最小流程冒烟测试。

用法：python tests/smoke_test_mfa.py
依赖：tests/smoke_test_cosyvoice2.py 已生成 /tmp/smoke_en.wav
成功标志：/tmp/mfa_smoke/aligned/smoke_en.TextGrid 包含 words 与 phones 层。
"""

import os
import shutil
import subprocess
import sys

WAV = "/tmp/smoke_en.wav"
TEXT = "And then later on fully acquiring that company"
CORPUS = "/tmp/mfa_smoke/corpus"
OUTPUT = "/tmp/mfa_smoke/aligned"
MFA_BIN = os.environ.get("MFA_BIN") or shutil.which("mfa")


def main():
    assert MFA_BIN, "set MFA_BIN or add mfa to PATH"
    assert os.path.isfile(
        WAV
    ), f"missing {WAV}; run tests/smoke_test_cosyvoice2.py first"

    if os.path.isdir("/tmp/mfa_smoke"):
        shutil.rmtree("/tmp/mfa_smoke")
    os.makedirs(CORPUS, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    shutil.copy(WAV, os.path.join(CORPUS, "smoke_en.wav"))
    with open(os.path.join(CORPUS, "smoke_en.lab"), "w") as f:
        f.write(TEXT + "\n")

    cmd = [
        MFA_BIN,
        "align",
        CORPUS,
        "english_mfa",
        "english_mfa",
        OUTPUT,
        "--clean",
        "--overwrite",
        "--num_jobs",
        "1",
        "--output_format",
        "long_textgrid",
    ]
    print("[cmd]", " ".join(cmd))
    # MFA needs openfst binaries (fstcompile, etc.) on PATH; ensure the conda
    # env's bin directory is front of PATH for the subprocess even when this
    # script is launched without the conda env activated.
    env = os.environ.copy()
    env_bin = os.path.dirname(MFA_BIN)
    extra_bins = [env_bin]
    # MFA also shells out to `sqlite3` (not bundled in the emofilm env).
    # Prefer a conda sqlite3 if present, fall back to system PATH.
    sqlite3_bin = shutil.which("sqlite3")
    if sqlite3_bin:
        extra_bins.append(os.path.dirname(sqlite3_bin))
    env["PATH"] = os.pathsep.join(extra_bins) + os.pathsep + env.get("PATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        print(f"FAIL: mfa align exited {r.returncode}")
        sys.exit(1)

    tg_path = os.path.join(OUTPUT, "smoke_en.TextGrid")
    assert os.path.isfile(tg_path), f"missing {tg_path}"

    from praatio import textgrid

    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
    tier_names = list(tg.tierNames)
    print(f"[tiers] {tier_names}")
    assert any("word" in t.lower() for t in tier_names), f"no word tier in {tier_names}"
    assert any(
        "phone" in t.lower() for t in tier_names
    ), f"no phone tier in {tier_names}"

    words = tg.getTier([t for t in tier_names if "word" in t.lower()][0]).entries
    print(f"[words] {[w.label for w in words]}")
    assert len(words) >= 3, f"too few words aligned: {len(words)}"
    print("OK")


if __name__ == "__main__":
    main()
