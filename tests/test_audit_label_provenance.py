"""audit_label_provenance.py 的门禁行为测试（ADR-0003）。"""
import json
import subprocess
import sys

import pytest

PY = sys.executable
AUDIT = "tools/audit_label_provenance.py"


def _run(manifest_path, profile, reference=None, cwd=None):
    cmd = [PY, AUDIT, "--manifest", manifest_path, "--profile", profile]
    if reference:
        cmd += ["--reference", reference]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def _write(tmp_path, name, records):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)


def test_esd_control_passes_valid_global_label(tmp_path):
    rec = {"utt_id": "0011_000001",
           "text": "<emotion type='ang' intensity='medium'>the text</emotion>",
           "label_role": "control", "label_source": "dataset_global_label",
           "intensity_policy": "fixed_medium", "granularity": "utterance"}
    ref = {"utt_id": "0011_000001", "sentence_emotion": "ang"}
    m = _write(tmp_path, "esd.jsonl", [rec])
    r = _write(tmp_path, "raw.jsonl", [ref])
    res = _run(m, "esd_control", reference=r)
    assert res.returncode == 0, res.stdout + res.stderr


def test_esd_control_fails_target_derived_pseudo_label(tmp_path):
    # 模拟当前漂移：label_source 不是 dataset_global_label + emotion 与 raw 冲突
    rec = {"utt_id": "0011_000001",
           "text": "<emotion type='sur' intensity='high'>the text</emotion>",
           "label_source": "word_annotator_pseudo_label"}
    ref = {"utt_id": "0011_000001", "sentence_emotion": "ang"}
    m = _write(tmp_path, "esd.jsonl", [rec])
    r = _write(tmp_path, "raw.jsonl", [ref])
    res = _run(m, "esd_control", reference=r)
    assert res.returncode == 1
    assert "label_source" in res.stdout
    assert "sentence_emotion" in res.stdout  # emotion 冲突被检出


def test_fedd_control_fails_missing_method(tmp_path):
    rec = {"utt_id": "fedd_b_0001", "text": "<emotion type='ang' intensity='medium'>a</emotion> <emotion type='hap' intensity='medium'>b</emotion>",
           "emo_from": "ang", "emo_to": "hap"}  # 缺 method/label_source/granularity
    m = _write(tmp_path, "fedd.jsonl", [rec])
    res = _run(m, "fedd_control")
    assert res.returncode == 1
    assert "method" in res.stdout


def test_no_word_detects_neu_low_collapse(tmp_path):
    # plain text 全塌 neu/low：非中性情感缺失 → 须 fail
    recs = [{"utt_id": f"u{i}", "text": f"<emotion type='neu' intensity='low'>text{i}</emotion>",
             "label_source": "dataset_global_label"} for i in range(3)]
    m = _write(tmp_path, "nw.jsonl", recs)
    res = _run(m, "no_word")
    assert res.returncode == 1
    assert "非中性情感未覆盖" in res.stdout
