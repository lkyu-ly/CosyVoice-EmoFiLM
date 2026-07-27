"""Ticket 07（experiment-readiness）— FEDD label 行级失败清单 + eval 排除（B14）。

B14：label_fedd_emotion2vec 的 report 加行级 ``failed_utts``（emotion2vec 一致性
失败的 utt）；eval_local_control ``--exclude_utts_file`` 读该 report 从评测排除
这些 utt（参考构造音频情感模糊，不进 exact 分母）。
"""
from __future__ import annotations

import json
import sys


class _FakeModel:
    """模拟 funasr AutoModel.generate：所有 wav 返回 ang 标签。"""

    def generate(self, wavs, **kw):
        return [{"labels": ["english/angry"], "scores": [0.9]} for _ in wavs]


def test_label_report_includes_failed_utts():
    """label_manifest report 含行级 failed_utts（B14 排除清单来源）。"""
    from tools.label_fedd_emotion2vec import label_manifest

    entries = [
        {"utt_id": "u1", "wav_path": "x", "emo_from": "ang", "emo_to": "hap", "part": "B"},
        {"utt_id": "u2", "wav_path": "x", "emo_from": "sad", "emo_to": "neu", "part": "B"},
    ]
    new_entries, report = label_manifest(_FakeModel(), entries)
    # u1: ang ∈ {ang,hap} → pass；u2: ang ∉ {sad,neu} → fail
    assert report["n_failed"] == 1
    assert report["failed_utts"][0]["utt_id"] == "u2"
    assert report["failed_utts"][0]["emo2vec_label"] == "ang"


def test_eval_main_excludes_failed_utts(tmp_path, monkeypatch):
    """eval main --exclude_utts_file 过滤掉 failed_utts（B14）。"""
    from eval.eval_local_control import main

    wav = tmp_path / "u.wav"
    wav.write_bytes(b"x" * 100)
    control = tmp_path / "control.jsonl"
    control.write_text(json.dumps({
        "utt_id": "u_bad", "text": "hello world",
        "emo_from": "ang", "emo_to": "hap",
        "boundary_word_index": None,
        "method": "midpoint_two_span_approximation",
        "part": "A", "label_source": "x", "intensity_policy": "fixed_medium",
    }) + "\n", encoding="utf-8")
    gen = tmp_path / "gen.jsonl"
    gen.write_text(json.dumps({
        "utt_id": "u_bad", "finish_reason": "eos", "wav_path": str(wav),
        "source_revision": "abc", "checkpoint_sha256": "d" * 64,
        "control_row_ref": "c/u", "prompt_row_ref": "p/u",
        "decode_config": {"max_len_hard_cap": 200}, "seed": 1986,
    }) + "\n", encoding="utf-8")
    exclude_report = tmp_path / "exclude.json"
    exclude_report.write_text(json.dumps({
        "total": 1, "passed": 0, "failed_utts": [{"utt_id": "u_bad"}], "n_failed": 1,
    }), encoding="utf-8")
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        "tools.write_emofilm_run_identity.capture_source_identity",
        lambda root, patch_bundle_path=None: {"git_head": "fake", "dirty": False},
    )
    monkeypatch.setattr(sys, "argv", [
        "eval_local_control.py",
        "--control_manifest", str(control),
        "--generation_manifest", str(gen),
        "--output", str(out),
        "--evaluator", "fake",
        "--exclude_utts_file", str(exclude_report),
    ])
    main()

    result = json.loads(out.read_text())
    # u_bad 被排除 → 无 control → n_samples 0
    assert result["n_samples"] == 0
