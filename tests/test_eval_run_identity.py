"""Ticket 05（experiment-readiness）— eval 运行身份接线测试（B15）。

main CLI 产出含 ``aggregate_identity`` 的 output JSON + identity sidecar，
使事后能回答"这份命中率来自哪次运行、哪个裁判、哪份输入"。identity API
（compute_aggregate_identity / write_emofilm_evaluation_identity）的库级行为
已在 test_emofilm_local_control_e2e_smoke 覆盖；本测试覆盖 main 的接线。
"""
from __future__ import annotations

import json
import sys


def test_main_writes_aggregate_identity_and_sidecar(tmp_path, monkeypatch):
    from eval.eval_local_control import main

    wav = tmp_path / "u0.wav"
    wav.write_bytes(b"x" * 100)
    control = tmp_path / "control.jsonl"
    control.write_text(json.dumps({
        "utt_id": "u0", "text": "hello world",
        "emo_from": "ang", "emo_to": "hap",
        "boundary_word_index": None,
        "method": "midpoint_two_span_approximation",
        "part": "A",
        "label_source": "construction_known_transition",
        "intensity_policy": "fixed_medium",
    }) + "\n", encoding="utf-8")
    gen = tmp_path / "gen.jsonl"
    gen.write_text(json.dumps({
        "utt_id": "u0", "finish_reason": "eos", "wav_path": str(wav),
        "source_revision": "abc", "checkpoint_sha256": "d" * 64,
        "control_row_ref": "control/u0", "prompt_row_ref": "prompt/u0",
        "decode_config": {"max_len_hard_cap": 200}, "seed": 1986,
    }) + "\n", encoding="utf-8")
    out = tmp_path / "out.json"

    # 避免 capture_source_identity 真跑 git（仓库 dirty 会产 patch bundle 副作用）
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
    ])
    main()

    result = json.loads(out.read_text())
    # B15: output 携带 aggregate_identity（绑定确定 rows 集合）
    assert "aggregate_identity" in result
    assert isinstance(result["aggregate_identity"], str)
    # B15: identity sidecar 落盘，含运行身份
    sidecar = tmp_path / "out.json.identity.json"
    assert sidecar.is_file()
    sid = json.loads(sidecar.read_text())
    assert sid["run_kind"] == "evaluate"
    assert sid["n_eval_rows"] == 1
    assert sid["aggregate_identity"] == result["aggregate_identity"]
