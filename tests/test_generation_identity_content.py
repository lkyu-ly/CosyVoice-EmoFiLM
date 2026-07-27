"""Ticket 03（experiment-readiness）— 生成身份指纹纳入合成输入 + 缺身份守卫（B6）。

B6 核心：skip-existing 指纹必须反映**实际合成内容**（``text_digest`` +
``prompt_audio_ref``），否则改 manifest 续跑会静默复用与当前控制条件不符的
旧 WAV。同时既有 row 缺任一身份族时必须拒绝复用（防"无身份"被当"身份匹配"，
即 v1 退化路径里缺身份 row 之间 ``"" == ""`` 误复用任意内容）。
"""
from __future__ import annotations

import hashlib

from tools.write_emofilm_run_identity import (
    check_skip_existing,
    generation_request_fingerprint,
    generation_row_identity_components,
)


def _base_request_kwargs(**over):
    kw = dict(
        source="abc123",
        checkpoint_sha256="d" * 64,
        control_row_ref="control/u1",
        prompt_row_ref="prompt/p.wav",
        decode_config={"max_len_hard_cap": 200},
        seed=1986,
        text_digest=hashlib.sha256(b"hello").hexdigest(),
        prompt_audio_ref="esd/p.wav",
    )
    kw.update(over)
    return kw


def test_fingerprint_includes_text_digest():
    """改变合成文本 → 指纹改变（B6 核心：合成内容进指纹）。"""
    fp1 = generation_request_fingerprint(
        **_base_request_kwargs(text_digest=hashlib.sha256(b"hello").hexdigest())
    )
    fp2 = generation_request_fingerprint(
        **_base_request_kwargs(text_digest=hashlib.sha256(b"world").hexdigest())
    )
    assert fp1 != fp2


def test_fingerprint_includes_prompt_audio_ref():
    """改变 prompt 音频 → 指纹改变。"""
    fp1 = generation_request_fingerprint(**_base_request_kwargs(prompt_audio_ref="esd/a.wav"))
    fp2 = generation_request_fingerprint(**_base_request_kwargs(prompt_audio_ref="esd/b.wav"))
    assert fp1 != fp2


def test_components_reports_empty_families():
    """缺身份族的 components 摘要为空（供 check_skip 守卫判定）。"""
    comps = generation_row_identity_components({"utt_id": "u1", "seed": 1})
    assert comps["source"] == ""
    assert comps["checkpoint"] == ""
    assert comps["control"] == ""
    assert comps["prompt"] == ""


def test_check_skip_rejects_missing_identity(tmp_path):
    """既有 row 缺身份族 → 拒绝复用（防退化路径 ""=="" 误复用）。"""
    (tmp_path / "u1.wav").write_bytes(b"x")
    existing = {
        "utt_id": "u1",
        "finish_reason": "eos",
        "wav_path": "u1.wav",
        "source_revision": "abc123",
        "checkpoint_sha256": "d" * 64,
        "decode_config": {"max_len_hard_cap": 200},
        "seed": 1986,
        # 缺 control_row_ref / prompt_row_ref
    }
    request_fp = generation_request_fingerprint(**_base_request_kwargs())
    decision = check_skip_existing(existing, request_fp, workspace_root=str(tmp_path))
    assert not decision.skip
    assert "missing" in decision.reason


def test_check_skip_matches_when_identity_and_content_same(tmp_path):
    """完整身份 + 相同合成内容 → 允许复用。"""
    (tmp_path / "u1.wav").write_bytes(b"x")
    td = hashlib.sha256(b"hello").hexdigest()
    existing = {
        "utt_id": "u1",
        "finish_reason": "eos",
        "wav_path": "u1.wav",
        "source_revision": "abc123",
        "checkpoint_sha256": "d" * 64,
        "control_row_ref": "control/u1",
        "prompt_row_ref": "prompt/p.wav",
        "decode_config": {"max_len_hard_cap": 200},
        "seed": 1986,
        "text_digest": td,
        "prompt_audio_ref": "esd/p.wav",
    }
    request_fp = generation_request_fingerprint(**_base_request_kwargs(text_digest=td))
    decision = check_skip_existing(existing, request_fp, workspace_root=str(tmp_path))
    assert decision.skip


def test_check_skip_rejects_when_content_differs(tmp_path):
    """身份齐全但合成内容（text_digest）不同 → 拒绝复用（B6 核心）。"""
    (tmp_path / "u1.wav").write_bytes(b"x")
    existing = {
        "utt_id": "u1",
        "finish_reason": "eos",
        "wav_path": "u1.wav",
        "source_revision": "abc123",
        "checkpoint_sha256": "d" * 64,
        "control_row_ref": "control/u1",
        "prompt_row_ref": "prompt/p.wav",
        "decode_config": {"max_len_hard_cap": 200},
        "seed": 1986,
        "text_digest": hashlib.sha256(b"old-text").hexdigest(),
        "prompt_audio_ref": "esd/p.wav",
    }
    # 请求侧用新文本摘要 → 指纹不匹配 → 不复用
    request_fp = generation_request_fingerprint(
        **_base_request_kwargs(text_digest=hashlib.sha256(b"new-text").hexdigest())
    )
    decision = check_skip_existing(existing, request_fp, workspace_root=str(tmp_path))
    assert not decision.skip
