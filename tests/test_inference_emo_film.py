"""Emo-FiLM 批量推理脚本测试。

不实际加载 CosyVoice2 模型（避免依赖 5GB 权重）；
仅测试 ckpt 过滤、prompt 选择等纯逻辑。
"""
import hashlib
import os
import sys

import pytest
import torch

ROOT = str(__import__("pathlib").Path(__file__).parents[1])
sys.path.insert(0, ROOT)


def test_import():
    from tools.inference_emo_film import filter_state_dict, select_prompt_wav
    assert callable(filter_state_dict)
    assert callable(select_prompt_wav)


def test_filter_state_dict_strips_epoch_step():
    """含 epoch/step 的 ckpt 字典应被过滤为纯 state_dict。"""
    from tools.inference_emo_film import filter_state_dict

    sd = {
        "layer1.weight": torch.zeros(2, 2),
        "layer1.bias": torch.zeros(2),
        "epoch": 5,
        "step": 1234,
    }
    filtered = filter_state_dict(sd)
    assert "epoch" not in filtered
    assert "step" not in filtered
    assert "layer1.weight" in filtered
    assert "layer1.bias" in filtered


def test_filter_state_dict_passes_pure_state_dict():
    """纯 state_dict（无 epoch/step）应原样返回。"""
    from tools.inference_emo_film import filter_state_dict

    sd = {"layer1.weight": torch.zeros(2, 2)}
    filtered = filter_state_dict(sd)
    assert filtered == sd


def test_select_prompt_wav_picks_same_speaker_neutral(tmp_path):
    """对 ESD 测试 manifest 行，prompt 选同 speaker 的 Neutral wav。"""
    from tools.inference_emo_film import select_prompt_wav

    utt = {
        "utt_id": "esd_0012_Angry_000001",
        "speaker_id": "0012",
        "sentence_emotion": "ang",
        "wav_path": "/data/ESD/0012/Angry/0012_000001.wav",
    }
    esd_root = tmp_path / "ESD"
    for emo in ["Angry", "Happy", "Neutral", "Sad", "Surprise"]:
        emo_dir = esd_root / "0012" / emo
        emo_dir.mkdir(parents=True)
        (emo_dir / "0012_000001.wav").write_bytes(b"dummy")
    prompt_path = select_prompt_wav(utt, esd_root=str(esd_root))
    assert "Neutral" in prompt_path
    assert "0012" in prompt_path


def test_select_prompt_wav_fallback_when_no_neutral(tmp_path):
    """缺 Neutral 时回退到第一个可用 emotion。"""
    from tools.inference_emo_film import select_prompt_wav

    utt = {"speaker_id": "0099", "sentence_emotion": "ang"}
    esd_root = tmp_path / "ESD"
    happy_dir = esd_root / "0099" / "Happy"
    happy_dir.mkdir(parents=True)
    (happy_dir / "0099_000001.wav").write_bytes(b"dummy")

    prompt_path = select_prompt_wav(utt, esd_root=str(esd_root))
    assert "Happy" in prompt_path  # fallback


def test_resolve_prompt_requires_manifest_prompt_for_fedd_part_a(tmp_path):
    from tools.inference_emo_film import resolve_prompt
    utt = {"utt_id": "fedd_a_pa_ang2hap_000", "speaker_id": "Mia",
           "part": "A", "source": "mimo_api", "text": "demo text"}
    with pytest.raises(FileNotFoundError, match="prompt_wav"):
        resolve_prompt(utt, esd_root=str(tmp_path))


def test_resolve_prompt_prefers_manifest_prompt_fields(tmp_path):
    from tools.inference_emo_film import resolve_prompt
    prompt_wav = tmp_path / "prompt.wav"
    prompt_wav.write_bytes(b"dummy")
    utt = {"utt_id": "fedd_a_pa_ang2hap_000", "speaker_id": "Mia", "part": "A",
           "source": "mimo_api", "prompt_wav": str(prompt_wav),
           "prompt_text": "reference prompt text"}
    resolved = resolve_prompt(utt, esd_root=str(tmp_path))
    assert resolved["ok"] is True
    assert resolved["prompt_wav"] == str(prompt_wav)
    assert resolved["prompt_text"] == "reference prompt text"
    assert resolved["prompt_source"] == "manifest"


def test_resolve_prompt_uses_esd_neutral_for_part_b(tmp_path):
    from tools.inference_emo_film import resolve_prompt
    neutral_dir = tmp_path / "0011" / "Neutral"
    neutral_dir.mkdir(parents=True)
    neutral_wav = neutral_dir / "0011_000001.wav"
    neutral_wav.write_bytes(b"dummy")
    utt = {"utt_id": "fedd_b_0011_ang2hap_0000", "speaker_id": "0011",
           "part": "B", "source": "esd_parallel_word_boundary",
           "prompt_text": "reference"}
    resolved = resolve_prompt(utt, esd_root=str(tmp_path))
    assert resolved["ok"] is True
    assert resolved["prompt_wav"] == str(neutral_wav)
    assert resolved["prompt_source"] == "esd_same_speaker_neutral"


def test_load_emofilm_model_uses_requested_device(monkeypatch):
    import tools.inference_emo_film as mod

    called = {}

    class _FakeLLM:
        def state_dict(self):
            return {"layer.weight": 1}

        def load_state_dict(self, state_dict, strict=True):
            called["strict"] = strict
            called["state_dict"] = state_dict

    class _FakeModel:
        def __init__(self):
            self.llm = _FakeLLM()

    class _FakeCV2:
        def __init__(self, model_dir, fp16=False):
            self.model = _FakeModel()

    def _fake_torch_load(path, map_location=None, weights_only=None):
        called["map_location"] = map_location
        return {"layer.weight": 1, "epoch": 5}

    monkeypatch.setattr(mod, "torch", type("T", (), {"load": staticmethod(_fake_torch_load)}))
    monkeypatch.setitem(__import__("sys").modules, "cosyvoice.cli.cosyvoice_emo",
                        type("M", (), {"CosyVoice2_Emotion": _FakeCV2}))

    mod.load_emofilm_model("dummy_model", "dummy_ckpt", fp16=False, device="cpu")

    assert called["map_location"] == "cpu"


def test_run_inference_fails_on_missing_prompt(tmp_path, monkeypatch):
    """Part A 缺 prompt 必须硬失败，不能静默跳过。"""
    import json
    from tools.inference_emo_film import run_inference

    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps({
            "utt_id": "fedd_a_pa_ang2hap_000",
            "speaker_id": "Mia",
            "part": "A",
            "text": "demo text",
        }) + "\n",
        encoding="utf-8",
    )

    class _FakeCV2:
        sample_rate = 24000

    with pytest.raises(FileNotFoundError, match="prompt"):
        run_inference(_FakeCV2(), str(manifest), str(tmp_path), str(tmp_path / "out"))


def test_run_inference_resolves_relative_prompt_against_workspace(tmp_path, monkeypatch):
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    workspace = tmp_path / "workspace"
    prompt = workspace / "datasets" / "ESD" / "prompt.wav"
    prompt.parent.mkdir(parents=True)
    prompt.write_bytes(b"dummy")
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(
        manifest,
        [{
            "utt_id": "u0",
            "text": "text",
            "prompt_wav": "datasets/ESD/prompt.wav",
            "prompt_text": "reference",
        }],
    )
    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    cv2 = _FakeCV2Runner()
    run_inference(
        cv2,
        str(manifest),
        str(workspace / "datasets" / "ESD"),
        str(tmp_path / "out"),
        workspace_root=str(workspace),
        source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
        decode_config={"max_len_hard_cap": 200},
    )

    assert cv2.calls[0]["prompt_wav_path"] == str(prompt)


def test_run_inference_empty_generation_produces_diagnostic_row(tmp_path, monkeypatch):
    """空生成（无 chunk）→ finish_reason=sampler_error 诊断 row，不写 WAV。"""
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(
        manifest,
        [{"utt_id": "u0", "text": "text", "prompt_wav": str(prompt), "prompt_text": "ref",
          "control_row_ref": "c0", "prompt_row_ref": "p0"}],
    )
    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    class _EmptyCV2:
        sample_rate = 24000

        def inference_emo_film(self, **kwargs):
            if False:
                yield kwargs

    results = run_inference(_EmptyCV2(), str(manifest), str(tmp_path), str(tmp_path / "out"),
                            seed=1986, source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                            decode_config={"min_token_text_ratio": 2, "max_token_text_ratio": 20,
                                           "max_len_hard_cap": 2000})
    r = results[0]
    # 空生成 → sampler_error（不 raise、不写 WAV）
    assert r["finish_reason"] == "sampler_error"
    assert "wav_path" not in r
    assert r["status"] == "non_eos_diagnostic"
    assert r["seed"] == 1986


def test_run_inference_propagates_runtime_failure(tmp_path, monkeypatch):
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(
        manifest,
        [{"utt_id": "u0", "text": "text", "prompt_wav": str(prompt), "prompt_text": "ref"}],
    )
    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    class _FailingCV2:
        sample_rate = 24000

        def inference_emo_film(self, **kwargs):
            raise RuntimeError("synthetic generation failure")

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        run_inference(_FailingCV2(), str(manifest), str(tmp_path), str(tmp_path / "out"))


def _write_jsonl(path, rows):
    import json
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class _FakeCV2Runner:
    """Fake CosyVoice2 for run_inference loop tests.

    Records each inference_emo_film call; optionally snapshots the current
    manifest line count at the start of each call so tests can verify that the
    implementation wrote the manifest mid-loop (periodic save).
    """
    sample_rate = 24000

    def __init__(self, manifest_path=None):
        self.calls = []
        self.manifest_snapshots = []
        self._manifest_path = manifest_path

    def inference_emo_film(self, text_with_emo, prompt_text, prompt_wav_path):
        if self._manifest_path is not None:
            n = 0
            if os.path.isfile(self._manifest_path):
                with open(self._manifest_path) as f:
                    n = sum(1 for line in f if line.strip())
            self.manifest_snapshots.append(n)
        self.calls.append({"text_with_emo": text_with_emo,
                           "prompt_wav_path": prompt_wav_path})
        yield {"tts_speech": torch.zeros(1, 100), "finish_reason": "eos"}


def test_run_inference_shard_selection(tmp_path, monkeypatch):
    """num_shards=3 shard_idx=1 over 6 samples → only entries[1::3] (2 items)."""
    import json
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    utts = [{"utt_id": f"u{i}", "text": f"txt{i}", "prompt_wav": str(prompt),
             "prompt_text": "ref"} for i in range(6)]
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, utts)

    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    out = tmp_path / "out"
    cv2 = _FakeCV2Runner()
    run_inference(cv2, str(manifest), str(tmp_path), str(out),
                  num_shards=3, shard_idx=1,
                  source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                  decode_config={"max_len_hard_cap": 200})

    # entries[1::3] of 6 = indices 1, 4
    assert sorted(c["text_with_emo"] for c in cv2.calls) == ["txt1", "txt4"]
    manifest_path = tmp_path / "inference_out.shard1.jsonl"
    assert manifest_path.is_file()
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    assert {r["utt_id"] for r in rows} == {"u1", "u4"}


def test_run_inference_skip_existing(tmp_path, monkeypatch):
    """skip_existing: 既有 row 身份匹配 + WAV 存在 → 跳过；身份不匹配 → 重生成。"""
    import json
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    decode_cfg = {"min_token_text_ratio": 5.0, "max_token_text_ratio": 25.0,
                  "max_len_hard_cap": 600}

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    utts = [{"utt_id": "u0", "text": "t0", "prompt_wav": str(prompt), "prompt_text": "ref",
             "control_row_ref": "ctrl/u0", "prompt_row_ref": "prompt/u0"},
            {"utt_id": "u1", "text": "t1", "prompt_wav": str(prompt), "prompt_text": "ref",
             "control_row_ref": "ctrl/u1", "prompt_row_ref": "prompt/u1"}]
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, utts)

    out = tmp_path / "out"
    out.mkdir()
    (out / "u0.wav").write_bytes(b"already-there")  # pre-place existing output

    # Pre-write existing manifest with matching identity for u0 (seed=1986)
    existing_manifest = tmp_path / "inference_out.jsonl"
    existing_row = {
        "utt_id": "u0", "finish_reason": "eos",
        "source_revision": "9c6d84b", "checkpoint_sha256": "a" * 64,
        "control_row_ref": "ctrl/u0", "prompt_row_ref": "prompt/u0",
        "decode_config": decode_cfg, "seed": 1986,
        "wav_path": "out/u0.wav",
        # B6: 合成输入摘要需与请求侧匹配，否则指纹不等 → 不 skip
        "text_digest": hashlib.sha256(b"t0").hexdigest(),
        "prompt_audio_ref": "prompt.wav",
    }
    existing_manifest.write_text(json.dumps(existing_row) + "\n")

    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    cv2 = _FakeCV2Runner()
    results = run_inference(cv2, str(manifest), str(tmp_path), str(out),
                            skip_existing=True, seed=1986,
                            source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                            decode_config=decode_cfg, workspace_root=str(tmp_path))

    by_id = {r["utt_id"]: r for r in results}
    # u0 identity matches → skipped (cv2 not called)
    assert by_id["u0"]["status"] == "skipped_existing"
    assert by_id["u0"]["wav_path"] == "out/u0.wav"
    # u1 no existing row → generated
    assert by_id["u1"]["finish_reason"] == "eos"
    # cv2.inference_emo_film called only for u1
    assert [c["text_with_emo"] for c in cv2.calls] == ["t1"]


def test_run_inference_periodic_save(tmp_path, monkeypatch):
    """save_every=2: manifest written mid-loop after 2nd item (seen during 3rd call)."""
    import json
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    utts = [{"utt_id": f"u{i}", "text": f"t{i}", "prompt_wav": str(prompt),
             "prompt_text": "ref"} for i in range(3)]
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, utts)

    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    out = tmp_path / "out"
    manifest_path = tmp_path / "inference_out.jsonl"
    cv2 = _FakeCV2Runner(str(manifest_path))
    run_inference(cv2, str(manifest), str(tmp_path), str(out), save_every=2,
                  source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                  decode_config={"max_len_hard_cap": 200})

    # During the 3rd inference call the manifest must already hold 2 rows —
    # proves a periodic write fired after the 2nd result was appended.
    assert 2 in cv2.manifest_snapshots
    # final manifest has all 3
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 3


def test_run_inference_logs_fixed_interval_progress(tmp_path, monkeypatch, caplog):
    import logging
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    utts = [{"utt_id": f"u{i}", "text": f"t{i}", "prompt_wav": str(prompt),
             "prompt_text": "ref"} for i in range(3)]
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, utts)

    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    with caplog.at_level(logging.INFO, logger="tools.inference_emo_film"):
        run_inference(
            _FakeCV2Runner(),
            str(manifest),
            str(tmp_path),
            str(tmp_path / "out"),
            save_every=2,
            shard_idx=0,
            source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
            decode_config={"max_len_hard_cap": 200},
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("shard=0 progress=2/3" in message for message in messages)
    assert any("elapsed=" in message and "avg_s_per_sample=" in message for message in messages)


def test_run_inference_shard_manifest_naming(tmp_path, monkeypatch):
    """num_shards=2 → manifest name contains .shard{idx}.jsonl (no plain manifest)."""
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    utts = [{"utt_id": "u0", "text": "t0", "prompt_wav": str(prompt), "prompt_text": "ref"}]
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, utts)
    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    out = tmp_path / "out"
    cv2 = _FakeCV2Runner()
    run_inference(cv2, str(manifest), str(tmp_path), str(out),
                  num_shards=2, shard_idx=0,
                  source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                  decode_config={"max_len_hard_cap": 200})

    assert (tmp_path / "inference_out.shard0.jsonl").is_file()
    # in shard mode the per-process script must NOT write the plain merged name
    assert not (tmp_path / "inference_out.jsonl").is_file()


# ============================================================
# Task 4 / #5: GenerationRow + seed + per-utt reset + 安全 skip
# ============================================================


def test_run_inference_writes_generation_row_with_seed(tmp_path, monkeypatch):
    """产物 row 含 seed + 四族身份，过 validate_generation_row。"""
    import json
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference
    from tools.build_emofilm_contract import validate_generation_row

    decode_cfg = {"min_token_text_ratio": 2, "max_token_text_ratio": 20,
                  "max_len_hard_cap": 2000}
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, [{"utt_id": "u1", "text": "hi", "tagged_text": "<emotion>hi</emotion>",
                             "prompt_wav": str(prompt), "prompt_text": "ref",
                             "control_row_ref": "ctrl/u1", "prompt_row_ref": "prompt/spk1"}])
    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    rows = run_inference(_FakeCV2Runner(), str(manifest), str(tmp_path),
                         str(tmp_path / "out"),
                         llm_ckpt_sha="a" * 64, source_revision="9c6d84b",
                         seed=1986, decode_config=decode_cfg,
                         workspace_root=str(tmp_path))
    r = rows[0]
    assert r["finish_reason"] == "eos"
    assert r["seed"] == 1986
    assert r["source_revision"] == "9c6d84b"
    assert r["checkpoint_sha256"] == "a" * 64
    assert r["control_row_ref"] == "ctrl/u1"
    assert r["prompt_row_ref"] == "prompt/spk1"
    assert r["decode_config"] == decode_cfg
    # wav_path 是 workspace-relative POSIX
    assert "/" in r["wav_path"]
    assert not r["wav_path"].startswith("/")
    # 过 validate_generation_row（含 seed + 四族身份齐全）
    validate_generation_row(r)


def test_per_utt_seed_reset_reproducible(tmp_path, monkeypatch):
    """per-utt RNG 重置：同 seed → 每个 utt 捕获相同的随机序列（可复现）。"""
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    decode_cfg = {"min_token_text_ratio": 2, "max_token_text_ratio": 20,
                  "max_len_hard_cap": 2000}
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    utts = [{"utt_id": f"u{i}", "text": "txt", "prompt_wav": str(prompt),
             "prompt_text": "ref", "control_row_ref": f"c{i}", "prompt_row_ref": f"p{i}"}
            for i in range(3)]
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, utts)
    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    class _SeedCapturingCV2:
        """捕获 inference_emo_film 入口时的 torch RNG 随机序列。"""
        sample_rate = 24000

        def __init__(self):
            self.captured = []

        def inference_emo_film(self, **kwargs):
            self.captured.append(torch.randn(4).clone())
            yield {"tts_speech": torch.zeros(1, 100), "finish_reason": "eos"}

    cv2_run1 = _SeedCapturingCV2()
    run_inference(cv2_run1, str(manifest), str(tmp_path), str(tmp_path / "out"),
                  seed=1986, source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                  decode_config=decode_cfg)

    cv2_run2 = _SeedCapturingCV2()
    run_inference(cv2_run2, str(manifest), str(tmp_path), str(tmp_path / "out2"),
                  seed=1986, source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                  decode_config=decode_cfg)

    # per-utt reset：同一 run 内每个 utt 捕获相同的随机序列
    assert len(cv2_run1.captured) == 3
    assert torch.equal(cv2_run1.captured[0], cv2_run1.captured[1])
    assert torch.equal(cv2_run1.captured[1], cv2_run1.captured[2])
    # 跨 run 可复现：两次 run 的序列完全一致
    assert len(cv2_run2.captured) == 3
    for i in range(3):
        assert torch.equal(cv2_run1.captured[i], cv2_run2.captured[i])


def test_non_eos_skips_wav(tmp_path, monkeypatch):
    """LLM 返 max_len_reached → 不写 wav、row 无 wav_path、finish_reason 正确。"""
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    decode_cfg = {"min_token_text_ratio": 2, "max_token_text_ratio": 20,
                  "max_len_hard_cap": 2000}
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, [{"utt_id": "u1", "text": "hi",
                             "prompt_wav": str(prompt), "prompt_text": "ref",
                             "control_row_ref": "c1", "prompt_row_ref": "p1"}])
    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    class _NonEosCV2:
        """模拟 LLM 到达 max_len 未出 EOS。"""
        sample_rate = 24000

        def inference_emo_film(self, **kwargs):
            yield {"tts_speech": None, "finish_reason": "max_len_reached"}

    rows = run_inference(_NonEosCV2(), str(manifest), str(tmp_path),
                         str(tmp_path / "wav"),
                         seed=1986, source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                         decode_config=decode_cfg, workspace_root=str(tmp_path))
    r = rows[0]
    assert r["finish_reason"] == "max_len_reached"
    assert "wav_path" not in r
    assert r["seed"] == 1986
    assert not (tmp_path / "wav" / "u1.wav").exists()


def test_skip_existing_seed_change_not_reused(tmp_path, monkeypatch):
    """既有 row seed=1986，请求 seed=42 → 指纹不同 → 不 skip（重生成）。"""
    import json
    import tools.inference_emo_film as mod
    from tools.inference_emo_film import run_inference

    decode_cfg = {"min_token_text_ratio": 5.0, "max_token_text_ratio": 25.0,
                  "max_len_hard_cap": 600}
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"dummy")
    manifest = tmp_path / "m.jsonl"
    _write_jsonl(manifest, [{"utt_id": "u0", "text": "t0",
                             "prompt_wav": str(prompt), "prompt_text": "ref",
                             "control_row_ref": "ctrl/u0", "prompt_row_ref": "prompt/u0"}])

    out = tmp_path / "out"
    out.mkdir()
    (out / "u0.wav").write_bytes(b"existing-wav")

    # 既有 manifest row（seed=1986，身份与请求匹配除了 seed）
    existing_manifest = tmp_path / "inference_out.jsonl"
    existing_row = {
        "utt_id": "u0", "finish_reason": "eos",
        "source_revision": "9c6d84b", "checkpoint_sha256": "a" * 64,
        "control_row_ref": "ctrl/u0", "prompt_row_ref": "prompt/u0",
        "decode_config": decode_cfg, "seed": 1986,
        "wav_path": "out/u0.wav",
    }
    existing_manifest.write_text(json.dumps(existing_row) + "\n")

    monkeypatch.setattr(mod.torchaudio, "save", lambda *a, **k: None)

    cv2 = _FakeCV2Runner()
    results = run_inference(cv2, str(manifest), str(tmp_path), str(out),
                            skip_existing=True, seed=42,  # 不同 seed
                            source_revision="9c6d84b", llm_ckpt_sha="a" * 64,
                            decode_config=decode_cfg, workspace_root=str(tmp_path))

    r = results[0]
    # seed=42 != 1986 → 指纹不同 → 不 skip → 重新生成
    assert r["finish_reason"] == "eos"
    assert r["seed"] == 42  # 新 seed（不是旧 1986）
    assert r["status"] == "success"
    # cv2 被调用了（证明未跳过）
    assert len(cv2.calls) == 1
