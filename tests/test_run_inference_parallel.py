"""多 GPU 数据并行推理 launcher 测试。

不真跑模型：mock subprocess.Popen，只验证每 GPU 一个分片子进程的启动参数、
manifest 合并、失败分片上报。
"""
import json
import os
import sys

import pytest
ROOT = str(__import__("pathlib").Path(__file__).parents[1])
sys.path.insert(0, ROOT)


def _flag(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_launches_one_process_per_gpu(monkeypatch, tmp_path):
    """gpus=[1,2,3] → 3 个 Popen，CUDA_VISIBLE_DEVICES 1/2/3，--shard_idx 0/1/2 --num_shards 3。"""
    import tools.run_inference_parallel as mod

    launches = []

    class _FakeProc:
        def wait(self):
            return 0

    def _fake_popen(cmd, cwd=None, env=None, **kw):
        launches.append({"cmd": list(cmd), "cwd": cwd, "env": dict(env or {})})
        return _FakeProc()

    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)

    out = tmp_path / "out"
    out.mkdir()
    mod.run_inference_parallel("model_dir", "llm.pt", "test.jsonl", "esd_root",
                               str(out), [1, 2, 3])

    assert len(launches) == 3
    # one process per GPU, env CUDA_VISIBLE_DEVICES matches
    assert [l["env"]["CUDA_VISIBLE_DEVICES"] for l in launches] == ["1", "2", "3"]
    # PYTHONPATH must carry REPO and Matcha-TTS (else matcha import fails in child)
    pp = launches[0]["env"]["PYTHONPATH"]
    assert mod.REPO in pp
    assert "Matcha-TTS" in pp
    # cwd is the repo root
    assert launches[0]["cwd"] == mod.REPO
    # shard_idx / num_shards propagated
    assert _flag(launches[0]["cmd"], "--shard_idx") == "0"
    assert _flag(launches[1]["cmd"], "--shard_idx") == "1"
    assert _flag(launches[2]["cmd"], "--shard_idx") == "2"
    assert _flag(launches[0]["cmd"], "--num_shards") == "3"
    assert _flag(launches[0]["cmd"], "--workspace_root") == mod.REPO
    # device is cuda
    assert _flag(launches[0]["cmd"], "--device") == "cuda"


def test_merges_shard_manifests(monkeypatch, tmp_path):
    """3 个 .shard*.jsonl（不同 utt）→ 合并为 1 个，计数=总和，按 utt_id 去重。"""
    import tools.run_inference_parallel as mod

    class _FakeProc:
        def wait(self):
            return 0

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: _FakeProc())

    out = tmp_path / "wav"
    out.mkdir()
    # three shard manifests with disjoint utt sets
    shards = [["u0", "u1"], ["u2", "u3"], ["u4"]]
    for i, utts in enumerate(shards):
        path = tmp_path / f"inference_wav.shard{i}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for u in utts:
                f.write(json.dumps({"utt_id": u, "status": "success"}) + "\n")

    res = mod.run_inference_parallel("md", "ckpt", "tm", "esd",
                                     str(out), [0, 1, 2])

    assert res["merged_count"] == 5
    merged_path = tmp_path / "inference_wav.jsonl"
    assert merged_path.is_file()
    rows = [json.loads(l) for l in merged_path.read_text().splitlines() if l.strip()]
    assert {r["utt_id"] for r in rows} == {"u0", "u1", "u2", "u3", "u4"}


def test_fails_when_any_shard_fails(monkeypatch, tmp_path):
    """某 shard returncode=1 → 合并前硬失败。"""
    import tools.run_inference_parallel as mod

    rcs = iter([0, 1, 0])

    class _FakeProc:
        def __init__(self, rc):
            self._rc = rc

        def wait(self):
            return self._rc

    monkeypatch.setattr(mod.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(next(rcs)))

    out = tmp_path / "wav"
    out.mkdir()
    with pytest.raises(RuntimeError, match="shard 1"):
        mod.run_inference_parallel("md", "ckpt", "tm", "esd", str(out), [5, 6, 7])
