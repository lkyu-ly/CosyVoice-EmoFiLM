"""emotion2vec_plus_large 加载与特征提取冒烟测试。

用法：python tests/smoke_test_emotion2vec.py
依赖：tests/smoke_test_cosyvoice2.py 已生成 /tmp/smoke_zh.wav
成功标志：utterance emb (1024,) + frame seq (T, 1024)，帧率约 50Hz；stdout 打印 OK。
"""

import os
import sys
import time

import numpy as np
from funasr import AutoModel

WAV = "/tmp/smoke_zh.wav"


def main():
    assert os.path.isfile(
        WAV
    ), f"missing {WAV}; run tests/smoke_test_cosyvoice2.py first"

    t0 = time.perf_counter()
    model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
    t_load = time.perf_counter() - t0
    print(
        f"[load] emotion2vec_plus_large ready in {t_load:.1f}s (first run includes download)"
    )

    res_utt = model.generate(WAV, granularity="utterance", extract_embedding=True)
    emb_utt = _extract_utt_emb(res_utt)
    print(f"[utt] shape={tuple(emb_utt.shape)}")

    res_frm = model.generate(WAV, granularity="frame", extract_embedding=True)
    frame_seq = _extract_frame_seq(res_frm)
    print(f"[frame] shape={tuple(frame_seq.shape)}")

    import torchaudio

    info = torchaudio.info(WAV)
    dur = info.num_frames / info.sample_rate
    est_fps = frame_seq.shape[0] / dur
    print(f"[frame-fps] estimated {est_fps:.1f} Hz (dur={dur:.2f}s)")

    assert emb_utt.shape == (1024,), f"utterance emb dim mismatch: {emb_utt.shape}"
    assert frame_seq.shape[1] == 1024, f"frame dim mismatch: {frame_seq.shape}"
    assert 40 <= est_fps <= 60, f"fps out of expected 50Hz range: {est_fps}"

    np.save(
        "/tmp/emo_frame.npy",
        frame_seq.numpy() if hasattr(frame_seq, "numpy") else frame_seq,
    )
    print("OK")


def _extract_utt_emb(res):
    """funasr 1.3.11 实测：generate 返回 [{'key', 'labels', 'scores', 'feats'}]。

    utterance granularity 下 feats 即 (1024,)；同时兜底老接口 last_hidden_state/tser_emb。
    """
    r = res[0]
    if "feats" in r:
        return r["feats"]
    if "last_hidden_state" in r:
        h = r["last_hidden_state"]
        return h.mean(dim=0) if hasattr(h, "mean") else np.mean(h, axis=0)
    if "tser_emb" in r:
        return r["tser_emb"]
    raise KeyError(f"no utterance emb field in {list(r.keys())}")


def _extract_frame_seq(res):
    """frame granularity 下 feats 形如 (T, 1024)；兜底 last_hidden_state。"""
    r = res[0]
    if "feats" in r:
        return r["feats"]
    if "last_hidden_state" in r:
        return r["last_hidden_state"]
    raise KeyError(f"no frame seq field in {list(r.keys())}")


if __name__ == "__main__":
    main()
