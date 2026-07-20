"""FEDD Part B v2 测试（ESD 平行文本 + MFA 词边界切拼）。

用合成正弦波 wav + 手写 TextGrid fixture 验证，不依赖真实 ESD/MFA 数据。
回归重点：旧版 build_fedd.py 只写出 50ms cross-fade 窗口（产物 0.05s 碎片），
本版必须断言产物是"前半句 + 过渡 + 后半句"的完整句子。
"""
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SR = 16000
EMOTIONS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

TEXTGRID_TMPL = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {dur}
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = {dur}
        intervals: size = {n}
{intervals}
"""


def _make_textgrid(path, words):
    """words: list of (xmin, xmax, text)，含空 interval。"""
    lines = []
    for i, (x0, x1, t) in enumerate(words, 1):
        lines.append(f"        intervals [{i}]:")
        lines.append(f"            xmin = {x0}")
        lines.append(f"            xmax = {x1}")
        lines.append(f'            text = "{t}"')
    content = TEXTGRID_TMPL.format(dur=words[-1][1], n=len(words), intervals="\n".join(lines))
    with open(path, "w") as f:
        f.write(content)


@pytest.fixture
def fake_esd(tmp_path):
    """1 个说话人、5 情感、3 个文本组（组 3 文本不平行），每条 wav 2.0s 正弦波。

    每条语音的 words tier: [静音 0-0.2][hello 0.2-0.9][world 0.9-1.8][静音 1.8-2.0]
    """
    esd = tmp_path / "ESD"
    mfa = tmp_path / "mfa"
    mfa.mkdir()
    spk = "0011"
    texts = {1: "Hello world.", 2: "Good morning friend.", 3: "SAME"}
    lines = []
    for ei, emo in enumerate(["Neutral", "Angry", "Happy", "Sad", "Surprise"]):
        edir = esd / spk / emo
        edir.mkdir(parents=True)
        for g in (1, 2, 3):
            num = g + ei * 350
            utt = f"{spk}_{num:06d}"
            # 组 3 在 Angry 里文本不同 → 应被跳过
            text = texts[g] if not (g == 3 and emo == "Angry") else "DIFFERENT"
            lines.append(f"{utt}\t{text}\t{emo}")
            freq = 200 + ei * 100  # 每情感不同频率，便于检查拼接来源
            y = 0.3 * np.sin(2 * np.pi * freq * np.arange(int(2.0 * SR)) / SR)
            sf.write(str(edir / f"{utt}.wav"), y.astype(np.float32), SR)
            _make_textgrid(
                str(mfa / f"{utt}.TextGrid"),
                [(0.0, 0.2, ""), (0.2, 0.9, "hello"), (0.9, 1.8, "world"), (1.8, 2.0, "")],
            )
    (esd / spk / f"{spk}.txt").write_text("\n".join(lines) + "\n")
    return {"esd_dir": str(esd), "mfa_dir": str(mfa), "spk": spk}


def test_import():
    from tools.build_fedd_part_b_v2 import build_part_b_v2, load_parallel_groups, word_boundary_times

    assert callable(build_part_b_v2)


def test_parallel_groups_skip_mismatch(fake_esd):
    """文本不严格相等的组必须被剔除。"""
    from tools.build_fedd_part_b_v2 import load_parallel_groups

    groups = load_parallel_groups(fake_esd["esd_dir"], fake_esd["spk"])
    assert set(groups.keys()) == {1, 2}  # 组 3 被跳过
    assert groups[1]["Neutral"][1] == "Hello world."
    assert set(groups[1].keys()) == set(EMOTIONS)


def test_word_boundary_times(fake_esd):
    """2 词句子：边界词 k=1，A 侧取第 1 词右边界，B 侧取第 2 词左边界。"""
    from tools.build_fedd_part_b_v2 import word_boundary_times

    tg = os.path.join(fake_esd["mfa_dir"], "0011_000001.TextGrid")
    t_end, n_words = word_boundary_times(tg, word_index=1)
    assert n_words == 2
    assert abs(t_end - 0.9) < 1e-6
    t_start, _ = word_boundary_times(tg, word_index=1, side="start_next")
    assert abs(t_start - 0.9) < 1e-6  # 第 2 词 xmin=0.9


def test_full_sentence_output(fake_esd, tmp_path):
    """核心回归：产物必须是完整句子（≈ A 前半 + B 后半），不是 50ms 碎片。"""
    from tools.build_fedd_part_b_v2 import build_part_b_v2

    out = tmp_path / "fedd"
    entries = build_part_b_v2(
        esd_dir=fake_esd["esd_dir"],
        mfa_dirs=[fake_esd["mfa_dir"]],
        output_dir=str(out),
        speakers=[fake_esd["spk"]],
        num_per_pair=1,
        seed=42,
    )
    # 1 spk × 20 有序情感对 × 1 = 20 条
    assert len(entries) == 20
    for e in entries:
        assert e["part"] == "B"
        assert e["level"] == "strong"
        assert e["source"] == "esd_parallel_word_boundary"
        assert e["emo_from"] != e["emo_to"]
        assert e["text"] in ("Hello world.", "Good morning friend.")  # 真实完整文本
        y, sr = sf.read(e["wav_path"])
        assert sr == SR
        # A 取 [0, 0.9]，B 取 [0.9, 2.0]，减去 50ms overlap → ≈1.95s；绝不允许 ~0.05s
        assert 1.5 < len(y) / sr < 2.5, f"duration {len(y)/sr:.3f}s — 疑似只写了 cross-fade 窗口"
        rms = float(np.sqrt((y ** 2).mean()))
        assert rms > 0.05  # 非静音


def test_source_utts_and_manifest_written(fake_esd, tmp_path):
    from tools.build_fedd_part_b_v2 import build_part_b_v2

    out = tmp_path / "fedd"
    entries = build_part_b_v2(
        esd_dir=fake_esd["esd_dir"],
        mfa_dirs=[fake_esd["mfa_dir"]],
        output_dir=str(out),
        speakers=[fake_esd["spk"]],
        num_per_pair=1,
        seed=42,
    )
    src_file = out / "part_b_source_utts.txt"
    assert src_file.is_file()
    srcs = set(src_file.read_text().split())
    # 每条产物消耗 2 个源 utt；有去重（同 utt 可被不同 pair 复用）
    assert len(srcs) >= 2
    manifest = out / "manifest.jsonl"
    assert manifest.is_file()
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    assert len(rows) == len(entries)
    expected_keys = {"utt_id", "wav_path", "text", "emo_from", "emo_to",
                     "speaker_id", "boundary_word_index", "source", "part", "level"}
    assert expected_keys.issubset(rows[0].keys())
