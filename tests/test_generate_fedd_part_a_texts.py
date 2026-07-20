"""FEDD Part A 文本生成器测试（LLM 生成 500 条渐进情感过渡文本）。

不实际调用 API；fake chat client 验证：数量/情感对均衡/去重/词数过滤/音色轮换/失败兜底。
"""
import json
import os
import sys
from unittest.mock import MagicMock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _fake_chat_client():
    """fake client：每次返回请求数量的文本；首条固定为重复句（测试去重），
    其余全局唯一且 ≥8 词。计数器加锁保证并发（generate_texts 默认 10 并发）下句子仍唯一。"""
    import threading
    counter = {"n": 0}
    lock = threading.Lock()

    def fake_create(model=None, messages=None, **kw):
        # 从 user message 里解析请求条数
        user = messages[-1]["content"]
        import re
        n = int(re.search(r"Generate (\d+) ", user).group(1))
        texts = ["This duplicate sentence appears in every single batch of the fake response."]
        texts.append("too short")  # 应被词数过滤
        for _ in range(n):
            with lock:
                counter["n"] += 1
                i = counter["n"]
            texts.append(
                f"I started out feeling one way about item {i}, and slowly my "
                f"feelings changed completely by the end."
            )
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps(texts)
        return resp

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_create
    return client


def test_import():
    from tools.generate_fedd_part_a_texts import generate_texts, ORDERED_PAIRS, VOICES

    assert callable(generate_texts)
    assert len(ORDERED_PAIRS) == 20
    assert len(VOICES) == 4


def test_full_generation_500():
    from tools.generate_fedd_part_a_texts import generate_texts, VOICES

    records = generate_texts(_fake_chat_client(), model="gpt-4o", per_pair=25)
    assert len(records) == 500
    # 20 情感对 × 25
    from collections import Counter
    pairs = Counter((r["emo_from"], r["emo_to"]) for r in records)
    assert len(pairs) == 20 and set(pairs.values()) == {25}
    # 音色轮换均衡：4 音色 × 125
    voices = Counter(r["voice"] for r in records)
    assert set(voices.keys()) == set(VOICES) and set(voices.values()) == {125}
    # 全局去重（normalized）
    norm = lambda s: " ".join("".join(c.lower() for c in s if c.isalnum() or c == " ").split())
    assert len({norm(r["text"]) for r in records}) == 500
    # 词数约束
    assert all(8 <= len(r["text"].split()) <= 25 for r in records)
    # schema + instructions 提及两端情感
    r0 = records[0]
    for k in ("prompt_id", "text", "emo_from", "emo_to", "voice", "tts_instructions"):
        assert k in r0
    from tools.generate_fedd_part_a_texts import EMO_WORDS
    for r in records[:40]:
        assert EMO_WORDS[r["emo_from"]] in r["tts_instructions"]
        assert EMO_WORDS[r["emo_to"]] in r["tts_instructions"]


def test_limit_smoke():
    """--limit 10 + per_pair 1：10 条覆盖 10 个不同情感对（冒烟形态）。"""
    from tools.generate_fedd_part_a_texts import generate_texts

    records = generate_texts(_fake_chat_client(), model="gpt-4o", per_pair=1, limit=10)
    assert len(records) == 10
    assert len({(r["emo_from"], r["emo_to"]) for r in records}) == 10


def test_fail_loudly_when_not_enough_unique():
    """LLM 反复返回同一句时，不允许悄悄缩水——必须抛错。"""
    from tools.generate_fedd_part_a_texts import generate_texts

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(
        ["The same exact sentence returned every time no matter what we ask for today."] * 30
    )
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    with pytest.raises(RuntimeError):
        generate_texts(client, model="gpt-4o", per_pair=25, limit=50)
