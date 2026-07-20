#!/usr/bin/env python3
"""FEDD Part A 文本生成器：chat LLM 生成 500 条渐进情感过渡文本。

对齐论文（学位论文 §FEDD）："平滑过渡样本通过自然语言指令引导 GPT-4 生成具有
渐进语义变化的文本"。替代已废弃的 50 种子 × 10 机械后缀方案。

- 20 个有序情感对 × 25 条 = 500 条互不重复的文本（normalized 去重）
- 每条 8-25 词；4 音色轮换（MiMo 英文预置音色 = Part A 说话人）
- 每条附 tts_instructions（mild transition 指令），供下游指令 TTS 使用

用法:
  # 把 chat LLM 凭证填进下方 HARDCODED_OPENAI_* 即可直接跑；
  # 或设环境变量 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_TEXT_MODEL。
  # 冒烟（10 条，覆盖 10 个情感对）:
  python tools/generate_fedd_part_a_texts.py --output /tmp/pa_smoke.jsonl --per_pair 1 --limit 10
  # 全量:
  python tools/generate_fedd_part_a_texts.py --output data/fedd_part_a_prompts.jsonl
"""
import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ── 快速测试硬编码（留空则回退环境变量）────────────────────────────
HARDCODED_OPENAI_API_KEY = ""       # 已脱敏；设环境变量 OPENAI_API_KEY（main 回退读取）
HARDCODED_OPENAI_BASE_URL = "https://yunwu.ai/v1"      # chat LLM base_url
HARDCODED_OPENAI_TEXT_MODEL = "gpt-4o"    # 如 gpt-4o
# ────────────────────────────────────────────────────────────────

EMO_WORDS = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad", "sur": "surprised"}
EMOTIONS = list(EMO_WORDS.keys())
ORDERED_PAIRS = [(a, b) for a in EMOTIONS for b in EMOTIONS if a != b]  # 20
# 4 个 MiMo-V2.5-TTS 英文预置音色（= Part A 说话人，2 女 2 男；官方文档确认）。
# 相对论文 5 人为已声明轻微偏差（MiMo 英文预置音色仅 4 个）。下游用 build_fedd_part_a_mimo.py
# 合成，voice 字段即 audio.voice，可直接传参。
VOICES = ["Mia", "Chloe", "Milo", "Dean"]
MIN_WORDS, MAX_WORDS = 8, 25

INSTRUCTIONS_TMPL = (
    "Speak this sentence starting in a {frm} tone and gradually, smoothly "
    "transition to a {to} tone by the end of the sentence. The emotional shift "
    "should be subtle and natural (a mild transition), not abrupt."
)

LLM_PROMPT_TMPL = (
    "You are helping build an evaluation dataset for fine-grained emotional "
    "speech synthesis. Generate {n} distinct English sentences a single speaker "
    "could say aloud. Requirements:\n"
    "- Each sentence is {min_w} to {max_w} words.\n"
    "- Within each sentence, the semantic content progresses naturally from a "
    "{frm} feeling at the beginning to a {to} feeling at the end (a gradual, "
    "mild emotional transition expressed through the words themselves).\n"
    "- First-person everyday speech; vary topics and sentence structures; no "
    "stage directions, no quotation marks, no emojis.\n"
    "Return ONLY a JSON array of {n} strings, no other text."
)


def _normalize(s):
    return " ".join("".join(c.lower() for c in s if c.isalnum() or c == " ").split())


def _extract_json_array(content):
    """从 LLM 回复中提取第一个 JSON 数组（容忍代码块包裹等噪声）。"""
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        raise ValueError("no JSON array in LLM response")
    arr = json.loads(m.group(0))
    if not isinstance(arr, list):
        raise ValueError("parsed JSON is not a list")
    return [str(x).strip() for x in arr if str(x).strip()]


def _collect_unique_texts(client, model, need, frm, to, seen, lock, max_attempts):
    """为一个情感对收集至多 need 条唯一、合规（词数+全局去重）的文本。

    每轮多要 5 条冗余以吸收去重/过滤损耗；凑够即停。不足则返回已收集的（由调用方判失败）。
    frm/to 为 EMO_WORDS 里的英文情感词。seen 为跨对共享的去重集合，lock 守护其并发访问
    （check-and-add 原子化，防止两个线程各自认为同一句是新句而重复入选）。
    """
    collected = []
    for attempt in range(max_attempts):
        if len(collected) >= need:
            break
        prompt = LLM_PROMPT_TMPL.format(
            n=need - len(collected) + 5, min_w=MIN_WORDS, max_w=MAX_WORDS, frm=frm, to=to)
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}])
            texts = _extract_json_array(resp.choices[0].message.content)
        except Exception as e:
            print(f"[{frm}->{to} attempt {attempt+1}] LLM/parse error: {e}")
            continue
        for t in texts:
            if len(collected) >= need:
                break
            key = _normalize(t)
            if not (MIN_WORDS <= len(t.split()) <= MAX_WORDS) or not key:
                continue
            with lock:
                if key in seen:
                    continue
                seen.add(key)
            collected.append(t)
    return collected


def _pair_needs(per_pair, limit):
    """按对顺序分配每对需求：无 limit 则每对 per_pair；有 limit 则顺序摊分至满。"""
    needs = []
    remaining = limit
    for pair in ORDERED_PAIRS:
        n = per_pair if limit is None else min(per_pair, remaining)
        if limit is not None:
            remaining -= n
        if n > 0:
            needs.append((pair, n))
    return needs


def generate_texts(client, model, per_pair=25, limit=None, max_attempts=3, concurrency=10):
    """按有序情感对**并发**生成过渡文本，返回 record 列表（凑不够即抛错，不悄悄缩水）。

    每对一个任务并发跑（默认 10）；跨对去重集合 seen 由锁守护。进度条按**目标条数**推进
    （非按对数），故 --limit 冒烟也能到 100%。records 按对顺序组装，voice 轮换/prompt_id 稳定。
    """
    needs = _pair_needs(per_pair, limit)
    target = sum(n for _, n in needs)
    seen = set()
    lock = threading.Lock()

    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut2pair = {
            ex.submit(_collect_unique_texts, client, model, n,
                      EMO_WORDS[a], EMO_WORDS[b], seen, lock, max_attempts): (a, b)
            for (a, b), n in needs
        }
        with tqdm(total=target, desc="Part A texts") as pbar:
            for fut in as_completed(fut2pair):
                collected = fut.result()
                results[fut2pair[fut]] = collected
                pbar.update(len(collected))

    records = []
    for (emo_from, emo_to), need in needs:
        collected = results[(emo_from, emo_to)]
        if len(collected) < need:
            raise RuntimeError(
                f"pair {emo_from}->{emo_to}: only {len(collected)}/{need} unique "
                f"texts after {max_attempts} attempts — refusing to under-deliver")
        for j, t in enumerate(collected):
            records.append({
                "prompt_id": f"pa_{emo_from}2{emo_to}_{j:03d}",
                "text": t,
                "emo_from": emo_from,
                "emo_to": emo_to,
                "voice": VOICES[len(records) % len(VOICES)],
                "tts_instructions": INSTRUCTIONS_TMPL.format(
                    frm=EMO_WORDS[emo_from], to=EMO_WORDS[emo_to]),
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--per_pair", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None,
                    help="总条数上限（冒烟用，如 --per_pair 1 --limit 10）")
    ap.add_argument("--concurrency", type=int, default=10, help="情感对并发数（默认 10）")
    args = ap.parse_args()

    base_url = HARDCODED_OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL")
    api_key = HARDCODED_OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    model = HARDCODED_OPENAI_TEXT_MODEL or os.environ.get("OPENAI_TEXT_MODEL")
    if not base_url or not api_key or not model:
        raise SystemExit("ERROR: 填写脚本顶部 HARDCODED_OPENAI_* 或设环境变量 "
                         "OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_TEXT_MODEL")

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    records = generate_texts(client, model, per_pair=args.per_pair, limit=args.limit,
                             concurrency=args.concurrency)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
