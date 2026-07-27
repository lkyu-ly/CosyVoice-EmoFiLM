"""Ticket 12 — 贯通三分区端到端修复 smoke（capstone）。

本测试是 EmoFiLM v2 修复的最高层主接缝：在一个 CPU fake 闭环里把 01-11 各票的
**公开 API** 串起来，证明三分区（ESD utterance / FEDD-A approximate / FEDD-B exact）
控制 row 经公开 v2 推理入口 → 结构化 DecodeResult → generation identity（11）→
评测入口（09 FEDD span/transition + 10 triplet intensity）→ 逐样本/逐 span rows +
aggregate 能完整运行。

只组合各票公开 API（MAP §0 v1 只读；不改任何 v2 模块）：
  - 05 ``Qwen2LM_Emotion.inference`` + ``DecodeResult``（仅 eos 产 token / 落 WAV）
  - 11 ``check_skip_existing`` + ``write_emofilm_generation_identity``
  - 09 ``evaluate_fedd_dataset`` + ``FakeForcedAligner``
  - 10 ``evaluate_triplet_dataset`` + ``build_triplet_member_eval_row``
  - 01 ``validate_span`` / ``validate_generation_row`` / ``validate_eval_row`` /
       ``validate_aggregate`` / ``normalize_workspace_path``

CPU fake（MAP §4）：``_FakeQwen`` 透传 hidden，脚本采样器控制 EOS；fake WAV bytes；
``FakeAcousticEvaluator`` + ``FakeForcedAligner``。不加载真实模型，不需 GPU。

真实 GPU smoke（ESD/FEDD-A/FEDD-B 各一条真实生成 + 真实 emotion2vec/MFA）是
**延后门禁**（需真实 v2 checkpoint，属 ticket 13 前置），不在本 CPU DoD 内。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cosyvoice.llm.llm_emotion import (
    DEFAULT_MAX_LEN_HARD_CAP,
    DecodeResult,
    Qwen2LM_Emotion,
)
from eval.acoustic_evaluators import (
    EMOTION_LABEL_SPACE,
    FakeAcousticEvaluator,
    SyntheticReferenceClip,
)
from eval.eval_local_control import (
    build_aggregate_from_rows,
    evaluate_fedd_dataset,
)
from eval.triplet_eval import (
    INTENSITY_TIERS,
    INTENSITY_TIER_TO_ID,
    build_triplet_aggregate,
    build_triplet_member_eval_row,
    evaluate_triplet_dataset,
)
from tests._emofilm_fakes import (
    ClipMappedEvaluator,
    FakeForcedAligner,
    FakeWerEvaluator,
    _FakeBackbone,
    _FakeHF,
    _FakeQwen,
)
from tools.build_emofilm_contract import (
    normalize_workspace_path,
    validate_aggregate,
    validate_eval_row,
    validate_generation_row,
    validate_span,
)
from tools.write_emofilm_run_identity import (
    check_skip_existing,
    generation_request_fingerprint,
    write_emofilm_generation_identity,
)


# ============================================================
# fakes：从 tests._emofilm_fakes 复用（_FakeBackbone / _FakeHF / _FakeQwen
# 原先在本文件本地定义，现已 DRY 整合到共享测试辅助模块）。注：原本地版
# forward 的 mask tensor 未带显式 ``device=xs.device``，整合版统一带
# device；CPU 测试下行为完全等价。
# ============================================================


VOCAB = 10             # speech_token_size；eos_token = VOCAB
EMO = 6
INTEN = 4
MODEL_DIM = 4
TEXT_LEN = 3
# decode min_len = int(TEXT_LEN * min_token_text_ratio=2) = 6；采样器必须累积
# >= min_len 个 token 后才发 EOS，否则触发 EOS-before-min 重采样。
DECODE_MIN_LEN = 6

CKPT_SHA = "ab" * 32            # 64-char hex checkpoint sha256
SOURCE_REV = "0f" * 20          # 40-char hex git revision
SEED = 1986                     # per-request 固定随机种子（Task 4 / #5）
DECODE_CONFIG = {
    "min_token_text_ratio": 2,
    "max_token_text_ratio": 20,
    "max_len_hard_cap": DEFAULT_MAX_LEN_HARD_CAP,
}

PROMPT_ROW = {
    "utt_id": "prompt_speaker_0011",
    "speaker_id": "speaker_0011",
    "flow_path": "artifacts/flow/flow_model.pt",
    "hift_path": "artifacts/hift/hift_model.pt",
}

EMOTION_NAME_TO_ID = {"ang": 1, "hap": 2, "neu": 3, "sad": 4, "sur": 5}
EMOTION_ID_TO_NAME = {v: k for k, v in EMOTION_NAME_TO_ID.items()}


# ============================================================
# 采样器
# ============================================================


def _sampler_eos_after(min_len, eos=VOCAB, valid=2):
    """累积 >= min_len 个 token 后发 EOS。"""
    def _s(scores, decoded, sampling):
        if len(decoded) >= min_len:
            return eos
        return valid
    return _s


def _sampler_always_valid(valid=2):
    """恒发合法 token，永不发 EOS → max_len_reached。"""
    def _s(scores, decoded, sampling):
        return valid
    return _s


def _make_model(sampling):
    return Qwen2LM_Emotion(
        llm_input_size=MODEL_DIM,
        llm_output_size=MODEL_DIM,
        speech_token_size=VOCAB,
        emotion_vocab_size=EMO,
        intensity_vocab_size=INTEN,
        llm=_FakeQwen(MODEL_DIM),
        sampling=sampling,
    )


# ============================================================
# 控制 row 构造（均为合法 SupervisionSpan + 各分区构造元数据）
# ============================================================


def _one_hot(emotion_name: str) -> list[float]:
    idx = EMOTION_LABEL_SPACE.index(emotion_name)
    dist = [0.0] * 5
    dist[idx] = 1.0
    return dist


def _esd_control_row() -> dict[str, Any]:
    """ESD utterance：utterance-level emotion（one-hot 硬标签），intensity_mask=False
    （fixed_medium 仅控制输入，**不伪造强度真值**；无 arousal / vad / raw_score）。"""
    emotion = "hap"
    return {
        "utt_id": "esd-0001",
        "label_source": "esd_fixed_medium_control",
        "supervision_granularity": "utterance",
        "start_sec": 0.0,
        "end_sec": 3.0,
        "emotion_soft_distribution": _one_hot(emotion),
        "control_emotion_id": EMOTION_NAME_TO_ID[emotion],
        "control_intensity_id": 2,           # medium（控制输入）
        "calibrated": False,
        "emotion_mask": True,
        "intensity_mask": False,             # NO fabricated intensity GT
        "supervision_weight": 1.0,
        "provenance": "esd:0001:fixed_medium",
        "intensity_policy": "fixed_medium",
        # 评测消费的非合同字段
        "source_dataset": "esd",
        "text": "the bright sun is shining today",
        "emotion": emotion,
    }


def _fedd_a_control_row() -> dict[str, Any]:
    """FEDD-A approximate：midpoint 两段近似（MiMo TTS 无真实词边界）。"""
    emo_from, emo_to = "ang", "hap"
    return {
        "utt_id": "fedd-a-0001",
        "label_source": "construction_known_transition",
        "supervision_granularity": "span",
        "start_sec": 0.0,
        "end_sec": 3.0,
        "emotion_soft_distribution": _one_hot(emo_from),
        "control_emotion_id": EMOTION_NAME_TO_ID[emo_from],
        "control_intensity_id": 2,
        "calibrated": False,
        "emotion_mask": True,
        "intensity_mask": False,
        "supervision_weight": 1.0,
        "provenance": "fedd:part_a:0001:approx_midpoint",
        "intensity_policy": "fixed_medium",
        # FEDD 构造字段（09 evaluate_fedd_dataset 消费）
        "source_dataset": "fedd_rebuilt",
        "part": "A",
        "method": "midpoint_two_span_approximation",
        "emo_from": emo_from,
        "emo_to": emo_to,
        "text": "angry beginning then happy ending",
    }


def _fedd_b_control_row() -> dict[str, Any]:
    """FEDD-B exact：真实 MFA 词边界拼接（boundary_word_index 已知）。"""
    emo_from, emo_to = "neu", "sad"
    return {
        "utt_id": "fedd-b-0001",
        "label_source": "construction_known_transition",
        "supervision_granularity": "span",
        "start_sec": 0.0,
        "end_sec": 3.0,
        "emotion_soft_distribution": _one_hot(emo_from),
        "control_emotion_id": EMOTION_NAME_TO_ID[emo_from],
        "control_intensity_id": 2,
        "calibrated": False,
        "emotion_mask": True,
        "intensity_mask": False,
        "supervision_weight": 1.0,
        "provenance": "fedd:part_b:0001:exact_concat_boundary",
        "intensity_policy": "fixed_medium",
        # FEDD 构造字段
        "source_dataset": "fedd_rebuilt",
        "part": "B",
        "method": "exact_concatenation_boundary",
        "boundary_word_index": 3,            # 前 3 个词属于 emo_from 段
        "emo_from": emo_from,
        "emo_to": emo_to,
        # 5 个词；前 3 = neu，后 2 = sad
        "text": "calm neutral steady then sad ending",
    }


# ============================================================
# 推理 + generation row 构造
# ============================================================


def _run_inference(model, control_row, text_len=TEXT_LEN):
    """经公开 v2 推理入口生成；返回 (DecodeResult, yielded_tokens)。"""
    emotion_ids = torch.full((1, text_len), control_row["control_emotion_id"], dtype=torch.long)
    intensity_ids = torch.full((1, text_len), control_row["control_intensity_id"], dtype=torch.long)
    text_token = torch.tensor([[2 + i for i in range(text_len)]])
    tokens = list(model.inference(
        text_token=text_token,
        text_len=torch.tensor([text_len], dtype=torch.int32),
        emotion_ids=emotion_ids,
        intensity_ids=intensity_ids,
        prompt_speech_token=torch.zeros(1, 0, dtype=torch.long),
        prompt_speech_token_len=torch.tensor([0], dtype=torch.int32),
        embedding=torch.zeros(1, MODEL_DIM),
        sampling=25,
        max_token_text_ratio=DECODE_CONFIG["max_token_text_ratio"],
        min_token_text_ratio=DECODE_CONFIG["min_token_text_ratio"],
        max_len_hard_cap=DECODE_CONFIG["max_len_hard_cap"],
    ))
    return model.last_decode_result, tokens


def _build_generation_row(control_row, decode_result, prompt_row, wav_rel_path=None):
    """构造合法 GenerationRow（通过 validate_generation_row）。

    仅 EOS 行携 wav_path；非 EOS 不得带 wav_path（合同强制）。
    """
    row: dict[str, Any] = {
        "utt_id": control_row["utt_id"],
        "finish_reason": decode_result.finish_reason,
        "source_revision": SOURCE_REV,
        "checkpoint_sha256": CKPT_SHA,
        "control_row": dict(control_row),
        "prompt_row": dict(prompt_row),
        "decode_config": dict(DECODE_CONFIG),
        "seed": SEED,
    }
    if decode_result.finish_reason == "eos":
        assert wav_rel_path is not None, "eos row requires a wav_path"
        row["wav_path"] = wav_rel_path
    validate_generation_row(row)
    return row


def _request_fingerprint(control_row, prompt_row):
    """构造与 generation row 比对的请求指纹（11）。"""
    return generation_request_fingerprint(
        source=SOURCE_REV,
        checkpoint_sha256=CKPT_SHA,
        control_row=dict(control_row),
        prompt_row=dict(prompt_row),
        decode_config=DECODE_CONFIG,
        seed=SEED,
    )


# ============================================================
# A. 三分区端到端闭环（capstone）
# ============================================================


def test_three_partition_e2e_closed_loop(tmp_path):
    """三分区控制 row → v2 推理 → generation identity → 评测 → aggregate 的完整闭环。

    覆盖 brief 12 §A 全部 DoD：
      - 三个 v2 控制 row 经公开推理入口生成三个结构化 DecodeResult；
      - 仅 EOS 完成项写（fake）WAV；
      - 每 generation row 可追溯到 control/prompt/checkpoint/source/decode_config
        （validate_generation_row）；
      - ESD：utterance-level emotion，intensity_mask=False（不伪造强度真值）；
      - FEDD-A：evidence_tier=approximate，不进 exact-boundary aggregate；
      - FEDD-B：前/后 span + transition direction + exact-boundary（fake aligner 已知边界）；
      - 至少一个 intensity triplet 经同一闭环 → 逐组 + aggregate；
      - 所有 aggregate 能从持久化 rows 重算（determinism）。
    """
    # --- 输出目录（fake WAV；workspace-relative POSIX）---
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    workspace = tmp_path

    # --- 三个 v2 控制 row（均通过 validate_span）---
    esd = _esd_control_row()
    fedd_a = _fedd_a_control_row()
    fedd_b = _fedd_b_control_row()
    for span in (esd, fedd_a, fedd_b):
        validate_span(span)  # 所有控制 row 是合法 SupervisionSpan

    # ESD：诚实性断言 —— 无伪造强度真值
    assert esd["intensity_mask"] is False
    assert "arousal" not in esd
    assert esd["intensity_policy"] == "fixed_medium"

    # --- 公开 v2 推理入口（05）：三分区各一条 → 结构化 DecodeResult → generation row ---
    model = _make_model(_sampler_eos_after(min_len=DECODE_MIN_LEN))
    gen_rows: list[dict[str, Any]] = []
    for ctrl in (esd, fedd_a, fedd_b):
        decode_result, tokens = _run_inference(model, ctrl)
        # 仅 eos 产 speech token（05 门控不变量）
        assert decode_result.finish_reason == "eos"
        assert len(tokens) == decode_result.num_valid_speech_tokens
        assert len(tokens) >= decode_result.min_len

        # 仅 EOS 写（fake）WAV；wav_path 必须 workspace-relative POSIX
        wav_rel = f"wav/{ctrl['utt_id']}.wav"
        (workspace / wav_rel).write_bytes(b"RIFFfake-wav-bytes-for-smoke")
        normalize_workspace_path(wav_rel, workspace)  # 路径规整（schema §7）

        gen_row = _build_generation_row(ctrl, decode_result, PROMPT_ROW, wav_rel_path=wav_rel)
        gen_rows.append(gen_row)

    esd_gen, fedd_a_gen, fedd_b_gen = gen_rows

    # --- 组合 11 的 check_skip_existing 进闭环：同请求二次到达 → 安全复用 ---
    # wav_path 是 workspace-relative POSIX，必须传 workspace_root 才能 isfile 命中
    # （上方已 (workspace / wav_rel).write_bytes(...) 创建真实 wav）。
    for ctrl, gen_row in zip((esd, fedd_a, fedd_b), gen_rows):
        decision = check_skip_existing(
            gen_row, _request_fingerprint(ctrl, PROMPT_ROW),
            workspace_root=workspace,
        )
        assert decision.skip is True, (
            f"expected safe skip for {ctrl['utt_id']}, got: {decision.reason}"
        )

    # --- 写 generation identity（11）---
    gen_manifest = workspace / "generation.jsonl"
    gen_manifest.write_text(
        "\n".join(json.dumps(r, default=str) for r in gen_rows) + "\n",
        encoding="utf-8",
    )
    gen_identity = write_emofilm_generation_identity(
        workspace / "generation_identity.json",
        code_root=workspace,
        command="test_emofilm_local_control_e2e_smoke::three_partition_closed_loop",
        checkpoint_sha256=CKPT_SHA,
        decode_config_defaults=DECODE_CONFIG,
        generation_manifest_path=str(gen_manifest),
        n_generation_rows=len(gen_rows),
    )
    assert gen_identity["schema_version"] == 2
    assert gen_identity["contract_name"] == "emofilm"
    assert gen_identity["n_generation_rows"] == 3
    assert (workspace / "generation_identity.json").is_file()

    # ============================================================
    # 09 FEDD span/transition 评测：FEDD-A（approximate）+ FEDD-B（exact）
    # ============================================================
    fedd_controls = [fedd_a, fedd_b]
    fedd_gen_rows = [fedd_a_gen, fedd_b_gen]

    # Fake emotion evaluator + 合成 clip（已知 transition 时刻 / 情感）。
    emotion_eval = FakeAcousticEvaluator(kind="emotion")
    clip_map = {
        fedd_a["utt_id"]: SyntheticReferenceClip(
            wav_path=fedd_a_gen["wav_path"], duration_sec=3.0,
            known_transition_sec=1.5,
            known_transition_from=fedd_a["emo_from"],
            known_transition_to=fedd_a["emo_to"],
        ),
        fedd_b["utt_id"]: SyntheticReferenceClip(
            wav_path=fedd_b_gen["wav_path"], duration_sec=3.0,
            known_transition_sec=1.5,
            known_transition_from=fedd_b["emo_from"],
            known_transition_to=fedd_b["emo_to"],
        ),
    }
    mapped_eval = ClipMappedEvaluator(emotion_eval, clip_map)

    # Fake aligner：为 FEDD-B 注册已知词边界（boundary_word_index=3 → words[2].end=1.5s）。
    b_words = fedd_b["text"].split()
    b_step = 3.0 / len(b_words)
    b_boundaries = [
        (i * b_step, (i + 1) * b_step, b_words[i]) for i in range(len(b_words))
    ]
    aligner = FakeForcedAligner(boundaries_by_utt={fedd_b["utt_id"]: b_boundaries})

    fedd_result = evaluate_fedd_dataset(
        fedd_controls, fedd_gen_rows, mapped_eval, aligner=aligner,
    )

    # --- FEDD-B：exact tier，前/后 span + transition direction + 精确边界误差 ---
    b_row = next(r for r in fedd_result["rows"] if r["utt_id"] == fedd_b["utt_id"])
    assert b_row["boundary_evidence_tier"] == "exact"
    b_metrics = b_row["metrics"]
    assert b_metrics["front_span"]["target_emotion"] == fedd_b["emo_from"]
    assert b_metrics["back_span"]["target_emotion"] == fedd_b["emo_to"]
    assert b_metrics["front_span"]["hit"] is True
    assert b_metrics["back_span"]["hit"] is True
    assert b_metrics["transition_direction"] == "correct"
    assert b_metrics["front_back_both_hit"] is True
    # exact-boundary：fake aligner 已知边界 → boundary_error_sec 非 null
    assert b_metrics["boundary_error_sec"] is not None
    assert b_metrics["aligned_boundary_sec"] == pytest.approx(1.5, abs=0.05)
    assert b_metrics["alignment_status"] == "aligned"
    validate_eval_row(b_row)

    # --- FEDD-A：approximate tier，不进 exact-boundary aggregate ---
    a_row = next(r for r in fedd_result["rows"] if r["utt_id"] == fedd_a["utt_id"])
    assert a_row["boundary_evidence_tier"] == "approximate"
    # approximate tier 不计算精确边界误差
    assert a_row["metrics"]["boundary_error_sec"] is None
    assert a_row["metrics"]["alignment_status"] == "not_attempted"
    validate_eval_row(a_row)

    # --- aggregate 按 evidence_tier 分离：FEDD-A 不进 exact aggregate ---
    agg_exact = fedd_result["aggregate_exact"]
    agg_approx = fedd_result["aggregate_approximate"]
    validate_aggregate(agg_exact)
    validate_aggregate(agg_approx)
    assert agg_exact["n_samples"] == 1      # 仅 FEDD-B
    assert agg_approx["n_samples"] == 1     # 仅 FEDD-A
    assert agg_exact["metrics"]["front_back_both_hit_rate"] == 1.0
    assert agg_exact["metrics"]["transition_correct_rate"] == 1.0

    # 写评测 identity（11）：aggregate 身份绑定确定 rows 集合。
    from tools.write_emofilm_run_identity import (
        compute_aggregate_identity,
        verify_aggregate_identity,
        write_emofilm_evaluation_identity,
    )
    fedd_rows = fedd_result["rows"]
    agg_identity = compute_aggregate_identity(fedd_rows)
    ok, reason = verify_aggregate_identity(fedd_rows, agg_identity)
    assert ok, reason
    eval_identity = write_emofilm_evaluation_identity(
        workspace / "fedd_eval_identity.json",
        code_root=workspace,
        command="test_emofilm_local_control_e2e_smoke::fedd_eval",
        generation_identity_ref=workspace / "generation_identity.json",
        n_eval_rows=len(fedd_rows),
        aggregate_identity=agg_identity,
        evaluator_info=emotion_eval.identity(),
    )
    assert eval_identity["schema_version"] == 2
    assert eval_identity["n_eval_rows"] == 2

    # ============================================================
    # 10 ESD utterance-level emotion 评测（单成员 eval row）
    # ============================================================
    arousal_eval = FakeAcousticEvaluator(kind="arousal")
    emotion_eval_esd = FakeAcousticEvaluator(kind="emotion")
    wer_eval = FakeWerEvaluator(hypotheses={esd["utt_id"]: esd["text"]})
    esd_clip = SyntheticReferenceClip(
        wav_path=esd_gen["wav_path"], duration_sec=2.0,
        known_emotion=esd["emotion"], known_arousal_rank=1,
    )
    esd_base_case = {
        "text": esd["text"],
        "emotion": esd["emotion"],
        "speaker": PROMPT_ROW["speaker_id"],
        "prompt_ref": "prompt/speaker_0011",
        "checkpoint_sha256": CKPT_SHA,
        "source_revision": SOURCE_REV,
        "decode_config": dict(DECODE_CONFIG),
    }
    esd_eval_row = build_triplet_member_eval_row(
        esd["utt_id"], esd_base_case, "medium", esd_gen, esd,
        arousal_eval.predict_frames(esd_clip),
        emotion_eval_esd.predict_frames(esd_clip),
        esd["text"],
        arousal_eval.identity(),
        emotion_eval_esd.identity(),
        wer_eval.identity(),
    )
    validate_eval_row(esd_eval_row)
    # ESD：utterance-level emotion 命中（独立 evaluator 预测 == 控制情感）
    assert esd_eval_row["metrics"]["emotion_prediction"] == esd["emotion"]
    assert esd_eval_row["boundary_evidence_tier"] == "exact"  # 整句，无近似边界

    # ============================================================
    # 10 intensity triplet：同一闭环，仅 intensity 不同 → 逐组 + aggregate
    # ============================================================
    triplet_base = {
        **esd_base_case,
        "emotion": "ang",
        "text": "the dog barks loudly at the mail carrier",
    }
    triplet_text = triplet_base["text"]
    triplet_emo = triplet_base["emotion"]

    # 三档控制 span：仅 control_intensity_id 不同（low/medium/high）。
    triplet_spans = {
        tier: {
            "utt_id": f"triplet_{tier}",
            "label_source": "triplet_intensity_sweep",
            "supervision_granularity": "utterance",
            "start_sec": 0.0,
            "end_sec": 2.0,
            "control_emotion_id": EMOTION_NAME_TO_ID[triplet_emo],
            "control_intensity_id": INTENSITY_TIER_TO_ID[tier],
            "calibrated": False,
            "emotion_mask": False,
            "intensity_mask": False,      # 控制 sweep，不声称强度真值
            "supervision_weight": 1.0,
            "provenance": f"triplet-intensity-sweep/{triplet_emo}/{tier}",
            "intensity_policy": f"fixed_{tier}",
            "text": triplet_text,
            "emotion": triplet_emo,
            "speaker": triplet_base["speaker"],
        }
        for tier in INTENSITY_TIERS
    }
    for span in triplet_spans.values():
        validate_span(span)

    # 三档 generation row：经同一推理闭环。
    triplet_gen_rows = []
    triplet_clip_map: dict[str, SyntheticReferenceClip] = {}
    arousal_rank_by_tier = {"low": 0, "medium": 1, "high": 2}
    for tier in INTENSITY_TIERS:
        ctrl = triplet_spans[tier]
        decode_result, tokens = _run_inference(model, ctrl)
        assert decode_result.finish_reason == "eos"
        wav_rel = f"wav/{ctrl['utt_id']}.wav"
        (workspace / wav_rel).write_bytes(b"RIFFfake-triplet-wav")
        gen = _build_generation_row(ctrl, decode_result, PROMPT_ROW, wav_rel_path=wav_rel)
        gen["intensity_tier"] = tier
        gen["group_id"] = "triplet_ang_sweep"
        triplet_gen_rows.append(gen)
        triplet_clip_map[ctrl["utt_id"]] = SyntheticReferenceClip(
            wav_path=wav_rel, duration_sec=2.0,
            known_arousal_rank=arousal_rank_by_tier[tier],
            known_emotion=triplet_emo,
        )

    triplet_spec = {
        "group_id": "triplet_ang_sweep",
        "base_case": triplet_base,
        "control_spans": triplet_spans,
    }
    mapped_arousal = ClipMappedEvaluator(
        FakeAcousticEvaluator(kind="arousal"), triplet_clip_map
    )
    mapped_emotion = ClipMappedEvaluator(
        FakeAcousticEvaluator(kind="emotion"), triplet_clip_map
    )
    triplet_result = evaluate_triplet_dataset(
        [triplet_spec], triplet_gen_rows,
        arousal_eval=mapped_arousal,
        emotion_eval=mapped_emotion,
        wer_eval=FakeWerEvaluator(
            hypotheses={f"triplet_{t}": triplet_text for t in INTENSITY_TIERS}
        ),
    )
    assert len(triplet_result["group_rows"]) == 1
    group = triplet_result["group_rows"][0]
    assert group["valid"] is True
    # low < med < high arousal 单调（fake evaluator 已知 rank 0/1/2）
    assert group["metrics"]["arousal_monotonic"] is True
    assert group["metrics"]["arousal_strict_monotonic"] is True
    assert group["metrics"]["emotion_preserved"] is True
    validate_aggregate(triplet_result["aggregate"])
    assert triplet_result["aggregate"]["metrics"]["monotonicity_rate"] == 1.0
    assert triplet_result["aggregate"]["metrics"]["n_valid_groups"] == 1

    # ============================================================
    # determinism：所有 aggregate 能从持久化 rows 重算
    # ============================================================
    agg_exact_recompute = build_aggregate_from_rows(fedd_rows, "exact")
    agg_approx_recompute = build_aggregate_from_rows(fedd_rows, "approximate")
    assert agg_exact_recompute == agg_exact
    assert agg_approx_recompute == agg_approx

    triplet_agg_recompute = build_triplet_aggregate(triplet_result["group_rows"])
    assert triplet_agg_recompute == triplet_result["aggregate"]

    # aggregate 身份在重算后仍一致（rows 未被替换 / 遗漏 / 混入）
    ok2, reason2 = verify_aggregate_identity(fedd_rows, agg_identity)
    assert ok2, reason2


# ============================================================
# B. 非 EOS 注入 → smoke 失败（hard-fail，不静默）
# ============================================================


def test_non_eos_finish_reason_no_wav_and_eval_hard_fails(tmp_path):
    """注入一个非 EOS 结果：不写 WAV；generation row 不得携 wav_path；
    评测入口对非 EOS 行 hard-fail（不静默跳过算均值）。"""
    model = _make_model(_sampler_always_valid())  # 永不发 EOS → max_len_reached
    ctrl = _fedd_b_control_row()

    decode_result, tokens = _run_inference(model, ctrl)
    assert decode_result.finish_reason == "max_len_reached"
    # 非 EOS → inference 不向声学侧产出任何 token（05 门控）
    assert tokens == []

    # generation row 不得携 wav_path（validate_generation_row 强制）
    row = _build_generation_row(ctrl, decode_result, PROMPT_ROW, wav_rel_path=None)
    assert "wav_path" not in row
    assert row["finish_reason"] == "max_len_reached"

    # 写不出 WAV（只有 EOS 落正式 WAV）
    assert not (tmp_path / "wav" / f"{ctrl['utt_id']}.wav").exists()

    # 评测入口对非 EOS 行 hard-fail（携 utt_id，不静默跳过）
    emotion_eval = FakeAcousticEvaluator(kind="emotion")
    with pytest.raises(RuntimeError, match="hard-fail"):
        evaluate_fedd_dataset(
            [ctrl], [row],
            ClipMappedEvaluator(emotion_eval, {}),
            aligner=FakeForcedAligner(),
        )

    # skip-existing 也拒绝非 EOS 既有行（11）（非 EOS 无 wav_path，
    # 在 isfile 检查前即返回；workspace_root 仅保持调用一致性）
    decision = check_skip_existing(
        row, _request_fingerprint(ctrl, PROMPT_ROW), workspace_root=tmp_path,
    )
    assert decision.skip is False
    assert "eos" in decision.reason.lower()


def test_input_rejected_finish_reason_no_wav():
    """hard cap 不足 → input_rejected；不进采样；不写 WAV。"""
    model = _make_model(_sampler_eos_after(min_len=DECODE_MIN_LEN))
    ctrl = _fedd_b_control_row()
    # max_len_hard_cap=1 < min_len=int(3*2)=6 → max_len <= min_len → input_rejected
    decode_result = model.decode(
        text_token=torch.tensor([[2, 3, 4]]),
        text_len=torch.tensor([3], dtype=torch.int32),
        emotion_ids=torch.full((1, 3), ctrl["control_emotion_id"], dtype=torch.long),
        intensity_ids=torch.full((1, 3), ctrl["control_intensity_id"], dtype=torch.long),
        max_token_text_ratio=20,
        min_token_text_ratio=2,
        max_len_hard_cap=1,
    )
    assert decode_result.finish_reason == "input_rejected"
    assert decode_result.tokens == []
    row = _build_generation_row(ctrl, decode_result, PROMPT_ROW, wav_rel_path=None)
    assert "wav_path" not in row


# ============================================================
# C. 身份不一致注入 → check_skip_existing 拒绝（不静默复用）
# ============================================================


def test_identity_mismatch_rejected_by_check_skip_existing(tmp_path):
    """既有 EOS 行与请求身份不一致（checkpoint 换）→ 拒绝复用（v1 会静默复用）。"""
    workspace = tmp_path
    ctrl = _fedd_b_control_row()
    decode_result = DecodeResult(
        tokens=[2, 2, 2, 2, 2, 2], finish_reason="eos",
        min_len=DECODE_MIN_LEN, max_len=60,
        num_valid_speech_tokens=6, invalid_token_retries=0, text_len=TEXT_LEN,
    )
    gen_row = _build_generation_row(
        ctrl, decode_result, PROMPT_ROW, wav_rel_path="wav/fedd-b-0001.wav",
    )

    # 创建真实 WAV 文件（workspace-relative），让 isfile 检查通过 → 走到身份比对。
    wav_rel = gen_row["wav_path"]
    (workspace / wav_rel).parent.mkdir(parents=True, exist_ok=True)
    (workspace / wav_rel).write_bytes(b"RIFFfake-wav-mismatch-case")

    # 请求使用**不同**的 checkpoint sha256。
    mismatched_fp = generation_request_fingerprint(
        source=SOURCE_REV,
        checkpoint_sha256="ef" * 32,           # 不同的 checkpoint
        control_row=dict(ctrl),
        prompt_row=dict(PROMPT_ROW),
        decode_config=DECODE_CONFIG,
        seed=SEED,
    )
    decision = check_skip_existing(
        gen_row, mismatched_fp, workspace_root=workspace,
    )
    assert decision.skip is False
    assert "mismatch" in decision.reason.lower()

    # 一致的请求 → 安全复用（对照）。
    ok_fp = _request_fingerprint(ctrl, PROMPT_ROW)
    ok_decision = check_skip_existing(gen_row, ok_fp, workspace_root=workspace)
    assert ok_decision.skip is True


def test_control_row_drift_rejected_by_check_skip_existing(tmp_path):
    """控制 row 漂移（情感控制值换）→ 指纹不一致 → 拒绝复用。"""
    workspace = tmp_path
    ctrl = _fedd_b_control_row()
    decode_result = DecodeResult(
        tokens=[2, 2, 2, 2, 2, 2], finish_reason="eos",
        min_len=DECODE_MIN_LEN, max_len=60,
        num_valid_speech_tokens=6, invalid_token_retries=0, text_len=TEXT_LEN,
    )
    gen_row = _build_generation_row(
        ctrl, decode_result, PROMPT_ROW, wav_rel_path="wav/fedd-b-0001.wav",
    )

    # 创建真实 WAV 文件（workspace-relative），让 isfile 检查通过 → 走到身份比对。
    wav_rel = gen_row["wav_path"]
    (workspace / wav_rel).parent.mkdir(parents=True, exist_ok=True)
    (workspace / wav_rel).write_bytes(b"RIFFfake-wav-drift-case")

    drifted_ctrl = dict(ctrl)
    drifted_ctrl["control_emotion_id"] = 5  # sur ≠ 原 neu(4) → 不同控制输入

    drifted_fp = generation_request_fingerprint(
        source=SOURCE_REV,
        checkpoint_sha256=CKPT_SHA,
        control_row=drifted_ctrl,
        prompt_row=dict(PROMPT_ROW),
        decode_config=DECODE_CONFIG,
        seed=SEED,
    )
    decision = check_skip_existing(
        gen_row, drifted_fp, workspace_root=workspace,
    )
    assert decision.skip is False
    assert "mismatch" in decision.reason.lower()


# ============================================================
# D. 聚合身份可检测 rows 被替换 / 遗漏（11）
# ============================================================


def test_aggregate_identity_detects_row_replacement():
    """aggregate 身份能检测 eval rows 被替换 / 遗漏 / 混入其他运行。"""
    from tools.write_emofilm_run_identity import (
        compute_aggregate_identity,
        verify_aggregate_identity,
    )

    # 复用 FEDD-A / FEDD-B 闭环产出的两条 eval row（走完整 evaluate_fedd_dataset）。
    fedd_a = _fedd_a_control_row()
    fedd_b = _fedd_b_control_row()
    model = _make_model(_sampler_eos_after(min_len=DECODE_MIN_LEN))

    gen_rows = []
    for ctrl in (fedd_a, fedd_b):
        decode_result, _ = _run_inference(model, ctrl)
        gen_rows.append(_build_generation_row(
            ctrl, decode_result, PROMPT_ROW,
            wav_rel_path=f"wav/{ctrl['utt_id']}.wav",
        ))

    emotion_eval = FakeAcousticEvaluator(kind="emotion")
    clip_map = {
        fedd_a["utt_id"]: SyntheticReferenceClip(
            wav_path=gen_rows[0]["wav_path"], duration_sec=3.0,
            known_transition_sec=1.5,
            known_transition_from=fedd_a["emo_from"], known_transition_to=fedd_a["emo_to"],
        ),
        fedd_b["utt_id"]: SyntheticReferenceClip(
            wav_path=gen_rows[1]["wav_path"], duration_sec=3.0,
            known_transition_sec=1.5,
            known_transition_from=fedd_b["emo_from"], known_transition_to=fedd_b["emo_to"],
        ),
    }
    result = evaluate_fedd_dataset(
        [fedd_a, fedd_b], gen_rows,
        ClipMappedEvaluator(emotion_eval, clip_map),
        aligner=FakeForcedAligner(),
    )
    rows = result["rows"]
    identity = compute_aggregate_identity(rows)

    # 原始 rows → 一致。
    ok, _ = verify_aggregate_identity(rows, identity)
    assert ok

    # 遗漏一条 → 不一致。
    ok_missing, _ = verify_aggregate_identity(rows[:1], identity)
    assert not ok_missing

    # 替换一条（改动 metrics）→ 不一致。
    tampered = [dict(r) for r in rows]
    tampered[0] = dict(tampered[0])
    tampered[0]["metrics"] = {**tampered[0]["metrics"], "injected": True}
    ok_tampered, _ = verify_aggregate_identity(tampered, identity)
    assert not ok_tampered
