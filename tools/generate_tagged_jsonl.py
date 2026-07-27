#!/usr/bin/env python3
"""EmoFiLM 监督 manifest 生成器 —— 保留 IEMOCAP 弱监督分布、VAD 与校准状态。

本模块是 EmoFiLM 主线数据准备的唯一入口（ADR-0020 扁平化后；原 v2 修复
替代了 v1 argmax 词级标注器，后者仅存于 git 基线 ``9c6d84b``）。它**不丢
信息**：把标注器 ``predict_words`` 的 soft distribution / 全 VAD / 连续
arousal / raw score / 校准状态 / 词边界完整保留到 SupervisionSpan（通过
``build_emofilm_contract.validate_span``）。

设计要点（MAP.md §2、§3；ADR-0019；ADR-0020）：
- **Predictor-agnostic（依赖注入）**：核心合并/构造逻辑是纯函数，不绑定具体
  标注器；测试注入 ``_FakePredictor``（无 GPU/重模型），CLI 用真实标注器。
- **只读导入数据流水线**：``build_emofilm_contract.py`` 的 ``load_word_sequence_model``
  / ``classify_text_coverage`` 等数据流水线函数仍被本文件只读引用。
- **诚实表达**：ESD/FEDD 无 VAD/arousal/model score → 不得伪造（schema 条件必需）；
  IEMOCAP 词级标签明确标记为句级广播弱监督（``weak_supervision="sentence_broadcast"``）。
- **raw_score 定义**：= max softmax 概率（5 类中最大者），明确不是校准 confidence；
  未校准（``calibrated=False``）时不得在任何字段名里出现 ``confidence``。
- **相邻词合并规则**：仅当 ``(control_emotion_id, control_intensity_id,
  calibrated, label_source)`` 兼容时合并；``calibrated`` 不一致 → raise
  ValueError 携 ``utt_id``（Grilling 决策 #6）。合并保留成员词分布/边界
  （``member_words``）以便溯源（schema §1 合并规则）。

范围限制：本模块交付**生成器代码 + 小样本产物**；真实全量 20774+1092 重建属
后续数据运行（非代码 DoD）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# WordSequenceModel 内部标签顺序：[ang, hap, neu, sad, sur]
# 与 ``cosyvoice/tokenizer/emo_tokenizer.py`` 的 EMOTIONS 列表**同序**（MAP §5）。
# 即：soft_dist[0]=ang, soft_dist[1]=hap, ..., soft_dist[4]=sur。
# tokenizer emotion_to_id 把同序列映射为 1..5：ang=1, hap=2, neu=3, sad=4, sur=5。
WORDSEQ_EMOTION_ORDER: tuple[str, ...] = ("ang", "hap", "neu", "sad", "sur")

# 控制 id 空间（与 ``emo_tokenizer.py`` 一致；见 schema §标签 id 空间）。
EMOTION_TO_CONTROL_ID: dict[str, int] = {
    emo: idx + 1 for idx, emo in enumerate(WORDSEQ_EMOTION_ORDER)
}
INTENSITY_TO_CONTROL_ID: dict[str, int] = {"low": 1, "medium": 2, "high": 3}

# arousal → 离散强度控制输入的阈值（仅作**控制接口**用，不作强度真值；
# 连续 arousal 仍是监督 target）。
_AROUSAL_HIGH_THRESHOLD = 3.5
_AROUSAL_MEDIUM_THRESHOLD = 2.5

# IEMOCAP 弱监督来源戳记（schema §1 典型值）。
IEMOCAP_LABEL_SOURCE = "word_annotator_pseudo_label"
WEAK_SUPERVISION_TAG = "sentence_broadcast"

# ESD 来源戳记。
ESD_LABEL_SOURCE = "esd_fixed_medium_control"
ESD_DATASET_LABEL = "dataset_global_label"


# ============================================================
# Predictor 协议（依赖注入；测试与 CLI 共用接口）
# ============================================================


class Predictor(Protocol):
    """Per-word-block predictor protocol.

    实现类必须能对单个 word_block ``.pt`` 文件运行推理并返回保留全部输出的 dict。

    可选输出键 ``calibrated``（bool，缺省 False）与 ``calibration``
    （``{method, version, ...}`` 或 None，缺省 None）：透传到 SupervisionSpan，
    使校准链不断裂（brief 07 / Task 7）。``calibrated=True`` 时 ``calibration``
    必须为非空 dict（合同 validator ``validate_span`` 会校验完整字段）。
    """

    def predict_word(self, word_block_path: Path) -> dict[str, Any]: ...


# ============================================================
# 控制接口工具（纯函数）
# ============================================================


def intensity_from_arousal(arousal: float) -> str:
    """arousal → 离散强度控制输入的阈值映射（仅作离散控制输入用，不作强度真值）。

    连续 arousal 本身作为监督 target 由 ``intensity_mask=True`` 门控。
    """
    if arousal > _AROUSAL_HIGH_THRESHOLD:
        return "high"
    if arousal > _AROUSAL_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _softmax(logits: list[float]) -> list[float]:
    """数值稳定 softmax（std-only；不引入 numpy）。"""
    if not logits:
        raise ValueError("softmax requires non-empty input")
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ============================================================
# Per-word 预测 → SupervisionSpan 合并（纯函数；Predictor-agnostic）
# ============================================================


def predict_words_full(
    predictor: Predictor,
    word_files: Iterable[str],
    utt_dir: Path,
) -> list[dict[str, Any]]:
    """对每个 word_block ``.pt`` 运行 ``predictor``，返回保留全部输出的 list。

    单调透传 predictor 输出；不 argmax、不分桶、不丢字段。word_files 顺序
    决定成员词顺序（sorted ``*.pt`` 文件名）。
    """
    results: list[dict[str, Any]] = []
    for word_file in word_files:
        word_path = utt_dir / word_file
        pred = predictor.predict_word(word_path)
        results.append(pred)
    return results


def _build_member_record(pred: Mapping[str, Any]) -> dict[str, Any]:
    """从单条 predictor 输出构造可溯源的成员词记录（合并后保留）。"""
    member: dict[str, Any] = {
        "word": str(pred.get("word", "")),
        "start_sec": float(pred["start_sec"]),
        "end_sec": float(pred["end_sec"]),
        "start_frame": int(pred["start_frame"]),
        "end_frame": int(pred["end_frame"]),
        "frame_rate_hz": float(pred["frame_rate_hz"]),
        "emotion_soft_distribution": [float(p) for p in pred["emotion_soft_distribution"]],
        "arousal": float(pred["arousal"]),
        "raw_score": float(pred["raw_score"]),
        # 校准状态透传（brief 07 / Task 7）：predictor 未提供时默认未校准。
        "calibrated": bool(pred.get("calibrated", False)),
    }
    # VAD 可选（checkpoint 可能只输出 1D arousal）
    vad = pred.get("vad")
    if vad is not None:
        member["vad"] = [float(v) for v in vad]
    # calibration 可选：calibrated=False 时应缺省；predictor 显式 None / 未提供 → None。
    calibration = pred.get("calibration")
    if calibration is not None:
        member["calibration"] = calibration
    else:
        member["calibration"] = None
    return member


def _span_from_members(
    utt_id: str,
    members: list[dict[str, Any]],
    *,
    sentence_emotion: Optional[str],
    sentence_vad: Optional[list[float]],
    annotator_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """从一组合并成员词构造一条 SupervisionSpan（含可溯源 provenance）。"""
    if not members:
        raise ValueError("cannot build span from empty members")

    first = members[0]
    last = members[-1]
    n = len(members)

    # 控制值由首成员决定（兼容键已保证所有成员 control_emotion_id/control_intensity_id 一致）。
    pred_idx = max(range(5), key=lambda i: first["emotion_soft_distribution"][i])
    control_emotion_id = pred_idx + 1  # ang=1..sur=5（WORDSEQ_EMOTION_ORDER 同序）
    control_intensity_id = INTENSITY_TO_CONTROL_ID[
        intensity_from_arousal(float(first["arousal"]))
    ]

    # Span-level soft distribution = 成员 soft dist 均值（保留分布，不 argmax）。
    avg_soft = [
        sum(m["emotion_soft_distribution"][i] for m in members) / n for i in range(5)
    ]
    # 归一化（理论上输入概率和为 1 → 均值仍和为 1；FP 安全起见显式归一）。
    soft_sum = sum(avg_soft)
    if soft_sum > 0:
        avg_soft = [p / soft_sum for p in avg_soft]

    # Span-level arousal = 成员 arousal 均值（连续，不分桶）。
    span_arousal = sum(float(m["arousal"]) for m in members) / n

    # Span-level VAD：仅当所有成员都有 VAD 时取均值（不伪造部分缺失）。
    span_vad: Optional[list[float]] = None
    if all(m.get("vad") is not None for m in members):
        span_vad = [
            sum(float(m["vad"][i]) for m in members) / n for i in range(3)
        ]

    # raw_score = 成员 max-softmax 均值（明确文档：未校准，非 confidence）。
    span_raw_score = sum(float(m["raw_score"]) for m in members) / n

    span: dict[str, Any] = {
        "utt_id": utt_id,
        "label_source": IEMOCAP_LABEL_SOURCE,
        "supervision_granularity": "word",
        "start_sec": float(first["start_sec"]),
        "end_sec": float(last["end_sec"]),
        # 供 span→token 对齐消费的 frame 边界（brief 02 §A）：
        "start_frame": int(first["start_frame"]),
        "end_frame": int(last["end_frame"]),
        "frame_rate_hz": float(first["frame_rate_hz"]),
        "emotion_soft_distribution": [float(p) for p in avg_soft],
        "arousal": float(span_arousal),
        "control_emotion_id": int(control_emotion_id),
        "control_intensity_id": int(control_intensity_id),
        "raw_score": float(span_raw_score),
        # 校准状态从首成员透传（合并已保证一致，否则 raise；brief 07 / Task 7）。
        "calibrated": bool(first["calibrated"]),
        "calibration": first.get("calibration"),
        "emotion_mask": True,  # 有 soft dist 监督
        "intensity_mask": True,  # 有连续 arousal 监督 target
        "supervision_weight": 1.0,
        "intensity_policy": "predicted_arousal",
        "provenance": {
            "label_source": IEMOCAP_LABEL_SOURCE,
            "weak_supervision": WEAK_SUPERVISION_TAG,  # 句级广播来源（非词级真值）
            "sentence_emotion": sentence_emotion,
            "annotator": dict(annotator_provenance),
            "member_words": [
                {k: v for k, v in m.items()} for m in members
            ],
        },
    }
    if sentence_vad is not None:
        span["provenance"]["sentence_vad"] = [float(v) for v in sentence_vad]
    if span_vad is not None:
        span["vad"] = [float(v) for v in span_vad]
    return span


def merge_word_predictions_to_v2_spans(
    *,
    utt_id: str,
    word_preds: list[Mapping[str, Any]],
    sentence_emotion: Optional[str],
    sentence_vad: Optional[list[float]],
    annotator_provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """合并相邻兼容词为 SupervisionSpan；不兼容则保留独立 span。

    兼容键：``(control_emotion_id, control_intensity_id, calibrated, label_source)``
    （brief 02 §A）。``label_source`` 对所有 IEMOCAP 词恒定，故有效键等价于
    ``(control_emotion_id, control_intensity_id, calibrated)``。合并后保留
    成员词的 per-word 分布/边界（``provenance.member_words``）以支持溯源
    （schema §1 合并规则）。

    Grilling 决策 #6：``calibrated`` 不一致（同 span 内有成员校准、有成员未校准）
    → raise ValueError 携 ``utt_id``。否则 span 只能写一个 calibrated 状态，
    会丢失另一部分成员的校准信息（信息不丢是本模块核心约束）。
    """
    if not word_preds:
        return []

    spans: list[dict[str, Any]] = []
    current_members: list[dict[str, Any]] = []
    current_key: Optional[tuple[int, int]] = None
    current_calibrated: Optional[bool] = None

    for pred in word_preds:
        member = _build_member_record(pred)
        soft = member["emotion_soft_distribution"]
        if len(soft) != 5:
            raise ValueError(
                f"emotion_soft_distribution must have length 5; got {len(soft)} "
                f"for word {member['word']!r}"
            )
        pred_idx = max(range(5), key=lambda i: soft[i])
        control_emotion_id = pred_idx + 1
        control_intensity_id = INTENSITY_TO_CONTROL_ID[
            intensity_from_arousal(float(member["arousal"]))
        ]
        key = (control_emotion_id, control_intensity_id)
        calibrated = bool(member["calibrated"])

        if current_key is None:
            current_key = key
            current_calibrated = calibrated
            current_members = [member]
        elif key == current_key:
            # calibrated 不一致 → raise（决策 #6）：合并会丢校准状态，必须分桶。
            if calibrated != current_calibrated:
                raise ValueError(
                    f"utt_id={utt_id!r}: cannot merge word {member['word']!r} "
                    f"(calibrated={calibrated}) with prior members "
                    f"(calibrated={current_calibrated}) under same "
                    f"(control_emotion_id={control_emotion_id}, "
                    f"control_intensity_id={control_intensity_id}); "
                    f"calibration state must be consistent within a span"
                )
            current_members.append(member)
        else:
            spans.append(
                _span_from_members(
                    utt_id,
                    current_members,
                    sentence_emotion=sentence_emotion,
                    sentence_vad=sentence_vad,
                    annotator_provenance=annotator_provenance,
                )
            )
            current_key = key
            current_calibrated = calibrated
            current_members = [member]

    if current_members:
        spans.append(
            _span_from_members(
                utt_id,
                current_members,
                sentence_emotion=sentence_emotion,
                sentence_vad=sentence_vad,
                annotator_provenance=annotator_provenance,
            )
        )
    return spans


# ============================================================
# ESD utterance-level span（无 predictor；dataset 全局标签）
# ============================================================


def build_esd_utterance_span(
    *,
    sample: Mapping[str, Any],
    utterance_duration_sec: float,
    intensity: str = "medium",
) -> dict[str, Any]:
    """构造 ESD utterance-level SupervisionSpan。

    ESD 由数据集提供 sentence_emotion（硬标签），无词级模型预测。``fixed_*``
    强度仅作控制输入 → ``intensity_mask=False``（schema §1：无强度真值）。
    one-hot soft dist 是硬标签的诚实表示（schema §1 允许）。**不伪造** VAD /
    arousal / raw_score / calibration。
    """
    emo = str(sample.get("sentence_emotion", "")).strip()
    if emo not in EMOTION_TO_CONTROL_ID:
        raise ValueError(
            f"ESD sample {sample.get('utt_id')!r} sentence_emotion={emo!r} "
            f"not in {list(EMOTION_TO_CONTROL_ID.keys())}; "
            "ESD label must come from dataset global label (ADR-0003)."
        )
    if intensity not in INTENSITY_TO_CONTROL_ID:
        raise ValueError(
            f"intensity {intensity!r} not in {list(INTENSITY_TO_CONTROL_ID.keys())}"
        )
    if utterance_duration_sec <= 0.0:
        raise ValueError(
            f"utterance_duration_sec must be > 0 (start_sec < end_sec); "
            f"got {utterance_duration_sec}"
        )

    control_emotion_id = EMOTION_TO_CONTROL_ID[emo]
    control_intensity_id = INTENSITY_TO_CONTROL_ID[intensity]
    # one-hot soft distribution（硬标签的诚实表示）。
    one_hot = [0.0] * 5
    one_hot[control_emotion_id - 1] = 1.0

    utt_id = str(sample.get("utt_id", "")).strip()

    return {
        "utt_id": utt_id,
        "label_source": ESD_LABEL_SOURCE,
        "supervision_granularity": "utterance",
        "start_sec": 0.0,
        "end_sec": float(utterance_duration_sec),
        # frame 边界由 span-align 从 speech-token 对齐导出，此处不伪造。
        "emotion_soft_distribution": one_hot,
        # arousal / vad / raw_score 故意缺省：ESD 无词级模型分数（schema 条件必需）。
        "control_emotion_id": int(control_emotion_id),
        "control_intensity_id": int(control_intensity_id),
        "calibrated": False,
        "emotion_mask": True,
        "intensity_mask": False,  # fixed_medium 仅控制输入；无强度监督 target。
        "supervision_weight": 1.0,
        "intensity_policy": f"fixed_{intensity}",
        "provenance": {
            "label_source": ESD_DATASET_LABEL,
            "dataset": "esd",
            "sentence_emotion": emo,
            "speaker_id": sample.get("speaker_id", ""),
            "method": "dataset_global_label",
        },
    }


# ============================================================
# TextGrid xmax 抽取（ESD utterance 时长来源）
# ============================================================


_TEXTGRID_XMAX_RE = re.compile(r"^\s*xmax\s*=\s*([0-9.]+)\s*$", re.MULTILINE)


def extract_textgrid_xmax(textgrid_path: Path) -> Optional[float]:
    """从 Praat TextGrid 文件头抽取顶层 ``xmax``（utterance 总时长秒）。

    TextGrid 第一段元数据 ``xmin = 0 \\n xmax = <dur>`` 是 utterance 总时长。
    用于 ESD span 的 ``end_sec``（start=0.0）。解析失败返回 None。
    """
    try:
        text = textgrid_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = _TEXTGRID_XMAX_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


# ============================================================
# Default predictor：WordSequenceModel（768d/5emo/3VAD 合同）
# ============================================================


def _file_sha256(path: Path) -> str:
    """流式 sha256（适配大 checkpoint）。"""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WordSequenceModelPredictor:
    """默认 predictor：调用 ``load_word_sequence_model`` 加载 768d/3VAD 合同 checkpoint。

    本类调用 ``WordSequenceModel.forward``，保留完整 ``(class_logits, vad_pred)``
    输出：softmax → soft dist；``vad_pred*4+1`` → [1,5] VAD；``raw_score = max(softmax)``。
    """

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        # 只读导入数据流水线函数。
        from tools.build_emofilm_contract import load_word_sequence_model
        import torch  # 延迟导入：合同测试无需 torch

        self._torch = torch
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = _file_sha256(checkpoint)
        self.model = load_word_sequence_model(checkpoint, device=device)

    def predict_word(self, word_block_path: Path) -> dict[str, Any]:
        torch = self._torch
        data = torch.load(
            word_block_path,
            map_location=self.device,
            weights_only=True,
        )
        frames = data["frames"].unsqueeze(0).float().to(self.device)
        padding_mask = torch.zeros(
            1,
            frames.shape[1],
            dtype=torch.bool,
            device=self.device,
        )
        with torch.no_grad():
            class_logits, vad_pred = self.model(frames, padding_mask=padding_mask)

        soft = torch.softmax(class_logits, dim=1).squeeze(0).cpu().tolist()
        vad_scaled = (vad_pred.squeeze(0).cpu() * 4.0 + 1.0).tolist()
        return {
            "word": str(data["word"]),
            "start_sec": float(data["start_sec"]),
            "end_sec": float(data["end_sec"]),
            "start_frame": int(data["start_frame"]),
            "end_frame": int(data["end_frame"]),
            "frame_rate_hz": float(data["frame_rate_hz"]),
            "emotion_soft_distribution": [float(p) for p in soft],
            "vad": [float(v) for v in vad_scaled],
            "arousal": float(vad_scaled[1]),
            "raw_score": float(max(soft)),  # max softmax 概率（明确非 confidence）
        }


class FlexWordSequenceModelPredictor:
    """形状自适应 predictor：用于本地 checkpoint 与合同维度不一致时的实跑 fallback。

    背景（``docs/reports/2026-07-17-emofilm-global-vs-word-annotation.md``）：
    本地 ``checkpoints/word_sequence_model/best.pt`` 实测为 1024d/5emo/1arousal
    （regression_head 输出 1 维），与 ``WordSequenceModel`` 合同要求的
    768d/5emo/3VAD **不兼容**（``load_word_sequence_model`` strict load 会失败）。
    本类用形状自适应的方式构造等价结构加载 checkpoint，用于小样本产物生成；
    **不输出 VAD**（checkpoint 只产 1D arousal），诚实地缺省 schema 可选字段 ``vad``。

    本类**不是模型实现**（不参与训练；frozen 推理 helper）。
    """

    def __init__(self, checkpoint: Path, device: str = "cpu"):
        import torch
        import torch.nn as nn

        self._torch = torch
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = _file_sha256(checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=True)

        # 从 state_dict 推断架构维度（不写死）。
        cls_w = state["classification_head.weight"]
        input_dim = int(cls_w.shape[1])
        num_classes = int(cls_w.shape[0])
        reg_dim = int(state["regression_head.0.weight"].shape[0])
        ffn_hidden = int(state["ffn.0.weight"].shape[0])
        # MultiheadAttention num_heads 无法从 state_dict 推断；8 是合同默认。
        num_heads = 8 if input_dim % 8 == 0 else 1

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.reg_dim = reg_dim

        class _FlexModel(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.attention = nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    dropout=0.0,  # 推理模式
                    batch_first=True,
                )
                self_inner.norm1 = nn.LayerNorm(input_dim)
                self_inner.norm2 = nn.LayerNorm(input_dim)
                self_inner.ffn = nn.Sequential(
                    nn.Linear(input_dim, ffn_hidden),
                    nn.GELU(),
                    nn.Linear(ffn_hidden, input_dim),
                )
                self_inner.classification_head = nn.Linear(input_dim, num_classes)
                self_inner.regression_head = nn.Sequential(
                    nn.Linear(input_dim, reg_dim),
                    nn.Sigmoid(),
                )

            def forward(self_inner, x, padding_mask=None):
                attn_out, _ = self_inner.attention(x, x, x, key_padding_mask=padding_mask)
                x = self_inner.norm1(x + attn_out)
                x = self_inner.norm2(x + self_inner.ffn(x))
                if padding_mask is None:
                    pooled = x.mean(dim=1)
                else:
                    valid = ~padding_mask
                    x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
                    pooled = x.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
                return self_inner.classification_head(pooled), self_inner.regression_head(pooled)

        self.model = _FlexModel().to(device).eval()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # ffn dropout 层在 state_dict 里没有对应键（ffn 含 Dropout）—— 允许。
        # 其他缺失/多余键应触发告警但继续（fallback 容错）。
        self._load_missing = missing
        self._load_unexpected = unexpected

    def predict_word(self, word_block_path: Path) -> dict[str, Any]:
        torch = self._torch
        data = torch.load(
            word_block_path,
            map_location=self.device,
            weights_only=True,
        )
        frames = data["frames"].unsqueeze(0).float().to(self.device)
        padding_mask = torch.zeros(
            1,
            frames.shape[1],
            dtype=torch.bool,
            device=self.device,
        )
        with torch.no_grad():
            class_logits, reg_pred = self.model(frames, padding_mask=padding_mask)

        soft = torch.softmax(class_logits, dim=1).squeeze(0).cpu().tolist()
        reg_scaled = (reg_pred.squeeze(0).cpu() * 4.0 + 1.0).tolist()
        # reg_dim=1 → 只有 arousal；reg_dim=3 → [v,a,d]。
        out: dict[str, Any] = {
            "word": str(data["word"]),
            "start_sec": float(data["start_sec"]),
            "end_sec": float(data["end_sec"]),
            "start_frame": int(data["start_frame"]),
            "end_frame": int(data["end_frame"]),
            "frame_rate_hz": float(data["frame_rate_hz"]),
            "emotion_soft_distribution": [float(p) for p in soft],
            "arousal": float(reg_scaled[0]) if self.reg_dim == 1 else float(reg_scaled[1]),
            "raw_score": float(max(soft)),
        }
        if self.reg_dim >= 3:
            out["vad"] = [float(v) for v in reg_scaled[:3]]
        return out


def load_default_predictor(
    checkpoint: Path,
    device: str = "cpu",
    *,
    allow_flexible_fallback: bool = True,
) -> Predictor:
    """优先用 ``load_word_sequence_model``（768d/3VAD 合同路径）；
    失败时回退到 ``FlexWordSequenceModelPredictor``（本地 1024d/1arousal checkpoint）。

    回退会打印告警并记录缺失/意外键，便于审计。
    """
    try:
        return WordSequenceModelPredictor(checkpoint, device=device)
    except Exception as exc:  # noqa: BLE001 — 合同 strict load 可能抛多种异常
        if not allow_flexible_fallback:
            raise
        print(
            f"[generate_tagged_jsonl] WARNING: load_word_sequence_model failed "
            f"({type(exc).__name__}: {exc}); falling back to FlexWordSequenceModelPredictor "
            f"(checkpoint shape may not match the 768d/5emo/3VAD contract).",
            file=sys.stderr,
        )
        flex = FlexWordSequenceModelPredictor(checkpoint, device=device)
        if flex._load_missing:
            print(
                f"[generate_tagged_jsonl] flex load missing keys: {flex._load_missing}",
                file=sys.stderr,
            )
        if flex._load_unexpected:
            print(
                f"[generate_tagged_jsonl] flex load unexpected keys: {flex._load_unexpected}",
                file=sys.stderr,
            )
        return flex


class StubPredictor:
    """Schema 演示用 predictor：词边界透传（真），分布用确定性占位（伪）。

    背景：本地 ``checkpoints/word_sequence_model/best.pt`` 实测为 1024d/1arousal
    （emotion2vec_plus_large 特征），与既有 768d word_blocks（emotion2vec_base 特征）
    **维度不兼容**，无法实跑推理（详见
    ``docs/reports/2026-07-17-emofilm-global-vs-word-annotation.md``）。
    本类让小样本产物生成不被此环境问题阻塞：
    - **真实字段**：``word / start_sec / end_sec / start_frame / end_frame /
      frame_rate_hz`` 从 word_block ``.pt`` 透传（与真实推理结果一致）；
    - **占位字段**：``emotion_soft_distribution / vad / arousal / raw_score`` 用
      基于内容哈希的确定性分布生成（**非真实模型输出**）。

    下游**不得**把本类的分布当作训练真值；真实全量重建必须用合同匹配的 checkpoint
    （768d/5emo/3VAD）重跑（后续数据运行）。
    """

    PREDICTOR_CLASS = "StubPredictor_SchemaDemo"

    def __init__(self, *, seed_salt: int = 0x5F1D):
        self.seed_salt = seed_salt
        self.checkpoint_sha256 = "0" * 64  # 占位；StubPredictor 无 checkpoint
        self.checkpoint = Path("<stub>")
        self._torch = None  # 延迟 import

    def _hash_word(self, word_block_path: Path) -> int:
        h = hashlib.sha256()
        h.update(str(word_block_path).encode("utf-8"))
        h.update(self.seed_salt.to_bytes(4, "little"))
        # 取前 8 字节做整数 seed
        return int.from_bytes(h.digest()[:8], "little")

    def _deterministic_soft_dist(self, seed: int) -> list[float]:
        """基于 seed 生成一个合理的、非 uniform 的 5 类概率分布。"""
        import random

        rng = random.Random(seed)
        # 选一个主导类（0..4），其余给小概率
        dominant = rng.randrange(5)
        raw = [rng.uniform(0.01, 0.05) for _ in range(5)]
        raw[dominant] = rng.uniform(0.75, 0.92)
        total = sum(raw)
        return [r / total for r in raw]

    def _deterministic_vad(self, seed: int) -> list[float]:
        """基于 seed 生成 [1,5] 区间的 3 维 VAD。"""
        import random

        rng = random.Random(seed ^ 0x7AD)
        return [round(rng.uniform(1.5, 4.5), 4) for _ in range(3)]

    def predict_word(self, word_block_path: Path) -> dict[str, Any]:
        import torch  # 只为读 .pt；推理本身不需要模型

        data = torch.load(word_block_path, map_location="cpu", weights_only=True)
        seed = self._hash_word(word_block_path)
        soft = self._deterministic_soft_dist(seed)
        vad = self._deterministic_vad(seed)
        return {
            "word": str(data["word"]),
            "start_sec": float(data["start_sec"]),
            "end_sec": float(data["end_sec"]),
            "start_frame": int(data["start_frame"]),
            "end_frame": int(data["end_frame"]),
            "frame_rate_hz": float(data["frame_rate_hz"]),
            "emotion_soft_distribution": soft,
            "vad": vad,
            "arousal": float(vad[1]),
            "raw_score": float(max(soft)),
        }


# ============================================================
# Orchestrators
# ============================================================


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_iemocap_v2_spans(
    *,
    manifest_path: Path,
    word_blocks_dir: Path,
    predictor: Predictor,
    output_jsonl: Path,
    max_utterances: int | None = None,
    require_text_coverage: bool = True,
) -> dict[str, int]:
    """IEMOCAP 监督 span 生成入口。

    对 manifest 每条 utterance：取其 word_blocks 子目录，逐词运行 predictor，
    按兼容键合并为 span。**不重分成员**（沿用 manifest 的冻结成员关系；
    MAP §0 与 brief 02 §A）。
    """
    manifest = _load_jsonl(manifest_path)
    if max_utterances is not None:
        manifest = manifest[:max_utterances]

    from tools.build_emofilm_contract import classify_text_coverage  # 只读

    spans: list[dict[str, Any]] = []
    kept = 0
    skipped = 0
    for sample in manifest:
        utt_id = sample["utt_id"]
        utt_dir = word_blocks_dir / utt_id
        if not utt_dir.is_dir():
            skipped += 1
            continue
        word_files = sorted(path.name for path in utt_dir.glob("*.pt"))
        if not word_files:
            skipped += 1
            continue

        word_preds = predict_words_full(predictor, word_files, utt_dir)

        if require_text_coverage:
            tagged_text = " ".join(str(wp["word"]) for wp in word_preds)
            coverage = classify_text_coverage(
                str(sample.get("plain_text") or sample.get("text", "")),
                tagged_text,
            )
            if coverage["decision"] == "reject":
                skipped += 1
                continue

        utt_spans = merge_word_predictions_to_v2_spans(
            utt_id=utt_id,
            word_preds=word_preds,
            sentence_emotion=sample.get("sentence_emotion"),
            sentence_vad=sample.get("sentence_vad"),
            annotator_provenance={
                "model_class": "WordSequenceModel",
                "checkpoint_sha256": getattr(predictor, "checkpoint_sha256", ""),
                "checkpoint": str(getattr(predictor, "checkpoint", "")),
                "contract": "768d/5emo/3VAD",
                "predictor_class": type(predictor).__name__,
            },
        )
        spans.extend(utt_spans)
        kept += 1

    _write_jsonl(output_jsonl, spans)
    return {"utterances_kept": kept, "utterances_skipped": skipped, "spans": len(spans)}


def generate_esd_v2_spans(
    *,
    manifest_path: Path,
    textgrid_dir: Path | None,
    output_jsonl: Path,
    intensity: str = "medium",
    max_utterances: int | None = None,
    fallback_duration_sec: float = 1.0,
) -> dict[str, int]:
    """ESD 监督 span 生成入口（utterance-level，无 predictor）。

    每条 utterance 一个 one-hot span；``end_sec`` 优先取 MFA TextGrid ``xmax``，
    缺失时用 ``fallback_duration_sec``（仅占位，span-align 会从 speech-token 对齐覆盖）。
    """
    manifest = _load_jsonl(manifest_path)
    if max_utterances is not None:
        manifest = manifest[:max_utterances]

    spans: list[dict[str, Any]] = []
    kept = 0
    skipped = 0
    for sample in manifest:
        utt_id = sample.get("utt_id", "")
        emo = sample.get("sentence_emotion")
        if emo not in EMOTION_TO_CONTROL_ID:
            skipped += 1
            continue

        duration: float | None = None
        if textgrid_dir is not None:
            tg_path = textgrid_dir / f"{utt_id}.TextGrid"
            if tg_path.is_file():
                duration = extract_textgrid_xmax(tg_path)
        if duration is None or duration <= 0.0:
            duration = float(fallback_duration_sec)

        span = build_esd_utterance_span(
            sample=sample,
            utterance_duration_sec=duration,
            intensity=intensity,
        )
        spans.append(span)
        kept += 1

    _write_jsonl(output_jsonl, spans)
    return {"utterances_kept": kept, "utterances_skipped": skipped, "spans": len(spans)}


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate EmoFiLM supervision spans preserving weak-supervision "
            "distributions, VAD, continuous arousal, raw score, calibration state, "
            "and word boundaries."
        )
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    p_iemocap = sub.add_parser("iemocap", help="IEMOCAP word-level weak supervision")
    p_iemocap.add_argument("--manifest", type=Path, required=True)
    p_iemocap.add_argument("--word_blocks_dir", type=Path, required=True)
    p_iemocap.add_argument("--checkpoint", type=Path, required=True)
    p_iemocap.add_argument("--output_jsonl", type=Path, required=True)
    p_iemocap.add_argument("--device", default="cpu")
    p_iemocap.add_argument("--max_utterances", type=int, default=None)
    p_iemocap.add_argument(
        "--no_flexible_fallback",
        action="store_true",
        help="Disable fallback to FlexWordSequenceModelPredictor when the contract loader fails",
    )
    p_iemocap.add_argument(
        "--stub_predictor",
        action="store_true",
        help=(
            "Use StubPredictor (deterministic placeholder distributions; real "
            "boundaries). For schema-demo small-sample artifacts only; downstream "
            "must NOT consume stub distributions as training truth."
        ),
    )

    p_esd = sub.add_parser("esd", help="ESD utterance-level dataset label")
    p_esd.add_argument("--manifest", type=Path, required=True)
    p_esd.add_argument("--textgrid_dir", type=Path, default=None)
    p_esd.add_argument("--output_jsonl", type=Path, required=True)
    p_esd.add_argument("--intensity", default="medium", choices=list(INTENSITY_TO_CONTROL_ID.keys()))
    p_esd.add_argument("--max_utterances", type=int, default=None)
    p_esd.add_argument("--fallback_duration_sec", type=float, default=1.0)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset == "iemocap":
        if args.stub_predictor:
            predictor = StubPredictor()
        else:
            predictor = load_default_predictor(
                args.checkpoint,
                device=args.device,
                allow_flexible_fallback=not args.no_flexible_fallback,
            )
        report = generate_iemocap_v2_spans(
            manifest_path=args.manifest,
            word_blocks_dir=args.word_blocks_dir,
            predictor=predictor,
            output_jsonl=args.output_jsonl,
            max_utterances=args.max_utterances,
        )
    elif args.dataset == "esd":
        report = generate_esd_v2_spans(
            manifest_path=args.manifest,
            textgrid_dir=args.textgrid_dir,
            output_jsonl=args.output_jsonl,
            intensity=args.intensity,
            max_utterances=args.max_utterances,
            fallback_duration_sec=args.fallback_duration_sec,
        )
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
