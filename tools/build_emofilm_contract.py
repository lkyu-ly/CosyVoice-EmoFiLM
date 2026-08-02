#!/usr/bin/env python3
"""EmoFiLM 语义数据合同原语库 —— 单一活跃权威（ADR-0020 扁平化）。

本模块是 EmoFiLM 主线数据合同的唯一权威。它承载两类职责：

1. **合同原语（schema_version=2，合同名 ``emofilm``）**：TypedDict schema +
   手写校验器（``validate_span`` / ``validate_generation_row`` /
   ``validate_contract_config``）。
   所有下游票据（数据生成 / 评测 / 身份）以本文件定义的 schema 与校验器为唯一来源。
   人类可读 schema 单一来源：``docs/contracts/emofilm_v2_schema.md``。

2. **v1 数据流水线函数**：manifest 规整 / parquet 打包 / 词级标注合并 /
   membership 校验 / eval 资产校验等。这些函数构建冻结的 v1 数据产物，
   仍被 v1 数据合同测试与监督 span 生成器引用，故原地保留（重构不留旧遗留
   指的是 v1 合同*身份*代码遗留——``CONTRACT_NAME="emofilm_v1"``——已删除；
   数据流水线基础设施不在 v1 身份校验路径意义上）。

设计要点（MAP.md §3、ADR-0019、ADR-0020）：
- **合同原语 stdlib-only**：core schema 与校验器仅用 stdlib，便于 CPU 合同测试。
  torch / pyarrow / cosyvoice_emo 仅在数据流水线函数内**延迟导入**。
- schema 用 TypedDict（文档 + 可选类型检查）表述；校验以手写函数为唯一权威，
  不引入第三方 jsonschema 依赖。
- **禁止用文件内容哈希标定文件**（ADR-0020）：GenerationRow 的 WAV 内容哈希
  字段已移除；产物身份用 ``wav_path`` + 结构化身份字段。
- v1 不可变性由 git 基线锚点 ``9c6d84b`` 保证，不由工作树哈希保证。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, TypedDict, Union


# ============================================================
# 合同身份与 schema 常量
# ============================================================

CONTRACT_NAME = "emofilm"
SCHEMA_VERSION = 2

# Generation row 的结构化 finish reason；仅 ``eos`` 进声学与正式 WAV（MAP §3）。
FINISH_REASONS = frozenset({
    "eos",
    "max_len_reached",
    "invalid_token_retry_exhausted",
    "sampler_error",
    "input_rejected",
})

# 边界证据等级：FEDD-B=exact（真实 MFA 词边界拼接），FEDD-A=approximate
# （MiMo TTS 无词边界，词数中点两段近似）。approximate 不进 exact aggregate
# （分离由评测侧实现，此处只定义合法 tier 值）。
BOUNDARY_EVIDENCE_TIERS = frozenset({"exact", "approximate"})

# SupervisionSpan 的监督粒度：utterance=句级广播；span=连续段（FEDD 两段）；
# word=逐词（保留分布的弱监督）。
SPAN_GRANULARITIES = frozenset({"utterance", "span", "word"})

# 死配置字段：有效 resolved 配置不得携带（ADR-0019）。
#   - mix_ratio：双流交错文本/语音比例，v2 协议为单流 target-only，已删除。
#   - alpha：v1 conf 中 ``alpha: 0.05  # 配置占位`` 是未参与计算的占位；
#     采样超参（top_p / top_k / RAS alpha）属于逐生成 decode_config，
#     不得出现在共享 resolved 配置顶层。
#   注：``emo_loss_weight`` 曾列死字段（v2 删除输入端 classifier 时加的门禁），
#   现已恢复为可选 input-end 句级监督 CE 权重（``Qwen2LM_Emotion.emo_loss_weight``，
#   作为 ``llm`` 构造参数消费，默认 ``0.0``=关闭），不再属顶层死字段。
DEAD_CONFIG_KEYS = frozenset({"mix_ratio", "alpha"})

# Generation row 身份引用键族（至少各满足其一）。
SOURCE_IDENTITY_KEYS = ("source_revision", "source_patch_bundle", "source_patch_sha256")
CHECKPOINT_IDENTITY_KEYS = ("checkpoint_sha256", "checkpoint_ref")

# SupervisionSpan 无条件必需字段。连续输出（emotion_soft_distribution / vad /
# arousal / raw_score / calibration）**按 mask 与 calibrated 条件必需**
# （见 SPAN_CONDITIONAL_RULES 与 validate_span），避免强迫仅有数据集全局情感标签
# 的 ESD、或仅有构造 emo_from/emo_to 的 FEDD 伪造 VAD / arousal / model score。
SPAN_REQUIRED_FIELDS = (
    "label_source",
    "supervision_granularity",
    "start_sec",
    "end_sec",
    "control_emotion_id",
    "control_intensity_id",
    "calibrated",
    "emotion_mask",
    "intensity_mask",
    "supervision_weight",
    "provenance",
)

# SupervisionSpan 连续输出的条件必需规则（validate_span 实现）：
#   emotion_mask=True   ⇒ emotion_soft_distribution 必需（one-hot 是硬标签的诚实表示）；
#                          emotion_mask=False 时可选/缺省。
#   intensity_mask=True ⇒ arousal 必需（连续强度目标存在）；
#   intensity_mask=False ⇒ arousal 必须缺省（无连续强度目标，ESD/FEDD fixed-*）。
#   calibrated=True     ⇒ calibration 记录（含 method/version + 校准样本集合溯源
#                          calibration_sample_set_ref / n_calibration_samples，票据 05）
#                          AND raw_score 必需；
#   calibrated=False    ⇒ calibration 必须缺省；raw_score 可选（IEMOCAP 未校准
#                          标注器分数可保留；ESD/FEDD 无模型分数则缺省）。
#   vad                 ⇒ 始终可选；存在时必须长度 3 数值。
SPAN_CONDITIONAL_FIELDS = (
    "emotion_soft_distribution",
    "arousal",
    "raw_score",
    "calibration",
    "vad",
)

# 控制标签 id 空间（与 ``cosyvoice/tokenizer/emo_tokenizer.py`` 一致）：
#   emotion_to_id = {emo: i+1} → 1..5（0=pad，不作为控制值）；5 类 emo。
#   intensity_to_id = {intensity: i+1} → 1..3（0=pad）；low/medium/high。
CONTROL_EMOTION_ID_RANGE = (1, 5)
CONTROL_INTENSITY_ID_RANGE = (1, 3)

_SOFT_DIST_SUM_TOL = 1e-6
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
# Windows 盘符绝对路径（C:\ / C:/）。合同路径字段必须 workspace-relative POSIX。
_DRIVE_LETTER_RE = re.compile(r"^[a-zA-Z]:[\\/]")
# 绝对/非 POSIX 路径泄漏：不允许前导 ``/``、前导 ``\``、任何反斜杠、盘符前缀。
_LEADING_SLASH_RE = re.compile(r"^[/\\]")

# v1 词级标注常量（数据流水线函数使用）。
_EMOTION_TAG_RE = re.compile(r"</?emotion\b[^>]*>", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*", re.IGNORECASE)


# ============================================================
# TypedDict schema（文档 / 可选类型检查；校验权威在下方 validator）
# ============================================================


class Calibration(TypedDict, total=False):
    """校准记录：仅当 ``calibrated=True`` 时必需。

    含校准样本集合溯源（票据 05 / 审查 #2 修复）：``calibration_sample_set_ref``
    标识校准所用的样本集（引用或名称），``n_calibration_samples`` 是该样本集
    的样本数。两者使校准 score 的数据范围可审计——此前 calibrated span 缺
    该字段仍能通过校验，导致校准曲线来源不可追溯。
    """

    method: str
    version: str
    # 校准样本集引用/标识（审查 #2）：使校准 score 数据范围可审计。
    calibration_sample_set_ref: str
    # 校准样本数（正整数）：与 calibration_sample_set_ref 共同锚定校准数据范围。
    n_calibration_samples: int


class SupervisionSpan(TypedDict, total=False):
    """一条监督 span —— 控制值 + 监督属性 + 可溯源来源。

    必需字段见 ``SPAN_REQUIRED_FIELDS``；``utt_id`` 亦必需。``calibration``
    仅在 ``calibrated=True`` 时必需，且必须含 ``method`` / ``version`` 与
    校准样本集合溯源（``calibration_sample_set_ref`` / ``n_calibration_samples``，
    票据 05）；``intensity_policy`` 可选，但若为 ``fixed_*`` 则
    ``intensity_mask`` 必须为 False（ESD/FEDD fixed-medium 仅有控制输入、
    无强度真值）。
    """

    utt_id: str
    label_source: str
    supervision_granularity: str
    start_sec: float
    end_sec: float
    emotion_soft_distribution: list[float]
    vad: list[float]
    arousal: float
    control_emotion_id: int
    control_intensity_id: int
    raw_score: float
    calibrated: bool
    calibration: Optional[Calibration]
    emotion_mask: bool
    intensity_mask: bool
    supervision_weight: float
    provenance: Union[str, dict]
    intensity_policy: str


class GenerationRow(TypedDict, total=False):
    """一条生成结果：身份 + 控制/prompt 引用 + 解码配置 + finish_reason + WAV。

    身份引用至少满足：source（source_revision / source_patch_bundle /
    source_patch_sha256 之一）、checkpoint（checkpoint_sha256 /
    checkpoint_ref 之一）、control（control_row_ref / control_row 之一）、
    prompt（prompt_row_ref / prompt_row 之一）。

    注意（ADR-0020）：本 schema **不含** WAV 内容哈希字段——产物身份用
    ``wav_path`` + 结构化身份字段绑定，禁止用 WAV 内容哈希标定产物。
    """

    utt_id: str
    finish_reason: str
    source_revision: str
    source_patch_bundle: dict
    source_patch_sha256: str
    checkpoint_sha256: str
    checkpoint_ref: dict
    control_row_ref: str
    control_row: dict
    prompt_row_ref: str
    prompt_row: dict
    decode_config: dict
    seed: int
    wav_path: str


class Evaluator(TypedDict, total=False):
    """评测器身份：冻结、与训练任务头隔离（MAP §3 evaluator）。"""

    name: str
    version: str
    label_space: list[str]
    sample_rate_hz: int
    frame_rate_hz: float
    window_sec: float
    calibration: dict
    self_evidence_risk: bool


class EvaluationRow(TypedDict, total=False):
    """一条逐样本/逐 span 评测结果：绑定 generation row + 控制 span + evaluator。"""

    utt_id: str
    generation_row_ref: str
    generation_row: dict
    control_span_ref: str
    control_span: dict
    evaluator: dict
    boundary_evidence_tier: str
    metrics: dict


class Aggregate(TypedDict, total=False):
    """聚合指标：携带 ``evidence_tier`` 以支持 exact/approximate 分离。"""

    evidence_tier: str
    metric_contract_version: str
    metrics: dict
    n_samples: int


# ============================================================
# 路径规整（workspace-relative, POSIX；合同原语与数据流水线共用）
# ============================================================


def normalize_workspace_path(path: str | Path, workspace_root: str | Path) -> str:
    """将工作区内路径规范化为 POSIX 相对路径，禁止绝对路径泄漏到合同产物。

    仓库外路径 → ValueError。合同原语与 v1 数据流水线共用同一语义。
    """
    workspace = Path(workspace_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"path is outside workspace: {candidate}") from exc
    return relative.as_posix()


# ============================================================
# 合同原语内部校验 helper
# ============================================================


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_non_empty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and len(value) > 0


def _require_number(result: Mapping[str, Any], key: str) -> float:
    value = result.get(key)
    if not _is_number(value):
        raise ValueError(f"{key} must be numeric, got {value!r}")
    return float(value)


def _require_bool(result: Mapping[str, Any], key: str) -> bool:
    value = result.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be bool, got {value!r}")
    return value


def _require_non_empty_str(result: Mapping[str, Any], key: str) -> str:
    value = result.get(key)
    if not _is_non_empty_str(value):
        raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _present_str(result: Mapping[str, Any], key: str) -> bool:
    return _is_non_empty_str(result.get(key))


def _present_mapping(result: Mapping[str, Any], key: str) -> bool:
    return _is_non_empty_mapping(result.get(key))


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def _is_relative_posix_path(value: Any) -> bool:
    r"""路径是否为 workspace-relative POSIX 形态（拒绝绝对路径与反斜杠泄漏）。

    不需要 workspace_root：只做格式校验。拒绝前导斜杠或反斜杠、任何反斜杠、
    Windows 盘符前缀（如 ``C:\`` 或 ``C:/``）。生产侧仍用 ``normalize_workspace_path``
    做完整 resolve；本函数用于校验器侧防止绝对/非 POSIX 路径溜进合同产物。
    """
    if not isinstance(value, str) or value.strip() == "":
        return False
    if _LEADING_SLASH_RE.match(value):
        return False
    if "\\" in value:
        return False
    if _DRIVE_LETTER_RE.match(value):
        return False
    return True


# ============================================================
# 合同级校验器
# ============================================================


def assert_no_dead_config(resolved_conf: Mapping[str, Any]) -> Mapping[str, Any]:
    """拒绝 resolved 配置中的死配置字段（mix_ratio / alpha）。

    单流协议删除双流 ``mix_ratio``；顶层 ``alpha`` 是 v1 遗留占位，采样超参
    归属逐生成 ``decode_config``。任一出现 → ValueError。

    注：``emo_loss_weight`` 不再是死字段——它现在是 ``Qwen2LM_Emotion`` 的可选
    input-end 句级监督 CE 权重（作为 ``llm`` 构造参数消费，默认 ``0.0``=关闭）。
    """
    if not isinstance(resolved_conf, Mapping):
        raise ValueError("resolved config must be a mapping")
    present = sorted(k for k in DEAD_CONFIG_KEYS if k in resolved_conf)
    if present:
        raise ValueError(
            f"dead config fields forbidden in {CONTRACT_NAME} resolved config: {present} "
            "(mix_ratio=bistream removed; "
            "alpha=top-level placeholder, sampling params belong in decode_config)"
        )
    return resolved_conf


def validate_contract_config(conf: Mapping[str, Any]) -> dict[str, Any]:
    """合同级统一入口：拒绝未知 contract_name / schema_version 与死配置字段。"""
    if not isinstance(conf, Mapping):
        raise ValueError("contract config must be a mapping")
    name = conf.get("contract_name")
    if name != CONTRACT_NAME:
        raise ValueError(
            f"contract_name must be {CONTRACT_NAME!r}, got {name!r}"
        )
    version = conf.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )
    assert_no_dead_config(conf)
    return dict(conf)


# ============================================================
# SupervisionSpan 校验
# ============================================================


def validate_span(row: Mapping[str, Any]) -> dict[str, Any]:
    """校验一条 SupervisionSpan；返回规范化 dict，失败抛 ValueError。"""
    if not isinstance(row, Mapping):
        raise ValueError("supervision span must be a mapping")
    result = dict(row)

    utt_id = str(result.get("utt_id", "")).strip()
    if not utt_id:
        raise ValueError("SupervisionSpan requires non-empty utt_id")
    result["utt_id"] = utt_id

    for key in SPAN_REQUIRED_FIELDS:
        if key not in result or result[key] is None:
            raise ValueError(f"SupervisionSpan missing required field: {key}")

    # ``confidence`` 永远不是合法字段：诚实字段是 ``raw_score`` + ``calibration``，
    # 无论是否校准都不得出现（MAP §3）。先于校准块检查，确保 calibrated=True 也拒。
    if "confidence" in result and result["confidence"] is not None:
        raise ValueError(
            "'confidence' is never a valid SupervisionSpan field "
            "(use raw_score + calibration; MAP §3)"
        )

    _require_non_empty_str(result, "label_source")

    granularity = result["supervision_granularity"]
    if granularity not in SPAN_GRANULARITIES:
        raise ValueError(
            f"supervision_granularity must be one of {sorted(SPAN_GRANULARITIES)}, "
            f"got {granularity!r}"
        )

    start_sec = _require_number(result, "start_sec")
    end_sec = _require_number(result, "end_sec")
    if start_sec < 0.0 or end_sec < 0.0:
        raise ValueError(
            f"start_sec/end_sec must be >= 0: start={start_sec} end={end_sec}"
        )
    if not (end_sec > start_sec):
        raise ValueError(
            f"start_sec must strictly precede end_sec: start={start_sec} end={end_sec}"
        )

    emotion_mask = _require_bool(result, "emotion_mask")
    dist = result.get("emotion_soft_distribution")
    if emotion_mask:
        if dist is None:
            raise ValueError(
                "SupervisionSpan missing required field: emotion_soft_distribution "
                "(required when emotion_mask=True; one-hot is an honest hard label)"
            )
    # 若 soft_dist 存在（无论 mask），都必须合法：长度 5、每项 [0,1]、和为 1。
    if dist is not None:
        if not isinstance(dist, (list, tuple)) or len(dist) != 5:
            raise ValueError(
                "emotion_soft_distribution must have length 5 (5 emotions), got "
                f"{type(dist).__name__} len={len(dist) if hasattr(dist, '__len__') else 'n/a'}"
            )
        for index, prob in enumerate(dist):
            if not _is_number(prob) or not (0.0 <= float(prob) <= 1.0):
                raise ValueError(
                    f"emotion_soft_distribution[{index}] must be in [0,1], got {prob!r}"
                )
        if abs(sum(float(p) for p in dist) - 1.0) > _SOFT_DIST_SUM_TOL:
            raise ValueError(
                "emotion_soft_distribution must sum to 1.0 "
                f"(got {sum(float(p) for p in dist):.6f})"
            )

    # vad 始终可选（IEMOCAP 标注器有完整 VAD；ESD/FEDD 无）；存在时必须长度 3 数值。
    vad = result.get("vad")
    if vad is not None:
        if not isinstance(vad, (list, tuple)) or len(vad) != 3:
            raise ValueError("vad must have length 3 [valence, arousal, dominance]")
        for index, value in enumerate(vad):
            if not _is_number(value):
                raise ValueError(f"vad[{index}] must be numeric, got {value!r}")

    # arousal 由 intensity_mask 门控：True ⇒ 必需（连续强度目标）；False ⇒ 必须缺省。
    intensity_mask = _require_bool(result, "intensity_mask")
    arousal = result.get("arousal")
    if intensity_mask:
        if not _is_number(arousal):
            raise ValueError(
                "arousal is required (continuous intensity target) when intensity_mask=True"
            )
    else:
        if arousal is not None:
            raise ValueError(
                "arousal must be absent when intensity_mask=False "
                "(no continuous intensity target; e.g. ESD/FEDD fixed-*)"
            )

    emotion_lo, emotion_hi = CONTROL_EMOTION_ID_RANGE
    cei = result["control_emotion_id"]
    if not isinstance(cei, int) or isinstance(cei, bool) or not (emotion_lo <= cei <= emotion_hi):
        raise ValueError(
            f"control_emotion_id must be int in [{emotion_lo},{emotion_hi}], got {cei!r}"
        )
    intensity_lo, intensity_hi = CONTROL_INTENSITY_ID_RANGE
    cii = result["control_intensity_id"]
    if not isinstance(cii, int) or isinstance(cii, bool) or not (intensity_lo <= cii <= intensity_hi):
        raise ValueError(
            f"control_intensity_id must be int in [{intensity_lo},{intensity_hi}], got {cii!r}"
        )

    # calibrated / raw_score / calibration 条件必需（见 SPAN_CONDITIONAL_FIELDS）。
    calibrated = _require_bool(result, "calibrated")
    calibration = result.get("calibration")
    raw_score = result.get("raw_score")
    if calibrated:
        if not isinstance(calibration, Mapping):
            raise ValueError(
                "calibrated=True requires a calibration record "
                "{method, version, calibration_sample_set_ref, n_calibration_samples}"
            )
        for key in ("method", "version"):
            mv = calibration.get(key)
            if not _is_non_empty_str(mv):
                raise ValueError(f"calibration.{key} must be a non-empty string")
        # 校准样本集合溯源（票据 05 / 审查 #2）：校准 score 数据范围必须可审计。
        # calibration_sample_set_ref：指向校准样本集的引用/标识；缺失或空 → 拒绝。
        sample_set_ref = calibration.get("calibration_sample_set_ref")
        if not _is_non_empty_str(sample_set_ref):
            raise ValueError(
                f"calibration.calibration_sample_set_ref must be a non-empty string "
                f"(span {utt_id}; calibrated score data range must be auditable)"
            )
        # n_calibration_samples：校准样本数，正整数；缺失或非正 → 拒绝（拒绝 bool）。
        n_cal = calibration.get("n_calibration_samples")
        if (
            not isinstance(n_cal, int)
            or isinstance(n_cal, bool)
            or n_cal <= 0
        ):
            raise ValueError(
                f"calibration.n_calibration_samples must be a positive int "
                f"(span {utt_id}; got {n_cal!r})"
            )
        if not _is_number(raw_score):
            raise ValueError(
                "raw_score is required when calibrated=True (the value that was calibrated)"
            )
    else:
        if calibration is not None:
            raise ValueError(
                "calibrated=False must not carry a calibration record"
            )
        if raw_score is not None and not _is_number(raw_score):
            raise ValueError(f"raw_score must be numeric, got {raw_score!r}")

    weight = result["supervision_weight"]
    if not _is_number(weight) or not (0.0 <= float(weight) <= 1.0):
        raise ValueError(
            f"supervision_weight must be float in [0,1], got {weight!r}"
        )

    provenance = result["provenance"]
    if not (_is_non_empty_str(provenance) or _is_non_empty_mapping(provenance)):
        raise ValueError(
            "provenance must be a non-empty string or mapping"
        )

    policy = result.get("intensity_policy")
    if isinstance(policy, str) and policy.startswith("fixed_") and intensity_mask:
        raise ValueError(
            "intensity_mask must be False when intensity_policy is 'fixed_*' "
            "(no intensity ground truth; e.g. ESD/FEDD fixed-medium control input)"
        )

    return result


# ============================================================
# GenerationRow 校验
# ============================================================


def validate_generation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """校验一条 GenerationRow；返回 dict，失败抛 ValueError。

    注意（ADR-0020）：WAV 内容哈希字段已从 schema 移除；产物身份仅由
    ``wav_path`` + 结构化身份字段绑定。本函数不读、不校验任何 WAV 内容哈希。
    """
    if not isinstance(row, Mapping):
        raise ValueError("generation row must be a mapping")
    result = dict(row)

    utt_id = str(result.get("utt_id", "")).strip()
    if not utt_id:
        raise ValueError("GenerationRow requires non-empty utt_id")
    result["utt_id"] = utt_id

    finish_reason = result.get("finish_reason")
    if finish_reason not in FINISH_REASONS:
        raise ValueError(
            f"finish_reason must be one of {sorted(FINISH_REASONS)}, got {finish_reason!r}"
        )

    if not any(_present_str(result, key) or _present_mapping(result, key) for key in SOURCE_IDENTITY_KEYS):
        raise ValueError(
            "generation row missing source identity "
            f"(one of {list(SOURCE_IDENTITY_KEYS)})"
        )

    checkpoint_present = any(
        _present_str(result, key) or _present_mapping(result, key)
        for key in CHECKPOINT_IDENTITY_KEYS
    )
    if not checkpoint_present:
        raise ValueError(
            "generation row missing checkpoint identity "
            f"(one of {list(CHECKPOINT_IDENTITY_KEYS)})"
        )
    if "checkpoint_sha256" in result and result["checkpoint_sha256"] is not None:
        if not _is_sha256_hex(result["checkpoint_sha256"]):
            raise ValueError(
                "checkpoint_sha256 must be a 64-char SHA-256 hex string"
            )

    if not (_present_str(result, "control_row_ref") or _present_mapping(result, "control_row")):
        raise ValueError(
            "generation row missing control row reference "
            "(control_row_ref or control_row)"
        )
    if not (_present_str(result, "prompt_row_ref") or _present_mapping(result, "prompt_row")):
        raise ValueError(
            "generation row missing prompt row reference "
            "(prompt_row_ref or prompt_row)"
        )

    decode_config = result.get("decode_config")
    if not isinstance(decode_config, Mapping):
        raise ValueError(
            "decode_config must be a mapping (min/max token_text_ratio, hard cap, ...)"
        )

    # seed（per-request 固定随机种子）：非负 int，bool 拒绝（bool 是 int 子类）
    seed = result.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(
            "GenerationRow requires seed (non-negative int); "
            f"got {type(seed).__name__}: {seed!r}"
        )
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    wav_path = result.get("wav_path")
    if finish_reason == "eos":
        if not _is_non_empty_str(wav_path):
            raise ValueError(
                "finish_reason=eos requires wav_path (only eos enters acoustics)"
            )
        if not _is_relative_posix_path(wav_path):
            raise ValueError(
                f"wav_path must be workspace-relative POSIX (no leading slash, "
                f"backslashes, or drive letters): got {wav_path!r}"
            )
    else:
        if wav_path is not None:
            raise ValueError(
                f"finish_reason={finish_reason!r} must not carry a formal wav_path "
                "(only eos enters acoustics / formal WAV)"
            )

    return result


# ============================================================
# v1 数据流水线函数（构建冻结 v1 数据产物；延迟导入重依赖）
# ============================================================


def normalize_manifest_row(
    row: Mapping[str, object],
    *,
    dataset: str,
    workspace_root: str | Path,
    label_source: str | None = None,
) -> dict[str, object]:
    """统一来源/训练/eval manifest 行的路径、文本和来源字段。"""
    result = dict(row)
    utt_id = str(result.get("utt_id", "")).strip()
    if not utt_id:
        raise ValueError("manifest row requires utt_id")

    wav_value = result.get("wav_path") or result.get("audio_filepath")
    if not wav_value:
        raise ValueError(f"manifest row {utt_id} requires wav_path")
    result["wav_path"] = normalize_workspace_path(str(wav_value), workspace_root)

    for key in ("prompt_wav", "reference_wav", "target_wav"):
        if result.get(key):
            result[key] = normalize_workspace_path(str(result[key]), workspace_root)

    tagged_text = result.get("tagged_text")
    text_value = str(result.get("text", "") or "")
    plain_text = str(result.get("plain_text", "") or "")
    if not plain_text:
        plain_text = text_value if "<emotion" not in text_value else ""
    if not tagged_text and "<emotion" in text_value:
        tagged_text = text_value
    if not plain_text:
        raise ValueError(f"manifest row {utt_id} requires plain text")

    result["utt_id"] = utt_id
    result["text"] = plain_text
    result["plain_text"] = plain_text
    if tagged_text:
        result["tagged_text"] = str(tagged_text)
    result["speaker_id"] = str(result.get("speaker_id", ""))
    result["source_dataset"] = dataset
    result["label_source"] = label_source or str(result.get("label_source", ""))
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_contract_provenance(
    contract_dir: str | Path,
    *,
    contract: Mapping[str, object],
    sources: Sequence[Mapping[str, object]],
    membership: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """写入新版合同要求的四个 provenance 文件。

    注意：``contract`` 字典若携带 ``contract_name`` 则以调用方值为准（覆盖模块
    默认 ``CONTRACT_NAME``），便于冻结的 v1 产物仍写出 ``emofilm_v1`` 身份。
    """
    contract_dir = Path(contract_dir)
    provenance_dir = contract_dir / "provenance"
    contract_value = {"contract_name": CONTRACT_NAME, **dict(contract)}
    _write_json(provenance_dir / "contract.json", contract_value)
    _write_json(provenance_dir / "sources.json", list(sources))
    _write_json(provenance_dir / "membership.json", dict(membership))
    provenance_dir.mkdir(parents=True, exist_ok=True)
    with (provenance_dir / "artifacts.jsonl").open("w", encoding="utf-8") as handle:
        for artifact in artifacts:
            handle.write(json.dumps(dict(artifact), ensure_ascii=False) + "\n")
    return contract_value


def validate_frame_artifact(frame_path: Path, provenance_path: Path) -> dict:
    """验证 emotion2vec-base 帧特征为 768d/50Hz/20ms 且有来源记录。"""
    import torch  # 延迟导入：合同原语测试无需 torch

    artifact = torch.load(frame_path, map_location="cpu", weights_only=True)
    feats = artifact.get("feats")
    if not torch.is_tensor(feats) or feats.ndim != 2 or feats.shape[1] != 768:
        raise ValueError(f"frame artifact must have shape (T, 768): {frame_path}")
    frame_rate_hz = float(artifact.get("frame_rate_hz", 0.0))
    frame_step_ms = float(artifact.get("frame_step_ms", 1000.0 / frame_rate_hz if frame_rate_hz else 0.0))
    if abs(frame_rate_hz - 50.0) > 1e-6 or abs(frame_step_ms - 20.0) > 1e-6:
        raise ValueError(f"frame artifact must be 50Hz/20ms: {frame_path}")

    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    required = {"model_id", "revision", "checkpoint_sha256", "upstream"}
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"missing frame provenance fields: {sorted(missing)}")
    if len(provenance["checkpoint_sha256"]) != 64:
        raise ValueError("checkpoint_sha256 must be a SHA-256 hex string")
    upstream = provenance["upstream"]
    if not isinstance(upstream, dict) or len(str(upstream.get("sha256", ""))) != 64:
        raise ValueError("upstream provenance must contain a SHA-256 hash")
    return {
        "feature_dim": int(feats.shape[1]),
        "frame_rate_hz": frame_rate_hz,
        "frame_step_ms": frame_step_ms,
    }


def load_word_sequence_model(checkpoint: Path, device: str = "cpu") -> "WordSequenceModel":
    """strict load EmoFiLM 768d/5 类/3D VAD WordSequenceModel。"""
    import torch  # 延迟导入：合同原语测试无需 torch
    from cosyvoice_emo.emo_annotator import WordSequenceModel

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = WordSequenceModel(input_dim=768, num_classes=5, num_heads=8, dropout_rate=0.3, reg_dim=3)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _tag(emotion: str, intensity: str, words: Sequence[str]) -> str:
    text = " ".join(words)
    return f"<emotion type='{emotion}' intensity='{intensity}'>{text}</emotion>"


def merge_word_predictions(words: Iterable[Mapping[str, str]]) -> str:
    """按相邻词的 emotion 和 intensity 双键合并。"""
    segments = []
    current_key = None
    current_words: list[str] = []
    for item in words:
        key = (item["predicted_emotion"], item["predicted_intensity"])
        if current_key != key:
            if current_key is not None:
                segments.append(_tag(*current_key, current_words))
            current_key = key
            current_words = []
        current_words.append(item["word"])
    if current_key is not None:
        segments.append(_tag(*current_key, current_words))
    return " ".join(segments)


def text_tokens(text: str) -> list[str]:
    """提取用于音频、TextGrid 与 tagged text 覆盖比较的规范词序。"""
    plain = html.unescape(_EMOTION_TAG_RE.sub(" ", text)).lower()
    return _WORD_RE.findall(plain.replace("’", "'"))


def classify_text_coverage(plain_text: str, compared_text: str) -> dict[str, object]:
    """只放行精确词覆盖或撇号切分等价；其他差异统一拒绝。"""
    plain_tokens = text_tokens(plain_text)
    compared_tokens = text_tokens(compared_text)
    if plain_tokens == compared_tokens:
        return {"decision": "keep", "category": "exact"}
    if "".join(plain_tokens).replace("'", "") == "".join(compared_tokens).replace("'", ""):
        return {"decision": "keep", "category": "apostrophe_tokenization"}

    compared_counts: dict[str, int] = {}
    for token in compared_tokens:
        compared_counts[token] = compared_counts.get(token, 0) + 1
    missing = []
    for token in plain_tokens:
        if compared_counts.get(token, 0):
            compared_counts[token] -= 1
        else:
            missing.append(token)
    return {
        "decision": "reject",
        "category": "audio_text_mismatch",
        "plain_tokens": plain_tokens,
        "tagged_tokens": compared_tokens,
        "missing_from_tagged": missing,
    }


def validate_membership(
    train_ids: set[str],
    cv_ids: set[str],
    rejected_ids: set[str],
    frozen_train_ids: set[str],
    frozen_cv_ids: set[str],
) -> None:
    """验证 train/cv 只移除 rejected，且两边无交集。"""
    frozen_union = frozen_train_ids | frozen_cv_ids
    if not rejected_ids <= frozen_union:
        raise ValueError("rejected ids must come from frozen union membership")
    if train_ids & cv_ids:
        raise ValueError("train/cv membership overlap")
    if train_ids != frozen_train_ids - rejected_ids:
        raise ValueError("train membership differs from frozen ids minus rejected")
    if cv_ids != frozen_cv_ids - rejected_ids:
        raise ValueError("cv membership differs from frozen ids minus rejected")


def validate_rejected_manifest(rejected: Sequence[Mapping[str, str]], original: Sequence[Mapping[str, str]]) -> dict:
    """验证 rejected 逐条有原因、来自 source 且分布可审计。"""
    total = len(original)
    if total == 0:
        raise ValueError("original manifest is empty")
    if any(not row.get("utt_id") or not row.get("reason") for row in rejected):
        raise ValueError("each rejected row needs utt_id and reason")
    fraction = len(rejected) / total
    original_by_id = {row["utt_id"]: row for row in original}
    if len(original_by_id) != total:
        raise ValueError("original manifest contains duplicate utt_id")
    rejected_ids = [row["utt_id"] for row in rejected]
    if len(set(rejected_ids)) != len(rejected_ids):
        raise ValueError("rejected manifest contains duplicate utt_id")
    if not set(row["utt_id"] for row in rejected) <= original_by_id.keys():
        raise ValueError("rejected row is not in original")
    speaker_counts = Counter(row.get("speaker_id") for row in rejected)
    emotion_counts = Counter(row.get("sentence_emotion") for row in rejected)
    concentration_limit = 0.75
    if len(rejected) >= 2:
        for dimension, counts in (("speaker", speaker_counts), ("emotion", emotion_counts)):
            if any(count / len(rejected) > concentration_limit for count in counts.values()):
                raise ValueError(f"rejected samples are concentrated by {dimension}")
    return {
        "rejected_count": len(rejected),
        "fraction": fraction,
        "speaker_counts": dict(speaker_counts),
        "emotion_counts": dict(emotion_counts),
    }


def validate_eval_assets(
    rows: Sequence[Mapping[str, str]],
    expected_count: int,
    workspace_root: str | Path | None = None,
) -> dict:
    """验证固定评测 population 的 target/reference/prompt/文本/标签完整。"""
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} eval rows, got {len(rows)}")
    required = ("utt_id", "target_wav", "reference_wav", "prompt_wav", "text", "label", "prompt_text")
    missing = []
    seen = set()
    for row in rows:
        utt_id = row.get("utt_id")
        if not utt_id or utt_id in seen:
            missing.append(f"duplicate-or-empty:{utt_id}")
        seen.add(utt_id)
        for key in required[1:]:
            value = row.get(key)
            if key.endswith("wav") and value and workspace_root is not None:
                value = Path(workspace_root) / value if not Path(value).is_absolute() else Path(value)
            if not value or (key.endswith("wav") and not Path(value).is_file()):
                missing.append(f"{utt_id}:{key}")
    if missing:
        raise ValueError(f"incomplete eval assets: {missing}")
    return {"count": len(rows), "missing": missing}


def _read_shard_list(data_list: Path) -> list[Path]:
    paths = []
    for raw in data_list.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            shard = Path(raw.strip())
            if shard.is_absolute():
                paths.append(shard)
                continue
            paths.append(data_list.parent / shard)
    return paths


def validate_train_cv_parquet(
    train_list: Path,
    cv_list: Path,
) -> dict:
    """读取 train/cv 全部 shard，并拒绝共享 shard。"""
    import pyarrow.parquet as pq  # 延迟导入：合同原语测试无需 pyarrow

    train_shards = _read_shard_list(train_list)
    cv_shards = _read_shard_list(cv_list)
    shared = sorted(str(path.resolve()) for path in set(train_shards) & set(cv_shards))
    if shared:
        raise ValueError(f"train/cv share parquet shards: {shared}")
    train_rows = sum(pq.read_table(path).num_rows for path in train_shards)
    cv_rows = sum(pq.read_table(path).num_rows for path in cv_shards)
    return {"train_rows": train_rows, "cv_rows": cv_rows, "shared_shards": shared}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_train_cv_contract(
    contract_dir: Path,
    *,
    train_rows: Sequence[Mapping[str, object]],
    cv_rows: Sequence[Mapping[str, object]],
    frozen_train_ids: set[str],
    frozen_cv_ids: set[str],
    rejected_rows: Sequence[Mapping[str, object]],
    num_utts_per_parquet: int = 1000,
    num_processes: int = 1,
    original_rows: Sequence[Mapping[str, object]] | None = None,
    source_root: str | Path | None = None,
    optional_maps: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
) -> dict[str, dict[str, object]]:
    """构建新版独立 train/cv 合同目录，不复制全量音频视图。"""
    import torch  # 延迟导入：合同原语测试无需 torch
    from tools.jsonl_to_cosyvoice_src import write_src_dir
    from tools.make_parquet_list import pack_src_dir

    contract_dir = Path(contract_dir)
    rejected_ids = {str(row["utt_id"]) for row in rejected_rows}
    train_ids = {str(row["utt_id"]) for row in train_rows}
    cv_ids = {str(row["utt_id"]) for row in cv_rows}
    validate_membership(train_ids, cv_ids, rejected_ids, frozen_train_ids, frozen_cv_ids)
    if original_rows is not None:
        validate_rejected_manifest(rejected_rows, original_rows)

    splits_dir = contract_dir / "splits"
    staging_dir = contract_dir / ".splits.staging"
    backup_dir = contract_dir / ".splits.backup"
    for path in (staging_dir, backup_dir):
        if path.exists():
            shutil.rmtree(path)

    reports: dict[str, dict[str, object]] = {}
    try:
        for split, rows in (("train", train_rows), ("cv", cv_rows)):
            split_root = staging_dir / split
            split_root.mkdir(parents=True, exist_ok=True)
            _write_jsonl(split_root / "manifest.jsonl", rows)
            split_src = split_root / "src"
            write_src_dir(str(split_src), [dict(row) for row in rows], use_tagged_text=True)
            for filename, mapping in (optional_maps or {}).get(split, {}).items():
                if not isinstance(mapping, Mapping):
                    raise ValueError(f"optional map must be a mapping: {split}/{filename}")
                if filename == "spk2embedding.pt":
                    expected_keys = {str(row.get("speaker_id", "")) for row in rows}
                else:
                    expected_keys = {str(row["utt_id"]) for row in rows}
                actual_keys = {str(key) for key in mapping}
                if actual_keys != expected_keys:
                    raise ValueError(
                        f"optional map coverage mismatch for {split}/{filename}: "
                        f"missing={sorted(expected_keys - actual_keys)[:5]} "
                        f"extra={sorted(actual_keys - expected_keys)[:5]}"
                    )
                torch.save(dict(mapping), split_src / filename)
            pack_report = pack_src_dir(
                split_src,
                split_root / "parquet",
                num_utts_per_parquet=num_utts_per_parquet,
                num_processes=num_processes,
                source_root=source_root,
                shard_prefix=f"{split}_",
            )
            data_list = split_root / "parquet" / "data.list"
            reports[split] = {
                "rows": len(rows),
                "shards": pack_report["shards"],
                "data_list": str(contract_dir / "splits" / split / "parquet" / "data.list"),
                "data_list_sha256": _sha256_file(data_list),
            }

        if splits_dir.exists():
            splits_dir.rename(backup_dir)
        staging_dir.rename(splits_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if backup_dir.exists():
            if splits_dir.exists():
                shutil.rmtree(splits_dir)
            backup_dir.rename(splits_dir)
        raise

    _write_json(
        contract_dir / "provenance" / "membership.json",
        {
            "train": sorted(train_ids),
            "cv": sorted(cv_ids),
            "rejected": sorted(rejected_ids),
            "frozen_train": sorted(frozen_train_ids),
            "frozen_cv": sorted(frozen_cv_ids),
        },
    )
    return reports


def build_annotation_parser() -> argparse.ArgumentParser:
    """生产 EmoFiLM 标注 CLI；刻意不提供历史 smoothing/majority 开关。"""
    parser = argparse.ArgumentParser(description="build EmoFiLM word-level emotion tags")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--textgrid_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="validate emofilm data contract")
    parser.add_argument("--contract_dir", type=Path)
    parser.add_argument("--word_sequence_checkpoint", type=Path)
    return parser


if __name__ == "__main__":
    build_parser().parse_args()
