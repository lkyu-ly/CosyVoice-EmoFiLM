"""EmoFiLM 合同 schema 与校验器 focused 测试。

覆盖（ADR-0020 扁平化后；合同原语库合并为单一活跃权威 ``tools/build_emofilm_contract.py``）：
- 合同常量（CONTRACT_NAME / SCHEMA_VERSION / FINISH_REASONS）；
- 活跃合同身份为 ``emofilm``（v1 ``emofilm_v1`` / v2 ``emofilm_v2`` 标识已从活跃
  合同代码删除——反转语义锁，替代原"v1 文件字节 sha256 冻结"源码哈希锁）；
- ADR 0001-0018 历史决策冻结（拼接 sha256 锁保留；ADR-0020 decision 4）；
- SupervisionSpan / GenerationRow / EvaluationRow / Aggregate 合法构造通过；
- 合同级校验器拒绝：未知 contract_name / schema_version、缺失身份引用
  （generation row 缺 source/checkpoint/control/prompt 之一）、无效
  boundary_evidence_tier、死配置字段（mix_ratio / emo_loss_weight / alpha）；
- ESD fixed-medium span 的 intensity_mask 必须为 False（ESD 不伪造强度）；
- 未校准 raw_score 不得标 calibrated=True，未校准不得命名 confidence。

纯 stdlib 合同逻辑，CPU（合同原语 stdlib-only；重依赖在数据流水线函数内延迟导入）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import pytest

from tools.build_emofilm_contract import (
    BOUNDARY_EVIDENCE_TIERS,
    CONTRACT_NAME,
    DEAD_CONFIG_KEYS,
    FINISH_REASONS,
    SCHEMA_VERSION,
    assert_no_dead_config,
    validate_contract_config,
    validate_generation_row,
    validate_span,
)

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "tools" / "build_emofilm_contract.py"
ADR_DIR = ROOT / "docs" / "adr"

# ADR 0001-0018 历史决策拼接摘要（每文件后加 \x00 分隔符；ADR-0020 decision 4：
# 历史决策记录冻结不动，此 sha256 锁保留——它是文档决策冻结锁，非源码哈希锁）。
EXPECTED_ADR_COMBINED_SHA256 = (
    "7cce72f65ba704bbcb8989b17cf1b8a67285bf5a476c638037f733ef7b955aef"
)
ADR_BASENAMES = [f"{i:04d}" for i in range(1, 19)]


def _combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
        digest.update(b"\x00")  # 分隔符，避免拼接歧义
    return digest.hexdigest()


# ============================================================
# 常量与活跃身份（反转语义锁）
# ============================================================

def test_contract_constants():
    assert CONTRACT_NAME == "emofilm"
    assert SCHEMA_VERSION == 2
    assert set(FINISH_REASONS) == {
        "eos",
        "max_len_reached",
        "invalid_token_retry_exhausted",
        "sampler_error",
        "input_rejected",
    }
    assert set(BOUNDARY_EVIDENCE_TIERS) == {"exact", "approximate"}
    assert set(DEAD_CONFIG_KEYS) == {"mix_ratio", "emo_loss_weight", "alpha"}


def test_contract_module_has_active_identity_not_v1_or_v2():
    """反转语义锁：活跃合同代码身份为 ``emofilm``，v1/v2 并行副本标识已删除。

    取代原"v1 build_emofilm_contract.py 字节 sha256 冻结 + AST CONTRACT_NAME=='emofilm_v1'"
    源码哈希锁（ADR-0020：禁源码哈希标定文件）。现在断言反模式已从活跃合同代码移除：
    既不写入 ``CONTRACT_NAME = "emofilm_v1"`` 也不写入 ``"emofilm_v2"`` 合同身份。
    v1 基线由 git 锚点 ``9c6d84b`` 保证，不由工作树哈希保证。
    """
    # 导入侧直接断言活跃身份（合同原语 stdlib-only，导入轻量）。
    assert CONTRACT_NAME == "emofilm"

    # 源码侧反模式扫描：活跃合同模块不得再定义 v1/v2 合同身份字面量。
    source = CONTRACT_PATH.read_text(encoding="utf-8")
    assert 'CONTRACT_NAME = "emofilm_v1"' not in source, (
        "v1 合同身份 'emofilm_v1' 必须从活跃合同代码删除（ADR-0020 单一活跃权威）"
    )
    assert 'CONTRACT_NAME = "emofilm_v2"' not in source, (
        "v2 合同身份 'emofilm_v2' 必须从活跃合同代码删除（ADR-0020 单一活跃权威）"
    )


def test_adrs_0001_0018_are_frozen():
    """ADR 0001-0018 历史决策记录冻结（ADR-0020 decision 4）。

    ADR 是已发生的历史决策文档（非源码），其冻结 sha256 锁保留。原伴随的
    ``conf/emo_film.yaml`` 源码 sha256 锁已删除（禁源码哈希标定；活跃配置
    ``conf/emo_film.yaml`` 在票据 02 由原 v2 副本合并后随主线演化，v2 副本
    已在票据 04 删除）。
    """
    adr_files = []
    missing = []
    for basename in ADR_BASENAMES:
        matches = sorted(ADR_DIR.glob(f"{basename}-*.md"))
        if not matches:
            missing.append(basename)
            continue
        adr_files.extend(matches)
    assert not missing, f"ADR 文件缺失: {missing}"
    # 每个 basename 恰好一个文件（防意外重复/覆盖）。
    assert len(adr_files) == len(ADR_BASENAMES), (
        f"ADR 文件数异常：期望 {len(ADR_BASENAMES)}，实际 {len(adr_files)}"
    )

    combined = _combined_sha256(adr_files)
    assert combined == EXPECTED_ADR_COMBINED_SHA256, (
        "docs/adr/0001-0018 被 modified；ADR 0001-0018 历史决策冻结（ADR-0020 decision 4）。"
    )


# ============================================================
# 合同级校验器（contract_name / schema_version / 死配置）
# ============================================================

def test_validate_contract_config_accepts_active():
    validate_contract_config(
        {"contract_name": "emofilm", "schema_version": 2}
    )


def test_validate_contract_config_rejects_unknown_name():
    with pytest.raises(ValueError, match="contract_name"):
        validate_contract_config(
            {"contract_name": "emofilm_v3", "schema_version": 2}
        )


def test_validate_contract_config_rejects_unknown_version():
    with pytest.raises(ValueError, match="schema_version"):
        validate_contract_config(
            {"contract_name": "emofilm", "schema_version": 99}
        )


def test_assert_no_dead_config_accepts_clean():
    assert_no_dead_config({"lr": 1e-5, "token_text_ratio": [5.0, 25.0]})


@pytest.mark.parametrize("dead_key", sorted(DEAD_CONFIG_KEYS))
def test_assert_no_dead_config_rejects_dead_fields(dead_key):
    with pytest.raises(ValueError, match="dead config"):
        assert_no_dead_config({dead_key: 0.05})


def test_validate_contract_config_also_rejects_dead_config():
    """合同级校验器是统一入口：含死配置字段时一并拒绝。"""
    with pytest.raises(ValueError, match="dead config"):
        validate_contract_config(
            {
                "contract_name": "emofilm",
                "schema_version": 2,
                "mix_ratio": [5, 15],
            }
        )


# ============================================================
# SupervisionSpan
# ============================================================

def _valid_span() -> dict[str, Any]:
    """构造一条合法的句级广播弱监督 span（IEMOCAP 词级标签语义）。"""
    return {
        "utt_id": "iemocap_sess1_0001",
        "label_source": "word_weak_sentence_broadcast",
        "supervision_granularity": "utterance",
        "start_sec": 0.0,
        "end_sec": 2.5,
        "emotion_soft_distribution": [0.02, 0.83, 0.05, 0.07, 0.03],
        "vad": [3.1, 2.4, 3.6],
        "arousal": 2.4,
        "control_emotion_id": 2,  # hap（tokenizer id space 1..5）
        "control_intensity_id": 2,  # medium（1..3）
        "raw_score": 0.83,
        "calibrated": True,
        "calibration": {
            "method": "isotonic",
            "version": "emotion2vec-v1",
            # 校准样本集合溯源（票据 05）：使校准 score 数据范围可审计。
            "calibration_sample_set_ref": "iemocap_calib_split:v1",
            "n_calibration_samples": 512,
        },
        "emotion_mask": True,
        "intensity_mask": True,
        "supervision_weight": 0.5,
        "provenance": "iemocap_word_sequence_model:v1",
    }


def test_valid_supervision_span_passes():
    validate_span(_valid_span())


def test_span_rejects_missing_required_field():
    span = _valid_span()
    del span["emotion_soft_distribution"]
    with pytest.raises(ValueError, match="emotion_soft_distribution"):
        validate_span(span)


def test_span_rejects_bad_soft_distribution_length():
    span = _valid_span()
    span["emotion_soft_distribution"] = [0.5, 0.5]  # 不是 5 维
    with pytest.raises(ValueError, match="emotion_soft_distribution"):
        validate_span(span)


def test_span_rejects_soft_distribution_not_normalizing():
    span = _valid_span()
    span["emotion_soft_distribution"] = [0.5, 0.4, 0.0, 0.0, 0.0]  # 和=0.9 != 1
    with pytest.raises(ValueError, match="emotion_soft_distribution"):
        validate_span(span)


def test_span_rejects_bad_time_order():
    span = _valid_span()
    span["start_sec"] = 3.0
    span["end_sec"] = 2.0
    with pytest.raises(ValueError, match="start_sec"):
        validate_span(span)


def test_span_rejects_out_of_range_control_ids():
    span = _valid_span()
    span["control_emotion_id"] = 6  # 超出 1..5
    with pytest.raises(ValueError, match="control_emotion_id"):
        validate_span(span)
    span2 = _valid_span()
    span2["control_intensity_id"] = 4  # 超出 1..3
    with pytest.raises(ValueError, match="control_intensity_id"):
        validate_span(span2)


def test_span_rejects_inconsistent_calibrated_false_with_calibration():
    """calibrated=False 但仍携带 calibration 记录 → 不一致，拒绝。"""
    span = _valid_span()  # calibrated=True + calibration + raw_score
    span["calibrated"] = False  # calibration 记录仍在 → 不一致
    with pytest.raises(ValueError, match="calibrat"):
        validate_span(span)


def test_span_uncalibrated_must_not_name_confidence():
    """MAP §3：未校准不得命名 confidence。"""
    span = _valid_span()
    span["calibrated"] = False
    span["calibration"] = None
    span["confidence"] = 0.9  # 未校准却命名 confidence
    with pytest.raises(ValueError, match="confidence"):
        validate_span(span)


def _honest_esd_span() -> dict[str, Any]:
    """一条诚实的 ESD span：仅有数据集全局情感标签，无 VAD/arousal/model score。

    ESD builder（``tools/build_esd_tagged_text.py``）不产出 VAD/arousal/raw_score/
    soft_dist；合同不得强迫 ESD 伪造这些连续输出（review C1）。one-hot soft_dist
    是硬标签的诚实表示。
    """
    return {
        "utt_id": "esd_0001_ang",
        "label_source": "esd_fixed_medium_control",
        "supervision_granularity": "span",
        "start_sec": 0.0,
        "end_sec": 2.5,
        "emotion_soft_distribution": [0.0, 1.0, 0.0, 0.0, 0.0],  # one-hot 硬标签
        "control_emotion_id": 2,  # hap
        "control_intensity_id": 2,  # medium（控制输入，非真值）
        "calibrated": False,
        "emotion_mask": True,
        "intensity_mask": False,  # ESD 无强度真值
        "supervision_weight": 0.25,
        "provenance": "esd_dataset:fixed_medium",
        "intensity_policy": "fixed_medium",
        # 故意缺省：vad / arousal / raw_score / calibration
    }


def test_honest_esd_span_without_continuous_outputs_passes():
    """(a) ESD 仅硬标签 + 无连续输出：validate_span 通过（不伪造监督）。"""
    validate_span(_honest_esd_span())


def test_iemocap_full_annotator_span_passes():
    """(b) IEMOCAP 完整标注器输出：intensity_mask=True + arousal 存在，通过。"""
    span = _valid_span()  # emotion_mask=True + soft_dist；intensity_mask=True + arousal
    validate_span(span)


def test_intensity_mask_true_requires_arousal():
    """(c) intensity_mask=True 但缺 arousal → 拒绝（连续强度目标必需）。"""
    span = _valid_span()
    del span["arousal"]
    with pytest.raises(ValueError, match="arousal"):
        validate_span(span)


def test_calibrated_true_without_calibration_record_fails():
    """(d) calibrated=True 但缺 calibration 记录 → 拒绝。"""
    span = _valid_span()
    del span["calibration"]
    with pytest.raises(ValueError, match="calibration"):
        validate_span(span)


def test_calibrated_true_requires_raw_score():
    """calibrated=True ⇒ raw_score 必需（被校准的那个值）。"""
    span = _valid_span()
    del span["raw_score"]
    with pytest.raises(ValueError, match="raw_score"):
        validate_span(span)


# -- 票据 05：校准样本集合溯源字段（审查 #2 修复）---------------------------

def test_calibrated_true_requires_calibration_sample_set_ref():
    """(a) calibrated=True 但 calibration 缺 calibration_sample_set_ref → 拒绝。

    校准 score 数据范围必须可审计：缺样本集引用 → 校准曲线来源不可追溯。
    """
    span = _valid_span()
    del span["calibration"]["calibration_sample_set_ref"]
    with pytest.raises(ValueError, match="calibration_sample_set_ref"):
        validate_span(span)


def test_calibrated_true_requires_n_calibration_samples():
    """(b) calibrated=True 但 calibration 缺 n_calibration_samples → 拒绝。"""
    span = _valid_span()
    del span["calibration"]["n_calibration_samples"]
    with pytest.raises(ValueError, match="n_calibration_samples"):
        validate_span(span)


def test_calibrated_true_rejects_non_positive_n_calibration_samples():
    """n_calibration_samples 必须是正整数：0 / 负数 / bool 均拒绝。"""
    for bad in (0, -3, True, 1.0, "512"):
        span = _valid_span()
        span["calibration"]["n_calibration_samples"] = bad
        with pytest.raises(ValueError, match="n_calibration_samples"):
            validate_span(span)


def test_calibrated_true_rejects_empty_calibration_sample_set_ref():
    """calibration_sample_set_ref 空字符串/空白 → 拒绝。"""
    for bad in ("", "   "):
        span = _valid_span()
        span["calibration"]["calibration_sample_set_ref"] = bad
        with pytest.raises(ValueError, match="calibration_sample_set_ref"):
            validate_span(span)


def test_calibrated_span_with_full_sample_set_fields_passes():
    """(c) calibrated=True + 完整校准样本集合溯源字段 → 通过。"""
    span = _valid_span()
    validate_span(span)


def test_intensity_mask_false_forbids_arousal():
    """intensity_mask=False ⇒ arousal 必须缺省（无连续强度目标）。"""
    span = _honest_esd_span()
    span["arousal"] = 2.4  # ESD 无强度真值却塞 arousal
    with pytest.raises(ValueError, match="arousal"):
        validate_span(span)


def test_emotion_mask_false_allows_absent_soft_distribution():
    """emotion_mask=False ⇒ emotion_soft_distribution 可缺省。"""
    span = _honest_esd_span()
    span["emotion_mask"] = False
    del span["emotion_soft_distribution"]
    validate_span(span)


def test_one_hot_soft_distribution_is_valid_for_hard_label():
    """one-hot soft_dist（硬标签诚实表示）合法。"""
    span = _honest_esd_span()
    span["emotion_soft_distribution"] = [0.0, 0.0, 1.0, 0.0, 0.0]
    validate_span(span)


def test_confidence_forbidden_even_when_calibrated_true():
    """(I2) confidence 永远不是合法字段：calibrated=True 时也拒绝。"""
    span = _valid_span()  # calibrated=True
    span["confidence"] = 0.9
    with pytest.raises(ValueError, match="confidence"):
        validate_span(span)


def test_esd_fixed_medium_span_intensity_mask_must_be_false():
    """ESD fixed-medium 仅有控制输入、无强度真值：intensity_mask 必须 False。

    规则一般化为：任何 fixed_* intensity_policy 都不得声明强度监督
    （覆盖 ESD 与 FEDD 的 fixed-medium 控制输入）。
    """
    esd_ok = _honest_esd_span()
    validate_span(esd_ok)  # intensity_mask=False 通过

    esd_bad = _valid_span()  # IEMOCAP full（含 arousal，intensity_mask=True 合法）
    esd_bad["label_source"] = "esd_fixed_medium_control"
    esd_bad["supervision_granularity"] = "span"
    esd_bad["intensity_policy"] = "fixed_medium"
    esd_bad["intensity_mask"] = True  # fixed_* 却声明强度监督 → 伪造
    esd_bad["provenance"] = "esd_dataset:fixed_medium"
    with pytest.raises(ValueError, match="intensity_mask"):
        validate_span(esd_bad)


# ============================================================
# GenerationRow
# ============================================================

def _valid_generation_row() -> dict[str, Any]:
    return {
        "utt_id": "esd_0001_ang",
        "finish_reason": "eos",
        "source_revision": "git_sha_abc123",
        "checkpoint_sha256": "a" * 64,
        "control_row_ref": "ctrl/esd_0001_ang.json",
        "prompt_row_ref": "prompt/esd_0001_ang.json",
        "decode_config": {
            "min_token_text_ratio": 5.0,
            "max_token_text_ratio": 25.0,
            "max_len_hard_cap": 600,
        },
        "seed": 1984,
        "wav_path": "wav/emofilm/esd_0001_ang.wav",
    }


def test_valid_generation_row_passes():
    validate_generation_row(_valid_generation_row())


@pytest.mark.parametrize(
    "missing",
    ["source_revision", "checkpoint_sha256", "control_row_ref", "prompt_row_ref"],
)
def test_generation_row_rejects_missing_identity_ref(missing):
    row = _valid_generation_row()
    # source_revision 是 source 身份唯一键时删除它应失败
    if missing == "source_revision":
        row.pop("source_revision", None)
        key = "source"
    elif missing == "checkpoint_sha256":
        row.pop("checkpoint_sha256", None)
        key = "checkpoint"
    elif missing == "control_row_ref":
        row.pop("control_row_ref", None)
        key = "control"
    else:
        row.pop("prompt_row_ref", None)
        key = "prompt"
    with pytest.raises(ValueError, match=key):
        validate_generation_row(row)


def test_generation_row_rejects_unknown_finish_reason():
    row = _valid_generation_row()
    row["finish_reason"] = "bogus"
    with pytest.raises(ValueError, match="finish_reason"):
        validate_generation_row(row)


def test_generation_row_non_eos_must_not_carry_wav():
    """MAP §3：仅 eos 进声学与正式 WAV。非 eos 不得带 wav_path。"""
    row = _valid_generation_row()
    row["finish_reason"] = "max_len_reached"
    # 仍带 wav_path → 拒绝
    with pytest.raises(ValueError, match="wav"):
        validate_generation_row(row)


def test_generation_row_non_eos_without_wav_passes():
    row = _valid_generation_row()
    row["finish_reason"] = "sampler_error"
    row["wav_path"] = None
    validate_generation_row(row)


def test_generation_row_eos_requires_wav():
    row = _valid_generation_row()
    row["wav_path"] = None
    with pytest.raises(ValueError, match="wav"):
        validate_generation_row(row)


@pytest.mark.parametrize(
    "bad_path",
    [
        "/abs/emofilm/esd_0001_ang.wav",  # 前导 / 绝对路径
        "wav\\emofilm\\esd_0001_ang.wav",  # 反斜杠
        "C:\\foo\\bar.wav",  # Windows 盘符
        "C:/foo/bar.wav",  # Windows 盘符 POSIX 风格
    ],
)
def test_generation_row_rejects_non_posix_wav_path(bad_path):
    """(I1) wav_path 必须是 workspace-relative POSIX：拒绝绝对路径/反斜杠/盘符。"""
    row = _valid_generation_row()
    row["wav_path"] = bad_path
    with pytest.raises(ValueError, match="wav_path"):
        validate_generation_row(row)


def test_generation_row_accepts_relative_posix_wav_path():
    row = _valid_generation_row()
    row["wav_path"] = "wav/emofilm/esd_0001_ang.wav"
    validate_generation_row(row)


# -- Task 4 / #5：seed 校验（per-request 固定随机种子）------------------------

def test_generation_row_rejects_missing_seed():
    """seed 是必需字段：缺失 → 拒绝。"""
    row = _valid_generation_row()
    del row["seed"]
    with pytest.raises(ValueError, match="seed"):
        validate_generation_row(row)


def test_generation_row_rejects_negative_seed():
    """seed 必须非负 int。"""
    row = _valid_generation_row()
    row["seed"] = -1
    with pytest.raises(ValueError, match="seed"):
        validate_generation_row(row)


@pytest.mark.parametrize("bad", [3.14, "1984", True, None, [1]])
def test_generation_row_rejects_non_int_seed(bad):
    """seed 必须是 int（bool / float / str / None / list 全拒绝）。"""
    row = _valid_generation_row()
    row["seed"] = bad
    with pytest.raises(ValueError, match="seed"):
        validate_generation_row(row)


def test_generation_row_accepts_zero_seed():
    """seed=0 合法（非负 int）。"""
    row = _valid_generation_row()
    row["seed"] = 0
    validate_generation_row(row)


# ============================================================
# EvaluationRow + Aggregate（boundary_evidence_tier）
# ============================================================

