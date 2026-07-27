"""冻结声学 evaluator 包装 —— EmoFiLM v2 ticket 08。

本模块定义与训练任务头**隔离、冻结**的 emotion / arousal 评测器接口、
一个 CPU 确定性 Fake evaluator（供 09/10/12 测试用），以及纯函数形式的
方向性 / transition 定位 / arousal 方向校验逻辑。

设计依据（MAP §3 evaluator 不变量）：
- evaluator 冻结（``requires_grad=False``），不复用训练下游任务头；
- 记录标签空间 / 采样率 / 帧率 / 窗口 / 语义 / 限制 / 校准；
- 与 IEMOCAP 弱监督生成器共享模型 → 标自证风险
  （``shares_source_with_iemocap_weak_supervision=True``）；
- 未校准（``calibration=None``）时 score 不得命名 confidence。

**自证风险链**（详见 ``docs/contracts/emofilm_v2_evaluators.md``）：
emotion2vec-base 768d/50Hz 帧特征 → ``WordSequenceModel`` 分类/回归头 →
IEMOCAP 词级弱监督标签（句级广播）。因此用 emotion2vec+WordSequenceModel
做 evaluator 与 IEMOCAP 训练标签共享上游模型/特征来源，是同源验收。

真实 emotion2vec wrapper 为 best-effort：现有资产**不满足**独立 frame-level
emotion/arousal 门禁（WordSequenceModel 是 utterance-level 池化架构，逐帧
应用分类头超出训练分布）。09/10 消费 ``FakeAcousticEvaluator`` + 接口。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


# ============================================================
# 常量
# ============================================================

# 5 类情感标签空间（与 v2 schema 一致；evaluator 用 0-indexed 数组）。
EMOTION_LABEL_SPACE = ("ang", "hap", "neu", "sad", "sur")
EMOTION_LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABEL_SPACE)}

# Fake evaluator 身份常量。
FAKE_MODEL_ID = "fake-acoustic-evaluator"
FAKE_REVISION = "deterministic-v0"

# emotion2vec 门禁状态：当前资产不满足独立 frame-level 评测门禁。
EMOTION2VEC_GATE_STATUS = "not_met_frame_level"
EMOTION2VEC_GATE_REASON = (
    "emotion2vec-base produces 768d frame features; the only emotion/arousal "
    "classifier available (WordSequenceModel) is the same model used for IEMOCAP "
    "weak-supervision annotation (self-evidence) and is an utterance-level pooled "
    "architecture — per-frame application of its classification/regression head is "
    "outside its training distribution. An independent, calibrated frame-level "
    "emotion/arousal evaluator requires an external asset."
)

# output dict 中永远禁止出现的键名（MAP §3：confidence 不是合法字段）。
_BANNED_OUTPUT_KEYS = frozenset({"confidence"})


# ============================================================
# 合成参考片段
# ============================================================


@dataclass
class SyntheticReferenceClip:
    """合成参考片段 —— 携带已知属性供方向性校验逻辑使用。

    用于 ``validate_emotion_label_mapping``（``known_emotion``）、
    ``validate_transition_localization``（``known_transition_sec`` 等）、
    ``validate_arousal_direction``（``known_arousal_rank``）。
    """

    wav_path: str | Path
    duration_sec: float
    known_emotion: str | None = None
    known_transition_sec: float | None = None
    known_transition_from: str | None = None
    known_transition_to: str | None = None
    known_arousal_rank: int | None = None


# ============================================================
# 冻结 evaluator 接口（Protocol）
# ============================================================


@runtime_checkable
class EmotionEvaluator(Protocol):
    """冻结的逐帧情感评测器接口。

    实现者必须保证：
    - ``is_frozen`` 恒 True（包装不持有 trainable 任务头）；
    - ``identity()`` 返回完整身份记录（见 ``_IDENTITY_KEYS``）；
    - ``predict_frames()`` 返回逐帧情感分布，每帧和为 1，不含 ``confidence`` 键。
    """

    @property
    def is_frozen(self) -> bool: ...

    def identity(self) -> dict[str, Any]: ...

    def predict_frames(self, wav_path_or_clip: Any) -> dict[str, Any]: ...


@runtime_checkable
class ArousalEvaluator(Protocol):
    """冻结的逐帧 arousal 评测器接口。

    实现者必须保证：
    - ``is_frozen`` 恒 True；
    - ``identity()`` 返回完整身份记录；
    - ``predict_frames()`` 返回逐帧 arousal 标量轨迹，不含 ``confidence`` 键；
    - 未校准时 score 不得命名为强度真值。
    """

    @property
    def is_frozen(self) -> bool: ...

    def identity(self) -> dict[str, Any]: ...

    def predict_frames(self, wav_path_or_clip: Any) -> dict[str, Any]: ...


# identity() 必需键（brief 08 §A + 合同 Evaluator TypedDict 的并集）。
_IDENTITY_KEYS = frozenset({
    "model_id", "revision", "label_mapping", "sample_rate_hz",
    "frame_rate_hz", "window_strategy", "output_semantics",
    "known_limitations", "calibration",
    "shares_source_with_iemocap_weak_supervision",
    # 合同 Evaluator TypedDict（build_emofilm_contract.Evaluator）
    "name", "version", "label_space", "self_evidence_risk",
})


def assert_identity_complete(ident: dict[str, Any]) -> None:
    """校验 identity() 返回值包含所有必需键；缺失抛 ValueError。"""
    missing = _IDENTITY_KEYS - ident.keys()
    if missing:
        raise ValueError(
            f"evaluator identity() missing required keys: {sorted(missing)}"
        )
    for key in ("name", "version", "model_id"):
        val = ident.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"identity.{key} must be a non-empty string")
    shares = ident.get("shares_source_with_iemocap_weak_supervision")
    if not isinstance(shares, bool):
        raise ValueError(
            "shares_source_with_iemocap_weak_supervision must be bool"
        )


def assert_output_honest(output: dict[str, Any]) -> None:
    """拒绝输出 dict 中的禁止键名（confidence）。"""
    leaked = _BANNED_OUTPUT_KEYS & output.keys()
    if leaked:
        raise ValueError(
            f"predict_frames output must not contain {sorted(leaked)} "
            "(MAP §3: use raw_score + calibration, never 'confidence')"
        )


# ============================================================
# FakeAcousticEvaluator —— CPU 确定性合成 evaluator
# ============================================================


class FakeAcousticEvaluator:
    """确定性合成 evaluator（CPU，不加载真实模型）。

    根据 ``kind`` 决定输出 emotion 分布（``(T, 5)``）还是 arousal 轨迹
    （``(T,)``）。通过 ``SyntheticReferenceClip`` 的已知属性产生可控合成输出，
    供方向性 / transition / arousal 校验逻辑测试。

    合成规则（全部确定性，无随机种子）：
    - **emotion**：已知 ``known_emotion`` → 该列均值为 0.82，其余均分 0.18；
      已知 transition → 在 ``known_transition_sec`` 处切换主导列；
      无已知属性 → 均匀分布。
    - **arousal**：已知 ``known_arousal_rank`` → 均值线性映射到 [0.2, 0.8]；
      ``flat_arousal=True`` → 恒定 0.5（无区分度，用于测试 fail 路径）。

    可选 ``emotion_override``：将已知 emotion 名映射到不同列（测试 label mapping
    fail 路径）；``ignore_transition=True``：忽略 transition 信息（测试 transition
    fail 路径）。
    """

    def __init__(
        self,
        kind: Literal["emotion", "arousal"] = "emotion",
        *,
        frame_rate_hz: float = 50.0,
        sample_rate_hz: int = 16000,
        emotion_override: dict[str, str] | None = None,
        ignore_transition: bool = False,
        flat_arousal: bool = False,
    ):
        if kind not in ("emotion", "arousal"):
            raise ValueError(f"kind must be 'emotion' or 'arousal', got {kind!r}")
        self._kind = kind
        self._frame_rate_hz = float(frame_rate_hz)
        self._sample_rate_hz = int(sample_rate_hz)
        self._emotion_override = dict(emotion_override) if emotion_override else {}
        self._ignore_transition = bool(ignore_transition)
        self._flat_arousal = bool(flat_arousal)

    # ---- 接口方法 ----

    @property
    def is_frozen(self) -> bool:
        return True

    def identity(self) -> dict[str, Any]:
        ident: dict[str, Any] = {
            # 合同 Evaluator TypedDict 字段
            "name": f"{FAKE_MODEL_ID}-{self._kind}",
            "version": FAKE_REVISION,
            "label_space": list(EMOTION_LABEL_SPACE) if self._kind == "emotion" else [],
            "sample_rate_hz": self._sample_rate_hz,
            "frame_rate_hz": self._frame_rate_hz,
            "self_evidence_risk": False,
            # brief 08 §A 扩展字段
            "model_id": FAKE_MODEL_ID,
            "revision": FAKE_REVISION,
            "label_mapping": (
                dict(EMOTION_LABEL_TO_IDX) if self._kind == "emotion" else None
            ),
            "window_strategy": "synthetic_fixed_hop_deterministic",
            "output_semantics": (
                "deterministic synthetic emotion distribution from known clip properties"
                if self._kind == "emotion"
                else "deterministic synthetic arousal trajectory from known clip properties"
            ),
            "known_limitations": [
                "fake evaluator; not a real model",
                "output is derived from clip metadata, not audio content",
            ],
            "calibration": None,
            "shares_source_with_iemocap_weak_supervision": False,
        }
        assert_identity_complete(ident)
        return ident

    def predict_frames(self, wav_path_or_clip: Any) -> dict[str, Any]:
        clip = _coerce_clip(wav_path_or_clip)
        n_frames = max(1, int(round(clip.duration_sec * self._frame_rate_hz)))
        times = np.arange(n_frames, dtype=np.float64) / self._frame_rate_hz

        if self._kind == "emotion":
            frames = self._synthesize_emotion(clip, n_frames, times)
            output: dict[str, Any] = {
                "frames": frames,
                "frame_rate_hz": self._frame_rate_hz,
                "times_sec": times,
                "label_space": list(EMOTION_LABEL_SPACE),
            }
        else:
            frames = self._synthesize_arousal(clip, n_frames, times)
            output = {
                "frames": frames,
                "frame_rate_hz": self._frame_rate_hz,
                "times_sec": times,
            }
        assert_output_honest(output)
        return output

    # ---- 合成逻辑 ----

    def _resolve_emotion(self, emotion: str) -> str:
        return self._emotion_override.get(emotion, emotion)

    def _synthesize_emotion(
        self,
        clip: SyntheticReferenceClip,
        n_frames: int,
        times: np.ndarray,
    ) -> np.ndarray:
        """确定性合成逐帧情感分布 (T, 5)。"""
        frames = np.full(
            (n_frames, len(EMOTION_LABEL_SPACE)), 0.2, dtype=np.float64
        )

        has_transition = (
            not self._ignore_transition
            and clip.known_transition_sec is not None
            and clip.known_transition_from is not None
            and clip.known_transition_to is not None
        )
        if has_transition:
            from_emotion = self._resolve_emotion(clip.known_transition_from)
            to_emotion = self._resolve_emotion(clip.known_transition_to)
            from_idx = EMOTION_LABEL_TO_IDX[from_emotion]
            to_idx = EMOTION_LABEL_TO_IDX[to_emotion]
            transition_frame = int(
                round(clip.known_transition_sec * self._frame_rate_hz)
            )
            transition_frame = max(1, min(transition_frame, n_frames - 1))
            for t in range(n_frames):
                dominant = from_idx if t < transition_frame else to_idx
                frames[t, :] = 0.04  # 0.04 * 5 = 0.20（非主导维度）
                frames[t, dominant] = 0.80
            return frames

        if clip.known_emotion is not None:
            emotion = self._resolve_emotion(clip.known_emotion)
            if emotion in EMOTION_LABEL_TO_IDX:
                idx = EMOTION_LABEL_TO_IDX[emotion]
                frames[:, :] = 0.045  # 0.045 * 4 = 0.18
                frames[:, idx] = 0.82
            return frames

        # 无已知属性 → 均匀分布
        frames[:, :] = 0.2
        return frames

    def _synthesize_arousal(
        self,
        clip: SyntheticReferenceClip,
        n_frames: int,
        times: np.ndarray,
    ) -> np.ndarray:
        """确定性合成逐帧 arousal 轨迹 (T,)。"""
        if self._flat_arousal:
            return np.full(n_frames, 0.5, dtype=np.float64)

        if clip.known_arousal_rank is not None:
            # rank 0 → 0.2, rank 1 → 0.5, rank 2 → 0.8（线性映射）
            rank = max(0, clip.known_arousal_rank)
            base = 0.2 + 0.3 * rank
            # 确定性微扰（基于帧索引，非随机）
            perturbation = 0.02 * np.sin(np.arange(n_frames) * 0.1)
            return np.clip(
                np.full(n_frames, base, dtype=np.float64) + perturbation,
                0.0,
                1.0,
            )

        # 无已知属性 → 中等 arousal + 确定性微扰
        base = 0.5
        perturbation = 0.02 * np.sin(np.arange(n_frames) * 0.1)
        return np.clip(base + perturbation, 0.0, 1.0)


# ============================================================
# 纯函数：方向性 / transition / arousal 校验逻辑
# ============================================================


def validate_emotion_label_mapping(
    evaluator: EmotionEvaluator,
    reference_clips: list[SyntheticReferenceClip],
) -> dict[str, Any]:
    """在已知 emotion 参考片段上验证类别映射与基本方向性。

    对每个 ``known_emotion`` 非空的片段调用 ``evaluator.predict_frames()``，
    取帧均值后 argmax，检查与已知标签是否一致。不止凭模型名。

    Returns:
        ``{passed: bool, n_total, n_passed, details: [...]}``
    """
    label_space = evaluator.identity().get("label_space") or list(EMOTION_LABEL_SPACE)
    details = []
    n_total = 0
    n_passed = 0

    for clip in reference_clips:
        if clip.known_emotion is None:
            continue
        n_total += 1
        output = evaluator.predict_frames(clip)
        frames = np.asarray(output["frames"])
        if frames.ndim != 2 or frames.shape[1] != len(label_space):
            details.append({
                "wav_path": str(clip.wav_path),
                "known_emotion": clip.known_emotion,
                "predicted": None,
                "passed": False,
                "error": f"frame shape {frames.shape} != label_space {len(label_space)}",
            })
            continue
        mean_dist = frames.mean(axis=0)
        # 有限性门禁（Ticket #12）：全 NaN / 含 inf / 空 frames 会让
        # ``np.argmax`` 退化到首类（返回 0），若 known_emotion 恰为
        # label_space[0] 会误通过校准。在 argmax 前拦截。
        if mean_dist.size == 0 or not np.isfinite(mean_dist).all():
            details.append({
                "wav_path": str(clip.wav_path),
                "known_emotion": clip.known_emotion,
                "predicted": None,
                "passed": False,
                "error": "non-finite or empty distribution",
            })
            continue
        argmax_idx = int(np.argmax(mean_dist))
        predicted = label_space[argmax_idx]
        ok = predicted == clip.known_emotion
        if ok:
            n_passed += 1
        details.append({
            "wav_path": str(clip.wav_path),
            "known_emotion": clip.known_emotion,
            "predicted": predicted,
            "passed": ok,
            "mean_distribution": mean_dist.tolist(),
        })

    return {
        "passed": n_total > 0 and n_passed == n_total,
        "n_total": n_total,
        "n_passed": n_passed,
        "details": details,
    }


def validate_transition_localization(
    evaluator: EmotionEvaluator,
    reference_clips: list[SyntheticReferenceClip],
    *,
    tolerance_sec: float = 0.5,
) -> dict[str, Any]:
    """验证逐帧输出能定位已知 transition 并量化时间偏差。

    检测方法：对每帧取 argmax label，找到 label 从 ``known_transition_from``
    切换到 ``known_transition_to`` 的首个帧索引，换算为秒，与
    ``known_transition_sec`` 比较偏差。

    Args:
        evaluator: emotion evaluator（需有 label_space）。
        reference_clips: 带 ``known_transition_sec`` 的片段列表。
        tolerance_sec: 允许的时间偏差（秒）。

    Returns:
        ``{passed: bool, n_total, n_passed, details: [...]}``
    """
    label_space = evaluator.identity().get("label_space") or list(EMOTION_LABEL_SPACE)
    label_to_idx = {lab: i for i, lab in enumerate(label_space)}
    details = []
    n_total = 0
    n_passed = 0

    for clip in reference_clips:
        if clip.known_transition_sec is None:
            continue
        if (
            clip.known_transition_from is None
            or clip.known_transition_to is None
        ):
            continue
        n_total += 1
        output = evaluator.predict_frames(clip)
        frames = np.asarray(output["frames"])
        frame_rate = float(output.get("frame_rate_hz", 50.0))
        per_frame_label = np.argmax(frames, axis=1)

        from_idx = label_to_idx.get(clip.known_transition_from)
        to_idx = label_to_idx.get(clip.known_transition_to)
        if from_idx is None or to_idx is None:
            details.append({
                "wav_path": str(clip.wav_path),
                "known_transition_sec": clip.known_transition_sec,
                "detected_transition_sec": None,
                "passed": False,
                "error": f"label {clip.known_transition_from}/{clip.known_transition_to} not in label_space",
            })
            continue

        # 找首个 from→to 切换帧
        detected_frame = None
        for t in range(1, len(per_frame_label)):
            if per_frame_label[t - 1] == from_idx and per_frame_label[t] == to_idx:
                detected_frame = t
                break
        # 退化检测：若没有严格相邻切换，找首个 == to_idx 的帧
        if detected_frame is None:
            to_frames = np.where(per_frame_label == to_idx)[0]
            if len(to_frames) > 0:
                detected_frame = int(to_frames[0])

        if detected_frame is None:
            details.append({
                "wav_path": str(clip.wav_path),
                "known_transition_sec": clip.known_transition_sec,
                "detected_transition_sec": None,
                "passed": False,
                "error": "no transition detected in per-frame argmax",
            })
            continue

        detected_sec = detected_frame / frame_rate
        deviation = abs(detected_sec - clip.known_transition_sec)
        ok = deviation <= tolerance_sec
        if ok:
            n_passed += 1
        details.append({
            "wav_path": str(clip.wav_path),
            "known_transition_sec": clip.known_transition_sec,
            "detected_transition_sec": detected_sec,
            "deviation_sec": deviation,
            "tolerance_sec": tolerance_sec,
            "passed": ok,
        })

    return {
        "passed": n_total > 0 and n_passed == n_total,
        "n_total": n_total,
        "n_passed": n_passed,
        "details": details,
    }


def validate_arousal_direction(
    evaluator: ArousalEvaluator,
    reference_clips: list[SyntheticReferenceClip],
) -> dict[str, Any]:
    """验证 arousal 在可排序参考上单调。

    按 ``known_arousal_rank`` 排序，检查各 rank 的均值 arousal 是否单调递增。
    若任意两个相邻 rank 的均值差 <= 0（无区分度）→ fail。

    未校准 evaluator 不得命名强度真值：本函数仅检验**方向性**
    （高 rank → 高均值），不输出绝对真值标签。

    Returns:
        ``{passed: bool, details: {mean_arousal_by_rank, ...}}``
    """
    rank_means: dict[int, float] = {}
    for clip in reference_clips:
        if clip.known_arousal_rank is None:
            continue
        output = evaluator.predict_frames(clip)
        frames = np.asarray(output["frames"])
        mean_val = float(np.mean(frames)) if frames.size > 0 else 0.0
        if clip.known_arousal_rank not in rank_means:
            rank_means[clip.known_arousal_rank] = []
        rank_means[clip.known_arousal_rank].append(mean_val)

    if len(rank_means) < 2:
        return {
            "passed": False,
            "details": {
                "mean_arousal_by_rank": {},
                "error": "need >= 2 distinct arousal ranks to verify direction",
            },
        }

    sorted_ranks = sorted(rank_means.keys())
    mean_by_rank = {
        rank: float(np.mean(rank_means[rank])) for rank in sorted_ranks
    }
    values = [mean_by_rank[r] for r in sorted_ranks]
    monotonic = all(
        values[i] < values[i + 1] for i in range(len(values) - 1)
    )

    return {
        "passed": bool(monotonic),
        "details": {
            "mean_arousal_by_rank": mean_by_rank,
            "sorted_ranks": sorted_ranks,
            "monotonic_increasing": bool(monotonic),
        },
    }


# ============================================================
# 辅助函数
# ============================================================


def _coerce_clip(wav_path_or_clip: Any) -> SyntheticReferenceClip:
    """将多种输入统一为 SyntheticReferenceClip。"""
    if isinstance(wav_path_or_clip, SyntheticReferenceClip):
        return wav_path_or_clip
    if isinstance(wav_path_or_clip, (str, Path)):
        # 无元数据：返回默认 clip（均匀分布 / 中等 arousal）
        return SyntheticReferenceClip(
            wav_path=wav_path_or_clip, duration_sec=2.0
        )
    # 尝试作为 (waveform, sample_rate) 元组
    if isinstance(wav_path_or_clip, (tuple, list)) and len(wav_path_or_clip) == 2:
        wav, sr = wav_path_or_clip
        if hasattr(wav, "__len__") and isinstance(sr, (int, float)) and sr > 0:
            duration = len(wav) / float(sr)
            return SyntheticReferenceClip(
                wav_path="<tensor>", duration_sec=duration
            )
    raise TypeError(
        f"predict_frames expects SyntheticReferenceClip, str/Path, or "
        f"(waveform, sample_rate); got {type(wav_path_or_clip).__name__}"
    )


# ============================================================
# best-effort: Emotion2Vec 包装（不满足门禁；见模块 docstring）
# ============================================================


class _Emotion2VecBaseWrapper:
    """emotion2vec-base 帧特征提取的共享包装逻辑。

    **不满足 frame-level 门禁**：emotion2vec-base 只输出 768d 原始特征，
    不直接输出 emotion/arousal。分类/回归需要下游头（WordSequenceModel），
    而该头是 IEMOCAP 弱监督标注器的同一模型 → 自证风险。

    本类提供 ``_extract_frame_features()`` 供子类复用；子类负责将 768d 特征
    映射到 emotion 分布或 arousal 轨迹。**构造时不加载模型**；模型对象由
    调用方注入（dependency injection），避免 import 时拉起 fairseq/funasr。
    """

    MODEL_ID = "emotion2vec-base"
    REVISION = "emofilm-frozen-v1"
    FEATURE_DIM = 768
    FRAME_RATE_HZ = 50.0
    FRAME_STEP_MS = 20.0
    SAMPLE_RATE_HZ = 16000
    WINDOW_STRATEGY = "sliding_20ms_hop_no_overlap_extract_features"

    def __init__(
        self,
        model: Any | None = None,
        task: Any | None = None,
        *,
        checkpoint_sha256: str | None = None,
    ):
        self._model = model
        self._task = task
        self._checkpoint_sha256 = checkpoint_sha256

    @property
    def is_frozen(self) -> bool:
        if self._model is None:
            return True
        # 检查模型是否处于 eval 模式且无 requires_grad
        try:
            import torch.nn as nn
            if isinstance(self._model, nn.Module):
                if self._model.training:
                    return False
                return not any(p.requires_grad for p in self._model.parameters())
        except Exception:
            pass
        return True

    def _extract_frame_features(self, wav_path: str | Path) -> np.ndarray:
        """提取 768d/50Hz 帧特征（需要 model+task 已注入）。"""
        if self._model is None or self._task is None:
            raise RuntimeError(
                "emotion2vec model/task not injected; construct with "
                "model=<loaded>, task=<loaded> to use real extraction"
            )
        import torch
        import soundfile as sf
        import torch.nn.functional as F

        wav, actual_rate = sf.read(str(wav_path), dtype="float32")
        if actual_rate != self.SAMPLE_RATE_HZ:
            raise ValueError(
                f"sample rate mismatch: expected {self.SAMPLE_RATE_HZ}, "
                f"got {actual_rate}"
            )
        if wav.ndim == 2:
            wav = wav[:, 0]
        source = torch.from_numpy(np.asarray(wav)).float().view(1, -1)
        if getattr(getattr(self._task, "cfg", None), "normalize", False):
            source = F.layer_norm(source, source.shape)
        with torch.no_grad():
            result = self._model.extract_features(source, padding_mask=None)
        features = result["x"]
        if features.ndim == 3:
            features = features.squeeze(0)
        return features.detach().cpu().float().numpy()

    def _base_identity(self) -> dict[str, Any]:
        return {
            "model_id": self.MODEL_ID,
            "revision": self.REVISION,
            "sample_rate_hz": self.SAMPLE_RATE_HZ,
            "frame_rate_hz": self.FRAME_RATE_HZ,
            "window_strategy": self.WINDOW_STRATEGY,
            "calibration": None,
            "shares_source_with_iemocap_weak_supervision": True,
            "self_evidence_risk": True,
            "known_limitations": [
                EMOTION2VEC_GATE_REASON,
                "per-frame classification/regression head application is outside "
                "WordSequenceModel training distribution (utterance-level pooling)",
                "uncalibrated; raw_score not named confidence",
            ],
        }


class Emotion2VecEmotionEvaluator(_Emotion2VecBaseWrapper):
    """best-effort: emotion2vec-base + WordSequenceModel 分类头 → 逐帧 emotion 分布。

    **不满足独立 emotion 评测门禁**（见模块 docstring + ``EMOTION2VEC_GATE_REASON``）。
    WordSequenceModel 的 ``classification_head`` 在训练时作用于 **mean-pooled**
    utterance 表示；此处逐帧应用是 best-effort 近似，超出训练分布。

    使用方法（需外部注入模型）::

        model, task = load_emotion2vec(upstream_dir, checkpoint, device)
        wsm = WordSequenceModel()  # 加载冻结 checkpoint
        evaluator = Emotion2VecEmotionEvaluator(
            emotion2vec_model=model, emotion2vec_task=task,
            classifier=wsm,
        )
    """

    def __init__(
        self,
        *,
        emotion2vec_model: Any | None = None,
        emotion2vec_task: Any | None = None,
        classifier: Any | None = None,
        checkpoint_sha256: str | None = None,
    ):
        super().__init__(
            model=emotion2vec_model,
            task=emotion2vec_task,
            checkpoint_sha256=checkpoint_sha256,
        )
        self._classifier = classifier

    def identity(self) -> dict[str, Any]:
        ident = self._base_identity()
        ident.update({
            "name": f"{self.MODEL_ID}+word_sequence_model-emotion",
            "version": self.REVISION,
            "label_space": list(EMOTION_LABEL_SPACE),
            "label_mapping": dict(EMOTION_LABEL_TO_IDX),
            "output_semantics": (
                "per-frame softmax of WordSequenceModel.classification_head "
                "applied to raw 768d features (outside training distribution; "
                "utterance-level pooled in training)"
            ),
            "gate_status": EMOTION2VEC_GATE_STATUS,
        })
        assert_identity_complete(ident)
        return ident

    def predict_frames(self, wav_path_or_clip: Any) -> dict[str, Any]:
        if self._classifier is None:
            raise RuntimeError(
                "classifier (WordSequenceModel) not injected; cannot predict"
            )
        clip = _coerce_clip(wav_path_or_clip)
        features = self._extract_frame_features(clip.wav_path)  # (T, 768)
        import torch
        x = torch.from_numpy(features).unsqueeze(0)  # (1, T, 768)
        with torch.no_grad():
            logits, _ = self._classifier(x)  # (1, 5)
        # 注意：分类头训练时作用于 pooled，此处逐帧应用
        # 需要手动调用分类头 per-frame
        cls_head = self._classifier.classification_head
        logits = cls_head(x).squeeze(0)  # (T, 5)
        probs = torch.softmax(logits, dim=-1).numpy()
        times = np.arange(probs.shape[0], dtype=np.float64) / self.FRAME_RATE_HZ
        output = {
            "frames": probs,
            "frame_rate_hz": self.FRAME_RATE_HZ,
            "times_sec": times,
            "label_space": list(EMOTION_LABEL_SPACE),
        }
        assert_output_honest(output)
        return output


class Emotion2VecArousalEvaluator(_Emotion2VecBaseWrapper):
    """best-effort: emotion2vec-base + WordSequenceModel 回归头 → 逐帧 arousal。

    **不满足独立 arousal 评测门禁**。WordSequenceModel 的 ``regression_head``
    输出 3D VAD（含 arousal = VAD[1]），训练时同样作用于 pooled 表示。
    逐帧应用超出训练分布，且与 IEMOCAP 弱监督标注器同源 → 自证风险。
    """

    def __init__(
        self,
        *,
        emotion2vec_model: Any | None = None,
        emotion2vec_task: Any | None = None,
        regressor: Any | None = None,
        checkpoint_sha256: str | None = None,
    ):
        super().__init__(
            model=emotion2vec_model,
            task=emotion2vec_task,
            checkpoint_sha256=checkpoint_sha256,
        )
        self._regressor = regressor

    def identity(self) -> dict[str, Any]:
        ident = self._base_identity()
        ident.update({
            "name": f"{self.MODEL_ID}+word_sequence_model-arousal",
            "version": self.REVISION,
            "label_space": [],
            "label_mapping": None,
            "output_semantics": (
                "per-frame WordSequenceModel.regression_head VAD[1] (arousal) "
                "applied to raw 768d features (outside training distribution; "
                "utterance-level pooled in training)"
            ),
            "gate_status": EMOTION2VEC_GATE_STATUS,
        })
        assert_identity_complete(ident)
        return ident

    def predict_frames(self, wav_path_or_clip: Any) -> dict[str, Any]:
        if self._regressor is None:
            raise RuntimeError(
                "regressor (WordSequenceModel) not injected; cannot predict"
            )
        clip = _coerce_clip(wav_path_or_clip)
        features = self._extract_frame_features(clip.wav_path)  # (T, 768)
        import torch
        x = torch.from_numpy(features).unsqueeze(0)  # (1, T, 768)
        reg_head = self._regressor.regression_head
        with torch.no_grad():
            vad = reg_head(x).squeeze(0).numpy()  # (T, 3)
        arousal = vad[:, 1]  # VAD[1] = arousal
        times = np.arange(arousal.shape[0], dtype=np.float64) / self.FRAME_RATE_HZ
        output = {
            "frames": arousal,
            "frame_rate_hz": self.FRAME_RATE_HZ,
            "times_sec": times,
        }
        assert_output_honest(output)
        return output


# ============================================================
# 外部异源 SER 裁判（ticket 04 / B8）：消除自证风险（决策3 A）
# ============================================================


class ExternalSerEmotionEvaluator:
    """外部异源 SER 裁判：wav2vec2-large-xlsr fine-tuned on speech emotion。

    与 EmoFiLM 训练数据**无关**（不基于 emotion2vec / WordSequenceModel），故
    ``self_evidence_risk=False``——消除决策3 所述"出题兼阅卷"的自证风险。
    模型为 utterance-level 分类；本包装用**滑窗**（窗 ``window_sec``、步
    ``hop_sec``）逐窗推理，把每窗的类别分布赋给该窗覆盖的帧，产出 frame-level
    5 类分布（供 ``evaluate_spans_from_frames`` 切前后段判 emo_from/emo_to）。

    外部标签空间 → schema 5 类映射（丢弃 calm/disgust/fear 等非目标类，重归一化）。
    """

    # 外部标签 → schema 5 类（ang/hap/neu/sad/sur）。非映射类（calm/disgust/fear）丢弃。
    _EXTERNAL_TO_5 = {
        "angry": "ang", "happy": "hap", "neutral": "neu", "sad": "sad",
        "surprised": "sur", "surprise": "sur",
    }

    def __init__(
        self,
        model_id: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device: str = "cuda",
        window_sec: float = 1.0,
        hop_sec: float = 0.2,
        frame_rate_hz: float = 50.0,
        sample_rate_hz: int = 16000,
    ):
        import torch
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

        self._torch = torch
        self._model = AutoModelForAudioClassification.from_pretrained(model_id)
        self._model.to(device).eval()
        self._fe = AutoFeatureExtractor.from_pretrained(model_id)
        self._device = device
        self._window_sec = float(window_sec)
        self._hop_sec = float(hop_sec)
        self._frame_rate_hz = float(frame_rate_hz)
        self._sample_rate_hz = int(sample_rate_hz)
        self._model_id = model_id
        # 外部类别 id → schema 5 类 idx（None = 丢弃）
        id2label = self._model.config.id2label
        self._ext_to_5: list[int | None] = []
        for ext_idx in range(len(id2label)):
            five = self._EXTERNAL_TO_5.get(str(id2label[ext_idx]).lower())
            self._ext_to_5.append(EMOTION_LABEL_TO_IDX[five] if five else None)
        self._n_5 = len(EMOTION_LABEL_SPACE)

    @property
    def is_frozen(self) -> bool:
        return True

    def identity(self) -> dict[str, Any]:
        ident: dict[str, Any] = {
            "name": f"external-ser-{self._model_id.split('/')[-1]}",
            "version": "external-v0",
            "label_space": list(EMOTION_LABEL_SPACE),
            "sample_rate_hz": self._sample_rate_hz,
            "frame_rate_hz": self._frame_rate_hz,
            "self_evidence_risk": False,  # 异源（决策3 A 核心）
            "model_id": self._model_id,
            "revision": "external-hub",
            "label_mapping": dict(EMOTION_LABEL_TO_IDX),
            "window_strategy": f"sliding_window_{self._window_sec}s_hop_{self._hop_sec}s",
            "output_semantics": (
                "external wav2vec2-large-xlsr SER logits mapped to 5-class, "
                "applied per sliding window → frame-level distribution"
            ),
            "known_limitations": [
                "utterance-level SER applied per sliding window (not native frame-level)",
                "external label space subset-mapped to 5 classes (calm/disgust/fear dropped)",
            ],
            "calibration": None,
            "shares_source_with_iemocap_weak_supervision": False,
        }
        assert_identity_complete(ident)
        return ident

    def predict_frames(self, wav_path_or_clip: Any) -> dict[str, Any]:
        import torchaudio

        clip = _coerce_clip(wav_path_or_clip)
        wav, sr = torchaudio.load(str(clip.wav_path))
        if sr != self._sample_rate_hz:
            wav = torchaudio.functional.resample(wav, sr, self._sample_rate_hz)
        audio = wav[0].cpu().numpy()
        total_sec = len(audio) / self._sample_rate_hz
        win_n = int(self._window_sec * self._sample_rate_hz)

        n_frames = max(1, int(round(total_sec * self._frame_rate_hz)))
        times = np.arange(n_frames, dtype=np.float64) / self._frame_rate_hz
        frames = np.zeros((n_frames, self._n_5), dtype=np.float32)

        # 滑窗：每窗跑 SER → 8 类 softmax → 5 类映射 → 赋给窗覆盖的帧区间
        start = 0.0
        while start + self._window_sec <= total_sec + 1e-6:
            s_n = int(start * self._sample_rate_hz)
            e_n = min(s_n + win_n, len(audio))
            if e_n - s_n >= int(0.1 * self._sample_rate_hz):
                chunk = audio[s_n:e_n]
                inp = self._fe(chunk, sampling_rate=self._sample_rate_hz, return_tensors="pt")
                inp = {k: v.to(self._device) for k, v in inp.items()}
                with self._torch.no_grad():
                    logits = self._model(**inp).logits[0]
                probs = self._torch.softmax(logits, dim=0).cpu().numpy()
                five = np.zeros(self._n_5, dtype=np.float32)
                for ext_idx, five_idx in enumerate(self._ext_to_5):
                    if five_idx is not None:
                        five[five_idx] += float(probs[ext_idx])
                s_total = float(five.sum())
                if s_total > 0:
                    five = five / s_total
                else:
                    five = np.ones(self._n_5, dtype=np.float32) / self._n_5
                # 赋给 [start, start+hop) 的帧（窗中心代表区）
                f0 = max(0, int(start * self._frame_rate_hz))
                f1 = min(n_frames, int((start + self._hop_sec) * self._frame_rate_hz) + 1)
                frames[f0:f1] = five
            start += self._hop_sec

        # 未覆盖帧（开头/结尾不足一窗）→ 均匀分布（不伪造某类）
        uncovered = frames.sum(axis=1) == 0
        if uncovered.any():
            frames[uncovered] = np.ones(self._n_5, dtype=np.float32) / self._n_5

        output = {
            "frames": frames,
            "frame_rate_hz": self._frame_rate_hz,
            "times_sec": times,
            "label_space": list(EMOTION_LABEL_SPACE),
        }
        assert_output_honest(output)
        return output


__all__ = [
    # 常量
    "EMOTION_LABEL_SPACE",
    "EMOTION_LABEL_TO_IDX",
    "FAKE_MODEL_ID",
    "FAKE_REVISION",
    "EMOTION2VEC_GATE_STATUS",
    "EMOTION2VEC_GATE_REASON",
    # 外部异源裁判（ticket 04 / B8）
    "ExternalSerEmotionEvaluator",
    # 接口
    "EmotionEvaluator",
    "ArousalEvaluator",
    # Fake 伪评测器
    "FakeAcousticEvaluator",
    "SyntheticReferenceClip",
    # 校验逻辑
    "validate_emotion_label_mapping",
    "validate_transition_localization",
    "validate_arousal_direction",
    # best-effort 真实包装
    "Emotion2VecEmotionEvaluator",
    "Emotion2VecArousalEvaluator",
    # 工具
    "assert_identity_complete",
    "assert_output_honest",
]
