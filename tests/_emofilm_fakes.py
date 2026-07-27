"""EmoFiLM v2 共享测试替身（test-only）。

本模块集中存放 v2 评测链路中供 CPU 合同 / 行为测试使用的确定性替身，
避免将 test-only 实现耦合到生产模块（ADR-0014：只锁外部行为与 schema，
不锁私有函数名 / 层数 / 内部张量顺序）。

内容：
- ``FakeForcedAligner``：确定性合成强制对齐器（CPU，不调用 MFA）。
  原先定义于 ``eval/eval_local_control.py``，仅被测试引用。
- ``FakeWerEvaluator``：确定性合成 ASR 转写器（CPU，不加载真实模型）。
  原先定义于 ``eval/triplet_eval.py``，仅被测试引用。
- ``ClipMappedEvaluator``：将 utt_id 映射到 ``SyntheticReferenceClip`` 的
  evaluator 包装器。复用 ``acoustic_evaluators._coerce_clip`` 做输入归一化，
  不重复 coercion 逻辑。原先以 ``_ClipMappedEvaluator``（私有符号）形式
  定义于 ``eval/eval_local_control.py`` 并被 12+ 测试文件导入——此处改为
  测试辅助模块的公开 API。

生产模块的真实类（``ForcedAligner`` Protocol、``WerEvaluator`` Protocol、
``MfaForcedAligner``、``Emotion2Vec*Evaluator``）保留在各自生产模块中不动。

所有输出确定性生成，无随机种子；不加载真实模型、不调用真实 ASR/MFA。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from eval.acoustic_evaluators import SyntheticReferenceClip, _coerce_clip
from eval.eval_local_control import AlignmentResult, WordBoundary


# ============================================================
# FakeForcedAligner（原 eval/eval_local_control.py）
# ============================================================


class FakeForcedAligner:
    """确定性合成强制对齐器（CPU，不调用 MFA）。

    供 09 / 12 测试使用。行为：
    - 若 ``always_fail=True``：始终返回 ``status="failed"``。
    - 若 ``boundaries_by_utt`` 注册了 utt_id → [(start, end, word), ...]：
      按 utt_id 返回对应边界（从 wav_path 中提取 utt_id stem）。
    - 否则：按 ``text`` 词数均匀分割 ``default_duration``。

    所有时间戳确定性生成，无随机种子。
    """

    def __init__(
        self,
        *,
        boundaries_by_utt: dict[str, list[tuple[float, float, str]]] | None = None,
        default_duration: float = 3.0,
        always_fail: bool = False,
    ):
        self._boundaries = dict(boundaries_by_utt) if boundaries_by_utt else {}
        self._default_duration = float(default_duration)
        self._always_fail = bool(always_fail)

    def align(self, wav_path: str, text: str) -> AlignmentResult:
        if self._always_fail:
            return AlignmentResult(status="failed", reason="fake_aligner_forced_fail")

        utt_id = Path(wav_path).stem
        if utt_id in self._boundaries:
            words = [
                WordBoundary(start_sec=s, end_sec=e, word=w)
                for s, e, w in self._boundaries[utt_id]
            ]
            return AlignmentResult(status="aligned", words=words)

        # 均匀分割默认时长
        tokens = text.split()
        if not tokens:
            return AlignmentResult(
                status="failed", reason="empty_text",
            )
        n = len(tokens)
        step = self._default_duration / n
        words = [
            WordBoundary(
                start_sec=i * step,
                end_sec=(i + 1) * step,
                word=tokens[i],
            )
            for i in range(n)
        ]
        return AlignmentResult(status="aligned", words=words)


# ============================================================
# FakeWerEvaluator（原 eval/triplet_eval.py）
# ============================================================


class FakeWerEvaluator:
    """确定性合成 ASR 转写器（CPU，不加载真实模型）。

    供 10 / 12 测试使用。行为：
    - 若 ``hypotheses`` 注册了 utt_id → text：按 wav_path stem 取 utt_id，
      返回对应 hypothesis_text（用于测试 WER 退化场景）。
    - 否则：返回 ``{"hypothesis_text": ""}``，WER 由调用方与参考文本计算
      （空 hypothesis + 非空 reference → WER=1.0）。

    所有输出确定性生成，无随机种子。
    """

    def __init__(
        self,
        *,
        hypotheses: dict[str, str] | None = None,
        default_hypothesis: str = "",
    ):
        self._hypotheses = dict(hypotheses) if hypotheses else {}
        self._default = default_hypothesis

    @property
    def is_frozen(self) -> bool:
        return True

    def identity(self) -> dict[str, Any]:
        return {
            "name": "fake-wer-evaluator",
            "version": "deterministic-v0",
            "label_space": [],
            "sample_rate_hz": 16000,
            "frame_rate_hz": None,
            "self_evidence_risk": False,
            "model_id": "fake-wer-evaluator",
            "revision": "deterministic-v0",
            "label_mapping": None,
            "window_strategy": "synthetic_lookup",
            "output_semantics": (
                "deterministic synthetic hypothesis from per-utt lookup table"
            ),
            "known_limitations": [
                "fake evaluator; not a real ASR model",
                "output is derived from injected hypotheses, not audio content",
            ],
            "calibration": None,
            "shares_source_with_iemocap_weak_supervision": False,
        }

    def transcribe(self, wav_path: Any) -> dict[str, Any]:
        utt_id = Path(str(wav_path)).stem
        text = self._hypotheses.get(utt_id, self._default)
        return {"hypothesis_text": text, "status": "ok"}


# ============================================================
# ClipMappedEvaluator（原 eval/eval_local_control.py:_ClipMappedEvaluator）
# ============================================================


class ClipMappedEvaluator:
    """将 utt_id 映射到 SyntheticReferenceClip 的 evaluator 包装器。

    供 09 / 10 / 12 测试使用：FakeAcousticEvaluator 从 SyntheticReferenceClip
    的已知属性合成 frame 轨迹。本包装器维护 ``{utt_id: clip}`` 注册表，
    根据传入的 wav_path 推断 utt_id（取 stem），返回对应 clip 的合成输出。

    复用 ``acoustic_evaluators._coerce_clip`` 做输入归一化（单一 coercion
    真理源），不重复 isinstance / Path stem 提取等逻辑。当 wav_path 的
    utt_id 未在注册表中时，由 ``_coerce_clip`` 生成默认 clip 并透传给
    内层 evaluator（与直接调用 evaluator.predict_frames(wav_path) 等价）。

    生产代码不使用此类——真实 evaluator 直接读 WAV 文件。
    """

    def __init__(self, evaluator: Any, clip_map: dict[str, Any]):
        self._evaluator = evaluator
        self._clip_map = dict(clip_map)

    @property
    def is_frozen(self) -> bool:
        return self._evaluator.is_frozen

    def identity(self) -> dict[str, Any]:
        return self._evaluator.identity()

    def predict_frames(self, wav_path_or_clip: Any) -> dict[str, Any]:
        # SyntheticReferenceClip 直接透传.
        if isinstance(wav_path_or_clip, SyntheticReferenceClip):
            clip = wav_path_or_clip
        else:
            # 从 wav_path 提取 utt_id stem 查注册表；
            # 未命中则由 _coerce_clip 归一化（与内层 evaluator 行为一致）.
            utt_id = Path(str(wav_path_or_clip)).stem
            clip = self._clip_map.get(utt_id)
            if clip is None:
                clip = _coerce_clip(wav_path_or_clip)
        return self._evaluator.predict_frames(clip)


# ============================================================
# 最小 Qwen2 backbone fake（_FakeBackbone / _FakeHF / _FakeQwen）
# ============================================================
#
# 原先在 7 个测试文件里逐字复制（test_emofilm_protocol.py /
# test_emofilm_optimizer.py / test_emofilm_training_contract.py /
# test_emofilm_length_finish.py / test_emofilm_local_control_e2e_smoke.py /
# test_emofilm_downstream_heads.py / test_emofilm_inference_contract.py）。
#
# 说明：
# - 这 7 处的 ``_FakeBackbone`` / ``_FakeHF`` 完全相同；
# - 6 处的 ``_FakeQwen`` 也功能等价（仅 ``forward`` 产出的 mask tensor 是否
#   带显式 ``device=xs.device`` 有细微差别；CPU 测试下 ``xs.device`` 为 CPU、
#   ``torch.ones`` 默认也在 CPU，行为完全等价）。此处统一用带
#   ``device=xs.device`` 的写法（更稳健，兼容非-CPU 设备）；
# - ``test_emofilm_training_contract.py`` 的 ``_FakeQwen`` 多了
#   ``output_bias``（用于反捷径梯度断言），**保留在该测试文件本地**，
#   不在此整合；但其 ``_FakeBackbone`` / ``_FakeHF`` 改用本模块的整合版。
#
# 这些 fake 是 test-only 内部辅助（非生产 Protocol 实现），保留带下划线的
# 模块私有命名（``_FakeBackbone`` 等），与原 7 处一致。


class _FakeBackbone(nn.Module):
    """最小 backbone：仅一个 ``embed_tokens``，满足 v2 模型所需的属性探测。"""

    def __init__(self, model_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, model_dim)


class _FakeHF(nn.Module):
    """最小 HF wrapper：``.model`` 指向 ``_FakeBackbone``。"""

    def __init__(self, model_dim):
        super().__init__()
        self.model = _FakeBackbone(model_dim)


class _FakeQwen(nn.Module):
    """恒等 backbone：``forward`` / ``forward_one_step`` 透传 ``xs``。

    用于 CPU 合同测试，满足 v2 ``Qwen2LM_Emotion`` 所需的 backbone 接口
    （``.model.embed_tokens`` / ``forward`` / ``forward_one_step``）。
    不加载真实权重，不需 GPU。
    """

    def __init__(self, model_dim=4):
        super().__init__()
        self.model = _FakeHF(model_dim)

    def forward_one_step(self, xs, masks=None, cache=None):
        return xs, cache

    def forward(self, xs, xs_lens):
        return xs, torch.ones(
            xs.shape[0], 1, xs.shape[1], dtype=torch.bool, device=xs.device
        )


__all__ = [
    "FakeForcedAligner",
    "FakeWerEvaluator",
    "ClipMappedEvaluator",
    "_FakeBackbone",
    "_FakeHF",
    "_FakeQwen",
]
