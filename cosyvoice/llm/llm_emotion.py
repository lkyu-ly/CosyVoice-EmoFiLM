"""Emo-FiLM LLM —— target-only 单流训练-推理协议（活跃主线权威）。

训练与产品推理统一到同一套**非流式 target-only** 条件序列，消除历史双流/
fill-token 状态与未消费的 prompt emotion/intensity 死接口。本模块是 Emo-FiLM
的单一活跃 LLM 权威（ADR-0020 扁平化整合）；v1 双流/classifier 逻辑仅存于
git 基线锚点 ``9c6d84b``，不再于工作树维护并行副本。

协议（MAP §3、``docs/contracts/emofilm_v2_schema.md`` 顶层不变量）：

- 训练 ``lm_input = [SOS, FiLM(target text), task_id, teacher-forced target speech]``，
  ``lm_target = [IGNORE] * (1 + text_len) + target speech + [EOS]``。
- 推理前缀 = ``[SOS, FiLM(target text), task_id]``（训练前缀减去 teacher speech）。
- **绝不**产生 fill_token / 交错文本块 / 双流状态转移，无论 speech/text 比例。
- ``prompt_text`` / ``prompt_speech_token`` / ``prompt_emotion`` /
  ``prompt_intensity`` 不进 LLM 条件；speaker / Flow / HiFT 的 prompt 条件
  （``prompt_speech_token`` / ``embedding``）保持透传给声学侧。

监督路径（两条，可叠加，loss 键名分离）：

- **input-end 句级监督**（``emo_loss_weight>0`` 计入 loss，默认 ``0.0``=不计）：
  ``emotion_classifier``（随机初始化 ``Linear(llm_input_size,
  emotion_vocab_size)``，``requires_grad_(False)`` 恒冻结）**恒构造**，作用在
  FiLM 输出 ``modulated_text_emb`` 上，``CrossEntropyLoss(ignore_index=0)`` 对
  token 级 ``emotion_ids``（同句 token 共享句级标签，padding=0 自动忽略）。
  梯度经冻结探针回流 ``emotion_encoder``/``emotion_adapter``（FiLM）。恒构造使
  模型拓扑与 checkpoint schema 不随配置变化（训练/推理/基线同一 state_dict
  键集）；推理不调用该探针。loss 键：``loss_emotion_input``。
- **downstream span 词级监督**（``downstream_supervision='enabled'`` + span 数据）：
  ``emotion_head``/``arousal_head`` 可训练，仅消费 ``lm_output`` 在 speech-token
  span 区段上 masked-mean 池化的 feature（反捷径：辅助监督落在生成因果链下游，
  不接收 control ID/loss target 作为特征）。loss 键：``loss_emotion_span`` /
  ``loss_intensity``。

无 span 且 ``emo_loss_weight==0`` 时 loss 仅 ``loss_tts``。
- ``__init__`` 不接受 ``mix_ratio`` / ``alpha``（仍为死字段，
  ``assert_no_dead_config`` 拒绝）。
- **不复用** v1 ``prepare_lm_input_target``（含 bistream 分支），改用专用
  ``_prepare_target_only_input`` 恒定单流。
- ``inference`` 签名不接受 ``prompt_emotion_ids`` / ``prompt_intensity_ids`` /
  ``prompt_text``（死字段已删）。
- 下游 emotion/arousal 任务头**仅消费** ``lm_output`` 在 span speech-token 区段上
  masked-mean 池化的 feature（``_pool_span_features``），不接收控制 ID/loss target
  作为特征（反捷径：辅助监督落在生成因果链下游）。
"""
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.llm.emo_film import EmotionEncoder, FiLMLayer
from cosyvoice.utils.common import IGNORE_ID, th_accuracy


# ============================================================
# 长度合同 + 结构化 finish reason 常量
# ============================================================

# Hard cap 默认值：为所有合理文本长度留下足够的 speech token 窗口，
# 同时保证长度有界（~80s @ 25Hz token_frame_rate）。
# 覆盖历史 ``max_len=200`` 硬编码 bug。
DEFAULT_MAX_LEN_HARD_CAP: int = 2000

# 单步内连续非法 token（EOS-before-min_len / 辅助 token / fill_token）
# 的最大重试次数；超过即结构化 ``invalid_token_retry_exhausted``。
MAX_INVALID_TOKEN_RETRIES: int = 100


@dataclass
class DecodeResult:
    """结构化解码结果（长度/finish 合同）。

    所有 finish_reason ∈ ``tools.build_emofilm_contract.FINISH_REASONS``：
    ``eos`` / ``max_len_reached`` / ``invalid_token_retry_exhausted`` /
    ``sampler_error`` / ``input_rejected``。仅 ``eos`` 表示完整生成、
    可进 Flow/HiFT + 正式 WAV；其余 finish reason 不落正式 WAV
    （调用方据 ``finish_reason`` 决定，``validate_generation_row`` 强制
    非 eos 不得携 ``wav_path``）。

    字段（MAP §3 长度/finish 不变量 + schema §2 decode_config）：
        tokens: 解码产出的合法 speech token id 列表（input_rejected 时为空）。
        finish_reason: 互斥、稳定的终止原因。
        min_len: ``int(text_len * min_token_text_ratio)``。
        max_len: ``min(int(text_len * max_token_text_ratio), max_len_hard_cap)``。
        num_valid_speech_tokens: ``len(tokens)``（显式冗余，便于审计/聚合）。
        invalid_token_retries: 累计非法 token 重试次数（EOS-before-min + 辅助）。
        text_len: 输入目标文本 token 数（input_rejected 诊断携带）。
    """

    tokens: List[int]
    finish_reason: str
    min_len: int
    max_len: int
    num_valid_speech_tokens: int
    invalid_token_retries: int
    text_len: int


def _cache_seq_length(cache) -> int:
    """KV cache 已存序列长度（兼容 Transformers 两种 cache 格式）。"""
    if cache is None:
        return 0
    get_seq_length = getattr(cache, "get_seq_length", None)
    if get_seq_length is not None:
        return int(get_seq_length())
    try:
        return int(cache[0][0].size(2))
    except (AttributeError, IndexError, TypeError):
        raise TypeError("unsupported KV cache format") from None


# ============================================================
# 下游监督 span 张量契约（与 span_align.collate_aligned_spans 对齐）
# ============================================================

#: ``collate_aligned_spans`` 产出的 span 张量键（forward 消费子集）。
#: 这些键存在 ⇒ batch 携带下游监督 span → forward 计算 head loss。
_SPAN_TENSOR_KEYS = (
    "span_mask",
    "span_valid",
    "span_tok_start",
    "span_tok_end",
    "span_emotion_mask",
    "span_intensity_mask",
    "span_emotion_soft_dist",
    "span_arousal",
    "span_supervision_weight",
    "span_control_emotion_id",
    "span_control_intensity_id",
)

#: 判定 batch 是否携带下游监督 span 所需的最小键集（区间 + mask + target）。
_SPAN_REQUIRED_KEYS = (
    "span_mask",
    "span_valid",
    "span_tok_start",
    "span_tok_end",
    "span_emotion_mask",
    "span_intensity_mask",
    "span_emotion_soft_dist",
    "span_arousal",
    "span_supervision_weight",
)


def _batch_has_spans(batch: Mapping) -> bool:
    """batch 是否携带对齐的下游监督 span（决定是否计算 head loss）。"""
    return all(k in batch for k in _SPAN_REQUIRED_KEYS)


def _weighted_span_mean(
    per_span: torch.Tensor,
    active: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """监督加权均值：``Σ(per_span·active·weight) / Σ(active·weight)``。

    - ``active`` 为 bool mask（per-span 有效性，含 ``valid`` & 对应 ``emotion/intensity_mask``）。
    - ``weight`` 为 ``span_supervision_weight``（弱监督按 resolved config 生效）。
    - 无活跃 span（``active`` 全 False）→ 分母 clamp 为 1，返回 0（不贡献）。
    """
    w = active.to(per_span.dtype) * weight.to(per_span.dtype)
    numerator = (per_span * w).sum()
    denominator = w.sum().clamp(min=1e-8)
    return numerator / denominator


class Qwen2LM_Emotion(Qwen2LM):
    """target-only 单流 Emo-FiLM LLM（``Qwen2LM`` 子类）。

    FiLM 保留（复用 ``emo_film.py`` 的 ``EmotionEncoder`` + ``FiLMLayer``），
    仅注入 text-token embedding 区段。

    **下游监督任务头**：``emotion_head``（5 类）+ ``arousal_head``
    （连续回归），二者**仅消费** ``lm_output`` 在每条 span 的 speech-token 区段
    上 masked-mean 池化的 feature（``_pool_span_features``）—— 任务头不接收
    ``emotion_ids`` / ``intensity_ids`` / ``modulated_text_emb`` / loss target
    作为特征（反捷径：辅助监督落在生成因果链下游，消除输入端标签回读捷径）。
    总 loss = ``loss_tts + w_e·loss_emotion_span + w_i·loss_intensity``（权重
    ``emotion_head_weight`` / ``intensity_head_weight``）。emotion 用 soft CE
    （one-hot 特例即 hard CE）；intensity 用连续 arousal MSE；per-span
    ``emotion_mask`` / ``intensity_mask`` 独立门控；无效 span（``valid=False``）
    不贡献。

    **input-end 句级监督**：``emotion_classifier``（随机初始化、冻结）恒构造，
    ``emo_loss_weight>0`` 时对 FiLM 输出 ``modulated_text_emb`` 做句级情感 CE
    （token 级 ``emotion_ids``，padding=0 由 ``ignore_index=0`` 忽略），梯度经
    冻结探针回流 FiLM（``emotion_encoder``/``emotion_adapter``）。
    ``emo_loss_weight==0``（默认）时不计入 loss（disabled 路径 loss 仅
    ``loss_tts``，loss_dict 无 ``loss_emotion_input``）。
    """

    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        emotion_vocab_size: int,
        intensity_vocab_size: int,
        llm: nn.Module,
        sampling: Callable,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        emotion_head_weight: float = 1.0,
        intensity_head_weight: float = 1.0,
        emo_loss_weight: float = 0.0,
        downstream_supervision: str = "disabled",
    ):
        # 父类 ``Qwen2LM.__init__`` 仍接受 ``mix_ratio``（默认 [5,15]）。
        # 本类**永不读取** ``self.mix_ratio``（不调用 ``prepare_lm_input_target``），
        # 该属性仅作为父类遗留存在；resolved 配置不得出现 ``mix_ratio`` 键
        # （``assert_no_dead_config`` 拒绝）。亦不接受 ``alpha``（历史死占位）；
        # ``emo_loss_weight`` 现为可选 input-end 句级监督开关（默认 ``0.0``=关闭）。
        super().__init__(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=speech_token_size,
            llm=llm,
            sampling=sampling,
            length_normalized_loss=length_normalized_loss,
            lsm_weight=lsm_weight,
        )
        # FiLM（保留输入层调制）：复用 emo_film.py，仅注入 text-token 区段。
        self.emotion_encoder = EmotionEncoder(
            emotion_vocab_size, intensity_vocab_size, llm_input_size
        )
        self.emotion_adapter = FiLMLayer(llm_input_size)

        # input-end 句级情感监督探针（恒构造、冻结；emo_loss_weight>0 才计入
        # loss）：emotion_classifier 随机初始化并 requires_grad_(False)，
        # loss_emotion 梯度经固定读出器回流 FiLM（emotion_encoder /
        # emotion_adapter）。恒构造使模型拓扑与 state_dict 键集不随配置变化
        # （训练/推理/基线同一 schema），推理不调用该探针；emo_loss_weight==0
        # 时仅不计入 loss。
        self.emo_loss_weight = float(emo_loss_weight)
        self.emotion_classifier = nn.Linear(
            llm_input_size, emotion_vocab_size
        )
        self.emotion_classifier.requires_grad_(False)
        self.criterion_emotion_cls = nn.CrossEntropyLoss(ignore_index=0)

        # ------------------------------------------------------------
        # 下游 speech-token 监督任务头
        # ------------------------------------------------------------
        # 两个线性层（随机初始化、可训练），**仅消费** ``lm_output`` 在每条
        # span 的 speech-token 区段上 masked-mean 池化得到的 feature（见
        # ``_pool_span_features``）。任务头**绝不**接收 ``emotion_ids`` /
        # ``intensity_ids`` / condition embedding / loss target 作为特征输入
        # （反捷径，见 ``_pool_span_features`` docstring）。
        #   - ``emotion_head``：5 类情感 logits（soft CE / hard CE 的统一实现）。
        #   - ``arousal_head``：连续 arousal 标量回归（intensity weak target）。
        # 注意 emotion 输出 5（非 ``emotion_vocab_size=6``），因为 pad 类不参与
        # 监督；soft distribution 始终是 5 维。
        self.emotion_head = nn.Linear(llm_output_size, 5)
        self.arousal_head = nn.Linear(llm_output_size, 1)
        # 总 loss 权重（语义不同于输入端 ``emo_loss_weight``：此处是生成
        # 因果链下游监督头的权重，不是输入端 classifier CE 权重）。
        self.emotion_head_weight = float(emotion_head_weight)
        self.intensity_head_weight = float(intensity_head_weight)
        # 下游监督开关（B1 静默→显式）。'enabled'：期望 batch 携带 span 张量，
        # 无 span 时 forward 显式报错（防监督头未接线被静默吞，见 ticket 02）；
        # 'disabled'：允许 FiLM-only 训练（无 span 时只算 loss_tts）。本次实验口径。
        if downstream_supervision not in ("enabled", "disabled"):
            raise ValueError(
                "downstream_supervision must be 'enabled' or 'disabled', "
                f"got {downstream_supervision!r}"
            )
        self.downstream_supervision = downstream_supervision

    # ------------------------------------------------------------
    # 专用 input-prep：恒定单流，绝不进入 bistream/fill-token 分支
    # ------------------------------------------------------------

    def _prepare_target_only_input(
        self,
        sos_emb: torch.Tensor,
        text_token_emb: torch.Tensor,
        text_token_len: torch.Tensor,
        task_id_emb: torch.Tensor,
        speech_token: torch.Tensor,
        speech_token_emb: torch.Tensor,
        speech_token_len: torch.Tensor,
    ):
        """构造 target-only 单流 lm_input / lm_target（无 fill_token、无双流）。

        无论 speech/text 比例或随机种子如何，恒走单流（剥离历史 bistream 分支）。

        每 sample：
          lm_input  = [sos, text_emb(modulated), task_id, speech_emb]
          lm_target = [IGNORE] * (1 + text_len) + speech_token + [eos]

        lm_input 与 lm_target 等长（单流 next-token teacher-forcing 对齐），
        二者 pad 到 batch 内最大长度（padding_value=IGNORE_ID）。
        """
        text_len_cpu = text_token_len.cpu()
        speech_len_cpu = speech_token_len.cpu()

        text_token_emb_unpad = unpad_sequence(
            text_token_emb, text_len_cpu, batch_first=True
        )
        speech_token_unpad = unpad_sequence(
            speech_token, speech_len_cpu, batch_first=True
        )
        speech_token_emb_unpad = unpad_sequence(
            speech_token_emb, speech_len_cpu, batch_first=True
        )

        sos_vec = sos_emb.squeeze(dim=0)
        task_vec = task_id_emb.squeeze(dim=0)
        eos = self.eos_token

        lm_input_list = []
        lm_target_list = []
        for i in range(len(text_token_emb_unpad)):
            t_len = int(text_len_cpu[i].item())
            this_input = torch.concat(
                [
                    sos_vec,
                    text_token_emb_unpad[i],
                    task_vec,
                    speech_token_emb_unpad[i],
                ],
                dim=0,
            )
            this_target = torch.tensor(
                [IGNORE_ID] * (1 + t_len)
                + speech_token_unpad[i].tolist()
                + [eos]
            )
            lm_input_list.append(this_input)
            lm_target_list.append(this_target)

        lm_input_len = torch.tensor(
            [t.size(0) for t in lm_input_list], dtype=torch.int32
        )
        lm_input = pad_sequence(
            lm_input_list, batch_first=True, padding_value=IGNORE_ID
        )
        lm_target = pad_sequence(
            lm_target_list, batch_first=True, padding_value=IGNORE_ID
        )
        return lm_target, lm_input, lm_input_len

    # ------------------------------------------------------------
    # 训练前向
    # ------------------------------------------------------------

    def forward(
        self, batch: dict, device: torch.device
    ) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        emotion_ids = batch["emotion_ids"].to(device)
        intensity_ids = batch["intensity_ids"].to(device)

        text_token_emb = self.llm.model.model.embed_tokens(text_token)
        speech_token_emb = self.speech_embedding(speech_token)

        # FiLM 调制 text-token embedding（仅 text 区段）
        emotion_features = self.emotion_encoder(emotion_ids, intensity_ids)
        modulated_text_emb = self.emotion_adapter(text_token_emb, emotion_features)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 恒定单流：绝不进入 bistream / fill-token 分支
        lm_target, lm_input, lm_input_len = self._prepare_target_only_input(
            sos_emb,
            modulated_text_emb,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_emb,
            speech_token_len,
        )
        lm_target = lm_target.to(device)

        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss_tts = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 3),
            lm_target,
            ignore_label=IGNORE_ID,
        )

        # ------------------------------------------------------------
        # 监督组合点（两条路径可叠加；loss 键名分离）
        # ------------------------------------------------------------
        # ``lm_output`` 为最后一层 hidden (B, T, llm_output_size)；
        # speech-token 区段 = ``lm_target != IGNORE_ID`` 的列。
        # ``speech_token_mask`` 既标识 supervised 列，也作为池化的安全网
        # （排除 IGNORE/padding 列）。
        speech_token_mask = lm_target != IGNORE_ID  # (B, T) bool, True = supervised
        loss = loss_tts
        loss_dict = {"loss": loss, "acc": acc, "loss_tts": loss_tts.detach()}

        # 1) 下游 span 词级监督（batch 携带 span 张量时计算；无 span 时由
        #    downstream_supervision 显式裁决，禁止静默降级）。
        if _batch_has_spans(batch):
            spans = {
                k: batch[k].to(device)
                for k in _SPAN_TENSOR_KEYS
                if k in batch
            }
            # feature 仅由 ``lm_output`` + span 几何区间决定（反捷径核心）。
            span_feature = self._pool_span_features(
                lm_output,
                speech_token_mask,
                text_token_len,
                spans["span_tok_start"],
                spans["span_tok_end"],
                spans["span_mask"],
                spans["span_valid"],
            )
            loss_emotion_span, loss_intensity = self._compute_downstream_losses(
                span_feature, spans
            )
            loss = (
                loss
                + self.emotion_head_weight * loss_emotion_span
                + self.intensity_head_weight * loss_intensity
            )
            loss_dict["loss_emotion_span"] = loss_emotion_span.detach()
            loss_dict["loss_intensity"] = loss_intensity.detach()
        elif self.downstream_supervision == "enabled":
            raise RuntimeError(
                "downstream_supervision='enabled' 但 batch 未携带 span 张量——"
                "下游监督头未接入数据管线（span→parquet→batch 链断）。"
                "若本次为 FiLM-only 实验，请在配置设 downstream_supervision='disabled'。"
            )

        # 2) input-end 句级监督（emo_loss_weight>0 时计入；与 span 路径可叠加，
        #    不再被 span 分支短路丢弃）。
        if self.emo_loss_weight > 0:
            emotion_logits = self.emotion_classifier(modulated_text_emb)
            loss_emotion_input = self.criterion_emotion_cls(
                emotion_logits.reshape(-1, emotion_logits.size(-1)),
                emotion_ids.reshape(-1),
            )
            loss = loss + self.emo_loss_weight * loss_emotion_input
            loss_dict["loss_emotion_input"] = loss_emotion_input.detach()

        loss_dict["loss"] = loss
        return loss_dict

    # ------------------------------------------------------------
    # 下游任务头：span 池化 + loss
    # ------------------------------------------------------------

    def _pool_span_features(
        self,
        lm_output: torch.Tensor,
        speech_token_mask: torch.Tensor,
        text_token_len: torch.Tensor,
        span_tok_start: torch.Tensor,
        span_tok_end: torch.Tensor,
        span_mask: torch.Tensor,
        span_valid: torch.Tensor,
    ) -> torch.Tensor:
        """按每条 span 的 speech-token 区间对 ``lm_output`` 做 masked-mean 池化。

        **反捷径契约**：本函数的输出（任务头输入 feature）**仅由** ``lm_output``
        与 span 几何区间（``tok_start/tok_end``）决定。本函数**绝不读取**
        ``modulated_text_emb`` / ``emotion_ids`` / ``intensity_ids`` / condition
        embedding / loss target（结构上由函数签名与源码保证）。改变 control ID
        只能经 FiLM→LLM→``lm_output`` **间接**影响本 feature。

        speech-token 区段定位：target = ``[IGNORE]*(1+text_len) + speech +
        [eos]``，故 supervised 列（``lm_target != IGNORE``）从 ``1 + text_len``
        起；span 的 ``tok_start/tok_end``（相对 speech-token 序列）的绝对列 =
        ``1 + text_len + tok``。``tok_end <= speech_token_len`` 保证 EOS 列被
        exclusive 切片排除。

        Returns:
            (B, S, llm_output_size) 每 span 一个 masked-mean feature 向量；
            padding / 无效 span 返回零向量（其 loss 由 mask 门控，不贡献）。
        """
        B, T, D = lm_output.shape
        S = span_tok_start.shape[1]
        if S == 0:
            return lm_output.new_zeros((B, 0, D))

        arange_T = torch.arange(T, device=lm_output.device)  # (T,)
        speech_start = (1 + text_token_len).to(torch.long).to(lm_output.device)  # (B,)
        abs_start = speech_start[:, None] + span_tok_start.long()  # (B, S)
        abs_end = speech_start[:, None] + span_tok_end.long()  # (B, S)
        # span × 列区间（exclusive end）
        interval = (
            (arange_T[None, None, :] >= abs_start[:, :, None])
            & (arange_T[None, None, :] < abs_end[:, :, None])
        )  # (B, S, T)
        # 与 speech_token_mask 求交（排除 IGNORE/padding 列，安全网）
        valid_cols = interval & speech_token_mask[:, None, :]  # (B, S, T)
        # 仅对真实且对齐有效的 span 池化（padding / 无效 span → 零 feature）
        active = (span_mask & span_valid)[:, :, None]  # (B, S, 1)
        valid_cols = valid_cols & active

        weights = valid_cols.to(lm_output.dtype)  # (B, S, T)
        counts = weights.sum(dim=-1).clamp(min=1.0)  # (B, S)，避免除零
        pooled = torch.einsum("bst,btd->bsd", weights, lm_output)  # (B, S, D)
        return pooled / counts.unsqueeze(-1)

    def _compute_downstream_losses(
        self, span_feature: torch.Tensor, spans: Mapping[str, torch.Tensor]
    ):
        """计算 emotion soft-CE loss 与 intensity (arousal) MSE loss。

        - emotion：``emotion_head`` 出 5 类 logits → soft CE（``-Σ p·log_softmax``；
          one-hot 分布即 hard CE 的特例）。per-span 门控 =
          ``span_mask & span_valid & span_emotion_mask``，加权
          ``span_supervision_weight``。
        - intensity：``arousal_head`` 出标量 → MSE 对连续 ``span_arousal``；
          per-span 门控 = ``span_mask & span_valid & span_intensity_mask``
          （ESD ``intensity_mask=False`` → 不贡献 intensity loss）。
        - 缺失/无效 target（``valid=False`` 或对应 mask=False）→ 不贡献。
        - 无活跃 span 时返回 0（监督加权均值分母 clamp）。
        """
        emotion_logits = self.emotion_head(span_feature)  # (B, S, 5)
        arousal_pred = self.arousal_head(span_feature).squeeze(-1)  # (B, S)

        log_probs = F.log_softmax(emotion_logits, dim=-1)  # (B, S, 5)
        soft_dist = spans["span_emotion_soft_dist"]  # (B, S, 5)
        soft_ce = -(soft_dist * log_probs).sum(dim=-1)  # (B, S)

        arousal_tgt = spans["span_arousal"]  # (B, S)
        mse = (arousal_pred - arousal_tgt) ** 2  # (B, S)

        sw = spans["span_supervision_weight"]  # (B, S)
        base = spans["span_mask"] & spans["span_valid"]
        emo_active = base & spans["span_emotion_mask"]
        int_active = base & spans["span_intensity_mask"]

        loss_emotion = _weighted_span_mean(soft_ce, emo_active, sw)
        loss_intensity = _weighted_span_mean(mse, int_active, sw)
        return loss_emotion, loss_intensity

    # ------------------------------------------------------------
    # 产品推理（target-only 前缀）+ 结构化长度/finish 合同
    # ------------------------------------------------------------

    @torch.inference_mode()
    def decode(
        self,
        text_token: torch.Tensor,
        text_len: torch.Tensor,
        emotion_ids: torch.Tensor,
        intensity_ids: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        max_len_hard_cap: int = DEFAULT_MAX_LEN_HARD_CAP,
    ) -> DecodeResult:
        """target-only 解码 + 结构化长度/finish 合同。

        前缀 = ``[sos, FiLM(target text), task_id]``（训练前缀减去
        teacher-forced target speech）。FiLM 仅调制 text-token embedding 区段。

        长度推导（MAP §3 长度不变量 / schema §2 decode_config）::

            min_len = int(text_len * min_token_text_ratio)
            max_len = min(int(text_len * max_token_text_ratio), max_len_hard_cap)

        解码前不变量：若 ``max_len <= min_len``（hard cap 不足以容纳输入）
        → **不进 token sampling**，直接返回 ``finish_reason="input_rejected"``
        的 ``DecodeResult``（携带 min_len / max_len / text_len）。

        finish_reason 互斥稳定：``eos`` / ``max_len_reached`` /
        ``invalid_token_retry_exhausted`` / ``sampler_error`` /
        ``input_rejected``。仅 ``eos`` 表示完整生成、可进 Flow/HiFT + 正式 WAV。
        """
        text_len_int = int(text_len.item())
        min_len = int(text_len_int * min_token_text_ratio)
        ratio_max = int(text_len_int * max_token_text_ratio)
        max_len = min(ratio_max, int(max_len_hard_cap))

        # 解码前不变量：hard cap 必须严格容纳 min_len（max_len > min_len）。
        # 等号也拒绝：min_len == max_len 时 EOS 永远不可能在 min_len 之后被采到，
        # 等价于无合法完成路径，必须结构化拒绝而非静默跑满。
        if max_len <= min_len:
            return DecodeResult(
                tokens=[],
                finish_reason="input_rejected",
                min_len=min_len,
                max_len=max_len,
                num_valid_speech_tokens=0,
                invalid_token_retries=0,
                text_len=text_len_int,
            )

        # 构造推理前缀 [SOS, FiLM(target text), task_id]（FiLM 仅注入 text 区段）。
        text_emb = self.llm.model.model.embed_tokens(text_token)
        emo_feats = self.emotion_encoder(emotion_ids, intensity_ids)
        text_emb = self.emotion_adapter(text_emb, emo_feats)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([sos_emb, text_emb, task_id_emb], dim=1)

        # 通过 ``inference_wrapper`` 生成器驱动解码（保持协议测试对
        # ``inference_wrapper`` 的 monkeypatch 钩子有效：那些测试替换本方法
        # 以捕获 lm_input/min_len/max_len，不消费结构化结果）。真实 wrapper
        # 将 ``DecodeResult`` 写入 ``self._wrapper_result``；decode 读取之。
        # text_len_int 不在 wrapper 签名里，故 decode 事后补全诊断字段。
        self._wrapper_result = None
        list(self.inference_wrapper(lm_input, sampling, min_len, max_len, ""))
        if self._wrapper_result is not None:
            result = self._wrapper_result
            result.text_len = text_len_int
            return result
        # 退化路径（monkeypatched / 协议测试）：wrapper 未产出结构化结果，
        # 合成一个保守的 max_len_reached 占位（这些调用方不检查 finish_reason）。
        return DecodeResult(
            tokens=[],
            finish_reason="max_len_reached",
            min_len=min_len,
            max_len=max_len,
            num_valid_speech_tokens=0,
            invalid_token_retries=0,
            text_len=text_len_int,
        )

    def _decode_loop(
        self,
        lm_input: torch.Tensor,
        sampling: int,
        min_len: int,
        max_len: int,
        text_len_int: int = 0,
    ) -> DecodeResult:
        """结构化解码循环（单流、EOS 终止、可审计）。

        KV-cache mask 构造与 EOS-before-min 重采样逻辑保留；但
          (1) 重试上限不再 raise，改为 ``invalid_token_retry_exhausted``；
          (2) 采样器异常不再向上传播，改为 ``sampler_error``；
          (3) ``max_len`` 步耗尽不再静默成功，改为 ``max_len_reached``；
          (4) 累计 ``invalid_token_retries`` 写入结构化结果。
        """
        out_tokens: List[int] = []
        invalid_token_retries = 0
        cache = None
        eos = self.eos_token
        speech_token_size = self.speech_token_size

        for step in range(max_len):
            current_len = lm_input.shape[1]
            past_len = _cache_seq_length(cache)
            total_len = past_len + current_len
            if cache is None:
                mask = torch.tril(
                    torch.ones(
                        (1, current_len, total_len),
                        device=lm_input.device,
                    )
                ).to(torch.bool)
            else:
                mask = torch.ones(
                    (1, current_len, total_len),
                    device=lm_input.device,
                    dtype=torch.bool,
                )
            y_pred, cache = self.llm.forward_one_step(
                lm_input, masks=mask, cache=cache
            )
            scores = (
                self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1).squeeze(0)
            )

            # Inner resample loop：EOS-before-min / 辅助 token / fill_token
            # 都触发重采样，但重试有界。
            consecutive_invalid = 0
            while True:
                try:
                    top_id = self.sampling(scores.clone(), out_tokens, sampling)
                except Exception:
                    # 采样器异常 → 结构化 sampler_error，不向上传播。
                    return DecodeResult(
                        tokens=list(out_tokens),
                        finish_reason="sampler_error",
                        min_len=min_len,
                        max_len=max_len,
                        num_valid_speech_tokens=len(out_tokens),
                        invalid_token_retries=invalid_token_retries,
                        text_len=text_len_int,
                    )
                top_id = (
                    int(top_id.item()) if torch.is_tensor(top_id) else int(top_id)
                )

                if top_id == eos:
                    if len(out_tokens) >= min_len:
                        return DecodeResult(
                            tokens=list(out_tokens),
                            finish_reason="eos",
                            min_len=min_len,
                            max_len=max_len,
                            num_valid_speech_tokens=len(out_tokens),
                            invalid_token_retries=invalid_token_retries,
                            text_len=text_len_int,
                        )
                    # EOS before min_len → 重采样（计入重试统计）。
                    consecutive_invalid += 1
                    invalid_token_retries += 1
                    if consecutive_invalid > MAX_INVALID_TOKEN_RETRIES:
                        return DecodeResult(
                            tokens=list(out_tokens),
                            finish_reason="invalid_token_retry_exhausted",
                            min_len=min_len,
                            max_len=max_len,
                            num_valid_speech_tokens=len(out_tokens),
                            invalid_token_retries=invalid_token_retries,
                            text_len=text_len_int,
                        )
                    continue

                if top_id >= speech_token_size:
                    # 辅助 token / fill_token（speech_token_size+1, +2）→ 重采样。
                    consecutive_invalid += 1
                    invalid_token_retries += 1
                    if consecutive_invalid > MAX_INVALID_TOKEN_RETRIES:
                        return DecodeResult(
                            tokens=list(out_tokens),
                            finish_reason="invalid_token_retry_exhausted",
                            min_len=min_len,
                            max_len=max_len,
                            num_valid_speech_tokens=len(out_tokens),
                            invalid_token_retries=invalid_token_retries,
                            text_len=text_len_int,
                        )
                    continue

                # 合法 speech token。
                break

            out_tokens.append(top_id)
            lm_input = self.speech_embedding.weight[top_id].reshape(1, 1, -1)

        # 循环耗尽仍未出现 EOS → max_len_reached（不再静默成功）。
        return DecodeResult(
            tokens=list(out_tokens),
            finish_reason="max_len_reached",
            min_len=min_len,
            max_len=max_len,
            num_valid_speech_tokens=len(out_tokens),
            invalid_token_retries=invalid_token_retries,
            text_len=text_len_int,
        )

    @torch.inference_mode()
    def inference_wrapper(
        self,
        lm_input: torch.Tensor,
        sampling: int,
        min_len: int,
        max_len: int,
        uuid: str,
    ) -> Generator[torch.Tensor, None, None]:
        """结构化解码生成器（协议兼容签名 + 结构化结果桥）。

        保留 generator 签名 ``(lm_input, sampling, min_len, max_len, uuid)``
        以维持协议测试（monkeypatch 本方法捕获 lm_input/min_len/max_len）。

        真实现委托 ``_decode_loop`` 获得完整 ``DecodeResult``，将其写入
        ``self._wrapper_result`` 供 ``decode`` 读取，再 ``yield from`` 产出
        speech token。``decode`` 据此构造结构化结果并按 ``finish_reason`` 门控
        是否进 Flow/HiFT（仅 ``eos``）。

        注意：``_decode_loop`` 先跑完整循环再 yield（buffered），因为
        finish_reason 只有循环终止后才能确定 —— "仅 eos 进声学"的合同要求
        在向 Flow 产出任何 token 前就知道是否 eos，故无法逐 token 流式输出。
        """
        del uuid
        result = self._decode_loop(lm_input, sampling, min_len, max_len)
        self._wrapper_result = result
        yield from result.tokens

    @torch.inference_mode()
    def inference(
        self,
        text_token: torch.Tensor,
        text_len: torch.Tensor,
        emotion_ids: torch.Tensor,
        intensity_ids: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        max_len_hard_cap: int = DEFAULT_MAX_LEN_HARD_CAP,
        uuid: str = "",
    ) -> Generator[torch.Tensor, None, None]:
        """target-only 推理生成器：委托 ``decode`` 并按 finish_reason 门控输出。

        与训练目标端前缀一致，只缺少 teacher-forced target speech。
        **不接受** ``prompt_emotion_ids`` / ``prompt_intensity_ids`` /
        ``prompt_text``（死字段已删，MAP §3 协议）。

        ``prompt_speech_token`` / ``embedding`` 仅透传给 Flow / HiFT（声学侧），
        **不进** LLM ``lm_input``；LLM 条件只含 target text + emotion/intensity 控制。

        长度：min/max 由 target 文本长度 + resolved ratio + hard cap 推导
        （MAP §3 长度不变量、schema §2 decode_config）。

        **门控不变量**：仅 ``finish_reason == "eos"`` 时向 Flow/HiFT 产出
        speech token；非 eos（max_len_reached / invalid_token_retry_exhausted /
        sampler_error / input_rejected）不产出任何 token，因此绝不进声学与正式
        WAV。结构化 ``DecodeResult`` 通过 ``self.last_decode_result`` 暴露给调用方。
        """
        del prompt_speech_token, prompt_speech_token_len, embedding, uuid

        result = self.decode(
            text_token=text_token,
            text_len=text_len,
            emotion_ids=emotion_ids,
            intensity_ids=intensity_ids,
            sampling=sampling,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
            max_len_hard_cap=max_len_hard_cap,
        )
        self.last_decode_result = result

        # 仅 eos → Flow/HiFT（formal WAV 路径）；非 eos 不向声学侧产出任何 token，
        # 消除"无 EOS 自然结束却被调用者视为成功"的路径（MAP §3）。
        if result.finish_reason == "eos":
            for token in result.tokens:
                yield token
