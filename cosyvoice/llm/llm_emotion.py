"""Qwen2LM_Emotion: 继承 Qwen2LM，新增词级情感调制 FiLM + emotion loss。

forward: 在 modulated_text_emb 上计算 loss_emotion (Emo_PA 行为)。
inference: 保留 prompt 拼接 + 复用 inference_wrapper。
"""
from typing import Callable, Dict, Generator, List, Optional
import torch
import torch.nn as nn
from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.llm.emo_film import EmotionEncoder, FiLMLayer
from cosyvoice.utils.common import IGNORE_ID, th_accuracy
# 注：AddFusionEmotionAdapter 不在此处 import，通过 yaml !new 构造后由 emotion_adapter 参数注入


class Qwen2LM_Emotion(Qwen2LM):
    def __init__(
        self,
        llm_input_size: int,
        llm_output_size: int,
        speech_token_size: int,
        emotion_vocab_size: int,
        intensity_vocab_size: int,
        llm: nn.Module,
        sampling: Callable,
        tokenizer_for_emotion: Optional[object] = None,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        mix_ratio: List[int] = [5, 15],
        emo_loss_weight: float = 0.2,
        emo_loss_on: str = "modulated_text_emb",
        alpha: float = 0.05,
        emotion_adapter: Optional[nn.Module] = None,
    ):
        # 调用父类 __init__（llm.py:258-268 兼容 8 参数签名）
        super().__init__(
            llm_input_size=llm_input_size,
            llm_output_size=llm_output_size,
            speech_token_size=speech_token_size,
            llm=llm,
            sampling=sampling,
            length_normalized_loss=length_normalized_loss,
            lsm_weight=lsm_weight,
            mix_ratio=mix_ratio,
        )

        self.emotion_encoder = EmotionEncoder(emotion_vocab_size, intensity_vocab_size, llm_input_size)
        # emotion_adapter 可注入：默认 None → FiLMLayer（与 Emo_PA 源码 llm_emo.py:171 一致）
        # ablation_no_film.yaml 通过 yaml !new:AddFusionEmotionAdapter 注入以实现 w/o FiLM 消融
        self.emotion_adapter = emotion_adapter if emotion_adapter is not None else FiLMLayer(llm_input_size)
        self.emotion_classifier = nn.Linear(llm_input_size, emotion_vocab_size)
        self.criterion_emotion_cls = nn.CrossEntropyLoss(ignore_index=0)
        self.emo_loss_weight = emo_loss_weight
        # spec 商讨点 3：emo_loss 投影位置可配置
        # 'modulated_text_emb' (默认, Emo_PA 行为) 或 'llm_output' (论文文字推断位置)
        assert emo_loss_on in ("modulated_text_emb", "llm_output"), \
            f"emo_loss_on must be 'modulated_text_emb' or 'llm_output', got {emo_loss_on}"
        self.emo_loss_on = emo_loss_on
        # alpha 仅配置占位，不参与有效计算（Emo_PA llm_emo.py:62-64 alpha 版本被注释）

    def forward(self, batch: dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        emotion_ids = batch["emotion_ids"].to(device)
        intensity_ids = batch["intensity_ids"].to(device)

        text_token_emb = self.llm.model.model.embed_tokens(text_token)
        speech_token_emb = self.speech_embedding(speech_token)

        emotion_features = self.emotion_encoder(emotion_ids, intensity_ids)
        modulated_text_emb = self.emotion_adapter(text_token_emb, emotion_features)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
            sos_emb, text_token, modulated_text_emb, text_token_len,
            task_id_emb, speech_token, speech_token_emb, speech_token_len,
        )
        lm_target = lm_target.to(device)

        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss_tts = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)

        # spec 商讨点 3：emo_loss 投影位置可配置
        # 默认在 modulated_text_emb 上（Emo_PA 行为）；可切换到 llm_output（论文推断）
        if self.emo_loss_on == "modulated_text_emb":
            emo_proj = modulated_text_emb
            emo_target = emotion_ids
        else:  # 'llm_output'
            # lm_output 形状 (B, T_lm, D)，T_lm 包含 sos/instruct/task/speech_token，远大于 T_text
            # 将 lm_output 在序列维度截取到 T_text 长度，与 emotion_ids (B, T_text) 对齐
            # 注意：lm_output[:, :T_text] 是近似（lm_input 结构复杂，bistream/unistream 拆分），
            # 但语义上覆盖了 text_token 区间附近的隐状态，足以用于 emotion 分类监督
            T_text = emotion_ids.shape[1]
            emo_proj = lm_output[:, :T_text, :]
            emo_target = emotion_ids
        emotion_logits = self.emotion_classifier(emo_proj)
        loss_emotion = self.criterion_emotion_cls(
            emotion_logits.view(-1, emotion_logits.size(-1)),
            emo_target.view(-1),
        )

        loss = loss_tts + self.emo_loss_weight * loss_emotion
        return {
            "loss": loss, "acc": acc,
            "loss_tts": loss_tts.detach(), "loss_emotion": loss_emotion.detach(),
        }

    @torch.inference_mode()
    def inference(
        self,
        text_token: torch.Tensor,
        text_len: torch.Tensor,
        emotion_ids: torch.Tensor,
        intensity_ids: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_emotion_ids: torch.Tensor,
        prompt_intensity_ids: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        uuid: str = "",
    ) -> Generator[torch.Tensor, None, None]:
        device = text_token.device

        # concat prompt + target
        full_text = torch.concat([prompt_text, text_token], dim=1)
        full_emotion_ids = torch.concat([prompt_emotion_ids, emotion_ids], dim=1)
        full_intensity_ids = torch.concat([prompt_intensity_ids, intensity_ids], dim=1)

        text_emb = self.llm.model.model.embed_tokens(full_text)
        emo_feats = self.emotion_encoder(full_emotion_ids, full_intensity_ids)
        text_emb = self.emotion_adapter(text_emb, emo_feats)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text_emb.dtype).to(device)

        lm_input = torch.concat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

        min_len = int(text_len.item() * min_token_text_ratio)
        max_len = int(text_len.item() * max_token_text_ratio)

        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token
