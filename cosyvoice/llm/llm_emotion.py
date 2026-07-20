"""基础 Emo-FiLM 的训练前向和目标文本推理。"""
from typing import Callable, Dict, Generator, List, Optional
import torch
import torch.nn as nn
from cosyvoice.llm.llm import Qwen2LM
from cosyvoice.llm.emo_film import EmotionEncoder, FiLMLayer
from cosyvoice.utils.common import IGNORE_ID, th_accuracy


def _cache_seq_length(cache) -> int:
    """Return cached key/value length for both Transformers cache formats."""
    if cache is None:
        return 0
    get_seq_length = getattr(cache, "get_seq_length", None)
    if get_seq_length is not None:
        return int(get_seq_length())
    try:
        return int(cache[0][0].size(2))
    except (AttributeError, IndexError, TypeError):
        raise TypeError("unsupported KV cache format") from None


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
        alpha: float = 0.05,
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
        self.emotion_adapter = FiLMLayer(llm_input_size)
        self.emotion_classifier = nn.Linear(llm_input_size, emotion_vocab_size)
        self.emotion_classifier.requires_grad_(False)
        self.criterion_emotion_cls = nn.CrossEntropyLoss(ignore_index=0)
        self.emo_loss_weight = emo_loss_weight
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

        empty_instruct_token = text_token.new_empty((text_token.size(0), 0))
        empty_instruct_emb = text_token_emb.new_empty(
            (text_token.size(0), 0, self.llm_input_size)
        )
        empty_instruct_len = text_token_len.new_zeros(text_token.size(0))
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(
            sos_emb, text_token, modulated_text_emb, text_token_len,
            task_id_emb, speech_token, speech_token_emb, speech_token_len,
            instruct_token=empty_instruct_token,
            instruct_token_emb=empty_instruct_emb,
            instruct_token_len=empty_instruct_len,
        )
        lm_target = lm_target.to(device)

        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss_tts = self.criterion_ce(logits, lm_target.to(device))
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)

        emotion_logits = self.emotion_classifier(modulated_text_emb)
        loss_emotion = self.criterion_emotion_cls(
            emotion_logits.reshape(-1, emotion_logits.size(-1)),
            emotion_ids.reshape(-1),
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

        del prompt_text, prompt_text_len, prompt_emotion_ids
        del prompt_intensity_ids, prompt_speech_token, prompt_speech_token_len, embedding

        text_emb = self.llm.model.model.embed_tokens(text_token)
        emo_feats = self.emotion_encoder(emotion_ids, intensity_ids)
        text_emb = self.emotion_adapter(text_emb, emo_feats)

        sos_emb = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        lm_input = torch.concat([sos_emb, text_emb, task_id_emb], dim=1)

        min_len = int(text_len.item() * min_token_text_ratio)
        max_len = 200

        for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
            yield token

    @torch.inference_mode()
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):
        """只把真实 EOS 作为停止符，保留原始 scores 的重采样分布。"""
        del uuid
        out_tokens = []
        cache = None
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
            y_pred, cache = self.llm.forward_one_step(lm_input, masks=mask, cache=cache)
            scores = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1).squeeze(0)

            trials = 0
            while True:
                trials += 1
                if trials > 100:
                    raise RuntimeError(
                        "sampling repeatedly returned a non-speech token"
                    )
                top_id = self.sampling(scores.clone(), out_tokens, sampling)
                top_id = int(top_id.item()) if torch.is_tensor(top_id) else int(top_id)
                if top_id == self.eos_token and len(out_tokens) >= min_len:
                    break
                if top_id == self.eos_token or top_id >= self.speech_token_size:
                    continue
                break

            if top_id == self.eos_token:
                break

            yield top_id
            out_tokens.append(top_id)
            lm_input = self.speech_embedding.weight[top_id].reshape(1, 1, -1)
