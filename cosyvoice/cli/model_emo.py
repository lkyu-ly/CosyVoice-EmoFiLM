"""CosyVoice2Model_Emotion: 重写 tts/llm_job 支持情感控制。"""
import threading
import uuid as uuid_mod
import torch
from cosyvoice.cli.model import CosyVoice2Model


class CosyVoice2Model_Emotion(CosyVoice2Model):
    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32),
            emotion_ids=None, intensity_ids=None,
            prompt_emotion_ids=None, prompt_intensity_ids=None,
            flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            source_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            stream=False, speed=1.0, **kwargs):
        if stream is True:
            raise NotImplementedError("stream inference not supported for Emo-FiLM")

        this_uuid = str(uuid_mod.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None

        # 默认值 (缺失时报错)
        if emotion_ids is None or intensity_ids is None:
            raise ValueError("emotion_ids and intensity_ids are required for Emo-FiLM inference")
        if prompt_emotion_ids is None:
            prompt_emotion_ids = torch.full((1, prompt_text.shape[1]), 3, dtype=torch.long).to(self.device)
        if prompt_intensity_ids is None:
            prompt_intensity_ids = torch.full((1, prompt_text.shape[1]), 1, dtype=torch.long).to(self.device)

        p = threading.Thread(target=self.llm_job, args=(
            text, emotion_ids, intensity_ids,
            prompt_emotion_ids, prompt_intensity_ids,
            prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid,
        ))
        p.start()

        # non-stream: 等待完成
        p.join()
        this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                         prompt_token=flow_prompt_speech_token,
                                         prompt_feat=prompt_speech_feat,
                                         embedding=flow_embedding,
                                         token_offset=0,
                                         uuid=this_uuid,
                                         finalize=True,
                                         speed=speed)
        yield {"tts_speech": this_tts_speech.cpu()}

        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)

    def llm_job(self, text_token, emotion_ids, intensity_ids,
                prompt_emotion_ids, prompt_intensity_ids,
                prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            text_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
            prompt_text_len = torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device)

            for i in self.llm.inference(
                text_token=text_token.to(self.device),
                text_len=text_len,
                emotion_ids=emotion_ids.to(self.device),
                intensity_ids=intensity_ids.to(self.device),
                prompt_text=prompt_text.to(self.device),
                prompt_text_len=prompt_text_len,
                prompt_emotion_ids=prompt_emotion_ids.to(self.device),
                prompt_intensity_ids=prompt_intensity_ids.to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                embedding=llm_embedding.to(self.device),
                uuid=uuid,
            ):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True
