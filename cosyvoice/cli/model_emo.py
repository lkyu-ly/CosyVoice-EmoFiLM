"""CosyVoice2Model_Emotion: 重写 tts/llm_job 支持情感控制（v2 单流协议）。

llm_job 以 target-only 单流 ``Qwen2LM_Emotion.inference`` 签名调用，不再传
v1 死参数（``prompt_text`` / ``prompt_text_len`` / ``prompt_emotion_ids`` /
``prompt_intensity_ids``）。声学侧 prompt 条件（``flow_prompt_speech_token`` /
``prompt_speech_feat`` / ``flow_embedding``）透传给 Flow/HiFT，不进 LLM 条件。
"""
import threading
import uuid as uuid_mod
import torch
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.emo_checkpoint import load_base_state


class CosyVoice2Model_Emotion(CosyVoice2Model):
    def load(self, llm_model, flow_model, hift_model):
        """加载基础模型；只允许基础 checkpoint 缺失新版情感/任务头模块。"""
        load_base_state(
            self.llm,
            torch.load(llm_model, map_location=self.device, weights_only=True),
        )
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(
            torch.load(flow_model, map_location=self.device, weights_only=True), strict=True)
        self.flow.to(self.device).eval()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in
                           torch.load(hift_model, map_location=self.device, weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32),
            emotion_ids=None, intensity_ids=None,
            flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80),
            stream=False, speed=1.0, decode_config=None, **kwargs):
        if stream is True:
            raise NotImplementedError("stream inference not supported for Emo-FiLM")

        this_uuid = str(uuid_mod.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None

        p = None
        thread_errors = []
        try:
            if emotion_ids is None or intensity_ids is None:
                raise ValueError("emotion_ids and intensity_ids are required for Emo-FiLM inference")

            def run_llm_job():
                try:
                    self.llm_job(
                        text, emotion_ids, intensity_ids,
                        llm_prompt_speech_token, llm_embedding, this_uuid,
                        decode_config=decode_config,
                    )
                except BaseException as exc:
                    thread_errors.append(exc)

            p = threading.Thread(target=run_llm_job)
            p.start()
            p.join()
            if thread_errors:
                raise thread_errors[0]
            # 门控（Task 1 / MAP §3）：非 eos 不得进 Flow/HiFT、不得落正式 WAV。
            # ``llm_job`` 已把 ``Qwen2LM_Emotion.last_decode_result`` 暴露到
            # ``self._last_decode_result``；``inference`` 在非 eos 时不 yield token，
            # 因此 ``tts_speech_token_dict[this_uuid]`` 此时为空。
            decode_result = getattr(self, "_last_decode_result", None)
            finish_reason = getattr(decode_result, "finish_reason", None)
            if finish_reason != "eos":
                # 透传结构化 decode_result，便于下游（T4）审计/写 manifest
                # （schema 强制非 eos 不得携 wav_path）
                yield {
                    "tts_speech": None,
                    "finish_reason": finish_reason,
                    "decode_result": decode_result,
                }
                return
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {
                "tts_speech": this_tts_speech.cpu(),
                "finish_reason": "eos",
                "decode_result": decode_result,
            }
        finally:
            if p is not None and p.is_alive():
                p.join()
            with self.lock:
                for state_dict in (
                    self.tts_speech_token_dict,
                    self.llm_end_dict,
                    self.hift_cache_dict,
                ):
                    state_dict.pop(this_uuid, None)

    def llm_job(self, text_token, emotion_ids, intensity_ids,
                llm_prompt_speech_token, llm_embedding, uuid, decode_config=None):
        """v2 单流推理：以 ``Qwen2LM_Emotion.inference`` 签名调用（无 v1 死参数）。

        ``decode_config`` 透传 yaml 长度参数（min/max_token_text_ratio +
        max_len_hard_cap）到 ``inference``，覆盖其硬编码默认（Task 2 / #3）。
        ``None`` 时**不**传这些 kwargs，回退到 ``Qwen2LM_Emotion.inference`` 默认
        （与 ``Qwen2LM_Emotion.decode`` 默认一致）。
        """
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            text_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)

            inference_kwargs = dict(
                text_token=text_token.to(self.device),
                text_len=text_len,
                emotion_ids=emotion_ids.to(self.device),
                intensity_ids=intensity_ids.to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                prompt_speech_token_len=torch.tensor(
                    [llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                embedding=llm_embedding.to(self.device),
                uuid=uuid,
            )
            # decode_config 非空时覆盖 inference 硬编码默认（历史 bug：透传缺失
            # 导致改 yaml 不生效）。schema 保证三项齐全（build_emofilm_contract）。
            if decode_config is not None:
                inference_kwargs.update(
                    min_token_text_ratio=decode_config["min_token_text_ratio"],
                    max_token_text_ratio=decode_config["max_token_text_ratio"],
                    max_len_hard_cap=decode_config["max_len_hard_cap"],
                )

            for i in self.llm.inference(**inference_kwargs):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True
        # 暴露结构化解码结果供 ``tts`` 门控（Task 1）：
        # ``Qwen2LM_Emotion.inference`` 总会把 ``DecodeResult`` 写入
        # ``self.llm.last_decode_result``（无论 eos / 非 eos）；非 eos 时
        # generator 不 yield token，但 ``DecodeResult`` 仍可读。线程内写、
        # 主线程读（``p.join`` + ``thread_errors`` 已保证可见性）。
        self._last_decode_result = getattr(self.llm, "last_decode_result", None)
