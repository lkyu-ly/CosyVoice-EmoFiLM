#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper


def extract_speech_tokens(wav_path, ort_session):
    """提取单条语音 token；长于 30 秒的音频不再静默丢弃。"""
    audio, sample_rate = torchaudio.load(str(wav_path), backend="soundfile")
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if audio.ndim != 2 or audio.shape[1] == 0:
        raise ValueError(f"invalid audio shape for {wav_path}: {tuple(audio.shape)}")

    feat = whisper.log_mel_spectrogram(audio, n_mels=128)
    input_names = [item.name for item in ort_session.get_inputs()]
    values = ort_session.run(
        None,
        {
            input_names[0]: feat.detach().cpu().numpy(),
            input_names[1]: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0]
    return np.asarray(values).reshape(-1).astype(np.int64).tolist()


def single_job(utt):
    speech_token = extract_speech_tokens(utt2wav[utt], ort_session)
    return utt, speech_token


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2speech_token = {}
    for future in tqdm(as_completed(all_task)):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    parser.add_argument("--num_thread", type=int, default=8)
    args = parser.parse_args()

    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
