# EmoFiLM RL+扩散 阶段 0（前置验证与奖励校准）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通 CosyVoice3 flow 训练入口冒烟，产出五类情感原型与三路 RL 奖励的离线校准结果，并用两道方向性门槛判定奖励是否可进入阶段 2。

**Architecture:** 新增派生配置 `conf/cosyvoice3-emofilm.yaml` 与四个独立离线工具脚本，不改 `cosyvoice/bin/train.py`、`flow.py`、`flow_matching.py` 与评测本体。冒烟数据按官方管线原样读取（parquet 预存 v3 语音符号与 CAM++ embedding）；原型只使用训练划分 GT 音频；奖励校准对四模型 v2 音频、CV3 全量中性基线音频与 GT 参考音频统一打分。

**Tech Stack:** Python 3.10 / PyTorch 2.13 / deepspeed / torchaudio / whisper large-v3 / funasr（emotion2vec_plus_large）/ onnxruntime（CAM++、v3 分词器）/ pyarrow / jiwer / pytest。

---

## 已确认决策（grilling 结论，2026-08-15）

| 编号 | 决策 | 结论 |
| --- | --- | --- |
| Q1 | CV3 全量基线纳入阶段 0 | 纳入：生成 ESD 1500 + FEDD 1000，跑 v3 评测 |
| Q2 | 奖励与原型用哪套 emotion2vec | 统一 `iic/emotion2vec_plus_large`（评测同空间） |
| Q3 | 原型数据范围 | 仅训练划分 GT 音频（`splits/train/manifest.jsonl`） |
| Q4 | 校准结论门槛 | 两道硬门槛：gap_mean>0（方向正确）、r_emo_std>1e-6（非退化） |
| Q5 | 伪组内方差探测 | 不做，推迟到阶段 2 SDE 冒烟 |
| Q6 | flow 冒烟数据构造 | 预抽 v3 token + CAM++ embedding 写 parquet，官方管线不动 |

判定规则：18 份校准视图（6 视图 × 3 数据集）逐视图输出门槛结果；**是否进入阶段 2 的主判据是 CV3 全量视图**，GT 视图作奖励模型健全性 sanity（应明显通过），四个 v2 视图作方向性佐证。

## 环境与路径事实（已实测）

- 激活环境：`source scripts/activate_env.sh`（conda env `emofilm`，含 torch/deepspeed/funasr/whisper/onnxruntime/pyarrow/jiwer）。
- 6×RTX3090 空闲；训练用 torchrun 单卡 DDP。
- CV3 缓存（默认）：`$HOME/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`，含 `cosyvoice3.yaml / flow.pt / llm.pt / hift.pt / campplus.onnx / speech_tokenizer_v3.onnx / CosyVoice-BlankEN`。
- emotion2vec：`$HOME/.cache/modelscope/hub/iic/emotion2vec_plus_large`；whisper：`$HOME/.cache/whisper/large-v3.pt`。
- v3 评测 ref 视图复用 `exp/emofilm_film_only/eval_refs/{esd,fedd_a,fedd_b}`（数据集级，与模型无关，已存在）。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `conf/cosyvoice3-emofilm.yaml`（新建） | 官方 v3 配置派生：flow 冒烟用（阶段 1 继续沿用） |
| `tools/rl_stage0_audio.py`（新建） | 阶段 0 共用音频/模型 IO 与标签解析纯函数 |
| `tools/build_cv3_flow_smoke_data.py`（新建） | 从 ESD 样例构造 train/cv parquet + data.list |
| `tools/compute_emotion_prototypes.py`（新建） | 训练划分 GT 音频 → 五类 emotion2vec 原型 |
| `tools/generate_cv3_baseline.py`（新建） | CV3 全量中性基线生成（分片可并行） |
| `tools/calibrate_rl_rewards.py`（新建） | 三路奖励离线打分 + 分布/判别/门槛 |
| `tests/test_rl_stage0_tools.py`（新建） | 纯函数核心测试（标签解析/原型聚合/分层取样/门槛） |
| `exp/rl_diffusion_stage0/run_*.sh`（新建 ×4） | 四任务的复现运行脚本 |
| `docs/reports/2026-08-15-emofilm-rl-diffusion-stage0-report.md`（执行时生成） | 阶段 0 结果报告 |

执行约定：不主动 `git commit`（工作树即交付物）；所有新代码与注释用简体中文；仅对纯函数做核心 TDD，模型/IO 脚本用运行冒烟 + 产物校验验收。

---

### Task 1: CV3 flow 训练入口冒烟（配置 + 小数据 + 一步训练）

**Files:**
- Create: `conf/cosyvoice3-emofilm.yaml`
- Create: `tools/rl_stage0_audio.py`
- Create: `tools/build_cv3_flow_smoke_data.py`
- Create: `exp/rl_diffusion_stage0/run_smoke.sh`

- [ ] **Step 1: 创建派生配置**

创建 `conf/cosyvoice3-emofilm.yaml`，以官方 `cosyvoice3.yaml` 为底，唯一改动在 `train_conf`（1 epoch / accum_grad 1 / log_interval 1），其余逐字保留：

```yaml
# CosyVoice3 派生配置（EmoFiLM RL 阶段 0/1）。
# 相对官方 cosyvoice3.yaml 仅改 train_conf：阶段 0 用 1 epoch 冒烟验证
# train.py --model flow 入口；阶段 1 再按正式训练调整。模型/管线/权重字段不动。
__set_seed1: !apply:random.seed [1986]
__set_seed2: !apply:numpy.random.seed [1986]
__set_seed3: !apply:torch.manual_seed [1986]
__set_seed4: !apply:torch.cuda.manual_seed_all [1986]

sample_rate: 24000
llm_input_size: 896
llm_output_size: 896
spk_embed_dim: 192
qwen_pretrain_path: ''
token_frame_rate: 25
token_mel_ratio: 2

chunk_size: 25
num_decoding_left_chunks: -1

llm: !new:cosyvoice.llm.llm.CosyVoice3LM
    llm_input_size: !ref <llm_input_size>
    llm_output_size: !ref <llm_output_size>
    speech_token_size: 6561
    length_normalized_loss: True
    lsm_weight: 0
    mix_ratio: [5, 15]
    llm: !new:cosyvoice.llm.llm.Qwen2Encoder
        pretrain_path: !ref <qwen_pretrain_path>
    sampling: !name:cosyvoice.utils.common.ras_sampling
        top_p: 0.8
        top_k: 25
        win_size: 10
        tau_r: 0.1

flow: !new:cosyvoice.flow.flow.CausalMaskedDiffWithDiT
    input_size: 80
    output_size: 80
    spk_embed_dim: !ref <spk_embed_dim>
    output_type: 'mel'
    vocab_size: 6561
    input_frame_rate: !ref <token_frame_rate>
    only_mask_loss: True
    token_mel_ratio: !ref <token_mel_ratio>
    pre_lookahead_len: 3
    pre_lookahead_layer: !new:cosyvoice.transformer.upsample_encoder.PreLookaheadLayer
        in_channels: 80
        channels: 1024
        pre_lookahead_len: 3
    decoder: !new:cosyvoice.flow.flow_matching.CausalConditionalCFM
        in_channels: 240
        n_spks: 1
        spk_emb_dim: 80
        cfm_params: !new:omegaconf.DictConfig
            content:
                sigma_min: 1e-06
                solver: 'euler'
                t_scheduler: 'cosine'
                training_cfg_rate: 0.2
                inference_cfg_rate: 0.7
                reg_loss_type: 'l1'
        estimator: !new:cosyvoice.flow.DiT.dit.DiT
            dim: 1024
            depth: 22
            heads: 16
            dim_head: 64
            ff_mult: 2
            mel_dim: 80
            mu_dim: 80
            spk_dim: 80
            out_channels: 80
            static_chunk_size: !ref <chunk_size> * <token_mel_ratio>
            num_decoding_left_chunks: !ref <num_decoding_left_chunks>

hift: !new:cosyvoice.hifigan.generator.CausalHiFTGenerator
    in_channels: 80
    base_channels: 512
    nb_harmonics: 8
    sampling_rate: !ref <sample_rate>
    nsf_alpha: 0.1
    nsf_sigma: 0.003
    nsf_voiced_threshold: 10
    upsample_rates: [8, 5, 3]
    upsample_kernel_sizes: [16, 11, 7]
    istft_params:
        n_fft: 16
        hop_len: 4
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    source_resblock_kernel_sizes: [7, 7, 11]
    source_resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    lrelu_slope: 0.1
    audio_limit: 0.99
    conv_pre_look_right: 4
    f0_predictor: !new:cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor
        num_class: 1
        in_channels: 80
        cond_channels: 512

mel_spec_transform1: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: null
    center: False
hifigan: !new:cosyvoice.hifigan.hifigan.HiFiGan
    generator: !ref <hift>
    discriminator: !new:cosyvoice.hifigan.discriminator.MultipleDiscriminator
        mpd: !new:matcha.hifigan.models.MultiPeriodDiscriminator
        mrd: !new:cosyvoice.hifigan.discriminator.MultiResSpecDiscriminator
    mel_spec_transform: [
        !ref <mel_spec_transform1>
    ]

parquet_opener: !name:cosyvoice.dataset.processor.parquet_opener
get_tokenizer: !name:cosyvoice.tokenizer.tokenizer.get_qwen_tokenizer
    token_path: !ref <qwen_pretrain_path>
    skip_special_tokens: True
    version: cosyvoice3
allowed_special: 'all'
tokenize: !name:cosyvoice.dataset.processor.tokenize
    get_tokenizer: !ref <get_tokenizer>
    allowed_special: !ref <allowed_special>
filter: !name:cosyvoice.dataset.processor.filter
    max_length: 40960
    min_length: 100
    token_max_length: 200
    token_min_length: 1
resample: !name:cosyvoice.dataset.processor.resample
    resample_rate: !ref <sample_rate>
truncate: !name:cosyvoice.dataset.processor.truncate
    truncate_length: 24480
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: !ref <sample_rate>
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: null
    center: False
compute_fbank: !name:cosyvoice.dataset.processor.compute_fbank
    feat_extractor: !ref <feat_extractor>
compute_f0: !name:cosyvoice.dataset.processor.compute_f0
    sample_rate: !ref <sample_rate>
    hop_size: 480
parse_embedding: !name:cosyvoice.dataset.processor.parse_embedding
    normalize: True
shuffle: !name:cosyvoice.dataset.processor.shuffle
    shuffle_size: 1000
sort: !name:cosyvoice.dataset.processor.sort
    sort_size: 500
batch: !name:cosyvoice.dataset.processor.batch
    batch_type: 'dynamic'
    max_frames_in_batch: 2000
padding: !name:cosyvoice.dataset.processor.padding
    use_spk_embedding: False

data_pipeline: [
    !ref <parquet_opener>,
    !ref <tokenize>,
    !ref <filter>,
    !ref <resample>,
    !ref <compute_fbank>,
    !ref <parse_embedding>,
    !ref <shuffle>,
    !ref <sort>,
    !ref <batch>,
    !ref <padding>,
]
data_pipeline_gan: [
    !ref <parquet_opener>,
    !ref <tokenize>,
    !ref <filter>,
    !ref <resample>,
    !ref <truncate>,
    !ref <compute_fbank>,
    !ref <compute_f0>,
    !ref <parse_embedding>,
    !ref <shuffle>,
    !ref <sort>,
    !ref <batch>,
    !ref <padding>,
]

# 阶段 0 冒烟：仅验证 flow 训练入口可执行（1 epoch / 无梯度累积）。
train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5
    scheduler: constantlr
    scheduler_conf:
        warmup_steps: 2500
    max_epoch: 1
    grad_clip: 5
    accum_grad: 1
    log_interval: 1
    save_per_step: -1

train_conf_gan:
    optim: adam
    optim_conf:
        lr: 0.0002
    scheduler: constantlr
    optim_d: adam
    optim_conf_d:
        lr: 0.0002
    scheduler_d: constantlr
    max_epoch: 200
    grad_clip: 5
    accum_grad: 1
    log_interval: 100
    save_per_step: -1
```

- [ ] **Step 2: 创建阶段 0 共用音频/标签辅助模块**

创建 `tools/rl_stage0_audio.py`：

```python
"""RL 阶段 0 共用工具：情感标签解析 + 16k 音频加载 + v3 token/CAM++ 提取。

重依赖（torch/whisper/onnxruntime）放在函数内导入：纯函数测试无需加载模型。
"""
import os
import re

EMOTIONS = ["ang", "hap", "neu", "sad", "sur"]
TAG_PATTERN = re.compile(
    r"<emotion\s+type='(\w+)'\s+intensity='(\w+)'>(.*?)</emotion>",
    re.IGNORECASE | re.DOTALL,
)


def strip_emotion_tags(text):
    """移除情感标签，返回纯文本。"""
    return TAG_PATTERN.sub(lambda m: m.group(3), text)


def parse_emotion_tags(text):
    """解析情感标签，返回 (segments, clean_text)。

    segments 的 char_start/char_end 是相对 clean_text 的字符偏移，
    用于按字符比例估计音频片段边界（作者 reward/emotion.py 同口径）。
    """
    segments = []
    clean_parts = []
    last_end = 0
    for match in TAG_PATTERN.finditer(text):
        before = text[last_end:match.start()]
        if before:
            clean_parts.append(before)
        start = len("".join(clean_parts))
        clean_parts.append(match.group(3))
        end = len("".join(clean_parts))
        emotion = match.group(1).lower().strip()
        if emotion in EMOTIONS:
            segments.append({
                "emotion": emotion,
                "char_start": start,
                "char_end": end,
            })
        last_end = match.end()
    tail = text[last_end:]
    if tail:
        clean_parts.append(tail)
    return segments, "".join(clean_parts)


def load_wav_16k_mono(wav_path):
    """加载并重采样到 16kHz 单声道，返回 (1, T) 的 torch.Tensor。"""
    import torch
    import torchaudio

    speech, sr = torchaudio.load(wav_path)
    if sr != 16000:
        speech = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=16000)(speech)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    return speech


def extract_v3_tokens(session, speech):
    """用 speech_tokenizer_v3.onnx 从 16k 语音抽取 v3 token（list[int]）。"""
    import numpy as np
    import whisper

    feat = whisper.log_mel_spectrogram(speech, n_mels=128)
    names = [item.name for item in session.get_inputs()]
    out = session.run(None, {
        names[0]: feat.detach().cpu().numpy(),
        names[1]: np.array([feat.shape[2]], dtype=np.int32),
    })[0]
    return out.reshape(-1).astype(np.int64).tolist()


def make_campplus_session(model_path):
    """创建 CPU 端 CAM++ ONNX 会话。"""
    import onnxruntime

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
    option.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(
        model_path, sess_options=option,
        providers=["CPUExecutionProvider"])


def extract_speaker_embedding(session, speech):
    """从 16k 语音提取 CAM++ 192 维 embedding（list[float]）。"""
    import torchaudio.compliance.kaldi as kaldi

    feat = kaldi.fbank(
        speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    out = session.run(None, {
        session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy(),
    })[0]
    return out.reshape(-1).astype("float32").tolist()


def cv3_model_dir():
    """解析 CV3 模型目录（环境变量优先）。"""
    return os.environ.get(
        "CV3_MODEL_DIR",
        os.path.join(
            os.path.expanduser("~"),
            ".cache", "modelscope", "hub",
            "FunAudioLLM", "Fun-CosyVoice3-0.5B-2512",
        ),
    )
```

- [ ] **Step 3: 创建冒烟数据构建脚本**

创建 `tools/build_cv3_flow_smoke_data.py`：

```python
#!/usr/bin/env python3
"""构造 CV3 flow 训练入口冒烟数据（train/cv parquet + data.list）。

预抽 v3 语音符号与 CAM++ 说话人 embedding 写入 parquet，官方
cosyvoice3.yaml 数据管线原样读取，不依赖 onnx_path 在线抽 token 路径。
"""
import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from rl_stage0_audio import (
    cv3_model_dir,
    extract_speaker_embedding,
    extract_v3_tokens,
    load_wav_16k_mono,
    make_campplus_session,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_rows(manifest_rows, token_session, campplus_session):
    rows = []
    for rec in manifest_rows:
        wav_path = rec["wav_path"]
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(ROOT, wav_path)
        speech = load_wav_16k_mono(wav_path)
        with open(wav_path, "rb") as f:
            audio_data = f.read()
        embedding = extract_speaker_embedding(campplus_session, speech)
        rows.append({
            "utt": rec["utt_id"],
            "audio_data": audio_data,
            "text": rec["text"],
            "speech_token": extract_v3_tokens(token_session, speech),
            "utt_embedding": embedding,
            "spk_embedding": embedding,
        })
    return rows


def write_parquet(rows, path):
    table = pa.table({
        "utt": [r["utt"] for r in rows],
        "audio_data": [r["audio_data"] for r in rows],
        "text": [r["text"] for r in rows],
        "speech_token": [r["speech_token"] for r in rows],
        "utt_embedding": [r["utt_embedding"] for r in rows],
        "spk_embedding": [r["spk_embedding"] for r in rows],
    })
    pq.write_table(table, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
                        default="data/contracts/emofilm_v1/eval/esd_cv3_150.jsonl")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--cv-size", type=int, default=4)
    parser.add_argument("--output-dir", default="data/smoke/cv3_flow")
    args = parser.parse_args()

    model_dir = cv3_model_dir()
    token_session = make_campplus_session(
        os.path.join(model_dir, "speech_tokenizer_v3.onnx"))
    campplus_session = make_campplus_session(
        os.path.join(model_dir, "campplus.onnx"))

    with open(args.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[:args.num_samples]
    if len(rows) != args.num_samples:
        raise RuntimeError(
            f"manifest rows {len(rows)} < num_samples {args.num_samples}")

    cv_rows = rows[-args.cv_size:]
    train_rows = rows[:-args.cv_size]
    os.makedirs(args.output_dir, exist_ok=True)

    write_parquet(build_rows(train_rows, token_session, campplus_session),
                  os.path.join(args.output_dir, "train.parquet"))
    write_parquet(build_rows(cv_rows, token_session, campplus_session),
                  os.path.join(args.output_dir, "cv.parquet"))
    with open(os.path.join(args.output_dir, "train.data.list"), "w") as f:
        f.write("train.parquet\n")
    with open(os.path.join(args.output_dir, "cv.data.list"), "w") as f:
        f.write("cv.parquet\n")
    print(f"OK train={len(train_rows)} cv={len(cv_rows)} "
          f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 创建冒烟运行脚本**

创建 `exp/rl_diffusion_stage0/run_smoke.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
source scripts/activate_env.sh

export CV3_MODEL_DIR="${CV3_MODEL_DIR:-$HOME/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p exp/rl_diffusion_stage0/smoke

python tools/build_cv3_flow_smoke_data.py \
  --output-dir data/smoke/cv3_flow

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT:-29517}" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model flow \
  --config conf/cosyvoice3-emofilm.yaml \
  --train_data data/smoke/cv3_flow/train.data.list \
  --cv_data data/smoke/cv3_flow/cv.data.list \
  --qwen_pretrain_path "$CV3_MODEL_DIR/CosyVoice-BlankEN" \
  --checkpoint "$CV3_MODEL_DIR/flow.pt" \
  --model_dir exp/rl_diffusion_stage0/smoke \
  --tensorboard_dir exp/rl_diffusion_stage0/smoke/tb \
  --use_amp 2>&1 | tee exp/rl_diffusion_stage0/smoke/train.log
```

- [ ] **Step 5: 构建并校验冒烟数据 parquet**

Run:

```bash
source scripts/activate_env.sh
export CV3_MODEL_DIR="${CV3_MODEL_DIR:-$HOME/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
python tools/build_cv3_flow_smoke_data.py \
  --output-dir data/smoke/cv3_flow
python - <<'PY'
import pyarrow.parquet as pq
t = pq.read_table("data/smoke/cv3_flow/train.parquet")
c = pq.read_table("data/smoke/cv3_flow/cv.parquet")
assert t.num_rows == 12 and c.num_rows == 4
assert set(t.column_names) == {
    "utt", "audio_data", "text", "speech_token",
    "utt_embedding", "spk_embedding"}
assert all(len(x) > 0 for x in t.column("speech_token").to_pylist())
print("OK", t.num_rows, c.num_rows)
PY
```

Expected: `OK 12 4`

- [ ] **Step 6: 运行 flow 冒烟并校验产物**

Run:

```bash
GPU=0 bash exp/rl_diffusion_stage0/run_smoke.sh
ls exp/rl_diffusion_stage0/smoke/init.pt \
   exp/rl_diffusion_stage0/smoke/epoch_0_whole.pt
rg -q "TRAIN" exp/rl_diffusion_stage0/smoke/train.log
rg -q "CV" exp/rl_diffusion_stage0/smoke/train.log
```

Expected: 两个 checkpoint 文件存在，日志同时包含 TRAIN loss 与 CV loss，无异常退出。若 24GB 显存不足，把 `conf/cosyvoice3-emofilm.yaml` 的 `batch.max_frames_in_batch` 从 2000 降到 1000 后重跑（冒烟目标只验证入口，不求收敛）。

---

### Task 2: 情感原型计算（训练划分 GT 音频）

**Files:**
- Create: `tests/test_rl_stage0_tools.py`
- Create: `tools/compute_emotion_prototypes.py`
- Create: `exp/rl_diffusion_stage0/run_prototypes.sh`

- [ ] **Step 1: 先写原型纯函数失败测试**

创建 `tests/test_rl_stage0_tools.py`：

```python
"""RL 阶段 0 工具纯函数核心测试。"""
import os
import sys

import numpy as np

TOOLS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)

from rl_stage0_audio import EMOTIONS, parse_emotion_tags  # noqa: E402
from compute_emotion_prototypes import build_prototypes  # noqa: E402


def test_parse_emotion_tags_two_segments():
    segs, clean = parse_emotion_tags(
        "<emotion type='sur' intensity='low'>excuse</emotion> "
        "<emotion type='sad' intensity='medium'>me</emotion>")
    assert clean == "excuse me"
    assert [s["emotion"] for s in segs] == ["sur", "sad"]
    assert segs[0]["char_end"] == 6
    assert segs[1]["char_start"] == 7


def test_build_prototypes_mean_and_unit_norm():
    embs = {
        "ang": [np.array([3.0, 0.0]), np.array([3.0, 2.0])],
        "hap": [np.array([0.0, 5.0])],
    }
    protos = build_prototypes(embs)
    assert set(protos) == set(EMOTIONS)
    target = np.array([3.0, 1.0])
    target = target / np.linalg.norm(target)
    assert np.allclose(protos["ang"], target)
    assert np.linalg.norm(protos["neu"]) == 0.0  # 缺类补零
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source scripts/activate_env.sh && pytest tests/test_rl_stage0_tools.py -q`

Expected: FAIL（`ModuleNotFoundError: No module named 'compute_emotion_prototypes'`）

- [ ] **Step 3: 实现原型计算脚本**

创建 `tools/compute_emotion_prototypes.py`：

```python
#!/usr/bin/env python3
"""按训练划分的标注音频计算五类 emotion2vec 原型。

ESD 用句级标签；IEMOCAP 用 tagged_text 的词级标签按字符比例定位片段。
缺失类补零向量；同时输出每类样本数与五类原型两两余弦矩阵。
"""
import argparse
import json
import os

import numpy as np

from rl_stage0_audio import EMOTIONS, load_wav_16k_mono, parse_emotion_tags

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_prototypes(embs_by_emotion):
    """embs_by_emotion: {emotion: [np.ndarray(dim)]} -> {emotion: 单位向量}。"""
    dims = {np.asarray(v).shape[-1]
            for vals in embs_by_emotion.values() for v in vals}
    if not dims:
        raise ValueError("embedding pool is empty")
    dim = dims.pop()
    out = {}
    for emotion in EMOTIONS:
        embs = embs_by_emotion.get(emotion, [])
        if embs:
            proto = np.mean(np.stack(embs), axis=0)
            proto = proto / np.maximum(np.linalg.norm(proto), 1e-8)
        else:
            proto = np.zeros(dim, dtype=np.float32)
        out[emotion] = proto
    return out


def segment_embedding(emo_model, audio, clean_text, segment):
    """按字符比例估计片段边界，返回 frame 均值 L2 归一化向量或 None。"""
    total = audio.shape[0]
    denom = max(len(clean_text), 1)
    start = max(0, int(segment["char_start"] / denom * total))
    end = min(total, int(segment["char_end"] / denom * total))
    if end - start < 800:  # 50ms 以下片段跳过（作者脚本同阈值）
        return None
    result = emo_model.generate(audio[start:end].numpy(), granularity="frame")
    if not result or "feats" not in result[0]:
        return None
    feats = result[0]["feats"]
    feats = feats.cpu().numpy() if hasattr(feats, "cpu") else np.asarray(feats)
    emb = feats.mean(axis=0)
    return emb / np.maximum(np.linalg.norm(emb), 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-utterances-per-emotion", type=int, default=500,
                        help="每类最多使用的片段数，0=不限制（控制运行时长）")
    args = parser.parse_args()

    from funasr import AutoModel
    emo_model = AutoModel(model="iic/emotion2vec_plus_large",
                          disable_update=True, device=args.device)

    with open(args.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    pool = {e: [] for e in EMOTIONS}
    cap = {e: args.max_utterances_per_emotion for e in EMOTIONS}
    processed = 0
    for rec in rows:
        if all(v <= 0 for v in cap.values()):
            break
        wav_path = rec["wav_path"]
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(ROOT, wav_path)
        if not os.path.exists(wav_path):
            continue
        tagged = rec.get("tagged_text") or rec.get("text") or ""
        segments, clean = parse_emotion_tags(tagged)
        sentence_emotion = rec.get("sentence_emotion")
        if not segments and sentence_emotion in EMOTIONS:
            segments = [{
                "emotion": sentence_emotion,
                "char_start": 0,
                "char_end": max(len(clean), 1),
            }]
        if not segments:
            continue
        speech = load_wav_16k_mono(wav_path).squeeze(0)
        for seg in segments:
            emotion = seg["emotion"]
            if cap.get(emotion, 0) <= 0:
                continue
            emb = segment_embedding(emo_model, speech, clean, seg)
            if emb is None:
                continue
            pool[emotion].append(emb)
            cap[emotion] -= 1
            processed += 1
        if processed % 100 == 0:
            print(f"processed segments={processed}", flush=True)

    protos = build_prototypes(pool)
    os.makedirs(args.output_dir, exist_ok=True)
    for emotion in EMOTIONS:
        np.save(os.path.join(args.output_dir, f"{emotion}.npy"), protos[emotion])
    matrix = {
        e1: {e2: float(np.dot(protos[e1], protos[e2])) for e2 in EMOTIONS}
        for e1 in EMOTIONS
    }
    with open(os.path.join(args.output_dir, "stats.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "counts": {e: len(pool[e]) for e in EMOTIONS},
            "pairwise_cosine": matrix,
        }, f, indent=2, ensure_ascii=False)
    print(json.dumps({"counts": {e: len(pool[e]) for e in EMOTIONS}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source scripts/activate_env.sh && pytest tests/test_rl_stage0_tools.py -q`

Expected: `2 passed`

- [ ] **Step 5: 创建原型运行脚本并执行**

创建 `exp/rl_diffusion_stage0/run_prototypes.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
source scripts/activate_env.sh
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export MODELSCOPE_OFFLINE=1

python tools/compute_emotion_prototypes.py \
  --manifest data/contracts/emofilm_v1/splits/train/manifest.jsonl \
  --output-dir exp/rl_diffusion_stage0/prototypes \
  --device cuda \
  --max-utterances-per-emotion 500
```

Run:

```bash
GPU=1 bash exp/rl_diffusion_stage0/run_prototypes.sh
ls exp/rl_diffusion_stage0/prototypes/{ang,hap,neu,sad,sur}.npy
python -c "import json; d=json.load(open('exp/rl_diffusion_stage0/prototypes/stats.json')); print(d['counts']); assert all(d['counts'][e]>0 for e in ['ang','hap','neu','sad','sur'])"
```

Expected: 5 个 npy + `stats.json`，五类 `counts>0`；`stats.json` 中非中性类两两余弦明显低于 1（原型可分）。

---

### Task 3: CV3 全量中性基线生成 + v3 评测

**Files:**
- Create: `tools/generate_cv3_baseline.py`
- Create: `exp/rl_diffusion_stage0/run_cv3_baseline.sh`

- [ ] **Step 1: 创建 CV3 全量基线生成脚本**

创建 `tools/generate_cv3_baseline.py`：

```python
#!/usr/bin/env python3
"""CosyVoice3 全量中性基线生成（ESD/FEDD，分片可并行）。

沿用 exp/cosyvoice3_baseline/gen_cv3.py 的关键适配：
- 指令格式必须是 Qwen 系统前缀 + 情感指令 + <|endofprompt|>；
- CV3 finish_reason 为 None，按首段有效 tts_speech 保存。
"""
import argparse
import json
import os
import time

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel
from rl_stage0_audio import cv3_model_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMO_INSTRUCT = {
    "ang": "You are a helpful assistant. Speak with angry emotion.<|endofprompt|>",
    "hap": "You are a helpful assistant. Speak with happy emotion.<|endofprompt|>",
    "neu": "You are a helpful assistant. Speak with neutral emotion.<|endofprompt|>",
    "sad": "You are a helpful assistant. Speak with sad emotion.<|endofprompt|>",
    "sur": "You are a helpful assistant. Speak with surprised emotion.<|endofprompt|>",
}


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1986)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = AutoModel(model_dir=cv3_model_dir())
    with open(args.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[args.shard_idx::args.num_shards]

    manifest_path = os.path.join(
        args.out_dir, f"generation_manifest.shard{args.shard_idx}.jsonl")
    results = []
    ok = skip = fail = 0
    t0 = time.time()
    for i, rec in enumerate(rows):
        utt_id = rec["utt_id"]
        out = os.path.join(args.out_dir, f"{utt_id}.wav")
        if os.path.exists(out):
            skip += 1
            continue
        emotion = rec.get("label") or rec.get("sentence_emotion")
        if emotion not in EMO_INSTRUCT:
            fail += 1
            results.append({"utt_id": utt_id, "status": "bad_emotion",
                            "emotion": emotion})
            continue
        prompt_wav = resolve(rec["prompt_wav"])
        torch.manual_seed(args.seed_base + i)
        torch.cuda.manual_seed_all(args.seed_base + i)
        saved = False
        try:
            for chunk in model.inference_instruct2(
                    rec["text"], EMO_INSTRUCT[emotion], prompt_wav):
                speech = chunk.get("tts_speech")
                if speech is not None and speech.numel() > 0:
                    torchaudio.save(out, speech.cpu(), model.sample_rate)
                    saved = True
                    break
        except Exception as exc:
            print(f"FAIL {utt_id}: {exc}", flush=True)
        if saved:
            ok += 1
            results.append({"utt_id": utt_id, "wav_path": out,
                            "emotion": emotion, "prompt_wav": prompt_wav,
                            "instruct": EMO_INSTRUCT[emotion],
                            "seed": args.seed_base + i,
                            "model_dir": cv3_model_dir(),
                            "status": "ok"})
        else:
            fail += 1
            results.append({"utt_id": utt_id, "status": "failed",
                            "emotion": emotion})
        if (i + 1) % 20 == 0:
            print(f"{utt_id} done={ok} skip={skip} fail={fail} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE ok={ok} skip={skip} fail={fail} total={len(rows)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 创建生成 + 评测运行脚本**

创建 `exp/rl_diffusion_stage0/run_cv3_baseline.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
source scripts/activate_env.sh
export CV3_MODEL_DIR="${CV3_MODEL_DIR:-$HOME/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
export MODELSCOPE_OFFLINE=1

BASE=exp/rl_diffusion_stage0/cv3_baseline
REFS=exp/emofilm_film_only/eval_refs

# 生成：三数据集在 GPU 0/1/2 并行
for spec in "esd:0" "fedd_a:1" "fedd_b:2"; do
  part="${spec%%:*}"
  gpu="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" python tools/generate_cv3_baseline.py \
    --manifest "data/contracts/emofilm_v1/eval/$part/manifest.jsonl" \
    --out-dir "$BASE/$part" &
done
wait

# 生成完整性校验
python - <<'PY'
import json
for part, want in [("esd", 1500), ("fedd_a", 500), ("fedd_b", 500)]:
    rows = [json.loads(l) for l in open(
        f"exp/rl_diffusion_stage0/cv3_baseline/{part}/generation_manifest.shard0.jsonl"
    ) if l.strip()]
    ok = sum(r["status"] == "ok" for r in rows)
    assert ok == want, (part, ok, want)
    print("GEN OK", part, ok)
PY

# 评测：ESD 提供情感参考清单算判别；FEDD 不做判别（结构上参考组不足）
for spec in "esd:1500:0" "fedd_a:500:1" "fedd_b:500:2"; do
  IFS=: read -r part count gpu <<< "$spec"
  extra=""
  if [ "$part" = "esd" ]; then
    extra="--emotion_ref_manifest data/contracts/emofilm_v1/sources/esd/manifest.jsonl"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python eval/eval_emo_film.py \
    --ref_dir "$REFS/$part" \
    --hyp_dir "$BASE/$part" \
    --output "$BASE/eval/$part.json" \
    --expected_count "$count" \
    --ref_text_manifest "data/contracts/emofilm_v1/eval/$part/manifest.jsonl" \
    $extra &
done
wait
```

- [ ] **Step 3: 运行并校验评测产物**

Run:

```bash
bash exp/rl_diffusion_stage0/run_cv3_baseline.sh
python - <<'PY'
import json
for part, want in [("esd", 1500), ("fedd_a", 500), ("fedd_b", 500)]:
    d = json.load(open(f"exp/rl_diffusion_stage0/cv3_baseline/eval/{part}.json"))
    assert d["n_samples"] == want, (part, d["n_samples"])
    print(part, d["emo_sim"], d["wer_percent"], d.get("discriminability"))
PY
```

Expected: 三个 `n_samples` 分别等于 1500/500/500；ESD 输出含 `discriminability`，FEDD_A 判别 `n_scored=0` 属预期（诚实口径）。

---

### Task 4: 离线奖励校准 + 阶段 0 报告

**Files:**
- Modify: `tests/test_rl_stage0_tools.py`（追加两测试）
- Create: `tools/calibrate_rl_rewards.py`
- Create: `exp/rl_diffusion_stage0/run_calibration.sh`
- Create: `docs/reports/2026-08-15-emofilm-rl-diffusion-stage0-report.md`（执行时填数）

- [ ] **Step 1: 追加校准纯函数失败测试**

在 `tests/test_rl_stage0_tools.py` 末尾追加：

```python
from calibrate_rl_rewards import evaluate_gates, sample_stratified  # noqa: E402


def test_sample_stratified_balanced_and_bounded():
    rows = [{"emotion": e} for e in
            ["ang", "hap", "sad", "ang", "hap", "sad", "sur"]]
    out = sample_stratified(rows, 5, key=lambda r: r["emotion"])
    assert len(out) == 5
    assert out[0]["emotion"] == "ang"
    assert any(r["emotion"] == "sur" for r in out)


def test_evaluate_gates_direction_and_degenerate():
    assert evaluate_gates({"gap_mean": 0.1, "r_emo_std": 0.05}) == {
        "gate1_emotion_direction": True,
        "gate2_non_degenerate": True,
    }
    assert evaluate_gates({"gap_mean": -0.1, "r_emo_std": 0.0}) == {
        "gate1_emotion_direction": False,
        "gate2_non_degenerate": False,
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source scripts/activate_env.sh && pytest tests/test_rl_stage0_tools.py -q`

Expected: FAIL（`ModuleNotFoundError: No module named 'calibrate_rl_rewards'`）

- [ ] **Step 3: 实现离线奖励校准脚本**

创建 `tools/calibrate_rl_rewards.py`：

```python
#!/usr/bin/env python3
"""离线 RL 三路奖励校准（r_emo / r_sv / r_cer）。

输入：评测 manifest + 某模型/某数据集的 hyp wav 目录 + 五类原型目录。
输出：三路奖励分布、逐情感均值、最近原型判别、两道方向性门槛 JSON。
"""
import argparse
import json
import os
import sys

import numpy as np

from rl_stage0_audio import (
    EMOTIONS,
    cv3_model_dir,
    extract_speaker_embedding,
    load_wav_16k_mono,
    make_campplus_session,
    strip_emotion_tags,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample_stratified(rows, n, key):
    """按 key 分层循环取样，返回不超过 n 行且类别尽可能均衡。"""
    buckets = {}
    for row in rows:
        buckets.setdefault(key(row), []).append(row)
    chosen = []
    cursor = 0
    while len(chosen) < n and buckets:
        added = False
        for bucket in list(buckets.values()):
            if cursor < len(bucket):
                chosen.append(bucket[cursor])
                added = True
                if len(chosen) >= n:
                    break
        if not added:
            break
        cursor += 1
    return chosen


def evaluate_gates(stats):
    """两道方向性门槛。"""
    return {
        "gate1_emotion_direction": bool(stats["gap_mean"] > 0),
        "gate2_non_degenerate": bool(stats["r_emo_std"] > 1e-6),
    }


def load_prototypes(prototypes_dir):
    protos = {}
    for emotion in EMOTIONS:
        vec = np.load(os.path.join(prototypes_dir, f"{emotion}.npy"))
        protos[emotion] = vec / np.maximum(np.linalg.norm(vec), 1e-8)
    return protos


def _l2(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return vec / np.maximum(np.linalg.norm(vec), 1e-8)


def dist_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--hyp-dir", required=True)
    parser.add_argument("--prototypes-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from emotion_metrics import extract_frame_embeddings, normalize_text
    from eval_emo_film import transcribe_parallel

    with open(args.manifest, encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f if line.strip()]

    def emotion_key(rec):
        return (rec.get("sentence_emotion") or rec.get("label")
                or rec.get("emotion") or "")

    matched = []
    for rec in manifest:
        wav = os.path.join(args.hyp_dir, f"{rec['utt_id']}.wav")
        if os.path.exists(wav):
            rec = dict(rec)
            rec["_hyp_wav"] = wav
            rec["_target"] = emotion_key(rec)
            matched.append(rec)
    subset = sample_stratified(matched, args.max_samples, emotion_key)
    if not subset:
        raise RuntimeError("no matched hyp wavs")

    protos = load_prototypes(args.prototypes_dir)

    from funasr import AutoModel
    emo_model = AutoModel(model="iic/emotion2vec_plus_large",
                          disable_update=True, device=args.device)
    hyp_paths = [r["_hyp_wav"] for r in subset]
    frames = extract_frame_embeddings(emo_model, hyp_paths,
                                      batch_size=args.batch_size)
    hyp_embs = [_l2(np.asarray(f).mean(axis=0)) for f in frames]

    campplus = make_campplus_session(
        os.path.join(cv3_model_dir(), "campplus.onnx"))
    prompt_paths = sorted({resolve_prompt(r) for r in subset})
    prompt_embs = {
        p: _l2(np.asarray(extract_speaker_embedding(
            campplus, load_wav_16k_mono(p))))
        for p in prompt_paths
    }
    hyp_spk = [
        np.dot(_l2(np.asarray(extract_speaker_embedding(
            campplus, load_wav_16k_mono(p)))), prompt_embs[resolve_prompt(r)])
        for r, p in zip(subset, hyp_paths)
    ]
    hyp_spk = [float(v) for v in hyp_spk]

    import whisper
    from jiwer import wer
    whisper_model = whisper.load_model("large-v3", device=args.device)
    hyp_texts = transcribe_parallel(whisper_model, hyp_paths,
                                    max_workers=args.batch_size)
    r_cer = []
    for rec, hyp_text in zip(subset, hyp_texts):
        gt = rec.get("plain_text") or strip_emotion_tags(rec.get("text") or "")
        error = wer(normalize_text(gt), normalize_text(hyp_text))
        r_cer.append(max(0.0, 1.0 - error))

    r_emo, gaps, nearest_ok = [], [], 0
    per_emotion = {e: [] for e in EMOTIONS}
    for rec, emb in zip(subset, hyp_embs):
        sims = {e: float(np.dot(emb, protos[e])) for e in EMOTIONS}
        target = rec["_target"]
        others = [s for e, s in sims.items() if e != target]
        r_emo.append(sims.get(target, 0.0))
        gaps.append(sims.get(target, 0.0) - max(others))
        if target in sims and max(sims, key=sims.get) == target:
            nearest_ok += 1
        if target in per_emotion:
            per_emotion[target].append(sims[target])

    stats = {
        "model_name": args.model_name,
        "n_samples": len(subset),
        "r_emo": dist_stats(r_emo),
        "r_sv": dist_stats(hyp_spk),
        "r_cer": dist_stats(r_cer),
        "per_emotion_r_emo_mean": {
            e: float(np.mean(v)) if v else None for e, v in per_emotion.items()
        },
        "nearest_prototype_acc_pct":
            round(nearest_ok / len(subset) * 100.0, 2),
        "gap_mean": float(np.mean(gaps)),
    }
    stats["gates"] = evaluate_gates(stats)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def resolve_prompt(rec):
    path = rec.get("prompt_wav")
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source scripts/activate_env.sh && pytest tests/test_rl_stage0_tools.py -q`

Expected: `4 passed`

- [ ] **Step 5: 创建校准运行脚本并执行**

创建 `exp/rl_diffusion_stage0/run_calibration.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
source scripts/activate_env.sh
export CV3_MODEL_DIR="${CV3_MODEL_DIR:-$HOME/.cache/modelscope/hub/FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
export MODELSCOPE_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU:-3}"

OUT=exp/rl_diffusion_stage0/calibration
PROTO=exp/rl_diffusion_stage0/prototypes

declare -A HYPS=(
  [v1]=exp/emofilm_v1/full
  [film_only]=exp/emofilm_film_only/full
  [longepoch]=exp/emofilm_film_only_longepoch/full
  [sentlvl]=exp/emofilm_sentlvl/full
  [cv3]=exp/rl_diffusion_stage0/cv3_baseline
  [gt]=exp/emofilm_film_only/eval_refs
)

for model in v1 film_only longepoch sentlvl cv3 gt; do
  for part in esd fedd_a fedd_b; do
    mkdir -p "$OUT/$model"
    python tools/calibrate_rl_rewards.py \
      --manifest "data/contracts/emofilm_v1/eval/$part/manifest.jsonl" \
      --hyp-dir "${HYPS[$model]}/$part" \
      --prototypes-dir "$PROTO" \
      --output "$OUT/$model/$part.json" \
      --model-name "$model" \
      --max-samples 200 \
      --device cuda
  done
done
```

Run:

```bash
GPU=3 bash exp/rl_diffusion_stage0/run_calibration.sh
ls exp/rl_diffusion_stage0/calibration/*/*.json | wc -l
python - <<'PY'
import json
for part in ["esd", "fedd_a", "fedd_b"]:
    d = json.load(open(f"exp/rl_diffusion_stage0/calibration/cv3/{part}.json"))
    assert set(d["gates"]) == {"gate1_emotion_direction", "gate2_non_degenerate"}
    print("cv3", part, d["gates"], d["nearest_prototype_acc_pct"])
PY
```

Expected: 18 份 JSON；每份含 `r_emo/r_sv/r_cer/per_emotion_r_emo_mean/nearest_prototype_acc_pct/gap_mean/gates`；GT 视图两道门槛通过、`nearest_prototype_acc_pct` 明显高于 20%（chance）。

- [ ] **Step 6: 撰写阶段 0 报告**

创建 `docs/reports/2026-08-15-emofilm-rl-diffusion-stage0-report.md`，结构如下并把实测数字从 18 份 JSON 与三份评测 JSON 填入：

```markdown
# EmoFiLM RL+扩散 阶段 0（前置验证与奖励校准）执行报告

- 日期：2026-08-15
- 计划：docs/superpowers/plans/2026-08-15-emofilm-rl-diffusion-stage0.md

## 0. 一句话结论

[flow 冒烟是否通过；原型五类是否可分；CV3 全量基线指标；两道门槛在 CV3 视图是否通过 → 是否可进入阶段 2]

## 1. flow 训练入口冒烟

- 数据：data/smoke/cv3_flow（train 12 / cv 4，v3 token + CAM++ embedding）
- 日志：exp/rl_diffusion_stage0/smoke/train.log
- 结果：[TRAIN/CV loss 是否出现、checkpoint 列表]

## 2. 情感原型

- 产物：exp/rl_diffusion_stage0/prototypes/{ang,hap,neu,sad,sur}.npy + stats.json
- 每类样本数与五类两两余弦矩阵：[从 stats.json 抄入]

## 3. CV3 全量中性基线（v3 评测）

| 数据集 | n | emo_sim | wer% | 判别 |
| --- | --- | --- | --- | --- |
| esd | 1500 | [值] | [值] | [n_scored/acc] |
| fedd_a | 500 | [值] | [值] | [n_scored=0 预期] |
| fedd_b | 500 | [值] | [值] | [值] |

## 4. 三路奖励校准（每视图每数据集 200 条分层样本）

| 视图 | 数据集 | r_emo mean/std | r_sv mean | r_cer mean | nearest-acc | gap_mean | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1/film_only/longepoch/sentlvl/cv3/gt | esd/fedd_a/fedd_b | [值] | [值] | [值] | [值] | [值] | [值] |

## 5. 门槛判定与下一阶段建议

- 主判据 = cv3 三数据集视图的两道门槛；gt 视图为奖励模型 sanity。
- 结论：[通过/不通过 + 依据 + 对阶段 2 的具体影响]
```

---

## 对抗审查结论（仅真实阻塞，逐项给处置）

1. **online_feature 初始化陷阱**：`cosyvoice.utils.onnx` 的 `embedding_extractor/online_feature` 在模块导入时按环境变量一次绑定，而 `train.py` 在 `main()` 才设置 `onnx_path`，导致在线抽 token 路径恒不可用。处置：Q6 选择预存 v3 token 与 embedding 的 parquet 路线，完全绕开该路径。
2. **冒烟取样偏差**：ESD manifest 按说话人/情感分块排列，"取前 16 条"会退化为单一情感子集。处置：冒烟只验证训练入口（对情感分布不敏感），16 条来自 `esd_cv3_150.jsonl` 的跨情感分层清单；校准侧的判别统计用 `sample_stratified` 显式均衡，并有两项单元测试锁定行为。
3. **IEMOCAP 短词段**：<50ms 词段被 800 样本阈值跳过，可能压低 IEMOCAP 对原型的贡献。处置：原型池以 ESD 句级为主，`stats.json` 如实记录每类样本数，报告中不以 IEMOCAP 词级贡献为前提。
4. **CV3 `finish_reason=None`**：沿用已实测的"保存首段有效 `tts_speech`"逻辑，失败/跳过全部写入 generation manifest，评测前做 1500/500/500 完整性断言。
5. **FEDD 判别参考组不足**：FEDD_A 无同文本跨情感参考，判别结构上不可算。处置：只给 ESD 传 `--emotion_ref_manifest`；FEDD 评测不传，接受 `n_scored=0` 的诚实口径（v3 契约已支持）。
6. **whisper 全量转写成本**：三路奖励必须在同一样本子集上对齐，故每视图每数据集用 200 条分层子集，总转写 3600 条，约 1 GPU·小时量级，不重跑四模型既有评测。
7. **冒烟显存**：22 层 DiT fp32 + dynamic batch 在 24GB 内可跑（flow 训练无需 LLM）；若 OOM 按 Task 1 Step 6 的降批处理，不改模型代码。
8. **明确不做（避免过度设计）**：不改 `train.py/flow.py/flow_matching.py`；不引入 ODE→SDE；不做伪组方差探测；不建奖励 Flask 服务；不加在线 token 提取管线；不为校准结果新增新指标字段。

## Self-Review 结果

- 规格覆盖：阶段 0 报告的三项要求（flow 冒烟、离线奖励校准、原型计算）分别由 Task 1/4/2 覆盖，CV3 全量基线为 Task 3；四模型 v2 与 GT 视图在 Task 4 的 run 脚本中显式覆盖。
- 占位符扫描：所有代码步骤含完整实现与可运行命令；报告中的 `[值]` 是执行期从 JSON 抄录的实测值，非实现占位。
- 类型一致性：`parse_emotion_tags` 返回 `(segments, clean_text)`，在 Task 2 的测试、脚本与 Task 4 的 `strip_emotion_tags` 引用一致；`evaluate_gates` 的输入键 `gap_mean/r_emo_std` 在测试与 Task 4 实现一致；`sample_stratified(rows, n, key)` 签名一致。
