"""tagged JSONL 生成单测。"""
import json
import os
import subprocess
import tempfile

import torch

EMOFILM_PY = "/home/hanlvyuan/miniconda3/envs/emofilm/bin/python"
ROOT = "/home/hanlvyuan/LLM-Audio/CosyVoice-EmoFiLM"
SCRIPT = f"{ROOT}/tools/generate_tagged_jsonl.py"


def test_generate_jsonl_format():
    """最小 fake 数据生成 tagged jsonl。"""
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)

    tmp = tempfile.mkdtemp(prefix="gen_jsonl_")

    # 生成 fake WordSequenceModel checkpoint
    from cosyvoice_emo.emo_annotator import WordSequenceModel
    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    ckpt_path = os.path.join(tmp, "model.pt")
    torch.save(model.state_dict(), ckpt_path)

    # 生成 fake word blocks（随机帧，使 predictions 非固定）
    for utt_id in ["utt_1", "utt_2"]:
        utt_dir = os.path.join(tmp, "word_blocks", utt_id)
        os.makedirs(utt_dir)
        for wi in range(4):
            n_frames = np.random.randint(5, 20)
            torch.save({
                "frames": torch.randn(n_frames, 1024),
                "word": f"word{wi}",
                "padding_mask": torch.zeros(n_frames, dtype=torch.bool),
            }, os.path.join(utt_dir, f"{wi:04d}_0_{n_frames}.pt"))

    manifest = os.path.join(tmp, "manifest.jsonl")
    with open(manifest, "w") as f:
        for utt_id in ["utt_1", "utt_2"]:
            f.write(json.dumps({
                "utt_id": utt_id,
                "wav_path": f"/tmp/{utt_id}.wav",
                "speaker_id": "0011",
            }) + "\n")

    output = os.path.join(tmp, "tagged.jsonl")
    cmd = [
        EMOFILM_PY, SCRIPT,
        f"--data_dir={os.path.join(tmp, 'word_blocks')}",
        f"--manifest={manifest}",
        f"--checkpoint={ckpt_path}",
        f"--output_jsonl={output}",
        "--device=cpu",
    ]
    subprocess.run(cmd, check=True)

    assert os.path.isfile(output)
    with open(output) as f:
        lines = f.readlines()
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert "audio_filepath" in row
        assert "text" in row
        assert "speaker_id" in row
        # text 包含 emotion 标签或裸文本
        text = row["text"]
        if "<emotion" in text:
            assert "</emotion>" in text


def test_merge_continuous_same_labels():
    """连续同标签合并后不应有相邻相同 tag。"""
    # 直接测试合并函数（不做子进程全 pipeline）
    from tools.generate_tagged_jsonl import merge_continuous_tags
    words = [
        {"word": "I", "predicted_emotion": "ang", "predicted_arousal": 4.0},
        {"word": "am", "predicted_emotion": "ang", "predicted_arousal": 4.0},
        {"word": "happy", "predicted_emotion": "hap", "predicted_arousal": 3.0},
        {"word": "today", "predicted_emotion": "hap", "predicted_arousal": 2.5},
    ]
    tagged = merge_continuous_tags(words)
    assert tagged.count("<emotion") == 2, f"expected 2 tags: {tagged}"
    assert "I am" in tagged
    assert "happy today" in tagged


def test_arousal_bucketing():
    from tools.generate_tagged_jsonl import arousal_to_intensity
    assert arousal_to_intensity(4.0) == "high"
    assert arousal_to_intensity(3.0) == "medium"
    assert arousal_to_intensity(2.0) == "low"
    assert arousal_to_intensity(1.5) == "low"
