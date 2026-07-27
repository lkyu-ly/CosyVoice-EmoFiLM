"""Config 可加载性冒烟测试。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
ASSET_ROOT = Path(os.environ.get("EMOFILM_PROJECT_ROOT", ROOT))
sys.path.insert(0, os.path.join(str(ROOT), "third_party", "Matcha-TTS"))
from hyperpyyaml import load_hyperpyyaml


def test_emo_film_yaml_loads():
    yaml_path = ROOT / "conf" / "emo_film.yaml"
    assert os.path.isfile(yaml_path), f"missing {yaml_path}"
    qwen_path = str(ASSET_ROOT / "pretrained_models" / "CosyVoice2-0.5B" / "CosyVoice-BlankEN")
    with yaml_path.open(encoding="utf-8") as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": qwen_path})
    # 验证关键值（v2 单流协议：无 emo_loss_weight；有下游任务头权重）
    assert configs["llm"].__class__.__name__ == "Qwen2LM_Emotion"
    assert not hasattr(configs["llm"], "emo_loss_weight"), (
        "活跃配置不得再携带 v1 输入端 emo_loss_weight 死字段"
    )
    assert configs["llm"].emotion_head_weight == 1.0
    assert configs["llm"].intensity_head_weight == 1.0
    # decode_config 长度合同存在
    assert "decode_config" in configs
    assert configs["decode_config"]["max_len_hard_cap"] == 2000
    # tokenizer factory 可调用
    tokenizer = configs["get_tokenizer"]()
    assert tokenizer is not None
    print("OK")


if __name__ == "__main__":
    test_emo_film_yaml_loads()
