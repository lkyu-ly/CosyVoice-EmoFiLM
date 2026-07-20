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
    # 验证关键值
    assert configs["llm"].__class__.__name__ == "Qwen2LM_Emotion"
    assert configs["llm"].emo_loss_weight == 0.2
    # tokenizer factory 可调用
    tokenizer = configs["get_tokenizer"]()
    assert tokenizer is not None
    print("OK")


if __name__ == "__main__":
    test_emo_film_yaml_loads()
