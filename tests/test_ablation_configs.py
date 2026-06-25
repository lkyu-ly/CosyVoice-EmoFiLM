"""消融 YAML 可加载性测试 + 差异断言。"""
import os
import re
import sys

ROOT = "/home/lkyu/LLM-Audio/CosyVoice-EmoFiLM"
sys.path.insert(0, os.path.join(ROOT, "third_party", "Matcha-TTS"))
from hyperpyyaml import load_hyperpyyaml


def _load_yaml(path):
    """加载 yaml 并做 CosyVoice-BlankEN 路径 override。"""
    qwen_path = os.path.join(ROOT, "pretrained_models", "CosyVoice2-0.5B", "CosyVoice-BlankEN")
    if not os.path.exists(qwen_path):  # CosyVoice-BlankEN 是目录，用 exists 不是 isfile
        qwen_path = os.path.join(ROOT, "pretrained_models", "CosyVoice2-0.5B")  # 目录式 fallback
    with open(path) as f:
        return load_hyperpyyaml(f, overrides={"qwen_pretrain_path": qwen_path})


def test_all_ablations_loadable():
    """4 个消融 YAML 均可加载，且 configs['llm'] 是 Qwen2LM_Emotion（no_film 例外，是 Qwen2LM_Emotion_NoFilm）。"""
    expected_llm_classes = {
        "no_global": "Qwen2LM_Emotion",
        "no_word": "Qwen2LM_Emotion",
        "no_emo_loss": "Qwen2LM_Emotion",
        "no_film": "Qwen2LM_Emotion",  # AddFusionEmotionAdapter 是 emotion_adapter 类型差异，llm 仍是子类
    }
    for name, expected_cls in expected_llm_classes.items():
        path = os.path.join(ROOT, "conf", f"ablation_{name}.yaml")
        assert os.path.isfile(path), f"missing {path}"
        configs = _load_yaml(path)
        assert configs["llm"] is not None, f"{name}: configs['llm'] is None"
        actual_cls = configs["llm"].__class__.__name__
        assert actual_cls == expected_cls, f"{name}: expected llm class {expected_cls}, got {actual_cls}"


def test_no_emo_loss_has_weight_zero():
    """ablation_no_emo_loss.yaml 必须显式 emo_loss_weight: 0。"""
    path = os.path.join(ROOT, "conf", "ablation_no_emo_loss.yaml")
    with open(path) as f:
        content = f.read()
    # 兼容多种格式：emo_loss_weight: 0 / emo_loss_weight: 0.0 / emo_loss_weight: 0.2e-10
    m = re.search(r"^emo_loss_weight:\s*([0-9.eE\-]+)\s*$", content, re.MULTILINE)
    assert m, f"emo_loss_weight not found in {path}"
    val = float(m.group(1))
    assert val == 0.0, f"emo_loss_weight should be 0, got {val}"


def test_no_global_has_only_iemocap_train_data():
    """ablation_no_global.yaml train_data 只含 annotated_IEMOCAP（剔除 ESD）。"""
    path = os.path.join(ROOT, "conf", "ablation_no_global.yaml")
    with open(path) as f:
        content = f.read()
    # train_data 必须含 'iemocap' 关键字，不应含 'esd'
    assert "iemocap" in content.lower(), f"ablation_no_global.yaml should contain IEMOCAP train_data"
    # 提取 train_data: 行的值
    m = re.search(r"^train_data:\s*(.+)$", content, re.MULTILINE)
    if m:
        train_val = m.group(1).lower()
        assert "esd" not in train_val, f"ablation_no_global.yaml train_data should NOT contain ESD: {train_val}"


def test_no_word_has_only_esd_train_data():
    """ablation_no_word.yaml train_data 只含 ESD 全局数据（剔除 annotated_IEMOCAP）。"""
    path = os.path.join(ROOT, "conf", "ablation_no_word.yaml")
    with open(path) as f:
        content = f.read()
    assert "esd" in content.lower(), f"ablation_no_word.yaml should contain ESD train_data"
    m = re.search(r"^train_data:\s*(.+)$", content, re.MULTILINE)
    if m:
        train_val = m.group(1).lower()
        assert "iemocap" not in train_val, f"ablation_no_word.yaml train_data should NOT contain IEMOCAP: {train_val}"


def test_no_film_uses_add_fusion_adapter():
    """ablation_no_film.yaml emotion_adapter 必须是 AddFusionEmotionAdapter（简单加法，非 FiLMLayer）。"""
    path = os.path.join(ROOT, "conf", "ablation_no_film.yaml")
    with open(path) as f:
        content = f.read()
    # 必须出现 AddFusionEmotionAdapter 关键字（新增类名）
    assert "AddFusionEmotionAdapter" in content, (
        f"ablation_no_film.yaml 必须用 AddFusionEmotionAdapter 替代 FiLMLayer。"
        f"FiLM 是 gamma*x+beta，no_film 消融应改为 x+emotion_features 简单加法。"
    )


def test_no_film_runtime_uses_add_fusion():
    """运行时验证：ablation_no_film.yaml 加载后 llm.emotion_adapter 是 AddFusionEmotionAdapter 实例。

    防止 yaml 写了但 Qwen2LM_Emotion 忽略参数的接口漂移（spec 12.2 消融有效性）。
    """
    from cosyvoice.llm.emo_film import AddFusionEmotionAdapter
    path = os.path.join(ROOT, "conf", "ablation_no_film.yaml")
    configs = _load_yaml(path)
    llm = configs["llm"]
    assert isinstance(llm.emotion_adapter, AddFusionEmotionAdapter), (
        f"ablation_no_film.yaml 加载后 llm.emotion_adapter 应为 AddFusionEmotionAdapter 实例，"
        f"实际为 {type(llm.emotion_adapter).__name__}。"
        f"请检查 Qwen2LM_Emotion.__init__ 是否接受 emotion_adapter 参数。"
    )
