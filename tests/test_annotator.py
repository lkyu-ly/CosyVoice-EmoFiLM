"""EmoFiLM WordSequenceModel 的 768d/5 类/3D VAD 合同测试。"""
import os
import tempfile

import pytest
import torch

from cosyvoice_emo.emo_annotator import WordSequenceModel


def test_model_forward_shape():
    model = WordSequenceModel()
    model.eval()
    x = torch.randn(2, 20, 768)
    mask = torch.zeros(2, 20, dtype=torch.bool)

    with torch.no_grad():
        class_logits, vad_pred = model(x, padding_mask=mask)
    assert class_logits.shape == (2, 5)
    assert vad_pred.shape == (2, 3)
    # VAD 输出在 [0,1]
    assert 0 <= vad_pred.min() <= 1
    assert 0 <= vad_pred.max() <= 1


def test_model_with_padding():
    model = WordSequenceModel()
    model.eval()
    x = torch.randn(2, 20, 768)
    # last 10 frames of first sample are padding
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[0, 10:] = True

    with torch.no_grad():
        class_logits, vad_pred = model(x, padding_mask=mask)
    assert class_logits.shape == (2, 5)
    assert vad_pred.shape == (2, 3)


def test_model_forward_no_mask():
    """None mask 等同于全 False。"""
    model = WordSequenceModel()
    model.eval()
    x = torch.randn(4, 15, 768)
    with torch.no_grad():
        class_logits, vad_pred = model(x, padding_mask=None)
    assert class_logits.shape == (4, 5)
    assert vad_pred.shape == (4, 3)


def test_emofilm_shape_is_fixed():
    model = WordSequenceModel()
    assert model.input_dim == 768
    assert model.num_classes == 5
    assert model.reg_dim == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 1024},
        {"num_classes": 4},
        {"reg_dim": 1},
    ],
)
def test_non_emofilm_shapes_are_rejected(kwargs):
    with pytest.raises(ValueError, match="EmoFiLM WordSequenceModel contract"):
        WordSequenceModel(**kwargs)


def test_vad_scale_range():
    """VAD *4+1 后应在 [1,5] 范围。"""
    model = WordSequenceModel()
    model.eval()
    x = torch.randn(3, 10, 768)
    with torch.no_grad():
        _, vad_pred = model(x, None)
    scaled = vad_pred * 4.0 + 1.0
    assert 1 <= scaled.min() <= 5
    assert 1 <= scaled.max() <= 5


def test_save_load():
    """WordSequenceModel 可序列化/反序列化。"""
    model = WordSequenceModel()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        p = f.name
    loaded = WordSequenceModel()
    loaded.load_state_dict(torch.load(p, map_location="cpu"))
    os.unlink(p)
    # forward 无报错即通过
    x = torch.randn(2, 10, 768)
    with torch.no_grad():
        loaded(x, None)
