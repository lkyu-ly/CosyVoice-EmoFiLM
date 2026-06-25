"""WordSequenceModel 单测。"""
import os
import tempfile
import torch
from cosyvoice_emo.emo_annotator import WordSequenceModel


def test_model_forward_shape():
    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    model.eval()
    x = torch.randn(2, 20, 1024)  # batch=2, T=20, D=1024
    mask = torch.zeros(2, 20, dtype=torch.bool)

    with torch.no_grad():
        class_logits, vad_pred = model(x, padding_mask=mask)
    assert class_logits.shape == (2, 5)
    assert vad_pred.shape == (2, 3)
    # VAD 输出在 [0,1]
    assert 0 <= vad_pred.min() <= 1
    assert 0 <= vad_pred.max() <= 1


def test_model_with_padding():
    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    model.eval()
    x = torch.randn(2, 20, 1024)
    # last 10 frames of first sample are padding
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[0, 10:] = True

    with torch.no_grad():
        class_logits, vad_pred = model(x, padding_mask=mask)
    assert class_logits.shape == (2, 5)
    assert vad_pred.shape == (2, 3)


def test_model_forward_no_mask():
    """None mask 等同于全 False。"""
    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    model.eval()
    x = torch.randn(4, 15, 1024)
    with torch.no_grad():
        class_logits, vad_pred = model(x, padding_mask=None)
    assert class_logits.shape == (4, 5)
    assert vad_pred.shape == (4, 3)


def test_vad_scale_range():
    """VAD *4+1 后应在 [1,5] 范围。"""
    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    model.eval()
    x = torch.randn(3, 10, 1024)
    with torch.no_grad():
        _, vad_pred = model(x, None)
    scaled = vad_pred * 4.0 + 1.0
    assert 1 <= scaled.min() <= 5
    assert 1 <= scaled.max() <= 5


def test_save_load():
    """WordSequenceModel 可序列化/反序列化。"""
    model = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        p = f.name
    loaded = WordSequenceModel(input_dim=1024, num_classes=5, num_heads=8, dropout_rate=0.3)
    loaded.load_state_dict(torch.load(p, map_location="cpu"))
    os.unlink(p)
    # forward 无报错即通过
    x = torch.randn(2, 10, 1024)
    with torch.no_grad():
        loaded(x, None)
