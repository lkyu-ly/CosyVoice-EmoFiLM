"""speech token 提取的长度边界合同测试。"""

import numpy as np


class _Input:
    def __init__(self, name):
        self.name = name


class _Session:
    def __init__(self):
        self.inputs = [_Input("feats"), _Input("feats_length")]
        self.seen_length = None

    def get_inputs(self):
        return self.inputs

    def run(self, _outputs, feed):
        self.seen_length = int(feed["feats_length"][0])
        return [np.asarray([[1, 2, 3]], dtype=np.int32)]


def test_extract_tokens_does_not_drop_audio_longer_than_30_seconds(tmp_path):
    from tools.extract_speech_token import extract_speech_tokens

    wav_path = tmp_path / "long.wav"
    import soundfile as sf

    sf.write(wav_path, np.zeros(31 * 16000, dtype=np.float32), 16000)
    session = _Session()

    tokens = extract_speech_tokens(wav_path, session)

    assert tokens == [1, 2, 3]
    assert session.seen_length > 3000
