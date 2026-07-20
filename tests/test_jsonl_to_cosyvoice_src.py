"""tagged jsonl → CosyVoice src_dir 转换器测试。

entries schema = generate_tagged_jsonl.py 实际输出：
  {utt_id, audio_filepath(=wav_path), text(tagged, <emotion>标签), plain_text(原), speaker_id, 可选 instruct}
标签格式 <emotion type='ang' intensity='high'>（下游 emo_tokenizer.py:16 正则证明）。
"""
import os
import sys

ROOT = str(__import__("pathlib").Path(__file__).parents[1])
sys.path.insert(0, ROOT)


def test_import():
    from tools.jsonl_to_cosyvoice_src import write_src_dir
    assert callable(write_src_dir)


def test_write_minimal_src_dir(tmp_path):
    """最小输入：写 wav.scp/text/utt2spk 三件套，无 instruct；text 用 tagged(<emotion>)。"""
    from tools.jsonl_to_cosyvoice_src import write_src_dir

    entries = [
        {"utt_id": "0011_000001", "audio_filepath": "/data/esd/0011_000001.wav",
         "text": "<emotion type='ang' intensity='high'>rare rabbit</emotion>",
         "plain_text": "Rare rabbit had a little apron.",
         "speaker_id": "0011"},
        {"utt_id": "0011_000002", "audio_filepath": "/data/esd/0011_000002.wav",
         "text": "<emotion type='neu' intensity='low'>hello world</emotion>",
         "plain_text": "Hello world.",
         "speaker_id": "0011"},
    ]
    src_dir = str(tmp_path / "src")
    write_src_dir(src_dir, entries, use_tagged_text=True, write_instruct=False)

    for fname in ["wav.scp", "text", "utt2spk"]:
        assert os.path.isfile(os.path.join(src_dir, fname)), f"missing {fname}"
    assert not os.path.isfile(os.path.join(src_dir, "instruct"))

    # wav.scp: "utt_id wav_path"
    with open(os.path.join(src_dir, "wav.scp")) as f:
        wlines = f.read().strip().split("\n")
    assert wlines[0].split()[0] == "0011_000001"
    assert wlines[0].split()[1] == "/data/esd/0011_000001.wav"

    # text 用 tagged_text（含 <emotion> 词级标签，下游 emo_tokenizer 解析）
    with open(os.path.join(src_dir, "text")) as f:
        text_content = f.read()
    assert "<emotion type='ang' intensity='high'>" in text_content
    assert "</emotion>" in text_content

    # utt2spk: "utt_id speaker_id"
    with open(os.path.join(src_dir, "utt2spk")) as f:
        ulines = f.read().strip().split("\n")
    assert ulines[0].split()[1] == "0011"


def test_use_tagged_text_false_uses_plain_text(tmp_path):
    """use_tagged_text=False 时，text 文件用 plain_text（消融 no_word：无词级标签）。"""
    from tools.jsonl_to_cosyvoice_src import write_src_dir

    entries = [{
        "utt_id": "u1", "audio_filepath": "/w.wav",
        "text": "<emotion type='neu' intensity='low'>hello</emotion>",
        "plain_text": "plain text here", "speaker_id": "s1",
    }]
    src_dir = str(tmp_path / "src")
    write_src_dir(src_dir, entries, use_tagged_text=False, write_instruct=False)

    with open(os.path.join(src_dir, "text")) as f:
        content = f.read()
    assert "plain text here" in content
    assert "<emotion" not in content


def test_write_instruct_creates_instruct_file(tmp_path):
    """write_instruct=True 且 entry 含 instruct 字段时，写 instruct 文件（zero-shot prompt 段）。"""
    from tools.jsonl_to_cosyvoice_src import write_src_dir

    entries = [{
        "utt_id": "u1", "audio_filepath": "/w.wav",
        "text": "<emotion type='neu' intensity='low'>hello</emotion>",
        "plain_text": "hello", "speaker_id": "s1",
        "instruct": "Please speak in a neutral tone.",
    }]
    src_dir = str(tmp_path / "src")
    write_src_dir(src_dir, entries, use_tagged_text=True, write_instruct=True)

    assert os.path.isfile(os.path.join(src_dir, "instruct"))
    with open(os.path.join(src_dir, "instruct")) as f:
        content = f.read()
    assert "Please speak in a neutral tone." in content
