from setuptools import setup, find_packages

setup(
    name="cosyvoice-emofilm",
    version="0.1.0",
    description="Emo-FiLM reproduction fork based on CosyVoice2",
    packages=find_packages(include=["cosyvoice", "cosyvoice.*"]),
    python_requires=">=3.10",
    install_requires=[
        "conformer==0.3.2",
        "HyperPyYAML==1.2.3",
        "librosa==0.10.2",
        "modelscope==1.20.0",
        "onnxruntime-gpu==1.18.0",
        "openai-whisper==20231117",
        "soundfile==0.12.1",
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "transformers==4.51.3",
        "wetext==0.0.4",
    ],
)
