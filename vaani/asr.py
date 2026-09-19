"""faster-whisper loader shared by eval and the clean-reference script."""
import os
from pathlib import Path


def load_whisper(device="cpu", size="small"):
    if device == "cuda":
        # ctranslate2 links cuBLAS/cuDNN 12 dynamically; torch's wheel already ships them
        import torch
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    from faster_whisper import WhisperModel
    return WhisperModel(size, device=device, compute_type="int8" if device == "cpu" else "float16")
