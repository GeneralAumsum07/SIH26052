"""faster-whisper loader shared by eval and the clean-reference script."""
import os
from pathlib import Path


def load_whisper(device="cpu", size="small", threads=1):
    if device == "cuda":
        # ctranslate2 links cuBLAS/cuDNN 12 dynamically; torch's wheel already ships them
        import torch
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    from faster_whisper import WhisperModel
    # num_workers lets several transcribe() calls run concurrently; ctranslate2 releases the GIL so a thread pool suffices
    return WhisperModel(size, device=device, compute_type="int8" if device == "cpu" else "float16", num_workers=threads)


def transcribe(model, audio):
    """One greedy pass, no temperature fallback: on -10 dB mixtures the fallback retries up to 5x per clip
    (~6 s instead of 0.3 s) and makes the hypothesis nondeterministic. Shared so eval and the clean reference match."""
    segs, _ = model.transcribe(audio, language=None, beam_size=1, temperature=0.0,
                               condition_on_previous_text=False, without_timestamps=True)
    return " ".join(s.text for s in segs).strip()  # generator: decoding happens here
