"""faster-whisper loader shared by eval and the clean-reference script, and the G4 Part 2 word and VAD hooks."""
import os
import re
from pathlib import Path


def load_whisper(device="cpu", size="small", threads=1):
    if device == "cuda":
        # ctranslate2 links cuBLAS/cuDNN 12 dynamically; torch's wheel already ships them
        import torch
        # Linux loads the wheel's shared libraries through torch; DLL search registration is Windows-only.
        if hasattr(os, "add_dll_directory"):
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


# ---------- G4 Part 2 hooks (plan 11.2): confident-word survival and VAD speech seconds ----------
# Both load lazily and take injectable callables, so tests (and a laptop without weights) never touch a model.

WORD_MODEL = "small"          # multilingual: the size eval and field_accept already use (en + hi speech)
VAD_THRESHOLD = 0.3


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def resolve_device(device: str = "auto") -> str:
    """'auto' -> cuda when ctranslate2 sees a CUDA device, else cpu; anything else is returned as given."""
    if device != "auto":
        return device
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def norm_word(w: str) -> str:
    return re.sub(r"[^\w']+", "", w.lower())


class WordTranscriber:
    """audio (16 kHz float32) -> [(word, start_s, end_s, p)] from faster-whisper word timestamps.

    Greedy, no temperature fallback, no VAD filter, language auto (as `transcribe`). The model loads on the first call;
    `loader(device, size, threads)` replaces `load_whisper` (tests pass a fake)."""

    def __init__(self, model: str = WORD_MODEL, device: str = "auto", threads: int = 1, loader=None):
        self.model_name, self.device, self.threads = model, device, threads
        self.loader, self._m = loader or load_whisper, None

    def __call__(self, audio):
        import numpy as np
        if self._m is None:
            self.device = resolve_device(self.device)
            self._m = self.loader(self.device, self.model_name, self.threads)
        segs, _ = self._m.transcribe(np.asarray(audio, np.float32), language=None, beam_size=1, temperature=0.0,
                                     vad_filter=False, condition_on_previous_text=False, word_timestamps=True)
        return [(norm_word(w.word), float(w.start), float(w.end), float(w.probability))
                for s in segs for w in (s.words or [])]


def vad_speech_seconds(y, sr: int = 16000, threshold: float = VAD_THRESHOLD, timestamps=None) -> float:
    """Speech seconds by the Silero VAD bundled with faster-whisper (faster_whisper/assets, run on onnxruntime; no
    download). Peak-normalised to 0.5 first so the level does not decide; no edge padding (speech_pad_ms=0) so the
    seconds are the detected speech itself. `timestamps(y, opts_dict)` replaces get_speech_timestamps (tests)."""
    import numpy as np
    y = np.asarray(y, np.float32)
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    opts = {"threshold": threshold, "min_silence_duration_ms": 200, "min_speech_duration_ms": 150, "speech_pad_ms": 0}
    if timestamps is None:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        ts = get_speech_timestamps(y, VadOptions(**opts), sampling_rate=sr)
    else:
        ts = timestamps(y, opts)
    return float(sum(t["end"] - t["start"] for t in ts) / sr)
