"""One STFT definition for the whole project.

Upstream GTCRN uses sqrt-Hann window twice (analysis+synthesis == COLA).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:   # the board timing script runs the numpy DSP with no torch installed (Zero 2 W, 512 MB)
    torch = None

N_FFT = 512
HOP = 256
WIN = 512


_WINDOWS: dict = {}


def window(device=None) -> "torch.Tensor":
    """The periodic sqrt-Hann window, cached per device (it used to be rebuilt on every call). The cached tensor is
    never modified in place; callers that need to must clone it."""
    key = str(torch.device(device) if device is not None else torch.device("cpu"))
    w = _WINDOWS.get(key)
    if w is None:
        w = _WINDOWS[key] = torch.hann_window(WIN, device=device).pow(0.5)
    return w


def stft(x: "torch.Tensor") -> "torch.Tensor":
    """x: (..., T) -> (..., F, T', 2) real/imag, matching upstream GTCRN input."""
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    s = torch.stft(x, N_FFT, HOP, WIN, window(x.device), center=True, return_complex=True)
    s = torch.view_as_real(s)
    return s.reshape(*shape[:-1], *s.shape[1:])


def istft(spec: "torch.Tensor", length: int | None = None) -> "torch.Tensor":
    shape = spec.shape
    spec = spec.reshape(-1, *shape[-3:])
    c = torch.view_as_complex(spec.contiguous())
    y = torch.istft(c, N_FFT, HOP, WIN, window(spec.device), center=True, length=length)
    return y.reshape(*shape[:-3], y.shape[-1])


_ENVELOPES: dict = {}


def _envelope(n_frames: int, dtype, device):
    """Overlap-added squared window of an n_frames centered transform, as torch.istft builds it."""
    key = (n_frames, dtype, str(device))
    e = _ENVELOPES.get(key)
    if e is None:
        w2 = window(device).pow(2)          # torch.istft folds the squared window in the window's own dtype
        total = N_FFT + HOP * (n_frames - 1)
        e = torch.nn.functional.fold(w2[None, :, None].expand(1, N_FFT, n_frames), (1, total), (1, N_FFT),
                                     stride=(1, HOP)).reshape(total).to(dtype)
        if len(_ENVELOPES) > 64:
            _ENVELOPES.clear()
        _ENVELOPES[key] = e
    return e


def istft_explicit(spec: "torch.Tensor", length: int | None = None) -> "torch.Tensor":
    """istft() as an explicit overlap-add with the same maths as torch.istft (center=True) and no host-side
    window-envelope check, so a training step that calls it stays sync-free and graph-capturable. The envelope is a
    constant of the frame count; bit-identical to istft() on CPU (tests/test_train_throughput.py)."""
    shape = spec.shape
    spec = spec.reshape(-1, *shape[-3:])
    b, _, t, _ = spec.shape
    c = torch.view_as_complex(spec.contiguous())
    fr = torch.fft.irfft(c, n=N_FFT, dim=1) * window(spec.device).to(spec.dtype)[None, :, None]   # (B, N_FFT, T)
    total = N_FFT + HOP * (t - 1)
    y = torch.nn.functional.fold(fr, (1, total), (1, N_FFT), stride=(1, HOP)).reshape(b, total)
    env = _envelope(t, spec.dtype, spec.device)
    start = N_FFT // 2
    end = total - N_FFT // 2 if length is None else start + length
    y = y[:, start:min(end, total)] / env[start:min(end, total)]
    if length is not None and y.shape[1] < length:
        y = torch.nn.functional.pad(y, (0, length - y.shape[1]))
    return y.reshape(*shape[:-3], y.shape[-1])


def np_stft(x: np.ndarray) -> np.ndarray:
    """NumPy twin of stft() for the DSP reference (portable to C).
    Reproduces torch's center=True reflect padding and framing exactly."""
    w = np.hanning(WIN + 1)[:-1] ** 0.5  # periodic sqrt-Hann == torch.hann_window
    xp = np.pad(x, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        xp, shape=(n_frames, N_FFT), strides=(xp.strides[0] * HOP, xp.strides[0]))
    return np.fft.rfft(frames * w, axis=1).T.astype(np.complex64)  # (F, T')
