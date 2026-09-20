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


def window(device=None) -> "torch.Tensor":
    return torch.hann_window(WIN, device=device).pow(0.5)


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


def np_stft(x: np.ndarray) -> np.ndarray:
    """NumPy twin of stft() for the DSP reference (portable to C).
    Reproduces torch's center=True reflect padding and framing exactly."""
    w = np.hanning(WIN + 1)[:-1] ** 0.5  # periodic sqrt-Hann == torch.hann_window
    xp = np.pad(x, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        xp, shape=(n_frames, N_FFT), strides=(xp.strides[0] * HOP, xp.strides[0]))
    return np.fft.rfft(frames * w, axis=1).T.astype(np.complex64)  # (F, T')
