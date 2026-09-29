"""Low-delay asymmetric STFT (plan Section 3.1): analysis K = 512, hop H, synthesis support L.

Offline, for N > 0 samples:
    data_hops = ceil(N / H); frames = data_hops + 1
    left_pad = K - H; right_pad = data_hops * H - N + H
Frame j covers padded [jH, jH + K); its windowed inverse FFT is non-zero only on its last L samples, which
overlap-add at padded [jH + K - L, jH + K). The output is padded positions [K - H, K - H + N).

Streaming (numpy, per stream, no shared mutable state): each step appends H samples to a (K - H)-sample history,
transforms the last K, and after the neural step takes the final L samples of the windowed inverse FFT, overlap-adds
them and releases H samples. Frame j releases stream samples [(j+1)H - L, (j+2)H - L): the earliest sample of a hop
waits L samples, the latest L - H + 1. The first L - H released samples precede stream sample 0 (pre-start positions,
discarded for aligned offline scoring).

C0 (the legacy contract) dispatches to vaani.dsp.stft's centered transform: the low-delay formulas never apply to it.
The synthesis is explicit; the legacy centered torch.istft is never used to interpret these windows.
"""
from __future__ import annotations

import numpy as np

from vaani.audio_contract import AudioContract, get_audio_contract

try:
    import torch
    import torch.nn.functional as F
except ImportError:   # board-side numpy streaming runs without torch
    torch = None

_WIN_CACHE: dict = {}


def _windows_t(c: AudioContract, dtype, device):
    key = (c.audio_contract_id, dtype, str(device))
    w = _WIN_CACHE.get(key)
    if w is None:
        a, _, s = c.windows()
        w = (torch.as_tensor(a, dtype=dtype, device=device), torch.as_tensor(s, dtype=dtype, device=device))
        _WIN_CACHE[key] = w
    return w


def pads(n: int, c: AudioContract):
    """(data_hops, frames, left_pad, right_pad) for an n-sample low-delay waveform."""
    if n <= 0:
        raise ValueError("empty input: low-delay framing needs N > 0 samples")
    data_hops = (n + c.hop - 1) // c.hop
    return data_hops, data_hops + 1, c.k - c.hop, data_hops * c.hop - n + c.hop


# ---- torch, differentiable -------------------------------------------------------------------
def analyze(x, contract):
    """(B, N) real waveforms -> (B, 257, T, 2) real/imag spectra."""
    c = get_audio_contract(contract)
    if c.is_legacy:
        from vaani.dsp import stft
        return stft.stft(x)
    shape = x.shape
    _, frames, lp, rp = pads(shape[-1], c)
    x = x.reshape(-1, shape[-1])
    a, _ = _windows_t(c, x.dtype, x.device)
    fr = F.pad(x, (lp, rp)).unfold(-1, c.k, c.hop)            # (B, T, K)
    assert fr.shape[1] == frames
    z = torch.view_as_real(torch.fft.rfft(fr * a, dim=-1))     # (B, T, 257, 2)
    z = z.transpose(1, 2)
    return z.reshape(*shape[:-1], *z.shape[1:])


def synthesize(z, lengths, contract):
    """(B, 257, T, 2) spectra, per-item lengths -> (aligned waveforms (B, max(lengths)), valid-sample mask).
    Samples past an item's length are zeroed and masked out. Differentiable."""
    c = get_audio_contract(contract)
    if torch.is_tensor(lengths):
        lengths = lengths.to(z.device).reshape(-1)
        n = int(lengths.max())
    else:   # host lengths: no copy or sync when all are equal (training crops), so the step stays graph-capturable
        lens = [int(v) for v in lengths]
        n = max(lens)
        lengths = None if min(lens) == n else torch.tensor(lens, device=z.device)
    if c.is_legacy:
        from vaani.dsp import stft
        y = stft.istft(z, length=n)
    else:
        b, _, t, _ = z.shape
        _, s = _windows_t(c, z.dtype, z.device)
        fr = torch.fft.irfft(torch.view_as_complex(z.transpose(1, 2).contiguous()), n=c.k, dim=-1) * s   # (B,T,K)
        total = c.k + (t - 1) * c.hop
        y = F.fold(fr.transpose(1, 2), (1, total), (1, c.k), stride=(1, c.hop)).reshape(b, total)
        start = c.k - c.hop
        if start + n > total:
            raise ValueError(f"{t} frames cannot cover {n} samples under {c.audio_contract_id}")
        y = y[:, start:start + n]
    if lengths is None:
        mask = torch.ones(z.shape[0], n, dtype=torch.bool, device=z.device)
    else:
        mask = torch.arange(n, device=z.device)[None, :] < lengths[:, None]
    return y * mask, mask


def frame_validity(avail, contract, n_frames: int | None = None):
    """Per-sample availability (B, N) {0,1} -> per-frame validity (B, T). A frame is invalid if any real sample
    of its 512-sample analysis support is unavailable; synthetic padding counts as known (available).
    For C0 this equals pipeline.frame_avail bit for bit."""
    c = get_audio_contract(contract)
    bad = 1.0 - avail.to(torch.float32)
    n = bad.shape[-1]
    t = c.n_frames(n) if n_frames is None else n_frames
    if c.is_legacy:
        # frame k covers [k*256 - 256, k*256 + 256) clipped to [0, N)
        lp = c.k // 2
        rp = max(0, (t - 1) * c.hop + c.k // 2 - n)
    else:
        lp = c.k - c.hop
        rp = max(0, (t - 1) * c.hop + c.hop - n)
    pooled = F.max_pool1d(F.pad(bad[:, None], (lp, rp)), c.k, c.hop)[:, 0]
    return 1.0 - pooled[:, :t]


def boundary_weights(n: int, contract):
    """(T,) fraction of each frame's synthesis support [(j+1)H - L, (j+1)H) inside [0, N): the native-domain loss
    weights (Section 3.1). Normalise weighted spectral terms by their sum."""
    c = get_audio_contract(contract)
    t = c.n_frames(n)
    j = np.arange(t)
    lo = np.clip((j + 1) * c.hop - c.support, 0, n)
    hi = np.clip((j + 1) * c.hop, 0, n)
    return (hi - lo).astype(np.float64) / c.support


# ---- numpy offline twins ---------------------------------------------------------------------
def np_analyze(x: np.ndarray, contract) -> np.ndarray:
    """(N,) -> (257, T) complex128 for float64 input (complex64 for float32)."""
    c = get_audio_contract(contract)
    if c.is_legacy:
        from vaani.dsp import stft
        return stft.np_stft(x)
    _, frames, lp, rp = pads(len(x), c)
    a, _, _ = c.windows()
    xp = np.pad(np.asarray(x), (lp, rp))
    fr = np.lib.stride_tricks.sliding_window_view(xp, c.k)[::c.hop]
    assert fr.shape[0] == frames
    out = np.fft.rfft(fr * a.astype(xp.dtype, copy=False), axis=1).T
    return out.astype(np.complex64) if xp.dtype == np.float32 else out


def np_synthesize(z: np.ndarray, n: int, contract) -> np.ndarray:
    c = get_audio_contract(contract)
    if c.is_legacy:
        raise ValueError("np_synthesize: C0 uses the legacy centered transform")
    _, s = c.windows()[0], c.windows()[2]
    t = z.shape[1]
    fr = np.fft.irfft(z.T, n=c.k, axis=1) * s
    y = np.zeros(c.k + (t - 1) * c.hop, fr.dtype)
    for j in range(t):
        y[j * c.hop:j * c.hop + c.k] += fr[j]
    start = c.k - c.hop
    return y[start:start + n]


# ---- numpy streaming -------------------------------------------------------------------------
class StreamAnalyzer:
    """One channel's analysis: push(H samples) -> (257,) spectrum of the last K samples. Initial history is known
    zero padding."""

    def __init__(self, contract, dtype=np.float32):
        self.c = get_audio_contract(contract)
        if self.c.is_legacy:
            raise ValueError("StreamAnalyzer is for low-delay contracts")
        self.dtype = dtype
        self.a = self.c.windows()[0].astype(dtype)
        self.reset()

    def reset(self):
        self.hist = np.zeros(self.c.k - self.c.hop, self.dtype)

    def push(self, hop: np.ndarray) -> np.ndarray:
        hop = np.asarray(hop, self.dtype)
        if hop.shape != (self.c.hop,):
            raise ValueError(f"expected one {self.c.hop}-sample hop, got {hop.shape}")
        frame = np.concatenate([self.hist, hop])
        self.hist = frame[self.c.hop:].copy()
        return np.fft.rfft(frame * self.a)

    def export_state(self) -> dict:
        return {"contract": self.c.audio_contract_id, "hist": self.hist.copy()}

    def import_state(self, st: dict):
        if st["contract"] != self.c.audio_contract_id or st["hist"].shape != self.hist.shape:
            raise ValueError("analysis state belongs to a different contract")
        self.hist = np.asarray(st["hist"], self.dtype).copy()


class StreamSynthesizer:
    """push((257,) spectrum) -> H released samples: the final L samples of the windowed inverse FFT, overlap-added
    with the previous frame's L - H pending samples."""

    def __init__(self, contract, dtype=np.float32):
        self.c = get_audio_contract(contract)
        if self.c.is_legacy:
            raise ValueError("StreamSynthesizer is for low-delay contracts")
        self.dtype = dtype
        self.s_tail = self.c.windows()[2][self.c.k - self.c.support:].astype(dtype)
        self.reset()

    def reset(self):
        self.pending = np.zeros(self.c.crossfade, self.dtype)

    def push(self, spec: np.ndarray) -> np.ndarray:
        y = np.fft.irfft(spec, n=self.c.k)[self.c.k - self.c.support:].astype(self.dtype) * self.s_tail
        y[:self.c.crossfade] += self.pending
        self.pending = y[self.c.hop:].copy()
        return y[:self.c.hop]

    def export_state(self) -> dict:
        return {"contract": self.c.audio_contract_id, "pending": self.pending.copy()}

    def import_state(self, st: dict):
        if st["contract"] != self.c.audio_contract_id or st["pending"].shape != self.pending.shape:
            raise ValueError("synthesis state belongs to a different contract")
        self.pending = np.asarray(st["pending"], self.dtype).copy()


def stream_identity(x: np.ndarray, contract, dtype=np.float64) -> np.ndarray:
    """Analysis -> synthesis through the streaming classes, aligned like the offline output (reference/test helper)."""
    c = get_audio_contract(contract)
    an, sy = StreamAnalyzer(c, dtype), StreamSynthesizer(c, dtype)
    data_hops, frames, _, _ = pads(len(x), c)
    xp = np.zeros(frames * c.hop, dtype)
    xp[:len(x)] = x
    out = np.concatenate([sy.push(an.push(xp[j * c.hop:(j + 1) * c.hop])) for j in range(frames)])
    return out[c.release_lead:c.release_lead + len(x)]
