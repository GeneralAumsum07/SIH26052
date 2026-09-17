"""Synthetic impulsive noise for training: fast onsets, varied decay, no corpus dependence."""
import numpy as np

KINDS = ("burst", "click_train", "gated_noise")


def _decay(rng, sr, tau_s):
    # Exponential-envelope colored noise; 1-pole lowpass gives a "thud"
    n = int(sr * min(2.0, tau_s * 6))
    t = np.arange(n) / sr
    x = rng.standard_normal(n)
    a = rng.uniform(0.6, 0.95)
    for i in range(1, n):
        x[i] += a * x[i - 1]
    return (x * np.exp(-t / tau_s)).astype(np.float32)


def generate(rng: np.random.Generator, sr: int = 16000, kind: str | None = None):
    kind = kind or rng.choice(KINDS)
    if kind == "burst":
        x = _decay(rng, sr, rng.uniform(0.02, 0.25))
        pre = int(rng.uniform(0.05, 0.3) * sr)
        x = np.concatenate([np.zeros(pre, np.float32), x])
        onsets = [pre / sr]
    elif kind == "click_train":
        n_clicks = int(rng.integers(3, 12))
        gap = rng.uniform(0.04, 0.15)
        pieces, onsets, pos = [], [], 0.0
        for _ in range(n_clicks):
            c = _decay(rng, sr, rng.uniform(0.003, 0.02))
            g = np.zeros(int(gap * sr), np.float32)
            onsets.append(pos); pos += (len(c) + len(g)) / sr
            pieces += [c, g]
        x = np.concatenate(pieces)
    else:  # gated_noise: wideband noise switched on/off abruptly
        dur = rng.uniform(0.3, 1.5)
        x = rng.standard_normal(int(dur * sr)).astype(np.float32)
        on, off = int(0.1 * sr), int(rng.uniform(0.3, 0.9) * dur * sr)
        x[:on] = 0; x[off:] = 0
        onsets = [on / sr]
    n = int(np.clip(len(x), 0.2 * sr, 2.0 * sr))
    x = np.pad(x, (0, max(0, n - len(x))))[:n]
    x = x / (np.abs(x).max() + 1e-9)
    return x.astype(np.float32), {"kind": str(kind), "onsets_s": [float(o) for o in onsets]}
