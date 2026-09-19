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


def detect_onsets(x: np.ndarray, sr: int = 16000, frame_ms: float = 5.0, rise_db: float = 12.0,
                  floor_ms: float = 100.0, min_gap_ms: float = 30.0) -> list[float]:
    """Onset times for a recorded impulse clip, which carries no generator metadata.
    An onset is a frame that jumps rise_db above the median energy of the preceding floor_ms;
    the loudest sample is returned when nothing stands out so callers never get an empty list."""
    fl = max(1, int(sr * frame_ms / 1000)); nf = len(x) // fl
    if nf == 0:
        return [0.0]
    e = 10 * np.log10((x[: nf * fl].reshape(nf, fl) ** 2).mean(axis=1) + 1e-12)
    look = max(1, int(floor_ms / frame_ms)); gap = max(1, int(min_gap_ms / frame_ms))
    # assume silence before the clip so a click on the very first frame still counts as an onset
    e = np.concatenate([np.full(look, e.min()), e])
    onsets, last = [], -gap
    for k in range(look, len(e)):
        floor = np.median(e[k - look:k])
        # crossing, not level: a decaying tail that stays above the floor is one event, not many
        if e[k] - floor >= rise_db and e[k - 1] - floor < rise_db and k - last >= gap:
            onsets.append((k - look) * fl / sr); last = k
    return onsets or [float(int(np.argmax(np.abs(x))) / sr)]
