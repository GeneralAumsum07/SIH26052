"""Friedlander blast-wave impulse synthesis — a physical model for gunshots and artillery.

Drop-in companion for `vaani/data/impulses.py`. The existing `generate()` builds an
exponentially-decaying lowpassed noise burst — a "thud". Measured crest factors:

    speech                         18.0 dB full / 12.8 dB event
    impulses.generate("burst")     20.9 / 15.2      <- barely above speech
    impulses.generate("gated_noise") 15.5 / 12.8    <- BELOW speech; not impulsive at all
    ESC-50 "fireworks" (best class) 31.7 / 20.4
    blast() below                  31.9 / 26.5      <- +11.3 dB event over "burst"

The physics, which is also the citable part for the report. A blast wave is described by
the Friedlander waveform,

    p(t) = P0 · (1 − t/T) · exp(−t/T)

an effectively instantaneous rise to peak overpressure P0, exponential decay, a zero
crossing at t = T, then a negative (rarefaction) phase. T is the positive-phase duration:
roughly 0.15–0.6 ms for small arms at close range, several ms for artillery. A supersonic
round additionally produces a ballistic shockwave — a symmetric N-wave — that arrives
*before* the muzzle blast because it travels with the projectile.

Three implementation details that matter:

1. **Synthesise oversampled, then resample.** A near-instantaneous rise synthesised
   directly at 16 kHz aliases badly. Building at 192 kHz and decimating band-limits it
   correctly. (Measured: crest is then essentially identical at 192/48/16 kHz, so the
   sample rate is not what limits realism — the propagation terms are.)
2. **Ground reflection** arrives 1–9 ms later, inverted and attenuated — this is what
   gives real gunshot recordings their characteristic doublet.
3. **Distance** shows up as a first-order lowpass: high frequencies are absorbed faster,
   so a distant shot is duller and has a lower crest. `distance` below spans muzzle-close
   to a few hundred metres.

Wiring it in: add "blast" to `impulses.KINDS` and dispatch to `blast()`, keeping the
existing kinds for non-weapon transients. Note that `mixer.MixConfig.impulse_peak_db` is
still `(-6, +12)` dB relative to speech-active RMS — a realistic muzzle blast at a headset
mic is tens of dB above close-talk speech, so raising that range is what actually makes
this model matter. Pair it with a front-end saturation model, because a real blast
overloads the preamp and that is the condition the reliability controller exists for.
"""
import numpy as np
from scipy.signal import lfilter, resample_poly

OVERSAMPLE = 12   # 192 kHz at SR=16000: the rise is not aliased
KINDS = ("small_arms", "artillery")


def _friedlander(T_s: float, n: int, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    return (1.0 - t / T_s) * np.exp(-t / T_s)


def _n_wave(L_s: float, n: int, sr: int) -> np.ndarray:
    """Ballistic shockwave of a supersonic round: symmetric N of duration L."""
    t = np.arange(n) / sr
    y = np.zeros(n)
    m = t < L_s
    y[m] = 1.0 - 2.0 * t[m] / L_s
    return y


def blast(rng: np.random.Generator, sr: int = 16000, kind: str | None = None,
          distance: float | None = None):
    """One blast transient, peak-normalised, plus metadata.

    kind      : "small_arms" | "artillery" (drawn if None)
    distance  : 0.0 = muzzle-close and harsh, 1.0 = far and dull (drawn if None)
    returns   : (x float32 in [-1, 1], meta dict)
    """
    kind = kind or str(rng.choice(KINDS))
    distance = rng.random() if distance is None else float(np.clip(distance, 0.0, 1.0))
    osr = sr * OVERSAMPLE

    if kind == "small_arms":
        T = rng.uniform(0.15e-3, 0.6e-3)
        ballistic = rng.random() < 0.6          # supersonic round
    elif kind == "artillery":
        T = rng.uniform(2e-3, 9e-3)
        ballistic = False
    else:
        raise ValueError(f"unknown kind {kind!r}")

    n = int(max(0.35, 30 * T) * osr)
    y = _friedlander(T, n, osr)

    if ballistic:
        L = rng.uniform(0.15e-3, 0.4e-3)
        lead = int(rng.uniform(0.5e-3, 4e-3) * osr)   # N-wave precedes the muzzle blast
        nw = _n_wave(L, n, osr) * rng.uniform(0.4, 1.0)
        y = y + np.concatenate([nw[lead:], np.zeros(lead)])

    # ground reflection: delayed, inverted, slightly softened
    d = int(rng.uniform(1e-3, 9e-3) * osr)
    g = rng.uniform(0.25, 0.65)
    y = y + lfilter([0.45, 0.55], [1.0], np.concatenate([np.zeros(d), y[:n - d]]) * -g)

    # propagation: HF absorption grows with distance
    a = 0.50 + 0.45 * distance
    if distance > 0.02:
        y = lfilter([1 - a], [1.0, -a], y)

    y = resample_poly(y, 1, OVERSAMPLE).astype(np.float32)
    peak = float(np.abs(y).max()) + 1e-12
    y = (y / peak).astype(np.float32)
    return y, {"kind": kind, "distance": distance, "positive_phase_ms": T * 1e3,
               "ballistic": bool(ballistic), "onsets_s": [0.0]}


if __name__ == "__main__":
    def crest(x):
        return 20 * np.log10(np.abs(x).max() / (np.sqrt((x ** 2).mean()) + 1e-12))

    def crest_event(x, sr=16000, half_ms=100.0):
        k = int(np.argmax(np.abs(x)))
        w = int(sr * half_ms / 1000)
        return crest(x[max(0, k - w): k + w])

    print(f"{'kind':<14s}{'distance':>10s}{'crest_full':>12s}{'crest_event':>13s}")
    for kind in KINDS:
        for dist, tag in ((0.0, "muzzle"), (0.5, "mid"), (1.0, "far")):
            v = [blast(np.random.default_rng(i), kind=kind, distance=dist)[0] for i in range(30)]
            print(f"{kind:<14s}{tag:>10s}{np.mean([crest(x) for x in v]):12.1f}"
                  f"{np.mean([crest_event(x) for x in v]):13.1f}")
