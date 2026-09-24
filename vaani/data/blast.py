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

Points 2 and 3 describe the default physics="v1" draw that r7 trained on. physics="v2"
(blast_v2, burst) corrects both: a same-sign reflection from image-source geometry, and
1/r spreading plus ISO 9613-1 absorption at SPL-referenced levels.

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
          distance: float | None = None, physics: str = "v1", **v2):
    """One blast transient, peak-normalised, plus metadata.

    kind      : "small_arms" | "artillery" (drawn if None)
    distance  : v1 only. 0.0 = muzzle-close and harsh, 1.0 = far and dull (drawn if None)
    physics   : "v1" (default; the r7 training draw, bit-exact) | "v2" (blast_v2; takes its keyword arguments)
    returns   : (x float32 in [-1, 1], meta dict)
    """
    if physics == "v2":
        x, m = blast_v2(rng, sr, kind=kind, **v2)
        return (x / (np.abs(x).max() + 1e-30)).astype(np.float32), m
    if physics != "v1" or v2:
        raise ValueError(f"physics={physics!r} with {sorted(v2)}: only v2 takes scene parameters")
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
        # N-wave at t=0, muzzle blast delayed by `lead`; the old nw[lead:] slice dropped it, since lead > L always
        y = np.concatenate([np.zeros(lead), y[:n - lead]]) + nw

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
               "ballistic": bool(ballistic), "ballistic_lead_ms": lead / osr * 1e3 if ballistic else None,
               "onsets_s": [0.0]}


# --- physics v2 (plan M5): SPL-referenced levels, ISO 9613-1 absorption, same-sign ground reflection, bursts ---
# v1 inverted the ground reflection, but rigid or grassy ground near grazing reflects with R > 0 (Maher 2006), and
# v1's distance term is a 1-pole lowpass with no physical scale. v2 returns pressure in Pa at the mic so the caller
# can calibrate SPL to dBFS; the v1 path above stays byte-for-byte what r7 trained on.
P_REF = 20e-6          # Pa, 0 dB SPL
P_ATM = 101325.0       # Pa
C_AIR = 343.0          # m/s at 20 C
V_BULLET = 915.0       # m/s, 5.56 mm INSAS muzzle velocity: sets how far the N-wave leads the muzzle blast
SMALL_ARMS_SPL_1M = (150.0, 160.0)   # dB peak at 1 m, muzzle side (Murphy 2018)
SMALL_ARMS_RANGE_M = (1.0, 300.0)    # log-uniform
ARTILLERY_CHARGE_KG = (0.1, 20.0)    # TNT equivalent, log-uniform
ARTILLERY_RANGE_M = (30.0, 3000.0)   # log-uniform
BURST_ROUNDS = (3, 30)               # inclusive
BURST_RPM = (650.0, 700.0)           # INSAS ~650, AK-203 ~700 cyclic
BURST_JITTER = 0.04                  # relative sd of the inter-shot interval; inferred, no measured value
HEIGHTS_M = (0.3, 1.8)               # source and mic height, prone to standing; inferred
GROUND_R = (0.5, 0.95)               # |R| of the ground reflection, same sign


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def iso9613_alpha_db_per_m(f, temp_c=20.0, rh=50.0, pa_kpa=101.325):
    """ISO 9613-1:1993 pure-tone atmospheric absorption in dB/m (20 C / 50 %: 1 kHz 4.66, 4 kHz 29.7 dB/km)."""
    f = np.asarray(f, dtype=np.float64)
    T, T0, T01, pr = temp_c + 273.15, 293.15, 273.16, 101.325
    pa = pa_kpa / pr
    h = rh * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151) / pa
    frO = pa * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pa * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    return 8.686 * f ** 2 * (1.84e-11 / pa * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / T) / (frO + f ** 2 / frO) + 0.1068 * np.exp(-3352.0 / T) / (frN + f ** 2 / frN)))


def absorb(y, sr, r_m, **atm):
    """Air absorption over r_m metres as a minimum-phase filter: a zero-phase one would ring ahead of the shock front."""
    n = 1 << int(np.ceil(np.log2(2 * len(y) + 1)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    logmag = -iso9613_alpha_db_per_m(f, **atm) * r_m * np.log(10) / 20   # natural log of the amplitude response
    c = np.fft.irfft(logmag, n)                                             # real cepstrum
    fold = np.zeros(n); fold[0] = c[0]; fold[1:n // 2] = 2 * c[1:n // 2]; fold[n // 2] = c[n // 2]
    return np.fft.irfft(np.fft.rfft(y, n) * np.exp(np.fft.rfft(fold)), n)[:len(y)]


def kinney_graham_spl(charge_kg, r_m, surface=True):
    """Peak overpressure of a TNT charge (Kinney & Graham 1985 eq. 13) in dB SPL; a surface burst doubles the charge."""
    w = charge_kg * (2.0 if surface else 1.0)
    z = r_m / w ** (1 / 3)
    ratio = 808 * (1 + (z / 4.5) ** 2) / np.sqrt((1 + (z / 0.048) ** 2) * (1 + (z / 0.32) ** 2) * (1 + (z / 1.35) ** 2))
    return float(20 * np.log10(ratio * P_ATM / P_REF))


def ground_reflection(y, sr, r_m, h_src, h_mic, R):
    """Image-source ground bounce: same sign, delayed by the path difference, scaled by R and the extra spreading."""
    r1, r2 = np.hypot(r_m, h_src - h_mic), np.hypot(r_m, h_src + h_mic)
    d = int(round((r2 - r1) / C_AIR * sr)); g = R * r1 / r2
    out = y.copy()
    if d < len(y):
        out[d:] += g * y[:len(y) - d]
    return out, {"reflection_delay_ms": d / sr * 1e3, "reflection_gain": float(g)}


def blast_v2(rng, sr=16000, kind=None, peak_spl_1m=None, distance_m=None, charge_kg=None, heights_m=None,
             ground_r=None, atm=None):
    """One shot or explosion as pressure in Pa at the mic (float64 at sr), plus meta carrying peak_spl_db.

    small_arms: peak_spl_1m ~ U(150, 160) dB, distance_m ~ logU(1, 300), 1/r spreading from 1 m.
    artillery : charge_kg ~ logU(0.1, 20) TNT, distance_m ~ logU(30, 3000), Kinney-Graham surface burst.
    Positive-phase durations keep the v1 draws: the Kinney-Graham duration formula is not verified (TBD).
    """
    kind = kind or str(rng.choice(KINDS)); osr = sr * OVERSAMPLE
    if kind == "small_arms":
        T = rng.uniform(0.15e-3, 0.6e-3); ballistic = rng.random() < 0.6
        spl1 = float(rng.uniform(*SMALL_ARMS_SPL_1M)) if peak_spl_1m is None else float(peak_spl_1m)
        r = _logu(rng, *SMALL_ARMS_RANGE_M) if distance_m is None else float(distance_m)
        spl = spl1 - 20 * np.log10(max(r, 1e-3)); extra = {"source_spl_1m": spl1, "charge_kg": None}
    elif kind == "artillery":
        T = rng.uniform(2e-3, 9e-3); ballistic = False
        w = _logu(rng, *ARTILLERY_CHARGE_KG) if charge_kg is None else float(charge_kg)
        r = _logu(rng, *ARTILLERY_RANGE_M) if distance_m is None else float(distance_m)
        spl = kinney_graham_spl(w, r); extra = {"source_spl_1m": None, "charge_kg": w}
    else:
        raise ValueError(f"unknown kind {kind!r}")
    # the round outruns its own muzzle blast: over r metres the shock front leads by r (1/c - 1/v)
    lead_s = r * (1 / C_AIR - 1 / V_BULLET) if ballistic else 0.0
    n = int((max(0.35, 30 * T) + lead_s) * osr); lead = int(lead_s * osr)
    y = np.concatenate([np.zeros(lead), _friedlander(T, n - lead, osr)])
    L = None
    if ballistic:
        b = _logu(rng, 1.0, 30.0)                     # miss distance, over the Whitham fit's measured range
        L = 144.39e-6 * b ** 0.1757                   # 5.56 mm N-wave duration fit (Volgyesi 2007)
        y = y + _n_wave(L, n, osr) * rng.uniform(0.4, 1.0)
    y = y * P_REF * 10 ** (spl / 20)                  # direct-path peak at the mic, before absorption
    hs, hm = heights_m if heights_m is not None else (rng.uniform(*HEIGHTS_M), rng.uniform(*HEIGHTS_M))
    R = float(rng.uniform(*GROUND_R)) if ground_r is None else float(ground_r)
    y, refl = ground_reflection(y, osr, r, hs, hm, R)
    y = absorb(resample_poly(y, 1, OVERSAMPLE), sr, r, **(atm or {}))
    pk = float(np.abs(y).max())
    return y, {"kind": kind, "physics": "v2", "distance": None, "distance_m": r, "peak_spl_db": float(20 * np.log10(pk / P_REF)),
               "direct_spl_db": float(spl), "positive_phase_ms": T * 1e3, "ballistic": bool(ballistic),
               "ballistic_lead_ms": lead_s * 1e3 if ballistic else None, "n_wave_ms": None if L is None else L * 1e3,
               "heights_m": (float(hs), float(hm)), "ground_r": R, **refl, **extra, "onsets_s": [0.0]}


def burst(rng, sr=16000, n_rounds=None, rpm=None, peak_spl_1m=None, distance_m=None, heights_m=None, atm=None):
    """Automatic fire from one shooter: n_rounds ~ U{3..30} small-arms shots every 60/rpm s, rpm ~ U(650, 700).
    Pressure in Pa at the mic plus meta; one geometry for the whole burst, per-shot waveform draws."""
    n_rounds = int(rng.integers(BURST_ROUNDS[0], BURST_ROUNDS[1] + 1)) if n_rounds is None else int(n_rounds)
    rpm = float(rng.uniform(*BURST_RPM)) if rpm is None else float(rpm)
    spl1 = float(rng.uniform(*SMALL_ARMS_SPL_1M)) if peak_spl_1m is None else float(peak_spl_1m)
    r = _logu(rng, *SMALL_ARMS_RANGE_M) if distance_m is None else float(distance_m)
    hs, hm = heights_m if heights_m is not None else (rng.uniform(*HEIGHTS_M), rng.uniform(*HEIGHTS_M))
    gaps = 60.0 / rpm * (1 + BURST_JITTER * rng.standard_normal(n_rounds - 1)).clip(0.8, 1.2)
    starts = np.round(np.concatenate([[0.0], np.cumsum(gaps)]) * sr).astype(int)
    shots = [blast_v2(rng, sr, "small_arms", spl1, r, heights_m=(hs, hm), atm=atm)[0] for _ in range(n_rounds)]
    y = np.zeros(int(starts[-1] + max(len(s) for s in shots)))
    for a, s in zip(starts, shots):
        y[a:a + len(s)] += s
    pk = float(np.abs(y).max())
    return y, {"kind": "small_arms_burst", "physics": "v2", "distance": None, "distance_m": r, "source_spl_1m": spl1,
               "peak_spl_db": float(20 * np.log10(pk / P_REF)), "n_rounds": n_rounds, "rpm": rpm,
               "heights_m": (float(hs), float(hm)), "onsets_s": [float(a) / sr for a in starts]}


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
