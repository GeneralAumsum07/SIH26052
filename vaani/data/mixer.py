"""Two-channel mixture synthesis.

The reference-mic channel is the whole point of the dual-channel model, so
it is modelled physically (room path) or with an explicit speech-leakage +
independent-noise-filter model (parametric path) - never by copying the
primary. SNR is defined on the *primary* over speech-active frames; the
impulse level is drawn independently and recorded because a file-average
SNR hides how loud a 50 ms burst really was.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, lfilter

from vaani.data import calib

SR = 16000


@dataclass
class MixConfig:
    snr_range: tuple[float, float] = (-10.0, 15.0)
    snr_sampling: str = "uniform"
    snr_bins: tuple[float, ...] = (-10., -5., 0., 5., 15.)
    snr_weights: tuple[float, ...] = (.4, .3, .2, .1)
    p_room: float = 0.6
    p_clean: float = 0.05
    p_clip: float = 0.10
    p_ref_dropout: float = 0.05
    p_wind: float = 0.15
    ref_speech_gain_db: tuple[float, float] = (-20.0, -8.0)
    ref_delay_ms: tuple[float, float] = (0.1, 0.5)
    mic_mismatch_db: float = 3.0
    impulse_peak_db: tuple[float, float] = (-6.0, 12.0)   # relative to speech-active RMS on primary
    # r3 physics knobs; defaults keep the r1/r2 eval-set contract (same rng stream, same hashes)
    impulse_kinds: tuple[str, ...] | None = None   # synthetic kinds to draw from; None = impulses.KINDS
    impulse_room: bool = False                     # room path: impulse arrives through a noise RIR, not a 2-tap decorrelator
    overload_softclip: bool = False                # a burst past full scale saturates the ADC instead of scaling the speech down
    speech_rms_db: tuple[float, float] | None = None  # recorder gain: speech-active RMS in dBFS; None = corpus level as stored
    # mixer v2 (plan 11.5): 1 keeps every r1-r7 render and training item bit-exact; 2 routes to mix_v2 with the v2 block
    version: int = 1
    v2: dict | None = None


def sample_snr(rng, cfg: MixConfig):
    """Default keeps precisely the old RNG draw; opt-in recipes alter training only."""
    lo, hi = cfg.snr_range
    if not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("invalid snr_range")
    if cfg.snr_sampling == "uniform":
        return float(rng.uniform(lo, hi))
    if cfg.snr_sampling == "triangular_low":
        return float(rng.triangular(lo, lo, hi)) if lo < hi else float(lo)
    if cfg.snr_sampling != "stratified":
        raise ValueError("snr_sampling must be uniform, triangular_low or stratified")
    bins, weights = np.asarray(cfg.snr_bins, float), np.asarray(cfg.snr_weights, float)
    if bins.ndim != 1 or len(bins) < 2 or not np.isfinite(bins).all() or not (np.diff(bins) > 0).all() or bins[0] != lo or bins[-1] != hi:
        raise ValueError("snr_bins must increase and span snr_range exactly")
    if weights.shape != (len(bins)-1,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("snr_weights must be nonnegative with positive sum, one per interval")
    i = int(rng.choice(len(weights), p=weights / weights.sum()))
    return float(rng.uniform(bins[i], bins[i+1]))


def softclip(x: np.ndarray, knee: float = 0.7) -> np.ndarray:
    """Linear below `knee`, tanh-compressed above, asymptote at 1.0: an ADC front-end that saturates, not a peak normaliser."""
    a = np.abs(x)
    over = a > knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return y.astype(np.float32)


def speech_active_power(x: np.ndarray, frame: int = 320, thresh_db: float = -30.0) -> float:
    f = x[: len(x) // frame * frame].reshape(-1, frame)
    e = (f ** 2).mean(axis=1) + 1e-12
    keep = e > e.max() * 10 ** (thresh_db / 10)
    return float(e[keep].mean()) if keep.any() else float(e.mean())


def _fit(x: np.ndarray, n: int, rng) -> np.ndarray:
    """Loop or crop noise to length n with a random offset."""
    if len(x) >= n:
        o = int(rng.integers(0, len(x) - n + 1)); return x[o:o + n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, (reps,) + (1,) * (x.ndim - 1))[:n]   # tile along time only; (m, 2) rows must stay 2 wide


def _frac_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = len(x); k = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * k * delay_samples), n).astype(np.float32)


def _tilt(x: np.ndarray, db: float) -> np.ndarray:
    """1st-order spectral tilt to emulate mic response mismatch."""
    a = np.clip(db / 40.0, -0.3, 0.3)
    return lfilter([1.0, a], [1.0], x).astype(np.float32)


def _conv2(x: np.ndarray, h2: np.ndarray, n: int) -> np.ndarray:
    return np.stack([fftconvolve(x, h2[m])[:n] for m in range(2)]).astype(np.float32)


def mix(rng, speech, noises, impulse, impulse_onsets_s, bank, cfg: MixConfig, norm_gain: float | None = None,
        scene: dict | None = None):
    """norm_gain: reuse another clip's final peak scaling (the burst clip's, when rendering its twin)
    so the pair differs only by the impulse and not by a level step. scene: mixer v2 only (vaani.data.scenes)."""
    if cfg.version == 2:
        return mix_v2(rng, speech, noises, impulse, impulse_onsets_s, bank, cfg, norm_gain=norm_gain, scene=scene)
    if cfg.version != 1:
        raise ValueError("mix version must be 1 or 2")
    n = len(speech)
    meta = {"clean_bucket": False, "clipped": False, "ref_dropout": False, "impulse_peak_db": None,
            "impulse_onsets_s": [], "ref_speech_gain_db": None, "norm_gain": 1.0, "snr_achieved_db": None}
    speech = speech.astype(np.float32)
    if cfg.speech_rms_db is not None:
        # the overload headroom is whatever sits between the speech and full scale, so the recorder gain must vary too
        tgt = 10 ** (float(rng.uniform(*cfg.speech_rms_db)) / 20)
        speech = speech * (tgt / (np.sqrt(speech_active_power(speech)) + 1e-9))

    use_room = bank is not None and rng.random() < cfg.p_room
    meta["path"] = "room" if use_room else "param"
    if cfg.overload_softclip:
        meta["overloaded"] = False  # only under the r3 model: r1/r2 eval-set hashes cover the meta JSON byte-for-byte

    r = None
    if use_room:
        r = bank.sample(rng)
        # normalise so the primary direct path has unit gain: SNR is defined at the primary
        h_s = r["speech"] / (np.abs(r["speech"][0]).max() + 1e-9)
        s2 = _conv2(speech, h_s, n)
        clean = s2[0].copy()
        noise2 = np.zeros((2, n), np.float32)
        for i, nz in enumerate(noises):
            if nz.ndim == 2:   # measured two-mic pair (DEMAND): its inter-channel relation is real, keep it
                noise2 += _fit(nz, n, rng).T; continue
            h_n = r["noise"][i % len(r["noise"])]
            h_n = h_n / (np.abs(h_n[0]).max() + 1e-9)
            noise2 += _conv2(_fit(nz, n, rng), h_n, n)
        meta["ref_speech_gain_db"] = float(10 * np.log10((s2[1] ** 2).sum() / ((s2[0] ** 2).sum() + 1e-12) + 1e-12))
    else:
        clean = speech.copy()
        g_db = float(rng.uniform(*cfg.ref_speech_gain_db))
        d = rng.uniform(*cfg.ref_delay_ms) * SR / 1000
        s2 = np.stack([speech, _frac_delay(speech, d) * 10 ** (g_db / 20)])
        meta["ref_speech_gain_db"] = g_db
        noise2 = np.zeros((2, n), np.float32)
        for nz in noises:
            nz = _fit(nz, n, rng)
            if nz.ndim == 2:   # measured pair: no synthetic arrival paths either
                noise2 += nz.T; continue
            # independent short random filters per channel: different arrival paths
            for m in range(2):
                taps = rng.normal(0, 1, 3); taps[0] = 1.0; taps[1:] *= 0.3
                noise2[m] += lfilter(taps, [1.0], nz).astype(np.float32)

    if rng.random() < cfg.p_clean:
        meta["clean_bucket"] = True; meta["snr_db"] = meta["snr_achieved_db"] = np.inf; meta["noise_class"] = "clean"
        return s2.astype(np.float32), clean, meta

    # --- scale noise to target SNR on the primary, speech-active region ---
    snr = sample_snr(rng, cfg)
    ps = speech_active_power(clean); pn = (noise2[0] ** 2).mean() + 1e-12
    noise2 *= np.sqrt(ps / (pn * 10 ** (snr / 10)))
    meta["snr_db"] = snr  # target SNR the noise was scaled to hit; later augmentations deliberately perturb it, not recomputed

    out = s2 + noise2
    sp = s2[0].copy()  # speech-only primary, carried through the linear augmentations to measure achieved SNR

    # --- impulse event: level set independently of SNR, recorded as peak ---
    # always draw one int to advance rng, so the main stream is identical whether or
    # not an impulse is present (needed for the twin-clip recovery-time comparison)
    imp_rng = np.random.default_rng(int(rng.integers(2**31)))
    if impulse is not None:
        pk_db = float(imp_rng.uniform(*cfg.impulse_peak_db))
        start = int(imp_rng.integers(0, max(1, n - len(impulse))))
        seg = impulse[: n - start]
        ref_rms = np.sqrt(ps)
        # impulses are far-field: similar level at both mics, small decorrelation
        decor = imp_rng.uniform(-0.2, 0.2)  # drawn unconditionally so the stream matches whichever branch runs
        if cfg.impulse_room and r is not None:
            # the same room the noise came through: real reverberant tail on both mics, ref no longer a 2-tap copy
            h_i = r["noise"][-1]; h_i = h_i / (np.abs(h_i[0]).max() + 1e-9)
            imp2 = _conv2(seg, h_i, len(seg))
        else:
            imp2 = np.stack([seg, lfilter([1.0, decor], [1.0], seg)]).astype(np.float32)
        imp2 *= ref_rms * 10 ** (pk_db / 20)
        out[:, start:start + len(seg)] += imp2
        meta["impulse_peak_db"] = pk_db
        meta["impulse_onsets_s"] = [start / SR + o for o in impulse_onsets_s]
        if cfg.overload_softclip and np.abs(out).max() > 1.0:
            # both mics saturate; the clean target is untouched (nothing recovers a slammed ADC) and the final
            # normaliser then leaves speech at its recorded level instead of burying it under a 45 dB burst
            out = softclip(out); meta["overloaded"] = True

    # --- common augmentations ---
    g = rng.uniform(-cfg.mic_mismatch_db, cfg.mic_mismatch_db)
    out[1] *= 10 ** (g / 20)
    t0 = rng.uniform(-3, 3); out[0] = _tilt(out[0], t0); sp = _tilt(sp, t0); out[1] = _tilt(out[1], rng.uniform(-3, 3))

    if rng.random() < cfg.p_wind:
        w = lfilter([1.0], [1.0, -0.995], rng.standard_normal(n)).astype(np.float32)
        w *= np.sqrt(ps) * 10 ** (rng.uniform(-20, -5) / 20) / (w.std() + 1e-9)
        out[int(rng.integers(0, 2))] += w

    if rng.random() < cfg.p_ref_dropout:
        a = int(rng.integers(0, n)); b = min(n, a + int(rng.uniform(0.2, 1.0) * SR))
        out[1, a:b] *= 10 ** (-40 / 20); meta["ref_dropout"] = True

    if rng.random() < cfg.p_clip:
        lvl = np.abs(out[0]).max() * rng.uniform(0.3, 0.8)
        out[0] = np.clip(out[0], -lvl, lvl); meta["clipped"] = True

    # keep everything inside [-1, 1] without changing SNR: scale mix and clean together
    peak = np.abs(out).max()
    gain = norm_gain if norm_gain is not None else (0.99 / peak if peak > 0.99 else 1.0)
    out *= gain; clean *= gain; meta["norm_gain"] = float(gain)
    # what the primary actually carries after impulse/wind/clip/tilt: snr_db is only the pre-augmentation target
    resid = out[0] - sp * gain
    meta["snr_achieved_db"] = float(10 * np.log10(speech_active_power(sp * gain) / ((resid ** 2).mean() + 1e-12)))
    return out.astype(np.float32), clean.astype(np.float32), meta


# ---------------------------------------------------------------------------------------------------------------
# Mixer v2 (plan 11.5 M1-M4, M7, M10, M11). Built in dB SPL at each mic, mapped to float by the ICS-43434 chain
# (vaani.data.calib); SNR is an output of the scene levels, not a draw. Only reached with MixConfig(version=2).
# ---------------------------------------------------------------------------------------------------------------

C_SOUND = 343.0
V2_DEFAULTS = dict(
    # M2 boom -> reference speech transfer (1/r geometry: r_boom 2-3 cm, r_ref 8-14 cm) plus head shadow
    ild_db=(8.5, 16.9), ref_delay_ms=(0.15, 0.35), hf_shadow_db=(-6.0, 0.0), head_radius_m=0.0875,
    # M2 out-of-physics tail (diag requirements 1 and 4); tail_share is ablation 3b (0 / 0.10 / 0.25)
    # tail items all land in -6..+3 dB, so the share of items there equals tail_share (mono 10 %, stereo 5 % at 0.25)
    tail_share=0.25, tail_mix={"mono": 0.4, "stereo": 0.2, "low_ild": 0.4}, low_ild_gain_db=(-6.0, 3.0),
    p_quiet=0.05, quiet_gain_db=(-20.0, -16.9),
    stereo_corr=(0.6, 0.95), stereo_gain_db=(-1.0, 1.0), stereo_delay_ms=(-0.5, 0.5),
    # M3 noise field
    mic_spacing_m=0.12, p_cylindrical=0.3, point_ild_db=12.0, near_ild_db=(6.0, 12.0), near_pos_share=0.5, far_ild_db=2.0,
    stft_n=512, stft_hop=128,
    # M4 wind: level at 5 m/s unprotected (inferred 75-90 dB SPL, TBD), +12 dB per doubling (rho V^2)
    wind_ref_db=(75.0, 90.0), wind_ref_mps=5.0, windscreen_db=0.0, wind_frame_s=0.02,
    # M7 front end
    front_end=True, mic_gain_db=1.0, self_noise_db=-93.0,
    # M10 seams, M11 Lombard
    xfade_s=0.05, lombard=True,
    path=None,   # force "room" or "param" (data gates); None = scene rir + cfg.p_room
)


def v2_params(cfg: MixConfig) -> dict:
    p = dict(V2_DEFAULTS); p.update(cfg.v2 or {})
    bad = set(p) - set(V2_DEFAULTS)
    if bad:
        raise ValueError(f"unknown mix.v2 keys: {sorted(bad)}")
    return p


def fit_xfade(x: np.ndarray, n: int, rng, xfade: int) -> np.ndarray:
    """M10: crop, or loop with equal-power crossfades (seams are uncorrelated, so sin/cos keeps the power flat)."""
    L = len(x)
    if L >= n:
        o = int(rng.integers(0, L - n + 1)); return x[o:o + n]
    xf = int(min(xfade, L // 4))
    step = L - xf
    reps = int(np.ceil((n + L) / step)) + 1
    y = np.zeros((reps * step + xf,) + x.shape[1:], np.float32)
    w = np.ones(L, np.float32)
    if xf > 0:
        t = (np.arange(xf) + 0.5) / xf
        w[:xf] = np.sin(0.5 * np.pi * t); w[-xf:] = np.cos(0.5 * np.pi * t)
    w = w.reshape((L,) + (1,) * (x.ndim - 1))
    for k in range(reps):
        y[k * step:k * step + L] += x * w
    o = xf + int(rng.integers(0, L))
    return y[o:o + n]


def concat_xfade(clips: list, xfade: int) -> np.ndarray:
    """M10: concatenate distinct clips with equal-power crossfades (for the scene sampler to fill a long crop)."""
    out = np.asarray(clips[0], np.float32)
    for c in clips[1:]:
        c = np.asarray(c, np.float32); xf = int(min(xfade, len(out) // 4, len(c) // 4))
        if xf == 0:
            out = np.concatenate([out, c]); continue
        t = ((np.arange(xf) + 0.5) / xf).reshape((xf,) + (1,) * (c.ndim - 1))
        mid = out[-xf:] * np.cos(0.5 * np.pi * t) + c[:xf] * np.sin(0.5 * np.pi * t)
        out = np.concatenate([out[:-xf], mid, c[xf:]])
    return out


def diffuse_coherence(f, d: float = 0.12, model: str = "spherical") -> np.ndarray:
    """Habets-Cohen-Gannot target coherence for a diffuse field: spherical sinc(2 pi f d / c) or cylindrical J0."""
    x = 2 * np.pi * np.asarray(f, float) * d / C_SOUND
    if model == "spherical":
        return np.sinc(x / np.pi)
    from scipy.special import j0
    return j0(x)


def _stft(x, nfft, hop):
    from scipy.signal import stft
    return stft(x, nperseg=nfft, noverlap=nfft - hop, boundary="zeros", padded=True)


def _istft(X, nfft, hop, n):
    from scipy.signal import istft
    return istft(X, nperseg=nfft, noverlap=nfft - hop, boundary=True)[1][:n].astype(np.float32)


def diffuse_pair(rng, x: np.ndarray, gamma=None, d: float = 0.12, model: str = "spherical", nfft: int = 512,
                 hop: int = 128) -> np.ndarray:
    """M3 two-channel bed per STFT bin: X1 = N1, X2 = g N1 + sqrt(1 - g^2) N2 (Habets, Cohen, Gannot 2008). N2 keeps
    N1's local envelope (3x3-smoothed power) with independent Rayleigh fine structure, so a non-stationary bed has
    the same envelope on both mics, as a diffuse field does. gamma: a constant overrides the diffuse curve."""
    from scipy.ndimage import uniform_filter
    n = len(x)
    f, _, X1 = _stft(x, nfft, hop)
    g = diffuse_coherence(f * SR, d, model) if gamma is None else np.full(len(f), float(gamma))
    env = np.sqrt(uniform_filter(np.abs(X1) ** 2, size=3, mode="nearest"))
    # W is the STFT of real white noise, not i.i.d. complex draws: random STFT coefficients are not a consistent
    # STFT and the ISTFT keeps only ~1/4 of their energy at 75 % overlap (coherence 0.63, not 0.37, at 1 kHz)
    W = _stft(rng.standard_normal(n), nfft, hop)[2]
    W = W / np.sqrt(np.mean(np.abs(W) ** 2) + 1e-30)
    X2 = g[:, None] * X1 + np.sqrt(np.maximum(0.0, 1 - g ** 2))[:, None] * env * W
    return np.stack([x.astype(np.float32), _istft(X2, nfft, hop, n)])


def _fft_transfer(x: np.ndarray, H_fn) -> np.ndarray:
    n = len(x); m = n + 256   # zero pad: delays and the shelf's phase must not wrap onto the start
    f = np.fft.rfftfreq(m, 1 / SR)
    return np.fft.irfft(np.fft.rfft(x, m) * H_fn(f), m)[:n].astype(np.float32)


def head_transfer(x: np.ndarray, gain_db: float, delay_s: float, shadow_db: float = 0.0, a: float = 0.0875) -> np.ndarray:
    """M2 boom -> reference: broadband gain, delay, and a Brown-Duda rigid-sphere head-shadow shelf
    (1 + j alpha w / 2w0) / (1 + j w / 2w0), w0 = c / a: unity at LF, alpha = 10^(shadow/20) at HF, corner ~1.25 kHz."""
    al = 10 ** (shadow_db / 20); w0 = C_SOUND / a

    def H(f):
        w = 2 * np.pi * f
        return 10 ** (gain_db / 20) * np.exp(-1j * w * delay_s) * (1 + 1j * al * w / (2 * w0)) / (1 + 1j * w / (2 * w0))
    return _fft_transfer(x, H)


def _ar_from_psd(psd: np.ndarray, order: int) -> np.ndarray:
    from scipy.linalg import solve_toeplitz
    r = np.fft.irfft(psd)[:order + 1]
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    return np.concatenate([[1.0], a])


_WIND_AR: dict = {}


def wind_ar5(fc_hz: float) -> np.ndarray:
    """AR(5) fitted (Yule-Walker) to a low-pass wind PSD 1 / (1 + (f / fc)^4). The shape follows Nelke-Vary 2014
    (AR(5), energy almost all below 1 kHz); their measured coefficients are TBD, so the fit stands in for them."""
    k = round(float(fc_hz), 0)
    if k not in _WIND_AR:
        f = np.linspace(0, SR / 2, 8193)
        _WIND_AR[k] = _ar_from_psd(1 / (1 + (f / k) ** 4), 5)
    return _WIND_AR[k]


try:   # numba is optional: the Markov chain is the only per-frame Python loop in v2
    from numba import njit as _njit
except Exception:   # noqa: BLE001 - any import failure means the pure-Python path
    _njit = None


def _markov_states_py(u, P, s0):
    s = np.empty(len(u), np.int64); cur = s0
    for i in range(len(u)):
        c = 0.0; nxt = 2
        for j in range(3):
            c += P[cur, j]
            if u[i] < c:
                nxt = j; break
        cur = nxt; s[i] = cur
    return s


_markov_states = _njit(cache=False)(_markov_states_py) if _njit is not None else _markov_states_py
# 3-state long-term gain (none / low / high), per 20 ms frame; dwell ~1 s. Values inferred (Nelke-Vary's are TBD).
WIND_P = np.array([[0.97, 0.03, 0.00], [0.02, 0.95, 0.03], [0.00, 0.04, 0.96]])
WIND_STATE_DB = np.array([-30.0, -10.0, 0.0])
WIND_WEIBULL_K = 2.0   # short-term gain shape (Nelke-Vary use Weibull short-term gains; k inferred)


def wind_pair(rng, n: int, speed_mps: float, ref_db: float, ref_mps: float = 5.0, windscreen_db: float = 0.0,
              frame_s: float = 0.02) -> tuple[np.ndarray, dict]:
    """M4: independent AR(5) turbulence per mic (Corcos coherence is ~0 above 20-100 Hz at 12 cm) under a shared
    3-state Markov gust envelope, with independent per-mic Weibull short-term gains. Level in the 'high' state is
    ref_db + 40 log10(V / ref_mps) dB SPL (rho V^2 -> +12 dB per doubling); it may exceed the speech."""
    spl = ref_db + 40 * np.log10(max(speed_mps, 1e-3) / ref_mps) + windscreen_db
    fc = float(np.clip(40 + 20 * speed_mps, 50, 250))   # corner rises with V/D (Strasberg); slope inferred
    a = wind_ar5(fc)
    hop = int(frame_s * SR); nf = n // hop + 2
    pi = np.linalg.matrix_power(WIND_P, 200)[0]
    s0 = int(rng.choice(3, p=pi / pi.sum()))
    states = _markov_states(rng.random(nf), WIND_P, s0)
    shared = 10 ** (WIND_STATE_DB[states] / 20)
    t_f = np.arange(nf) * hop; t = np.arange(n)
    out = np.zeros((2, n), np.float32)
    amp = calib.spl_to_float_rms(spl)
    for m in range(2):
        w = lfilter([1.0], a, rng.standard_normal(n + 2048))[2048:]   # drop the AR start-up transient
        w /= (w.std() + 1e-12)
        g = rng.weibull(WIND_WEIBULL_K, nf); g /= np.sqrt(np.mean(g ** 2))
        out[m] = (w * np.interp(t, t_f, shared * g) * amp).astype(np.float32)
    return out, {"wind_spl_db": float(spl), "wind_fc_hz": fc, "wind_states": np.bincount(states, minlength=3).tolist()}


def _level(x, weighting):
    return {"active": calib.active_rms_db, "rms": calib.rms_db, "A": calib.a_weighted_rms_db}[weighting](x)


def _to_spl(pair: np.ndarray, spl: float, weighting: str) -> np.ndarray:
    """Scale a (2, n) pair so the primary channel sits at spl dB SPL; the inter-mic relation is kept."""
    lvl = _level(pair[0], weighting)
    return (pair * np.float32(10 ** ((calib.spl_to_float_rms_db(spl) - lvl) / 20))).astype(np.float32)


def _point_pair(rng, x: np.ndarray, ild_db: float, d: float) -> np.ndarray:
    tau = d * np.cos(rng.uniform(0, np.pi)) / C_SOUND   # far-field arrival difference, |tau| <= d / c
    return np.stack([x.astype(np.float32), head_transfer(x, -ild_db, tau)])


def mix_v2_ref_draw(rng, p: dict) -> tuple[str, float, float, float]:
    """M2 reference draw: (mode, broadband gain dB re primary, delay s, HF shadow dB). Tail modes (mono, produced
    stereo, low ILD) all sit in -6..+3 dB, so the share of items there is tail_share; physical items carry the
    boom->reference ILD, with p_quiet of them at the obstructed -20 dB end."""
    tail = p["tail_mix"]
    if rng.random() < p["tail_share"]:
        keys = sorted(tail); w = np.asarray([tail[k] for k in keys], float)
        mode = keys[int(rng.choice(len(keys), p=w / w.sum()))]
    else:
        mode = "physical"
    if mode == "mono":
        return mode, 0.0, 0.0, 0.0
    if mode == "stereo":
        return mode, float(rng.uniform(*p["stereo_gain_db"])), float(rng.uniform(*p["stereo_delay_ms"])) / 1000, 0.0
    if mode == "low_ild":
        gain = float(rng.uniform(*p["low_ild_gain_db"]))
    elif rng.random() < p["p_quiet"]:
        gain = float(rng.uniform(*p["quiet_gain_db"]))
    else:
        gain = -float(rng.uniform(*p["ild_db"]))
    return mode, gain, float(rng.uniform(*p["ref_delay_ms"])) / 1000, float(rng.uniform(*p["hf_shadow_db"]))


def mix_v2(rng, speech, noises, impulse, impulse_onsets_s, bank, cfg: MixConfig, norm_gain: float | None = None,
           scene: dict | None = None):
    """Returns (mix (2, n), clean (n,), meta) like mix(). clean is the boom speech after the linear front end
    (mic gain, 60 Hz HPF) at its calibrated level; nothing is peak-normalised, so overload is physical."""
    from vaani.data import scenes as _scenes
    p = v2_params(cfg)
    n = len(speech); xf = int(p["xfade_s"] * SR)
    if scene is None:
        scene = _scenes.sample_scene(rng, crop_s=n / SR)
    meta = {"mix_version": 2, "scene": scene["name"], "clean_bucket": False, "clipped": False, "ref_dropout": False,
            "impulse_peak_db": None, "impulse_onsets_s": [], "norm_gain": 1.0, "effort": scene["effort"],
            "speech_spl_db": float(scene["speech_spl"])}
    speech = np.asarray(speech, np.float32)
    meta["lombard"] = bool(p["lombard"] and scene.get("lombard"))
    if meta["lombard"]:
        speech = calib.lombard_tilt(speech)   # alpha ratio only; the +1.9 st F0 shift is TBD (not applied)

    # --- reference mode (M2 physics or its deliberate tail) ---
    mode, gain, delay, shadow = mix_v2_ref_draw(rng, p)
    meta["ref_mode"] = mode

    if p["path"] is not None:
        use_room = p["path"] == "room" and bank is not None
    else:
        use_room = bank is not None and scene["rir"] in ("room", "armoured") and rng.random() < cfg.p_room
    meta["path"] = "room" if use_room else "param"
    r = None
    if use_room:
        try:
            r = bank.sample(rng, armoured=scene["rir"] == "armoured")
        except TypeError:   # a bank without armoured selection (tests, older loaders)
            r = bank.sample(rng)
        h_s = r["speech"] / (np.abs(r["speech"][0]).max() + 1e-9)
        s_p = fftconvolve(speech, h_s[0])[:n].astype(np.float32)
        s_r_room = fftconvolve(speech, h_s[1])[:n].astype(np.float32)
    else:
        s_p = speech.copy(); s_r_room = None
    g0 = np.float32(10 ** ((calib.spl_to_float_rms_db(scene["speech_spl"]) - calib.active_rms_db(s_p)) / 20))
    s_p = s_p * g0
    if s_r_room is not None:
        s_r_room = s_r_room * g0

    d_mic = p["mic_spacing_m"]
    if mode in ("physical", "low_ild"):
        # the shadow shapes the spectrum only; the broadband level is re-set to the drawn gain so the M2 histogram holds
        if s_r_room is not None:   # the room's own reference RIR keeps its reverberant tail
            s_r = head_transfer(s_r_room, 0.0, 0.0, shadow, p["head_radius_m"])
        else:
            s_r = head_transfer(s_p, 0.0, delay, shadow, p["head_radius_m"])
        s_r *= np.float32(10 ** ((gain - (calib.rms_db(s_r) - calib.rms_db(s_p))) / 20))
    elif mode == "stereo":
        s_r = head_transfer(s_p, gain, delay)
    else:   # mono: the whole reference becomes a copy of the primary after the front end
        s_r = s_p.copy()
    s2 = np.stack([s_p, s_r]).astype(np.float32)
    meta["ref_speech_gain_db"] = float(calib.rms_db(s_r) - calib.rms_db(s_p))

    # --- noise field (M3), scene levels (M1), seams (M10) ---
    noise2 = np.zeros((2, n), np.float32)
    srcs = scene.get("sources") or []
    rho = float(rng.uniform(*p["stereo_corr"]))
    comp = []
    for i, nz in enumerate(noises):
        spec = srcs[i] if i < len(srcs) else dict(role="bed", spl=srcs[0]["spl"] if srcs else 60.0, weighting="A")
        nz = fit_xfade(np.asarray(nz, np.float32), n, rng, xf)
        role = spec["role"]; ild = None
        if nz.ndim == 2 and role != "bed":   # a point or near source is spatialised here: one channel of the pair
            nz = np.ascontiguousarray(nz[:, 0])
        if nz.ndim == 2:   # measured two-mic bed (DEMAND): its inter-channel relation is real, keep it
            pair = nz.T.astype(np.float32)
        elif mode == "stereo":   # produced stereo: broadband channel correlation rho, no spatial physics
            pair = diffuse_pair(rng, nz, gamma=rho, nfft=p["stft_n"], hop=p["stft_hop"])
        elif role == "bed":
            model = "cylindrical" if rng.random() < p["p_cylindrical"] else "spherical"
            pair = diffuse_pair(rng, nz, d=d_mic, model=model, nfft=p["stft_n"], hop=p["stft_hop"])
        else:
            if role == "near":
                ild = float(rng.uniform(*p["near_ild_db"])) * (1 if rng.random() < p["near_pos_share"] else -1)
            else:
                ild = float(rng.uniform(-p["point_ild_db"], p["point_ild_db"]))
            if use_room:
                h_n = r["noise"][i % len(r["noise"])]; h_n = h_n / (np.abs(h_n[0]).max() + 1e-9)
                pair = _conv2(nz, h_n, n)
                pair[1] *= np.float32(10 ** (-ild / 20))   # the source's level step between the mics, on top of the room
            else:
                pair = _point_pair(rng, nz, ild, d_mic)
        pair = _to_spl(pair, float(spec["spl"]), spec.get("weighting", "A"))
        noise2 += pair
        comp.append({"role": role, "spl_db": float(spec["spl"]), "ild_db": ild, "tag": spec.get("tag")})
    meta["noise_sources"] = comp

    wind_mps = float(scene.get("wind_mps") or 0.0)
    if wind_mps > 0:
        w2, wm = wind_pair(rng, n, wind_mps, float(rng.uniform(*p["wind_ref_db"])), p["wind_ref_mps"],
                           p["windscreen_db"], p["wind_frame_s"])
        noise2 += w2; meta.update(wm)
    meta["wind_mps"] = wind_mps

    if rng.random() < cfg.p_clean:
        noise2[:] = 0.0; meta["clean_bucket"] = True

    # --- impulse event: peak SPL from the scene, arriving far-field (small ILD) or through the room ---
    imp2 = None
    if impulse is not None and not meta["clean_bucket"]:
        ev = scene.get("event") or {}
        pk_spl = float(ev["peak_spl"]) if "peak_spl" in ev else float(scene["speech_spl"] + rng.uniform(*cfg.impulse_peak_db))
        start = int(rng.integers(0, max(1, n - len(impulse))))
        seg = np.asarray(impulse[: n - start], np.float32)
        if use_room:
            h_i = r["noise"][-1]; h_i = h_i / (np.abs(h_i[0]).max() + 1e-9)
            ip = _conv2(seg, h_i, len(seg))
        else:
            ip = _point_pair(rng, seg, float(rng.uniform(-p["far_ild_db"], p["far_ild_db"])), d_mic)
        ip *= np.float32(10 ** ((pk_spl - calib.SPL_TO_FLOAT_RMS_DB) / 20) / (np.abs(ip[0]).max() + 1e-12))
        imp2 = np.zeros((2, n), np.float32); imp2[:, start:start + len(seg)] = ip
        meta["impulse_peak_db"] = pk_spl   # dB SPL peak at the primary (v1 stores dB re speech rms)
        meta["impulse_onsets_s"] = [start / SR + o for o in impulse_onsets_s]

    # --- front end (M7): mic gain, 60 Hz HPF, saturation, rails, self-noise ---
    gains = rng.uniform(-p["mic_gain_db"], p["mic_gain_db"], 2) if p["front_end"] else np.zeros(2)
    acoustic = s2 + noise2 + (imp2 if imp2 is not None else 0.0)
    if p["front_end"]:
        lin = calib.front_end_linear(acoustic, gains)
        clean = calib.front_end_linear(s_p[None], gains[:1])[0]
        out, fe = calib.front_end_nonlinear(rng, lin, p["self_noise_db"])
    else:
        lin = acoustic.astype(np.float32); clean = s_p.copy()
        out, fe = lin.copy(), {"clip_frac": 0.0, "saturated": False}
    if mode == "mono":
        out[1] = out[0]
    meta["mic_gain_db"] = [float(g) for g in gains]
    meta["clip_frac"] = fe["clip_frac"]; meta["clipped"] = fe["clip_frac"] > 0; meta["overloaded"] = fe["saturated"]

    # SNR is an output here: speech-active boom speech against everything else on the primary, before the nonlinearity
    ps = speech_active_power(clean)
    pre = lin[0] - clean
    meta["snr_db"] = np.inf if meta["clean_bucket"] else float(10 * np.log10(ps / ((pre ** 2).mean() + 1e-20)))
    if norm_gain is not None:
        out = out * np.float32(norm_gain); clean = clean * np.float32(norm_gain); meta["norm_gain"] = float(norm_gain)
    resid = out[0] - clean
    meta["snr_achieved_db"] = float(10 * np.log10(speech_active_power(clean) / ((resid ** 2).mean() + 1e-20)))
    meta["noise_class"] = "clean" if meta["clean_bucket"] else scene["name"]
    return out.astype(np.float32), clean.astype(np.float32), meta
