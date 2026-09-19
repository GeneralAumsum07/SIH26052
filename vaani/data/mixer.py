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

SR = 16000


@dataclass
class MixConfig:
    snr_range: tuple[float, float] = (-10.0, 15.0)
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
    return np.tile(x, reps)[:n]


def _frac_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = len(x); k = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * k * delay_samples), n).astype(np.float32)


def _tilt(x: np.ndarray, db: float) -> np.ndarray:
    """1st-order spectral tilt to emulate mic response mismatch."""
    a = np.clip(db / 40.0, -0.3, 0.3)
    return lfilter([1.0, a], [1.0], x).astype(np.float32)


def _conv2(x: np.ndarray, h2: np.ndarray, n: int) -> np.ndarray:
    return np.stack([fftconvolve(x, h2[m])[:n] for m in range(2)]).astype(np.float32)


def mix(rng, speech, noises, impulse, impulse_onsets_s, bank, cfg: MixConfig, norm_gain: float | None = None):
    """norm_gain: reuse another clip's final peak scaling (the burst clip's, when rendering its twin)
    so the pair differs only by the impulse and not by a level step."""
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
            # independent short random filters per channel: different arrival paths
            for m in range(2):
                taps = rng.normal(0, 1, 3); taps[0] = 1.0; taps[1:] *= 0.3
                noise2[m] += lfilter(taps, [1.0], nz).astype(np.float32)

    if rng.random() < cfg.p_clean:
        meta["clean_bucket"] = True; meta["snr_db"] = meta["snr_achieved_db"] = np.inf; meta["noise_class"] = "clean"
        return s2.astype(np.float32), clean, meta

    # --- scale noise to target SNR on the primary, speech-active region ---
    snr = float(rng.uniform(*cfg.snr_range))
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
