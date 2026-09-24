"""SPL calibration chain and front-end model for mixer v2 (plan 11.5 M1, M7, M11).

Everything in mixer v2 is built in dB SPL at each mic and mapped to float samples through the ICS-43434
transfer: -26 dBFS (sine-referenced) at 94 dB SPL, so a float signal's rms in dB is SPL - 123.01 and float
1.0 instantaneous is 28.3 Pa (123.01 dB peak). Clipping, the mic HPF and self-noise then follow from the
physics instead of being tuned. Sources: ICS-43434 DS-000069, ITU-T P.64, Alghamdi et al. 2018 (JASA EL523).
"""
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfilt

SR = 16000
P_REF = 20e-6
SPL_TO_FLOAT_RMS_DB = 123.01          # float rms dB = SPL - 123.01 (94 dB SPL sine -> -26 dBFS sine-ref = -29.01 dB rms)
ICS43434_SENS_DBFS = -26.0            # at 94 dB SPL, 1 kHz sine (datasheet +-1 dB)
AOP_DB_SPL = 120.0                    # acoustic overload point, 10 % THD
HARD_CLIP_PEAK_DB_SPL = SPL_TO_FLOAT_RMS_DB   # float |x| = 1.0: the rails
SELF_NOISE_FLOAT_DB = -93.0           # EIN 30 dBA -> about -93 dB float rms
HPF_HZ = 60.0                         # datasheet low-frequency corner
MIC_GAIN_TOL_DB = 1.0                 # sensitivity tolerance per unit

SPEECH_NORMAL_BOOM_DB_SPL = 89.3      # P.64: -4.7 dBPa at the MRP (25 mm), normal effort
# TBD-verify: ANSI S3.5-1997 overall levels per vocal effort; +6/+12/+20 are the commonly quoted offsets, unverified
EFFORT_OFFSETS_DB = {"normal": 0.0, "raised": 6.0, "loud": 12.0, "shout": 20.0}
# Alghamdi 2018 Table 1 (80 dB SPL noise, 54 talkers): alpha ratio -12.2 -> -7.7 dB. Applied to every non-normal class;
# a per-class value is not in the source (TBD).
LOMBARD_ALPHA_DB = 4.5
LOMBARD_F0_ST = 1.9                   # TBD: no cheap artefact-free F0 shift implemented; recorded, not applied


def spl_to_float_rms_db(spl_db):
    return np.asarray(spl_db, float) - SPL_TO_FLOAT_RMS_DB


def float_rms_db_to_spl(db):
    return np.asarray(db, float) + SPL_TO_FLOAT_RMS_DB


def spl_to_float_rms(spl_db) -> float:
    return float(10 ** ((float(spl_db) - SPL_TO_FLOAT_RMS_DB) / 20))


def pascal_to_float(p):
    return np.asarray(p) / (P_REF * 10 ** (SPL_TO_FLOAT_RMS_DB / 20))


def float_to_pascal(x):
    return np.asarray(x) * (P_REF * 10 ** (SPL_TO_FLOAT_RMS_DB / 20))


def rms_db(x) -> float:
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-30))


def active_rms_db(x, frame: int = 320, thresh_db: float = -30.0) -> float:
    """Speech-active rms, same rule as mixer.speech_active_power, so SPL means the talker's active level."""
    f = np.asarray(x, np.float64)[: len(x) // frame * frame].reshape(-1, frame)
    e = (f ** 2).mean(axis=1) + 1e-30
    keep = e > e.max() * 10 ** (thresh_db / 10)
    return float(10 * np.log10(e[keep].mean() if keep.any() else e.mean()))


def a_weight_db(f):
    """IEC 61672 A-weighting in dB (0 dB at 1 kHz)."""
    f2 = np.maximum(np.asarray(f, float), 1e-3) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    return 20 * np.log10(ra) + 2.0


def a_weighted_rms_db(x, sr: int = SR) -> float:
    """Leq in dBA terms (float domain): the scene beds are specified in dBA, and LF-heavy noise reads lower weighted."""
    X = np.fft.rfft(np.asarray(x, np.float64))
    w = 10 ** (a_weight_db(np.fft.rfftfreq(len(x), 1 / sr)) / 20)
    p = (np.abs(X * w) ** 2).sum() * 2 / len(x) ** 2
    return float(10 * np.log10(p + 1e-30))


def scale_to_spl(x, spl_db: float, weighting: str = "active") -> np.ndarray:
    """Scale x so its level is spl_db at the mic. 'active' = speech-active rms, 'rms' = plain, 'A' = A-weighted Leq."""
    lvl = {"active": active_rms_db, "rms": rms_db, "A": a_weighted_rms_db}[weighting](x)
    return (np.asarray(x, np.float32) * np.float32(10 ** ((spl_to_float_rms_db(spl_db) - lvl) / 20))).astype(np.float32)


def effort_class(offset_db: float) -> str:
    """Nearest effort class for a talker level offset re normal (midpoints between the class offsets)."""
    if offset_db < 3.0: return "normal"
    if offset_db < 9.0: return "raised"
    if offset_db < 16.0: return "loud"
    return "shout"


def lombard_tilt(x, alpha_db: float = LOMBARD_ALPHA_DB, sr: int = SR) -> np.ndarray:
    """Flatten the spectral tilt: +alpha_db above 1 kHz against 50-1000 Hz via a smooth shelf, level restored after.
    The alpha ratio is the 1-5 kHz vs 50 Hz-1 kHz energy ratio, so a shelf at 1 kHz moves it by about alpha_db."""
    if alpha_db == 0:
        return np.asarray(x, np.float32)
    n = len(x); X = np.fft.rfft(np.asarray(x, np.float64)); f = np.fft.rfftfreq(n, 1 / sr)
    s = (f / 1000.0) ** 4 / (1 + (f / 1000.0) ** 4)         # 0 below ~700 Hz, 1 above ~1.4 kHz
    y = np.fft.irfft(X * 10 ** (alpha_db * s / 20), n)
    return (y * np.sqrt(np.mean(np.square(x, dtype=np.float64)) / (np.mean(y ** 2) + 1e-30))).astype(np.float32)


def softsat(x, knee: float) -> np.ndarray:
    """Linear below knee, tanh above with its asymptote at the rails (float 1.0 = 123 dB peak)."""
    x = np.asarray(x, np.float32); a = np.abs(x)
    return np.where(a > knee, np.sign(x) * (knee + (1 - knee) * np.tanh((a - knee) / (1 - knee))), x).astype(np.float32)


def _thd(a: float, knee: float, n: int = 4096) -> float:
    t = np.arange(n) / n
    Y = np.abs(np.fft.rfft(softsat(a * np.sin(2 * np.pi * 8 * t), knee).astype(np.float64))) ** 2
    return float(np.sqrt(Y[16::8].sum() / Y[8]))


THD_POINT = (105.0, 0.002)   # datasheet: 0.2 % THD at 105 dB SPL


@lru_cache(maxsize=4)
def soft_knee(spl_db: float = THD_POINT[0], thd: float = THD_POINT[1]) -> float:
    """Knee (float) such that a sine at spl_db reaches the datasheet THD; bisection, THD falls as the knee rises.
    Fitted at the 105 dB point, not the AOP: with the asymptote at the rails even a zero knee gives only ~6.7 % THD
    at 120 dB, so the datasheet's 10 % at the AOP is unreachable by this curve (the model under-distorts there)."""
    a = np.sqrt(2) * spl_to_float_rms(spl_db)
    lo, hi = 0.0, 0.999
    for _ in range(50):
        mid = (lo + hi) / 2
        if _thd(a, mid) > thd: lo = mid
        else: hi = mid
    return float((lo + hi) / 2)


def hpf(x, hz: float = HPF_HZ, sr: int = SR) -> np.ndarray:
    sos = butter(2, hz, "highpass", fs=sr, output="sos")
    return sosfilt(sos, x, axis=-1).astype(np.float32)


def front_end_linear(x2, gains_db, sr: int = SR) -> np.ndarray:
    """Per-mic sensitivity offset then the 60 Hz HPF: the linear half, also applied to the clean target."""
    g = (10 ** (np.asarray(gains_db, float) / 20)).astype(np.float32)
    return hpf(np.asarray(x2, np.float32) * (g[:, None] if np.ndim(x2) == 2 else g), sr=sr)


def front_end_nonlinear(rng, x2, self_noise_db: float = SELF_NOISE_FLOAT_DB, saturate: bool = True) -> tuple[np.ndarray, dict]:
    """Soft saturation from the AOP knee, hard clip at the rails (123 dB peak), then self-noise (white; A-shaping TBD).
    clip_frac counts input samples past the rails, i.e. pressure above 123 dB peak."""
    y = np.asarray(x2, np.float32)
    meta = {"clip_frac": 0.0, "saturated": False}
    if saturate:
        knee = soft_knee()
        a = np.abs(y)
        meta["clip_frac"] = float((a >= 1.0).mean())
        if a.max(initial=0.0) > knee:
            y = np.clip(softsat(y, knee), -1.0, 1.0); meta["saturated"] = True
    if self_noise_db is not None:
        y = y + (rng.standard_normal(y.shape) * 10 ** (self_noise_db / 20)).astype(np.float32)
    return y.astype(np.float32), meta
