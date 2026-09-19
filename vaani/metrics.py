"""Metric definitions. SNR here is error-based against the clean reference
(distortion counts as noise); SI-SDR is scale-invariant. They are different
numbers and are reported separately - never call SI-SDR 'output SNR'.
PESQ: wideband P.862.2 at 16 kHz. ITU has withdrawn P.862 in favour of P.863;
we report it because the brief requests it.
"""
import numpy as np
from pesq import pesq as _pesq
from pystoi import stoi as _stoi

SR = 16000


def snr_db(clean, est):
    return float(10 * np.log10((clean ** 2).sum() / (((est - clean) ** 2).sum() + 1e-12) + 1e-12))


def si_sdr_db(clean, est):
    a = (est * clean).sum() / ((clean ** 2).sum() + 1e-12); s = a * clean
    return float(10 * np.log10((s ** 2).sum() / (((est - s) ** 2).sum() + 1e-12) + 1e-12))


def stoi(clean, est):
    return float(_stoi(clean, est, SR, extended=False))


def pesq_wb(clean, est):
    try: return float(_pesq(SR, clean, est, "wb"))
    except Exception: return float("nan")  # NoUtterancesError etc.: never crash a run


def _envelope_db(x, frame=320):
    f = x[: len(x) // frame * frame].reshape(-1, frame)
    return 10 * np.log10((f ** 2).mean(axis=1) + 1e-10)


def recovery_time_s(est_burst, est_twin, burst_onset_s, thresh_db=3.0, hold_s=0.2, frame=320):
    """Time after the burst until the speech envelope of the burst run stays
    within thresh_db of the no-burst twin for hold_s. inf if never (NaN is reserved for "no burst")."""
    d = np.abs(_envelope_db(est_burst, frame) - _envelope_db(est_twin, frame))
    k0 = int(burst_onset_s * SR / frame); hold = int(hold_s * SR / frame)
    ok = d < thresh_db
    for k in range(k0, len(ok) - hold):
        if ok[k:k + hold].all():
            return (k - k0) * frame / SR
    return float("inf")
