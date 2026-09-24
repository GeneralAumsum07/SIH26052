"""Known-answer tests for the headline metrics and the n=617 nominal filter.

The wrappers in vaani/metrics.py are thin, but a changed default (extended STOI, narrowband PESQ, a
different rate) would move every reported number silently. These pin the wrappers to direct pystoi/pesq
calls on deterministic signals, and pin the filter behind the headline table to its committed n.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pesq import pesq as _pesq
from pystoi import stoi as _stoi

from vaani import metrics

SR = 16000
R7_EVAL_R2 = Path("results_r2/r7/r7_e256_wr64_cascade_eval_r2.csv")


def _voiced(seconds=3.0):
    """Deterministic voiced-speech stand-in: gliding 120 Hz harmonic stack, 3 Hz syllable envelope."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 30))
    env = np.clip(np.sin(2 * np.pi * 3 * t), 0, None) ** 2
    return 0.3 * x * env / np.abs(x).max()


def _noisy(clean, scale):
    return clean + scale * np.random.default_rng(0).standard_normal(len(clean))


def test_identical_signals_hit_the_ceilings():
    c = _voiced()
    assert metrics.stoi(c, c) == pytest.approx(1.0, abs=1e-6)
    assert metrics.pesq_wb(c, c) == pytest.approx(4.644, abs=0.01)   # P.862.2 WB ceiling (MOS-LQO mapping)


@pytest.mark.parametrize("scale", [3e-4, 5e-2])
def test_wrappers_equal_direct_library_calls(scale):
    c = _voiced(); y = _noisy(c, scale)
    assert metrics.stoi(c, y) == _stoi(c, y, SR, extended=False)      # classic STOI, not ESTOI
    assert metrics.pesq_wb(c, y) == float(_pesq(SR, c, y, "wb"))       # wideband mode at 16 kHz


def test_known_values_on_the_fixed_signal():
    # measured with pesq 0.0.4 / pystoi 0.4.1; a library or wrapper change that moves these is a reporting change
    c = _voiced()
    assert metrics.stoi(c, _noisy(c, 5e-2)) == pytest.approx(0.88715, abs=1e-4)
    assert metrics.pesq_wb(c, _noisy(c, 3e-4)) == pytest.approx(2.378, abs=0.01)


def test_metrics_fall_monotonically_with_noise():
    c = _voiced()
    stoi = [metrics.stoi(c, _noisy(c, s)) for s in (1e-3, 1e-2, 5e-2)]
    pesq = [metrics.pesq_wb(c, _noisy(c, s)) for s in (1e-4, 3e-4, 1e-3)]
    assert stoi[0] > stoi[1] > stoi[2]
    assert pesq[0] > pesq[1] > pesq[2]


def test_pesq_failure_is_nan_not_a_crash():
    assert np.isnan(metrics.pesq_wb(np.zeros(SR), np.zeros(SR)))     # no utterance: NaN, dropped downstream


def test_nominal_filter_selects_the_headline_617():
    from scripts import optimization_report, r6_table
    assert R7_EVAL_R2.exists(), f"missing committed eval CSV {R7_EVAL_R2}"
    df = pd.read_csv(R7_EVAL_R2, dtype={"id": str})
    env = r6_table.envelope(df)
    assert len(df) == 2280 and len(env) == 617
    assert optimization_report.nominal(df).index.equals(env.index)   # both copies of the filter agree
    # the README headline (SNR_out / STOI / PESQ-WB) is the mean over exactly these rows
    assert env.snr_out.mean() == pytest.approx(14.864, abs=5e-4)
    assert env.stoi.mean() == pytest.approx(0.917, abs=5e-4)
    assert env.pesq_wb.mean() == pytest.approx(2.462, abs=5e-4)
