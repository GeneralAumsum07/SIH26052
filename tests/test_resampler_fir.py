"""Low-delay plan Task 0 / Section 2.4: the committed R1/R2 coefficient files load with their hashes, R1 keeps R0's
magnitude, R2 meets every recorded requirement, and each streams exactly like offline filtering."""
import json

import numpy as np
import pytest

from vaani import resampler as rs
from vaani.audio_contract import RESAMPLER_R1, RESAMPLER_R2


def _resp(h, f):
    n = np.arange(len(h))
    return np.exp(-2j * np.pi * np.outer(f, n) / rs.FS_HI) @ h


def test_files_load_with_their_hashes():
    for rid in (RESAMPLER_R1, RESAMPLER_R2):
        pair = rs.load(rid)
        assert len(pair.h) == 193 and pair.sha256 == rs.coef_sha256(pair.h)


def test_tampered_coefficients_are_refused(tmp_path):
    j = json.loads((rs.COEF_DIR / f"{RESAMPLER_R1}.json").read_text())
    j["coefficients"][10] += 1e-9
    (tmp_path / f"{RESAMPLER_R1}.json").write_text(json.dumps(j))
    with pytest.raises(ValueError, match="sha256"):
        rs.load(RESAMPLER_R1, tmp_path)


def test_r1_is_r0s_magnitude_at_minimum_phase_delay():
    h0, h1 = rs.r0_coefficients(), rs.load(RESAMPLER_R1).h
    f = np.linspace(0, 7000, 701)
    d = 20 * np.log10(np.abs(_resp(h1, f))) - 20 * np.log10(np.abs(_resp(h0, f)))
    assert np.abs(d).max() < 1e-4
    assert rs.load(RESAMPLER_R1).pair_delay_ms == pytest.approx(1 / 3, abs=1e-9)


def test_r2_meets_the_task0_requirements():
    j = json.loads((rs.COEF_DIR / f"{RESAMPLER_R2}.json").read_text())
    assert all(j["design"]["checks"].values()), j["design"]["checks"]
    m, req = j["measurements"], j["design"]["requirements"]
    assert m["passband_dev_vs_r0_db"] <= req["passband_dev_db"]
    assert m["transition_7_9k_max_db"] <= 0.0
    assert m["reject_8_9k_db"] <= -60 and m["reject_above_9k_db"] <= -80
    lo, hi = m["group_delay_0p1_6k_ms"]
    assert abs(lo - 0.75) <= 0.02 and abs(hi - 0.75) <= 0.02
    assert m["identity_snr_db"] >= 30
    # an independent spot check of the stopband from the coefficients themselves
    h = rs.load(RESAMPLER_R2).h
    assert 20 * np.log10(np.abs(_resp(h, np.linspace(9000, 24000, 1501))).max()) <= -80


@pytest.mark.parametrize("rid", rs.IDS)
def test_streaming_equals_offline(rid):
    pair = rs.load(rid)
    g = np.random.default_rng(0)
    x = (g.standard_normal((1, 2400)) * 0.1).astype(np.float32)
    dec = pair.decimator(1)
    y = np.concatenate([dec(x[:, a:a + 240]) for a in range(0, 2400, 240)], 1)
    ref = np.convolve(x[0].astype(np.float64), pair.h)[:2400:3]
    np.testing.assert_allclose(y[0], ref, atol=1e-6)
