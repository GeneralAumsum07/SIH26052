"""physics="v2" in vaani/data/blast.py (plan M5): reflection polarity, absorption, Kinney-Graham, burst timing, v1 intact."""
import numpy as np
import pytest

from vaani.data import blast, impulses

SR = 16000


def test_iso9613_matches_the_standard_table():
    a = blast.iso9613_alpha_db_per_m([1000.0, 4000.0]) * 1000   # dB/km at 20 C, 50 %
    assert a[0] == pytest.approx(4.66, abs=0.05) and a[1] == pytest.approx(29.7, abs=0.3)


def test_kinney_graham_levels():
    assert blast.kinney_graham_spl(1.0, 100.0) == pytest.approx(154.5, abs=0.5)
    assert blast.kinney_graham_spl(15.0, 3000.0) == pytest.approx(132.8, abs=0.5)
    assert blast.kinney_graham_spl(1.0, 100.0) > blast.kinney_graham_spl(1.0, 300.0)


def test_ground_reflection_keeps_the_sign():
    y = np.zeros(4000); y[10] = 1.0
    out, m = blast.ground_reflection(y, SR, r_m=1.0, h_src=1.5, h_mic=1.5, R=0.8)
    d = int(round(m["reflection_delay_ms"] * 1e-3 * SR))
    assert d > 0 and out[10 + d] > 0 and m["reflection_gain"] == pytest.approx(0.8 / np.hypot(1, 3), rel=1e-6)


def test_v2_shot_shows_a_positive_echo_where_v1_had_a_negative_one():
    # non-ballistic muzzle blast at 1 m, standing heights: the echo lands ~6 ms after the direct pulse
    kw = dict(kind="small_arms", distance_m=1.0, heights_m=(1.5, 1.5), ground_r=0.9)
    for s in range(40):
        y, m = blast.blast_v2(np.random.default_rng(s), SR, **kw)
        if not m["ballistic"]:
            break
    d = int(round(m["reflection_delay_ms"] * 1e-3 * SR))
    win = y[d - 2:d + 6]
    assert win.max() > 0.2 * np.abs(y).max() and win.max() > -win.min()


def test_absorption_dulls_a_distant_shot():
    def hf_fraction(r):
        y, _ = blast.blast_v2(np.random.default_rng(3), SR, kind="artillery", distance_m=r, charge_kg=1.0,
                              heights_m=(1.0, 1.0), ground_r=0.5)
        P = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / SR)
        return P[f > 4000].sum() / P.sum()
    assert hf_fraction(3000.0) < 0.1 * hf_fraction(30.0)


def test_small_arms_level_follows_spreading_from_1m():
    y, m = blast.blast_v2(np.random.default_rng(0), SR, kind="small_arms", peak_spl_1m=160.0, distance_m=10.0)
    assert m["direct_spl_db"] == pytest.approx(140.0) and m["source_spl_1m"] == 160.0
    assert abs(m["peak_spl_db"] - 20 * np.log10(np.abs(y).max() / blast.P_REF)) < 1e-9


def test_v2_draws_scene_parameters_in_range():
    for s in range(30):
        _, m = blast.blast_v2(np.random.default_rng(s), SR, kind="small_arms")
        assert 1.0 <= m["distance_m"] <= 300.0 and 150.0 <= m["source_spl_1m"] <= 160.0
        _, m = blast.blast_v2(np.random.default_rng(s), SR, kind="artillery")
        assert 30.0 <= m["distance_m"] <= 3000.0 and 0.1 <= m["charge_kg"] <= 20.0


def test_burst_timing_follows_the_cyclic_rate():
    y, m = blast.burst(np.random.default_rng(0), SR, n_rounds=10, rpm=660.0, distance_m=20.0)
    on = np.asarray(m["onsets_s"])
    assert len(on) == 10 and on[0] == 0.0
    gaps = np.diff(on)
    assert np.all(np.abs(gaps / (60 / 660) - 1) <= 0.2 + 1e-9) and abs(gaps.mean() / (60 / 660) - 1) < 0.05
    # every round is present: a shot-level peak within 60 ms of each onset (the muzzle blast trails a ballistic N-wave by 36 ms at 20 m)
    pk = np.abs(y).max()
    for t in on:
        a = int(round(t * SR)); assert np.abs(y[a:a + int(0.06 * SR)]).max() > 0.3 * pk


def test_burst_draws_rounds_and_rate_in_range():
    for s in range(30):
        _, m = blast.burst(np.random.default_rng(s), SR, distance_m=50.0)
        assert 3 <= m["n_rounds"] <= 30 and 650.0 <= m["rpm"] <= 700.0 and len(m["onsets_s"]) == m["n_rounds"]


def test_generate_v2_burst_meta_and_onsets():
    x, m = impulses.generate(np.random.default_rng(1), kind="blast", blast_kind="burst", physics="v2",
                             n_rounds=5, rpm=700.0, distance_m=5.0)
    assert m["physics"] == "v2" and m["blast_kind"] == "small_arms_burst" and m["n_rounds"] == 5
    assert len(m["onsets_s"]) == 5 and abs(np.abs(x).max() - 1.0) < 1e-5 and len(x) <= impulses.V2_MAX_S * SR
    assert np.allclose(np.diff(m["onsets_s"]).mean(), 60 / 700, rtol=0.05)


def test_v1_is_the_default_and_rejects_scene_parameters():
    for s in range(5):
        a, ma = blast.blast(np.random.default_rng(s))
        b, mb = blast.blast(np.random.default_rng(s), physics="v1")
        assert np.array_equal(a, b) and ma == mb and "physics" not in ma
        a, ma = impulses.generate(np.random.default_rng(s), kind="blast")
        b, mb = impulses.generate(np.random.default_rng(s), kind="blast", physics="v1")
        assert np.array_equal(a, b) and ma == mb
    with pytest.raises(ValueError):
        blast.blast(np.random.default_rng(0), distance_m=5.0)
    with pytest.raises(ValueError):
        impulses.generate(np.random.default_rng(0), kind="burst", physics="v2")
