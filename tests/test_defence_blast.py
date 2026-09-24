"""Ballistic N-wave in vaani/data/blast.py (review plan 3.10): the old nw[lead:] slice always removed it."""
import numpy as np
import pytest

from vaani.data import blast, impulses

SR = 16000


def _ballistic_seeds(k=5):
    out = [s for s in range(200) if blast.blast(np.random.default_rng(s), kind="small_arms", distance=0.0)[1]["ballistic"]]
    assert len(out) >= k
    return out[:k]


def test_n_wave_survives_into_the_output(monkeypatch):
    for seed in _ballistic_seeds():
        x, m = blast.blast(np.random.default_rng(seed), kind="small_arms", distance=0.0)
        monkeypatch.setattr(blast, "_n_wave", lambda L, n, sr: np.zeros(n))
        x0, m0 = blast.blast(np.random.default_rng(seed), kind="small_arms", distance=0.0)
        monkeypatch.undo()
        assert m == m0   # same rng stream: the N-wave is the only difference
        lead = int(m["ballistic_lead_ms"] * 1e-3 * SR)
        # before the fix x == x0 exactly; now the N-wave sits in the lead window ahead of the muzzle blast
        assert np.abs(x[: max(1, lead - 1)] - x0[: max(1, lead - 1)]).max() > 0.1, seed
        assert np.abs(x0[: max(1, lead - 2)]).max() < 0.05, seed   # muzzle blast itself arrives after the lead


def test_n_wave_precedes_the_muzzle_blast_peak():
    for seed in _ballistic_seeds():
        x, m = blast.blast(np.random.default_rng(seed), kind="small_arms", distance=0.0)
        lead = m["ballistic_lead_ms"] * 1e-3 * SR
        first = int(np.argmax(np.abs(x) > 0.1))
        assert first < lead, seed


def test_non_ballistic_meta_and_artillery_unchanged_shape():
    x, m = blast.blast(np.random.default_rng(0), kind="artillery")
    assert m["ballistic"] is False and m["ballistic_lead_ms"] is None
    assert abs(np.abs(x).max() - 1.0) < 1e-5


@pytest.mark.parametrize("bk", ["small_arms", "artillery"])
def test_generate_can_pin_the_blast_sub_kind(bk):
    for s in range(5):
        x, m = impulses.generate(np.random.default_rng(s), kind="blast", blast_kind=bk)
        assert m["blast_kind"] == bk and 0.2 * SR <= len(x) <= 2.0 * SR


def test_default_generate_draw_is_unchanged_by_the_new_kwarg():
    for s in range(5):
        a = impulses.generate(np.random.default_rng(s))
        b = impulses.generate(np.random.default_rng(s), blast_kind=None)
        assert np.array_equal(a[0], b[0]) and a[1] == b[1]
