"""Parallel RIR bank generation must be byte-identical to the sequential path: the anchor recipes name banks by
path, so a bank that differs in content would silently change the training rooms."""
import numpy as np, pytest

from vaani.data import rirs


def _load(p):
    z = np.load(p); return {k: z[k] for k in ("speech", "noise", "rt60", "armoured")}


def test_split_draw_and_simulate_matches_the_composed_call():
    a = rirs.simulate_pair_set(np.random.default_rng(3), n_noise=2, max_len=1600)
    p = rirs.draw_room_params(np.random.default_rng(3), n_noise=2)
    b = rirs.simulate_from_params(p, max_len=1600)
    assert np.array_equal(a["speech"], b["speech"]) and np.array_equal(a["noise"], b["noise"]) and a["rt60"] == b["rt60"]


def test_parallel_bank_is_bit_identical_to_sequential(tmp_path):
    rirs.build_bank(tmp_path / "s.npz", n=6, seed=0, max_len=1600, workers=1)
    rirs.build_bank(tmp_path / "p.npz", n=6, seed=0, max_len=1600, workers=3)
    s, p = _load(tmp_path / "s.npz"), _load(tmp_path / "p.npz")
    for k in s: assert np.array_equal(s[k], p[k]), k


@pytest.mark.xfail(strict=True, reason="pyroomacoustics ray tracing is nondeterministic even within one process "
                   "(measured 2026-09-21: same params twice, max |diff| 1.02); armoured bank entries were never "
                   "reproducible. Ship banks by hash instead of regenerating them.")
def test_armoured_ray_tracing_is_deterministic(tmp_path):
    rirs.build_bank(tmp_path / "a.npz", n=2, seed=1, armoured_frac=1.0, max_len=1600, workers=1)
    rirs.build_bank(tmp_path / "b.npz", n=2, seed=1, armoured_frac=1.0, max_len=1600, workers=2)
    a, b = _load(tmp_path / "a.npz"), _load(tmp_path / "b.npz")
    assert np.array_equal(a["speech"], b["speech"]) and np.array_equal(a["noise"], b["noise"])
