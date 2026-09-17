import numpy as np
from vaani.data import rirs


def test_pair_set_shapes_and_leakage():
    rng = np.random.default_rng(0)
    s = rirs.simulate_pair_set(rng, n_noise=2)
    assert s["speech"].shape[0] == 2 and s["noise"].shape[:2] == (2, 2)
    # near-mouth geometry: speech RIR energy must be much higher at primary
    e = (s["speech"] ** 2).sum(axis=1)
    assert 10 * np.log10(e[0] / e[1]) > 6
    # far-field noise: roughly equal at both mics
    en = (s["noise"][0] ** 2).sum(axis=1)
    assert abs(10 * np.log10(en[0] / en[1])) < 4


def test_bank_roundtrip(tmp_path):
    p = tmp_path / "bank.npz"
    rirs.build_bank(p, n=3, seed=1)
    b = rirs.RirBank(p)
    s = b.sample(np.random.default_rng(0))
    assert s["speech"].shape[0] == 2
