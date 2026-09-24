import pytest
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


def test_simulate_pair_set_never_raises_across_seeds():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        s = rirs.simulate_pair_set(rng, n_noise=2)
        assert s["speech"].shape == (2, rirs.MAX_LEN)
        assert s["speech"].dtype == np.float32
        for h in s["noise"]:
            assert h.shape == (2, rirs.MAX_LEN)
            assert h.dtype == np.float32


def test_bank_roundtrip(tmp_path):
    p = tmp_path / "bank.npz"
    rirs.build_bank(p, n=3, seed=1)
    b = rirs.RirBank(p)
    s = b.sample(np.random.default_rng(0))
    assert s["speech"].shape[0] == 2
    assert s["speech"].dtype == np.float32
    assert s["noise"].dtype == np.float32
    assert isinstance(s["rt60"], float)


def test_bank_is_memmapped_and_reloads_from_npy(tmp_path):
    p = tmp_path / "b.npz"
    rirs.build_bank(p, n=2, seed=0)
    b = rirs.RirBank(p)
    assert isinstance(b.speech, np.memmap) and (tmp_path / "b.speech.npy").exists()
    b2 = rirs.RirBank(p)
    assert np.array_equal(b.speech, b2.speech) and len(b2) == 2


def _t60(h, sr=16000):
    e = np.cumsum(h[::-1] ** 2)[::-1]; e = 10 * np.log10(e / e[0] + 1e-12)
    return float(np.argmax(e < -60) / sr) if (e < -60).any() else float("inf")


def test_armoured_pair_set_is_small_box_with_long_tail():
    rng = np.random.default_rng(1)
    s = rirs.simulate_pair_set(rng, n_noise=1, armoured=True, max_len=16000)
    assert s["armoured"] and s["speech"].shape == (2, 16000) and s["room_dims"].max() <= 2.5
    # the ray-traced tail must outlast what a 12th-order image method gives a normal room (measured ~0.26 s)
    assert _t60(s["noise"][0, 0]) > 0.4
    # geometry still holds: speech dominates the primary, noise is roughly equal at both mics
    e = (s["speech"] ** 2).sum(axis=1)
    assert 10 * np.log10(e[0] / e[1]) > 6


def test_bank_armoured_share_is_exact_and_ignored_by_loader(tmp_path):
    p = tmp_path / "b.npz"
    rirs.build_bank(p, n=5, seed=0, n_noise=1, armoured_frac=0.4, max_len=8000)
    z = np.load(p)
    assert z["armoured"].sum() == 2 and z["speech"].shape == (5, 2, 8000)
    b = rirs.RirBank(p)
    assert len(b) == 5 and b.sample(np.random.default_rng(0))["speech"].shape == (2, 8000)


def test_m6_defaults_unchanged_and_eval_namespace_disjoint(tmp_path):
    # default stream and file layout are the legacy ones
    assert rirs.bank_rng(3).random() == np.random.default_rng(3).random()
    rirs.build_bank(tmp_path / "a.npz", n=2, seed=0, n_noise=1, max_len=4000, workers=1)
    assert set(np.load(tmp_path / "a.npz").files) == {"speech", "noise", "rt60", "armoured"}
    # an eval bank with the same seed shares no room with the training stream
    tr, ev = rirs.bank_rng(0), rirs.bank_rng(0, "eval")
    dims_tr = {tuple(np.round(rirs.draw_room_params(tr)["dims"], 9)) for _ in range(200)}
    dims_ev = {tuple(np.round(rirs.draw_room_params(ev)["dims"], 9)) for _ in range(200)}
    assert not dims_tr & dims_ev
    with pytest.raises(ValueError):
        rirs.bank_rng(0, "test")


def test_m6_receiver_radius_below_half_spacing_scales_rays(monkeypatch):
    assert rirs.M6_RECEIVER_RADIUS < rirs.MIC_SPACING / 2
    seen = {}
    orig = rirs.pra.ShoeBox.set_ray_tracing

    def spy(self, n_rays=None, receiver_radius=0.5, **kw):
        seen.update(n_rays=n_rays, receiver_radius=receiver_radius)
        return orig(self, n_rays=2000, receiver_radius=receiver_radius, **kw)   # cheap run, the request is what we check
    monkeypatch.setattr(rirs.pra.ShoeBox, "set_ray_tracing", spy)
    rirs.simulate_pair_set(np.random.default_rng(1), n_noise=1, armoured=True, max_len=4000,
                           receiver_radius=rirs.M6_RECEIVER_RADIUS)
    assert seen["receiver_radius"] == rirs.M6_RECEIVER_RADIUS and seen["n_rays"] == 360000
    rirs.simulate_pair_set(np.random.default_rng(1), n_noise=1, armoured=True, max_len=4000)
    assert seen["receiver_radius"] == 0.3 and seen["n_rays"] == rirs.ARMOURED_RAYS
