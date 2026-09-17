import numpy as np
from vaani.data import impulses


def test_all_kinds_peak_at_one_and_have_onsets():
    rng = np.random.default_rng(0)
    for kind in ("burst", "click_train", "gated_noise"):
        x, meta = impulses.generate(rng, kind=kind)
        assert x.dtype == np.float32
        assert abs(np.abs(x).max() - 1.0) < 1e-5
        assert meta["kind"] == kind and len(meta["onsets_s"]) >= 1
        assert 0.2 * 16000 <= len(x) <= 2.0 * 16000


def test_burst_is_actually_impulsive():
    # 1 s after onset even the slowest burst (tau=0.25 s) has decayed >30 dB
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x, meta = impulses.generate(rng, kind="burst")
        on = int(meta["onsets_s"][0] * 16000)
        x = np.pad(x, (0, max(0, on + 16160 - len(x))))  # short bursts end in silence
        e0 = (x[on:on + 160] ** 2).mean()
        e1 = (x[on + 16000:on + 16160] ** 2).mean() + 1e-12
        assert 10 * np.log10(e0 / e1) > 20, seed
