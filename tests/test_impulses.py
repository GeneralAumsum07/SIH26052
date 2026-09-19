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


def test_detect_onsets_finds_corpus_style_burst():
    # a recorded impulse file has no metadata: the onset must come from the waveform
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32000).astype(np.float32) * 0.01  # -40 dB floor
    on = 11200  # 0.7 s
    x[on:on + 800] += np.exp(-np.arange(800) / 200).astype(np.float32) * rng.standard_normal(800).astype(np.float32)
    got = impulses.detect_onsets(x, 16000)
    assert len(got) == 1 and abs(got[0] - 0.7) < 0.02


def test_detect_onsets_matches_generator_for_click_train():
    for seed in range(5):
        x, meta = impulses.generate(np.random.default_rng(seed), kind="click_train")
        got = impulses.detect_onsets(x, 16000)
        assert abs(got[0] - meta["onsets_s"][0]) < 0.02, seed
        assert len(got) >= 3, seed  # separate clicks, not one smeared event


def test_detect_onsets_falls_back_to_peak_when_nothing_stands_out():
    x = np.ones(16000, np.float32) * 0.1  # flat: no onset -> loudest sample, never an empty list
    assert impulses.detect_onsets(x, 16000) == [0.0]
