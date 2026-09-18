import numpy as np
from vaani.dsp import pipeline


def _voiced(n, sr=16000):
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def test_burst_detected_and_gate_recovers():
    sr = 16000; n = 3 * sr
    prim = _voiced(n) * 0.2; ref = np.roll(prim, 5) * 0.2
    # far-field impulse: similar level at both mics
    imp = np.exp(-np.arange(800) / 100).astype(np.float32) * np.random.default_rng(0).standard_normal(800).astype(np.float32)
    prim[sr:sr + 800] += imp; ref[sr:sr + 800] += imp * 0.9
    out = pipeline.run(np.stack([prim, ref]))
    f0 = sr // 256
    assert out["burst"][f0:f0 + 4].any()
    assert out["gate"][f0 + 1] < 0.1
    assert out["gate"][-1] > 0.9                     # ramped back well before the end


def test_consonant_does_not_trip_burst():
    sr = 16000; n = 2 * sr
    prim = _voiced(n) * 0.3; ref = np.roll(prim, 5) * 0.15   # near-mouth: ref much quieter
    rng = np.random.default_rng(1)
    # 20 ms wideband transient at -10 dB rel. voicing, primary only (a consonant)
    prim[sr:sr + 320] += rng.standard_normal(320).astype(np.float32) * 0.3 * 10 ** (-10 / 20)
    out = pipeline.run(np.stack([prim, ref]))
    assert not out["burst"].any()
