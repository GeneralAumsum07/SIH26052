import numpy as np
import pytest

from vaani.dnsmos import DNSMOS, MODEL, WINDOW


@pytest.mark.skipif(not MODEL.exists(), reason="deploy/dnsmos/sig_bak_ovr.onnx not fetched")
def test_dnsmos_ranks_clean_tone_burst_above_the_same_in_white_noise():
    rng = np.random.default_rng(0); sr = 16000; t = np.arange(4 * sr) / sr
    # speech-like stand-in: AM-modulated harmonic stack; DNSMOS is a speech-quality model, so a pure tone is unfair to it
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560, 700), 1)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    x = (0.1 * x / np.abs(x).max()).astype(np.float32)
    noisy = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
    m = DNSMOS()
    a, b = m(x), m(noisy)
    assert set(a) == {"sig", "bak", "ovrl"} and all(1.0 <= v <= 5.0 for v in a.values())
    assert a["bak"] > b["bak"] and a["ovrl"] > b["ovrl"]   # noise floor must read as worse background
    assert len(x) < WINDOW                                   # the 4 s clip exercised the repeat-to-fill path
