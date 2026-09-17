import numpy as np, soundfile as sf
from vaani.data import sources


def test_estimate_snr_clean_vs_noisy():
    sr = 16000
    t = np.arange(sr * 2) / sr
    speech = (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    clean = speech
    noisy = speech + 0.1 * np.random.randn(len(t)).astype(np.float32)
    assert sources.estimate_snr_db(clean, sr) > sources.estimate_snr_db(noisy, sr) + 10


def test_to_flac16k_resamples(tmp_path):
    x = np.random.randn(48000).astype(np.float32) * 0.1
    sf.write(tmp_path / "a.wav", x, 48000)
    dur = sources.to_flac16k(tmp_path / "a.wav", tmp_path / "a.flac")
    y, sr = sf.read(tmp_path / "a.flac")
    assert sr == 16000 and abs(dur - 1.0) < 0.01 and len(y) == 16000
