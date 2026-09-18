from pathlib import Path

import numpy as np, soundfile as sf
from vaani.data import sources

SR = 16000
_T = np.arange(SR * 3) / SR


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


def _white_noise():
    return (np.random.default_rng(0).standard_normal(len(_T)).astype(np.float32) * 0.1)


def _am_hum():
    return ((1 + 0.3 * np.sin(2 * np.pi * 0.5 * _T)) * np.sin(2 * np.pi * 150 * _T)).astype(np.float32) * 0.1


def _gated_bursts():
    gate = (np.sin(2 * np.pi * 1.5 * _T) > 0).astype(np.float32)
    return _white_noise() * gate


def _chirp():
    return (0.2 * np.sin(2 * np.pi * (100 + (4000 - 100) * _T / 3) * _T)).astype(np.float32)


def test_stationarity_class_white_and_am_hum_are_stationary():
    assert sources.stationarity_class(_white_noise(), SR) == "stationary"
    assert sources.stationarity_class(_am_hum(), SR) == "stationary"


def test_stationarity_class_bursts_and_chirp_are_changing():
    assert sources.stationarity_class(_gated_bursts(), SR) == "changing"
    assert sources.stationarity_class(_chirp(), SR) == "changing"


def test_scan_dns_noise_classifies_and_sets_licence(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    sf.write(root / "white.wav", _white_noise(), SR)
    sf.write(root / "bursts.wav", _gated_bursts(), SR)
    out = tmp_path / "out"
    rows = sources.scan_dns_noise(root, out)
    by_stem = {Path(r["path"]).stem: r for r in rows}
    assert by_stem["white"]["noise_class"] == "stationary"
    assert by_stem["bursts"]["noise_class"] == "changing"
    assert all(r["licence"] == "DNS-4 archive noise_fullband (see DNS README per-clip licences)" for r in rows)


def test_scan_esc50_excludes_vocal_and_labels_impulsive(tmp_path):
    root = tmp_path / "root"; (root / "meta").mkdir(parents=True); (root / "audio").mkdir()
    rows_csv = [("a.wav", "rain", "100"), ("b.wav", "door_wood_knock", "101"), ("c.wav", "coughing", "102")]
    with open(root / "meta" / "esc50.csv", "w") as fh:
        fh.write("filename,fold,target,category,esc10,src_file,take\n")
        for fn, cat, src in rows_csv:
            fh.write(f"{fn},1,0,{cat},False,{src},A\n")
            sf.write(root / "audio" / fn, _white_noise(), SR)
    rows = sources.scan_esc50(root, tmp_path / "out")
    by_cat = {r["source_id"].split(":")[1].split("/")[0]: r for r in rows}
    assert set(by_cat) == {"rain", "door_wood_knock"}
    assert by_cat["rain"]["noise_class"] == "stationary"
    assert by_cat["door_wood_knock"]["noise_class"] == "impulsive"
    assert by_cat["rain"]["group_id"] == "esc50-100"
