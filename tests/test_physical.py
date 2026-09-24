"""Physical-test path: CSV loader, scoring with and without a clean reference, and the CLI's columns."""
import csv
import math

import numpy as np
import pytest

from vaani import live, physical

HAVE_R7 = physical.ONNX.exists() and physical.CONFIG.exists()


def _speechish(seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000); t = np.arange(n) / 16000
    clean = (0.1 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32) * 0.02
    return clean, np.stack([clean + noise, 0.3 * clean + np.roll(noise, 3)]).astype(np.float32)


def _write_set(tmp_path, with_clean):
    (tmp_path / "clips").mkdir()
    clean, mix = _speechish()
    live.write_wav(tmp_path / "clips" / "a.wav", mix, 16000)
    rows = [{"id": "a", "speaker": "s1", "language": "en", "condition": "engine", "transcript": "", "has_clean": 0}]
    if with_clean:
        live.write_wav(tmp_path / "clips" / "b.wav", mix, 16000)
        live.write_wav(tmp_path / "clips" / "b.clean.wav", clean, 16000)
        rows.append({"id": "b", "speaker": "s1", "language": "hi", "condition": "quiet", "transcript": "x",
                     "has_clean": 1})
    with open(tmp_path / "physical.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=physical.CSV_COLUMNS); w.writeheader(); w.writerows(rows)
    return tmp_path / "physical.csv"


def test_loader_parses_rows_and_paths(tmp_path):
    rows = physical.load_physical_csv(_write_set(tmp_path, with_clean=True))
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["has_clean"] is False and rows[0]["clean"] is None
    assert rows[1]["has_clean"] is True and rows[1]["clean"] == tmp_path / "clips" / "b.clean.wav"
    mix, clean = physical.load_clip(rows[1])
    assert mix.shape == (2, 16000) and clean.shape == (16000,)


@pytest.mark.parametrize("bad, msg", [("id,speaker,condition,transcript,has_clean\n", "missing columns"),
                                      ("id,speaker,language,condition,transcript,has_clean\na,s,en,jungle,,0\n",
                                       "condition"),
                                      ("id,speaker,language,condition,transcript,has_clean\na,s,en,quiet,,0\n"
                                       "a,s,en,quiet,,0\n", "duplicate")])
def test_loader_rejects_bad_csv(tmp_path, bad, msg):
    p = tmp_path / "physical.csv"; p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        physical.load_physical_csv(p)


def test_attenuation_proxy_counts_vanished_frames():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    assert physical.attenuation_proxy(x, x)["atten20_frac"] == 0.0
    assert physical.attenuation_proxy(x, x * 0.01)["atten20_frac"] == 1.0      # -40 dB everywhere
    half = x.copy(); half[8000:] *= 0.01
    assert abs(physical.attenuation_proxy(x, half)["atten20_frac"] - 0.5) < 0.05


def test_to_16k_resamples_441():
    y = physical.to_16k(np.zeros((2, 44100), np.float32), 44100)
    assert y.shape == (2, 16000)


@pytest.mark.skipif(not HAVE_R7, reason="deploy/r7 artefacts missing")
def test_run_engine_is_aligned_and_full_length():
    _, mix = _speechish(0.5)
    y, diag = physical.run_engine(mix)
    assert y.shape == (mix.shape[1],) and np.isfinite(y).all()
    assert set(diag) == {"gate_mean", "burst_frac", "limiter_frac"}


@pytest.mark.skipif(not HAVE_R7, reason="deploy/r7 artefacts missing")
def test_cli_columns_with_and_without_clean(tmp_path):
    out = tmp_path / "scores.csv"
    _write_set(tmp_path, with_clean=True)
    physical.main(["--dir", str(tmp_path), "--out", str(out)])
    with open(out, encoding="utf-8") as fh:
        rd = csv.DictReader(fh); rows = list(rd)
    assert rd.fieldnames == physical.OUT_COLUMNS
    a, b = rows
    for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "atten20_frac"):
        assert math.isfinite(float(a[k])) and math.isfinite(float(b[k]))
    assert all(math.isnan(float(a[k])) for k in ("snr_out", "stoi", "pesq"))      # no clean: intrusive metrics blank
    assert math.isfinite(float(b["snr_out"])) and math.isfinite(float(b["stoi"]))  # clean: scored


def test_score_real_subset_is_one_clip_per_video_and_deterministic():
    import importlib.util
    spec = importlib.util.spec_from_file_location("score_real", physical.REPO / "scripts/score_real.py")
    sr = importlib.util.module_from_spec(spec); spec.loader.exec_module(sr)
    rows = [{"clip": f"c{v}_{k}", "video": f"v{v}", "path": None} for v in range(10) for k in range(3)]
    a, b = sr.select_subset(rows, 4, 0), sr.select_subset(rows, 4, 0)
    assert [r["clip"] for r in a] == [r["clip"] for r in b]
    assert len(a) == 4 and len({r["video"] for r in a}) == 4
    assert len(sr.select_subset(rows, 200, 0)) == 10
    assert sr._video_id("https://www.youtube.com/watch?v=abc123&t=5", "x") == "abc123"
    assert sr._video_id("not a url", "fb") == "fb"
