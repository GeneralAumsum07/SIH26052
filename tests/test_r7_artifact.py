"""The shipped r7 artifact itself: the exact graph bytes, live == offline on it, and a real enhancement gain.

tests/test_live.py proves the engine against a random-weight graph; this pins the file that ships
(deploy/r7/cascade.onnx) and the DSP configuration committed beside it.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vaani import export, live, metrics
from vaani.dsp import pipeline, stft

R7 = Path("deploy/r7")
ONNX = R7 / "cascade.onnx"
ONNX_SHA256 = "e67a2c42a2fd53ec4d7e6c4dfbf9cef920c249e499a509a2b40af401c5cefe3c"
VECTOR = Path("deploy/dsp_reference/vectors_cascade/speech_plus_noise.npz")
SR = 16000
# measured 12.29 dB (2.16 -> 14.45 dB SI-SDR) on _voiced_mix(); the floor keeps a ~4 dB margin for ORT/BLAS drift
SI_SDR_GAIN_FLOOR_DB = 8.0


def _cfg():
    return live.load_model_config(R7 / "model_config.json")


def _offline(mix, cfg):
    """The eval path: whole-clip DSP, torch STFT, frame-by-frame ORT with caches, torch iSTFT."""
    r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=cfg["dsp"])
    x = torch.from_numpy(r["mix"])[None]
    spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1).numpy()
    sess = export.load_session(ONNX); names, zero = export.zero_caches(sess)
    out, _ = export.stream_onnx(sess, spec6, np.ascontiguousarray(r["features"][None], np.float32), names, zero)
    return r, stft.istft(torch.from_numpy(out), length=mix.shape[1])[0].numpy()


def _live(mix, r, cfg):
    # the offline STFT reflects the clip start into frame 0; the engine gets the same left context
    lc = np.stack([r["mix"][0][256:0:-1], r["mix"][1][256:0:-1], r["n_hat"][256:0:-1]])
    eng = live.StreamEngine(ONNX, cfg["controller_on"], cfg["dsp"], left_context=lc)
    B = mix.shape[1] // live.HOP
    y = np.concatenate([eng.process(mix[0, j * live.HOP:(j + 1) * live.HOP], mix[1, j * live.HOP:(j + 1) * live.HOP])
                        for j in range(B)])
    return y[live.HOP:]                                   # one hop behind


def _voiced_mix(seconds=2.0, seed=0, noise_rms=0.05):
    """Deterministic speech-like primary (gliding harmonic stack, 3 Hz syllables), near-field geometry as the
    golden vectors (reference: 0.3x speech, shifted noise). A pure tone is not usable here: r7 treats a steady
    150 Hz sine as noise (the committed vector goes 12.55 -> 0.28 dB SI-SDR), so it cannot carry a gain floor."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 30))
    s = (0.3 * x * np.clip(np.sin(2 * np.pi * 3 * t), 0, None) ** 2 / np.abs(x).max()).astype(np.float32)
    noise = (np.random.default_rng(seed).standard_normal(len(s)) * noise_rms).astype(np.float32)
    return s, np.stack([s + noise, np.roll(s, 5) * 0.3 + np.roll(noise, 2)]).astype(np.float32)


def test_shipped_graph_is_the_recorded_bytes():
    assert ONNX.exists(), f"missing shipped graph {ONNX}"
    assert hashlib.sha256(ONNX.read_bytes()).hexdigest() == ONNX_SHA256


def test_live_matches_offline_on_the_committed_vector():
    mix = np.ascontiguousarray(np.load(VECTOR)["mix"], np.float32)   # inputs only; expected outputs may be regenerated
    cfg = _cfg()
    r, y_off = _offline(mix, cfg)
    y = _live(mix, r, cfg)
    assert len(y) == (mix.shape[1] // live.HOP - 1) * live.HOP
    assert np.abs(y - y_off[:len(y)]).max() < 1e-5
    assert np.abs(y_off).max() > 1e-3                     # a vacuous pass on silence would prove nothing


def test_live_matches_offline_and_enhances_speech_like_input():
    clean, mix = _voiced_mix()
    cfg = _cfg()
    r, y_off = _offline(mix, cfg)
    y = _live(mix, r, cfg)
    assert np.abs(y - y_off[:len(y)]).max() < 1e-5
    n = len(y)
    gain_off = metrics.si_sdr_db(clean, y_off) - metrics.si_sdr_db(clean, mix[0])
    gain_live = metrics.si_sdr_db(clean[:n], y) - metrics.si_sdr_db(clean[:n], mix[0, :n])
    assert gain_off > SI_SDR_GAIN_FLOOR_DB, gain_off
    assert gain_live > SI_SDR_GAIN_FLOOR_DB, gain_live


@pytest.mark.timeout(120)
def test_live_path_runs_without_torch():
    """The board has no torch. vaani/dsp/stft.py imports it opportunistically, so on this machine it would load
    anyway; block it (sys.modules['torch'] = None makes `import torch` raise) and the engine must still run."""
    code = "\n".join([
        "import sys, numpy as np",
        "sys.modules['torch'] = None",
        "from vaani import live",
        f"cfg = live.load_model_config({str(R7 / 'model_config.json')!r})",
        f"eng = live.StreamEngine({str(ONNX)!r}, cfg['controller_on'], cfg['dsp'])",
        "z = np.random.default_rng(0).standard_normal((2, 4 * live.HOP)).astype(np.float32) * 0.05",
        "y = np.concatenate([eng.process(z[0, j * live.HOP:(j + 1) * live.HOP], z[1, j * live.HOP:(j + 1) * live.HOP])"
        " for j in range(4)])",
        "print(y.shape[0], bool(np.isfinite(y).all()))",
    ])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path.cwd(), timeout=110)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == f"{4 * live.HOP} True"
