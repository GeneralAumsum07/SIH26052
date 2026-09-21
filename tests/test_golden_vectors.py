import numpy as np, soundfile as sf
import json
import pytest
from pathlib import Path
from vaani.dsp import pipeline

VEC = Path("deploy/dsp_reference/vectors")


def test_cascade_vectors_exercise_limiter_and_blocking():
    directory = Path("deploy/dsp_reference/vectors_cascade")
    cfg = json.loads((directory / "config.json").read_text())["dsp"]
    x, _ = sf.read(directory / "burst.wav", dtype="float32")
    assert np.max(np.abs(np.load(directory / "burst.npz")["mix"] - x.T)) > 0.01
    x, _ = sf.read(directory / "speech_onset.wav", dtype="float32")
    without_blocking = pipeline.run(x.T, dsp_cfg={**cfg, "blocking": False})
    expected = np.load(directory / "speech_onset.npz")
    assert np.max(np.abs(expected["n_hat"] - without_blocking["n_hat"])) > 0.01


@pytest.mark.parametrize("directory", [VEC, Path("deploy/dsp_reference/vectors_cascade")])
def test_golden_vectors_replay_from_their_own_wavs(directory):
    """The C port reads the .wav; the .npz must be what pipeline.run gives for exactly that audio."""
    cfg = json.loads((directory / "config.json").read_text()) if (directory / "config.json").exists() else {}
    vectors = sorted(directory.glob("*.npz"))
    assert vectors, f"No golden vectors in {directory}"
    for npz in vectors:
        x, _ = sf.read(npz.with_suffix(".wav"), dtype="float32"); ref = np.load(npz)
        r = pipeline.run(np.ascontiguousarray(x.T), controller_on=cfg.get("controller_on", True), dsp_cfg=cfg.get("dsp"))
        assert np.abs(r["n_hat"] - ref["n_hat"]).max() < 1e-4, npz.name
        assert np.abs(r["features"] - ref["features"]).max() < 1e-4, npz.name
        for k in ("gate", "burst", "reliability"):
            assert np.array_equal(r[k], ref[k]), (npz.name, k)
        if cfg:  # legacy vectors predate the limiter and did not store mix
            assert np.max(np.abs(r["mix"] - ref["mix"])) < 1e-4, npz.name
