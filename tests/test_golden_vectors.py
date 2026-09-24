import numpy as np, soundfile as sf
import hashlib
import json
import pytest
import torch
from pathlib import Path
from vaani.dsp import pipeline

VEC = Path("deploy/dsp_reference/vectors")
CASCADE = Path("deploy/dsp_reference/vectors_cascade")


def test_cascade_vectors_are_bound_to_the_shipping_r7_artifacts():
    cfg = json.loads((CASCADE / "config.json").read_text())
    assert cfg["checkpoint"] == "results_r2/runs/r7_e256_wr64_refiner/best.pt"
    assert cfg["onnx"] == "deploy/r7/cascade.onnx"
    for key in ("checkpoint", "onnx"):  # the tracked copies, not the ignored runs/ tree, so a clone can check this
        assert hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest() == cfg[f"{key}_sha256"], key
    assert cfg["checkpoint_sha256"] == json.loads(Path("deploy/r7/model_config.json").read_text())["checkpoint_sha256"]
    ck = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)["config"]
    assert (ck["controller_on"], ck["dsp"]) == (cfg["controller_on"], cfg["dsp"])  # vectors use the checkpoint's own DSP settings
    assert len(cfg["git_sha"]) == 40


def test_cascade_vectors_exercise_limiter_and_blocking():
    directory = CASCADE
    cfg = json.loads((directory / "config.json").read_text())["dsp"]
    x, _ = sf.read(directory / "burst.wav", dtype="float32")
    assert np.max(np.abs(np.load(directory / "burst.npz")["mix"] - x.T)) > 0.01
    x, _ = sf.read(directory / "speech_onset.wav", dtype="float32")
    without_blocking = pipeline.run(x.T, dsp_cfg={**cfg, "blocking": False})
    expected = np.load(directory / "speech_onset.npz")
    assert np.max(np.abs(expected["n_hat"] - without_blocking["n_hat"])) > 0.01


@pytest.mark.parametrize("directory", [VEC, CASCADE])
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
