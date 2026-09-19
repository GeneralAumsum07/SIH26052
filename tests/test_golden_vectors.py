import numpy as np, soundfile as sf
from pathlib import Path
from vaani.dsp import pipeline

VEC = Path("deploy/dsp_reference/vectors")


def test_golden_vectors_replay_from_their_own_wavs():
    """The C port reads the .wav; the .npz must be what pipeline.run gives for exactly that audio."""
    for npz in sorted(VEC.glob("*.npz")):
        x, _ = sf.read(npz.with_suffix(".wav"), dtype="float32"); ref = np.load(npz)
        r = pipeline.run(np.ascontiguousarray(x.T))
        assert np.abs(r["n_hat"] - ref["n_hat"]).max() < 1e-4, npz.name
        assert np.abs(r["features"] - ref["features"]).max() < 1e-4, npz.name
        for k in ("gate", "burst", "reliability"):
            assert np.array_equal(r[k], ref[k]), (npz.name, k)
