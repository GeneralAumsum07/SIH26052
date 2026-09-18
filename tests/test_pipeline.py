import numpy as np
from vaani.dsp import pipeline, features, stft


def test_shapes_and_ablation():
    x = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32) * 0.1
    out = pipeline.run(x)
    T = stft.np_stft(x[0]).shape[1]
    assert out["features"].shape == (T, features.N_FEATURES) and out["n_hat"].shape == (16000,)
    off = pipeline.run(x, controller_on=False)
    assert np.all(off["gate"] == 1.0) and np.all(off["features"] == 0.0)
