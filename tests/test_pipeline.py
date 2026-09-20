import numpy as np
from vaani.dsp import pipeline, features, stft


def test_shapes_and_ablation():
    x = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32) * 0.1
    out = pipeline.run(x)
    T = stft.np_stft(x[0]).shape[1]
    assert out["features"].shape == (T, features.N_FEATURES) and out["n_hat"].shape == (16000,)
    off = pipeline.run(x, controller_on=False)
    assert np.all(off["gate"] == 1.0) and np.all(off["features"] == 0.0)


def test_dsp_cfg_limiter_returns_the_limited_mix_and_defaults_are_untouched():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 16000)).astype(np.float32) * 0.01
    x[:, 8000:8320] += np.exp(-np.arange(320) / 60) * 0.9      # far-field blast on both mics
    base = pipeline.run(x)
    assert np.array_equal(base["mix"], x)                        # default path hands the input back
    lim = pipeline.run(x, dsp_cfg={"limiter": True, "controller": {"diff_jump_max_db": 3.0}})
    assert np.abs(lim["mix"][:, 8000:8320]).max() < 0.3 * 0.9 and np.allclose(lim["mix"][:, :7900], x[:, :7900])
    assert lim["burst"].any()
