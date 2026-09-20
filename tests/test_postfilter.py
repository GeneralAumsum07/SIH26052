import numpy as np, pytest

from vaani.dsp.postfilter import ResidualPostFilter


def _spec(T=80, seed=46, scale=1.0):
    rng = np.random.default_rng(seed)
    return (scale * (rng.normal(size=(257, T)) + 1j * rng.normal(size=(257, T)))).astype(np.complex64)


def _run(pf, y):
    return np.stack([pf.process_frame(y[:, t]) for t in range(y.shape[1])], 1)


def test_unity_floor_is_identity():
    y = _spec()
    z = _run(ResidualPostFilter(gain_floor=1.0), y)
    np.testing.assert_array_equal(z, y)


def test_zero_input_is_finite_zero_through_startup():
    z = _run(ResidualPostFilter(), np.zeros((257, 100), np.complex64))
    assert np.isfinite(z).all() and not z.any()


def test_gain_bounded_and_phase_preserved():
    y = _spec(); pf = ResidualPostFilter(gain_floor=0.7); z = _run(pf, y)
    ratio = np.abs(z) / np.abs(y)
    assert (ratio >= 0.7 - 1e-6).all() and (ratio <= 1 + 1e-6).all()
    assert np.allclose(np.angle(z[np.abs(y) > 0.1]), np.angle(y[np.abs(y) > 0.1]), atol=1e-5)
    assert (ratio[:, :16] > 1 - 1e-6).all()      # warm-up frames pass through
    assert (ratio[:, 16:] < 1 - 1e-6).any()      # and something is actually attenuated afterwards on noise


def test_causal_prefix_unchanged_by_future_frames():
    y = _spec(); y2 = y.copy(); y2[:, 50:] = _spec(seed=7)[:, 50:] * 10
    a = _run(ResidualPostFilter(), y); b = _run(ResidualPostFilter(), y2)
    np.testing.assert_array_equal(a[:, :50], b[:, :50])


def test_reset_equals_fresh_instance_and_chunking_is_invisible():
    y = _spec(T=93)
    pf = ResidualPostFilter(); _run(pf, _spec(seed=1)); pf.reset()
    ref = _run(pf, y)
    np.testing.assert_array_equal(ref, _run(ResidualPostFilter(), y))
    for chunk in (1, 7, 23):
        pf = ResidualPostFilter(); out = []
        for s in range(0, 93, chunk): out.append(_run(pf, y[:, s:s + chunk]))
        np.testing.assert_array_equal(np.concatenate(out, 1), ref)
    # clip order does not leak state
    pf = ResidualPostFilter(); pf.process(_spec(seed=3)); np.testing.assert_array_equal(pf.process(y), ref)


def test_onset_restores_gain_immediately():
    y = _spec(T=120, scale=0.05); y[:, 90:] = _spec(T=120, seed=9)[:, 90:] * 5   # quiet residual, then a loud onset
    pf = ResidualPostFilter(gain_floor=0.7); z = _run(pf, y)
    g = np.abs(z) / np.maximum(np.abs(y), 1e-9)
    assert g[:, 89].mean() < 0.9 and g[:, 90].mean() > 0.97


def test_unknown_key_rejected():
    with pytest.raises(TypeError): ResidualPostFilter(bogus=1)
