"""mixer._istft replaces scipy's istft inside diffuse_pair: it must be bit-identical (resumed runs re-render items)."""
import numpy as np
import pytest
from scipy.signal import istft

from vaani.data import mixer


def _ref(X, nfft, hop, n):
    return istft(X, nperseg=nfft, noverlap=nfft - hop, boundary=True)[1][:n].astype(np.float32)


@pytest.mark.parametrize("nfft,hop", [(512, 128), (512, 256), (256, 64), (1024, 256)])
def test_fast_istft_is_bit_identical_to_scipy(nfft, hop):
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(nfft, 70000))
        x = rng.standard_normal(n) * 10 ** rng.uniform(-6, 2)
        if trial % 7 == 0:
            x[: n // 3] = 0.0                                        # silence: signed zeros must survive too
        X = mixer._stft(x, nfft, hop)[2]
        X = X * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))   # a modified STFT
        if trial % 5 == 0:
            X.imag[:, ::3] = -0.0
        a, b = mixer._istft(X, nfft, hop, n), _ref(X, nfft, hop, n)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes(), (nfft, hop, n, trial)


def test_diffuse_pair_unchanged_on_real_draws():
    for seed in range(300):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(int(rng.integers(8000, 64000))).astype(np.float32)
        for model, gamma in (("spherical", None), ("cylindrical", None), ("spherical", 0.3)):
            r1, r2 = np.random.default_rng(seed + 1), np.random.default_rng(seed + 1)
            fast = mixer.diffuse_pair(r1, x, gamma, model=model)
            orig = mixer._istft
            try:
                mixer._istft = lambda X, nfft, hop, n: _ref(X, nfft, hop, n)
                ref = mixer.diffuse_pair(r2, x, gamma, model=model)
            finally:
                mixer._istft = orig
            assert fast.tobytes() == ref.tobytes(), (seed, model)
