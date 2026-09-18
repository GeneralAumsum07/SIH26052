import time

import numpy as np

from vaani.dsp import nlms as nlms_mod
from vaani.dsp.nlms import NLMS


def test_converges_on_filtered_reference():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000 * 4).astype(np.float32)
    h = np.array([0.0, 0.5, -0.3, 0.1], np.float32)
    prim = np.convolve(ref, h)[: len(ref)].astype(np.float32)   # noise-only at primary
    f = NLMS()
    for i in range(0, len(ref) - 256, 256):
        n_hat, _ = f.process_block(prim[i:i + 256], ref[i:i + 256], gate=1.0)
    # last block error must be well below the input energy
    err = prim[i:i + 256] - n_hat
    assert 10 * np.log10((prim[i:i + 256] ** 2).mean() / ((err ** 2).mean() + 1e-12)) > 20


def test_gate_zero_freezes_taps():
    rng = np.random.default_rng(0)
    f = NLMS(); ref = rng.standard_normal(256).astype(np.float32); prim = ref.copy()
    f.process_block(prim, ref, gate=1.0); w = f.w.copy()
    f.process_block(prim, ref, gate=0.0)
    assert np.array_equal(w, f.w)


def test_state_persists_across_blocks():
    # one 512-block must equal two consecutive 256-blocks (same running state)
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(512).astype(np.float32)
    prim = (np.convolve(ref, [0.0, 0.4, -0.2], mode="full")[:512]).astype(np.float32)

    f_one = NLMS()
    n_hat_one, ratio_one = f_one.process_block(prim, ref, gate=1.0)

    f_two = NLMS()
    n_hat_a, _ = f_two.process_block(prim[:256], ref[:256], gate=1.0)
    n_hat_b, ratio_two = f_two.process_block(prim[256:], ref[256:], gate=1.0)
    n_hat_two = np.concatenate([n_hat_a, n_hat_b])

    assert np.allclose(n_hat_one, n_hat_two, rtol=1e-5)
    assert np.array_equal(f_one.w, f_two.w)


def test_reset_restores_initial_behaviour():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(256).astype(np.float32)
    prim = ref.copy()

    f = NLMS()
    n_hat_first, ratio_first = f.process_block(prim, ref, gate=1.0)
    f.process_block(prim, ref, gate=1.0)  # perturb state further
    f.reset()
    n_hat_after_reset, ratio_after_reset = f.process_block(prim, ref, gate=1.0)

    assert np.allclose(n_hat_first, n_hat_after_reset)
    assert ratio_first == ratio_after_reset


def test_numba_path_matches_pure_path():
    if not nlms_mod._HAVE_NUMBA:
        return  # numba not installed; fallback is the only path, nothing to compare
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(4000).astype(np.float32)
    h = np.array([0.0, 0.5, -0.3, 0.1], np.float32)
    prim = np.convolve(ref, h)[: len(ref)].astype(np.float32)

    f_numba = NLMS()
    f_pure = NLMS(force_pure=True)
    n_hat_numba, ratio_numba = f_numba.process_block(prim, ref, gate=1.0)
    n_hat_pure, ratio_pure = f_pure.process_block(prim, ref, gate=1.0)

    assert np.allclose(n_hat_numba, n_hat_pure, rtol=1e-5, atol=1e-6)
    assert np.isclose(ratio_numba, ratio_pure, rtol=1e-5, atol=1e-6)


def test_timing_4s_clip():
    # informational: measures pure-path throughput on a realistic clip length
    rng = np.random.default_rng(4)
    ref = rng.standard_normal(16000 * 4).astype(np.float32)
    prim = ref.copy()
    f = NLMS(force_pure=True)
    start = time.perf_counter()
    for i in range(0, len(ref) - 256, 256):
        f.process_block(prim[i:i + 256], ref[i:i + 256], gate=1.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 60  # generous ceiling; real number reported separately
