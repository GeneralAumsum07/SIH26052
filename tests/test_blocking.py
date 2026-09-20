import numpy as np

from vaani.dsp import pipeline
from vaani.dsp.blocking import BlockingMatrix

SR, HOP = 16000, 256


def test_blocking_matrix_learns_the_speech_path_and_keeps_the_noise():
    rng = np.random.default_rng(0)
    n = 4 * SR
    speech = rng.standard_normal(n).astype(np.float32) * 0.1
    noise_p = rng.standard_normal(n).astype(np.float32) * 0.01
    noise_r = noise_p * 0.9 + rng.standard_normal(n).astype(np.float32) * 0.004   # far-field: near-equal at both mics
    # mouth -> reference path: -10 dB, 5 samples late
    sp_ref = np.concatenate([np.zeros(5, np.float32), speech[:-5]]) * 10 ** (-10 / 20)
    prim, ref = speech + noise_p, sp_ref + noise_r
    bm = BlockingMatrix(); out = np.zeros_like(ref)
    for i in range(0, n, HOP):
        out[i:i + HOP] = bm.process_block(prim[i:i + HOP], ref[i:i + HOP], 1.0)
    tail = slice(2 * SR, n)
    leak_before = np.corrcoef(ref[tail], sp_ref[tail])[0, 1]
    leak_after = np.corrcoef(out[tail], sp_ref[tail])[0, 1]
    assert leak_before > 0.9 and abs(leak_after) < 0.2          # speech gone from the reference
    assert np.corrcoef(out[tail], noise_r[tail])[0, 1] > 0.8    # the noise it is there for survives
    assert abs(bm.f.w[5]) > 0.2 and abs(bm.f.w[5] - 10 ** (-10 / 20)) < 0.08   # it found the 5-sample, -10 dB path


def test_blocking_matrix_frozen_at_gate_zero_and_inert_in_pipeline_without_controller():
    rng = np.random.default_rng(1)
    prim = rng.standard_normal(HOP * 8).astype(np.float32); ref = 0.3 * prim
    bm = BlockingMatrix()
    for i in range(0, len(prim), HOP):
        out = bm.process_block(prim[i:i + HOP], ref[i:i + HOP], 0.0)
    assert np.array_equal(out, ref[-HOP:]) and not bm.f.w.any()
    x = np.stack([prim, ref])
    a = pipeline.run(x, controller_on=False, dsp_cfg={"blocking": True})
    b = pipeline.run(x, controller_on=False)
    assert np.array_equal(a["n_hat"], b["n_hat"])
