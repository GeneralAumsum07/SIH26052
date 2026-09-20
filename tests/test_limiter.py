import numpy as np

from vaani.dsp.limiter import Limiter

SR, HOP = 16000, 256


def _run(lim, p, r):
    op, orr = np.zeros_like(p), np.zeros_like(r)
    for i in range(0, len(p), HOP):
        op[i:i + HOP], orr[i:i + HOP] = lim.process_block(p[i:i + HOP], r[i:i + HOP])
    return op, orr


def _scene(rng, n=SR):
    # far-field noise equal at both mics, near-mouth "speech" 12 dB louder at the primary with a 20 dB crest
    noise = rng.standard_normal(n) * 0.003
    sp = rng.standard_normal(n) * 0.01 * (1 + 4 * (np.sin(2 * np.pi * 4 * np.arange(n) / SR) > 0.9))
    return noise + sp, noise + sp * 10 ** (-12 / 20)


def test_speech_and_its_onsets_pass_untouched():
    rng = np.random.default_rng(0)
    p, r = _scene(rng)
    p[: SR // 2] = 0.0005 * rng.standard_normal(SR // 2); r[: SR // 2] = p[: SR // 2]  # silence, then speech
    lim = Limiter(); op, orr = _run(lim, p, r)
    assert np.allclose(op, p) and np.allclose(orr, r) and lim.engaged == 0


def test_far_field_blast_is_clamped_to_the_headroom_and_both_mics_share_the_gain():
    rng = np.random.default_rng(1)
    p, r = _scene(rng)
    k = SR // 2; burst = np.exp(-np.arange(320) / 60) * 0.9   # 20 ms decaying blast, ~38 dB over the programme
    p[k:k + 320] += burst; r[k:k + 320] += burst
    lim = Limiter(); op, orr = _run(lim, p, r)
    assert lim.engaged > 0
    assert np.abs(op[k:k + 320]).max() < 0.3 * np.abs(p[k:k + 320]).max()
    # shared gain: the per-sample ratio between the mics is exactly preserved inside the burst
    g = op[k:k + 64] / p[k:k + 64]
    assert np.allclose(orr[k:k + 64], r[k:k + 64] * g)
    # and bit-identical once the release has run out (~350 ms)
    assert np.allclose(op[k + 7000:], p[k + 7000:])   # 30 dB of gain reduction needs ~7 time constants to release


def test_block_processing_is_causal_and_stateful():
    rng = np.random.default_rng(2)
    p, r = _scene(rng); p[SR // 2:SR // 2 + 64] += 0.9; r[SR // 2:SR // 2 + 64] += 0.9
    a = _run(Limiter(), p, r)[0]
    b = _run(Limiter(), p[: SR // 2 + HOP], r[: SR // 2 + HOP])[0]   # same prefix, later samples absent
    assert np.allclose(a[: SR // 2 + HOP], b)
