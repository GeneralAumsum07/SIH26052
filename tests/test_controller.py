import numpy as np
from vaani.dsp import pipeline


def _voiced(n, sr=16000):
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def test_burst_detected_and_gate_recovers():
    sr = 16000; n = 3 * sr
    # far-field background: prim and ref at the same level (0 dB diff) -> speech_presence ~0
    prim = _voiced(n) * 0.2; ref = np.roll(prim, 5)
    # far-field impulse: similar level at both mics
    imp = np.exp(-np.arange(800) / 100).astype(np.float32) * np.random.default_rng(0).standard_normal(800).astype(np.float32)
    prim[sr:sr + 800] += imp; ref[sr:sr + 800] += imp * 0.9
    out = pipeline.run(np.stack([prim, ref]))
    f0 = sr // 256
    assert out["burst"][f0:f0 + 4].any()
    assert out["gate"][f0 + 1] < 0.1
    assert out["gate"][-1] > 0.9                     # ramped back well before the end


def test_gate_freezes_during_near_end_speech():
    sr = 16000; n = 3 * sr
    # near-mouth: primary >> reference (~14 dB) for the whole clip -> speech_presence ~1
    prim = _voiced(n) * 0.2; ref = np.roll(prim, 5) * 0.2
    out = pipeline.run(np.stack([prim, ref]))
    f0 = sr // 256
    assert (out["gate"][f0:] < 0.1).all()


def test_consonant_does_not_trip_burst():
    sr = 16000; n = 2 * sr
    prim = _voiced(n) * 0.3; ref = np.roll(prim, 5) * 0.15   # near-mouth: ref much quieter
    rng = np.random.default_rng(1)
    # 20 ms wideband transient at -10 dB rel. voicing, primary only (a consonant)
    prim[sr:sr + 320] += rng.standard_normal(320).astype(np.float32) * 0.3 * 10 ** (-10 / 20)
    out = pipeline.run(np.stack([prim, ref]))
    assert not out["burst"].any()


def test_speech_onset_freezes_within_one_frame():
    """Fast attack: the first voiced frame after a pause must already be frozen, otherwise a 0.05-step
    64-tap NLMS (80 ms time constant) learns the speech path before the smoothed detector reacts."""
    sr = 16000; n = 3 * sr
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n).astype(np.float32) * 0.02
    prim = noise.copy(); ref = np.roll(noise, 3).copy()          # far-field noise, ~0 dB inter-mic
    v = _voiced(n) * 0.2; v[: sr] = 0; v[2 * sr:] = 0              # speech only in the middle second
    prim += v; ref += np.roll(v, 5) * 0.25                          # near-mouth: ref ~12 dB down
    out = pipeline.run(np.stack([prim, ref]))
    hop = 256; f_on = sr // hop
    assert out["gate"][: f_on - 2].mean() > 0.9                     # adapting on pure noise
    assert out["gate"][f_on + 1] < 0.1                               # frozen by the frame after onset
    assert out["gate"][f_on + 2: 2 * f_on - 6].mean() < 0.02              # and stays frozen through the speech (brief AM-dip openings tolerated)
    assert out["gate"][-1] > 0.9                                     # released after speech ends


def test_speech_presence_survives_weak_reference_gain():
    """A reference only 8 dB below the primary (the mixer's weakest headset) is still speech, after a
    short noise-only lead-in has shown the detector where the far-field ratio sits."""
    sr = 16000; n = 3 * sr
    rng = np.random.default_rng(0); noise = rng.standard_normal(n).astype(np.float32) * 0.01
    v = _voiced(n) * 0.2; v[: sr // 2] = 0
    prim = noise + v; ref = np.roll(noise, 3) + np.roll(v, 5) * 10 ** (-8 / 20)
    out = pipeline.run(np.stack([prim, ref]))
    f0 = sr // 2 // 256 + 2
    assert (out["gate"][f0:] < 0.1).mean() > 0.95


def test_subframe_jump_sees_a_4ms_click():
    """A 4 ms click inside a 32 ms frame barely moves the frame energy; the sub-frame jump must still fire."""
    from vaani.dsp.features import FrameFeatures, N_FEATURES
    ff = FrameFeatures(); rng = np.random.default_rng(0)
    base = rng.standard_normal(512).astype(np.float32) * 0.01
    P = np.fft.rfft(base); R = P.copy()
    for _ in range(10):
        ff.compute(base, base, P, R, 0.0, 1.0)
    frame = base.copy(); frame[300:364] += 0.5 * rng.standard_normal(64).astype(np.float32)
    f = ff.compute(frame, frame, P, R, 0.0, 1.0)
    assert f.shape == (N_FEATURES,)
    assert f[0] >= 12.0


def test_differential_jump_separates_far_field_click_from_near_mouth_onset():
    """Same primary onset either way; only the reference tells a burst (equal at both mics) from a consonant."""
    from vaani.dsp.features import FrameFeatures
    ff = FrameFeatures(); rng = np.random.default_rng(0)
    base = rng.standard_normal(512).astype(np.float32) * 0.01
    P = np.fft.rfft(base); R = P.copy()
    for _ in range(10):
        ff.compute(base, base, P, R, 0.0, 1.0)
    click = np.zeros(512, np.float32); click[300:364] = 0.5 * rng.standard_normal(64)
    ff.compute(base + click, base + click, P, R, 0.0, 1.0)
    assert abs(ff.diff_jump) < 3.0                       # far-field: both mics jump alike
    ff.reset()
    for _ in range(10):
        ff.compute(base, base, P, R, 0.0, 1.0)
    ff.compute(base + click, base + click * 10 ** (-12 / 20), P, R, 0.0, 1.0)
    assert ff.diff_jump > 8.0                             # near-mouth: the primary jumps ~12 dB more


def test_controller_diff_jump_rule_is_opt_in():
    from vaani.dsp.controller import Controller
    from vaani.dsp.features import FEATURE_NAMES
    f = np.zeros(len(FEATURE_NAMES), np.float32); f[0] = 20.0; f[14] = 10.0   # big onset, primary 10 dB louder
    assert Controller().step(f)[1] is False                                   # legacy rule: level_diff says speech
    assert Controller(diff_jump_max_db=3.0).step(f, diff_jump=0.5)[1] is True  # 2.7a rule: both mics jumped alike
    assert Controller(diff_jump_max_db=3.0).step(f, diff_jump=9.0)[1] is False
