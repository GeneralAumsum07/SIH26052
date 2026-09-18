import numpy as np
from scipy.signal import coherence

from vaani.data import mixer, rirs, impulses


def _speech(rng, n=32000):
    t = np.arange(n) / 16000
    env = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)
    return (np.sin(2 * np.pi * 180 * t) * env * 0.3).astype(np.float32)


def test_param_path_hits_target_snr_and_channels_differ():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=0.0, p_clip=0.0, p_ref_dropout=0.0, p_wind=0.0, p_clean=0.0, snr_range=(5.0, 5.0))
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], None, cfg)
    assert mix.shape == (2, len(s)) and clean.shape == (len(s),)
    assert not np.allclose(mix[0], mix[1])
    achieved = 10 * np.log10(mixer.speech_active_power(clean) / mixer.speech_active_power(mix[0] - clean))
    assert abs(achieved - 5.0) < 0.5
    # reference carries much less speech than primary
    assert meta["path"] == "param" and meta["ref_speech_gain_db"] <= -8


def test_room_path_runs(tmp_path):
    rirs.build_bank(tmp_path / "b.npz", n=2, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=1.0, p_clean=0.0)
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], bank, cfg)
    assert meta["path"] == "room" and np.isfinite(mix).all()


def test_clean_bucket_is_identity():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_clean=1.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0, p_room=0.0)
    s = _speech(rng)
    mix, clean, meta = mixer.mix(rng, s, [rng.standard_normal(len(s)).astype(np.float32)], None, [], None, cfg)
    assert meta["clean_bucket"] and np.allclose(mix[0], clean, atol=1e-6)


def test_room_path_reference_speech_level_within_physical_range(tmp_path):
    # geometry-driven leakage: primary is louder than reference by a plausible, but not huge, margin
    rirs.build_bank(tmp_path / "b.npz", n=4, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    cfg = mixer.MixConfig(p_room=1.0, p_clean=0.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0, snr_range=(30.0, 30.0))
    for seed in range(5):
        rng = np.random.default_rng(seed)
        s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
        mix, clean, meta = mixer.mix(rng, s, n, None, [], bank, cfg)
        diff = 10 * np.log10(mixer.speech_active_power(mix[0]) / mixer.speech_active_power(mix[1]))
        assert 3.0 < diff < 25.0, f"seed {seed}: observed {diff:.1f} dB"


def test_noise_coherence_param_high_room_lower():
    # noise-dominated mix (very low SNR) isolates the noise-path coherence structure
    cfg_param = mixer.MixConfig(p_room=0.0, p_clean=0.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0, snr_range=(-30.0, -30.0))
    rng = np.random.default_rng(0)
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, _, _ = mixer.mix(rng, s, n, None, [], None, cfg_param)
    f, cxy = coherence(mix[0], mix[1], fs=mixer.SR, nperseg=512)
    band = (f >= 100) & (f <= 4000)
    coh_param = float(cxy[band].mean())
    assert coh_param > 0.5, f"observed {coh_param:.2f}"


def test_noise_coherence_room_path_lower(tmp_path):
    rirs.build_bank(tmp_path / "b.npz", n=4, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    cfg_room = mixer.MixConfig(p_room=1.0, p_clean=0.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0, snr_range=(-30.0, -30.0))
    rng = np.random.default_rng(0)
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, _, _ = mixer.mix(rng, s, n, None, [], bank, cfg_room)
    f, cxy = coherence(mix[0], mix[1], fs=mixer.SR, nperseg=512)
    band = (f >= 100) & (f <= 4000)
    coh_room = float(cxy[band].mean())
    assert coh_room > 0.3, f"observed {coh_room:.2f}"


def test_determinism_param_and_room(tmp_path):
    rirs.build_bank(tmp_path / "b.npz", n=4, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    for bank_arg, p_room in [(None, 0.0), (bank, 1.0)]:
        cfg = mixer.MixConfig(p_room=p_room, p_clean=0.0)
        rng1 = np.random.default_rng(7); rng2 = np.random.default_rng(7)
        s = _speech(rng1); n = [np.random.default_rng(7).standard_normal(len(s)).astype(np.float32)]
        s2 = _speech(rng2); n2 = [np.random.default_rng(7).standard_normal(len(s)).astype(np.float32)]
        mix1, clean1, meta1 = mixer.mix(rng1, s, n, None, [], bank_arg, cfg)
        mix2, clean2, meta2 = mixer.mix(rng2, s2, n2, None, [], bank_arg, cfg)
        assert np.array_equal(mix1, mix2) and np.array_equal(clean1, clean2)
        assert meta1 == meta2


def test_impulse_injection_recorded_and_audible():
    rng = np.random.default_rng(0)
    imp, imp_meta = impulses.generate(np.random.default_rng(1), sr=mixer.SR, kind="burst")
    # force a loud, deterministic burst (well above the speech+noise floor) so the assertion isn't flaky
    cfg = mixer.MixConfig(p_room=0.0, p_clean=0.0, impulse_peak_db=(12.0, 12.0))
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32) * 0.01]
    mix, clean, meta = mixer.mix(rng, s, n, imp, imp_meta["onsets_s"], None, cfg)
    assert np.isfinite(meta["impulse_peak_db"])
    assert len(meta["impulse_onsets_s"]) > 0
    onset_sample = int(meta["impulse_onsets_s"][0] * mixer.SR)
    noise_floor = np.abs(mix[0][:onset_sample]).mean() if onset_sample > 0 else np.abs(mix[0]).mean()
    burst_peak = np.abs(mix[0][onset_sample:onset_sample + 200]).max()
    assert burst_peak > 3 * (noise_floor + 1e-9)


def test_twin_without_impulse_matches_outside_impulse_window():
    # same seed, with vs. without impulse: everything outside the impulse window must be
    # bit-identical so the recovery-time metric compares apples to apples. A rectangular,
    # constant-amplitude impulse (rather than impulses.generate's decay taper) gives sharp,
    # unambiguous edges -- a tapered edge can differ by less than float32 precision can show.
    imp = np.ones(1600, np.float32)  # 0.1 s pulse, onset offset 0.0
    cfg = mixer.MixConfig(p_room=0.0, p_clean=0.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0,
                           impulse_peak_db=(-3.0, -3.0))
    s = _speech(np.random.default_rng(2))
    n = [np.random.default_rng(2).standard_normal(len(s)).astype(np.float32)]
    rng_a = np.random.default_rng(0)
    mix_a, clean_a, meta_a = mixer.mix(rng_a, s, list(n), imp, [0.0], None, cfg)
    rng_b = np.random.default_rng(0)
    mix_b, clean_b, meta_b = mixer.mix(rng_b, s, list(n), None, [], None, cfg)

    start = int(round(meta_a["impulse_onsets_s"][0] * mixer.SR))  # onset == insertion point (offset 0.0)
    # mix() guarantees start + len(impulse) <= n, so seg is never truncated; the +1 guard covers
    # the 1-sample causal bleed from the post-impulse tilt IIR filter (memory of the last sample)
    end = start + len(imp) + 1
    assert np.array_equal(mix_a[:, :start], mix_b[:, :start])
    assert np.array_equal(mix_a[:, end:], mix_b[:, end:])
    assert np.array_equal(clean_a, clean_b)
    meta_keys = set(meta_a) - {"impulse_peak_db", "impulse_onsets_s"}
    for k in meta_keys:
        assert meta_a[k] == meta_b[k], k


def test_clip_bucket_forces_clip_and_stays_bounded():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=0.0, p_clean=0.0, p_clip=1.0, p_wind=0.0, p_ref_dropout=0.0)
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], None, cfg)
    assert meta["clipped"] and np.abs(mix[0]).max() <= 1.0


def test_ref_dropout_bucket_creates_low_energy_span():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=0.0, p_clean=0.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=1.0)
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], None, cfg)
    assert meta["ref_dropout"]
    # a 20 ms sliding window should find a near-silent span somewhere in the reference channel
    w = 320
    frames = mix[1][: len(mix[1]) // w * w].reshape(-1, w)
    assert (np.abs(frames).max(axis=1) < 0.05 * np.abs(mix[1]).max()).any()
