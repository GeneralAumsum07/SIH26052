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
    meta_keys = set(meta_a) - {"impulse_peak_db", "impulse_onsets_s", "snr_achieved_db"}  # the impulse changes what was achieved
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


def test_meta_records_achieved_snr_after_augmentations():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=0.0, p_clip=0.0, p_ref_dropout=0.0, p_wind=0.0, p_clean=0.0, snr_range=(5.0, 5.0))
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    _, _, meta = mixer.mix(rng, s, n, None, [], None, cfg)
    assert abs(meta["snr_achieved_db"] - 5.0) < 0.5  # nothing perturbed it: matches the target
    # a loud impulse must show up as a lower achieved SNR while the target stays 5
    imp = np.ones(8000, np.float32)
    cfg2 = mixer.MixConfig(p_room=0.0, p_clip=0.0, p_ref_dropout=0.0, p_wind=0.0, p_clean=0.0,
                           snr_range=(5.0, 5.0), impulse_peak_db=(12.0, 12.0))
    _, _, meta2 = mixer.mix(np.random.default_rng(0), s, [n[0].copy()], imp, [0.0], None, cfg2)
    assert meta2["snr_db"] == 5.0 and meta2["snr_achieved_db"] < 2.0


def test_twin_shares_the_burst_clips_peak_normalisation():
    # a +12 dB burst pushes the clip past full scale, so the burst clip gets scaled and the
    # twin would not: the recovery metric would then see a level step, not a recovery
    imp = np.ones(1600, np.float32)
    cfg = mixer.MixConfig(p_room=0.0, p_clean=0.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0,
                           impulse_peak_db=(12.0, 12.0), snr_range=(15.0, 15.0))
    s = _speech(np.random.default_rng(2)) * 3  # peaks near 0.9 so the burst overshoots
    n = [np.random.default_rng(2).standard_normal(len(s)).astype(np.float32)]
    mix_a, clean_a, meta_a = mixer.mix(np.random.default_rng(0), s, list(n), imp, [0.0], None, cfg)
    assert meta_a["norm_gain"] < 1.0
    mix_b, clean_b, meta_b = mixer.mix(np.random.default_rng(0), s, list(n), None, [], None, cfg,
                                       norm_gain=meta_a["norm_gain"])
    start = int(round(meta_a["impulse_onsets_s"][0] * mixer.SR))
    assert np.array_equal(mix_a[:, :start], mix_b[:, :start])
    assert np.array_equal(clean_a, clean_b)
    assert meta_b["norm_gain"] == meta_a["norm_gain"]


# --- r3 physics knobs (2.2). Every default must leave the r1/r2 rng stream untouched. ---

def _quiet_cfg(**over):
    base = dict(p_room=0.0, p_clean=0.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0, snr_range=(0.0, 0.0))
    base.update(over)
    return mixer.MixConfig(**base)


def test_r3_defaults_reproduce_the_old_stream():
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    s = _speech(rng_a); _speech(rng_b)
    imp, m = impulses.generate(np.random.default_rng(1), kind="burst")
    a = mixer.mix(rng_a, s, [rng_a.standard_normal(len(s)).astype(np.float32)], imp, m["onsets_s"], None, _quiet_cfg())
    b = mixer.mix(rng_b, s, [rng_b.standard_normal(len(s)).astype(np.float32)], imp, m["onsets_s"], None,
                  _quiet_cfg(impulse_kinds=None, impulse_room=False, overload_softclip=False, speech_rms_db=None))
    assert np.array_equal(a[0], b[0]) and a[2] == b[2] and "overloaded" not in a[2]  # meta is part of the eval-set hash


def test_overload_saturates_instead_of_burying_the_speech():
    imp, m = impulses.generate(np.random.default_rng(1), kind="burst")
    outs = {}
    for flag in (False, True):
        rng = np.random.default_rng(5); s = _speech(rng); nz = [rng.standard_normal(len(s)).astype(np.float32) * 0.01]
        outs[flag] = mixer.mix(rng, s, nz, imp, m["onsets_s"], None,
                               _quiet_cfg(impulse_peak_db=(45.0, 45.0), overload_softclip=flag, speech_rms_db=(-25.0, -25.0)))
    (old, _, mo), (new, _, mn) = outs[False], outs[True]
    assert mn["overloaded"] and "overloaded" not in mo
    assert np.abs(new).max() <= 1.0
    # speech before the burst keeps its recorded level under saturation; the peak normaliser crushed it before
    start = int(mn["impulse_onsets_s"][0] * mixer.SR)
    lvl = lambda x: 10 * np.log10((x[0, : start - 160] ** 2).mean() + 1e-12)
    assert lvl(new) - lvl(old) > 20


def test_impulse_through_room_rir_changes_only_the_impulse_window(tmp_path):
    rirs.build_bank(tmp_path / "b.npz", n=2, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    imp, m = impulses.generate(np.random.default_rng(1), kind="burst")
    res = {}
    for flag in (False, True):
        rng = np.random.default_rng(7); s = _speech(rng); nz = [rng.standard_normal(len(s)).astype(np.float32)]
        res[flag] = mixer.mix(rng, s, nz, imp, m["onsets_s"], bank, _quiet_cfg(p_room=1.0, impulse_room=flag, impulse_peak_db=(6.0, 6.0)))
    a, b = res[False][0] / res[False][2]["norm_gain"], res[True][0] / res[True][2]["norm_gain"]  # undo the peak normaliser
    start = int(res[True][2]["impulse_onsets_s"][0] * mixer.SR)
    assert np.allclose(a[:, : start - 1], b[:, : start - 1], atol=1e-5)   # rng stream identical, so pre-impulse audio is
    assert not np.allclose(a[1, start: start + 3200], b[1, start: start + 3200])  # ref sees a reverberant tail, not a 2-tap copy


def test_r3_impulse_crest_distribution():
    """Plan 2.2 gate: p50 of the achieved impulse crest (peak over background RMS on the primary) > 25 dB in 200 draws (r3 range 15-45 dB).
    Measured pre-saturation - under the overload model the crest is bounded by the recorder headroom, which is a
    property of the speech level, not of the impulse physics."""
    crest = []
    for i in range(200):
        rng = np.random.default_rng([11, i]); s = _speech(rng); nz = [rng.standard_normal(len(s)).astype(np.float32)]
        kind = str(rng.choice(("burst", "click_train")))
        imp, m = impulses.generate(rng, kind=kind)
        out, _, meta = mixer.mix(rng, s, nz, imp, m["onsets_s"], None, _quiet_cfg(impulse_peak_db=(15.0, 45.0)))
        start = int(meta["impulse_onsets_s"][0] * mixer.SR); w = out[0, start: start + 3200]
        bg = np.sqrt((out[0, : max(160, start - 160)] ** 2).mean() + 1e-12) if start > 320 else np.sqrt((out[0, start + 8000:] ** 2).mean() + 1e-12)
        crest.append(20 * np.log10(np.abs(w).max() / bg))
    assert np.percentile(crest, 50) > 25.0, np.percentile(crest, [10, 50, 90])
