"""Mixer v2 physics (plan 11.5 M1-M4, M7, M10, M11; verification 11.8) and v1 bit-exactness."""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import csd, welch

from vaani.data import calib, mixer, scenes

SR = 16000
REPO = Path(__file__).resolve().parents[1]
PRE_V2_COMMIT = "619205c"   # last commit before mixer v2: its mix() is the v1 reference


def _speech(n=4 * SR, f0=180.0):
    t = np.arange(n) / SR
    env = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)
    return (np.sin(2 * np.pi * f0 * t) * env * 0.3 + 0.05 * np.sin(2 * np.pi * 2 * f0 * t) * env).astype(np.float32)


def _real_coherence(a, b, nper=1024):
    f, sab = csd(a, b, fs=SR, nperseg=nper)
    _, saa = welch(a, fs=SR, nperseg=nper); _, sbb = welch(b, fs=SR, nperseg=nper)
    return f, np.real(sab) / np.sqrt(saa * sbb)


# --- M1 / M7: SPL chain ---

def test_spl_round_trip_and_datasheet_point():
    x = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    for spl in (40.0, 70.0, 94.0, 110.0, 120.0):
        y = calib.scale_to_spl(x, spl, "rms")
        assert abs(calib.float_rms_db_to_spl(calib.rms_db(y)) - spl) < 0.01
    y94 = calib.scale_to_spl(x, 94.0, "rms")
    assert abs(20 * np.log10(np.abs(y94).max()) + 26.0) < 0.05          # -26 dBFS sine at 94 dB SPL
    assert abs(float(calib.float_to_pascal(1.0)) - 28.3) < 0.1           # full scale = 28.3 Pa = 123 dB peak


def test_effort_classes_and_front_end():
    assert calib.effort_class(0) == "normal" and calib.effort_class(6) == "raised"
    assert calib.effort_class(12) == "loud" and calib.effort_class(20) == "shout"
    rng = np.random.default_rng(0)
    x = np.stack([np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)] * 2).astype(np.float32) * 1.5
    y, meta = calib.front_end_nonlinear(rng, x)
    assert np.abs(y).max() <= 1.0 + 1e-3 and meta["saturated"] and meta["clip_frac"] > 0
    lo = calib.front_end_linear(np.stack([np.sin(2 * np.pi * 20 * np.arange(SR) / SR)] * 2), [0.0, 0.0])
    assert calib.rms_db(lo[:, SR // 2:]) < calib.rms_db(np.sin(2 * np.pi * 20 * np.arange(SR) / SR)) - 15   # 60 Hz HPF


def test_lombard_tilt_raises_alpha_ratio_keeps_level():
    x = np.random.default_rng(0).standard_normal(2 * SR).astype(np.float32)

    def alpha(v):
        f, p = welch(v, fs=SR, nperseg=1024)
        return 10 * np.log10(p[(f >= 1000) & (f <= 5000)].sum() / p[(f >= 50) & (f < 1000)].sum())
    y = calib.lombard_tilt(x)
    assert 3.5 < alpha(y) - alpha(x) < 5.5
    assert abs(calib.rms_db(y) - calib.rms_db(x)) < 0.01


# --- M3: diffuse coherence ---

@pytest.mark.parametrize("f0,target", [(200.0, 0.97), (1000.0, 0.37), (1429.0, 0.0)])
def test_diffuse_coherence_curve(f0, target):
    assert abs(float(mixer.diffuse_coherence(f0, 0.12)) - target) < 0.02   # the analytic target itself
    rng = np.random.default_rng(1)
    x = rng.standard_normal(30 * SR).astype(np.float32)
    pair = mixer.diffuse_pair(rng, x, d=0.12)
    f, g = _real_coherence(pair[0], pair[1])
    k = int(np.argmin(np.abs(f - f0)))
    assert abs(g[k] - target) < 0.08, f"coherence {g[k]:.2f} at {f[k]:.0f} Hz, want {target}"


def test_point_and_near_sources_have_ild_of_either_sign():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2 * SR).astype(np.float32)
    for ild in (-12.0, 12.0):
        pair = mixer._point_pair(rng, x, ild, 0.12)
        assert abs(calib.rms_db(pair[0]) - calib.rms_db(pair[1]) - ild) < 0.3


# --- M4: wind ---

def test_wind_independent_per_mic_with_shared_envelope():
    rng = np.random.default_rng(3)
    w, meta = mixer.wind_pair(rng, 20 * SR, speed_mps=10.0, ref_db=85.0)
    f, g = _real_coherence(w[0], w[1])
    assert np.abs(g[(f > 100) & (f < 1000)]).mean() < 0.1               # independent turbulence
    fr = 320
    e = [10 * np.log10((w[m][: len(w[m]) // fr * fr].reshape(-1, fr) ** 2).mean(1) + 1e-20) for m in range(2)]
    assert np.corrcoef(e[0], e[1])[0, 1] > 0.6                         # shared gust envelope
    assert abs(meta["wind_spl_db"] - (85.0 + 12.04)) < 0.01            # +12 dB per doubling of speed
    f, p = welch(w[0], fs=SR, nperseg=1024)
    assert p[f < 1000].sum() / p.sum() > 0.95                          # energy below 1 kHz


# --- M2: reference speech gain histogram ---

@pytest.mark.parametrize("share", [0.0, 0.10, 0.25])
def test_m2_reference_gain_histogram(share):
    rng = np.random.default_rng(4)
    p = {**mixer.V2_DEFAULTS, "tail_share": share}
    d = [mixer.mix_v2_ref_draw(rng, p) for _ in range(20000)]
    g = np.asarray([x[1] for x in d]); modes = np.asarray([x[0] for x in d])
    assert g.min() >= -20.0 - 1e-9 and g.max() <= 3.0 + 1e-9
    band = ((g >= -6) & (g <= 3)).mean()
    assert abs(band - share) < 0.015
    if share == 0.25:
        assert abs((modes == "mono").mean() - 0.10) < 0.01 and abs((modes == "stereo").mean() - 0.05) < 0.01
    phys = g[modes == "physical"]
    assert ((phys <= -8.5) & (phys >= -20.0)).all()


def test_mix_v2_reference_gain_matches_draw_and_mono_is_duplicated():
    s = _speech()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        sc = scenes.sample_scene(rng, "command_post", crop_s=4.0)
        noises = [rng.standard_normal(3 * SR).astype(np.float32) for _ in sc["sources"]]
        m, clean, meta = mixer.mix(rng, s, noises, None, [], None, mixer.MixConfig(version=2, p_clean=0.0), scene=sc)
        assert m.shape == (2, len(s)) and clean.shape == (len(s),) and np.isfinite(m).all()
        if meta["ref_mode"] == "mono":
            assert np.array_equal(m[0], m[1])
        elif meta["ref_mode"] == "physical":
            assert -20.5 <= meta["ref_speech_gain_db"] <= -8.0


# --- M1: SNR is an output of the scene SPLs ---

def test_snr_follows_scene_levels():
    rng = np.random.default_rng(5)
    s = _speech()
    sc = dict(name="t", rir="outdoor", speech_spl=90.0, effort="normal", lombard=False, wind_mps=0.0, event=None,
              sources=[dict(role="bed", tags=["general"], spl=80.0, weighting="rms")])
    cfg = mixer.MixConfig(version=2, p_clean=0.0, v2={"front_end": False, "tail_share": 0.0})
    m, clean, meta = mixer.mix(rng, s, [rng.standard_normal(5 * SR).astype(np.float32)], None, [], None, cfg, scene=sc)
    assert abs(calib.float_rms_db_to_spl(calib.active_rms_db(clean)) - 90.0) < 0.05
    assert abs(calib.float_rms_db_to_spl(calib.rms_db(m[0] - clean)) - 80.0) < 0.2
    assert abs(meta["snr_db"] - 10.0) < 0.5


def test_unknown_v2_key_and_bad_version_raise():
    with pytest.raises(ValueError):
        mixer.v2_params(mixer.MixConfig(version=2, v2={"no_such": 1}))
    with pytest.raises(ValueError):
        mixer.mix(np.random.default_rng(0), _speech(), [], None, [], None, mixer.MixConfig(version=3))


# --- M10: seams ---

def test_crossfaded_loop_has_no_seam_step():
    rng = np.random.default_rng(6)
    x = (np.sin(2 * np.pi * 50 * np.arange(3000) / SR) + 1.0).astype(np.float32)   # DC offset: a hard seam would jump
    y = mixer.fit_xfade(x, 4 * SR, rng, int(0.05 * SR))
    assert len(y) == 4 * SR and np.abs(np.diff(y)).max() < 0.1
    c = mixer.concat_xfade([x, x[::-1]], 400)
    assert len(c) == 2 * len(x) - 400 and np.isfinite(c).all()


# --- v1 bit-exactness against the pre-v2 commit ---

def _pre_v2_mixer(tmp_path):
    try:
        src = subprocess.run(["git", "show", f"{PRE_V2_COMMIT}:vaani/data/mixer.py"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout
    except Exception:   # noqa: BLE001 - no git or a shallow clone: nothing to compare against
        pytest.skip("pre-v2 mixer not available from git")
    p = tmp_path / "mixer_v1_ref.py"; p.write_text(src)
    spec = importlib.util.spec_from_file_location("mixer_v1_ref", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_v1_default_is_bit_exact_with_pre_v2_mixer(tmp_path):
    from vaani.data import impulses, rirs
    ref = _pre_v2_mixer(tmp_path)
    rirs.build_bank(tmp_path / "b.npz", n=2, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    s = _speech()
    kw = [dict(), dict(p_room=1.0), dict(p_clip=1.0, p_wind=1.0, p_ref_dropout=1.0), dict(speech_rms_db=(-30.0, -20.0)),
          dict(impulse_room=True, overload_softclip=True, p_room=1.0)]
    for i, k in enumerate(kw):
        for seed in range(3):
            args = []
            for mod in (mixer, ref):
                rng = np.random.default_rng(100 * i + seed)
                noise = [rng.standard_normal(3 * SR).astype(np.float32)]
                imp, im = impulses.generate(rng); on = im["onsets_s"]
                args.append(mod.mix(rng, s, noise, imp, on, bank, mod.MixConfig(**k)))
            (a_m, a_c, a_meta), (b_m, b_c, b_meta) = args
            assert np.array_equal(a_m, b_m) and np.array_equal(a_c, b_c), (k, seed)
            assert a_meta == b_meta


# --- M8: scene-driven sampling ---

def test_scene_weights_and_draw_frequencies():
    assert abs(sum(scenes.SCENE_WEIGHTS.values()) - 1.0) < 1e-9 and set(scenes.SCENE_WEIGHTS) == set(scenes.SCENES)
    rng = np.random.default_rng(7)
    names = [scenes.sample_scene(rng)["name"] for _ in range(8000)]
    for k, w in scenes.SCENE_WEIGHTS.items():
        assert abs(names.count(k) / len(names) - w) < 0.02, k
    sc = scenes.sample_scene(np.random.default_rng(0), "apc")
    assert sc["rir"] == "armoured" and sc["effort"] != "normal" and sc["lombard"]
    assert all(s["role"] in ("bed", "point", "near") for s in sc["sources"])


def test_scene_pool_weights_classes_not_rows_and_drops_v1_only_corpora():
    import pandas as pd
    rows = [dict(corpus="mad", source_id=f"mad:helicopter/v{i % 50}", group_id=f"mad:v{i % 50}", noise_class="continuous",
                 path=f"h{i}.wav") for i in range(1000)]
    rows += [dict(corpus="mad", source_id="mad:shooting/v0", group_id="mad:s0", noise_class="continuous", path="g.wav")]
    rows += [dict(corpus="drone", source_id="drone:x", group_id="drone:x", noise_class="continuous", path="d.wav")]
    pool = scenes.ScenePool(pd.DataFrame(rows))
    assert "drone" not in pool.tags() and pool.tags()["helicopter"] == 1000
    rng = np.random.default_rng(8)
    sc = scenes.sample_scene(rng, "firefight", p_near=0.0)
    got = [pool.draw(rng, sc)[0][0]["path"] for _ in range(200)]
    assert got.count("g.wav") == 200            # the gunfire bed comes from its one row, not the 1000-row class
    sc = scenes.sample_scene(rng, "drone", p_near=0.0)
    r, _ = pool.draw(rng, sc)
    assert sc["sources"][-1]["tag"] in ("fallback", None) or r[-1] is None or r[-1]["corpus"] != "drone"
