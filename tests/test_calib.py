"""SPL level chain of mixer v2 (results_r2/r8/calib): A vs Z weighting, HPF order, front-end curves, level flags."""
import numpy as np
import pytest

from vaani.data import calib, mixer, scenes

SR = 16000
T = np.arange(2 * SR) / SR


def _sine(hz, spl):
    return calib.scale_to_spl(np.sin(2 * np.pi * hz * T), spl, "rms")


def test_a_weighting_of_an_lf_bed_reads_low_so_its_unweighted_level_is_higher():
    rng = np.random.default_rng(0)
    bed = np.sin(2 * np.pi * 40 * T) + 0.1 * rng.standard_normal(len(T))       # LF-heavy: a 40 Hz line over hiss
    y = calib.scale_to_spl(bed, 100.0, "A")
    assert abs(calib.float_rms_db_to_spl(calib.a_weighted_rms_db(y)) - 100.0) < 0.01
    z = float(calib.float_rms_db_to_spl(calib.rms_db(y)))
    assert z - 100.0 > 10.0                     # A(40 Hz) = -34.6 dB, so the dBA target puts Z far above it
    assert abs(calib.a_weight_db(40.0) + 34.6) < 0.1 and abs(calib.a_weight_db(1000.0)) < 0.01


def test_hpf_before_the_saturator_lowers_lf_peaks_and_post_order_keeps_them():
    x = np.stack([_sine(30.0, 121.0)] * 2)      # 30 Hz at 121 dB: peaks past the rails before any HPF
    pre = calib.front_end_linear(x, [0.0, 0.0])
    assert np.abs(x).max() >= 1.0 and np.abs(pre[:, SR // 2:]).max() < 0.35   # 2nd order at half the corner: about -12 dB
    assert calib.level_flags(x)["past_rails"] and not calib.level_flags(pre[:, SR // 2:])["past_rails"]


def test_curves_hit_their_datasheet_points():
    assert abs(calib.thd_of(105.0) - 0.002) < 2e-4                # knee105: 0.2 % THD at 105 dB (DS-000069 typ)
    assert abs(calib.thd_of(120.0, "tanh120") - 0.10) < 2e-3      # tanh120: 10 % THD at the 120 dB AOP
    assert calib.thd_of(120.0) < 0.08 and calib.thd_of(105.0, "tanh120") < 0.01   # each misses the other point, within max
    assert abs(20 * np.log10(calib.soft_knee()) + calib.SPL_TO_FLOAT_RMS_DB - 90.7) < 0.1
    with pytest.raises(ValueError):
        calib.saturate_curve(np.zeros(4), "nope")


def test_level_flags_thresholds():
    f = calib.level_flags(np.stack([_sine(1000.0, 94.0)] * 2))
    assert f["past_knee"] and not f["past_aop"] and not f["past_rails"] and abs(f["peak_db_spl"] - 97.0) < 0.1
    f = calib.level_flags(_sine(1000.0, 119.5)[None])
    assert f["past_knee"] and not f["past_aop"] and not f["past_rails"]
    f = calib.level_flags(_sine(1000.0, 120.5)[None])
    assert f["past_aop"] and f["past_rails"]                      # a sine at the AOP peaks at the rails
    rng = np.random.default_rng(1)
    g = calib.scale_to_spl(rng.standard_normal(len(T)), 112.0, "rms")[None]   # noise: crest ~13 dB, rails before AOP
    f = calib.level_flags(g)
    assert f["past_rails"] and not f["past_aop"]
    assert calib.level_flags(_sine(1000.0, 80.0)[None], fs_offset_db=10.0)["peak_db_spl"] > 92.9


def _mix(v2=None, seed=3, bed_spl=112.0):
    rng = np.random.default_rng(seed)
    sc = dict(name="t", rir="outdoor", speech_spl=100.0, effort="loud", lombard=False, wind_mps=0.0, event=None,
              sources=[dict(role="bed", tags=["general"], spl=bed_spl, weighting="A")])
    s = _sine(200.0, 100.0) * (np.sin(2 * np.pi * 2 * T) > 0)
    nz = rng.standard_normal(3 * SR).astype(np.float32) + 3 * np.sin(2 * np.pi * 35 * np.arange(3 * SR) / SR).astype(np.float32)
    cfg = mixer.MixConfig(version=2, p_clean=0.0, v2={"tail_share": 0.0, **(v2 or {})})
    return mixer.mix(rng, s.astype(np.float32), [nz], None, [], None, cfg, scene=sc)


def test_mix_v2_emits_explicit_flags_and_keeps_overloaded():
    m, c, meta = _mix()
    for k in ("past_knee", "past_aop", "past_rails", "peak_db_spl", "overloaded", "clipped"):
        assert k in meta
    assert meta["overloaded"] == meta["past_knee"] and meta["clipped"] == meta["past_rails"]
    m2, c2, meta2 = _mix(bed_spl=60.0)
    assert not meta2["past_rails"] and not meta2["past_aop"] and meta2["peak_db_spl"] < meta["peak_db_spl"]


def test_front_end_options():
    _, _, pre = _mix()
    _, _, post = _mix({"fe_hpf_order": "post"})
    assert post["peak_db_spl"] > pre["peak_db_spl"]              # the converter sees the 35 Hz line under "post"
    m0, c0, _ = _mix()
    m1, c1, q = _mix({"mic_fs_spl_db": 130.0})
    assert abs(calib.rms_db(c1) - calib.rms_db(c0) + 10.0) < 0.01  # same SPL, 10 dB less float level
    assert q["peak_db_spl"] == pytest.approx(pre["peak_db_spl"], abs=0.01) and not q["past_rails"]
    m2, _, _ = _mix({"fe_curve": "tanh120"})
    assert not np.array_equal(m2, m0) and np.abs(m2).max() <= calib.tanh_scale() + 1e-3
    with pytest.raises(ValueError):
        _mix({"fe_hpf_order": "sideways"})


def test_near_split_mode_holds_the_scene_level():
    for seed in range(40):
        a = scenes.sample_scene(np.random.default_rng(seed), "helicopter", near_mode="add")
        b = scenes.sample_scene(np.random.default_rng(seed), "helicopter", near_mode="split")
        near = [s["spl"] for s in b["sources"] if s["role"] == "near"]
        if not near:
            assert a == b; continue
        tot = 10 * np.log10(sum(10 ** (s["spl"] / 10) for s in b["sources"] if s["role"] in ("bed", "near")))
        assert abs(tot - a["sources"][0]["spl"]) < 1e-6          # bed + near = the drawn table level
        assert abs((near[0] - b["sources"][0]["spl"]) - ([s["spl"] for s in a["sources"] if s["role"] == "near"][0]
                                                          - a["sources"][0]["spl"])) < 1e-9
    with pytest.raises(ValueError):
        scenes.sample_scene(np.random.default_rng(0), "helicopter", p_near=1.0, near_mode="x")
