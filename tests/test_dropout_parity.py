"""Reference dropout: the live engine runs the trained ref_policy exactly as the offline (training) path does, so a
model sees at deployment what it learned. The same availability pattern goes through vaani.dsp.pipeline.run /
vaani.data.dataset.front_end + the offline graph run, and through StreamEngine hop by hop, for the refvalid C16 and
an untrained VaaniFE-Mini (inputs pr and pr_nhat). Untrained weights: plumbing and causality, never quality."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from vaani import backend as bk
from vaani import export, live
from vaani.data import dataset
from vaani.dsp import pipeline, stft
from tests.test_export import _refvalid_ckpt
from tests.test_live import _mix

H = live.HOP
ROOT = Path(__file__).resolve().parents[1]


def _recipe(name):
    """(controller_on, dsp) of the r8 recipe, so the test follows the configs the box will train."""
    c = yaml.safe_load((ROOT / "configs/retraining" / f"{name}.yaml").read_text(encoding="utf-8"))
    assert c["dsp"].get("ref_policy") is not None
    return bool(c["controller_on"]), c["dsp"]


RV_CTL, RV_DSP = _recipe("r8_refvalid_v2")
FE_CTL, FE_DSP = _recipe("r8_fe_mini")
N = 40000   # 2.5 s, 156 hops


def _pattern(name):
    """(per-sample availability, what the engine is told: per-sample or per-hop, reference content while absent)."""
    av = np.ones(N, bool)
    if name == "full":
        av[:] = False
    elif name in ("bursts", "bursts_reset"):           # sample-aligned, one inside a hop, one inside the last ramp
        av[7000:11000] = False; av[15000:15100] = False; av[18000:19500] = False; av[19700:20500] = False
    elif name == "reconnect":                          # hop-aligned, told per hop
        av[40 * H:80 * H] = False
    elif name == "start_absent":
        av[:6000] = False
    return av, name == "reconnect", "railed" if name == "bursts" else "zero"


class _OrtSpy(bk.OrtBackend):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.seen = []

    def _step(self, spec6, feats, state, valid=1.0):
        self.seen.append((spec6.copy(), feats.copy(), float(valid)))
        return super()._step(spec6, feats, state, valid)


class _FeSpy(bk.FeOrtBackend):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.seen = []

    def _step(self, spec6, feats, state, valid=1.0):
        self.seen.append((spec6.copy(), feats.copy(), float(valid)))
        return super()._step(spec6, feats, state, valid)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("parity")
    ck = _refvalid_ckpt(d / "rv.pt")
    out = {"rv": (None, export.export_refvalid(ck, d / "rv.onnx"))}
    for inputs in ("pr", "pr_nhat"):
        m = export.fe_untrained({"tier": "mini", "inputs": inputs}, 0)
        out[inputs] = (m, export.export_fe(m, d / f"fe_{inputs}.onnx", parity=False)["folded"])
    return out


def _scene(pattern, seed):
    av, per_hop, absent_ref = _pattern(pattern)
    mix = _mix(N / 16000, seed=seed)
    if absent_ref == "railed":                        # a dead mic railed at full scale: must drive nothing
        mix[1, ~av] = 1.0
    else:                                             # the training fault: an absent reference reads zero
        mix[1, ~av] = 0.0
    return mix, av, per_hop


def _offline(kind, models, mix, av, dsp, ctl):
    """Model inputs (spec (1,257,T,6), feats (T,18), validity (T,)), the model-facing time signals and the
    enhanced output, all by the training path."""
    r = pipeline.run(mix, controller_on=ctl, dsp_cfg=dsp, ref_avail=av)
    m2, fa = r["mix"], r["ref_avail"]
    if kind == "pr":                                  # VaaniFE pr trains on dataset.front_end, not pipeline.run
        m2, fa = dataset.front_end(mix, dsp, av)
        assert np.array_equal(m2, r["mix"]) and np.array_equal(fa, r["ref_avail"])
    x = torch.from_numpy(m2)[None]
    spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1).numpy()
    m, onnx = models[kind]
    sess = export.load_session(onnx)
    if kind == "rv":
        names, zero = export.zero_caches(sess)
        out, _ = export.stream_onnx(sess, spec6, np.ascontiguousarray(r["features"][None], np.float32), names, zero,
                                    ref_avail=fa)
    else:
        out, _ = export.fe_stream_ort(sess, m, np.ascontiguousarray(spec6[..., :m.n_raw]), fa[None].astype(np.float32))
    y = stft.istft(torch.from_numpy(out), length=mix.shape[1])[0].numpy()
    return spec6, r["features"], fa, m2, r["n_hat"], y


def _live(kind, models, mix, av, per_hop, dsp, ctl, lc, **kw):
    _, onnx = models[kind]
    spy = _OrtSpy(onnx) if kind == "rv" else _FeSpy(onnx)
    eng = live.StreamEngine(None, ctl, dsp, backend=spy, left_context=lc, **kw)
    ys, ref_m, nh_m = [], [], []
    for j in range(mix.shape[1] // H):
        s = slice(j * H, (j + 1) * H)
        ys.append(eng.process(mix[0, s], mix[1, s], bool(av[s].all()) if per_hop else av[s]))
        ref_m.append(eng.hist[1].copy()); nh_m.append(eng.hist[2].copy())
    return eng, spy, np.concatenate(ys), np.concatenate(ref_m), np.concatenate(nh_m)


@pytest.mark.parametrize("kind", ["rv", "pr", "pr_nhat"])
@pytest.mark.parametrize("pattern", ["full", "bursts", "bursts_reset", "reconnect", "start_absent"])
def test_live_dropout_policy_is_the_training_policy(models, kind, pattern):
    ctl, dsp = (RV_CTL, RV_DSP) if kind == "rv" else (FE_CTL, FE_DSP)
    if pattern == "bursts_reset":
        dsp = {**dsp, "ref_policy": {**dsp["ref_policy"], "absent": "reset"}}
    mix, av, per_hop = _scene(pattern, seed=31)
    spec6, feats, fa, m2, n_hat, y_off = _offline(kind, models, mix, av, dsp, ctl)
    lc = np.stack([m2[0][256:0:-1], m2[1][256:0:-1], n_hat[256:0:-1]])   # torch's reflect padding of frame 0
    eng, spy, y, ref_m, nh_m = _live(kind, models, mix, av, per_hop, dsp, ctl, lc)
    B = len(spy.seen)
    # the DSP policy is bit-identical: model-facing reference and n_hat, sample for sample
    assert np.array_equal(ref_m, m2[1][:B * H]) and np.array_equal(nh_m, n_hat[:B * H])
    # model inputs: validity exactly, spectra (torch STFT offline vs numpy rfft live) and features within 1e-6
    v = np.array([s[2] for s in spy.seen])
    assert np.array_equal(v, fa[:B])
    n_in = 6 if kind in ("rv", "pr_nhat") else 4
    ls = np.concatenate([s[0][..., :n_in] for s in spy.seen], axis=2)
    os_ = spec6[:, :, :B, :n_in]
    assert np.abs(ls - os_).max() <= 1e-6 * max(1.0, np.abs(os_).max())
    if kind == "rv":
        lf = np.concatenate([s[1][0] for s in spy.seen])
        assert np.abs(lf - feats[:B]).max() <= 1e-6
    # outputs
    yl = y[H:]
    assert np.abs(yl - y_off[:len(yl)]).max() <= 1e-5 and np.abs(y_off).max() > 1e-4
    if pattern != "full":
        assert 0 < v.sum() < B


def test_the_validity_input_is_live(models):
    mix, av, _ = _scene("bursts", seed=32)
    y_drop = _offline("rv", models, mix, av, RV_DSP, RV_CTL)[-1]
    y_all = _offline("rv", models, mix, np.ones(N, bool), RV_DSP, RV_CTL)[-1]
    assert np.abs(y_drop - y_all).max() > 1e-4


def test_all_valid_trained_path_matches_offline_and_ignores_the_policy(models):
    """With every sample valid the ref_policy is a no-op offline (bit-exact) and live."""
    mix, _, _ = _scene("none", seed=33)
    av = np.ones(N, bool)
    a = pipeline.run(mix, controller_on=RV_CTL, dsp_cfg={k: v for k, v in RV_DSP.items() if k != "ref_policy"})
    b = pipeline.run(mix, controller_on=RV_CTL, dsp_cfg={**RV_DSP, "ref_policy": {"absent": "freeze"}}, ref_avail=av)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    spec6, _, fa, m2, n_hat, y_off = _offline("rv", models, mix, av, RV_DSP, RV_CTL)
    lc = np.stack([m2[0][256:0:-1], m2[1][256:0:-1], n_hat[256:0:-1]])
    _, spy, y, _, _ = _live("rv", models, mix, av, True, RV_DSP, RV_CTL, lc)
    assert (fa == 1).all() and np.abs(y[H:] - y_off[:len(y) - H]).max() <= 1e-5


def test_streaming_ref_gain_is_bit_identical_to_ref_gain():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(1, 12000)); av = np.ones(n, bool)
        for _ in range(int(rng.integers(0, 5))):
            a = int(rng.integers(0, n)); av[a:a + int(rng.integers(1, 3000))] = False
        rf = int(rng.integers(1, 16)); blk = int(rng.choice([1, 64, 256, 300]))
        since, prev, out = rf * H, True, []
        for i in range(0, n, blk):
            g, since, prev = pipeline.ref_gain_step(av[i:i + blk], since, prev, rf); out.append(g)
        assert np.array_equal(np.concatenate(out), pipeline.ref_gain(av, rf))


def test_an_absent_reference_drives_nothing_offline_either():
    """A railed dead mic and a zero one give the same offline result: it is zeroed before the limiter."""
    mix, av, _ = _scene("bursts", seed=34)
    zero = mix.copy(); zero[1, ~av] = 0.0
    a = pipeline.run(mix, controller_on=RV_CTL, dsp_cfg=RV_DSP, ref_avail=av)
    b = pipeline.run(zero, controller_on=RV_CTL, dsp_cfg=RV_DSP, ref_avail=av)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert np.array_equal(dataset.front_end(mix, FE_DSP, av)[0], dataset.front_end(zero, FE_DSP, av)[0])


def test_adaptive_filters_freeze_on_absent_hops_only(models):
    mix, av, _ = _scene("reconnect", seed=35)
    eng = live.StreamEngine(None, RV_CTL, RV_DSP, backend=bk.OrtBackend(models["rv"][1]))
    w, gates = [], []
    for j in range(mix.shape[1] // H):
        s = slice(j * H, (j + 1) * H)
        gates.append(eng.gate)                                 # the controller's gate this hop's NLMS runs with
        eng.process(mix[0, s], mix[1, s], bool(av[s].all()))
        w.append((eng.nlms.w.copy(), eng.blk.f.w.copy(), eng.last["ref_weight"]))
    assert eng.nlms.robust is not None                        # ref_policy.nlms: the robust kernel, as offline
    assert all(np.array_equal(w[j][0], w[39][0]) and np.array_equal(w[j][1], w[39][1]) for j in range(40, 80))
    # after the reconnect the policy adds no freeze: the NLMS moves exactly when the controller lets it
    moved = [not np.array_equal(w[j][0], w[j - 1][0]) for j in range(80, len(w))]
    assert moved == [g > 0 for g in gates[80:]] and any(moved)
    ramp = np.array([x[2] for x in w[80:93]])                 # last-sample weight: reaches 1 at the ramp's 12th hop
    assert (ramp[:11] < 1).all() and (np.diff(ramp[:12]) > 0).all() and (ramp[11:] == 1.0).all()


def test_trained_path_state_round_trip_mid_ramp(models):
    mix, av, _ = _scene("bursts", seed=36)
    B = mix.shape[1] // H; k = 45                            # hop 45 sits inside the ramp after the 11000 reconnect
    mk = lambda: live.StreamEngine(None, RV_CTL, RV_DSP, backend=bk.OrtBackend(models["rv"][1]))
    run = lambda e, a, b: np.concatenate([e.process(mix[0, j * H:(j + 1) * H], mix[1, j * H:(j + 1) * H],
                                                    av[j * H:(j + 1) * H]) for j in range(a, b)])
    whole = run(mk(), 0, B)
    e1 = mk(); head = run(e1, 0, k)
    e2 = mk(); e2.import_state(bk.StreamState.from_bytes(e1.export_state().to_bytes()))
    assert np.array_equal(np.concatenate([head, run(e2, k, B)]), whole)


def test_guards_on_the_trained_path_fold_into_availability(models):
    """Guards that do not fire change nothing; a duplicated-mono reference becomes an absent one (validity 0, R = 0)."""
    mix, _, _ = _scene("none", seed=37)
    ones = np.ones(N, bool)
    _, _, y0, _, _ = _live("pr", models, mix, ones, True, FE_DSP, FE_CTL, None)
    _, _, y1, _, _ = _live("pr", models, mix, ones, True, FE_DSP, FE_CTL, None, guards=True)
    assert np.array_equal(y0, y1)
    dup = np.stack([mix[0], mix[0]])
    eng, spy, y, ref_m, _ = _live("pr", models, dup, ones, True, FE_DSP, FE_CTL, None, guards=True)
    v = np.array([s[2] for s in spy.seen]); on = int(np.argmax(v == 0))
    assert v[on:].max() == 0 and 0 < on <= 40 and np.abs(ref_m[on * H:]).max() == 0 and np.isfinite(y).all()


def test_a_skipped_hop_without_reference_runs_the_trained_clock(models):
    """skip(prim, None) is an absent hop to the ramp, the ref_policy reset and the next frame's left half."""
    mix, _, _ = _scene("none", seed=38)
    dsp = {**RV_DSP, "ref_policy": {**RV_DSP["ref_policy"], "absent": "reset"}}
    eng = live.StreamEngine(None, RV_CTL, dsp, backend=bk.OrtBackend(models["rv"][1]))
    for j in range(30):
        eng.process(mix[0, j * H:(j + 1) * H], mix[1, j * H:(j + 1) * H])
    w0 = eng.nlms.w.copy(); assert np.abs(w0).max() > 0
    eng.skip(mix[0, 30 * H:31 * H], None)
    assert np.abs(eng.nlms.w).max() == 0 and np.abs(eng.hist[1]).max() == 0
    av = np.ones(40 * H, bool); av[30 * H:31 * H] = False
    g_ref = pipeline.ref_gain(av, dsp["ref_policy"].get("ramp_frames", pipeline.RAMP_FRAMES))
    ws = []
    for j in range(31, 40):
        eng.process(mix[0, j * H:(j + 1) * H], mix[1, j * H:(j + 1) * H]); ws.append(eng.last["ref_weight"])
    assert ws == [float(g_ref[(j + 1) * H - 1]) for j in range(31, 40)]
