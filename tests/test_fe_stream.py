"""VaaniFE through the streaming contract (spec 6.1 / 6.2 / 6.4): the step graph behind StreamEngine, its validity
input driven by the capture path and the guards, state round trips and shared backends. Untrained weights: these
tests prove plumbing and causality, never quality."""
import json

import numpy as np
import pytest
import torch

from vaani import backend as bk
from vaani import export, live
from vaani.dsp import pipeline, stft
from tests.test_guards import _speechy
from tests.test_live import DSP, _mix

H = live.HOP
R7_ONNX = "deploy/r7/cascade.onnx"


@pytest.fixture(scope="module")
def fe(tmp_path_factory):
    d = tmp_path_factory.mktemp("fe")
    m = export.fe_untrained("mini", 0)
    rep = export.export_fe(m, d / "fe_mini.onnx", parity=False)
    onnx = rep["folded"]
    cfg = d / "model_config.json"                 # beside the graph, so verify_onnx finds its sha256
    live.write_fe_model_config(cfg, onnx, "mini", True, DSP, {"tier": "mini"}, note="untrained, tests only")
    return m, onnx, cfg


class _Spy(bk.FeOrtBackend):
    """Records the validity and the reference spectrum's peak of every step; `vanish` scales the output by 1e-3
    whenever validity is 1 (a model that deletes speech while it trusts the reference)."""

    def __init__(self, *a, vanish=False, **k):
        super().__init__(*a, **k)
        self.seen, self.vanish = [], vanish

    def _step(self, spec6, feats, state, valid=1.0):
        self.seen.append((float(valid), float(np.abs(spec6[0, :, 0, 2:4]).max())))
        y = super()._step(spec6, feats, state, valid)
        return y * 1e-3 if self.vanish and valid == 1.0 else y


def _stream(eng, mix, valid=None):
    B = mix.shape[1] // H
    return np.concatenate([eng.process(mix[0, j * H:(j + 1) * H], mix[1, j * H:(j + 1) * H],
                                       True if valid is None else bool(valid[j])) for j in range(B)])


def test_live_matches_offline_step_graph(fe):
    m, onnx, _ = fe
    mix = _mix()
    r = pipeline.run(mix, controller_on=True, dsp_cfg=DSP)
    x = torch.from_numpy(r["mix"])[None]
    spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1).numpy()
    T = spec6.shape[2]
    out, _ = export.fe_stream_ort(export.load_session(onnx), m, np.ascontiguousarray(spec6[..., :m.n_raw]),
                                  np.ones((1, T), np.float32))
    with torch.no_grad():                         # the training-time offline forward agrees with the step graph
        off = m(torch.from_numpy(spec6)).numpy()
    assert np.abs(out - off).max() <= 1e-5 * max(1.0, np.abs(off).max())
    y_off = stft.istft(torch.from_numpy(out), length=mix.shape[1])[0].numpy()
    lc = np.stack([r["mix"][0][256:0:-1], r["mix"][1][256:0:-1], r["n_hat"][256:0:-1]])
    eng = live.StreamEngine(onnx, True, DSP, left_context=lc)
    assert isinstance(eng.backend, bk.FeOrtBackend) and eng.takes_valid
    y = _stream(eng, mix)[H:]
    assert np.abs(y - y_off[:len(y)]).max() < 1e-5
    assert np.abs(y_off).max() > 1e-3


def test_from_config_picks_the_fe_backend_and_binds_the_graph(fe, tmp_path):
    _, onnx, cfg = fe
    c = live.load_model_config(cfg)
    assert c["kind"] == "vaani_fe" and c["profile"] == "mini"
    eng = live.StreamEngine.from_config(onnx, cfg)
    assert eng.backend.profile_id == "vaani_fe-mini" and eng.backend.kind == "vaani_fe"
    assert {k: v.shape for k, v in eng.state.caches.items()} == {"state": (1, 768)}   # K*F*C2 = 2*16*24
    with pytest.raises(ValueError):
        bk.open_onnx(onnx, kind="cascade")        # a config that names the wrong kind of graph is refused
    bad = json.loads(cfg.read_text()); bad["onnx_sha256"] = "0" * 64
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        live.StreamEngine.from_config(onnx, tmp_path / "bad.json")


def test_r7_graph_still_gets_the_plain_backend():
    b = bk.open_onnx(R7_ONNX)
    assert type(b) is bk.OrtBackend and not b.takes_valid and bk.graph_kind(R7_ONNX) == "cascade"


def test_validity_follows_the_capture_path_and_the_reference_ramps_back(fe):
    _, onnx, _ = fe
    mix = _mix(1.6, seed=4)
    valid = np.ones(mix.shape[1] // H, bool); valid[20:30] = False
    spy = _Spy(onnx)
    eng = live.StreamEngine(None, True, DSP, backend=spy)
    w = []
    for j in range(len(valid)):
        y = eng.process(mix[0, j * H:(j + 1) * H], mix[1, j * H:(j + 1) * H], bool(valid[j]))
        assert np.isfinite(y).all()
        w.append(eng.last["ref_weight"])
    v = np.array([s[0] for s in spy.seen]); rpk = np.array([s[1] for s in spy.seen])
    # frame k spans hops k-1 and k (pipeline.frame_avail): hop 30 is valid but its frame is not
    assert (v[:20] == 1).all() and (v[20:31] == 0).all() and (v[31:] == 1).all()
    assert (rpk[21:30] == 0).all()                # the reference is zeroed, not just masked, while absent
    assert eng.ref_ramp_hops == live.REF_RAMP_HOPS   # DSP has no ref_policy: the engine default
    ramp = np.array(w[30:30 + eng.ref_ramp_hops])
    assert (np.diff(ramp) > 0).all() and ramp[0] < 0.1 and w[30 + eng.ref_ramp_hops] == 1.0
    assert eng.state.discontinuity_flags & bk.DISC_REF_DROPOUT


def test_ramp_length_follows_the_training_ref_policy(fe):
    _, onnx, _ = fe
    eng = live.StreamEngine(onnx, True, {**DSP, "ref_policy": {"ramp_frames": 12}})
    assert eng.ref_ramp_hops == 12


def test_validity_zero_is_the_mono_path(fe):
    _, onnx, _ = fe
    b = bk.FeOrtBackend(onnx)
    rng = np.random.default_rng(0)
    spec = rng.standard_normal((1, 257, 1, 6)).astype(np.float32) * 0.1
    junk = spec.copy(); junk[..., 2:6] = rng.standard_normal((1, 257, 1, 4)).astype(np.float32) * 3
    zero = spec.copy(); zero[..., 2:6] = 0
    f = np.zeros((1, 1, 18), np.float32)
    y0 = b.step(zero, f, b.new_state(), 0.0)
    assert np.array_equal(b.step(junk, f, b.new_state(), 0.0), y0)      # validity 0: the reference cannot matter
    assert not np.array_equal(b.step(junk, f, b.new_state(), 1.0), b.step(zero, f, b.new_state(), 1.0))
    # engine level: a stream whose reference is invalid throughout is independent of what the reference carries
    mix = _mix(0.6, seed=5); other = mix.copy(); other[1] = np.random.default_rng(9).standard_normal(mix.shape[1]) * 0.5
    off = np.zeros(mix.shape[1] // H, bool)
    assert np.array_equal(_stream(live.StreamEngine(onnx, True, DSP), mix, off),
                          _stream(live.StreamEngine(onnx, True, DSP), other, off))


def test_duplicated_mono_guard_drives_validity_zero(fe):
    _, onnx, _ = fe
    v, n = _speechy(2.0, seed=1)
    dup = np.stack([v + n, v + n])
    spy = _Spy(onnx)
    eng = live.StreamEngine(None, True, DSP, backend=spy, guards={"never_vanish": False})
    y = _stream(eng, dup)
    val = np.array([s[0] for s in spy.seen])
    first0 = int(np.argmax(val == 0))
    assert val[0] == 1 and (val[first0:] == 0).all()
    assert first0 <= int(0.5 * live.SR / H) + 2 and np.isfinite(y).all()
    assert eng.telemetry.event_counts["ref_uninformative"] == 1
    # the two-mic mix never trips it, and guards that do not fire leave the stream bit-exact
    mix = _mix(1.5, seed=6)
    spy2 = _Spy(onnx)
    on = _stream(live.StreamEngine(None, True, DSP, backend=spy2, guards={"never_vanish": False}), mix)
    assert all(s[0] == 1 for s in spy2.seen)
    assert np.array_equal(on, _stream(live.StreamEngine(onnx, True, DSP), mix))


def test_never_vanish_guard_falls_back_to_validity_zero(fe):
    _, onnx, _ = fe
    v, n = _speechy(3.0, seed=2)
    mix = np.stack([v + n, 0.3 * v + np.roll(n, 3)])
    spy = _Spy(onnx, vanish=True)
    eng = live.StreamEngine(None, True, DSP, backend=spy, guards={"informativeness": False})
    y = _stream(eng, mix)
    val = np.array([s[0] for s in spy.seen])
    assert eng.telemetry.event_counts["never_vanish"] >= 1
    k = int(np.argmax(val == 0))                  # first hop on the reference-absent path
    assert k > int(0.5 * live.SR / H) and np.isfinite(y).all()
    # on the fallback the (unscaled) mono path is audible again
    assert np.abs(y[(k + 2) * H:(k + 20) * H]).max() > 10 * np.abs(y[(k - 20) * H:(k - 1) * H]).max()


def test_state_round_trip_resumes_bit_exact(fe):
    _, onnx, _ = fe
    mix = _mix(1.6, seed=7)
    valid = np.ones(mix.shape[1] // H, bool); valid[33:38] = False     # save mid-dropout-ramp
    ref = _stream(live.StreamEngine(onnx, True, DSP), mix, valid)
    a = live.StreamEngine(onnx, True, DSP)
    half = 40
    first = _stream(a, mix[:, :half * H], valid[:half])
    st = bk.StreamState.from_bytes(a.export_state().to_bytes())
    assert st.profile_id == a.backend.profile_id and st.caches["model/state"].shape == (1, 768)
    b = live.StreamEngine(onnx, True, DSP)
    b.import_state(st)
    rest = _stream(b, mix[:, half * H:], valid[half:])
    assert np.array_equal(np.concatenate([first, rest]), ref)
    # a state of another tier's shape, or another profile, is refused rather than resized
    with pytest.raises(ValueError):
        b.backend.from_host(bk.StreamState(b.backend.profile_id, "", {"state": np.zeros((1, 3840), np.float32)}))
    with pytest.raises(ValueError):
        b.backend.from_host(bk.StreamState("vaani_fe-mid", "", {"state": np.zeros((1, 768), np.float32)}))


def test_two_interleaved_streams_share_one_backend(fe):
    _, onnx, _ = fe
    for shared in (bk.FeOrtBackend(onnx), bk.FeOrtBackend(onnx, io_binding=True, device="cpu")):
        e1 = live.StreamEngine(None, True, DSP, backend=shared)
        e2 = live.StreamEngine(None, True, DSP, backend=shared)
        a, b = _mix(1.0, seed=13), _mix(1.0, seed=14)
        vb = np.ones(a.shape[1] // H, bool); vb[10:20] = False
        o1, o2 = [], []
        for j in range(a.shape[1] // H):
            o1.append(e1.process(a[0, j * H:(j + 1) * H], a[1, j * H:(j + 1) * H]))
            o2.append(e2.process(b[0, j * H:(j + 1) * H], b[1, j * H:(j + 1) * H], bool(vb[j])))
        assert np.array_equal(np.concatenate(o1), _stream(live.StreamEngine(onnx, True, DSP), a))
        assert np.array_equal(np.concatenate(o2), _stream(live.StreamEngine(onnx, True, DSP), b, vb))


def test_torch_backend_matches_ort_backend(fe):
    m, onnx, _ = fe
    o, t = bk.FeOrtBackend(onnx), bk.FeTorchBackend(m)
    assert t.profile_id == o.profile_id.replace("s768", "mini") == "vaani_fe-mini"
    so, st = o.new_state(), t.new_state()
    rng = np.random.default_rng(3)
    for k in range(40):
        spec = rng.standard_normal((1, 257, 1, 6)).astype(np.float32) * 0.1
        vv = float(k % 13 < 9)
        assert np.abs(o.step(spec, None, so, vv) - t.step(spec, None, st, vv)).max() < 1e-5
    assert np.abs(so.caches["state"] - t.to_host(st).caches["state"]).max() < 1e-5


def test_mono_inputs_graph_has_no_validity(tmp_path):
    m = export.fe_untrained({"tier": "mini", "inputs": "p"}, 0)
    onnx = export.export_fe(m, tmp_path / "p.onnx", parity=False)["folded"]
    eng = live.StreamEngine(onnx, True, DSP)
    eng.process(np.zeros(H), np.zeros(H))
    assert not eng.takes_valid and "validity" not in eng.last
    assert np.isfinite(_stream(eng, _mix(0.5, seed=8), np.zeros(31, bool))).all()
