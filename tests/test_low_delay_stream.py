"""Low-delay plan Task 6: the Python streaming reference (vaani.low_delay_live) against the offline route and the ORT
graph; startup and flush, chunk aggregation, reset, state continuation and rejection, interleaved streams,
causality at the release boundary, long streams, the resampler runner and the committed golden vectors."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vaani import audio_contract as ac
from vaani import backend as bk
from vaani import export as E
from vaani import live
from vaani import resampler as rs
from vaani.enhance_low_delay import enhance_low_delay
from vaani.low_delay_live import LowDelayStreamEngine
from vaani.models import vaani_fe as V

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
A, A144, B = ac.ARM_A_IDS[0], ac.ARM_A_IDS[1], ac.ARM_B_ID
DSP = {"limiter": True, "limiter_kernel": "numba", "ref_policy": {"absent": "freeze", "ramp_samples": 3072}}
ATOL, RTOL = 1e-5, 1e-4


def net(cid, seed=0):
    tiling = "p32" if cid == B else "p18"
    return E.fe_untrained({"tier": "mini", "audio_contract": cid, **V.MINI_P[tiling]}, seed)


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    d = tmp_path_factory.mktemp("ld_stream")
    out = {}
    for cid in (A, A144, B):
        m = net(cid)
        out[cid] = (m, E.export_fe(m, d / f"{cid}.onnx", parity=False)["folded"])
    return out


def signal(n=9601, seed=0, gap=True):
    g = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    p = (0.3 * np.sin(2 * np.pi * 150 * t) + g.standard_normal(n) * 0.05).astype(np.float32)
    r = (g.standard_normal(n) * 0.05).astype(np.float32)
    av = np.ones(n, bool)
    if gap:
        av[3000:4200] = False
    return p, r, av


def ort_engine(graphs, cid, **kw):
    return LowDelayStreamEngine(cid, bk.FeOrtBackend(graphs[cid][1], audio_contract=cid), DSP, **kw)


def close(a, b):
    np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("cid", [A, B])
def test_offline_python_and_ort_streams_agree(graphs, cid):
    m, _ = graphs[cid]
    p, r, av = signal()
    off, _ = enhance_low_delay(np.stack([p, r]), av, m, cid, dsp_cfg=DSP)
    y_t = LowDelayStreamEngine(cid, bk.FeTorchBackend(m), DSP).run(p, r, av)
    y_o = ort_engine(graphs, cid).run(p, r, av)
    assert y_t.shape == y_o.shape == (len(p),)
    close(y_t, off)
    close(y_o, off)
    ea, er = np.abs(y_o - off).max(), np.abs(y_o - off).max() / np.abs(off).max()
    assert ea <= ATOL and er <= RTOL                         # absolute and relative reported separately


def test_startup_release_schedule_and_flush(graphs):
    eng = ort_engine(graphs, A)
    c = eng.c
    p, r, av = signal(n=1000, gap=False)
    y = eng.push(p, r, av)
    assert len(y) == (len(p) // c.hop) * c.hop              # one hop out per completed hop in
    tail = eng.flush()
    assert len(tail) == 2 * c.hop                            # the zero-padded partial hop + the flush hop
    full = np.concatenate([y, tail])
    assert np.array_equal(full[c.release_lead:c.release_lead + len(p)], eng.run(p, r, av))
    assert eng.run(p[:0], r[:0]).shape == (0,)


def test_arbitrary_chunk_aggregation(graphs):
    p, r, av = signal()
    ref = ort_engine(graphs, A).run(p, r, av)
    eng = ort_engine(graphs, A)
    g = np.random.default_rng(3)
    out, a = [], 0
    while a < len(p):
        k = int(g.integers(1, 400))
        out.append(eng.push(p[a:a + k], r[a:a + k], av[a:a + k])); a += k
    out.append(eng.flush())
    y = np.concatenate(out)[eng.c.release_lead:eng.c.release_lead + len(p)]
    assert np.array_equal(y, ref)


def test_deterministic_reset(graphs):
    p, r, av = signal()
    eng = ort_engine(graphs, A)
    y1 = eng.run(p, r, av)
    eng.push(p[:5000] * 3, r[:5000], True)                   # dirty every piece of state
    y2 = eng.run(p, r, av)                                   # run() resets first
    assert np.array_equal(y1, y2)


def test_exact_state_continuation(graphs, tmp_path):
    p, r, av = signal()
    whole = ort_engine(graphs, A)
    y_whole = np.concatenate([whole.push(p, r, av), whole.flush()])
    a = ort_engine(graphs, A, guards=True)
    b = ort_engine(graphs, A, guards=True)
    ref = ort_engine(graphs, A, guards=True)
    y_ref = np.concatenate([ref.push(p, r, av), ref.flush()])
    cut = 4507                                               # mid-hop: pending samples are part of the state
    y1 = a.push(p[:cut], r[:cut], av[:cut])
    st = a.export_state()
    st.save(tmp_path / "s.npz")
    b.import_state(bk.StreamState.load(tmp_path / "s.npz"))
    y2 = np.concatenate([b.push(p[cut:], r[cut:], av[cut:]), b.flush()])
    assert np.array_equal(np.concatenate([y1, y2]), y_ref)
    assert len(y_whole) == len(y_ref)


def test_contract_mismatch_is_rejected(graphs):
    a160, a144 = ort_engine(graphs, A), ort_engine(graphs, A144)
    p, r, av = signal(n=2000)
    a160.push(p, r, av); a144.push(p, r, av)
    st = a160.export_state()
    assert set(st.caches) == set(a144.export_state().caches)          # identical layout
    assert st.caches["model/state"].shape == a144.export_state().caches["model/state"].shape   # same neural state
    with pytest.raises(ValueError):
        a144.import_state(st)
    with pytest.raises(ValueError):                                   # the contract is also checked on the backend
        LowDelayStreamEngine(A144, bk.FeOrtBackend(graphs[A][1]), DSP)
    c0 = bk.FeTorchBackend(E.fe_untrained("mini", 0))
    with pytest.raises(ValueError):
        LowDelayStreamEngine(A, c0, DSP)
    with pytest.raises(ValueError):
        LowDelayStreamEngine(ac.LEGACY_ID, c0, DSP)
    other_dsp = LowDelayStreamEngine(A, bk.FeOrtBackend(graphs[A][1]), {**DSP, "limiter": False})
    with pytest.raises(ValueError, match="config_hash"):
        other_dsp.import_state(st)


def test_two_interleaved_streams_share_one_backend(graphs):
    b = bk.FeOrtBackend(graphs[A][1])
    e1, e2 = LowDelayStreamEngine(A, b, DSP), LowDelayStreamEngine(A, b, DSP)
    p1, r1, a1 = signal(seed=1)
    p2, r2, a2 = signal(seed=2, gap=False)
    H = e1.hop
    o1, o2 = [], []
    for j in range(len(p1) // H):
        s = slice(j * H, (j + 1) * H)
        o1.append(e1.process(p1[s], r1[s], a1[s])); o2.append(e2.process(p2[s], r2[s], a2[s]))
    ref1 = ort_engine(graphs, A).push(p1[:len(o1) * H], r1[:len(o1) * H], a1[:len(o1) * H])
    ref2 = ort_engine(graphs, A).push(p2[:len(o2) * H], r2[:len(o2) * H], a2[:len(o2) * H])
    assert np.array_equal(np.concatenate(o1), ref1) and np.array_equal(np.concatenate(o2), ref2)


def test_future_prefix_causality_at_the_release_boundary(graphs):
    p, r, av = signal(n=4800, gap=False)
    H = ort_engine(graphs, A).hop
    y0 = ort_engine(graphs, A).push(p, r, av)
    for t0 in (1000, 2400, 2401):
        q = p.copy(); q[t0:] = np.random.default_rng(t0).standard_normal(len(p) - t0) * 0.5
        y1 = ort_engine(graphs, A).push(q, r, av)
        done = t0 // H                                    # hops fully received before the change
        assert np.array_equal(y0[:done * H], y1[:done * H])
        assert not np.array_equal(y0[done * H:(done + 1) * H], y1[done * H:(done + 1) * H])


def test_long_stream_state_and_queue_stay_bounded(graphs):
    eng = ort_engine(graphs, A, guards=True)
    g = np.random.default_rng(0)
    H = eng.hop
    shapes0 = None
    for j in range(3000):
        if j == 50:                                          # after warm-up (limiter and VAD values exist)
            shapes0 = {k: v.shape for k, v in eng.export_state().caches.items()}
        x = (g.standard_normal((2, H)) * (0.05 if j % 500 else 20.0)).astype(np.float32)
        y = eng.process(x[0], x[1], j % 700 > 30)
        assert np.isfinite(y).all()
    st = eng.export_state()
    assert {k: v.shape for k, v in st.caches.items()} == shapes0
    assert all(np.isfinite(v).all() for v in st.caches.values() if v.dtype.kind == "f")
    assert eng.backend.telemetry.summary()["steps"] == 3000


def test_nonfinite_primary_never_reaches_the_model(graphs):
    eng = ort_engine(graphs, A)
    p, r, av = signal(n=2000, gap=False)
    p[700] = np.nan
    y = np.concatenate([eng.push(p, r, av), eng.flush()])
    assert np.isfinite(y).all()
    assert eng.state.discontinuity_flags & bk.DISC_GAP and eng.state.discontinuity_flags & bk.DISC_RESET


def test_resampler_runner_matches_the_explicit_pipeline(graphs):
    pair = rs.load(rs.IDS[1])
    eng = LowDelayStreamEngine(A, bk.FeOrtBackend(graphs[A][1]), DSP, resampler=pair)
    g = np.random.default_rng(0)
    x48 = (g.standard_normal((2, 3 * 96 * 20)) * 0.05).astype(np.float32)
    got = eng.push(x48[0], x48[1], True)
    x16 = pair.decimator(2)(x48)
    y16 = ort_engine(graphs, A).push(x16[0], x16[1], True)
    ref = pair.interpolator(1)(y16[None])[0]
    assert np.array_equal(got, ref)
    assert "dec/state" in eng.export_state().caches and "interp/state" in eng.export_state().caches


def test_physical_run_engine_dispatches_low_delay(graphs, tmp_path):
    from vaani import physical
    m, onnx = graphs[A]
    cfgp = tmp_path / "model_config.json"
    live.write_fe_model_config(cfgp, onnx, "mini_p18", controller_on=False, dsp=DSP, model_cfg=dict(m.cfg))
    p, r, av = signal(gap=False)
    y, diag = physical.run_engine(np.stack([p, r]), onnx, cfgp)
    assert diag["audio_contract"] == A
    assert np.array_equal(y, ort_engine(graphs, A).run(p, r, av))


def test_eval_ld_onnx_route(graphs, tmp_path):
    from vaani import eval as ev
    m, onnx = graphs[A]
    ck = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "config": {"model": "vaani_fe", "model_cfg": dict(m.cfg), "dsp": DSP,
                                                    "controller_on": False}}, ck)
    f = ev.enhance_fn(f"ld_onnx:{onnx}@{ck}")
    p, r, _ = signal(gap=False)
    y = f(np.stack([p, r]))
    off, _ = enhance_low_delay(np.stack([p, r]), None, m, A, dsp_cfg=DSP)
    close(y, off)


def test_capture_loop_delay_comes_from_the_coefficient_file():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import capture_loop
    assert capture_loop.resampler_delay(3) == 192 == 2 * (len(live.lowpass_fir()) - 1) // 2
    assert capture_loop.resampler_delay(1) == 0
    for rid in rs.IDS[1:]:
        assert capture_loop.resampler_delay(3, rid) == round(rs.load(rid).pair_delay_ms * 48)


# ---- committed golden vectors -------------------------------------------------------------------
VEC = ROOT / "deploy/dsp_reference/vectors_ld"


def _manifest():
    return json.loads((VEC / "manifest.json").read_text())


def test_golden_vector_files_match_the_manifest():
    man = _manifest()
    assert set(man["contracts"]) == {*ac.ARM_A_IDS, ac.ARM_B_ID}
    for cid, e in man["contracts"].items():
        assert e["contract_hash"] == ac.get_audio_contract(cid).contract_hash
        assert bk.file_sha256(VEC / e["model"]["file"]) == e["model"]["sha256"]
        for case in e["cases"].values():
            assert bk.file_sha256(VEC / case["file"]) == case["sha256"]
    for rid, e in man["resampler"].items():
        assert bk.file_sha256(VEC / e["file"]) == e["sha256"] and rs.load(rid).sha256 == e["coef_sha256"]


@pytest.mark.parametrize("cid", [A, B])
def test_engine_reproduces_the_golden_vectors(cid):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_ld_golden_vectors as G
    man = _manifest()
    e = man["contracts"][cid]
    inputs = G.cases(man["contracts"][cid]["seed"])
    for name, meta in e["cases"].items():
        z = np.load(VEC / meta["file"])
        p, r, av = inputs[name]
        assert np.array_equal(z["frontend_in"][:-1].transpose(1, 0, 2).reshape(2, -1)[:, :len(p)],
                              np.stack([p, r]), equal_nan=True)
        eng = LowDelayStreamEngine(cid, bk.FeOrtBackend(VEC / e["model"]["file"], audio_contract=cid), man["dsp"])
        y = np.concatenate([eng.push(p, r, av), eng.flush()])
        np.testing.assert_allclose(y, z["synthesis"].reshape(-1), atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(eng.backend.to_host(eng.state).caches["state"][0], z["final_state"], atol=ATOL, rtol=RTOL)
        assert np.array_equal(z["discontinuity"].any(), name == "nonfinite")


def test_resampler_golden_vectors():
    man = _manifest()
    for rid, e in man["resampler"].items():
        z = np.load(VEC / e["file"])
        pair = rs.load(rid)
        itp, dec = pair.interpolator(2), pair.decimator(2)
        yi, yd, a = [], [], 0
        for b in z["blocks"]:
            yi.append(itp(z["x16"][:, a:a + b])); yd.append(dec(z["x48"][:, 3 * a:3 * (a + b)])); a += b
        np.testing.assert_allclose(np.concatenate(yi, 1), z["interpolated"], atol=1e-7)
        np.testing.assert_allclose(np.concatenate(yd, 1), z["decimated"], atol=1e-7)
