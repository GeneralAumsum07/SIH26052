"""The streaming contract (scalable-mini spec 6.1 / 6.4 / 8) through the real per-hop engine and streaming ONNX."""
from pathlib import Path

import numpy as np
import pytest

from vaani import backend as bk
from vaani import live
from tests.test_live import DSP, _graph, _mix, _offline

H = live.HOP
R7 = Path(__file__).resolve().parents[1] / "deploy" / "r7"


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    return _graph(tmp_path_factory.mktemp("sc"))


def _stream(eng, mix, valid=None):
    """Hop-by-hop output, NOT shifted (index i is output time i - H)."""
    B = mix.shape[1] // H
    return np.concatenate([eng.process(mix[0, j * H:(j + 1) * H], mix[1, j * H:(j + 1) * H],
                                       True if valid is None else bool(valid[j])) for j in range(B)])


def test_causal_under_future_perturbation(graph):
    _, onnx = graph
    mix = _mix(1.5, seed=10)
    base = _stream(live.StreamEngine(onnx, True, DSP), mix)
    p = 40 * H + 17                                          # perturb everything from sample p on
    pert = mix.copy()
    pert[:, p:] += np.random.default_rng(1).standard_normal((2, mix.shape[1] - p)).astype(np.float32) * 0.2
    y = _stream(live.StreamEngine(onnx, True, DSP), pert)
    safe = (p // H) * H                                      # calls before block p//H never saw the change
    assert np.array_equal(y[:safe], base[:safe])
    assert np.abs(y[safe:] - base[safe:]).max() > 1e-4       # and the change does reach the output


def test_reset_equals_fresh_engine_and_replays_nothing(graph):
    _, onnx = graph
    a, b = _mix(1.0, seed=11), _mix(1.0, seed=12)
    eng = live.StreamEngine(onnx, True, DSP)
    _stream(eng, a)
    eng.reset()
    assert eng.state.discontinuity_flags & bk.DISC_RESET and eng.state.sample_counter == 0
    after = _stream(eng, b)
    assert np.array_equal(after, _stream(live.StreamEngine(onnx, True, DSP), b))
    eng.reset()
    z = eng.process(np.zeros(H), np.zeros(H))                # first hop after reset: nothing of `b` comes back
    assert not np.any(z)


def test_two_interleaved_streams_share_a_backend(graph):
    _, onnx = graph
    shared = bk.OrtBackend(onnx)
    e1, e2 = live.StreamEngine(None, True, DSP, backend=shared), live.StreamEngine(None, True, DSP, backend=shared)
    a, b = _mix(1.0, seed=13), _mix(1.0, seed=14)
    o1, o2 = [], []
    for j in range(a.shape[1] // H):
        o1.append(e1.process(a[0, j * H:(j + 1) * H], a[1, j * H:(j + 1) * H]))
        o2.append(e2.process(b[0, j * H:(j + 1) * H], b[1, j * H:(j + 1) * H]))
    assert np.array_equal(np.concatenate(o1), _stream(live.StreamEngine(onnx, True, DSP), a))
    assert np.array_equal(np.concatenate(o2), _stream(live.StreamEngine(onnx, True, DSP), b))


def test_chunk_boundaries_are_invisible(graph):
    _, onnx = graph
    mix = _mix(1.0, seed=15)
    want = _stream(live.StreamEngine(onnx, True, DSP), mix)
    eng, fr, out, pos = live.StreamEngine(onnx, True, DSP), live.HopFramer(2), [], 0
    rng = np.random.default_rng(2)
    while pos < mix.shape[1]:
        n = int(rng.integers(1, 700))
        for hop in fr.push(mix[:, pos:pos + n]):
            out.append(eng.process(hop[0], hop[1]))
        pos += n
    assert np.array_equal(np.concatenate(out), want)


def test_long_silence_stays_silent_and_recovers(graph):
    _, onnx = graph
    eng = live.StreamEngine(onnx, True, DSP)
    z = np.zeros((2, 20 * 16000), np.float32)                # 20 s of digital silence
    y = _stream(eng, z)
    assert np.isfinite(y).all() and np.abs(y).max() < 1e-6
    x = _mix(1.0, seed=16)
    y2 = _stream(eng, x)
    assert np.isfinite(y2).all() and np.abs(y2).max() > 1e-3


def test_state_round_trip_resumes_the_whole_engine(graph, tmp_path):
    _, onnx = graph
    mix = _mix(1.5, seed=17)
    whole = _stream(live.StreamEngine(onnx, True, DSP), mix)
    k = 30
    e1 = live.StreamEngine(onnx, True, DSP)
    head = _stream(e1, mix[:, :k * H])
    e1.export_state().save(tmp_path / "st.npz")
    e2 = live.StreamEngine(onnx, True, DSP)
    e2.import_state(bk.StreamState.load(tmp_path / "st.npz"))
    tail = _stream(e2, mix[:, k * H:])
    assert np.array_equal(np.concatenate([head, tail]), whole)
    with pytest.raises(ValueError, match="config_hash"):
        live.StreamEngine(onnx, False, DSP).import_state(e1.export_state())


def test_all_valid_default_path_matches_offline(graph):
    _, onnx = graph
    mix = _mix(2.0, seed=18)
    r, y_off = _offline(onnx, mix)
    lc = np.stack([r["mix"][0][256:0:-1], r["mix"][1][256:0:-1], r["n_hat"][256:0:-1]])
    y = _stream(live.StreamEngine(onnx, True, DSP, left_context=lc), mix, np.ones(mix.shape[1] // H))[H:]
    assert np.abs(y - y_off[:len(y)]).max() < 1e-5 and np.abs(y_off).max() > 1e-3


@pytest.mark.skipif(not (R7 / "cascade.onnx").exists(), reason="shipped r7 graph absent")
def test_r7_all_valid_parity_and_hash_bound():
    from tests.test_r7_artifact import _offline as r7_off, _voiced_mix
    cfg = live.load_model_config(R7 / "model_config.json")
    assert cfg["onnx_sha256"] == bk.file_sha256(R7 / "cascade.onnx")
    _, mix = _voiced_mix(2.0)
    r, y_off = r7_off(mix, cfg)
    lc = np.stack([r["mix"][0][256:0:-1], r["mix"][1][256:0:-1], r["n_hat"][256:0:-1]])
    eng = live.StreamEngine.from_config(R7 / "cascade.onnx", R7 / "model_config.json", left_context=lc)
    y = _stream(eng, mix, np.ones(mix.shape[1] // H))[H:]
    assert np.abs(y - y_off[:len(y)]).max() < 1e-6          # measured 5.96e-8 (plan A6)


def test_hash_mismatch_is_refused_unless_overridden(graph, tmp_path):
    _, onnx = graph
    with pytest.raises(ValueError, match="does not match"):
        live.StreamEngine(onnx, True, DSP, onnx_sha256="0" * 64)
    with pytest.warns(UserWarning):
        live.StreamEngine(onnx, True, DSP, onnx_sha256="0" * 64, allow_hash_mismatch=True)
    (Path(onnx).parent / "model_config.json").write_text('{"controller_on": true, "onnx_sha256": "' + "1" * 64 + '"}')
    try:
        with pytest.raises(ValueError, match="model_config.json"):
            live.StreamEngine(onnx, True, DSP)                # the sidecar is checked automatically
    finally:
        (Path(onnx).parent / "model_config.json").unlink()


def test_reference_dropout_policy_and_reconnect_ramp(graph):
    _, onnx = graph
    mix = _mix(3.0, seed=19)
    B = mix.shape[1] // H
    valid = np.ones(B, bool); valid[60:120] = False
    ok = _stream(live.StreamEngine(onnx, True, DSP), mix)
    eng = live.StreamEngine(onnx, True, DSP, ref_ramp_hops=8)
    out, w, weights = [], [], None
    for j in range(B):
        if j == 60:
            weights = eng.nlms.w.copy()
        out.append(eng.process(mix[0, j * H:(j + 1) * H], mix[1, j * H:(j + 1) * H], bool(valid[j])))
        w.append(eng.last["ref_weight"])
        if j == 119:
            assert np.array_equal(eng.nlms.w, weights)       # adaptation frozen for the whole dropout
    y = np.concatenate(out)
    assert np.isfinite(y).all() and np.abs(y).max() <= 2 * np.abs(mix[0]).max()   # untrained weights: bounded, not quiet
    assert np.array_equal(y[:60 * H], ok[:60 * H])           # nothing changes before the dropout
    w = np.array(w)
    assert (w[60:120] == 0).all() and (w[:60] == 1).all() and (w[128:] == 1).all()
    assert (np.diff(w[119:128]) > 0).all()                   # linear ramp back over 8 hops
    assert eng.ref_invalid_hops == 60 and eng.state.discontinuity_flags & bk.DISC_REF_DROPOUT
    # a railed / dead reference drives nothing: the invalid hops' reference content is irrelevant
    mix2 = mix.copy(); mix2[1, 60 * H:120 * H] = 1.0
    y2 = _stream(live.StreamEngine(onnx, True, DSP, ref_ramp_hops=8), mix2, valid)
    assert np.array_equal(y2, y)


def test_overload_queue_drops_oldest_and_skip_never_reemits(graph):
    q = live.BoundedHopQueue(3)
    for i in range(5):
        q.push(np.full((2, H), i, np.float32))
    assert q.dropped_hops == 2 and q.dropped_samples == 2 * H
    got = [q.pop(0) for _ in range(3)]
    assert [int(b[0][0, 0]) for b in got] == [2, 3, 4] and got[0][1] == 2 and got[1][1] == 0
    _, onnx = graph
    mix = _mix(1.0, seed=20)
    eng = live.StreamEngine(onnx, True, DSP)
    _stream(eng, mix[:, :20 * H])
    eng.skip()                                               # dropped input: the pending OLA half is abandoned
    assert not np.any(eng.ola) and eng.state.discontinuity_flags & bk.DISC_GAP and eng.skipped == 1
    ola_before = eng.ola.copy()
    z = eng.process(np.zeros(H), np.zeros(H))
    assert not np.any(ola_before) and np.isfinite(z).all()  # only the new frame's first half is emitted
    assert eng.state.sample_counter == 22 * H


def test_ms_is_a_bounded_ring(graph):
    _, onnx = graph
    eng = live.StreamEngine(onnx, True, DSP, ms_window=5)
    _stream(eng, _mix(0.5, seed=21))
    assert len(eng.ms) == 5 and eng.telemetry.count == _mix(0.5).shape[1] // H
