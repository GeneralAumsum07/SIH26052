"""vaani.backend: the versioned stream state, the step backends and their telemetry (spec 6.1 / 6.4)."""
import numpy as np
import pytest
import torch

from vaani import backend as bk
from vaani import export
from tests.test_live import _graph


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    return _graph(tmp_path_factory.mktemp("bk"))


def _inputs(T=24, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, 257, T, 6)).astype(np.float32) * 0.1,
            rng.standard_normal((1, T, 18)).astype(np.float32) * 0.1)


def _run(b, st, spec, feats):
    return np.concatenate([b.step(spec[:, :, t:t + 1], feats[:, t:t + 1], st) for t in range(spec.shape[2])], axis=2)


def test_ort_backend_is_the_reference_stream_loop(graph):
    _, onnx = graph
    spec, feats = _inputs()
    ref, _ = export.stream_onnx(export.load_session(onnx), spec, feats)
    b = bk.OrtBackend(onnx)
    assert np.array_equal(_run(b, b.new_state(), spec, feats), ref)   # bit-exact: the same session call
    assert b.telemetry.count == spec.shape[2] and b.tested


def test_io_binding_path_matches_plain_path_on_cpu(graph):
    # the device-resident code path; only its CPU device is testable here (no CUDA/TensorRT EP installed)
    _, onnx = graph
    spec, feats = _inputs(seed=1)
    a = bk.OrtBackend(onnx); b = bk.OrtBackend(onnx, io_binding=True, device="cpu")
    ya, yb = _run(a, a.new_state(), spec, feats), _run(b, b.new_state(), spec, feats)
    assert np.array_equal(ya, yb)


def test_interleaved_streams_do_not_contaminate(graph):
    _, onnx = graph
    (s1, f1), (s2, f2) = _inputs(seed=2), _inputs(seed=3)
    b = bk.OrtBackend(onnx)
    solo1, solo2 = _run(b, b.new_state(), s1, f1), _run(b, b.new_state(), s2, f2)
    st1, st2, o1, o2 = b.new_state(), b.new_state(), [], []
    for t in range(s1.shape[2]):
        o1.append(b.step(s1[:, :, t:t + 1], f1[:, t:t + 1], st1))
        o2.append(b.step(s2[:, :, t:t + 1], f2[:, t:t + 1], st2))
    assert np.array_equal(np.concatenate(o1, 2), solo1) and np.array_equal(np.concatenate(o2, 2), solo2)


def test_reset_equals_fresh_state(graph):
    _, onnx = graph
    spec, feats = _inputs(seed=4)
    b = bk.OrtBackend(onnx)
    st = b.new_state()
    _run(b, st, *_inputs(seed=5))
    b.reset(st)
    assert st.sample_counter == 0 and st.discontinuity_flags & bk.DISC_RESET
    assert all(not np.any(v) for v in st.caches.values())
    assert np.array_equal(_run(b, st, spec, feats), _run(b, b.new_state(), spec, feats))


def test_state_round_trip_resumes_exactly(graph, tmp_path):
    _, onnx = graph
    spec, feats = _inputs(seed=6)
    b = bk.OrtBackend(onnx)
    whole = _run(b, b.new_state(), spec, feats)
    st = b.new_state("cfg"); st.sample_counter = 1234; st.channel_validity = (True, False); st.discontinuity_flags = 9
    head = _run(b, st, spec[:, :, :10], feats[:, :10])
    b.to_host(st).save(tmp_path / "s.npz")
    back = bk.StreamState.load(tmp_path / "s.npz")
    assert (back.profile_id, back.config_hash, back.sample_counter, back.channel_validity, back.discontinuity_flags) == \
        (st.profile_id, "cfg", 1234, (True, False), 9)
    tail = _run(b, b.from_host(back), spec[:, :, 10:], feats[:, 10:])
    assert np.array_equal(np.concatenate([head, tail], 2), whole)


def test_state_schema_and_shape_checks(graph):
    _, onnx = graph
    b = bk.OrtBackend(onnx)
    st = b.to_host(b.new_state())
    raw = st.to_bytes()
    bad = bk.StreamState(st.profile_id, "", st.caches, schema_version=99)
    with pytest.raises(ValueError, match="schema"):
        bk.StreamState.from_bytes(bad.to_bytes())
    k = next(iter(st.caches))
    wrong = bk.StreamState(st.profile_id, "", {**st.caches, k: np.zeros((1, 2), np.float32)})
    with pytest.raises(ValueError, match="backend expects"):
        b.from_host(wrong)                                    # a width change never reuses hidden state
    with pytest.raises(ValueError, match="profile"):
        b.from_host(bk.StreamState("other", "", st.caches))
    assert bk.StreamState.from_bytes(raw).caches.keys() == st.caches.keys()


def test_torch_backend_matches_ort(graph):
    ck, onnx = graph
    spec, feats = _inputs(seed=7)
    o = bk.OrtBackend(onnx); t = bk.TorchBackend.from_checkpoint(ck, "cpu", profile_id=o.profile_id)
    assert t.cache_specs() == o.cache_specs()
    assert np.abs(_run(t, t.new_state(), spec, feats) - _run(o, o.new_state(), spec, feats)).max() < 1e-4
    # two torch streams on one module stay independent (the twin must not keep hidden per-module state)
    st1, st2 = t.new_state(), t.new_state()
    a = [t.step(spec[:, :, i:i + 1], feats[:, i:i + 1], st1) for i in range(8)]
    _ = [t.step(spec[:, :, i:i + 1] * 0, feats[:, i:i + 1] * 0, st2) for i in range(8)]
    st3 = t.new_state()
    b = [t.step(spec[:, :, i:i + 1], feats[:, i:i + 1], st3) for i in range(8)]
    assert np.array_equal(np.concatenate(a, 2), np.concatenate(b, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device visible")
def test_torch_backend_cuda(graph):
    ck, onnx = graph
    spec, feats = _inputs(seed=8)
    c, o = bk.TorchBackend.from_checkpoint(ck, "cuda"), bk.OrtBackend(onnx)
    assert np.abs(_run(c, c.new_state(), spec, feats) - _run(o, o.new_state(), spec, feats)).max() < 1e-3


def test_gpu_execution_providers_absent_or_flagged_untested(graph):
    import onnxruntime as ort
    _, onnx = graph
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        with pytest.raises(RuntimeError, match="not available"):
            bk.OrtBackend(onnx, providers=["CUDAExecutionProvider"])
        pytest.skip("onnxruntime has no CUDA EP here; the IO-binding GPU path is untested")
    b = bk.OrtBackend(onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert b.io_binding and not b.tested


def test_telemetry_bounded_and_counts_misses():
    tm = bk.Telemetry(deadline_ms=16.0, window=100)
    for ms in list(np.linspace(1, 10, 990)) + [20.0] * 10:
        tm.add(ms)
    s = tm.summary()
    assert s["steps"] == 1000 and s["deadline_misses"] == 10 and s["max_ms"] == 20.0
    assert len(tm.recent) == 100
    assert abs(s["p50_ms"] - 5.5) < 0.05 and s["p99_ms"] <= 20.0


def test_config_hash_tracks_dsp():
    a = bk.config_hash(True, {"limiter": True})
    assert a == bk.config_hash(True, {"limiter": True}) != bk.config_hash(True, {"limiter": False})
