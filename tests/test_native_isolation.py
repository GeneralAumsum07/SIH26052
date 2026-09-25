"""pesq 0.0.4's native out-of-bounds read (results_r2/r8/native_crash/README.md) must never kill a run or hang a pool:
pesq_wb runs the C call in a child process, a child death is a NaN plus a warning, and normal values are unchanged."""
from pathlib import Path

import numpy as np
import pytest
from pesq import pesq as _pesq

from vaani import metrics

SR = 16000
ROOT = Path(__file__).resolve().parents[1]
# the two eval_r2_relabel/test raw inputs that ASan flags in utterance_split (a negative VAD index)
OFFENDING = ["changing_5/0018", "stationary_-10/0028"]

# a child that behaves like the real one except that a 12345-sample reference kills it with a native fault
_CRASH_SRC = metrics._WORKER_SRC.replace(
    "    try: v = float(pesq(16000, clean, est, \"wb\"))",
    "    if len(clean) == 12345:\n        import ctypes; ctypes.string_at(0)\n    try: v = float(pesq(16000, clean, est, \"wb\"))")


def _sig(seed=0, n=3 * SR):
    r = np.random.default_rng(seed); t = np.arange(n) / SR
    c = (np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 0.3).astype(np.float64)
    return c, c + r.normal(0, 0.02, n)


def test_isolated_values_are_bit_identical_to_in_process():
    for seed in range(3):
        c, y = _sig(seed)
        assert metrics.pesq_wb(c, y) == metrics.pesq_wb_inproc(c, y) == float(_pesq(SR, c, y, "wb"))
    c32, y32 = (a.astype(np.float32) for a in _sig(7))
    assert metrics.pesq_wb(c32, y32) == metrics.pesq_wb_inproc(c32, y32)   # dtype travels unchanged through pickle


def test_isolation_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("VAANI_PESQ_ISOLATE", "0")
    c, y = _sig(1)
    assert metrics.pesq_wb(c, y) == float(_pesq(SR, c, y, "wb"))


def test_no_utterance_is_still_nan():
    assert np.isnan(metrics.pesq_wb(np.zeros(SR), np.zeros(SR)))


def test_a_native_crash_in_the_child_is_nan_and_the_next_call_recovers(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("VAANI_PESQ_CRASH_DIR", str(tmp_path))
    w = metrics._PesqWorker(_CRASH_SRC)
    try:
        c, y = _sig(2)
        good = w(c, y)
        assert good == metrics.pesq_wb_inproc(c, y)
        assert np.isnan(w(np.ones(12345) * 0.1, np.ones(12345) * 0.1))      # the child segfaults; the caller lives
        assert w.failures == {"crash": 1, "timeout": 0}
        assert "pesq child crash" in capsys.readouterr().err
        assert len(list(tmp_path.glob("pesq_crash_*.npz"))) == 1              # the input is kept for a reproducer
        assert w(c, y) == good                                                 # a fresh child takes the next call
    finally:
        w.stop()


def test_a_hung_child_times_out_to_nan():
    hang = metrics._WORKER_SRC.replace("    try: v = float(pesq(16000, clean, est, \"wb\"))",
                                       "    import time; time.sleep(600)\n    try: v = float(pesq(16000, clean, est, \"wb\"))")
    w = metrics._PesqWorker(hang, timeout=3)
    try:
        c, y = _sig(3)
        assert np.isnan(w(c, y)) and w.failures == {"crash": 0, "timeout": 1}
    finally:
        w.stop()


def test_failures_are_counted_module_wide():
    f = metrics.pesq_failures()
    assert set(f) == {"crash", "timeout", "total"} and f["total"] == f["crash"] + f["timeout"]


@pytest.mark.parametrize("clip", OFFENDING)
def test_the_offending_inputs_never_kill_the_caller(clip):
    sf = pytest.importorskip("soundfile")
    d = ROOT / "data" / "eval_r2_relabel" / "test" / clip
    if not d.with_suffix(".mix.wav").exists():
        pytest.skip("eval_r2_relabel not on this machine")
    clean, _ = sf.read(str(d.with_suffix(".clean.wav"))); mix, _ = sf.read(str(d.with_suffix(".mix.wav")))
    raw = mix[:, 0] if mix.ndim == 2 else mix
    # in-process, these crash on the 2nd call in most fresh processes (Windows); 20 calls here must all return
    vals = [metrics.pesq_wb(clean, raw) for _ in range(20)]
    assert all(np.isnan(v) or 1.0 <= v <= 4.65 for v in vals)
