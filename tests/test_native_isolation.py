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


def test_the_composite_screen_survives_a_pesq_crash_and_reports_it(monkeypatch, capsys):
    import torch
    from types import SimpleNamespace
    from vaani import train
    w = metrics._PesqWorker(_CRASH_SRC); monkeypatch.setattr(metrics, "_worker", w)
    c, y = _sig(4, n=12345); c2, y2 = _sig(5)
    items = [dict(cond="present", clean=c, clean_item=False, snr_in=0.0, prim=y, out=y),       # crashes the child
             dict(cond="present", clean=c2, clean_item=False, snr_in=0.0, prim=y2, out=y2)]
    screen = SimpleNamespace(items=items, metrics=metrics, c=dict(train.COMPOSITE_DEFAULTS), base_d_snr=None,
                             er=SimpleNamespace(frame_stats=lambda clean, y, prim: {"speech_loss": 0.0}),
                             enhance=lambda model, it, device: it["out"])
    try:
        s = train.CompositeScreen.score(screen, torch.nn.Identity(), "cpu")
    finally:
        w.stop()
    assert s["pesq_failures"] == 1 and s["n_clips"] == 2
    assert s["pesq"] == metrics.pesq_wb_inproc(c2, y2)   # the crashed clip drops out of the mean, the other is exact
    assert "composite screen: 1 PESQ child failure" in capsys.readouterr().out


def test_validate_logs_pesq_over_the_scored_items_and_keeps_stoi(monkeypatch):
    import vaani.train_refiner as tr
    from types import SimpleNamespace
    from vaani import train
    arr = np.array([[5.0, 0.8, 2.0], [6.0, 0.9, np.nan], [7.0, 0.7, 3.0]])
    monkeypatch.setattr(tr, "screen_items", lambda root, split: (None, [0, 1, 2]))
    monkeypatch.setattr(tr, "score_items", lambda *a: arr)
    dl = SimpleNamespace()
    v = train.validate(None, dl, {"val": {"eval_root": "x"}}, "cpu")
    assert v == float(arr.mean(0)[1])
    assert dl._last_val_metrics == {"snr_out": 6.0, "stoi": v, "pesq_wb": 2.5, "pesq_nan": 1}


@pytest.mark.parametrize("clip", OFFENDING[:1])
def test_the_training_screen_pool_finishes_on_an_offending_input(clip):
    # the r8 val screen fans PESQ out to a spawn Pool; before isolation a dead worker left Pool.map waiting forever
    sf = pytest.importorskip("soundfile")
    from multiprocessing import get_context
    from vaani.train_refiner import _metric_item
    d = ROOT / "data" / "eval_r2_relabel" / "test" / clip
    if not d.with_suffix(".mix.wav").exists():
        pytest.skip("eval_r2_relabel not on this machine")
    clean, _ = sf.read(str(d.with_suffix(".clean.wav"))); mix, _ = sf.read(str(d.with_suffix(".mix.wav")))
    with get_context("spawn").Pool(2) as pool:
        out = pool.map_async(_metric_item, [(clean, mix[:, 0])] * 8, chunksize=1).get(timeout=300)
    assert len(out) == 8 and all(np.isfinite(o[1]) for o in out)
