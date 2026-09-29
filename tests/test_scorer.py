"""Validation beside training (plan Task 4b): async scoring equals synchronous validation."""
import copy
import json

import pytest
import torch
import yaml
import multiprocessing
import time

from vaani import scorer, train
from tests.test_low_delay_training import _cfg
from tests.test_train_smoke import _tiny


def _run(tmp_path, cfg):
    cp = tmp_path / f"{cfg['name']}.yaml"
    yaml.safe_dump(cfg, open(cp, "w"))
    train.main(str(cp))
    return tmp_path / "runs" / cfg["name"]


def _strip(h):
    return [{k: v for k, v in r.items()} for r in h]


@pytest.mark.parametrize("workers", [False, True])
@pytest.mark.parametrize("ema", [None, {"decay": 0.9}])
def test_async_history_and_best_equal_sync(tmp_path, ema, workers):
    m = _tiny(tmp_path)
    base = _cfg(tmp_path, m, name="sync", epochs=3, max_steps=6, ema=ema)
    base["data"]["epoch_len"] = 4
    rs = _run(tmp_path, base)
    a = copy.deepcopy(base); a["name"] = "async"
    a["perf"] = dict(a["perf"], ops={"scorer": "async", "priority": 1})
    ra = _run(tmp_path, a)
    assert (ra / "train_done.json").exists() and not (ra / "scored_done.json").exists()
    if workers:   # a measure worker takes every snapshot first; the scorer then only applies the selection rule
        w = scorer.Scorer([ra], "cpu")
        while w.measure_step():
            pass
        assert len(list((ra / "snapshots").glob("meas_*.json"))) == 3
    sc = scorer.Scorer([ra], "cpu")
    sc.run(0.0, until_done=True, max_idle=2)
    assert (ra / "scored_done.json").exists() and not list((ra / "snapshots").glob("*"))
    hs, ha = json.loads((rs / "run.json").read_text()), json.loads((ra / "run.json").read_text())
    assert _strip(hs["history"]) == _strip(ha["history"])
    assert hs["best_val_stoi"] == ha["best_val_stoi"]
    bs, ba = torch.load(rs / "best.pt", weights_only=True), torch.load(ra / "best.pt", weights_only=True)
    assert bs["step"] == ba["step"] and bs.get("weights") == ba.get("weights")
    for k in bs["model"]:
        assert torch.equal(bs["model"][k], ba["model"][k]), k


def test_scorer_restart_resumes_without_rescoring(tmp_path, monkeypatch):
    m = _tiny(tmp_path)
    a = _cfg(tmp_path, m, name="restart", epochs=3, max_steps=6, ema=None)
    a["data"]["epoch_len"] = 4
    a["perf"] = dict(a["perf"], ops={"scorer": "async"})
    ra = _run(tmp_path, a)
    sc = scorer.Scorer([ra], "cpu")
    assert sc.step()                       # scores snapshot 1 only, then "crashes"
    calls = []
    orig = scorer.Selector.measure

    def counting(self, row, cands, vdl, n_points, final):
        calls.append(n_points)
        return orig(self, row, cands, vdl, n_points, final)
    monkeypatch.setattr(scorer.Selector, "measure", counting)
    sc2 = scorer.Scorer([ra], "cpu")        # a fresh scorer process
    sc2.run(0.0, until_done=True, max_idle=2)
    assert calls == [2, 3]                 # neither re-scored nor skipped
    h = json.loads((ra / "run.json").read_text())["history"]
    assert all("val_stoi" in r for r in h) and len(h) == 3


def test_async_refused_with_patience(tmp_path):
    m = _tiny(tmp_path)
    a = _cfg(tmp_path, m, name="pat")
    a["perf"] = dict(a["perf"], ops={"scorer": "async"})
    a["early_stopping"] = {"patience": 2}
    with pytest.raises(ValueError, match="patience"):
        _run(tmp_path, a)


def test_priority_order():
    class R:
        def __init__(self, rd, items):
            self.run_dir, self.items = rd, items

        def pending(self):
            return self.items
    sc = scorer.Scorer([], "cpu", {"a": 3, "b": 1})
    import pathlib, os, tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    pa, pb = d / "a.pt", d / "b.pt"
    pa.write_bytes(b"x"); pb.write_bytes(b"x")
    os.utime(pb, (pa.stat().st_mtime + 5,) * 2)     # b arrived later but has the higher priority
    sc.runs = {"a": R(d, [(1, pa)]), "b": R(d, [(1, pb)])}
    assert [it[3] for it in sc.queue()] == ["b", "a"]


def test_validation_preserves_ema_eval_mode(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(tmp_path, m)
    model = train.build_model(cfg["model"], model_cfg=cfg["model_cfg"]).eval()
    dl = train.build_val_loader(cfg, torch.device("cpu"), num_workers=0)
    train.validate(model, dl, cfg, torch.device("cpu"))
    assert not model.training, "validation must not turn the EMA shadow into a training model"


def test_snapshot_claim_excludes_another_owner(tmp_path, monkeypatch):
    # Two independent scorer objects must not own a snapshot simultaneously.
    # Mock the old PID probe: os.kill(pid, 0) can terminate processes on Windows.
    monkeypatch.setattr(scorer.os, "kill", lambda *args: None)
    owners = [object.__new__(scorer.RunScorer) for _ in range(2)]
    for rs in owners:
        rs.run_dir = tmp_path
    (tmp_path / scorer.SNAP_DIR).mkdir()
    assert owners[0]._claim(1)
    try:
        assert not owners[1]._claim(1)
    finally:
        for rs in owners:
            if hasattr(rs, "_release"):
                rs._release(1)


def _hold_claim(run_dir, ready):
    rs = object.__new__(scorer.RunScorer)
    rs.run_dir = run_dir
    assert rs._claim(1)
    ready.set()
    time.sleep(60)


def test_dead_worker_releases_snapshot_lock(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_hold_claim, args=(tmp_path, ready))
    p.start()
    rs = object.__new__(scorer.RunScorer)
    rs.run_dir = tmp_path
    try:
        assert ready.wait(30)
        assert not rs._claim(1)
        p.terminate(); p.join(10)
        assert not p.is_alive()
        assert rs._claim(1), "a crashed worker must not strand its snapshot"
    finally:
        rs._release(1)
        if p.is_alive():
            p.terminate()
        p.join(10)


def _measure_in_process(rd):
    torch.set_num_threads(1)
    sc = scorer.Scorer([rd], "cpu")
    for _ in range(100):
        if (rd / scorer.SCORED_DONE).exists():
            return
        if not sc.measure_step():
            time.sleep(.05)


def test_concurrent_processes_match_serial_selection(tmp_path):
    import shutil
    m = _tiny(tmp_path)
    cfg = _cfg(tmp_path, m, name="parallel", epochs=4, max_steps=8)
    cfg["perf"]["ops"] = {"scorer": "async"}
    rd = _run(tmp_path, cfg)
    serial = tmp_path / "serial"
    shutil.copytree(rd, serial)
    scorer.Scorer([serial], "cpu").run(0., until_done=True, max_idle=2)
    ctx = multiprocessing.get_context("spawn")
    children = [ctx.Process(target=_measure_in_process, args=(rd,)) for _ in range(2)]
    for p in children:
        p.start()
    try:
        sc = scorer.Scorer([rd], "cpu")
        # Allow spawned imports to finish so helpers actually contend with the
        # selector; serial equivalence alone does not test process coordination.
        deadline = time.monotonic() + 60
        while not list((rd / scorer.SNAP_DIR).glob("meas_*.json")):
            assert time.monotonic() < deadline
            time.sleep(.05)
        sc.run(.02, until_done=True, max_idle=1000)
        assert (rd / scorer.SCORED_DONE).exists()
        for p in children:
            p.join(15)
            assert p.exitcode == 0
        a = json.loads((rd / "run.json").read_text())
        b = json.loads((serial / "run.json").read_text())
        assert a["history"] == b["history"]
        ca, cb = [torch.load(x / "best.pt", weights_only=True) for x in (rd, serial)]
        assert ca["step"] == cb["step"] and ca["weights"] == cb["weights"]
        assert all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"])
    finally:
        for p in children:
            if p.is_alive():
                p.terminate()
            p.join(10)


def test_busy_snapshot_does_not_block_other_runs(tmp_path):
    class R:
        def __init__(self, busy):
            self.run_dir = tmp_path
            self.busy = busy
            self.called = False
        def pending(self):
            return [(1, tmp_path / "snap_00001.pt")]
        def finalize(self):
            return False
        def score(self, idx, path):
            self.called = True
            return None if self.busy else {"step": 1}
    sc = scorer.Scorer([], "cpu")
    sc.runs = {"a": R(True), "b": R(False)}
    sc.queue = lambda: [(0, 0, 1, k, tmp_path / "snap_00001.pt") for k in sc.runs]
    assert sc.step()
    assert sc.runs["b"].called


def test_train_done_is_published_after_stream_statistics(tmp_path, monkeypatch):
    from vaani.data import stream_server
    m = _tiny(tmp_path)
    cfg = _cfg(tmp_path, m, name="publication")
    cfg["perf"]["ops"] = {"scorer": "async", "stream": "shared"}
    class Reader:
        hits, misses = 0, 2
        def get(self, *args):
            return None
        def detach(self):
            pass
    monkeypatch.setattr(stream_server.RingReader, "attach", lambda *a, **kw: Reader())
    original = scorer.mark_train_done
    def mark(rd, n):
        info = json.loads((rd / "run.json").read_text())
        assert info.get("stream_misses") == 2, "trainer must finish writing before scorer may finalize"
        original(rd, n)
    monkeypatch.setattr(scorer, "mark_train_done", mark)
    _run(tmp_path, cfg)
