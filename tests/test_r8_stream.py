"""Shared batch streams (scripts/r8_stream.py): join-or-private decisions, resume positions, and two runs of one
stream reading one group server end to end."""
import json, os, sys, time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import r8_stream as RS   # noqa: E402
from vaani.data import stream_server as ss   # noqa: E402
from tests.test_stream_server import _cfg, _local, _same   # noqa: E402
from tests.test_train_smoke import _tiny   # noqa: E402


def test_decide_joins_within_the_gap_only():
    assert RS.decide(0, None, 100)[0]                 # no server: start one
    assert RS.decide(9999, [], 100)[0]                # idle server: nobody waits
    assert RS.decide(250, [200, 300], 100)[0]         # between the readers
    assert RS.decide(150, [200, 300], 100)[0]         # behind by 50: the group rewinds 50
    assert RS.decide(400, [200, 300], 100)[0]         # ahead by 100: it waits 100
    assert not RS.decide(99, [200, 300], 100)[0]      # too far behind: private ring
    assert not RS.decide(401, [200, 300], 100)[0]     # too far ahead: private ring


def test_resume_seq_is_the_epoch_after_last_pt(tmp_path):
    cfg = {"batch_size": 32, "data": {"epoch_len": 20000}}
    assert RS.resume_seq(tmp_path, cfg) == 0
    torch.save({"epoch": 10, "step": 6875}, tmp_path / "last.pt")
    assert RS.resume_seq(tmp_path, cfg) == 11 * 625


def test_two_runs_share_one_group_server(tmp_path, monkeypatch):
    monkeypatch.delenv("VAANI_STREAM_ID", raising=False)
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    c = tmp_path / "s.yaml"; c.write_text(yaml.safe_dump(cfg))
    q = tmp_path / "q"; (q / "logs").mkdir(parents=True)
    sig = ss.stream_signature(cfg)
    try:
        assert RS.ensure(c, tmp_path / "run_a", "run_a", q, 0, 0, gap=10, exit_idle_s=5) == "g0"
        assert RS.ensure(c, tmp_path / "run_b", "run_b", q, 0, 0, gap=10, exit_idle_s=5) == "g0"
        # a run far from g0's start gets a group of its own, which a later run near it would join
        far = tmp_path / "run_far"; far.mkdir(); torch.save({"epoch": 40}, far / "last.pt")
        assert RS.ensure(c, far, "run_far", q, 0, 0, gap=10, exit_idle_s=5, dry=True, max_groups=2) == "g1"
        assert RS.ensure(c, far, "run_far", q, 0, 0, gap=10, exit_idle_s=5, dry=True, max_groups=1) == "run_far"
        pid = json.loads((q / f"stream.{sig[:16]}.g0.json").read_text())["pid"]
        monkeypatch.setenv("VAANI_STREAM_ID", "g0")   # what the lane passes the trainer
        a = ss.RingReader.attach(cfg, timeout_s=120)
        b = ss.RingReader.attach(cfg, timeout_s=120)
        assert a is not None and b is not None and a.shm.name == b.shm.name
        ds = _local(cfg)
        for s in range(6):
            ba, bb = a.get(s // 3, s % 3, wait_s=60), b.get(s // 3, s % 3, wait_s=60)
            _same(ba, ss.local_batch(ds, s // 3, s % 3, 2)); _same(bb, ba)
        assert a.misses == b.misses == 0
        a.detach(); b.detach()
        t0 = time.time()
        while ss._alive(pid) and time.time() - t0 < 60:   # no readers left: the group server exits by itself
            if hasattr(os, "WNOHANG"):   # our child here (the launcher's helper exits, init reaps it): reap the zombie
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            time.sleep(0.5)
        assert not ss._alive(pid)
    finally:
        rec = q / f"stream.{sig[:16]}.g0.json"
        if rec.exists() and ss._alive(json.loads(rec.read_text())["pid"]):
            import signal
            os.kill(json.loads(rec.read_text())["pid"], signal.SIGTERM)
