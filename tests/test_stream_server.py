"""Batch server per stream (plan Task 4b): ring batches bit-identical to a local loader, signature refusal,
back-pressure without reordering or drops."""
import multiprocessing as mp
import time

import pytest
import torch

from vaani.data import stream_server as ss
from vaani.data.dataset import DynamicMixDataset, collate
from vaani.data.mixer import MixConfig
from vaani import train
from tests.test_train_smoke import _tiny


def _cfg(m, seed=0):
    return dict(name="s", model="vaani_fe", controller_on=False, loss="fe", seed=seed, batch_size=2, epochs=3,
                dsp={"limiter": True, "limiter_kernel": "numpy"}, model_cfg={"tier": "mini"},
                data=dict(manifests=[str(m)], bank=None, crop_s=0.5, epoch_len=6, mix={"p_room": 0.0},
                          ref_corrupt={"p": 0.5, "p_absent": 0.3}))


def _local(cfg):
    d = cfg["data"]
    return DynamicMixDataset(d["manifests"], "train", None, MixConfig(**d["mix"]), d["crop_s"], d["epoch_len"],
                             cfg["seed"], **train.dataset_kwargs(cfg))


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def _start(cfg, **kw):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=ss.serve, args=(cfg,), kwargs=dict(workers=0, ready=ready, **kw), daemon=True)
    p.start()
    assert ready.wait(60)
    return p


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows process probe")
def test_windows_liveness_probe_never_sends_a_signal(monkeypatch):
    import os
    def forbidden(*args):
        raise AssertionError("os.kill(pid, 0) is not a Windows liveness probe")
    monkeypatch.setattr(os, "kill", forbidden)
    assert ss._alive(os.getpid())


def test_ring_batches_equal_local_render(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    p = _start(cfg, n_slots=4)
    try:
        r = ss.RingReader.attach(cfg, start_seq=0, timeout_s=10)
        ds = _local(cfg)
        for e in range(3):
            for i in range(3):
                b = r.get(e, i, wait_s=30)
                assert b is not None
                _same(b, ss.local_batch(ds, e, i, 2))
        assert r.hits == 9 and r.misses == 0
        r.detach()
    finally:
        p.join(30); p.kill()


def test_signature_mismatch_refuses(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    p = _start(cfg, n_slots=2)
    try:
        other = _cfg(m)
        sig = ss.stream_signature(cfg)
        assert ss.stream_signature(other) == sig
        for change in (lambda c: c.update(seed=1), lambda c: c["data"].update(crop_s=0.75),
                       lambda c: c["data"].update(ref_corrupt=None), lambda c: c["model_cfg"].update(inputs="p"),
                       lambda c: c.update(perf={"numerics": {"render": "gpu"}})):
            c = _cfg(m); change(c)
            assert ss.stream_signature(c) != sig
        # a ring whose stored signature differs from the run's is refused (the holder keeps the server alive)
        holder = ss.RingReader.attach(cfg, timeout_s=10)
        name = ss.shm_name(sig)
        forged = _cfg(m); forged["seed"] = 7
        import vaani.data.stream_server as mod
        orig = mod.shm_name
        mod.shm_name = lambda s: name
        try:
            with pytest.raises(ValueError, match="signature mismatch"):
                ss.RingReader.attach(forged, timeout_s=5)
        finally:
            mod.shm_name = orig
        holder.detach()
    finally:
        p.join(30); p.kill()


def test_back_pressure_never_reorders_or_drops(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    p = _start(cfg, n_slots=2)
    try:
        fast = ss.RingReader.attach(cfg, timeout_s=10)
        slow = ss.RingReader.attach(cfg, timeout_s=10)
        ds = _local(cfg)
        for e in range(3):
            for i in range(3):
                bf = fast.get(e, i, wait_s=30)
                time.sleep(0.05)
                bs = slow.get(e, i, wait_s=30)
                assert bf is not None and bs is not None     # the 2-slot ring waited for the slow reader
                _same(bf, bs)
        _same(bs, ss.local_batch(ds, 2, 2, 2))
        assert fast.misses == slow.misses == 0
        fast.detach(); slow.detach()
    finally:
        p.join(30); p.kill()


def test_sampler_start_batch_is_a_suffix():
    from vaani.data.dataset import EpochBatchSampler
    full = list(EpochBatchSampler(7, 2, 1, 4))
    for k in range(len(full)):
        s = EpochBatchSampler(7, 2, 1 + k // 4, 4, k % 4)
        assert list(s) == full[k:] and len(s) == len(full) - k


def test_reader_behind_the_ring_rewinds_the_server(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    import threading
    p = _start(cfg, n_slots=2, open_ended=True)
    try:
        ds = _local(cfg)
        lead = ss.RingReader.attach(cfg, start_seq=4, timeout_s=10)   # a run resumed at batch 4
        assert lead.get(1, 1, wait_s=30) is not None
        time.sleep(0.5)   # the server ran ahead: batches 0..3 are gone from the 2-slot ring
        got = {}
        def run_lead():   # its own process in production: reads on, waits at the front of the ring
            got["lead"] = [lead.get(s // 3, s % 3, wait_s=60) for s in range(5, 12)]
        t = threading.Thread(target=run_lead); t.start()
        late = ss.RingReader.attach(cfg, start_seq=0, timeout_s=10)
        for s in range(12):   # the late run gets every batch from the ring, not a local render
            b = late.get(s // 3, s % 3, wait_s=60)
            assert b is not None, s
            _same(b, ss.local_batch(ds, s // 3, s % 3, 2))
        t.join(60)
        for s, b in zip(range(5, 12), got["lead"]):   # the leader waited for it: every batch, in order
            _same(b, ss.local_batch(ds, s // 3, s % 3, 2))
        assert lead.misses == late.misses == 0
        lead.detach(); late.detach()
    finally:
        p.join(30); p.kill()


def test_per_reader_workers_restart_at_the_exact_batch(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=ss.serve, args=(cfg,), daemon=False,   # its loader spawns workers
                    kwargs=dict(workers=1, per_reader=True, max_workers=2, n_slots=2, ready=ready))
    p.start()
    assert ready.wait(60)
    try:
        ds = _local(cfg)
        a = ss.RingReader.attach(cfg, timeout_s=10)
        for s in range(3):
            _same(a.get(s // 3, s % 3, wait_s=60), ss.local_batch(ds, s // 3, s % 3, 2))
        b = ss.RingReader.attach(cfg, start_seq=3, timeout_s=10)   # 1 -> 2 readers: the loader restarts mid-stream
        for s in range(3, 9):
            ba, bb = a.get(s // 3, s % 3, wait_s=60), b.get(s // 3, s % 3, wait_s=60)
            _same(ba, ss.local_batch(ds, s // 3, s % 3, 2)); _same(bb, ba)
        assert a.misses == b.misses == 0
        a.detach(); b.detach()
    finally:
        p.join(30); p.kill()


def test_idle_server_exits_and_readers_fall_back_on_a_dead_server(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    p = _start(cfg, n_slots=2, open_ended=True, exit_idle_s=0.5)
    p.join(30)
    assert p.exitcode == 0                                     # no reader ever came: released everything
    assert ss.RingReader.attach(cfg, timeout_s=1) is None      # nothing to attach to: the run renders locally
    p = _start(cfg, n_slots=2)
    try:
        r = ss.RingReader.attach(cfg, start_seq=0, timeout_s=10)
        assert r.get(0, 0, wait_s=30) is not None
        p.kill(); p.join(10)
        assert r.get(2, 2, wait_s=1.0) is None and r.misses == 1   # never written, server gone: local, not a hang
        r.detach()
    finally:
        p.kill()
