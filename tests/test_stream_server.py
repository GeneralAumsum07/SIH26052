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


def test_reader_outside_window_renders_locally(tmp_path):
    m = _tiny(tmp_path)
    cfg = _cfg(m)
    p = _start(cfg, n_slots=2, stop_after=6)
    try:
        holder = ss.RingReader.attach(cfg, start_seq=4, timeout_s=10)   # a run resumed at batch 4
        time.sleep(1.0)   # the server ran ahead to batch 5: batches 0..3 were overwritten
        r = ss.RingReader.attach(cfg, start_seq=0, timeout_s=10)
        assert r.get(0, 0, wait_s=5) is None and r.misses == 1          # outside the window: render locally
        _same(ss.local_batch(_local(cfg), 0, 0, 2), ss.local_batch(_local(cfg), 0, 0, 2))
        assert holder.get(1, 1, wait_s=10) is not None                  # seq 4 is still in the ring
        r.detach(); holder.detach()
    finally:
        p.join(30); p.kill()
