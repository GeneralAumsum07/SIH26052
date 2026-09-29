"""One batch server per rendered data stream (low-delay plan, Section 3.10 / Task 4b, `perf.ops.stream: shared`).

Runs with the same seed and crop see identical data (items are a pure function of (seed, epoch, index)), so every run
of a stream can read one rendering. The server builds batches with the standard DynamicMixDataset semantics (one
EpochBatchSampler over all epochs), and publishes them to a shared-memory ring. Each training process reads batch
(epoch, i) from the ring; a batch outside the ring's window (for example after a resume) is rendered locally, which
yields the same batch. Back-pressure: the server never overwrites a slot an attached reader has not consumed, so a
batch is never reordered or dropped.

The stream signature covers every data-defining key (seed, crop, epoch length, batch size, manifests and their
content hash, bank, pack, mix, ref_corrupt, exclude_groups_file, scene_weights, the model's inputs, dsp, the audio
contract's frontend policy and perf.numerics.render). A reader whose signature differs refuses to attach.

Layout of the shared memory `vaani_ring_<sig16>`:
  header  int64[HDR]: magic, n_slots, slot_bytes, max_readers, batches_per_epoch, closed, 64-byte signature
  slots   int64[n_slots] sequence number held by each slot (-1 = empty)
  readers int64[max_readers] next sequence each reader needs (-1 = free), pid
  data    n_slots x slot_bytes: [int64 nbytes][pickle bytes]
Sequence numbers are global batch indices, epoch * batches_per_epoch + i.

usage (the launcher starts one per stream):
    python -m vaani.data.stream_server --config cfg.yaml [--slots 64] [--workers 8] [--start-epoch 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np

MAGIC = 0x56414E49524E4731   # "VANIRNG1"
HDR = 16
SIG_BYTES = 64


def stream_signature(cfg: dict) -> str:
    """sha256 over every key that defines the rendered data."""
    from vaani.audio_contract import contract_of
    d = cfg["data"]
    try:
        from vaani.data import manifests
        mhash = manifests.content_hash(d["manifests"])
    except Exception:   # an absent manifest: the paths alone (the dataset would not build anyway)
        mhash = None
    c = contract_of(cfg.get("model_cfg")) if cfg.get("model") == "vaani_fe" else None
    key = {
        "seed": cfg["seed"], "crop_s": d.get("crop_s", 4.0), "epoch_len": d.get("epoch_len", 20000),
        "batch_size": cfg["batch_size"], "manifests": list(d["manifests"]), "manifest_hash": mhash,
        "bank": d.get("bank"), "pack": d.get("pack", "data/pack"), "mix": d.get("mix", {}),
        "ref_corrupt": d.get("ref_corrupt"), "exclude_groups_file": d.get("exclude_groups_file"),
        "scene_weights": d.get("scene_weights"), "inputs": (cfg.get("model_cfg") or {}).get("inputs"),
        "model": cfg.get("model"), "dsp": cfg.get("dsp"), "controller_on": cfg.get("controller_on"),
        "frontend": None if c is None else {"contract": c.audio_contract_id if not c.is_legacy else None,
                                            "ramp": c.ramp_samples, "sub": c.limiter_sub},
        "render": ((cfg.get("perf") or {}).get("numerics") or {}).get("render", "cpu"),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def shm_name(sig: str) -> str:
    # $VAANI_STREAM_ID gives a run its own ring: runs sharing one drift apart after a resume and the one behind misses
    sid = os.environ.get("VAANI_STREAM_ID")
    return f"vaani_ring_{sig[:16]}" + (f"_{sid}" if sid else "")


class _Ring:
    def __init__(self, shm, create=False, n_slots=0, slot_bytes=0, max_readers=0, bpe=0, sig=""):
        self.shm = shm
        buf = shm.buf
        self.hdr = np.ndarray((HDR,), np.int64, buf, 0)
        if create:
            self.hdr[:] = 0
            self.hdr[:6] = [MAGIC, n_slots, slot_bytes, max_readers, bpe, 0]
            np.ndarray((SIG_BYTES,), np.uint8, buf, 6 * 8)[:] = np.frombuffer(sig.encode()[:SIG_BYTES].ljust(SIG_BYTES), np.uint8)
        if self.hdr[0] != MAGIC:
            raise ValueError("not a vaani ring")
        self.n_slots, self.slot_bytes, self.max_readers, self.bpe = (int(x) for x in self.hdr[1:5])
        off = HDR * 8
        self.slots = np.ndarray((self.n_slots,), np.int64, buf, off); off += self.n_slots * 8
        self.readers = np.ndarray((self.max_readers, 2), np.int64, buf, off); off += self.max_readers * 16
        self.data_off = off
        if create:
            self.slots[:] = -1
            self.readers[:] = -1

    @property
    def signature(self) -> str:
        return bytes(np.ndarray((SIG_BYTES,), np.uint8, self.shm.buf, 6 * 8)).decode("ascii", "replace").strip()

    @property
    def closed(self) -> bool:
        return bool(self.hdr[5])

    @staticmethod
    def size(n_slots, slot_bytes, max_readers):
        return HDR * 8 + n_slots * 8 + max_readers * 16 + n_slots * slot_bytes

    def _slot(self, k):
        a = self.data_off + k * self.slot_bytes
        return self.shm.buf[a:a + self.slot_bytes]

    def write(self, seq, payload: bytes):
        k = seq % self.n_slots
        if len(payload) + 8 > self.slot_bytes:
            raise ValueError(f"batch of {len(payload)} bytes exceeds the ring slot ({self.slot_bytes})")
        self.slots[k] = -1                                  # invalidate before overwriting
        v = self._slot(k)
        v[:8] = np.int64(len(payload)).tobytes()
        v[8:8 + len(payload)] = payload
        self.slots[k] = seq                                 # publish

    def read(self, seq):
        k = seq % self.n_slots
        if self.slots[k] != seq:
            return None
        v = self._slot(k)
        n = int(np.frombuffer(bytes(v[:8]), np.int64)[0])
        data = bytes(v[8:8 + n])
        if self.slots[k] != seq:                            # overwritten while copying
            return None
        return data

    def min_reader_pos(self):
        act = self.readers[:, 0]
        act = act[act >= 0]
        return int(act.min()) if act.size else None


def _alive(pid: int) -> bool:
    if os.name == "nt":
        # On Windows os.kill(pid, 0) calls TerminateProcess. Query a process
        # handle instead; access denied conservatively means it may be alive.
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such PID
        try:
            return k.WaitForSingleObject(handle, 0) != 0
        finally:
            k.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _live_min(ring, floor):
    """Lowest position among live readers above floor; frees rows of dead readers (a killed trainer never detaches)."""
    lo = None
    for r in range(ring.max_readers):
        pos, pid = int(ring.readers[r, 0]), int(ring.readers[r, 1])
        if pos < 0:
            continue
        if pid > 0 and not _alive(pid):
            ring.readers[r] = -1
            continue
        if (floor is None or pos > floor) and (lo is None or pos < lo):
            lo = pos
    return lo


def serve(cfg: dict, start_epoch: int = 0, n_slots: int = 64, max_readers: int = 32, workers: int = 4,
          poll_s: float = 0.002, ready=None, stop_after: int | None = None, render_device=None, idle_s: float = 300.0):
    """Render every batch of the stream in order into the ring (blocks). stop_after: batches (tests)."""
    import torch
    from torch.utils.data import DataLoader
    from vaani import runtime, train
    from vaani.data.dataset import DynamicMixDataset, EpochBatchSampler, collate
    from vaani.data.mixer import MixConfig
    d = cfg["data"]
    ds = DynamicMixDataset(d["manifests"], "train", d.get("bank"), MixConfig(**d.get("mix", {})), d.get("crop_s", 4.0),
                           d.get("epoch_len", 20000), cfg["seed"], **train.dataset_kwargs(cfg))
    batch_size, epochs = cfg["batch_size"], cfg["epochs"]
    bpe = EpochBatchSampler(len(ds), batch_size, 0, 1).batches_per_epoch
    end = epochs * bpe
    render = ((cfg.get("perf") or {}).get("numerics") or {}).get("render", "cpu")
    dev = torch.device(render_device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def loader(epoch0, nw=workers):
        """Batches from epoch0 on. Workers exist only while a reader is served (they are the box's CPU budget)."""
        bs = EpochBatchSampler(len(ds), batch_size, epoch0, epochs)
        kw = runtime.loader_kwargs(nw, torch.device("cpu"))
        if render == "gpu":   # CPU workers emit recipes (every draw and read); the server renders them on its GPU
            from vaani.data import mixer_gpu
            rl = DataLoader(mixer_gpu.RecipeDataset(ds), batch_sampler=bs, collate_fn=mixer_gpu.collate_recipes, **kw)
            return (mixer_gpu.render_and_finish(ds, recs, dev) for recs in rl)
        return iter(DataLoader(ds, batch_sampler=bs, collate_fn=collate, **kw))

    sig = stream_signature(cfg)
    ring = shm = None
    try:
        # size the ring from one batch rendered here, so readers can attach before any worker is spawned
        seq = start_epoch * bpe
        payload = pickle.dumps(next(loader(start_epoch, 0)), protocol=pickle.HIGHEST_PROTOCOL)
        slot = int(len(payload) * 1.25) + 4096
        try:
            shared_memory.SharedMemory(name=shm_name(sig)).unlink()   # a stale ring of a dead server
        except FileNotFoundError:
            pass
        shm = shared_memory.SharedMemory(name=shm_name(sig), create=True, size=_Ring.size(n_slots, slot, max_readers))
        ring = _Ring(shm, True, n_slots, slot, max_readers, bpe, sig)
        ring.write(seq, payload)
        if ready is not None:
            ready.set()
        n = 1
        while seq < end and (stop_after is None or n < stop_after):
            lo = _live_min(ring, None)
            if lo is None:   # nobody to serve: no workers, no CPU
                time.sleep(poll_s * 50)
                continue
            seq = (lo // bpe) * bpe   # the loader starts at an epoch boundary: restart at the lowest reader's epoch
            dl, idle_since = loader(seq // bpe), None
            for batch in dl:
                payload = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
                while True:   # back-pressure: the slot's previous occupant must be consumed by every live reader
                    lo = _live_min(ring, seq - n_slots - 1)   # readers already past the window render locally
                    if lo is None:
                        idle_since = idle_since or time.time()
                    else:
                        idle_since = None
                    if lo is None or lo > seq - n_slots:
                        break
                    time.sleep(poll_s)
                ring.write(seq, payload)
                seq += 1; n += 1
                if stop_after is not None and n >= stop_after:
                    break
                if idle_since is not None and time.time() - idle_since > idle_s:
                    break   # every reader left: release the workers, wait for the next one
            del dl
        ring.hdr[5] = 1   # closed: readers render locally past the end
        while _live_min(ring, None) is not None and _live_min(ring, None) < seq:
            time.sleep(poll_s * 10)
    finally:
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


class RingReader:
    """A training process's view of its stream's ring. get(epoch, i) -> batch or None (render locally)."""

    def __init__(self, ring, shm, reader_id, start_seq):
        self.ring, self.shm, self.id = ring, shm, reader_id
        self.ring.readers[reader_id] = [start_seq, os.getpid()]
        self.hits = self.misses = 0

    @classmethod
    def attach(cls, cfg, ds=None, start_seq=0, timeout_s=600.0):
        """Attach to the stream's ring. Raises ValueError on a signature mismatch; returns None when no server runs
        (the caller renders locally). A fallback lasts the whole run, so wait out a slow server start (17 at once >60 s)."""
        sig = stream_signature(cfg)
        t0 = time.time()
        while True:
            try:
                shm = shared_memory.SharedMemory(name=shm_name(sig))
                # py3.12 registers attaches too: a reader's exit would unlink the server's ring
                if os.name != "nt":  # Windows lifetime is managed by handles, with no POSIX resource tracker
                    resource_tracker.unregister(shm._name, "shared_memory")
                ring = _Ring(shm)
                break
            except FileNotFoundError:
                pass
            except ValueError:   # created but the header not yet written: the server is mid-start
                shm.close()
            if time.time() - t0 > timeout_s:
                return None
            time.sleep(0.05)
        if ring.signature != sig:
            shm.close()
            raise ValueError(f"stream signature mismatch: ring {ring.signature[:16]} vs run {sig[:16]}; refusing to attach")
        free = np.flatnonzero(ring.readers[:, 0] < 0)
        if not free.size:
            shm.close()
            raise RuntimeError("ring has no free reader slot")
        return cls(ring, shm, int(free[0]), start_seq)

    def get(self, epoch, i, wait_s=120.0):
        seq = epoch * self.ring.bpe + i
        t0 = time.time()
        while True:
            data = self.ring.read(seq)
            if data is not None:
                self.ring.readers[self.id, 0] = seq + 1
                self.hits += 1
                return pickle.loads(data)
            held = int(self.ring.slots[seq % self.ring.n_slots])
            if held > seq or self.ring.closed or time.time() - t0 > wait_s:
                # outside the window (overwritten or past the end): render locally, the same batch
                self.ring.readers[self.id, 0] = seq + 1
                self.misses += 1
                return None
            time.sleep(0.001)

    def detach(self):
        self.ring.readers[self.id] = [-1, -1]
        self.shm.close()


def local_batch(ds, epoch, i, batch_size):
    """Batch (epoch, i) rendered in this process: what the ring would have held."""
    from vaani.data.dataset import collate
    base = epoch * len(ds)
    a, b = i * batch_size, min((i + 1) * batch_size, len(ds))
    return collate([ds[base + j] for j in range(a, b)])


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--readers", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start-epoch", type=int, default=0)
    ap.add_argument("--idle-s", type=float, default=300.0)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    print(f"stream {stream_signature(cfg)[:16]}: serving {a.config}", flush=True)
    serve(cfg, a.start_epoch, a.slots, a.readers, a.workers, idle_s=a.idle_s)


if __name__ == "__main__":
    main()
