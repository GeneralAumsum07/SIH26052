"""Shared batch streams for the r8 low-delay lanes (lever 2): one batch server per stream signature, read by every run
of that stream, instead of one server per run.

`ensure` is called by a lane before each training try. It prints the VAANI_STREAM_ID the run must use:
  g<k>    group ring k of the stream: the first whose readers are near the run's start, else a new one that later
          runs can join (resumed full runs and fresh pilots of one stream sit in different groups)
  <name>  a private ring, as before, which the lane serves: every group slot is busy far from the run's start
The group server serves `--workers` per live reader (each reader is a lane with that budget), so sharing never
takes more CPU than the lanes it feeds; it outlives any one run and exits after --exit-idle-s without readers.
Readers never drift apart (stream_server rewinds to a reader behind the ring), so the only cost of joining is the
wait until readers meet: bounded here by --gap batches.

usage:  python scripts/r8_stream.py ensure --config C --run-dir D --name N --q Q --workers W --max-workers M
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani.data import stream_server as ss   # noqa: E402


def batches_per_epoch(cfg):
    return math.ceil(cfg["data"].get("epoch_len", 20000) / cfg["batch_size"])


def resume_seq(run_dir: Path, cfg) -> int:
    """Global batch the run starts at: the epoch after last.pt's (train.py resumes at epoch boundaries)."""
    last = Path(run_dir) / "last.pt"
    if not last.exists():
        return 0
    import torch
    # the run's own checkpoint, loaded as train.py's resume loads it (its config and optimizer are not tensors)
    ck = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
    return (int(ck.get("epoch", -1)) + 1) * batches_per_epoch(cfg)


def group_readers(sig: str, gid: str = "", stale_s: float = 60.0):
    """Live reader positions of group ring `gid` of the stream; None when no live, beating ring exists."""
    os.environ["VAANI_STREAM_ID"] = gid
    try:
        shm = shared_memory.SharedMemory(name=ss.shm_name(sig))
    except FileNotFoundError:
        return None
    try:
        if os.name != "nt":   # an attach must not unlink the server's ring at exit
            resource_tracker.unregister(shm._name, "shared_memory")
        ring = ss._Ring(shm)
        if ring.closed or ring.signature != sig:
            return None
        if time.monotonic_ns() // 1_000_000 - int(ring.hdr[ss.BEAT]) > stale_s * 1000:
            return None
        return [int(p) for p, pid in ring.readers if p >= 0 and ss._alive(int(pid))]
    except ValueError:   # the server is mid-start
        return []
    finally:
        shm.close()


class _Lock:
    """mkdir lock: two lanes starting runs of one stream must not both start its server."""

    def __init__(self, path: Path, stale_s: float = 120.0):
        self.path, self.stale_s = path, stale_s

    def __enter__(self):
        while True:
            try:
                os.mkdir(self.path)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:   # a killed holder
                        os.rmdir(self.path)
                except FileNotFoundError:
                    pass
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            os.rmdir(self.path)
        except FileNotFoundError:
            pass


def decide(resume: int, readers, gap: int):
    """(join, why): join the group when its readers' positions are within `gap` batches of the run's start."""
    if readers is None:
        return True, "no group server: starting one"
    if not readers:
        return True, "group server idle: joining (nobody waits)"
    lo, hi = min(readers), max(readers)
    if lo - gap <= resume <= hi + gap:
        return True, f"joining group readers at {lo}..{hi} (start {resume}, gap <= {gap})"
    return False, f"group readers at {lo}..{hi}, start {resume}: more than {gap} batches apart, private ring"


def ensure(config, run_dir, name, q, workers, max_workers, gap=5000, exit_idle_s=1800.0, py=sys.executable,
           dry=False, max_groups=8):
    """VAANI_STREAM_ID for the run: a group ring g<k> of its stream (joined, or started for later runs to join), or
    its own name (a private ring the lane serves) when every group slot is busy far from its start."""
    import yaml
    cfg = yaml.safe_load(open(config))
    sig = ss.stream_signature(cfg)
    q = Path(q); q.mkdir(parents=True, exist_ok=True)
    start = resume_seq(run_dir, cfg)
    with _Lock(q / f"stream.{sig[:16]}.lock"):
        free, why = None, None
        for k in range(max_groups):   # groups at different positions: each run joins the one near it
            gid = f"g{k}"
            pidf = q / f"stream.{sig[:16]}.{gid}.json"
            rec = json.loads(pidf.read_text()) if pidf.exists() else None
            alive = rec is not None and ss._alive(int(rec["pid"]))
            if not alive:
                free = gid if free is None else free
                continue
            readers = group_readers(sig, gid)
            if readers is None:   # started, ring not yet up: its start is where its first reader is
                readers = [int(rec["start_seq"])]
            join, why = decide(start, readers, gap)
            if join:
                break
        else:
            gid = free
            if gid is None:
                why = f"all {max_groups} group rings busy far from start {start}: private ring"
            else:
                why = f"no group near start {start}: starting {gid}"
                if not dry:
                    pidf = q / f"stream.{sig[:16]}.{gid}.json"
                    log = open(q / "logs" / f"stream.{sig[:16]}.{gid}.log", "a") if (q / "logs").is_dir() else subprocess.DEVNULL
                    env = dict(os.environ, VAANI_STREAM_ID=gid)
                    p = subprocess.Popen([py, "-m", "vaani.data.stream_server", "--config", str(config),
                                          "--workers", str(workers), "--per-reader", "--max-workers", str(max_workers),
                                          "--open-ended", "--exit-idle-s", str(exit_idle_s), "--start-seq", str(start)],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                    pidf.write_text(json.dumps(dict(pid=p.pid, start_seq=start, config=str(config), sig=sig)))
    print(f"stream {sig[:16]} {name}: {why}", file=sys.stderr, flush=True)
    return gid if gid is not None else name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["ensure"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--max-workers", type=int, required=True)
    ap.add_argument("--gap", type=int, default=int(os.environ.get("STREAM_GAP", "5000")))
    ap.add_argument("--exit-idle-s", type=float, default=1800.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    print(ensure(a.config, a.run_dir, a.name, a.q, a.workers, a.max_workers, a.gap, a.exit_idle_s, dry=a.dry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
