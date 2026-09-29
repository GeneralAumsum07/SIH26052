"""Conservative CPU/memory accounting for simultaneous trainers, loaders and scorers."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def budget(cpus, memory_gb, lanes, reserve=8, train_threads=2, scorer_procs=2,
           scorer_threads=4, screen_workers=2, worker_gb=1., workers=None):
    # Scorer pools are per PROCESS, not per GPU. Reserve room for the parent
    # trainer, batch server and OS; cap memory too. A lane waits for scoring to
    # drain before its next run, so old scorers cannot accumulate without bound.
    if min(cpus, lanes, train_threads, scorer_procs, scorer_threads, screen_workers, worker_gb) <= 0:
        raise ValueError("resource counts must be positive")
    fixed = train_threads + 1 + scorer_procs * (scorer_threads + screen_workers)
    cpu_workers = (cpus - reserve) // lanes - fixed
    mem_workers = int((memory_gb * .8 / lanes - 4 - 2 * scorer_procs) / worker_gb)
    count = min(cpu_workers, mem_workers) if workers is None else workers
    fits = count >= 1 and count <= cpu_workers and count <= mem_workers
    count = max(1, count)
    return dict(fits=fits, workers_per_lane=count, lanes=lanes, cpu_count=cpus,
                fixed_cpus_per_lane=fixed, total_cpu_budget=reserve + lanes * (fixed + count),
                memory_gb=memory_gb, estimated_process_gb=lanes * (count * worker_gb + 4 + 2 * scorer_procs))


def main():
    from vaani.runtime import cpu_count
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lanes", type=int, required=True)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    # Respect cgroup memory limits, not just the host's MemAvailable.
    mem = next(int(x.split()[1]) * 1024 / 1e9 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    for limit, used in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            mem = min(mem, (int(Path(limit).read_text()) - int(Path(used).read_text())) / 1e9)
        except (OSError, ValueError):
            pass
    env = lambda k, d: int(os.environ.get(k, d))
    b = budget(cpu_count(), mem, a.lanes, env("RESERVE_CPUS", 8), env("TRAIN_THREADS", 2),
               env("SCORER_PROCS", 2), env("SCORER_THREADS", 4), env("SCORER_SCREEN_WORKERS", 2),
               float(os.environ.get("VAANI_WORKER_RSS_GB", 1)), a.workers)
    if a.json:
        print(json.dumps(b, indent=2))
    else:
        print(b["workers_per_lane"])
    if not b["fits"]:
        print("r8 resources do not fit: reduce lanes or scorer processes/threads", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
