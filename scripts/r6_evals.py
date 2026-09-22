#!/usr/bin/env python3
"""r6 evaluation supervisor.

Twelve systems (six backbones, six cascades) plus the tier46 baseline, on two frozen sets, is about
four hours of scoring if it waits for the last model. Each eval is bound by its own main process
rather than the GPU, so several run side by side; this keeps MAX_PAR in flight and starts each the
moment its model is FINISHED.

Two things this has to get right, both learned the hard way:

  * "best.pt exists" does not mean training finished - best.pt is rewritten on every improvement, so
    scoring on it mid-run silently evaluates a half-trained model. A run counts as finished only when
    its log shows the last epoch of its configured budget.
  * "the .csv exists" does not mean the eval finished - vaani.eval streams rows as it goes, so an
    interrupted eval leaves a short file that looks like a result. A csv counts as done only when it
    has one row per item in the set.
"""
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/SIH26052")
OUT = ROOT / "results_r2/r6"
LOG = Path("/workspace/evals.log")
MAX_PAR = 3
WORKERS = 4
POLL = 60

EPOCHS = {"r6_e32": 32, "r6_ctl64": 64, "r6_e128": 128, "r6_e256": 256,
          "r6_demand64": 64, "r6_wham64": 64}
REFINER_EPOCHS = 32
SETS = [("eval_r2", ROOT / "data/eval_r2"), ("gen", ROOT / "data/eval_gen")]
EPOCH_RE = re.compile(r"^epoch (\d+)", re.M)


def log(msg):
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def finished(log_path: Path, budget: int) -> bool:
    if not log_path.exists():
        return False
    eps = EPOCH_RE.findall(log_path.read_text(errors="ignore"))
    return bool(eps) and max(int(e) for e in eps) >= budget - 1


def expected_rows(eval_root: Path) -> int:
    return len(list((eval_root / "test").rglob("*.json"))) + 1   # + header


def csv_done(csv: Path, want: int) -> bool:
    if not csv.exists():
        return False
    with csv.open() as f:
        return sum(1 for _ in f) >= want


def jobs():
    """(ready-predicate, --system, output name, eval root). Cascades first: they are the deployed
    system, so if time runs out it is a backbone number that goes missing, not a headline one."""
    base = ROOT / "results_r2/runs/vaani_tier46_refiner/best.pt"
    for tag, root in SETS:
        yield (lambda: base.exists()), f"cascade:{base}", f"tier46_{tag}", root
    for n in EPOCHS:
        for tag, root in SETS:
            yield ((lambda n=n: finished(OUT / f"train_{n}_refiner.log", REFINER_EPOCHS)),
                   f"cascade:runs/{n}_refiner/best.pt", f"{n}_cascade_{tag}", root)
    for n, budget in EPOCHS.items():
        for tag, root in SETS:
            yield ((lambda n=n, b=budget: finished(OUT / f"train_{n}.log", b)),
                   f"ckpt:runs/{n}/best.pt", f"{n}_{tag}", root)


def main():
    log("=== eval supervisor start ===")
    want = {str(root): expected_rows(root) for _, root in SETS}
    log("expected rows: " + str(want))
    pending, running, gpu = list(jobs()), [], 0
    while pending or running:
        for p, name in list(running):
            if p.poll() is not None:
                log(f"END   {name} rc={p.returncode}")
                running.remove((p, name))
        progressed = True
        while pending and len(running) < MAX_PAR and progressed:
            progressed = False
            for job in list(pending):
                ready, system, name, root = job
                csv = OUT / f"{name}.csv"
                if csv_done(csv, want[str(root)]):
                    log(f"SKIP  {name} (complete csv)")
                    pending.remove(job); progressed = True; break
                if not ready():
                    continue
                if csv.exists():
                    log(f"REDO  {name} (short csv, {csv.stat().st_size} B)")
                    csv.unlink()
                cmd = ["uv", "run", "python", "-m", "vaani.eval", "--system", system,
                       "--split", "test", "--eval-root", str(root), "--workers", str(WORKERS),
                       "--dnsmos", "--out", str(csv)]
                log(f"START {name} gpu={gpu} ({system})")
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=(OUT / f"eval_{name}.log").open("w"),
                                     stderr=subprocess.STDOUT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
                running.append((p, name)); gpu = 1 - gpu
                pending.remove(job); progressed = True
                break
        time.sleep(POLL)
    log("=== eval supervisor done ===")


if __name__ == "__main__":
    main()
