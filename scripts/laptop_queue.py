"""Single-GPU training queue for the laptop (Windows or Linux; no tmux, no /dev/shm, no cgroups).

    python scripts/laptop_queue.py status                 # every queued run: DONE / RUNNING / PARTIAL / PENDING ...
    python scripts/laptop_queue.py run [--dry-run]        # train the queue top to bottom, resumable
    python scripts/laptop_queue.py score [--helpers N]    # only the async scorer (+ N measure helpers), until idle
    python scripts/laptop_queue.py stop                   # finish nothing more: the current run is stopped now

The queue file (configs/retraining/laptop_queue.txt) is reread before every run, so it can be edited mid-queue.
Runs live in $RUNS (r8_runs_final, where the box's runs came back): runs_dir is not a resume key, so the box's
interrupted pilots resume from their last.pt. Every run trains with perf.ops.scorer async, as on the box, so a
resumed run keeps the box's selection state (best.pt, scorer_state.json); one background scorer
(scripts/r8_scorer.py, CPU, below-normal priority) scores every run's snapshots in priority order.
Env: RUNS, LAPTOP_WORKERS (loader workers, default 10), SCREEN_WORKERS (2), TRAIN_THREADS (4), SCORER_THREADS (3),
SCORER_HELPERS (0: measure-only processes beside the scorer), MAX_TRIES (3).
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/retraining/laptop_queue.txt"
RUNS = Path(os.environ.get("RUNS", "r8_runs_final"))
Q = RUNS / "laptop_queue"
PY = sys.executable
WIN = os.name == "nt"
LOW = subprocess.BELOW_NORMAL_PRIORITY_CLASS if WIN else 0   # scorer must not starve the loader


def log(msg):
    line = f"{time.strftime('%F %T')} [laptop] {msg}"
    print(line, flush=True)
    with open(Q / "queue.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def caps(n):
    # BLAS/OMP/numba pools size to the core count otherwise: 16 threads per process x workers thrashes
    return {f"{k}_NUM_THREADS": str(n) for k in ("OMP", "MKL", "OPENBLAS", "NUMBA", "NUMEXPR")}


def parse_queue(path=QUEUE):
    """[(config, opts)] in file order; opts holds priority/workers and dotted config overrides."""
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cfg, *kv = line.split()
        opts = {}
        for item in kv:
            k, v = item.split("=", 1)
            try:
                opts[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                opts[k] = v
        out.append((cfg, opts))
    return out


def run_name(cfg_path):
    return yaml.safe_load(open(ROOT / cfg_path, encoding="utf-8"))["name"]


def alive(pid_file):
    try:
        pid = int(pid_file.read_text())
    except (OSError, ValueError):
        return False
    if WIN:   # os.kill(pid, 0) terminates on Windows; ask tasklist instead
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def state(name):
    rd = RUNS / name
    for flag in ("DONE", "STOPPED", "FAILED"):
        if (rd / flag).exists():
            return flag
    if (Q / f"running.{name}").exists() and alive(Q / f"running.{name}"):
        return "RUNNING"
    try:
        info = json.loads((rd / "run.json").read_text())
    except (OSError, ValueError):
        return "PENDING"
    if "end" in info:
        return "DONE"
    return f"PARTIAL step {info.get('steps', '?')}" if (rd / "last.pt").exists() else "PENDING"


def derived(cfg_path, opts):
    """The config as trained here: runs_dir -> RUNS plus the line's dotted overrides; written under Q/cfg."""
    c = yaml.safe_load(open(ROOT / cfg_path, encoding="utf-8"))
    c["runs_dir"] = str(RUNS)
    for k, v in opts.items():
        if k in ("priority", "workers"):
            continue
        d = c
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = v
    out = Q / "cfg" / f"{c['name']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(c, open(out, "w", encoding="utf-8"), sort_keys=False)
    return out, c


def scorer_runs():
    """Every run dir the scorer should own: has run.json, not STOPPED (a ruled-out run is not worth CPU)."""
    return sorted(str(p.parent) for p in RUNS.glob("*/run.json") if not (p.parent / "STOPPED").exists())


def ensure_scorer(helpers):
    """One scorer process (plus measure-only helpers), restarted if it died; picks up new run dirs on restart."""
    env = {**os.environ, **caps(int(os.environ.get("SCORER_THREADS", 3))),
           "VAANI_SCREEN_WORKERS": os.environ.get("SCREEN_WORKERS", "2")}
    # explicit dirs, not a glob: the STOPPED overparam run's P1 snapshots would otherwise jump the queue. r8_scorer.py
    # re-globs every entry each poll, so a queued run's dir that does not exist yet is picked up once it does
    pats = sorted(set(scorer_runs()) | {str(RUNS / run_name(c)) for c, _ in parse_queue()})
    for k in range(helpers + 1):
        pid_file = Q / (f"scorer.m{k}.pid" if k else "scorer.pid")
        if alive(pid_file):
            continue
        cmd = [PY, "scripts/r8_scorer.py", "--runs", *pats, "--device", "cpu", "--poll", "30"]
        if k:
            cmd.append("--measure-only")
        f = open(Q / "scorer.log", "a", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=LOW)
        pid_file.write_text(str(p.pid))
        log(f"scorer{' helper %d' % k if k else ''} started (pid {p.pid})")


def train_one(cfg_path, opts, dry):
    name = run_name(cfg_path)
    out, c = derived(cfg_path, opts)
    workers = int(opts.get("workers", os.environ.get("LAPTOP_WORKERS", 10)))
    ops = {"scorer": "async", "stream": "local", "priority": int(opts.get("priority", 2))}
    env = {**os.environ, **caps(int(os.environ.get("TRAIN_THREADS", 4))), "CUDA_VISIBLE_DEVICES": "0",
           "VAANI_WORKERS": str(workers), "VAANI_SCREEN_WORKERS": os.environ.get("SCREEN_WORKERS", "2"),
           "VAANI_PERF_OPS": json.dumps(ops)}
    cmd = [PY, "-m", "vaani.train", str(out)]
    if dry:
        print(f"DRY {name}: VAANI_WORKERS={workers} VAANI_PERF_OPS={json.dumps(ops)} {' '.join(cmd)}")
        return True
    rd = RUNS / name
    for tries in range(1, int(os.environ.get("MAX_TRIES", 3)) + 1):
        if (Q / "STOP").exists():
            return False
        log(f"start {name} (try {tries}, {workers} workers): {cfg_path}")
        with open(Q / "logs" / f"{name}.log", "a", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            (Q / f"running.{name}").write_text(str(p.pid))
            rc = p.wait()
        (Q / f"running.{name}").unlink(missing_ok=True)
        if (Q / "STOP").exists():
            log(f"{name} stopped by request (rc {rc}); rerun 'run' to resume it")
            return False
        try:
            done = "end" in json.loads((rd / "run.json").read_text())
        except (OSError, ValueError):
            done = False
        if rc == 0 and done:
            (rd / "DONE").write_text(time.strftime("%F %T") + "\n")
            log(f"DONE {name}")
            return True
        log(f"{name} exited rc={rc}; relaunching resumes from last.pt")
        time.sleep(30)
    (rd / "FAILED").write_text(f"rc={rc} after {tries} tries\n")
    log(f"FAILED {name}")
    return True   # a failed run does not stop the queue


def cmd_run(a):
    (Q / "STOP").unlink(missing_ok=True)
    if not a.dry_run:
        ensure_scorer(int(os.environ.get("SCORER_HELPERS", 0)))
    seen = set()
    while True:
        todo = [(c, o) for c, o in parse_queue() if run_name(c) not in seen
                and state(run_name(c)) not in ("DONE", "STOPPED", "FAILED", "RUNNING")]
        if not todo:
            break
        c, o = todo[0]
        seen.add(run_name(c))
        if not a.dry_run:
            ensure_scorer(int(os.environ.get("SCORER_HELPERS", 0)))
        if not train_one(c, o, a.dry_run):
            return 1
    log("queue finished; the scorer keeps scoring (status: python scripts/laptop_queue.py status)")
    return 0


def cmd_score(a):
    ensure_scorer(a.helpers)
    return 0


def cmd_stop(_):
    (Q / "STOP").write_text(time.strftime("%F %T") + "\n")
    for pid_file in Q.glob("running.*"):
        if alive(pid_file):
            pid = pid_file.read_text().strip()
            # train.py checkpoints every epoch; last.pt is write-then-rename, so a kill loses at most one epoch
            subprocess.run(["taskkill", "/T", "/F", "/PID", pid] if WIN else ["kill", "-TERM", pid])
            log(f"stopped {pid_file.name[8:]} (pid {pid})")
    print("queue stopped; the scorer is left running (kill its pid in", Q / "scorer.pid", "to stop it)")
    return 0


def cmd_status(_):
    for c, o in parse_queue():
        n = run_name(c)
        print(f"{state(n):<22} {n:<28} {c}{'  ' + str(o) if o else ''}")
    for k, pf in enumerate(sorted(Q.glob("scorer*.pid"))):
        print(f"{pf.stem}: {'alive' if alive(pf) else 'dead'}")
    pend = []
    for rd in scorer_runs():
        s, st = Path(rd) / "snapshots", Path(rd) / "scorer_state.json"
        n_snap = len(list(s.glob("snap_*.pt"))) if s.exists() else 0
        try:
            n_sc = len(json.loads(st.read_text())["rows"]) if st.exists() else 0
        except (OSError, ValueError, KeyError):
            n_sc = 0
        if n_snap and not (Path(rd) / "scored_done.json").exists():
            pend.append(f"{Path(rd).name} {n_snap} snapshots on disk, {n_sc} scored rows")
    print("unscored:", *pend, sep="\n  ") if pend else print("unscored: none")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--dry-run", action="store_true")
    s = sp.add_parser("score"); s.add_argument("--helpers", type=int, default=int(os.environ.get("SCORER_HELPERS", 0)))
    sp.add_parser("stop"); sp.add_parser("status")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    (Q / "logs").mkdir(parents=True, exist_ok=True)
    return {"run": cmd_run, "score": cmd_score, "stop": cmd_stop, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
