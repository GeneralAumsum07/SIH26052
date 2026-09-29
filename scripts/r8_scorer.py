"""One validation scorer per box (low-delay plan Task 4b, `perf.ops.scorer: async`).

Scores the snapshots that async runs write at each validation point, in priority order (the run's perf.ops.priority,
overridable with --priority), then arrival order; writes best.pt by the unchanged selection rule and merges the
validation history into each run's run.json once its last snapshot is scored. Restart-safe: it resumes from each
run's scorer_state.json without re-scoring or skipping a snapshot. VAANI_SCREEN_WORKERS sizes its one CPU pool.
--measure-only: a helper beside the scorer that measures later snapshots in parallel (meas_*.json); the scorer still
applies the selection rule in history order, so the results equal one scorer's.

usage: python scripts/r8_scorer.py --runs 'runs/r8_ld_*' [--device cuda] [--priority runs/r8_ld_fe_mini=1] [--until-done]
"""
import argparse
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from vaani.scorer import Scorer
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories or globs (re-expanded every poll)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--priority", nargs="*", default=[], help="run_dir=class (1 highest)")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--until-done", action="store_true", help="exit once every run is scored and merged")
    ap.add_argument("--measure-only", action="store_true", help="measure snapshots ahead; exits when runs are merged")
    a = ap.parse_args(argv)
    pr = dict(x.split("=", 1) for x in a.priority)
    sc = Scorer([], a.device, pr)
    done = lambda: sc.runs and all((r.run_dir / "scored_done.json").exists() for r in sc.runs.values())
    while a.measure_only:
        for pat in a.runs:
            for rd in sorted(glob.glob(pat)):
                sc.add(rd)
        if done():
            return
        if not sc.measure_step():
            time.sleep(a.poll)
    while True:
        for pat in a.runs:
            for rd in sorted(glob.glob(pat)):
                sc.add(rd)
        sc.run(a.poll, until_done=a.until_done, max_idle=1)
        if a.until_done and done():
            return


if __name__ == "__main__":
    main()
