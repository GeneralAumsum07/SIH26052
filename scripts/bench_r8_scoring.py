"""Compare scorer processes vs threads on real validation audio and identical snapshots.

Creates disposable, seeded model snapshots (not trained quality results), checks
ordered history and best-checkpoint agreement, and times complete queue drains.
--limit bounds the frozen val screen; --limit 0 uses it in full. Composite scoring
is included at every fourth snapshot and the final one. Only writes under --out.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _worker(rd, threads, limit, screen_workers, ready, start, errors, selector):
    try:
        import torch
        from vaani.scorer import Scorer, SCORED_DONE
        from vaani.train_refiner import screen_items
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        os.environ["VAANI_SCREEN_WORKERS"] = str(screen_workers)
        sc = Scorer([rd], "cpu")
        rs = next(iter(sc.runs.values()))
        # Restrict only the benchmark's view, never the training config or split.
        cfg = rs.cfg
        ds, indices = screen_items(cfg["val"]["eval_root"], cfg["val"].get("split", "val"))
        rs.vdl._frozen_screen = (ds, indices[:limit] if limit else indices)
        ready.put(True)
        if not start.wait(120):
            raise TimeoutError("benchmark start barrier")
        while not (rd / SCORED_DONE).exists():
            if not (sc.step() if selector else sc.measure_step()):
                time.sleep(.01)
    except Exception:
        import traceback
        errors.put(traceback.format_exc())
        raise


def compare(reference, current, path="history"):
    """Different thread counts can change FP32 reduction rounding; selection must still match."""
    import math
    if isinstance(reference, dict):
        assert reference.keys() == current.keys(), path
        for k in reference:
            compare(reference[k], current[k], f"{path}.{k}")
    elif isinstance(reference, list):
        assert len(reference) == len(current), path
        for i, (x, y) in enumerate(zip(reference, current)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(reference, float):
        assert (math.isnan(reference) and math.isnan(current)) or math.isclose(reference, current, rel_tol=1e-6, abs_tol=1e-7), (path, reference, current)
    else:
        assert reference == current, (path, reference, current)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/retraining/r8_ld_fe_mini.yaml")
    ap.add_argument("--layouts", nargs="+", default=["1x8", "2x4", "4x2"], help="processes x threads per process")
    ap.add_argument("--snapshots", type=int, default=8)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--composite-limit", type=int, default=6)
    ap.add_argument("--screen-workers", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    import yaml
    from vaani import train, scorer
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg["val"]["composite"].update(limit=a.composite_limit or None)
    cfg["perf"]["ops"]["scorer"] = "async"
    torch.manual_seed(cfg["seed"])
    model = train.build_model(cfg["model"], model_cfg=cfg["model_cfg"]).eval()
    ctx = mp.get_context("spawn")
    rows, reference = [], None
    for layout in a.layouts:
        procs, threads = map(int, layout.split("x"))
        if min(procs, threads) < 1:
            ap.error("processes and threads must be positive")
        for repeat in range(a.repeats):
            rd = a.out / f"{layout}_{repeat}"
            rd.mkdir()
            history = [dict(epoch=i, step=i + 1) for i in range(a.snapshots)]
            (rd / "run.json").write_text(json.dumps(dict(config=cfg, history=history)))
            # Every layout sees exactly the same tensors and order. Small seeded
            # bias changes exercise checkpoint selection across snapshots.
            for i, row in enumerate(history):
                m = train.build_model(cfg["model"], model_cfg=cfg["model_cfg"]).eval()
                m.load_state_dict(model.state_dict())
                with torch.no_grad():
                    for name, p in m.named_parameters():
                        if name.endswith("bias"):
                            p.add_(i * .0001)
                scorer.write_snapshot(rd, i + 1, row, {"raw": m, "ema": m}, i == a.snapshots - 1)
            scorer.mark_train_done(rd, len(history))
            for pool in ("OMP", "MKL", "OPENBLAS", "NUMBA", "NUMEXPR"):
                os.environ[f"{pool}_NUM_THREADS"] = str(threads)
            os.environ["VAANI_SCREEN_WORKERS"] = str(a.screen_workers)
            ready, errors, start = ctx.Queue(), ctx.Queue(), ctx.Event()
            children = [ctx.Process(target=_worker, args=(rd, threads, a.limit, a.screen_workers, ready, start, errors, i == 0)) for i in range(procs)]
            t_start = time.perf_counter()
            print(f"scoring {layout}, repeat {repeat}", flush=True)
            try:
                for p in children:
                    p.start()
                for p in children:
                    ready.get(timeout=120)
                startup = time.perf_counter() - t_start
                t0 = time.perf_counter(); start.set()
                deadline = time.monotonic() + a.timeout
                for p in children:
                    p.join(max(0., deadline - time.monotonic()))
                if any(p.is_alive() or p.exitcode != 0 for p in children):
                    raise RuntimeError("scorer process failed or timed out")
                elapsed = time.perf_counter() - t0
                info = json.loads((rd / "run.json").read_text())
                ck = torch.load(rd / "best.pt", weights_only=True)
                result = dict(history=info["history"], step=ck["step"], weights=ck.get("weights"))
                if reference is None:
                    reference = (result, ck["model"])
                else:
                    compare(reference[0], result)
                    assert all(torch.equal(v, ck["model"][k]) for k, v in reference[1].items())
                row = dict(status="passed", layout=layout, repeat=repeat, seconds=elapsed,
                           startup_s=startup, snapshots_per_s=a.snapshots / elapsed, parity_passed=True)
            except Exception as e:
                row = dict(status="failed", layout=layout, repeat=repeat, reason=repr(e))
            finally:
                for p in children:
                    if p.is_alive():
                        p.terminate()
                    p.join(10)
            rows.append(row)
            print(json.dumps(row), flush=True)
            medians = {x: statistics.median(r["seconds"] for r in rows if r["layout"] == x) for x in a.layouts
                       if sum(r["layout"] == x for r in rows) == a.repeats and all(r["status"] == "passed" for r in rows if r["layout"] == x)}
            (a.out / "report.json").write_text(json.dumps(dict(config=a.config, limit=a.limit, composite_limit=a.composite_limit,
                snapshots=a.snapshots, screen_workers=a.screen_workers, cpu_count=os.cpu_count(), rows=rows,
                median_seconds=medians, winner=min(medians, key=medians.get) if medians else None), indent=2))


if __name__ == "__main__":
    main()
