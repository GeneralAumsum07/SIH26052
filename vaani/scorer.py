"""Validation beside training (low-delay plan Task 4b, `perf.ops.scorer: async`).

Selector is the unchanged per-epoch validation and checkpoint-selection rule of vaani.train (validate(), the
composite screen, composite_key, best.pt), factored out so synchronous validation and the async scorer run the same
code. With `perf.ops.scorer: async` the trainer writes a snapshot at each validation point (raw and EMA state dicts,
step, epoch, the history index and the training fields of the history row; over-parameterized runs write folded
weights) and continues. One scorer per box (scripts/r8_scorer.py) scores snapshots in priority order, and in arrival
order within a priority, writes best.pt by the unchanged rule and keeps a log it resumes from after a restart
(no snapshot is scored twice or skipped). A run is DONE only when its last snapshot is scored: the scorer then merges
the validation fields into run.json, whose history equals the synchronous one.

Async scoring is refused while early_stopping.patience is set: then selection would feed back into training.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

SNAP_DIR = "snapshots"
STATE = "scorer_state.json"
TRAIN_DONE = "train_done.json"
SCORED_DONE = "scored_done.json"


def _atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


class Selector:
    """One validation point: validate the candidates, fill the history row's validation fields, write best.pt."""

    def __init__(self, cfg, run_dir, device, best=-1.0, best_key=None, tb=None):
        from vaani.train import COMPOSITE_DEFAULTS
        self.cfg, self.run_dir, self.device, self.tb = cfg, Path(run_dir), device, tb
        self.select = cfg.get("val", {}).get("select", "stoi")
        if self.select not in ("stoi", "composite"):
            raise ValueError(f"val.select must be stoi or composite, got {self.select!r}")
        self.plain_best = not (cfg.get("ema") or {}).get("decay") and self.select == "stoi"
        self.best, self.best_key, self.screen = best, best_key, None
        self.every = int(((cfg.get("val") or {}).get("composite") or {}).get("every", COMPOSITE_DEFAULTS["every"]))

    def _scalar(self, k, v, step):
        if self.tb is not None:
            self.tb.add_scalar(k, v, step)

    def point(self, row: dict, cands: dict, vdl, n_points: int, final: bool) -> float:
        """row: the history row (epoch, step, training fields); cands: {"raw": model[, "ema": model]}.
        n_points: this point's 1-based index in the history. Returns the raw val_stoi."""
        self.measure(row, cands, vdl, n_points, final)
        return self.choose(row, cands)

    def measure(self, row: dict, cands: dict, vdl, n_points: int, final: bool):
        """The point's validation numbers into row; no selection state or files, so any process may run it."""
        from vaani.train import CompositeScreen, validate
        cfg, device = self.cfg, self.device
        v = validate(cands["raw"], vdl, cfg, device)
        row.update(val_stoi=v, val_metrics=getattr(vdl, "_last_val_metrics", {"stoi": v}))
        if "ema" in cands:
            ve = validate(cands["ema"], vdl, cfg, device)
            row.update(val_stoi_ema=ve, val_metrics_ema=getattr(vdl, "_last_val_metrics", {"stoi": ve}))
        if self.select == "composite" and (n_points % self.every == 0 or final):
            if self.screen is None:
                t0 = time.time(); self.screen = CompositeScreen(cfg, self.run_dir)
                print(f"composite screen: {len(self.screen.items)} clip-conditions, baseline dSNR "
                      f"{self.screen.base_d_snr}, built in {time.time() - t0:.1f}s", flush=True)
            for name, m in cands.items():
                row[f"composite_{name}"] = self.screen.score(m, device)

    def choose(self, row: dict, cands: dict) -> float:
        """The unchanged selection rule on a measured row (writes best.pt); runs in history order."""
        from vaani.train import _save
        from vaani.training_controls import composite_key
        cfg, step, v = self.cfg, row["step"], row["val_stoi"]
        self._scalar("val/stoi", v, step)
        if "ema" in cands:
            self._scalar("val/stoi_ema", row["val_stoi_ema"], step)
        rd = self.run_dir
        if self.plain_best:
            if v > self.best:
                self.best = v
                _save({"model": cands["raw"].state_dict(), "config": cfg, "step": step}, rd / "best.pt")
        elif self.select == "stoi":
            scores = {"raw": v, "ema": row["val_stoi_ema"]}
            pick = max(scores, key=scores.get)   # raw wins ties
            if scores[pick] > self.best:
                self.best = scores[pick]
                _save({"model": cands[pick].state_dict(), "config": cfg, "step": step, "weights": pick,
                       "selection": "stoi"}, rd / "best.pt")
        elif "composite_raw" in row:   # measured at every `every`-th point and the final one
            for name, m in cands.items():
                s = row[f"composite_{name}"]; key = composite_key(s)
                self._scalar(f"val/pass_rate_{name}", s["pass_rate"], step)
                if self.best_key is None or key > self.best_key:
                    self.best_key, self.best = key, s["pass_rate"]
                    _save({"model": m.state_dict(), "config": cfg, "step": step, "weights": name,
                           "selection": "composite", "composite": s}, rd / "best.pt")
        return v


# ---- trainer side ---------------------------------------------------------------------------------
def _deployable_state(m):
    return (m.fold() if getattr(m, "overparam", False) else m).state_dict()


def write_snapshot(run_dir, index: int, row: dict, cands: dict, final: bool, priority: int = 2):
    """Snapshot of one validation point (index is 1-based), written atomically."""
    d = Path(run_dir) / SNAP_DIR
    d.mkdir(parents=True, exist_ok=True)
    folded = any(getattr(m, "overparam", False) for m in cands.values())
    obj = {"index": index, "row": dict(row), "final": bool(final), "priority": int(priority), "folded": folded,
           "models": {k: {n: t.detach().cpu() for n, t in _deployable_state(m).items()} for k, m in cands.items()}}
    p = d / f"snap_{index:05d}.pt"
    tmp = p.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, p)
    return p


def mark_train_done(run_dir, n_points: int):
    _atomic_json(Path(run_dir) / TRAIN_DONE, {"n_points": n_points, "time": time.time()})


# ---- scorer side ------------------------------------------------------------------------------------
class RunScorer:
    """Scoring state of one run: the Selector, its validation loader and the scored log."""

    def __init__(self, run_dir, device):
        self.run_dir = Path(run_dir)
        info = json.loads((self.run_dir / "run.json").read_text())
        self.cfg = info["config"]
        pat = (self.cfg.get("early_stopping") or {}).get("patience")
        if pat is not None:
            raise ValueError(f"{run_dir}: async scoring is not allowed with early_stopping.patience set")
        self.device = torch.device(device)
        st = self._load_state()
        self.scored = {int(k): v for k, v in st.get("rows", {}).items()}
        self.sel = Selector(self.cfg, self.run_dir, self.device, st.get("best", -1.0),
                            tuple(st["best_key"]) if st.get("best_key") is not None else None)
        self._vdl = None

    def _load_state(self):
        p = self.run_dir / STATE
        return json.loads(p.read_text()) if p.exists() else {}

    def _save_state(self):
        _atomic_json(self.run_dir / STATE, {"rows": {str(k): v for k, v in self.scored.items()}, "best": self.sel.best,
                                            "best_key": list(self.sel.best_key) if self.sel.best_key is not None else None})

    @property
    def vdl(self):
        if self._vdl is None:
            from vaani.train import build_val_loader
            self._vdl = build_val_loader(self.cfg, self.device, num_workers=0)
        return self._vdl

    def pending(self):
        d = self.run_dir / SNAP_DIR
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("snap_*.pt")):
            idx = int(p.stem.split("_")[1])
            if idx not in self.scored:
                out.append((idx, p))
        return out

    def _model(self, state, folded):
        from vaani.train import build_model
        mc = dict(self.cfg.get("model_cfg") or {})
        if folded:
            mc.pop("overparam", None)
        m = build_model(self.cfg["model"], model_cfg=mc)
        m.load_state_dict(state)
        return m.to(self.device).eval()

    # ---- parallel measuring: measure workers fill meas_*.json ahead; score() applies selection in order ----
    def _meas(self, idx):
        return self.run_dir / SNAP_DIR / f"meas_{idx:05d}.json"

    def _claim(self, idx) -> bool:
        """Take snapshot idx for measuring (one process per snapshot); a dead claimant's claim is taken over."""
        c = self.run_dir / SNAP_DIR / f"meas_{idx:05d}.claim"
        for _ in range(2):
            try:
                fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return True
            except FileExistsError:
                try:
                    pid = int(c.read_text() or 0)
                    os.kill(pid, 0)
                    return pid == os.getpid()
                except (ProcessLookupError, ValueError, FileNotFoundError):
                    c.unlink(missing_ok=True)   # stale claim (or mid-write): retry once
                except PermissionError:
                    return False
        return False

    def _measure_snapshot(self, idx, path):
        snap = torch.load(path, map_location="cpu", weights_only=True)
        cands = {k: self._model(s, snap["folded"]) for k, s in snap["models"].items()}
        row = dict(snap["row"])
        self.sel.measure(row, cands, self.vdl, snap["index"], snap["final"])
        return snap, cands, row

    def measure_ahead(self) -> bool:
        """Worker: measure one pending snapshot nobody has claimed. False when there is none."""
        st = self._load_state()
        done = {int(k) for k in st.get("rows", {})}
        for idx, p in self.pending():
            if idx in done or self._meas(idx).exists() or not self._claim(idx):
                continue
            try:
                snap, _, row = self._measure_snapshot(idx, p)
            except FileNotFoundError:   # scored and cleaned up meanwhile
                continue
            _atomic_json(self._meas(idx), {k: v for k, v in row.items() if k not in snap["row"]})
            return True
        return False

    def score(self, idx, path):
        """Selection for the next snapshot in history order, from its measure file when a worker wrote one.
        None: a live worker is still measuring it."""
        snap = torch.load(path, map_location="cpu", weights_only=True)
        if snap["index"] != len(self.scored) + 1:
            raise RuntimeError(f"{path}: snapshot {snap['index']} out of order (scored {len(self.scored)})")
        mp = self._meas(idx)
        if mp.exists():
            cands = {k: self._model(s, snap["folded"]) for k, s in snap["models"].items()}
            row = dict(snap["row"]); row.update(json.loads(mp.read_text()))
        elif self._claim(idx):
            _, cands, row = self._measure_snapshot(idx, path)
        else:
            return None
        self.sel.choose(row, cands)
        self.scored[idx] = {k: v for k, v in row.items() if k not in snap["row"]}
        self._save_state()
        mp.unlink(missing_ok=True); mp.with_suffix(".claim").unlink(missing_ok=True)
        return row

    def finalize(self) -> bool:
        """Merge the validation fields into run.json once training is done and every snapshot is scored."""
        done = self.run_dir / TRAIN_DONE
        if not done.exists() or (self.run_dir / SCORED_DONE).exists():
            return (self.run_dir / SCORED_DONE).exists()
        n = json.loads(done.read_text())["n_points"]
        if len(self.scored) < n or self.pending():
            return False
        info = json.loads((self.run_dir / "run.json").read_text())
        for i, row in enumerate(info["history"], 1):
            row.update(self.scored[i])
        info["best_val_stoi"] = self.sel.best
        if self.sel.best_key is not None:
            info["best_key"] = list(self.sel.best_key)
        info["scoring"] = "async"
        _atomic_json(self.run_dir / "run.json", info)
        _atomic_json(self.run_dir / SCORED_DONE, {"n_points": n, "time": time.time()})
        for pat in ("snap_*.pt", "meas_*.json", "meas_*.claim"):
            for p in (self.run_dir / SNAP_DIR).glob(pat):
                p.unlink(missing_ok=True)
        return True


class Scorer:
    """One scorer per box over many runs: priority class first (lower = sooner), then arrival order."""

    def __init__(self, run_dirs, device="cpu", priorities: dict | None = None):
        self.device = device
        self.priorities = {str(Path(k)): int(v) for k, v in (priorities or {}).items()}
        self.runs = {}
        for rd in run_dirs:
            self.add(rd)

    def add(self, run_dir):
        rd = Path(run_dir)
        if str(rd) not in self.runs and (rd / "run.json").exists():
            try:
                self.runs[str(rd)] = RunScorer(rd, self.device)
            except json.JSONDecodeError:   # run.json being written by the trainer: take it next poll
                pass

    def queue(self):
        items = []
        for key, rs in self.runs.items():
            for idx, p in rs.pending():
                pr = self.priorities.get(key)
                if pr is None:
                    try:
                        pr = int(torch.load(p, map_location="cpu", weights_only=True).get("priority", 2))
                    except Exception:   # a snapshot being written: take it next round
                        continue
                items.append((pr, p.stat().st_mtime, idx, key, p))
        items.sort(key=lambda t: (t[0], t[1], t[3], t[2]))
        # within a run, scoring must follow the history order
        seen, out = set(), []
        for it in items:
            if it[3] in seen:
                continue
            seen.add(it[3]); out.append(it)
        return out

    def step(self) -> bool:
        """Score one snapshot (the highest priority); finalize finished runs. False when idle."""
        q = self.queue()
        for rs in self.runs.values():
            rs.finalize()
        if not q:
            return False
        _, _, idx, key, p = q[0]
        rs = self.runs[key]
        nxt = min(i for i, _ in rs.pending())
        if rs.score(nxt, rs.run_dir / SNAP_DIR / f"snap_{nxt:05d}.pt") is None:
            return False   # a measure worker holds it: wait a poll
        rs.finalize()
        return True

    def measure_step(self) -> bool:
        """Measure worker: measure one snapshot ahead of the scorer. False when there is nothing to take."""
        return any(rs.measure_ahead() for rs in self.runs.values() if not (rs.run_dir / SCORED_DONE).exists())

    def run(self, poll_s=10.0, until_done=False, max_idle=None):
        idle = 0
        while True:
            if self.step():
                idle = 0
                continue
            if until_done and all((rs.run_dir / SCORED_DONE).exists() for rs in self.runs.values()):
                return
            idle += 1
            if max_idle is not None and idle >= max_idle:
                return
            time.sleep(poll_s)
