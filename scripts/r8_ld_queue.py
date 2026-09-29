"""The r8 low-delay queue (low-delay plan Task 8, Section 3.10): name resolution, eligibility, waves, decisions, lanes.

scripts/run_r8.sh registers the queued names (LD_P1..LD_P4, LD_FULL) and calls this helper; the helper never invents
a job. Each name resolves to exactly one configuration:
    low-delay arms and C0 seeds 2-4      configs/retraining/r8_ld_ablations/<name>.yaml
    C0 seeds 0/1                          configs/retraining/r8_ablations/<name>.yaml (shared with the legacy queue)
    full runs                             configs/retraining/<name>.yaml
and must be recorded in r8_ld_ablations/arms.json (gen_r8_configs.py --low-delay). A missing or ambiguous
configuration is an error: the queue stops instead of skipping it, and a legacy configuration never stands in for a
low-delay one. Marker directories are keyed by the queued name ($RUNS_DIR/<name>/DONE, as the legacy queue keeps
them) and training writes $RUNS_DIR/<config name>, so C0 seeds 0/1 share both with the legacy queue and neither
queue trains them twice.

Training is ungated by default (owner decision 2026-09-28). REQUIRE_TRAINING_VALIDATION=1 restores the previous
validation policy described below. Missing confirmation configurations cannot run until generated; generated jobs
need no measurement, promotion or full-run release. Explicit stop decisions still apply.

Strict eligibility: Arm B runs (arms.json needs_gate0_arm_b) when Gate 0a pilots it or the owner enables the persistent
training-only allow-unvalidated-arm override. The override does not change measured Pi eligibility. Waves: wave 2 (the
promoted recipe's confirmation seeds and full run) waits for the Stage-2 decision. Decisions (run_r8.sh ld-decide)
stop every run whose recipe they rule out (arms.json stop_if). Full runs need the guarded release: an explicit
authorization file and readiness evidence (a passing preflight JSON younger than LD_READY_MAX_H hours).

usage (what run_r8.sh calls):
    python scripts/r8_ld_queue.py plan  --phase pilots|full|all --gpus N --slots S [--json out]
    python scripts/r8_ld_queue.py next  --phase ... --gpu g --slot s        # claims and prints one job, or nothing
    python scripts/r8_ld_queue.py status
    python scripts/r8_ld_queue.py decide stage1 arm_a|arm_b | stage2 none|<ld_s2 stem>[,...]   # prints runs to stop
    python scripts/r8_ld_queue.py streams --phase ...                       # one config per rendered stream
Names come from --names (run_r8.sh passes its registry) or the LD_NAMES environment variable.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LD_OUT = "configs/retraining/r8_ld_ablations"
LEGACY_OUT = "configs/retraining/r8_ablations"
FULL_DIR = "configs/retraining"
ARMS_JSON = f"{LD_OUT}/arms.json"
GATE0 = "results_r2/r8_ld/gate0/eligibility.json"
WAVE2 = tuple(f"ld_conf_s{s}" for s in range(5)) + ("r8_ld_fe_mini_conf",)
DECISIONS = "decisions.json"
FULL_GO = "ld_full_go"
OVERRIDES = "training_overrides.json"
READY_MAX_H = float(os.environ.get("LD_READY_MAX_H", "24"))
# CPU priority classes (Section 3.10), highest first: loader-worker share and nice level per class
CLASS_SHARE = {1: 2.0, 2: 1.0, 3: 1.0, 4: 0.5}
CLASS_NICE = {1: 0, 2: 5, 3: 10, 4: 15}
# data-defining keys of a rendered stream (vaani.data.stream_server's signature covers these and more; a reader with a
# different signature renders locally, so grouping is an optimisation, never a correctness condition)
STREAM_KEYS = ("seed", "batch_size", "data", "dsp", "controller_on", "model_cfg.inputs", "perf.numerics.render")


class QueueError(RuntimeError):
    pass


def unvalidated_arms(runs_dir):
    """Read training permission separately from measured deployment eligibility."""
    p = Path(runs_dir) / "r8_queue" / OVERRIDES
    if not p.exists():
        return []
    arms = json.loads(p.read_text(encoding="utf-8")).get("allow_unvalidated_arms", [])
    if not isinstance(arms, list) or any(a != "arm_b" for a in arms):
        raise QueueError(f"{p}: only arm_b supports unvalidated training")
    return arms


def _get(cfg, dotted):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def network_of(cfg):
    """Network label without torch (mirrors vaani_fe.profile_of for the named tiers)."""
    mc = cfg.get("model_cfg") or {}
    tier = mc.get("tier", "mini")
    if mc.get("freq_windows"):
        return f"{tier}_{mc['freq_windows']}"
    if mc.get("df_bins") is not None:
        return f"{tier}_df{mc['df_bins']}"
    return tier


def stream_key(cfg):
    d = {k: _get(cfg, k) for k in STREAM_KEYS}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


class Queue:
    def __init__(self, root, runs_dir, names, gate0=None, ready=None):
        self.root = Path(root)
        self.runs = Path(runs_dir)
        self.q = self.runs / "r8_queue"
        self.names = list(names)
        # Owner policy: validation is diagnostic by default, including on a fresh training box.
        self.require_validation = os.environ.get("REQUIRE_TRAINING_VALIDATION", "0") == "1"
        if len(set(self.names)) != len(self.names):
            dup = sorted({n for n in self.names if self.names.count(n) > 1})
            raise QueueError(f"names registered twice: {dup}")
        self.gate0_path = Path(gate0) if gate0 else self.root / GATE0
        self.ready_path = Path(ready) if ready else self.q / "preflight_ld.json"
        am = self.root / ARMS_JSON
        if not am.exists():
            raise QueueError(f"{ARMS_JSON} missing: run python scripts/gen_r8_configs.py --low-delay")
        self.man = json.loads(am.read_text(encoding="utf-8"))
        self.rec = {Path(a["config"]).stem: a for a in self.man["arms"]}
        self.g0 = self._gate0()
        self.decisions = self._decisions()
        self.unvalidated = unvalidated_arms(self.runs)

    def allow_unvalidated_arm(self, arm):
        if arm != "arm_b":
            raise QueueError("only arm_b supports unvalidated training")
        self.q.mkdir(parents=True, exist_ok=True)
        self.unvalidated = [arm]
        (self.q / OVERRIDES).write_text(json.dumps(dict(
            allow_unvalidated_arms=self.unvalidated, training_only=True,
            written=time.strftime("%Y-%m-%dT%H:%M:%S"),
            reason="Owner authorized training before Pi validation"), indent=2) + "\n", encoding="utf-8")

    def arm_b_allowed(self):
        return (not self.require_validation or
                bool(((self.g0.get("selection") or {}).get("arm_b") or {}).get("piloted")) or "arm_b" in self.unvalidated)

    # ---- inputs ----------------------------------------------------------------------------------------------
    def _gate0(self):
        if not self.gate0_path.exists():
            if not self.require_validation:
                return dict(status="missing", selection=dict(provisional=True,
                            support_contract=self.man["gate0"]["support_contract"]))
            raise QueueError(f"Gate 0a record {self.gate0_path} missing: no low-delay compute before Gate 0a")
        g0 = json.loads(self.gate0_path.read_text(encoding="utf-8"))
        sel = g0.get("selection") or {}
        if self.require_validation and sel.get("support_contract") != self.man["gate0"]["support_contract"]:
            raise QueueError(f"Gate 0a selects {sel.get('support_contract')}, the configs were generated for "
                             f"{self.man['gate0']['support_contract']}: regenerate (gen_r8_configs.py --low-delay)")
        return g0

    def gate0_ready(self):
        sel = self.g0.get("selection") or {}
        if self.g0.get("status") != "complete" or sel.get("provisional"):
            return False, f"Gate 0a record is {self.g0.get('status')} (provisional selection): board measurements missing"
        if sel.get("mini_p18_fails_8ms"):
            return False, "Gate 0a: Mini-P18 fails L = 8 ms (back to the owner)"
        return True, f"Gate 0a complete: {sel.get('support_contract')}"

    def _decisions(self):
        p = self.q / DECISIONS
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def readiness(self):
        """Full-run release: explicit authorization and fresh readiness evidence (a passing preflight JSON)."""
        if not self.require_validation:
            return True, "training gates disabled; validation is diagnostic"
        if not (self.q / FULL_GO).exists():
            return False, f"not authorized: run 'bash scripts/run_r8.sh ld-go-full' ({self.q / FULL_GO} missing)"
        if not self.ready_path.exists():
            return False, f"readiness evidence {self.ready_path} missing (r8_preflight.py --low-delay --json)"
        age_h = (time.time() - self.ready_path.stat().st_mtime) / 3600
        j = json.loads(self.ready_path.read_text(encoding="utf-8"))
        if age_h > READY_MAX_H:
            return False, f"readiness evidence is stale ({age_h:.1f} h > {READY_MAX_H:g} h): re-run the preflight"
        if j.get("status") != "pass":
            return False, f"readiness evidence {self.ready_path} does not pass ({j.get('status')})"
        return True, "released"

    # ---- resolution --------------------------------------------------------------------------------------------
    def resolve(self, name):
        """(config path, record) of one queued name; raises on a missing or ambiguous configuration."""
        cands = [f"{LD_OUT}/{name}.yaml", f"{LEGACY_OUT}/{name}.yaml", f"{FULL_DIR}/{name}.yaml"]
        found = [c for c in cands if (self.root / c).exists()]
        rec = self.rec.get(name)
        if rec is None:
            raise QueueError(f"{name}: not recorded in {ARMS_JSON} (not a registered low-delay job)")
        if len(found) > 1:
            raise QueueError(f"{name}: ambiguous, {found}")
        if not found or found[0] != rec["config"]:
            raise QueueError(f"{name}: configuration {rec['config']} missing (the queue fails; nothing substitutes)")
        cfg = yaml.safe_load((self.root / found[0]).read_text(encoding="utf-8"))
        if cfg["name"] != rec["name"]:
            raise QueueError(f"{name}: {found[0]} names {cfg['name']}, arms.json {rec['name']}: regenerate")
        return found[0], cfg, rec

    def _stop_reason(self, rec):
        for cond in rec.get("stop_if", []):
            key, _, val = cond.partition("!=") if "!=" in cond else cond.partition("=")
            neg = "!=" in cond
            got = self.decisions.get(key)
            if got is None:
                continue
            got_s = "none" if got in ([], "none") else ",".join(got) if isinstance(got, list) else got
            if (got_s != val) if neg else (got_s == val):
                return f"ruled out by the {key} decision ({got_s}): {cond}"
        return None

    def state(self, name):
        d = self.runs / name
        if (d / "DONE").exists():
            return "DONE"
        pid = self.q / f"running.{name}"
        if pid.exists():
            try:
                os.kill(int(pid.read_text().strip()), 0)
                return "RUNNING"
            except (OSError, ValueError):
                pass
        if (d / "STOPPED").exists():
            return "STOPPED"
        if (d / "FAILED").exists():
            return "FAILED"
        return "PENDING"

    def jobs(self, phase):
        """Every registered name of the phase with its resolution and why it would or would not start."""
        out = []
        promoted = self.decisions.get("stage2")
        for i, n in enumerate(self.names):
            if n in WAVE2 and not self.require_validation and n not in self.rec:
                out.append(dict(name=n, order=i, status="NOT_GENERATED",
                                why="no configuration: generate a confirmation recipe with gen_r8_configs.py --low-delay --promote"))
                continue
            if n in WAVE2 and self.require_validation:
                if promoted is None:
                    out.append(dict(name=n, order=i, status="WAITING", why="wave 2: waits for the Stage-2 decision"))
                    continue
                if promoted in ([], "none"):
                    out.append(dict(name=n, order=i, status="NOT_NEEDED", why="Stage 2 promoted nothing"))
                    continue
                if n == "r8_ld_fe_mini_conf" and promoted == ["ld_s2_overparam"]:
                    out.append(dict(name=n, order=i, status="NOT_NEEDED",
                                    why="the early overparam full run is the promoted recipe's full run"))
                    continue
                gen = sorted(self.man.get("promoted") or [])
                if gen != sorted(promoted):
                    if gen:   # wave 2 generated for another recipe: never train it under this decision
                        raise QueueError(f"wave 2 was generated for {gen}, the Stage-2 decision promoted {promoted}: "
                                         f"regenerate (gen_r8_configs.py --low-delay --promote {','.join(promoted)})")
                    out.append(dict(name=n, order=i, status="WAITING",
                                    why=f"promoted {','.join(promoted)}: generate wave 2 first "
                                        f"(python scripts/gen_r8_configs.py --low-delay --promote {','.join(promoted)})"))
                    continue
            c, cfg, rec = self.resolve(n)
            if rec.get("queued") is False:
                raise QueueError(f"{n}: {c} is not a queued configuration ({rec.get('stage')})")
            full = rec["kind"] == "full"
            if (phase == "pilots" and full) or (phase == "full" and not full):
                continue
            j = dict(name=n, order=i, config=c, cfg_name=cfg["name"], train_dir=str(self.runs / cfg["name"]),
                     network=network_of(cfg), contract=rec["audio_contract"], arm=rec["arm"], stage=rec["stage"],
                     priority=int(rec["priority"]), wave=int(rec["wave"]), full=full,
                     speculative=bool(rec.get("speculative")), stream=stream_key(cfg), epochs=cfg["epochs"])
            # Both B's main runs and its n_hat ablation share the owner's training override.
            j["unvalidated"] = (not self.require_validation and (not self.gate0_ready()[0] or
                                bool(rec.get("needs_gate0_arm_b") and not
                                     ((self.g0.get("selection") or {}).get("arm_b") or {}).get("piloted")))) or bool(
                                rec.get("needs_gate0_arm_b") and "arm_b" in self.unvalidated)
            st = self.state(n)
            why = self._stop_reason(rec)
            if rec.get("needs_gate0_arm_b") and not self.arm_b_allowed():
                j.update(status="NOT_ELIGIBLE", why="Arm B runs only when Gate 0a selects L = 10 ms and pilots it")
            elif why and st != "DONE":
                j.update(status="STOPPED" if st != "RUNNING" else "RUNNING", why=why, stop=True)
            else:
                j.update(status=st, why=f"unvalidated {rec['arm']}: training only; Pi eligibility unverified" if j["unvalidated"] else "")
            out.append(j)
        seen = {}
        for j in out:
            if "train_dir" in j:
                if j["train_dir"] in seen:
                    raise QueueError(f"{j['name']} and {seen[j['train_dir']]} write the same run directory")
                seen[j["train_dir"]] = j["name"]
        return out

    # ---- lanes -------------------------------------------------------------------------------------------------
    def runnable(self, phase):
        ok_g0, why_g0 = self.gate0_ready()
        rel, why_rel = self.readiness()
        js = []
        for j in sorted(self.jobs(phase), key=lambda j: (j.get("wave", 9), j.get("priority", 9), j["order"])):
            if j["status"] not in ("PENDING", "FAILED"):
                continue
            if self.require_validation and not ok_g0 and not j["unvalidated"]:
                j = dict(j, status="BLOCKED", why=why_g0)
            elif j["full"] and not rel:
                j = dict(j, status="BLOCKED", why=why_rel)
            js.append(j)
        return js

    def workers(self, prio, lanes, ncpu=None, reserve=4, meminfo="/proc/meminfo", rss_gb=1.0, screen=8):
        """Loader workers of one run by its priority class: a class-weighted share of the cores, capped by memory."""
        if not ncpu:   # lazy: vaani.runtime pulls in torch, too slow for every queue call
            sys.path.insert(0, str(ROOT))
            from vaani import runtime
            ncpu = runtime.cpu_count()   # the CFS quota, not nproc
        base = max(1.0, (ncpu - reserve) / max(1, lanes))
        w = max(1, int(base * CLASS_SHARE[prio]))
        try:
            kb = next(int(l.split()[1]) for l in open(meminfo) if l.startswith("MemAvailable:"))
            cap = int((kb * 1024 / 1e9 / rss_gb - screen) / max(1, lanes) / 2)
            w = max(1, min(w, cap))
        except (OSError, StopIteration, ValueError):
            pass
        return w

    def plan(self, phase, gpus, slots, **wk):
        """Static preview of the lane assignment in priority order (the live queue claims dynamically)."""
        lanes = [(g, s) for s in range(slots) for g in range(gpus)]
        fulls = {g: 0 for g in range(gpus)}
        rows, k = [], 0
        for j in self.runnable(phase):
            if j["full"]:   # full runs on different GPUs where possible
                g = min(fulls, key=lambda x: (fulls[x], x)); fulls[g] += 1
                lane = (g, 0)
            else:
                lane = lanes[k % len(lanes)]; k += 1
            rows.append(dict(j, gpu=lane[0], slot=lane[1], workers=self.workers(j["priority"], len(lanes), **wk),
                             nice=CLASS_NICE[j["priority"]]))
        return rows

    def claim_next(self, phase, gpu, gpus, **wk):
        """Claim the highest-priority runnable job for a lane on `gpu` (full runs only on their own GPU)."""
        (self.q / "claims").mkdir(parents=True, exist_ok=True)
        full_rank = 0
        for j in self.runnable(phase):
            if j["status"] == "BLOCKED":
                continue
            if j["full"]:
                home = full_rank % gpus; full_rank += 1
                if home != gpu:
                    continue
            try:
                os.mkdir(self.q / "claims" / j["name"])
            except FileExistsError:
                continue
            return dict(j, gpu=gpu, workers=self.workers(j["priority"], wk.pop("lanes", gpus), **wk),
                        nice=CLASS_NICE[j["priority"]])
        return None

    def decide(self, what, value):
        if what == "stage1":
            if value not in ("arm_a", "arm_b"):
                raise QueueError("stage1 is arm_a or arm_b (Arm R is never selected)")
            if value == "arm_b" and not self.arm_b_allowed():
                raise QueueError("Arm B was not piloted at this support")
            self.decisions["stage1"] = value
            self.decisions["stage1_unvalidated"] = not self.gate0_ready()[0] or (value == "arm_b" and
                ("arm_b" in self.unvalidated or not ((self.g0.get("selection") or {}).get("arm_b") or {}).get("piloted")))
        elif what == "stage2":
            v = [] if value == "none" else sorted(x for x in value.split(",") if x)
            bad = [x for x in v if not x.startswith("ld_s2_")]
            if bad:
                raise QueueError(f"not Stage-2 arms: {bad}")
            self.decisions["stage2"] = v
        else:
            raise QueueError("decide stage1|stage2")
        self.q.mkdir(parents=True, exist_ok=True)
        (self.q / DECISIONS).write_text(json.dumps(self.decisions, indent=2) + "\n", encoding="utf-8")
        return [j["name"] for j in self.jobs("all") if j.get("stop")]

    def streams(self, phase):
        """One configuration per rendered stream: the member with the most epochs (the server serves every epoch)."""
        best = {}
        for j in self.runnable(phase):
            if j["status"] == "BLOCKED":
                continue
            if "stream" not in j or j["status"] in ("DONE", "STOPPED", "NOT_ELIGIBLE", "NOT_NEEDED"):
                continue
            b = best.get(j["stream"])
            if b is None or j["epochs"] > b["epochs"]:
                best[j["stream"]] = j
        return best


def main(argv=None):
    # Bash consumes tab/newline records; Windows CRLF otherwise becomes part of job names and paths.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "next", "status", "decide", "streams", "resolve",
                                    "allow-unvalidated-arm", "check-start"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", "runs"))
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--phase", default="all", choices=["pilots", "full", "all"])
    ap.add_argument("--gpus", type=int, default=int(os.environ.get("LD_GPUS", "1")))
    ap.add_argument("--slots", type=int, default=int(os.environ.get("LD_SLOTS", "1")))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gate0", default=os.environ.get("GATE0_JSON"))
    ap.add_argument("--ready", default=os.environ.get("LD_READY_JSON"))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    names = a.names if a.names is not None else os.environ.get("LD_NAMES", "").split()
    wk = dict(meminfo=os.environ.get("MEMINFO", "/proc/meminfo"),
              rss_gb=float(os.environ.get("VAANI_WORKER_RSS_GB", "1") or 1),
              screen=int(os.environ.get("VAANI_SCREEN_WORKERS", "8")), reserve=int(os.environ.get("RESERVE_CPUS", "4")))
    try:
        q = Queue(a.root, a.runs_dir, names, a.gate0, a.ready)
        if a.cmd == "allow-unvalidated-arm":
            if len(a.args) != 1:
                ap.error("allow-unvalidated-arm arm_b")
            q.allow_unvalidated_arm(a.args[0])
            print(f"Enabled unvalidated arm_b training: {q.q / OVERRIDES}; Pi validation remains pending")
            return 0
        if a.cmd == "check-start":
            # The shell must use the same per-job decision as the claimant, not parse a gate's prose.
            return 0 if any(j["status"] in ("PENDING", "FAILED") for j in q.runnable(a.phase)) else 1
        if a.cmd == "resolve":
            for n in a.args or names:
                c, cfg, rec = q.resolve(n)
                print(f"{n}\t{c}\t{cfg['name']}\t{network_of(cfg)}\t{rec['audio_contract']}")
            return 0
        if a.cmd == "plan":
            rows = q.plan(a.phase, a.gpus, a.slots, **wk)
            for r in rows:
                print(f"gpu{r['gpu']}.{r['slot']}\tP{r['priority']}\tw{r['wave']}\t{r['status']}\t{r['name']}\t"
                      f"{r['config']}\t{r['network']}\t{r['contract']}\t{r['train_dir']}\tworkers={r['workers']}"
                      + (f"\t{r['why']}" if r.get("why") else ""))
            skipped = [j for j in q.jobs(a.phase) if j["status"] in ("WAITING", "NOT_ELIGIBLE", "NOT_NEEDED", "STOPPED", "NOT_GENERATED")]
            for j in skipped:
                print(f"-\t-\t-\t{j['status']}\t{j['name']}\t{j['why']}")
            ok, why = q.gate0_ready()
            print(f"# {why}")
            if not q.require_validation:
                print("# Training gates disabled for all arms; validation is diagnostic")
            if q.unvalidated:
                print(f"# Unvalidated training override: {','.join(q.unvalidated)} (training only)")
            if a.json:
                Path(a.json).write_text(json.dumps(dict(rows=rows, skipped=skipped, gate0=why), indent=2))
            return 0
        if a.cmd == "next":
            j = q.claim_next(a.phase, a.gpu, a.gpus, lanes=a.gpus * a.slots, **wk)
            if j:   # tab-separated for bash `read`
                print("\t".join(str(j[k]) for k in ("name", "config", "cfg_name", "priority", "workers", "nice",
                                                    "full", "speculative")))
            return 0
        if a.cmd == "status":
            for j in q.jobs("all"):
                print(f"  {j['status']:<12} {j['name']:<26} {j.get('why', '')}")
            print(f"# {q.gate0_ready()[1]}; full-run release: {q.readiness()[1]}")
            return 0
        if a.cmd == "decide":
            if len(a.args) != 2:
                ap.error("decide stage1 arm_a|arm_b  or  decide stage2 none|<stems>")
            for n in q.decide(*a.args):
                print(n)
            return 0
        if a.cmd == "streams":
            for key, j in sorted(q.streams(a.phase).items()):
                print(f"{key}\t{j['config']}\t{j['name']}")
            return 0
    except QueueError as e:
        print(f"LD QUEUE ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
