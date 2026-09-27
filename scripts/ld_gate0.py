"""Gate 0a eligibility report (low-delay plan Task 0 / Section 4): D_proc per arm, each support's eligibility under
R0, R1 and R2, the registered support selection, and the simulated release schedule.

    python scripts/ld_gate0.py \
        --timing arm_a=results_r2/r8_ld/gate0/step_mini_p18.jsonl --timing arm_b=.../step_mini_p32.jsonl \
        --timing arm_r=.../step_arm_r.jsonl --cyclictest-max-us 85 --period-test .../period_test_48.json \
        --out results_r2/r8_ld/gate0/eligibility.json

Inputs (board measurements, run by the owner):
  --timing ARM=FILE[,FILE...]  vld_step_bench JSON lines (primary) and/or scripts/ld_step_timing.py reports. The
                               measured maximum is the largest whole-hop maximum over every input with FZ set (the
                               deployment setting); FZ-cleared runs are reported, not used. A Python-only arm is marked
                               so, because the C++ loop is the timing vehicle.
  --cyclictest-max-us          cyclictest maximum wake-up latency under the intended RT configuration.
  --period-test FILE           vld_period_test JSON. 1 ms periods count as verified only with pass, 48 frames and at
                               least 30 minutes on the D6 hardware (same card, no xruns).
  --converters-ms              0.5 (Section 4 upper estimate) until Gate 0b measures the converter path.
Without the board inputs the report is written with status "pending_board": the budget arithmetic, the resampler
records and the schedule simulation are complete, and every missing measurement is listed. Nothing is inferred.

D_proc = measured maximum + wake-up maximum + reserve (0.2 ms, fixed before measuring), rounded up to whole playback
periods; it never exceeds H. Budget upper estimate = L + resampler pair (D3: maximum group delay over 300-4,000 Hz,
or the impulse-peak delay if larger) + (D_proc + one period) + converters + 0.1 ms FIFO; eligible at <= 13.0 ms.
"""
import argparse, hashlib, json, math, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani import audio_contract as ac  # noqa: E402

LIMIT_MS = 13.0
FIFO_MS = 0.1
CONVERTERS_UPPER_MS = 0.5
RESERVE_MS = 0.2
PERIOD_TEST_MIN_S = 1800.0
RESAMPLERS = {"R0": "r0_linphase_kaiser193_v1", "R1": "r1_minphase_kaiser193_v1", "R2": "r2_cdelay_ls193_v1"}
ARMS = {"arm_a": "Mini-P18 at Arm A's contracts", "arm_b": "Mini-P32 at Arm B's contract",
        "arm_r": "Arm R's network at Arm A's contracts (reference; never eligible)"}
ARM_CONTRACTS = {"arm_a": list(ac.ARM_A_IDS), "arm_b": [ac.ARM_B_ID], "arm_r": list(ac.ARM_A_IDS)}
DEFAULT_RUNNER = ROOT / "native/vaani_ld/build/vaani_ld_run"


# ---- inputs ----------------------------------------------------------------------------------------------------------
def load_resamplers(rdir: Path = ROOT / "deploy/resampler") -> dict:
    out = {}
    for tag, rid in RESAMPLERS.items():
        p = rdir / f"{rid}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        dl = d["delays"]
        out[tag] = {"id": d["id"], "file": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                    "sha256": d["sha256"], "pair_peak_ms": dl["pair_peak_ms"],
                    "pair_group_delay_300_4000_max_ms": dl["pair_group_delay_300_4000_max_ms"],
                    "budget_ms": max(dl["pair_peak_ms"], dl["pair_group_delay_300_4000_max_ms"])}
    return out


def _timing_rows(path: Path) -> list[dict]:
    """Rows {source, contract, input, fz, whole_hop_max_ms, step_max_ms, ...} from either timing tool."""
    text = path.read_text(encoding="utf-8").strip()
    rows = []
    try:
        doc = json.loads(text)
        docs = [doc]
    except json.JSONDecodeError:
        docs = [json.loads(l) for l in text.splitlines() if l.strip()]
    for d in docs:
        if "error" in d:
            raise ValueError(f"{path}: the timing run failed: {d['error']}")
        if "graphs" in d:                                           # scripts/ld_step_timing.py
            for g in d["graphs"]:
                for r in g["runs"]:
                    rows.append({"source": "python", "file": str(path), "contract": g["audio_contract"],
                                 "input": r["input"], "fz": r["fz"], "paced": r["paced"], "hops": r["hops"],
                                 "whole_hop": r["whole_hop"], "step": r["step"], "resampler": None,
                                 "reportable": bool(d.get("reportable"))})
        elif "whole_hop" in d:                                      # native vld_step_bench
            rows.append({"source": "cpp", "file": str(path), "contract": d["contract"], "input": d["input"],
                         "fz": d["fz"], "paced": d["paced"], "hops": d["hops"], "whole_hop": d["whole_hop"],
                         "step": d["step"], "resampler": d.get("resampler"), "machine": d.get("machine"),
                         "allocations_after_warmup": d.get("allocations_after_warmup")})
        else:
            raise ValueError(f"{path}: neither a vld_step_bench nor an ld_step_timing record")
    return rows


def measured_max(arm: str, rows: list[dict]) -> dict:
    """The arm's processing maximum: C++ rows with FZ set when there are any (the timing vehicle), else Python."""
    want = set(ARM_CONTRACTS[arm])
    foreign = sorted({r["contract"] for r in rows} - want)
    if foreign:
        raise ValueError(f"{arm}: timing for contracts {foreign} that are not {arm}'s ({sorted(want)})")
    cpp = [r for r in rows if r["source"] == "cpp"]
    fz_on = [r for r in cpp if r["fz"]]
    py = [r for r in rows if r["source"] == "python"]
    used = fz_on or py or cpp               # FZ-cleared runs only when nothing else exists; flagged below
    if not used:
        raise ValueError(f"{arm}: no timing rows")
    basis = ("cpp, FZ set" if fz_on else "python only (not the timing vehicle)" if py
             else "cpp, FZ cleared only (not the deployment setting)")
    unpaced = [r["file"] for r in used if not r["paced"]]
    top = max(used, key=lambda r: r["whole_hop"]["max_ms"])
    return {"max_ms": top["whole_hop"]["max_ms"], "from": {k: top[k] for k in ("source", "file", "contract", "input")},
            "basis": basis,
            "inputs_seen": sorted({r["input"] for r in used}), "contracts_seen": sorted({r["contract"] for r in used}),
            "unpaced_files": unpaced,
            "missing_inputs": sorted({"random", "silent", "lowlevel"} - {r["input"] for r in used}),
            "includes_resampler": all(r.get("resampler") for r in used) if used[0]["source"] == "cpp" else False,
            "p999_ms": max(r["whole_hop"]["p999_ms"] for r in used), "p99_ms": max(r["whole_hop"]["p99_ms"] for r in used),
            "fz_cleared_max_ms": max((r["whole_hop"]["max_ms"] for r in cpp if not r["fz"]), default=None),
            "python_max_ms": max((r["whole_hop"]["max_ms"] for r in rows if r["source"] == "python"), default=None)}


def period_verdict(path: Path | None) -> dict:
    if path is None:
        return {"verified_1ms": False, "reason": "no period test supplied (Gate 0a: without it the support is 8 ms)"}
    d = json.loads(path.read_text(encoding="utf-8"))
    a = d.get("alsa", {})
    period = a.get("period", a.get("period_frames"))
    why = []
    if d.get("error"): why.append(f"period test failed: {d['error']}")
    if not d.get("pass"): why.append("pass is false (xruns or capture/playback on different cards)")
    if d.get("xruns", 1) != 0: why.append(f"{d.get('xruns')} xruns")
    if period != 48: why.append(f"negotiated period {period} frames, not 48")
    if float(d.get("seconds", 0)) < PERIOD_TEST_MIN_S: why.append(f"ran {d.get('seconds')} s, under {PERIOD_TEST_MIN_S:.0f} s")
    return {"verified_1ms": not why, "file": str(path), "reason": "; ".join(why) or "verified",
            "kernel": d.get("kernel"), "xruns": d.get("xruns"), "negotiated": a}


# ---- arithmetic --------------------------------------------------------------------------------------------------------
def dproc_periods(max_ms: float, wake_ms: float, reserve_ms: float, period_ms: float) -> int:
    return math.ceil((max_ms + wake_ms + reserve_ms) / period_ms - 1e-9)


def budget(cid: str, pair_ms: float, dproc_ms: float, period_ms: float, converters_ms: float) -> dict:
    c = ac.get_audio_contract(cid)
    parts = {"support_ms": 1e3 * c.support / c.sr, "resampler_pair_ms": pair_ms, "io_ms": dproc_ms + period_ms,
             "converters_ms": converters_ms, "fifo_ms": FIFO_MS}
    total = sum(parts.values())
    return {**parts, "total_ms": total, "eligible": total <= LIMIT_MS + 1e-9}


def select_support(arm_a_dproc_ms: dict, period: dict, rs: dict, conv: float, arm_b_dproc_ms: dict | None) -> dict:
    """Section 4's registered rule with R1 and each arm's own D_proc (dict period_ms -> D_proc ms, None = unknown)."""
    a10, a9, a8 = ac.ARM_A_IDS
    r1 = rs["R1"]["budget_ms"]
    trail = []
    if not period["verified_1ms"]:
        trail.append(f"1 ms periods not verified ({period['reason']}): L = 8 ms")
        chosen = a8
        periods = [2.0]
    else:
        periods = [1.0, 2.0]
        chosen = None
        d1 = arm_a_dproc_ms.get(1.0)
        if d1 is None:
            return {"status": "pending", "reason": "Arm A's D_proc is not measured", "trail": trail}
        for cid in (a10, a9):
            b = budget(cid, r1, d1, 1.0, conv)
            trail.append(f"{cid}: {b['total_ms']:.4f} ms with D_proc {d1:g} ms and 1 ms periods -> "
                         f"{'eligible' if b['eligible'] else 'not eligible'}")
            if b["eligible"]:
                chosen = cid; break
        if chosen is None:
            chosen = a8
            trail.append("neither L = 10 nor L = 9 ms is eligible: L = 8 ms")
    b8 = [budget(a8, r1, arm_a_dproc_ms[p], p, conv) for p in periods if arm_a_dproc_ms.get(p) is not None]
    mini_p18_fails_8ms = chosen == a8 and b8 and not any(b["eligible"] for b in b8)
    arm_b = {"piloted": False}
    if chosen == a10 and arm_b_dproc_ms and arm_b_dproc_ms.get(1.0) is not None:
        bb = budget(ac.ARM_B_ID, r1, arm_b_dproc_ms[1.0], 1.0, conv)
        arm_b = {"piloted": bb["eligible"], "budget": bb}
    elif chosen == a10:
        arm_b = {"piloted": False, "reason": "Arm B's own D_proc is not measured"}
    else:
        arm_b = {"piloted": False, "reason": "L = 10 ms was not selected"}
    return {"status": "selected", "support_contract": chosen,
            "support_ms": 1e3 * ac.get_audio_contract(chosen).support / 16000, "trail": trail, "arm_b": arm_b,
            "mini_p18_fails_8ms": bool(mini_p18_fails_8ms),
            "failure_action": ("Mini-P18 cannot meet L = 8 ms: not queued for low-delay training; return the measured "
                               "budget to the owner") if mini_p18_fails_8ms else None}


# ---- schedule simulation ---------------------------------------------------------------------------------------------
def simulate(runner: Path, cid: str, period: int, dproc: int, rfile: str, conv: float, hops: int,
             proc_ms: float | None = None, profile: str | None = None) -> dict:
    cmd = [str(runner), "simulate", "--contract", cid, "--period", str(period), "--dproc", str(dproc),
           "--hops", str(hops), "--resampler", str(ROOT / rfile), "--converters-ms", repr(conv)]
    cmd += ["--proc-ms", repr(proc_ms)] if proc_ms is not None else ["--profile", profile]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode not in (0, 3):
        raise RuntimeError(f"{' '.join(cmd)} exited {r.returncode}: {r.stdout}{r.stderr}")
    d = json.loads(r.stdout)
    res = d["result"]
    ok = res is not None and res["late_hops"] == res["late_periods"] == res["stale_periods"] == res["misplaced"] == 0
    return {"contract": cid, "period": period, "dproc_periods": dproc, "proc": proc_ms if proc_ms is not None else profile,
            "refused": d["budget"]["refused"], "result": res, "no_late_hop": bool(ok)}


def schedule(runner: Path, rs: dict, conv: float, arm_max: dict, hops: int) -> dict:
    """H = 96 and 128, D_proc 1 and 2 ms (1 ms periods), with the edge profile (every hop finishing exactly at
    D_proc) and, where measured, every hop taking the arm's measured maximum plus wake-up and reserve."""
    if not runner.exists():
        return {"status": "not_run", "reason": f"{runner} not built (cmake -S native/vaani_ld -B native/vaani_ld/build)"}
    rows = []
    for cid in (ac.ARM_A_IDS[0], ac.ARM_B_ID):
        c = ac.get_audio_contract(cid)
        for d in (1, 2):
            for prof in ("edge", "jitter", "early"):
                rows.append(simulate(runner, cid, 48, d, rs["R1"]["file"], conv, hops, profile=prof))
            e = simulate(runner, cid, 48, d, rs["R1"]["file"], conv, hops, profile="late")
            e["expect_recoveries"] = True       # the negative control: 1 hop in 1,000 overruns D_proc by half a period
            e["no_late_hop"] = e["result"] is not None and e["result"]["late_hops"] > 0 and e["result"]["misplaced"] == 0
            rows.append(e)
        for arm, m in arm_max.items():
            if cid in ARM_CONTRACTS[arm] and m.get("total_ms") is not None:
                d = m["dproc_periods"][1.0]
                if d * 48 <= 3 * c.hop:
                    rows.append({**simulate(runner, cid, 48, d, rs["R1"]["file"], conv, hops, proc_ms=m["total_ms"]),
                                 "arm": arm})
    return {"status": "run", "hops": hops, "rows": rows, "pass": all(r["no_late_hop"] for r in rows),
            "l_io_ms": {f"dproc_{d}ms": d + 1.0 for d in (1, 2)}}


# ---- report ----------------------------------------------------------------------------------------------------------
def report(timing: dict, wake_us: float | None, period_file: Path | None, converters_ms: float, converters_source: str,
           reserve_ms: float, runner: Path, hops: int) -> dict:
    rs = load_resamplers()
    period = period_verdict(period_file)
    wake_ms = None if wake_us is None else wake_us / 1e3
    arms = {}
    for arm, files in timing.items():
        rows = [r for f in files for r in _timing_rows(Path(f))]
        m = measured_max(arm, rows)
        if wake_ms is not None:
            m["wake_ms"], m["reserve_ms"] = wake_ms, reserve_ms
            m["total_ms"] = m["max_ms"] + wake_ms + reserve_ms
            hop_ms = min(1e3 * ac.get_audio_contract(c).hop / 16000 for c in ARM_CONTRACTS[arm])
            m["dproc_periods"] = {p: dproc_periods(m["max_ms"], wake_ms, reserve_ms, p) for p in (1.0, 2.0)}
            m["dproc_ms"] = {p: n * p for p, n in m["dproc_periods"].items()}
            m["dproc_within_hop"] = {p: v <= hop_ms + 1e-9 for p, v in m["dproc_ms"].items()}
        arms[arm] = m
    elig = []
    for arm in ("arm_a", "arm_b"):
        for cid in ARM_CONTRACTS[arm]:
            for tag, r in rs.items():
                for p in (1.0, 2.0):
                    d = arms.get(arm, {}).get("dproc_ms", {}).get(p)
                    # unmeasured: the plan's range, D_proc 1 or 2 ms in whole periods
                    for dproc in ([d] if d is not None else ([1.0, 2.0] if p == 1.0 else [2.0])):
                        b = budget(cid, r["budget_ms"], dproc, p, converters_ms)
                        if tag == "R0": b["eligible"] = False       # the control, never eligible
                        if p == 2.0 and ac.get_audio_contract(cid).support > 128: b["eligible"] = False
                        elig.append({"arm": arm, "contract": cid, "resampler": tag, "period_ms": p, "dproc_ms": dproc,
                                     "dproc_measured": d is not None, "period_verified": p == 2.0 or period["verified_1ms"],
                                     **b})
    arm_dp = lambda a: {p: v for p, v in arms.get(a, {}).get("dproc_ms", {}).items()} or {}
    sel = select_support(arm_dp("arm_a"), period, rs, converters_ms, arm_dp("arm_b") or None)
    missing = []
    for arm in ("arm_a", "arm_b", "arm_r"):
        if arm not in arms: missing.append(f"step timing for {arm} ({ARMS[arm]})")
        else:
            if arms[arm]["basis"] != "cpp, FZ set": missing.append(f"C++ timing with FZ set for {arm}")
            if arms[arm]["missing_inputs"]: missing.append(f"{arm} inputs {arms[arm]['missing_inputs']}")
            if arms[arm]["unpaced_files"]: missing.append(f"{arm}: paced runs (unpaced: {arms[arm]['unpaced_files']})")
    if wake_ms is None: missing.append("cyclictest maximum wake-up latency")
    if period_file is None: missing.append("vld_period_test on the D6 hardware (48-frame periods, >= 30 min)")
    if converters_source == "section4_upper_estimate": missing.append("Gate 0b converter delay (0.5 ms upper estimate in use)")
    out = {"generated_by": "python scripts/ld_gate0.py", "limit_ms": LIMIT_MS, "reserve_ms": reserve_ms,
           "converters_ms": converters_ms, "converters_source": converters_source, "fifo_ms": FIFO_MS,
           "resamplers": rs, "period_test": period, "wake_ms": wake_ms, "arms": arms,
           "eligibility": elig, "selection": sel,
           "schedule": schedule(Path(runner), rs, converters_ms, arms, hops),
           "missing": [m for m in missing if not m.startswith("Gate 0b")],
           "pending_gate0b": [m for m in missing if m.startswith("Gate 0b")]}
    out["status"] = "pending_board" if out["missing"] else "complete"
    if out["status"] != "complete":
        out["selection"] = {**sel, "provisional": True,
                            "note": "not a Gate 0a selection: board measurements are missing (see 'missing')"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timing", action="append", default=[], metavar="ARM=FILE[,FILE]")
    ap.add_argument("--cyclictest-max-us", type=float)
    ap.add_argument("--period-test", type=Path)
    ap.add_argument("--converters-ms", type=float, default=CONVERTERS_UPPER_MS)
    ap.add_argument("--converters-source", default=None, help="required when --converters-ms is not 0.5: the Gate 0b "
                                                              "measurement file or record")
    ap.add_argument("--reserve-ms", type=float, default=RESERVE_MS)
    ap.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    ap.add_argument("--hops", type=int, default=20000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.reserve_ms != RESERVE_MS:
        ap.error("the reserve is fixed before measuring (0.2 ms, Section 4); it is not a tuning knob")
    if a.converters_ms != CONVERTERS_UPPER_MS and not a.converters_source:
        ap.error("--converters-ms other than the 0.5 ms upper estimate needs --converters-source (a Gate 0b record)")
    timing = {}
    for t in a.timing:
        arm, _, files = t.partition("=")
        if arm not in ARMS or not files:
            ap.error(f"--timing {t!r}: expected one of {sorted(ARMS)}=FILE[,FILE]")
        timing.setdefault(arm, []).extend(files.split(","))
    src = a.converters_source or "section4_upper_estimate"
    if a.converters_source and Path(a.converters_source).exists():
        src = f"{a.converters_source} sha256:{hashlib.sha256(Path(a.converters_source).read_bytes()).hexdigest()[:16]}"
    rep = report(timing, a.cyclictest_max_us, a.period_test, a.converters_ms, src, a.reserve_ms, a.runner, a.hops)
    s = json.dumps(rep, indent=2, default=str)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(s + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("status", "missing", "pending_gate0b")} |
                     {"selection": {k: rep["selection"].get(k) for k in ("status", "support_contract", "provisional")},
                      "schedule_pass": rep["schedule"].get("pass")}, indent=2))
    return rep


if __name__ == "__main__":
    main()
