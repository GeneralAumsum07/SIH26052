"""Matched arm comparison for the low-delay r8 (plan Task 9): the registered D2 uncertainty model, the decision rule,
the Stage-2 screen, the frame-rate modulation index, blinded listening examples and the recipe freeze.

    python scripts/compare_r8_ld.py register --c0-s0 C0_S0.csv --c0-s1 C0_S1.csv [--approved-by NAME] \
        --out results_r2/r8_ld/compare/registry.json
    python scripts/compare_r8_ld.py compare MANIFEST.json --registry results_r2/r8_ld/compare/registry.json \
        --out results_r2/r8_ld/compare/<name>
    python scripts/compare_r8_ld.py modindex --wavs OUT_WAVS --clean CLEAN_WAVS --hops 96 128 --out mod.csv
    python scripts/compare_r8_ld.py listen LISTEN.json --out results_r2/r8_ld/listening/<name> [--seed N]
    python scripts/compare_r8_ld.py freeze SELECTION.json --out results_r2/r8_ld/selection/<tier>.json

Inputs are per-clip tables, one per (arm, seed), scored by the unified runner on the frozen validation split: the
`vaani.eval` / `scripts/eval_refvalid.py` columns, plus `speech_loss` (the eval_refvalid definition) and, where
measured, `dnsmos_ovrl` and the `mod_idx_hop<N>_db` / `mod_idx_clean_hop<N>_db` columns written by `modindex`. A clip
is keyed by bucket/id (plus condition in reference-condition tables) and clustered by scene (`scene` column, else
input SNR and id, as `vaani.report.scene_clusters`).

The manifest names the comparison, its kind and both arms' runs:

    {"comparison": "ld_vs_c0" | "c0_vs_r7" | "arm_a_vs_arm_r" | "final_vs_r7",
     "kind": "screen" | "stage1" | "confirmation" | "full",
     "population": {...},                        # identical in every run: split, eval root, route, sample rate
     "disclosed_exposure": {"<key>": "reason"},  # every exposure key allowed to differ between the arms
     "seed_var_from": "<confirmation report.json>",   # kind "full" only
     "candidate": {"arm": "arm_a", "hop": 96, "runs": [RUN, ...]},
     "control":   {"arm": "c0", "hop": 256, "fixed_reference": false, "runs": [RUN, ...]}}
    RUN = {"seed": 0, "csv": "...", "recipe": "...", "contract_id": "...", "metric_defs": {...},
           "population": {...}, "exposure": {"epochs", "batch", "optimizer_steps", "scored_s", "prefix_s"},
           "numerics": {...}, "runtime": {"step_ms": .., "peak_mem_mb": ..}, "composite": ..}

Uncertainty (Section 2.9, registered): per seed s, d_s = mean over clips of (candidate - control); d_bar = mean_s d_s;
Var(d_bar) = s_d^2/S + V_clip, with s_d never below the D2 floor and V_clip the scene-clustered bootstrap variance of
the seed-averaged paired difference; 95 % interval d_bar +- t(0.975, S-1) sqrt(Var). Seeds are paired: the same seed
gives the same mixture stream. A single frozen reference (r7) pairs with every candidate seed. A single-seed full run
takes s_d and the degrees of freedom from its five-seed confirmation report.

Decision (per metric, oriented so that positive is better; lower-is-better metrics are mirrored): non-inferior when the
lower bound >= -margin, superior when it is > 0, inferior when the upper bound < -margin, otherwise inconclusive.
Overall results gate; the transient and reference-fault subsets gate only on "inferior". DNSMOS, the modulation index,
runtime and memory are reported, never gated. Every number in a report is measured from the listed tables.
"""
import argparse, hashlib, json, math, random, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000
TARGETS = {"snr_out": 15.0, "stoi": 0.85, "pesq_wb": 2.5}   # vaani.report.TARGETS: the all-three pass
# D2 (proposed; owner-approved at Gate A). Floors are between-seed standard deviations (Section 2.9's largest
# historical spreads); speech loss's floor is registered from C0 seeds 0/1 by `register`.
D2 = {
    "snr_out":     {"margin": 0.25,  "floor": 0.053,  "higher_better": True,  "unit": "dB"},
    "stoi":        {"margin": 0.005, "floor": 0.0009, "higher_better": True,  "unit": ""},
    "pesq_wb":     {"margin": 0.05,  "floor": 0.017,  "higher_better": True,  "unit": ""},
    "pass3":       {"margin": 2.0,   "floor": 0.8,    "higher_better": True,  "unit": "pp"},
    "speech_loss": {"margin": 0.005, "floor": None,   "higher_better": False, "unit": ""},
}
SPEECH_LOSS_FLOOR_MIN = 0.002
METRIC_DEFS = {
    "snr_out": "vaani.metrics.snr_db(clean, est), 16 kHz, no resampler",
    "stoi": "vaani.metrics.stoi(clean, est), 16 kHz",
    "pesq_wb": "vaani.metrics.pesq_wb(clean, est), 16 kHz",
    "pass3": "100 * (snr_out > 15 and stoi > 0.85 and pesq_wb > 2.5) per clip, clean clips excluded",
    "speech_loss": "share of speech-active 20 ms frames with projected speech gain below -15 dB (eval_refvalid)",
}
SUBSETS = ("overall", "transient", "ref_fault")
REPORTED = ("dnsmos_ovrl",)
MOD_SUBSETS = ("stationary", "clean")
EXPOSURE_KEYS = ("epochs", "batch", "optimizer_steps", "scored_s", "prefix_s")
RUN_KEYS = ("seed", "csv", "recipe", "contract_id", "metric_defs", "population", "exposure")
CONFIRMATION_MIN_SEEDS = 5
COMPARISONS = {   # name -> (allowed candidate arms, allowed control arms, control may be one frozen reference)
    "c0_vs_r7": ({"c0"}, {"r7"}, True),
    "ld_vs_c0": ({"arm_a", "arm_b", "arm_r", "ld_conf"}, {"c0"}, False),
    "arm_a_vs_arm_r": ({"arm_a"}, {"arm_r"}, False),
    "final_vs_r7": ({"arm_a", "arm_b", "ld_conf", "final"}, {"r7"}, True),
}
KINDS = ("screen", "stage1", "confirmation", "full")
N_BOOT, BOOT_SEED = 2000, 9
LISTEN_CATEGORIES = ("clean_speech", "onsets", "stationary_noise", "transients", "talker_leakage", "ref_reconnect")
# Absolute gates a recipe needs before it is frozen (Section 6). Quality comparisons never stand in for them.
FREEZE_GATES = ("gate0a", "gate_a", "spec62", "export")


class ComparisonError(ValueError):
    """The inputs cannot be compared; every reason is listed."""

    def __init__(self, reasons):
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        super().__init__("; ".join(self.reasons))


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(argv) -> dict:
    diff = _git("diff", "HEAD")
    return {"revision": _git("rev-parse", "HEAD"), "dirty_diff_sha256": _sha_text(diff) if diff else None,
            "command": " ".join(argv), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# ---- per-clip tables -------------------------------------------------------------------------------------------------

def _flag(df, col):
    return df[col].fillna(False).astype(str).str.lower().isin(("true", "1", "1.0"))


def load_table(path) -> "pd.DataFrame":
    """One (arm, seed) table: key, scene cluster, subset flags and the registered metrics. Rejects duplicate keys and
    non-finite registered metrics."""
    import pandas as pd
    df = pd.read_csv(path, dtype={"id": str, "bucket": str})
    miss = [c for c in ("id", "bucket", "snr_out", "stoi", "pesq_wb", "speech_loss") if c not in df]
    if miss:
        raise ComparisonError(f"{path}: missing columns {miss}")
    key = df.bucket.astype(str) + "/" + df.id.astype(str)
    if "condition" in df:
        key = key + "|" + df.condition.astype(str)
    df["key"] = key
    if df.key.duplicated().any():
        raise ComparisonError(f"{path}: duplicate clip keys {df.key[df.key.duplicated()].head(3).tolist()}")
    snr = df.snr_in if "snr_in" in df else pd.Series("", index=df.index)
    df["scene"] = df.scene.astype(str) if "scene" in df else snr.astype(str) + "|" + df.id.astype(str)
    nc = df.get("noise_class", pd.Series("", index=df.index)).fillna("").astype(str)
    fault = df.get("fault", pd.Series(np.nan, index=df.index)).fillna("").astype(str)
    clean = nc.eq("clean") | df.bucket.astype(str).str.startswith("clean")
    df["subset_clean"] = _flag(df, "subset_clean") if "subset_clean" in df else clean
    df["subset_transient"] = (_flag(df, "subset_transient") if "subset_transient" in df else
                              nc.str.contains("impulsive") | df.get("impulse_peak_db", pd.Series(np.nan, index=df.index)).notna())
    ref = fault.str.startswith("fault_ref")
    if "ref_dropout" in df:
        ref |= _flag(df, "ref_dropout")
    if "condition" in df:
        ref |= df.condition.astype(str).ne("present")
    df["subset_ref_fault"] = _flag(df, "subset_ref_fault") if "subset_ref_fault" in df else ref
    df["subset_stationary"] = (_flag(df, "subset_stationary") if "subset_stationary" in df else
                               nc.eq("stationary") & fault.eq("") & ~df.subset_ref_fault)
    df["subset_overall"] = True
    reg = df[["snr_out", "stoi", "pesq_wb", "speech_loss"]].to_numpy(float)
    if not np.isfinite(reg).all():
        bad = df.key[~np.isfinite(reg).all(1)].head(3).tolist()
        raise ComparisonError(f"{path}: non-finite registered metrics on {bad}")
    ok = np.ones(len(df), bool)
    for m, t in TARGETS.items():
        ok &= df[m].to_numpy(float) > t
    df["pass3"] = np.where(df.subset_clean, np.nan, 100.0 * ok)
    return df.set_index("key", drop=False)


def keys_sha(df) -> str:
    return _sha_text("\n".join(sorted(df.key)))


# ---- the registry (D2 margins and floors) ----------------------------------------------------------------------------

def register(c0_s0, c0_s1, approved_by=None) -> dict:
    """Fix the metric list, margins, floors and definitions before any low-delay arm is scored. The speech-loss floor is
    max(0.002, |mean C0 s1 - mean C0 s0|) over the clips both seeds scored."""
    a, b = load_table(c0_s0), load_table(c0_s1)
    if set(a.key) != set(b.key):
        raise ComparisonError("C0 seed 0 and seed 1 tables cover different clips")
    diff = abs(float(b.loc[a.key, "speech_loss"].mean() - a.speech_loss.mean()))
    d2 = json.loads(json.dumps(D2))
    d2["speech_loss"]["floor"] = max(SPEECH_LOSS_FLOOR_MIN, diff)
    return {"schema": "r8_ld_d2_registry_v1", "status": "approved" if approved_by else "proposed",
            "approved_by": approved_by, "metrics": d2, "metric_defs": METRIC_DEFS, "subsets": list(SUBSETS),
            "confirmation_min_seeds": CONFIRMATION_MIN_SEEDS, "speech_loss_floor_from": {
                "c0_s0": str(c0_s0), "c0_s1": str(c0_s1), "abs_mean_difference": diff,
                "c0_s0_sha256": hashlib.sha256(Path(c0_s0).read_bytes()).hexdigest(),
                "c0_s1_sha256": hashlib.sha256(Path(c0_s1).read_bytes()).hexdigest()},
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def registry_sha(reg: dict) -> str:
    return _sha_text(json.dumps({k: reg[k] for k in ("metrics", "metric_defs", "subsets", "confirmation_min_seeds")},
                                sort_keys=True))


def check_registry(reg: dict) -> list:
    """Every registered metric needs a positive finite margin and floor; the metric list and definitions are D2's."""
    bad = []
    if not isinstance(reg, dict) or reg.get("schema") != "r8_ld_d2_registry_v1":
        return ["registry missing or of the wrong schema"]
    mets = reg.get("metrics") or {}
    for m in D2:
        for f in ("margin", "floor"):
            v = (mets.get(m) or {}).get(f)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                bad.append(f"registry: {m} has no registered {f}")
        if m in mets and mets[m].get("higher_better") != D2[m]["higher_better"]:
            bad.append(f"registry: {m} direction differs from D2")
    extra = sorted(set(mets) - set(D2))
    if extra:
        bad.append(f"registry: unregistered metrics {extra}")
    if reg.get("metric_defs") != METRIC_DEFS:
        bad.append("registry: metric definitions differ from D2's")
    if reg.get("confirmation_min_seeds", 0) < CONFIRMATION_MIN_SEEDS:
        bad.append(f"registry: confirmation needs at least {CONFIRMATION_MIN_SEEDS} seeds")
    return bad


# ---- validation ------------------------------------------------------------------------------------------------------

def _load_runs(side, who, bad):
    tables = {}
    runs = side.get("runs") or []
    if not runs:
        bad.append(f"{who}: no runs")
    for r in runs:
        miss = [k for k in RUN_KEYS if k not in r]
        if miss:
            bad.append(f"{who} seed {r.get('seed')}: missing {miss}")
            continue
        if r["seed"] in tables:
            bad.append(f"{who}: seed {r['seed']} listed twice")
            continue
        try:
            tables[r["seed"]] = load_table(r["csv"])
        except (ComparisonError, OSError) as e:
            bad.append(f"{who} seed {r['seed']}: {e}")
    return tables


def validate(man: dict, reg: dict) -> dict:
    """Refuse incompatible metric definitions, unmatched populations or clip IDs, undisclosed exposure differences,
    mismatched configurations, unpaired seeds and too few seeds for the kind. Returns the loaded tables."""
    bad = check_registry(reg)
    comp, kind = man.get("comparison"), man.get("kind")
    if comp not in COMPARISONS:
        bad.append(f"unknown comparison {comp!r}")
    if kind not in KINDS:
        bad.append(f"unknown kind {kind!r}")
    cand, ctrl = man.get("candidate") or {}, man.get("control") or {}
    if comp in COMPARISONS:
        ok_c, ok_k, fixed_ok = COMPARISONS[comp]
        if cand.get("arm") not in ok_c:
            bad.append(f"{comp}: candidate arm {cand.get('arm')!r} not in {sorted(ok_c)}")
        if ctrl.get("arm") not in ok_k:
            bad.append(f"{comp}: control arm {ctrl.get('arm')!r} not in {sorted(ok_k)}")
        if ctrl.get("fixed_reference") and not fixed_ok:
            bad.append(f"{comp}: the control must be seed-paired, not a fixed reference")
    tc, tk = _load_runs(cand, "candidate", bad), _load_runs(ctrl, "control", bad)
    fixed = bool(ctrl.get("fixed_reference"))
    runs = [("candidate", r) for r in cand.get("runs") or []] + [("control", r) for r in ctrl.get("runs") or []]
    runs = [(w, r) for w, r in runs if all(k in r for k in RUN_KEYS)]
    # metric definitions and population
    for w, r in runs:
        if r["metric_defs"] != reg.get("metric_defs"):
            bad.append(f"{w} seed {r['seed']}: metric definitions differ from the registry")
        if r["population"] != man.get("population"):
            bad.append(f"{w} seed {r['seed']}: population {r['population']} differs from the manifest's")
    all_t = list(tc.values()) + list(tk.values())
    if all_t:
        ref = set(all_t[0].key)
        for w, t in [("candidate", s) for s in tc] + [("control", s) for s in tk]:
            got = set((tc if w == "candidate" else tk)[t].key)
            if got != ref:
                bad.append(f"{w} seed {t}: clip IDs do not match ({len(got ^ ref)} unmatched)")
        want = (man.get("population") or {}).get("clip_keys_sha256")
        if want and keys_sha(all_t[0]) != want:
            bad.append("clip IDs differ from the population's registered clip set")
    # configuration: one recipe, contract and numerics per arm; seeds paired
    for who, side in (("candidate", cand), ("control", ctrl)):
        rs = [r for r in side.get("runs") or [] if all(k in r for k in RUN_KEYS)]
        for f in ("recipe", "contract_id"):
            if len({json.dumps(r[f], sort_keys=True) for r in rs}) > 1:
                bad.append(f"{who}: runs differ in {f}")
        if len({json.dumps(r.get("numerics"), sort_keys=True) for r in rs}) > 1:
            bad.append(f"{who}: runs differ in perf.numerics")
    if not fixed and ctrl.get("arm") != "r7" and cand.get("arm") != "r7":
        nums = {json.dumps(r.get("numerics"), sort_keys=True) for _, r in runs}
        if len(nums) > 1:
            bad.append("candidate and control differ in perf.numerics")
    if fixed:
        if len(tk) != 1:
            bad.append("a fixed reference control has exactly one run")
    elif tc and tk and set(tc) != set(tk):
        bad.append(f"seeds are not paired: candidate {sorted(tc)} vs control {sorted(tk)}")
    # exposure: every difference disclosed
    disclosed = man.get("disclosed_exposure") or {}
    kruns = {r["seed"]: r for r in ctrl.get("runs") or [] if all(k in r for k in RUN_KEYS)}
    for r in cand.get("runs") or []:
        if not all(k in r for k in RUN_KEYS):
            continue
        k = next(iter(kruns.values()), None) if fixed else kruns.get(r["seed"])
        if k is None:
            continue
        for e in EXPOSURE_KEYS:
            if e not in r["exposure"] or e not in k["exposure"]:
                bad.append(f"seed {r['seed']}: exposure {e} not disclosed")
            elif r["exposure"][e] != k["exposure"][e] and not disclosed.get(e):
                bad.append(f"seed {r['seed']}: undisclosed exposure difference in {e} "
                           f"({r['exposure'][e]} vs {k['exposure'][e]})")
    # seed counts
    S = len(tc)
    if kind == "screen" and (S != 1 or (not fixed and len(tk) != 1)):
        bad.append("a Stage-2 screen pairs exactly one seed per arm")
    if kind == "stage1" and S < 2:
        bad.append("a Stage-1 comparison needs at least 2 seeds per arm")
    if kind == "confirmation" and (S < CONFIRMATION_MIN_SEEDS or (not fixed and len(tk) < CONFIRMATION_MIN_SEEDS)):
        bad.append(f"a confirmation needs at least {CONFIRMATION_MIN_SEEDS} seeds per arm "
                   f"(candidate {S}, control {len(tk)})")
    if kind == "full":
        if S != 1:
            bad.append("a full-exposure comparison is single-seed")
        if not man.get("seed_var_from"):
            bad.append("a full-exposure comparison takes its between-seed variance from the confirmation "
                       "(seed_var_from)")
    if kind in ("screen", "confirmation", "full") and man.get("candidate", {}).get("hop") is None:
        bad.append("candidate hop (samples at 16 kHz) is required for the modulation index")
    if bad:
        raise ComparisonError(bad)
    return {"candidate": tc, "control": tk, "fixed": fixed}


# ---- the registered uncertainty model and decision rule --------------------------------------------------------------

def t_quantile(df: int) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, df))


def cluster_boot_var(e, clusters, n=N_BOOT, seed=BOOT_SEED) -> float:
    """Scene-clustered bootstrap variance of the mean of e: whole scenes are resampled."""
    e = np.asarray(e, float); c = np.asarray(clusters)
    keep = np.isfinite(e); e, c = e[keep], c[keep]
    if len(e) < 2:
        return float("nan")
    _, inv = np.unique(c, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=e), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    return float(np.var(sums[idx].sum(1) / cnts[idx].sum(1), ddof=1))


def paired(tc, tk, fixed, col, subset):
    """Per-seed clip differences (candidate - control) on a subset: (d_s per seed, seed-averaged clip difference, scenes)."""
    seeds = sorted(tc)
    ref = next(iter(tk.values()))
    mask = ref[f"subset_{subset}"].to_numpy(bool)
    keys = ref.key.to_numpy()[mask]
    rows = []
    for s in seeds:
        k = ref if fixed else tk[s]
        rows.append(tc[s].loc[keys, col].to_numpy(float) - k.loc[keys, col].to_numpy(float))
    D = np.vstack(rows) if rows else np.zeros((0, len(keys)))
    fin = np.isfinite(D).all(0)
    D = D[:, fin]
    return D.mean(1) if D.shape[1] else np.full(len(seeds), np.nan), D.mean(0), ref.scene.to_numpy()[mask][fin], D.shape[1]


def interval(d_s, e, scenes, floor, seed_var=None) -> dict:
    """d_bar +- t(0.975, df) sqrt(s_d^2/S + V_clip); s_d >= floor. seed_var = (s_d, df) from a confirmation, for S = 1."""
    d_s = np.asarray(d_s, float)
    S = len(d_s)
    d_bar = float(d_s.mean())
    if seed_var is not None:
        s_obs, df = seed_var
    else:
        if S < 2:
            raise ComparisonError("an interval needs at least 2 seeds or a confirmation's between-seed variance")
        s_obs, df = float(d_s.std(ddof=1)), S - 1
    if floor is None or not math.isfinite(floor) or floor <= 0:
        raise ComparisonError("no registered between-seed floor")
    s_used = max(s_obs, floor)
    v_clip = cluster_boot_var(e, scenes)
    var = s_used ** 2 / S + (v_clip if math.isfinite(v_clip) else 0.0)
    t = t_quantile(df)
    h = t * math.sqrt(var)
    return {"S": S, "df": df, "d_seeds": [float(x) for x in d_s], "d_bar": d_bar, "s_d_observed": float(s_obs),
            "s_d_used": s_used, "floor": floor, "floor_binding": s_obs < floor, "v_clip": v_clip, "t": t,
            "lo": d_bar - h, "hi": d_bar + h}


def decide(lo: float, hi: float, margin: float, higher_better: bool = True) -> str:
    """The registered rule on the candidate - control interval; a lower-is-better metric is mirrored first."""
    if margin is None or not math.isfinite(margin) or margin <= 0:
        raise ComparisonError("no registered margin")
    if not higher_better:
        lo, hi = -hi, -lo
    if lo > 0:
        return "superior"
    if lo >= -margin:
        return "non_inferior"
    if hi < -margin:
        return "inferior"
    return "inconclusive"


def _seed_var(path, metric, subset, reg):
    rep = json.loads(Path(path).read_text())
    if rep.get("kind") != "confirmation" or rep.get("registry_sha256") != registry_sha(reg):
        raise ComparisonError(f"{path}: not a confirmation report under this registry")
    r = rep["registered"][metric][subset]
    if r["S"] < CONFIRMATION_MIN_SEEDS:
        raise ComparisonError(f"{path}: confirmation with {r['S']} seeds")
    return r["s_d_used"], r["S"] - 1


def mod_cols(hop):
    return f"mod_idx_hop{hop}_db", f"mod_idx_clean_hop{hop}_db"


def compare(man: dict, reg: dict, argv=()) -> dict:
    """The full comparison report. Raises ComparisonError when the inputs cannot be compared."""
    t = validate(man, reg)
    tc, tk, fixed = t["candidate"], t["control"], t["fixed"]
    kind = man["kind"]
    ref = next(iter(tk.values()))
    rep = {"schema": "r8_ld_comparison_v1", "comparison": man["comparison"], "kind": kind,
           "candidate": man["candidate"]["arm"], "control": man["control"]["arm"], "fixed_reference": fixed,
           "seeds": sorted(tc), "n_clips": len(ref), "clip_keys_sha256": keys_sha(ref),
           "population": man.get("population"), "disclosed_exposure": man.get("disclosed_exposure") or {},
           "registry_sha256": registry_sha(reg), "registry_status": reg.get("status"),
           "provenance": provenance(argv), "registered": {}, "reported": {}, "runtime": {}}
    mets = reg["metrics"]
    for m, spec in mets.items():
        rep["registered"][m] = {}
        for sub in SUBSETS:
            d_s, e, sc, n = paired(tc, tk, fixed, m, sub)
            if n == 0:
                rep["registered"][m][sub] = {"n_clips": 0, "status": "no_clips"}
                continue
            if kind == "screen":
                d = float(d_s[0])
                worse = (-d if spec["higher_better"] else d) > spec["margin"]
                rep["registered"][m][sub] = {"n_clips": n, "d": d, "worse_than_margin": bool(worse)}
                continue
            sv = _seed_var(man["seed_var_from"], m, sub, reg) if kind == "full" else None
            r = interval(d_s, e, sc, spec["floor"], sv)
            r.update(n_clips=n, margin=spec["margin"], higher_better=spec["higher_better"], unit=spec["unit"],
                     status=decide(r["lo"], r["hi"], spec["margin"], spec["higher_better"]),
                     gating="non_inferiority" if sub == "overall" else "inferior_only")
            rep["registered"][m][sub] = r
    # reported, never gated: DNSMOS, the modulation index, runtime and memory
    for m in REPORTED:
        if all(m in x for x in list(tc.values()) + list(tk.values())):
            d_s, e, sc, n = paired(tc, tk, fixed, m, "overall")
            if n:
                rep["reported"][m] = {"n_clips": n, "d_seeds": [float(x) for x in d_s], "d_bar": float(np.mean(d_s)),
                                      "clip_boot_sd": math.sqrt(max(cluster_boot_var(e, sc), 0.0))}
    hop = man["candidate"].get("hop")
    if hop is not None:
        out_c, clean_c = mod_cols(hop)
        have = all(out_c in x and clean_c in x for x in list(tc.values()) + list(tk.values()))
        rep["reported"]["modulation_index"] = {"hop": hop, "hop_hz": SR / hop, "measured": have}
        if have:
            for sub in MOD_SUBSETS:
                vs_ctrl, _, _, n = paired(tc, tk, fixed, out_c, sub)
                mask = ref[f"subset_{sub}"].to_numpy(bool)
                vs_clean = [float(np.nanmean(tc[s].loc[ref.key[mask], out_c].to_numpy(float)
                                             - tc[s].loc[ref.key[mask], clean_c].to_numpy(float))) for s in sorted(tc)]
                rep["reported"]["modulation_index"][sub] = {
                    "n_clips": n, "db_vs_control": float(np.mean(vs_ctrl)) if n else None,
                    "db_vs_clean": float(np.mean(vs_clean)) if n else None}
    for who, side in (("candidate", man["candidate"]), ("control", man["control"])):
        rt = [r.get("runtime") or {} for r in side["runs"]]
        keys = sorted({k for x in rt for k in x})
        rep["runtime"][who] = {k: float(np.mean([x[k] for x in rt if k in x])) for k in keys}
    if kind == "screen":
        comp = [r.get("composite") for r in man["candidate"]["runs"]] + [r.get("composite") for r in man["control"]["runs"]]
        worse = [m for m in mets if rep["registered"][m]["overall"].get("worse_than_margin")]
        if None in comp:
            rep["screen"] = {"promote": False, "reason": "composite score missing"}
        else:
            dc = float(comp[0] - comp[1])
            ok = dc >= 0 and not worse
            rep["screen"] = {"promote": ok, "d_composite": dc, "worse_than_margin": worse,
                             "reason": "promoted" if ok else ("composite decreased" if dc < 0 else f"worse than margin: {worse}")}
        return rep
    status = {m: rep["registered"][m]["overall"].get("status") for m in mets}
    subs_inferior = [f"{m}/{s}" for m in mets for s in SUBSETS[1:] if rep["registered"][m][s].get("status") == "inferior"]
    rep["summary"] = {
        "improvements": [m for m, s in status.items() if s == "superior"],
        "non_inferior": [m for m, s in status.items() if s == "non_inferior"],
        "regressions": [m for m, s in status.items() if s == "inferior"] + subs_inferior,
        "uncertain": [m for m, s in status.items() if s == "inconclusive"],
        "subsets_inferior": subs_inferior}
    ok = all(s in ("superior", "non_inferior") for s in status.values()) and not subs_inferior
    if kind == "confirmation" and man["comparison"] == "ld_vs_c0":
        rep["gate_b_quality"] = {"pass": ok, "reason": "every overall metric non-inferior or better, no subset inferior"
                                 if ok else "needs an explicit owner decision (D1/D2) stating the gap, with Arm R beside it"}
    elif kind == "full":
        rep["gate_c_full"] = {"not_inferior": not [m for m, s in status.items() if s == "inferior"] and not subs_inferior}
    return rep


def markdown(rep: dict) -> str:
    L = [f"# {rep['comparison']} ({rep['kind']}): {rep['candidate']} vs {rep['control']}", "",
         f"Seeds {rep['seeds']}{' against one fixed reference' if rep['fixed_reference'] else ', paired'}; "
         f"{rep['n_clips']} clips (keys sha256 `{rep['clip_keys_sha256'][:12]}`). Registry `{rep['registry_sha256'][:12]}` "
         f"({rep['registry_status']}). Revision `{rep['provenance']['revision']}`.", "",
         "Every number is measured from the listed per-clip tables; nothing here is an expected score. A non-significant "
         "difference is not equivalence: inconclusive stays inconclusive.", ""]
    if rep.get("disclosed_exposure"):
        L += ["Disclosed exposure differences: " + "; ".join(f"{k}: {v}" for k, v in rep["disclosed_exposure"].items()), ""]
    if "screen" in rep:
        L += [f"**Stage-2 screen:** {'promote' if rep['screen']['promote'] else 'reject'} ({rep['screen']['reason']}).", "",
              "| metric | subset | clips | candidate - control | worse than margin |", "|---|---|---:|---:|---|"]
        for m, subs in rep["registered"].items():
            for s, r in subs.items():
                if r.get("n_clips"):
                    L.append(f"| {m} | {s} | {r['n_clips']} | {r['d']:+.4f} | {r['worse_than_margin']} |")
        return "\n".join(L) + "\n"
    sm = rep["summary"]
    L += ["## Measured improvements (superior)", "", ", ".join(sm["improvements"]) or "none", "",
          "## Regressions (inferior)", "", ", ".join(sm["regressions"]) or "none", "",
          "## Uncertain (inconclusive)", "", ", ".join(sm["uncertain"]) or "none", "",
          "## Non-inferior", "", ", ".join(sm["non_inferior"]) or "none", ""]
    for g in ("gate_b_quality", "gate_c_full"):
        if g in rep:
            L += [f"**{g}:** {json.dumps(rep[g])}", ""]
    L += ["## Registered metrics (candidate - control, 95 % interval)", "",
          "| metric | subset | clips | d_bar | interval | margin | s_d obs / used | V_clip | status | gates on |",
          "|---|---|---:|---:|---|---:|---|---:|---|---|"]
    for m, subs in rep["registered"].items():
        for s, r in subs.items():
            if not r.get("n_clips"):
                L.append(f"| {m} | {s} | 0 | | | | | | no clips | |")
                continue
            L.append(f"| {m} | {s} | {r['n_clips']} | {r['d_bar']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                     f"{r['margin']} | {r['s_d_observed']:.4f} / {r['s_d_used']:.4f} | {r['v_clip']:.2e} | "
                     f"{r['status']} | {r['gating']} |")
    L += ["", "## Reported, not gated", "", "```json", json.dumps({"reported": rep["reported"], "runtime": rep["runtime"]},
                                                                  indent=1), "```", ""]
    return "\n".join(L)


# ---- frame-rate modulation index -------------------------------------------------------------------------------------

def modulation_index(y, hop, sr=SR, harmonics=3, bw_hz=2.0) -> float:
    """Energy of the normalized power-envelope spectrum at the hop rate and its first two harmonics, in dB. The
    envelope is divided by its mean, so the index measures modulation depth, not level. Compare it against the
    control's output and the clean target at the same hop rate."""
    y = np.asarray(y, float)
    env = y ** 2
    mu = env.mean()
    if not np.isfinite(mu) or mu <= 0:
        return float("nan")
    env = env / mu - 1.0
    w = np.hanning(len(env))
    spec = np.abs(np.fft.rfft(env * w)) ** 2 / (w ** 2).sum()
    f = np.fft.rfftfreq(len(env), 1.0 / sr)
    f0 = sr / hop
    band = np.zeros_like(f, bool)
    for k in range(1, harmonics + 1):
        band |= np.abs(f - k * f0) <= bw_hz
    return float(10 * np.log10(spec[band].sum() + 1e-20))


def modindex_table(wavs, clean, hops):
    """Per-clip rows {key, mod_idx_hop<N>_db, mod_idx_clean_hop<N>_db}; files are <bucket>/<id>.wav in both trees."""
    import soundfile as sf
    rows = []
    for p in sorted(Path(wavs).rglob("*.wav")):
        rel = p.relative_to(wavs)
        cp = Path(clean) / rel
        if not cp.exists():
            raise ComparisonError(f"no clean target for {rel}")
        y, sr = sf.read(p, dtype="float32")
        c, sr2 = sf.read(cp, dtype="float32")
        if sr != SR or sr2 != SR:
            raise ComparisonError(f"{rel}: the modulation index is scored at 16 kHz")
        row = {"bucket": str(rel.parent), "id": rel.stem}
        for h in hops:
            o, k = mod_cols(h)
            row[o], row[k] = modulation_index(y, h), modulation_index(c, h)
        rows.append(row)
    return rows


# ---- blinded listening examples --------------------------------------------------------------------------------------

def export_listening(spec: dict, out, seed=None) -> dict:
    """Aligned examples per category under neutral labels, the key in a separate file, and the unaligned recordings and
    timing kept apart. spec = {"categories": {cat: [clip, ...]}, "systems": {arm: aligned_dir},
    "unaligned": {arm: dir}, "timing": path}; clip files are <clip>.wav in each system's directory."""
    cats = spec.get("categories") or {}
    miss = [c for c in LISTEN_CATEGORIES if not cats.get(c)]
    if miss:
        raise ComparisonError(f"listening categories without clips: {miss}")
    systems = spec.get("systems") or {}
    if len(systems) < 2:
        raise ComparisonError("listening needs at least two systems")
    out = Path(out)
    rng = random.Random(seed if seed is not None else random.SystemRandom().randrange(2 ** 32))
    key = {}
    for cat, clips in cats.items():
        for clip in clips:
            arms = list(systems)
            rng.shuffle(arms)
            d = out / "aligned" / cat / Path(clip).name
            d.mkdir(parents=True, exist_ok=True)
            for i, arm in enumerate(arms):
                src = Path(systems[arm]) / f"{clip}.wav"
                if not src.exists():
                    raise ComparisonError(f"{arm}: no aligned output for {clip}")
                shutil.copyfile(src, d / f"S{i + 1}.wav")
                key[f"{cat}/{Path(clip).name}/S{i + 1}"] = arm
    (out / "key").mkdir(parents=True, exist_ok=True)
    (out / "key" / "key.json").write_text(json.dumps(key, indent=1))
    for arm, udir in (spec.get("unaligned") or {}).items():
        shutil.copytree(udir, out / "unaligned" / arm, dirs_exist_ok=True)
    if spec.get("timing"):
        (out / "unaligned").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(spec["timing"], out / "unaligned" / Path(spec["timing"]).name)
    return key


# ---- recipe freeze ---------------------------------------------------------------------------------------------------

def freeze(sel: dict) -> dict:
    """Freeze a per-tier recipe only after its checks. sel = {"tier", "variants": [{"name", "recipe",
    "confirmation": report.json, "gates": {gate: {"status": "pass"|"fail"|"pending", "evidence": path}},
    "owner_decision": {"by", "gap"} (quality only)}]}. Absolute gates are independent of the comparison: a quality
    win never rescues a failed or missing gate, and a gate pass never rescues a quality failure."""
    frozen, rejected = None, []
    for v in sel.get("variants") or []:
        why = []
        gates = v.get("gates") or {}
        for g in FREEZE_GATES:
            st = (gates.get(g) or {}).get("status")
            if st != "pass" or not (gates.get(g) or {}).get("evidence"):
                why.append(f"gate {g}: {'incomplete evidence' if st in (None, 'pending') or st == 'pass' else st}")
        rep = None
        try:
            rep = json.loads(Path(v["confirmation"]).read_text()) if v.get("confirmation") else None
        except (OSError, ValueError) as e:
            why.append(f"confirmation unreadable: {e}")
        if rep is None:
            why.append("no five-seed confirmation report")
        elif rep.get("kind") != "confirmation" or len(rep.get("seeds") or []) < CONFIRMATION_MIN_SEEDS:
            why.append("the comparison is not a five-seed confirmation")
        elif not (rep.get("gate_b_quality") or {}).get("pass"):
            od = v.get("owner_decision") or {}
            if not (od.get("by") and od.get("gap")):
                why.append(f"quality: {rep.get('summary', {}).get('regressions') or rep.get('summary', {}).get('uncertain')}"
                           " not non-inferior, and no owner decision states the gap")
        if why or frozen is not None:
            rejected.append({"name": v.get("name"), "reasons": why or [f"{frozen['name']} was frozen first"]})
        else:
            frozen = {"name": v.get("name"), "recipe": v.get("recipe"), "confirmation": v.get("confirmation"),
                      "owner_decision": v.get("owner_decision")}
    return {"schema": "r8_ld_selection_v1", "tier": sel.get("tier"), "frozen": frozen, "rejected": rejected,
            "status": "frozen" if frozen else "not_frozen"}


# ---- CLI -------------------------------------------------------------------------------------------------------------

def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("register"); r.add_argument("--c0-s0", required=True); r.add_argument("--c0-s1", required=True)
    r.add_argument("--approved-by"); r.add_argument("--out", required=True)
    c = sp.add_parser("compare"); c.add_argument("manifest"); c.add_argument("--registry", required=True)
    c.add_argument("--out", required=True)
    m = sp.add_parser("modindex"); m.add_argument("--wavs", required=True); m.add_argument("--clean", required=True)
    m.add_argument("--hops", type=int, nargs="+", default=[96, 128]); m.add_argument("--out", required=True)
    li = sp.add_parser("listen"); li.add_argument("spec"); li.add_argument("--out", required=True)
    li.add_argument("--seed", type=int)
    f = sp.add_parser("freeze"); f.add_argument("selection"); f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "register":
            if Path(a.out).exists():
                raise ComparisonError(f"{a.out} exists: margins and floors are never re-registered after results")
            _write(a.out, register(a.c0_s0, a.c0_s1, a.approved_by))
        elif a.cmd == "compare":
            rep = compare(json.loads(Path(a.manifest).read_text()), json.loads(Path(a.registry).read_text()),
                          ["compare_r8_ld.py", *argv])
            _write(Path(a.out) / "report.json", rep)
            (Path(a.out) / "report.md").write_text(markdown(rep))
        elif a.cmd == "modindex":
            import pandas as pd
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(modindex_table(a.wavs, a.clean, a.hops)).to_csv(a.out, index=False)
        elif a.cmd == "listen":
            export_listening(json.loads(Path(a.spec).read_text()), a.out, a.seed)
        elif a.cmd == "freeze":
            res = freeze(json.loads(Path(a.selection).read_text()))
            _write(a.out, res)
            print(json.dumps(res, indent=1))
            return 0 if res["frozen"] else 1
    except ComparisonError as e:
        print("refused:\n  " + "\n  ".join(e.reasons), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
