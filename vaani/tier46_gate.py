"""Paired kill gate for a Tier 4.6 second-stage candidate against the frozen anchor (plan §2).

Both CSVs come from `vaani.eval` on the same frozen split. Rows are paired on (bucket, id); the bootstrap resamples
item ids (all bucket variants of an id travel together) so per-item speech/noise draws are not treated as independent
across buckets. Two candidates are pre-registered per release, so the primary delta intervals are 97.5 % two-sided.
"""
import numpy as np, pandas as pd

N_BOOT, SEED, ALPHA = 10_000, 4606, 0.025   # 97.5 % two-sided for two pre-registered candidates
TARGETS = {"snr_out": 15.0, "stoi": 0.85, "pesq_wb": 2.5}
UTILITY = {"postfilter": (0.25, 0.02), "refiner": (0.50, 0.05)}   # (dSNR_out dB, dPESQ) both required on the nominal slice
HOP_S = 256 / 16000


def _load(p):
    df = pd.read_csv(p, dtype={"id": str})
    if "fault" not in df: df["fault"] = np.nan
    df["fault"] = df.fault.where(df.fault.notna(), None)
    df["nominal"] = (~df.clipped) & (~df.ref_dropout) & df.fault.isna() & df.snr_in.isin([0, 5, 10])
    return df


def _pair(a, c):
    """Inner join on (bucket, id); the caller checks completeness against the protocol's key list first."""
    return a.merge(c, on=["bucket", "id"], suffixes=("_a", "_c"))


def _boot(d, clusters, stat=np.mean):
    """Cluster bootstrap of the mean of paired deltas `d` (aligned with `clusters`)."""
    d = np.asarray(d, float); clusters = np.asarray(clusters)
    if len(d) == 0: return dict(mean=np.nan, lo=np.nan, hi=np.nan)
    ids, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, d, len(ids)); cnts = np.bincount(inv, None, len(ids)).astype(float)
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(ids), (N_BOOT, len(ids)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)   # resample whole clusters; mean over the resampled rows
    return dict(mean=float(d.mean()), lo=float(np.percentile(means, 100 * ALPHA)), hi=float(np.percentile(means, 100 * (1 - ALPHA))))


def compare(anchor_csv, candidate_csv, protocol, kind):
    """protocol: dict with `keys` = sorted [bucket, id] list of the frozen split (from tier46_protocol freeze).
    Returns a dict with completeness, nominal deltas + CIs, per-bucket/aggregate checks, each gate and `pass`."""
    a, c = _load(anchor_csv), _load(candidate_csv)
    want = sorted(map(tuple, protocol["keys"])) if protocol.get("keys") else sorted(map(tuple, a[["bucket", "id"]].values.tolist()))
    got = sorted(map(tuple, c[["bucket", "id"]].astype(str).values.tolist()))
    metrics = ["snr_out", "stoi", "pesq_wb"]
    finite = bool(np.isfinite(c[metrics].to_numpy(float)).all()) and bool(np.isfinite(a[metrics].to_numpy(float)).all())
    complete = got == list(map(tuple, want)) and len(set(got)) == len(got) and finite
    r = {"kind": kind, "n_anchor": len(a), "n_candidate": len(c), "complete": complete, "gates": {}, "nominal": {}, "buckets": {}}
    if not complete:
        r["pass"] = False; r["reason"] = "incomplete: key mismatch, duplicate keys or non-finite metrics"; return r
    p = _pair(a, c)
    for m in metrics: p["d_" + m] = p[m + "_c"] - p[m + "_a"]
    nom = p[p.nominal_a]
    for m in metrics: r["nominal"]["d_" + m] = _boot(nom["d_" + m], nom.id)
    r["nominal"]["candidate_means"] = {m: float(nom[m + "_c"].mean()) for m in metrics}
    r["nominal"]["targets_met"] = all(nom[m + "_c"].mean() > t for m, t in TARGETS.items())
    r["n_clusters"] = int(p.id.nunique())
    d_snr, d_pesq, d_stoi = r["nominal"]["d_snr_out"], r["nominal"]["d_pesq_wb"], r["nominal"]["d_stoi"]
    u_snr, u_pesq = UTILITY[kind]
    eps = 1e-9   # thresholds are inclusive; float noise on an exact shift must not flip them
    r["gates"]["utility"] = d_snr["mean"] >= u_snr - eps and d_pesq["mean"] >= u_pesq - eps
    r["gates"]["paired_uncertainty"] = d_snr["lo"] > 0 and d_pesq["lo"] > 0
    r["gates"]["intelligibility"] = (nom.stoi_c.mean() > TARGETS["stoi"] and d_stoi["mean"] >= -0.003 - eps and d_stoi["lo"] > -0.005)
    per = p.groupby("bucket").agg(d_stoi=("d_stoi", "mean"), d_pesq=("d_pesq_wb", "mean"), d_snr=("d_snr_out", "mean"))
    r["buckets"] = {b: {k: float(v) for k, v in row.items()} for b, row in per.iterrows()}
    r["gates"]["per_bucket"] = bool((per.d_stoi >= -0.01 - eps).all() and (per.d_pesq >= -0.05 - eps).all())
    severe, burst = p[~p.nominal_a], p[p.bucket.str.startswith("fault_burst")]
    r["aggregates"] = {"severe_d_snr_out": float(severe.d_snr_out.mean()) if len(severe) else np.nan,
                       "burst_d_snr_out": float(burst.d_snr_out.mean()) if len(burst) else np.nan}
    r["gates"]["severe_burst"] = all(np.isnan(v) or v >= -0.2 - eps for v in r["aggregates"].values())
    # recovery: inf = burst clip never re-converged; the candidate may not add any, nor slow finite recoveries by > 1 hop (median)
    ra, rc = p.recovery_s_a.to_numpy(float), p.recovery_s_c.to_numpy(float)
    both = np.isfinite(ra) & np.isfinite(rc)
    r["recovery"] = {"unrecovered_anchor": int(np.isinf(ra).sum()), "unrecovered_candidate": int(np.isinf(rc).sum()),
                     "median_delta_s": float(np.median(rc[both] - ra[both])) if both.any() else 0.0}
    r["gates"]["recovery"] = (r["recovery"]["unrecovered_candidate"] <= r["recovery"]["unrecovered_anchor"]
                              and r["recovery"]["median_delta_s"] <= HOP_S + eps)
    r["pass"] = all(r["gates"].values())
    return r
