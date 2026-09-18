"""Aggregate eval CSVs into the ablation matrix with bootstrap CIs.
Nominal envelope = unclipped, no ref dropout, SNR in {0,5,10}.
Severe envelope = everything else (reported, never claimed as target-met).
"""
import argparse

import numpy as np, pandas as pd

METRICS = ["snr_out", "si_sdr", "stoi", "pesq_wb"]
# problem-statement targets (SIH26052): SNR>15 dB, STOI>0.85, PESQ>2.5
TARGETS = {"snr_out": 15.0, "stoi": 0.85, "pesq_wb": 2.5}


def ci(x, n=1000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def fmt(t): return f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"


def _mark(metric, mean):
    if metric not in TARGETS or not np.isfinite(mean): return ""
    return " ✓" if mean > TARGETS[metric] else " ✗"


def _cell(g):
    """One system's row in the per-bucket table: SNR/SI-SDR/STOI/PESQ (✓/✗ against target) + recovery median."""
    if len(g) == 0: return "-"
    parts = []
    for m in METRICS:
        t = ci(g[m]); parts.append(f"{m}={t[0]:.2f}{_mark(m, t[0])}")
    if "recovery_s" in g and g.recovery_s.notna().any():
        r = pd.to_numeric(g.recovery_s, errors="coerce")
        parts.append(f"rec={r.median():.2f}s")
    return f"{', '.join(parts)} (n={len(g)})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csvs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.csvs])
    df["nominal"] = (~df.clipped) & (~df.ref_dropout) & df.snr_in.isin([0, 5, 10])
    lines = ["# Ablation matrix", "",
             "PESQ: wideband P.862.2 @16 kHz (`pesq` package). P.862 is withdrawn by ITU in favour of P.863; reported because the brief requests it.",
             "SNR_out = 10log10(||s||^2/||s_hat-s||^2) vs clean primary (distortion counts as error). SI-SDR reported separately.",
             "Targets (problem statement): SNR_out>15 dB, STOI>0.85, PESQ>2.5 - marked per bucket row and per overall nominal row.", "",
             "## Nominal envelope (unclipped, no reference fault, input SNR 0/5/10 dB)", "",
             "| system | n | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for sysname, g in df[df.nominal].groupby("system"):
        cells = []
        for m in METRICS:
            t = ci(g[m]); cells.append(fmt(t) + _mark(m, t[0]))
        lines.append(f"| {sysname} | {len(g)} | " + " | ".join(cells) + " |")
    systems = sorted(df.system.unique())
    lines += ["", "## Per bucket (all systems)", ""]
    lines += ["| bucket | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for bucket, gb in df.groupby("bucket"):
        cells = [_cell(gb[gb.system == s]) for s in systems]
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("| **Overall** | " + " | ".join(_cell(df[df.system == s]) for s in systems) + " |")
    if "recovery_s" in df and df.recovery_s.notna().any():
        lines += ["", "## Recovery time after burst (s)", ""]
        for sysname, g in df[df.recovery_s.notna()].groupby("system"):
            r = pd.to_numeric(g.recovery_s, errors="coerce")
            lines.append(f"- {sysname}: median={r.median():.3f} p90={r.quantile(0.9):.3f} failures={int(r.isna().sum())}/{len(r)}")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
