"""Defence-noise table (plan A3): per PS category x input SNR, each system's means with 95% CIs, per-clip all-three
pass rate, marks against the PS targets, and system-minus-baseline paired deltas. Reads vaani/eval.py CSVs scored on
data/eval_defence (their category/noise_source/impulse_source columns come from its index.csv).

CIs: clustered bootstrap (vaani/report.py cluster_ci). The cluster is the recording an item shares with others in its
cell: the gunshot recording for "gunshot", the noise-bed clip everywhere else (synthetic blasts are unique per item).
"""
import argparse
from pathlib import Path

import numpy as np, pandas as pd

from vaani.report import TARGETS, cluster_ci, pass_all

METRICS = [("snr_out", "SNR_out dB"), ("stoi", "STOI"), ("pesq_wb", "PESQ"), ("dnsmos_ovrl", "DNSMOS OVRL")]
ORDER = ["gunshot", "blast_small_arms", "blast_artillery", "helicopter", "vehicle", "siren"]
GAPS = [
    "drone: every drone row (1332) is in the train split; no held-out drone recording exists, so no drone category.",
    "NOISEX-92: 14 train + 1 val rows, 0 test rows; no NOISEX category.",
    "wind: ESC-50 has 8 wind test rows but no wind category was rendered; wind is not measured here.",
    "Lombard speech: all speech is read speech (LibriSpeech, Common Voice hi) recorded in quiet; no Lombard effect.",
    "siren: only 2 ESC-50 siren test clips exist, so every siren item reuses one of two recordings.",
    "all mixtures are synthetic (clean speech + noise through a simulated or parametric room), not recordings in noise.",
]


def load(paths):
    df = pd.concat([pd.read_csv(p).assign(sys_name=Path(p).stem) for p in paths], ignore_index=True)
    if "category" not in df or df.category.isna().all():
        df["category"] = df.bucket.str.rsplit("_", n=1).str[0]   # older CSVs: bucket is "<category>_<snr>"
    for c in ("noise_source", "impulse_source"):
        if c not in df: df[c] = ""
    df["cluster"] = np.where(df.category == "gunshot", df.impulse_source.fillna(""), df.noise_source.fillna(""))
    df["pass3"] = pass_all(df)
    return df


def mark(m, lo, mean):
    """PASS = lower 95% bound above target; ~ = mean above, bound not; FAIL = mean not above."""
    if m not in TARGETS or not np.isfinite(mean): return ""
    return "PASS" if lo > TARGETS[m] else ("~" if mean > TARGETS[m] else "FAIL")


def fmt(t, d=2):
    return "n/a" if not np.isfinite(t[0]) else f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def cell_stats(g, n_boot=1000):
    out = {}
    for m, _ in METRICS:
        out[m] = cluster_ci(g[m], g.cluster, n=n_boot) if m in g and g[m].notna().any() else (np.nan,) * 3
    out["pass3"] = cluster_ci(g.pass3.astype(float), g.cluster, n=n_boot)
    return out


def paired_delta(df, sys_a, sys_b, m, n_boot=1000):
    """Mean of (a - b) over the items both systems scored, per cell, clustered like the cell CIs."""
    key = ["bucket", "id"]
    a = df[df.sys_name == sys_a].set_index(key); b = df[df.sys_name == sys_b].set_index(key)
    j = a[[m, "cluster", "category", "snr_in"]].join(b[[m]], rsuffix="_b", how="inner")
    j["d"] = j[m] - j[f"{m}_b"]
    return {k: cluster_ci(g.d, g.cluster, n=n_boot) for k, g in j.groupby(["category", "snr_in"])}


def table(df, systems, baseline, evalset_hash=None, n_boot=1000):
    cats = [c for c in ORDER if c in set(df.category)] + sorted(set(df.category) - set(ORDER))
    snrs = sorted(df.snr_in.unique())
    L = ["# Defence-noise results (data/eval_defence, test split)", ""]
    if evalset_hash: L += [f"Eval-set hash `{evalset_hash}`. Items per system: "
                           + ", ".join(f"{s} {int((df.sys_name == s).sum())}" for s in systems) + ".", ""]
    L += ["Targets (PS SIH26052): SNR_out > 15 dB, STOI > 0.85, PESQ > 2.5. Cells are mean [95% CI]; mark PASS = lower "
          "bound above target, ~ = mean above but bound not, FAIL = mean not above. `pass3` = fraction of clips meeting "
          "all three at once. CIs: clustered bootstrap (1000 resamples) by shared recording (gunshot recording for "
          "`gunshot`, noise-bed clip otherwise). Input SNR is the speech-to-bed SNR; transients sit on top at 15-45 dB "
          "peak re speech RMS.", ""]
    for s in systems:
        L += [f"## {s}", "", "| category | SNR_in | " + " | ".join(h for _, h in METRICS) + " | pass3 |",
              "|---|---:|" + "---|" * (len(METRICS) + 1)]
        for c in cats:
            for snr in snrs:
                g = df[(df.sys_name == s) & (df.category == c) & (df.snr_in == snr)]
                if g.empty: continue
                st = cell_stats(g, n_boot)
                cells = [fmt(st[m], 3 if m == "stoi" else 2) + (f" {mark(m, st[m][1], st[m][0])}" if m in TARGETS else "")
                         for m, _ in METRICS]
                L.append(f"| {c} | {snr:g} | " + " | ".join(cells) + f" | {fmt(st['pass3'], 2)} |")
        L.append("")
    for s in systems:
        if s == baseline: continue
        L += [f"## {s} minus {baseline} (paired per item)", "",
              "| category | SNR_in | " + " | ".join(h for m, h in METRICS if m != "dnsmos_ovrl") + " |", "|---|---:|---|---|---|"]
        ds = {m: paired_delta(df, s, baseline, m, n_boot) for m, _ in METRICS if m != "dnsmos_ovrl"}
        for c in cats:
            for snr in snrs:
                if (c, snr) not in ds["snr_out"]: continue
                L.append(f"| {c} | {snr:g} | " + " | ".join(fmt(ds[m][(c, snr)], 3 if m == "stoi" else 2) for m in ds) + " |")
        L.append("")
    L += ["## Gaps (not measured by this set)", ""] + [f"- {g}" for g in GAPS] + [""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="vaani/eval.py CSVs; the system name is the file stem")
    ap.add_argument("--baseline", default="raw")
    ap.add_argument("--eval-root", default="data/eval_defence/test", help="for EVALSET_HASH")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results_r2/defence/table.md")
    a = ap.parse_args()
    df = load(a.results)
    systems = [Path(p).stem for p in a.results]
    h = Path(a.eval_root) / "EVALSET_HASH"
    Path(a.out).write_text(table(df, systems, a.baseline, h.read_text().strip() if h.exists() else None, a.boot),
                           encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
