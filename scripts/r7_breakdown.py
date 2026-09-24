#!/usr/bin/env python3
"""Break the r7 cascade's scores down by noise class, input SNR and clip, from the committed CSVs.

The headline tables report means. A mean can clear a target while most clips miss it, and the
pooled eval_gen number hides that the pre-registered stationary grid fails on its own. This writes:

- results_r2/r7/breakdown.md: headline rows with row-bootstrap and scene-clustered CIs, the per-clip
  all-three pass rate, a per-class x per-input-SNR table, the fault buckets, and a paired delta vs raw.
- results_r2/generalisation/per_grid.md: eval_gen split by grid (stationary = the registered grid,
  changing = the amendment), paired-by-bucket vs raw, and the per-bucket gap to eval_r2
  (PROTOCOL.md section 5).

Scene clusters are (snr_in, id): one seed at one input SNR shares its speech and room across the
noise classes, so resampling rows alone treats correlated clips as independent.

Regenerate (repo root): uv run python scripts/r7_breakdown.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vaani.report import TARGETS, ci, cluster_ci, load_results, nominal_mask, pass_all, scene_clusters

R7_R2 = "results_r2/r7/r7_e256_wr64_cascade_eval_r2.csv"
R7_GEN = "results_r2/r7/r7_e256_wr64_cascade_gen.csv"
RAW_R2 = "results_r2/r7/raw_eval_r2_relabel.csv"  # same render as the r7 CSV; results_r2/raw.csv is the earlier render
RAW_GEN = "results_r2/generalisation/raw_gen.csv"
TM = ["snr_out", "stoi", "pesq_wb"]  # the three problem-statement targets, in the order the docs quote them
KEYS = ["bucket", "id"]


def load(path):
    df = load_results([path])
    if "fault" not in df: df["fault"] = np.nan
    df["fault"] = df.fault.where(df.fault.notna(), None)
    df["nominal"] = nominal_mask(df)
    df["pass3"] = pass_all(df).astype(float)
    df["cluster"] = scene_clusters(df)
    return df


def _f(t, d=3):
    return "n/a" if not np.isfinite(t[0]) else f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def _pct(t):
    return "n/a" if not np.isfinite(t[0]) else f"{100 * t[0]:.1f}% [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"


def headline_rows(df, n_boot=1000):
    """(label, subset) pairs for the rows every doc quotes."""
    burst = df.fault.fillna("").str.startswith("fault_burst")
    rows = [("Nominal (0/5/10 dB, unclipped, no fault)", df[df.nominal]),
            ("Full test split", df),
            ("Non-fault, input 0 dB", df[df.fault.isna() & (df.snr_in == 0)]),
            ("Non-fault, input -5 dB", df[df.fault.isna() & (df.snr_in == -5)]),
            ("Non-fault, input -10 dB", df[df.fault.isna() & (df.snr_in == -10)]),
            ("Transients (fault_burst_*, input 0/5 dB)", df[burst])]
    out = []
    for label, g in rows:
        r = {"label": label, "n": len(g)}
        for m in TM + ["pass3"]:
            r[m] = ci(g[m], n=n_boot); r[m + "_cl"] = cluster_ci(g[m], g.cluster, n=n_boot)
        out.append(r)
    return out


def headline_table(rows):
    lines = ["| rows | n | SNR_out (row CI) | SNR_out (cluster CI) | STOI (row) | STOI (cluster) | PESQ-WB (row) | PESQ-WB (cluster) | all-three pass (row) | all-three pass (cluster) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for m in TM:
            cells += [_f(r[m]), _f(r[m + "_cl"])]
        cells += [_pct(r["pass3"]), _pct(r["pass3_cl"])]
        lines.append(f"| {r['label']} | {r['n']} | " + " | ".join(cells) + " |")
    return lines


def _means(g):
    return " | ".join(f"{np.nanmean(g[m]):.3f}" for m in TM) + f" | {100 * g.pass3.mean():.1f}%"


def class_snr_table(df):
    lines = ["| noise class | input SNR (dB) | n | SNR_out | STOI | PESQ-WB | all-three pass |", "|---|---|---|---|---|---|---|"]
    nf = df[df.fault.isna()]
    for (c, s), g in nf.groupby(["noise_class", "snr_in"]):
        lines.append(f"| {c} | {s:g} | {len(g)} | {_means(g)} |")
    return lines


def fault_table(df):
    lines = ["| fault | input SNR (dB) | n | SNR_out | STOI | PESQ-WB | all-three pass |", "|---|---|---|---|---|---|---|"]
    for (f, s), g in df[df.fault.notna()].groupby(["fault", "snr_in"]):
        lines.append(f"| {f} | {s:g} | {len(g)} | {_means(g)} |")
    return lines


def paired(sys_df, raw_df, n_boot=1000):
    """Per-clip system minus raw on the same (bucket, id); the pairing removes the clip's own difficulty."""
    m = sys_df.merge(raw_df[KEYS + TM], on=KEYS, suffixes=("", "_raw"), validate="one_to_one")
    for k in TM:
        m[k + "_d"] = m[k] - m[k + "_raw"]
    return m


def paired_table(m, by, n_boot=1000):
    lines = [f"| {by} | n | " + " | ".join(f"{k} r7 / raw / delta [cluster CI]" for k in TM) + " |", "|---|---|" + "---|" * len(TM)]
    for key, g in m.groupby(by):
        cells = [f"{np.nanmean(g[k]):.3f} / {np.nanmean(g[k + '_raw']):.3f} / {_f(cluster_ci(g[k + '_d'], g.cluster, n=n_boot))}" for k in TM]
        lines.append(f"| {key} | {len(g)} | " + " | ".join(cells) + " |")
    return lines


def gap_ci(a, b, n=1000, seed=0):
    """Unpaired mean(a) - mean(b), independent row bootstrap of each set (the two renders share no clips)."""
    a = np.asarray(a, float); b = np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b): return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return (a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))


def breakdown_md(df, raw, n_boot, src):
    L = ["# r7 cascade: per class, per input SNR, per clip", "",
         f"Source: `{src['r7']}` (system `{df.system.iloc[0]}`, eval_r2 test, current render). "
         f"Raw passthrough: `{src['raw']}`. Regenerate: `uv run python scripts/r7_breakdown.py`.", "",
         f"Targets: SNR_out > {TARGETS['snr_out']:g} dB, STOI > {TARGETS['stoi']:g}, PESQ-WB > {TARGETS['pesq_wb']:g}. "
         "All-three pass = the fraction of clips on which all three hold at once (strict >).",
         "Row CI: 1000-sample bootstrap over clips (`vaani.report.ci`, as in matrix.md). Cluster CI: 1000-sample bootstrap over "
         "scene clusters (snr_in, id) (`vaani.report.cluster_ci`); clips from one seed at one SNR share speech and room across classes.", "",
         "## Headline rows", ""] + headline_table(headline_rows(df, n_boot))
    L += ["", "## Per noise class x input SNR (non-fault buckets; clipped items included, so these are not nominal cells)", ""] + class_snr_table(df)
    L += ["", "## Fault buckets (input 0/5 dB; compare with fault_none, not with nominal)", ""] + fault_table(df)
    if raw is not None:
        m = paired(df, raw, n_boot)
        L += ["", "## Paired vs raw passthrough, nominal clips, by noise class (delta = r7 - raw on the same clip)", ""] + paired_table(m[m.nominal], "noise_class", n_boot)
    return L


def per_grid_md(gen, raw_gen, r2, n_boot, src):
    nom = gen[gen.nominal & gen.noise_class.isin(["stationary", "changing"])]
    L = ["# eval_gen split by grid", "",
         f"Source: `{src['gen']}` (system `{gen.system.iloc[0]}`, eval_gen test). Regenerate: `uv run python scripts/r7_breakdown.py`.", "",
         "PROTOCOL.md registered the **stationary** vehicle grid; the **changing** grid was added by amendment before scoring. "
         "The pooled nominal number averages the two; the registered grid is reported on its own here.", "",
         "Within one grid every scene cluster (snr_in, id) holds a single clip, so there the cluster CI equals the row CI.", "",
         "## Nominal (0/5/10 dB, unclipped), per grid", ""] + headline_table(
        [dict(label=grid, n=len(g), **{k: v for m in TM + ["pass3"] for k, v in
             ((m, ci(g[m], n=n_boot)), (m + "_cl", cluster_ci(g[m], g.cluster, n=n_boot)))})
         for grid, g in [("stationary (registered)", nom[nom.noise_class == "stationary"]),
                         ("changing (amendment)", nom[nom.noise_class == "changing"]), ("pooled", nom)]])
    if raw_gen is None:
        L += ["", f"## Paired vs raw", "", f"TBD: `{src['raw_gen']}` is missing; score the raw passthrough on data/eval_gen test "
              f"(`uv run python -m vaani.eval --system raw --split test --eval-root data/eval_gen --out {src['raw_gen']} --workers 2`) and rerun."]
    else:
        m = paired(gen, raw_gen, n_boot)
        mg = m[m.noise_class.isin(["stationary", "changing"])]
        L += ["", f"## Paired by bucket vs raw passthrough (all clips in the bucket, clipped included; raw: `{src['raw_gen']}`)", ""] + paired_table(mg, "bucket", n_boot)
        L += ["", "## Paired vs raw, nominal clips, per grid", ""] + paired_table(mg[mg.nominal], "noise_class", n_boot)
    L += ["", "## Per-bucket gap to eval_r2 (eval_gen mean - eval_r2 mean, same system and bucket; unpaired row bootstrap, the sets share no clips)", "",
          f"eval_r2 source: `{src['r7']}`. The speech mix differs between the sets, so the gap is not a pure noise-generalisation measure.", "",
          "| bucket | n gen / r2 | " + " | ".join(f"{k} gap [95% CI]" for k in TM) + " |", "|---|---|" + "---|" * len(TM)]
    for b, g in gen[gen.noise_class.isin(["stationary", "changing"])].groupby("bucket"):
        h = r2[r2.bucket == b]
        if not len(h): continue
        L.append(f"| {b} | {len(g)} / {len(h)} | " + " | ".join(_f(gap_ci(g[k], h[k], n=n_boot)) for k in TM) + " |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--r7", default=R7_R2); ap.add_argument("--gen", default=R7_GEN)
    ap.add_argument("--raw", default=RAW_R2); ap.add_argument("--raw-gen", default=RAW_GEN)
    ap.add_argument("--out", default="results_r2/r7/breakdown.md")
    ap.add_argument("--gen-out", default="results_r2/generalisation/per_grid.md")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    src = {"r7": a.r7, "gen": a.gen, "raw": a.raw, "raw_gen": a.raw_gen}
    r2, gen = load(a.r7), load(a.gen)
    raw = load(a.raw) if Path(a.raw).exists() else None
    raw_gen = load(a.raw_gen) if Path(a.raw_gen).exists() else None
    Path(a.out).write_text("\n".join(breakdown_md(r2, raw, a.n_boot, src)) + "\n", encoding="utf-8")
    Path(a.gen_out).write_text("\n".join(per_grid_md(gen, raw_gen, r2, a.n_boot, src)) + "\n", encoding="utf-8")
    print(f"wrote {a.out} and {a.gen_out}")


if __name__ == "__main__":
    main()
