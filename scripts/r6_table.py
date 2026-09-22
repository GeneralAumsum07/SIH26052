#!/usr/bin/env python3
"""Turn the r6 result CSVs into one comparison table.

Twelve systems on two frozen sets is too many numbers to read as raw CSV, and the interesting
comparisons are between systems rather than within one. This groups by eval set, puts one system per
row, and reports the problem-statement metrics over the protocol's nominal envelope with the project's
own 1000-sample bootstrap (vaani.report.ci), so the numbers match every other table in the repo.

A target counts as MET only when the CI lower bound clears it. A mean above target with a lower bound
below it is marked "~": that is the distinction the r5 review turned on and it should stay visible.

The low-SNR stationary bucket gets its own table. It is where this architecture is weakest, it
reproduced on an unseen corpus, and a headline envelope average hides it.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vaani.report import TARGETS, ci

# order must match the table header below: the three problem-statement targets first,
# then the supporting metrics
METRICS = ["snr_out", "stoi", "pesq_wb", "si_sdr", "dnsmos_ovrl"]


def mark(m, t):
    if m not in TARGETS or not np.isfinite(t[0]):
        return ""
    if t[1] > TARGETS[m]:
        return " **MET**"
    return " ~" if t[0] > TARGETS[m] else " FAIL"


def cell(series, m):
    t = ci(series)
    if not np.isfinite(t[0]):
        return "n/a"
    d = 3 if m in ("stoi",) else 2
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]{mark(m, t)}"


def envelope(df):
    """PROTOCOL's nominal envelope: the six noise classes at 0/5/10 dB, with the reliability-fault
    buckets excluded.

    The fault_* buckets are deliberately adversarial - hard clipping, a -12 dB reference, an obstructed
    reference - and averaging them into a headline number is meaningless. Filtering on the `clipped`
    and `ref_dropout` columns is NOT enough: those flag per-item conditions, while the reference-gain
    and reference-obstruction faults are carried by the bucket alone. Doing it that way pulled tier46's
    eval_r2 envelope down from 14.66 dB to 12.72 dB, which is a statement about fault buckets, not
    about the system."""
    return df[(~df.bucket.astype(str).str.startswith("fault_")) & df.snr_in.isin([0, 5, 10])]


def system_name(path):
    n = Path(path).stem
    for suffix in ("_eval_r2", "_gen"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


def eval_set(path):
    return "eval_gen" if Path(path).stem.endswith("_gen") else "eval_r2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for p in a.csvs:
        df = pd.read_csv(p)
        rows.append((eval_set(p), system_name(p), df))

    lines = ["# r6 results", "",
             "Nominal envelope: unclipped, no reference fault, input SNR 0/5/10 dB.",
             "Bootstrap 95 % CI, 1000 resamples (`vaani.report.ci`). **MET** = CI lower bound clears the "
             "target; `~` = mean clears it but the lower bound does not.", ""]

    for es in ("eval_r2", "eval_gen"):
        group = [(n, d) for e, n, d in rows if e == es]
        if not group:
            continue
        lines += [f"## {es}", "",
                  "| system | n | SNR_out (>15) | STOI (>0.85) | PESQ-WB (>2.5) | SI-SDR | DNSMOS |",
                  "|---|---|---|---|---|---|---|"]
        for name, df in sorted(group):
            e = envelope(df)
            cells = " | ".join(cell(e[m], m) for m in METRICS)
            lines.append(f"| `{name}` | {len(e)} | {cells} |")
        lines.append("")

        # the weak point, which the envelope average hides
        lines += [f"### {es} - low-SNR buckets", "",
                  "| system | bucket | n | SNR_out | STOI | PESQ-WB |", "|---|---|---|---|---|---|"]
        for name, df in sorted(group):
            for b in sorted(x for x in df.bucket.unique() if re.search(r"_-(5|10)$", str(x))):
                g = df[df.bucket == b]
                lines.append(f"| `{name}` | {b} | {len(g)} | "
                             + " | ".join(cell(g[m], m) for m in ("snr_out", "stoi", "pesq_wb")) + " |")
        lines.append("")

    Path(a.out).write_text("\n".join(lines))
    print(f"wrote {a.out} ({len(rows)} result files)")


if __name__ == "__main__":
    main()
