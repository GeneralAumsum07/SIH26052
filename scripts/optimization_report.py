"""Tables for SIH26052 clauses 14a/14b/14c: what ONNX export, INT8 quantization and magnitude
pruning actually cost, on the frozen round-2 test split.

The deltas here are **paired**. Every system is scored on the same 2280 rendered clips, so the
question "what did quantization cost" is answered per clip and then averaged, not by subtracting
two independently-bootstrapped means. Pairing removes clip-to-clip variance -- which dominates
this eval set, because a -10 dB stationary clip and a +10 dB clean one differ by far more than
any optimization does -- and gives an interval on the difference rather than two intervals the
reader has to compare by eye. An unpaired comparison here would hide a real effect inside two
overlapping confidence intervals.

    uv run python scripts/optimization_report.py --out results_r2/optim/optimization.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vaani.report import TARGETS, ci

METRICS = ["snr_out", "si_sdr", "stoi", "pesq_wb", "dnsmos_ovrl"]
KEYS = ["bucket", "id"]


def nominal(df):
    """The matrix's nominal envelope: unclipped, no reference fault, no fault bucket, SNR 0/5/10."""
    fault = df.fault if "fault" in df else pd.Series(np.nan, index=df.index)
    return df[(~df.clipped) & (~df.ref_dropout) & fault.isna() & df.snr_in.isin([0, 5, 10])]


def read(path):
    """A CSV plus any metric column it predates, so an older run reads n/a instead of raising."""
    df = pd.read_csv(path, dtype={"id": str})
    for m in METRICS:
        if m not in df:
            df[m] = np.nan
    return df


def paired_delta(ref, sys_df, metric, n=2000, seed=0):
    """Mean per-clip (system - reference) difference with a bootstrap CI over clips.

    Resampling clips rather than differences of means is what makes the interval a statement
    about this eval set's items. Clips where either side is NaN are dropped: a failed clip is
    not a zero difference.
    """
    m = ref[KEYS + [metric]].merge(sys_df[KEYS + [metric]], on=KEYS, suffixes=("_ref", "_sys"))
    d = (m[f"{metric}_sys"] - m[f"{metric}_ref"]).to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n, len(d))).mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), len(d)


def fmt_abs(g, metric):
    mean, lo, hi = ci(g[metric])
    if not np.isfinite(mean):
        return "n/a"
    mark = ""
    if metric in TARGETS:
        mark = " ✓" if lo > TARGETS[metric] else (" ~" if mean > TARGETS[metric] else " ✗")
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]{mark}"


def fmt_delta(ref, g, metric):
    mean, lo, hi, n = paired_delta(ref, g, metric)
    if not np.isfinite(mean):
        return "n/a"
    # A delta whose interval straddles zero is not a measured change, and saying so inline stops
    # a reader from reporting noise as a cost.
    flag = "" if lo <= 0 <= hi else "*"
    return f"{mean:+.3f} [{lo:+.3f},{hi:+.3f}]{flag}"


def quality_table(rows, ref_csv, title, note):
    """rows: list of (label, csv path). The first metric block is absolute, the second paired vs ref."""
    ref = nominal(read(ref_csv))
    out = [f"## {title}", "", note, "",
           "| system | n | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for label, path in rows:
        g = nominal(read(path))
        out.append(f"| {label} | {len(g)} | " + " | ".join(fmt_abs(g, m) for m in METRICS) + " |")
    out += ["", "Paired per-clip change against the reference row (`*` = 95 % interval excludes zero):", "",
            "| system | n paired | " + " | ".join("Δ " + m for m in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for label, path in rows[1:]:
        g = nominal(read(path))
        n = paired_delta(ref, g, "snr_out")[3]
        out.append(f"| {label} | {n} | " + " | ".join(fmt_delta(ref, g, m) for m in METRICS) + " |")
    return out + [""]


def graph_table(reports):
    out = ["## Graph size and single-core latency", "",
           "ORT CPU provider, one intra-op thread, 10 s synthetic stream (624 timed frames), best of 5 "
           "interleaved repeats. Model only: no DSP, STFT/iSTFT or audio I/O. The budget is one 16 ms hop.", "",
           "| graph | bytes | nodes | initializer bytes | ms/frame mean | ms/frame p99 |",
           "|---|---:|---:|---:|---:|---:|"]
    seen = set()
    for path in reports:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("fp32", "int8"):
            g = r[key]
            if g["onnx_sha256"] in seen:
                continue
            seen.add(g["onnx_sha256"])
            out.append(f"| `{Path(g['onnx']).name}` | {g['onnx_bytes']:,} | {g['nodes']:,} | "
                       f"{g['initializer_bytes']:,} | {g['ms_per_frame_mean']:.3f} | {g['ms_per_frame_p99']:.3f} |")
    return out + [""]


def sparsity_table(path):
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    inv = r["inventory"]
    out = ["## Pruning budget", "",
           f"Global magnitude pruning over {inv['prunable_weight_elements']:,} learned weight elements in "
           f"{inv['prunable_tensors']} tensors, out of {inv['total_parameters']:,} total parameters. "
           f"Excluded: {', '.join('`' + s + '`' for s in inv['excluded_substrings'])} -- the fixed ERB "
           "analysis/synthesis matrices, which are a signal transform rather than learned capacity.", "",
           "| level | requested | achieved (prunable) | achieved (all parameters) | weights zeroed |",
           "|---|---:|---:|---:|---:|"]
    for lv in r["levels"]:
        out.append(f"| p{int(round(lv['requested_sparsity'] * 100)):02d} | {lv['requested_sparsity']:.0%} | "
                   f"{lv['achieved_sparsity']:.4f} | {lv['sparsity_over_all_parameters']:.4f} | {lv['zeroed_weights']:,} |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--optim-dir", default="results_r2/optim")
    ap.add_argument("--reference", default="results_r2/vaani_tier46_refiner.csv",
                    help="the trained cascade's PyTorch row from the ablation matrix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.optim_dir)

    lines = ["# Optimization: ONNX, INT8 quantization and magnitude pruning", "",
             "Frozen round-2 test split (2280 items); tables below use the nominal envelope "
             "(unclipped, no reference fault, no fault bucket, input SNR 0/5/10 dB) so they are "
             "directly comparable with [the earlier-render ablation matrix](../matrix_prerelabel.md), the render they were measured on.", "",
             "Targets (problem statement): SNR_out > 15 dB, STOI > 0.85, PESQ > 2.5. "
             "✓ = lower 95 % bootstrap bound exceeds target, ~ = mean does but the bound does not, ✗ = mean does not.", ""]

    reports = [p for p in ["deploy/tier46/int8_report.json", "deploy/tier46/int8_perchannel_report.json"] if Path(p).exists()]
    if reports:
        lines += graph_table(reports)

    quant_rows = [("PyTorch checkpoint (reference)", a.reference)]
    for label, name in [("ONNX FP32 graph", "onnx_cascade"), ("ONNX INT8 dynamic", "onnx_cascade_int8")]:
        if (d / f"{name}.csv").exists():
            quant_rows.append((label, d / f"{name}.csv"))
    if len(quant_rows) > 1:
        lines += quality_table(quant_rows, a.reference, "Quality cost of export and quantization",
                               "The FP32 ONNX row is the control: it separates any export difference from the "
                               "quantization difference, so a delta on the INT8 row cannot be blamed on ONNX.")

    prune_rows = [("p00 (unpruned reference)", a.reference)]
    for p in ["p10", "p20", "p30", "p40", "p50"]:
        if (d / f"prune_{p}.csv").exists():
            prune_rows.append((f"{p} ({int(p[1:])} % of learned weights zeroed)", d / f"prune_{p}.csv"))
    if (d / "prune_sparsity.json").exists():
        lines += sparsity_table(d / "prune_sparsity.json")
    if len(prune_rows) > 1:
        lines += quality_table(prune_rows, a.reference, "Quality cost of magnitude pruning",
                               "Pruned in PyTorch and evaluated through the same path as the reference row, so the "
                               "only variable is which weights are zero. No fine-tuning after pruning: this measures "
                               "what the trained weights tolerate, which is the question a deployment budget asks.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
