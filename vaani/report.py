"""Aggregate eval CSVs into the ablation matrix with bootstrap CIs.
Nominal envelope = unclipped, no ref dropout, SNR in {0,5,10}.
Severe envelope = everything else (reported, never claimed as target-met).
"""
import argparse, re

import numpy as np, pandas as pd

METRICS = ["snr_out", "si_sdr", "stoi", "pesq_wb"]
# problem-statement targets (SIH26052): SNR>15 dB, STOI>0.85, PESQ>2.5
TARGETS = {"snr_out": 15.0, "stoi": 0.85, "pesq_wb": 2.5}
ENGLISH_PREFIXES = ("ls:", "ears:")  # source_id prefixes whose speech is English; everything else reports WER as n/a


def ci(x, n=1000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def _words(t):
    return re.sub(r"[^a-z0-9' ]+", " ", str(t).lower()).split()


def wer(hyp, ref):
    """Word error rate via Levenshtein on normalised tokens; NaN when the reference is empty."""
    h, r = _words(hyp), _words(ref)
    if not r: return np.nan
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / len(r)


def add_wer(df, ref_csv):
    """Attach per-item WER of asr_text against the clean-reference transcript (matched on bucket+id)."""
    ref = pd.read_csv(ref_csv, dtype={"id": str}).rename(columns={"asr_text": "ref_text"})
    df = df.astype({"id": str}).merge(ref, on=["bucket", "id"], how="left")
    # an absent hypothesis means the eval ran without ASR, not that the system erased the speech
    df["wer"] = [wer(h, r) if isinstance(h, str) and h else np.nan for h, r in zip(df.asr_text, df.ref_text.fillna(""))]
    # the WER normaliser is Latin-only and Whisper-small's Hindi is unreliable: non-English speech gets no WER at all
    if "speech_source" in df:
        df.loc[~df.speech_source.fillna("").str.startswith(ENGLISH_PREFIXES), "wer"] = np.nan
    return df


def fmt(t): return "n/a" if not np.isfinite(t[0]) else f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


def _mark(metric, mean):
    if metric not in TARGETS or not np.isfinite(mean): return ""
    return " ✓" if mean > TARGETS[metric] else " ✗"


def _cell(g, metrics=METRICS):
    """One system's row in the per-bucket table: SNR/SI-SDR/STOI/PESQ (✓/✗ against target) + recovery median."""
    if len(g) == 0: return "-"
    parts = []
    for m in metrics:
        t = ci(g[m]); parts.append(f"{m}=n/a" if not np.isfinite(t[0]) else f"{m}={t[0]:.3f}{_mark(m, t[0])}")
    if "recovery_s" in g and g.recovery_s.notna().any():
        r = pd.to_numeric(g.recovery_s, errors="coerce")
        parts.append(f"rec={r.median():.2f}s")
    return f"{', '.join(parts)} (n={len(g)})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csvs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--asr-ref", help="results/asr/clean.csv from scripts/asr_clean_reference.py; adds a WER column")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p, dtype={"id": str}) for p in a.csvs])
    metrics = list(METRICS)
    if a.asr_ref:
        df = add_wer(df, a.asr_ref); metrics.append("wer")
    # older CSVs predate the fault column; treat them as fault-free
    if "fault" not in df: df["fault"] = np.nan
    df["fault"] = df.fault.where(df.fault.notna(), None)
    df["snr_gain"] = df.snr_out - df.snr_in  # improvement reading; the target is judged on absolute snr_out
    df["nominal"] = (~df.clipped) & (~df.ref_dropout) & df.fault.isna() & df.snr_in.isin([0, 5, 10])
    lines = ["# Ablation matrix", "",
             "PESQ: wideband P.862.2 @16 kHz (`pesq` package). P.862 is withdrawn by ITU in favour of P.863; reported because the brief requests it.",
             "SNR_out = 10log10(||s||^2/||s_hat-s||^2) vs clean primary (distortion counts as error). SI-SDR reported separately.",
             "Targets (problem statement): SNR_out>15 dB, STOI>0.85, PESQ>2.5 - marked per bucket row and per overall nominal row.",
             *(["WER: faster-whisper small on the enhanced output vs the SAME model's transcript of the clean reference (no human transcripts in the eval set) - supporting evidence only. English speech only (LibriSpeech); Hindi rows report n/a."] if a.asr_ref else []), "",
             "## Nominal envelope (unclipped, no reference fault, no fault bucket, input SNR 0/5/10 dB)", "",
             "snr_gain = snr_out - snr_in, the improvement reading; targets are marked on absolute snr_out only.", "",
             "| system | n | " + " | ".join(metrics + ["snr_gain"]) + " |", "|---|---|" + "---|" * (len(metrics) + 1)]
    for sysname, g in df[df.nominal].groupby("system"):
        cells = []
        for m in metrics + ["snr_gain"]:
            t = ci(g[m]); cells.append(fmt(t) + _mark(m, t[0]))
        lines.append(f"| {sysname} | {len(g)} | " + " | ".join(cells) + " |")
    if df.fault.notna().any():
        # fault buckets share seeds with fault_none, so each row reads as a delta against that reference clip set
        lines += ["", "## Reliability faults (outside the nominal envelope; same speech/noise/room per seed, fault is the only variable)", "",
                  "| fault | " + " | ".join(systems := sorted(df.system.unique())) + " |", "|---|" + "---|" * len(systems)]
        fd = df[df.fault.notna()]
        for fault, gf in fd.groupby("fault"):
            lines.append(f"| {fault} | " + " | ".join(_cell(gf[gf.system == s], metrics) for s in systems) + " |")
    systems = sorted(df.system.unique())
    lines += ["", "## Per bucket (all systems)", ""]
    lines += ["| bucket | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for bucket, gb in df.groupby("bucket"):
        cells = [_cell(gb[gb.system == s], metrics) for s in systems]
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("| **Overall** | " + " | ".join(_cell(df[df.system == s], metrics) for s in systems) + " |")
    if "recovery_s" in df and df.recovery_s.notna().any():
        lines += ["", "## Recovery time after burst (s)", ""]
        for sysname, g in df[df.recovery_s.notna()].groupby("system"):
            r = pd.to_numeric(g.recovery_s, errors="coerce")  # inf = never recovered; quantiles keep it so p90 is honest
            lines.append(f"- {sysname}: median={r.median():.3f} p90={r.quantile(0.9):.3f} failures={int(np.isinf(r).sum())}/{len(r)}")
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines).encode("ascii", "replace").decode())  # cp1252 consoles choke on the check marks


if __name__ == "__main__":
    main()
