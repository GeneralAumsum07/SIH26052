"""Aggregate eval CSVs into the ablation matrix with bootstrap CIs.
Nominal envelope = unclipped, no ref dropout, SNR in {0,5,10}.
Severe envelope = everything else (reported, never claimed as target-met).
"""
import argparse, glob, hashlib, re, warnings
from pathlib import Path

import numpy as np, pandas as pd

METRICS = ["snr_out", "si_sdr", "stoi", "pesq_wb", "dnsmos_ovrl"]  # dnsmos_ovrl only where the CSV has it (r3+, --dnsmos)
# problem-statement targets (SIH26052): SNR>15 dB, STOI>0.85, PESQ>2.5
TARGETS = {"snr_out": 15.0, "stoi": 0.85, "pesq_wb": 2.5}
ENGLISH_PREFIXES = ("ls:", "ears:")  # source_id prefixes whose speech is English; everything else reports WER as n/a


def ci(x, n=1000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def nominal_mask(df):
    """The nominal envelope exactly as main() builds it: unclipped, no ref dropout, no fault bucket, input SNR 0/5/10 dB."""
    fault = df.fault if "fault" in df else pd.Series(np.nan, index=df.index)
    return ((~df.clipped.astype(bool)) & (~df.ref_dropout.astype(bool)) & fault.isna()
            & df.snr_in.isin([0, 5, 10])).to_numpy()


def scene_clusters(df):
    """Cluster key (snr_in, id): items rendered from one seed at one input SNR share speech and room across noise classes."""
    return df.snr_in.astype(str) + "|" + df.id.astype(str)


def cluster_ci(x, clusters, n=1000, seed=0):
    """Scene-clustered bootstrap of the item mean: resample whole clusters, so correlated items do not narrow the CI."""
    x = np.asarray(x, float); c = np.asarray(clusters)
    keep = np.isfinite(x); x, c = x[keep], c[keep]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    _, inv = np.unique(c, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=x), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return (x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def pass_all(df, targets=TARGETS):
    """Per clip: True only where every target metric strictly exceeds its target (a NaN metric fails)."""
    ok = np.ones(len(df), bool)
    for m, t in targets.items():
        ok &= np.asarray(df[m], float) > t
    return ok


def pass_rate(df, targets=TARGETS):
    """Fraction of clips meeting all targets at once; the mean-based marks hide how few clips do."""
    return float(pass_all(df, targets).mean()) if len(df) else np.nan


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
    if ref.duplicated(["bucket", "id"]).any():
        raise ValueError("ASR reference contains duplicate (bucket, id) keys")
    # Only attach the transcript: metadata in a future reference must not rename
    # evaluation columns or multiply observations (and thereby narrow the CIs).
    df = df.astype({"id": str}).merge(ref[["bucket", "id", "ref_text"]],
                                        on=["bucket", "id"], how="left", validate="many_to_one")
    # an absent hypothesis means the eval ran without ASR, not that the system erased the speech
    df["wer"] = [wer(h, r) if isinstance(h, str) and h else np.nan for h, r in zip(df.asr_text, df.ref_text.fillna(""))]
    # the WER normaliser is Latin-only and Whisper-small's Hindi is unreliable: non-English speech gets no WER at all
    if "speech_source" in df:
        df.loc[~df.speech_source.fillna("").str.startswith(ENGLISH_PREFIXES), "wer"] = np.nan
    return df


def fmt(t): return "n/a" if not np.isfinite(t[0]) else f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


def _mark(metric, interval):
    mean, lower, _ = interval
    if metric not in TARGETS or not np.isfinite(mean): return ""
    if mean <= TARGETS[metric]: return " ✗"
    return " ✓" if lower > TARGETS[metric] else " ~"


def load_results(paths):
    """Partial snapshots overlap the final result; never count them as new items."""
    frames = []
    # PowerShell leaves wildcards literal for native executables. Expand here as
    # well so the documented command consumes the same files on every platform.
    expanded = [p for pattern in paths for p in (sorted(glob.glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern])]
    for p in expanded:
        if ".partial" in Path(p).name:
            warnings.warn(f"Skipping partial evaluation snapshot: {p}", stacklevel=2)
            continue
        frames.append(pd.read_csv(p, dtype={"id": str}))
    if not frames:
        raise ValueError("No final evaluation CSVs supplied")
    df = pd.concat(frames, ignore_index=True)
    # the eval records the checkpoint path as given, so the box (/workspace/SIH26052/results_r2/runs/...)
    # and the laptop (results_r2/runs/...) name one checkpoint differently; key rows on runs/<name>/...
    df["system"] = df.system.astype(str).str.replace(r"^(ckpt|cascade):(?:.*[\\/])?(runs[\\/])", r"\1:\2", regex=True).str.replace("\\", "/", regex=False)
    keys = ["system", "bucket", "id"]
    if df.duplicated(keys).any():
        raise ValueError("Evaluation CSVs contain duplicate (system, bucket, id) keys; supply each final result once")
    return df


def _envelope(g):
    """Lowest input SNR (dB) from which all target metrics pass at every higher SNR present; 'none' if the top fails."""
    if len(g) == 0: return "-"
    passed = {snr: all(np.nanmean(gs[m]) > TARGETS[m] for m in TARGETS) for snr, gs in g.groupby("snr_in")}
    low = None
    for snr in sorted(passed, reverse=True):   # walk down from the top until the first failure
        if not passed[snr]: break
        low = snr
    if low is None: return "none"
    return "met" if np.isinf(low) else f">= {low:g} dB"   # clean_inf has no SNR axis: it either passes or not


def _cell(g, metrics=METRICS):
    """One system's row in the per-bucket table: SNR/SI-SDR/STOI/PESQ (✓/✗ against target) + recovery median."""
    if len(g) == 0: return "-"
    parts = []
    for m in metrics:
        t = ci(g[m]); parts.append(f"{m}=n/a" if not np.isfinite(t[0]) else f"{m}={t[0]:.3f}{_mark(m, t)}")
    if "recovery_s" in g and g.recovery_s.notna().any():
        r = pd.to_numeric(g.recovery_s, errors="coerce")
        parts.append(f"rec={r.median():.2f}s")
    return f"{', '.join(parts)} (n={len(g)})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csvs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--asr-ref", help="results/asr/clean.csv from scripts/asr_clean_reference.py; adds a WER column")
    ap.add_argument("--protocol", help="results_r2/tier46/anchor.json: prints the frozen split/anchor identity above the tables")
    ap.add_argument("--note", action="append", default=[], help="a paragraph printed under the title (e.g. which eval render); repeatable")
    a = ap.parse_args()
    df = load_results(a.csvs)
    metrics = list(METRICS)
    for m in metrics: df[m] = df[m] if m in df else np.nan   # r1/r2 CSVs predate dnsmos_ovrl; they read n/a
    if a.asr_ref:
        df = add_wer(df, a.asr_ref); metrics.append("wer")
    # older CSVs predate the fault column; treat them as fault-free
    if "fault" not in df: df["fault"] = np.nan
    df["fault"] = df.fault.where(df.fault.notna(), None)
    df["snr_gain"] = df.snr_out - df.snr_in  # improvement reading; the target is judged on absolute snr_out
    df["nominal"] = (~df.clipped) & (~df.ref_dropout) & df.fault.isna() & df.snr_in.isin([0, 5, 10])
    lines = ["# Ablation matrix", ""]
    for n in a.note: lines += [f"> {n}", ""]
    if "h_gtcrn_iva" in set(df.system):
        # Keep the measurements auditable, but label every table occurrence so
        # copying a row cannot silently turn our failed integration into a paper result.
        df["system"] = df.system.replace({"h_gtcrn_iva": "h_gtcrn_iva (unreproduced integration)"})
        lines += ["H-GTCRN IVA: our integration of the IVA variant; we have not reproduced the authors' configuration/results. "
                  "This diagnostic row is not presented as their result and must not support comparative superiority claims.", ""]
    if a.protocol:
        import json
        pr = json.loads(open(a.protocol, encoding="utf-8").read()); an = pr.get("anchor", {})
        lines += [f"Protocol: anchor `{an.get('source')}` sha256 `{str(an.get('sha256'))[:16]}`, git `{str(an.get('git'))[:12]}`; "
                  + "; ".join(f"{k} split {v['n_items']} items / {len(v['files'])} files content-hashed" for k, v in pr.get("splits", {}).items()), ""]
    lines += [
             "PESQ: wideband P.862.2 @16 kHz (`pesq` package). P.862 is withdrawn by ITU in favour of P.863; reported because the brief requests it.",
             "SNR_out = 10log10(||s||^2/||s_hat-s||^2) vs clean primary (distortion counts as error). SI-SDR reported separately.",
             "Targets (problem statement): SNR_out>15 dB, STOI>0.85, PESQ>2.5 - marked per bucket row and per overall nominal row.",
             "Legend: ✓ = lower 95% bootstrap bound exceeds target; ~ = mean exceeds target but lower bound does not; ✗ = mean does not exceed target. "
             "Intervals resample evaluation items for one checkpoint; they do not measure training-seed variability or give a joint three-target guarantee.",
             *(["WER: faster-whisper small on the enhanced output vs the SAME model's transcript of the clean reference (no human transcripts in the eval set) - supporting evidence only. English speech only (LibriSpeech); Hindi rows report n/a."] if a.asr_ref else []), "",
             *([f"ASR reference: `{Path(a.asr_ref).as_posix()}`, sha256 `{hashlib.sha256(Path(a.asr_ref).read_bytes()).hexdigest()}`.", ""] if a.asr_ref else []),
             "## Nominal envelope (unclipped, no reference fault, no fault bucket, input SNR 0/5/10 dB)", "",
             "snr_gain = snr_out - snr_in, the improvement reading; targets are marked on absolute snr_out only.", "",
             "| system | n | " + " | ".join(metrics + ["snr_gain"]) + " |", "|---|---|" + "---|" * (len(metrics) + 1)]
    for sysname, g in df[df.nominal].groupby("system"):
        cells = []
        for m in metrics + ["snr_gain"]:
            t = ci(g[m]); cells.append(fmt(t) + _mark(m, t))
        lines.append(f"| {sysname} | {len(g)} | " + " | ".join(cells) + " |")
    # The nominal average hides a large pass region under the 0 dB bucket. Report the declared operating envelope:
    # per noise class, the lowest input SNR from which every bucket up to +15 dB meets all three targets.
    lines += ["", "## Operating envelope (point estimates only: lowest input SNR at which all three means exceed targets and stay above at higher tested SNRs)", "",
              "| system | " + " | ".join(classes := sorted(df[df.fault.isna()].noise_class.dropna().unique())) + " |", "|---|" + "---|" * len(classes)]
    for sysname, g in df[df.fault.isna()].groupby("system"):
        lines.append(f"| {sysname} | " + " | ".join(_envelope(g[g.noise_class == c]) for c in classes) + " |")
    burst = df[df.fault.fillna("").str.startswith("fault_burst")]
    if len(burst):
        # loud transients live only in the burst fault buckets; the number the project is about must not omit them
        lines += ["", "## Transient-present envelope (fault_burst_* buckets: +24/+36 dB bursts and overload, input SNR 0/5 dB)", "",
                  "| system | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
        for sysname, g in burst.groupby("system"):
            lines.append(f"| {sysname} | {len(g)} | " + " | ".join(fmt(t := ci(g[m])) + _mark(m, t) for m in metrics) + " |")
    if df.fault.notna().any():
        # fault buckets share seeds with fault_none, so each row reads as a delta against that reference clip set
        lines += ["", "## Reliability faults (outside the nominal envelope; same speech/noise/room per seed, fault is the only variable)", "",
                  "Fault buckets, including fault_none, use input SNR 0/5 dB; nominal uses 0/5/10 dB. Compare faults with fault_none, not with the nominal aggregate.", "",
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
