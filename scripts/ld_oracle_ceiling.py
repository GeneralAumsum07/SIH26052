"""Oracle-mask ceiling per transform (low-delay plan Task 1b; training-free).

On the frozen validation split, for C0's transform, Arm A's three contracts and Arm B's contract, two realistic oracle
masks are applied to the primary microphone's spectrum and the result is re-synthesized through the same transform:

  irm   bounded ideal ratio mask, noisy phase: M = sqrt(|S|^2 / (|S|^2 + |N|^2)) in [0, 1], N = X - S, applied to X
  ccm   bounded compressed complex mask: in the model's power-law-compressed domain (c = 0.3, vaani_fe),
        M = S_c / X_c with |M| clipped to 1 and its phase kept; Y_c = M X_c is decompressed

The report gives SNR_out, STOI and PESQ-WB per transform with paired scene-clustered bootstrap intervals of the
difference from C0 under the same mask, and the frame-rate modulation index (compare_r8_ld.modulation_index) at each
low-delay hop rate, relative to C0 and to the clean target, on the stationary-noise and clean-speech clips. A
low-delay ceiling whose mean falls below C0's by more than the D2 margin is flagged to the owner before the pilots.
It is a ceiling of the transform, not a prediction of trained quality.

    python scripts/ld_oracle_ceiling.py --eval-root data/eval_r2 --split val [--per-bucket 4] [--limit N] \
        --out results_r2/r8_ld/oracle
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from vaani import audio_contract as ac  # noqa: E402

import compare_r8_ld as cmp  # noqa: E402

COMPRESS = 0.3
EPS = 1e-8
MASKS = ("irm", "ccm")
CONTRACTS = (ac.LEGACY_ID, *ac.ARM_A_IDS, ac.ARM_B_ID)
METRICS = ("snr_out", "stoi", "pesq_wb")
MARGINS = {m: cmp.D2[m]["margin"] for m in METRICS}
LD_HOPS = sorted({ac.get_audio_contract(c).hop for c in CONTRACTS if c != ac.LEGACY_ID})


def oracle(mix_p, clean, contract, mask):
    """Primary mixture and clean target (N,) -> the oracle-masked output (N,) through `contract`'s transform."""
    import torch
    from vaani.dsp import low_delay_stft as ld
    x = torch.as_tensor(np.asarray(mix_p, np.float64))[None]
    s = torch.as_tensor(np.asarray(clean, np.float64))[None]
    X = torch.view_as_complex(ld.analyze(x, contract).contiguous())
    S = torch.view_as_complex(ld.analyze(s, contract).contiguous())
    if mask == "irm":
        n2 = (X - S).abs() ** 2
        Y = X * torch.sqrt(S.abs() ** 2 / (S.abs() ** 2 + n2 + EPS))
    elif mask == "ccm":
        comp = lambda z: z * (z.abs() + EPS) ** (COMPRESS - 1)   # noqa: E731
        Xc, Sc = comp(X), comp(S)
        M = Sc / (Xc + EPS * (Xc.abs() < EPS))
        M = M * torch.clamp(M.abs(), max=1.0) / (M.abs() + EPS)
        Yc = M * Xc
        Y = Yc * (Yc.abs() + EPS) ** (1 / COMPRESS - 1)
    else:
        raise ValueError(f"unknown mask {mask!r}")
    y, _ = ld.synthesize(torch.view_as_real(Y), [x.shape[-1]], contract)
    return y[0].numpy().astype(np.float32)


def score_clip(meta, mix, clean, contracts=CONTRACTS, masks=MASKS):
    """One clip -> rows (one per contract and mask) with the registered metrics and the modulation index."""
    from vaani import metrics
    mix_p = np.asarray(mix)[0] if np.asarray(mix).ndim == 2 else np.asarray(mix)
    clean = np.asarray(clean, np.float32)
    rows = []
    base = {"id": str(meta.get("id")), "bucket": meta.get("bucket"), "noise_class": meta.get("noise_class"),
            "snr_in": meta.get("snr_db"), "fault": meta.get("fault")}
    clean_mod = {h: cmp.modulation_index(clean, h) for h in LD_HOPS}
    for c in contracts:
        for m in masks:
            y = oracle(mix_p, clean, c, m)
            r = dict(base, contract=c, mask=m, snr_out=metrics.snr_db(clean, y), stoi=metrics.stoi(clean, y),
                     pesq_wb=metrics.pesq_wb(clean, y))
            for h in LD_HOPS:
                o, k = cmp.mod_cols(h)
                r[o], r[k] = cmp.modulation_index(y, h), clean_mod[h]
            rows.append(r)
    return rows


def summarise(rows, n_boot=1000):
    """Per mask and low-delay contract: the paired difference from C0 with its scene-clustered interval, the flag, and
    the modulation index against C0 and the clean target."""
    import pandas as pd
    from vaani.report import cluster_ci
    df = pd.DataFrame(rows)
    df["scene"] = df.snr_in.astype(str) + "|" + df.id.astype(str)
    df["key"] = df.bucket.astype(str) + "/" + df.id.astype(str)
    nc = df.noise_class.fillna("").astype(str)
    df["stationary"] = nc.eq("stationary") & df.fault.isna()
    df["clean"] = nc.eq("clean") | df.bucket.astype(str).str.startswith("clean")
    out = {"n_clips": int(df.key.nunique()), "masks": {}, "flags": []}
    for m in MASKS:
        c0 = df[(df["mask"] == m) & (df.contract == ac.LEGACY_ID)].set_index("key")
        res = {"c0_means": {k: float(c0[k].mean()) for k in METRICS}, "contracts": {}}
        for c in CONTRACTS[1:]:
            x = df[(df["mask"] == m) & (df.contract == c)].set_index("key").loc[c0.index]
            r = {"means": {k: float(x[k].mean()) for k in METRICS}, "d_vs_c0": {}}
            for k in METRICS:
                d = x[k].to_numpy(float) - c0[k].to_numpy(float)
                mean, lo, hi = cluster_ci(d, c0.scene.to_numpy(), n=n_boot)
                below = bool(mean < -MARGINS[k])
                r["d_vs_c0"][k] = {"mean": mean, "lo": lo, "hi": hi, "margin": MARGINS[k], "below_c0_by_margin": below}
                if below:
                    out["flags"].append(f"{m} {c}: {k} ceiling {mean:+.4f} vs C0 (margin {MARGINS[k]})")
            hop = ac.get_audio_contract(c).hop
            o, k2 = cmp.mod_cols(hop)
            r["modulation_index"] = {"hop": hop, "hop_hz": ac.SR / hop}
            for sub in ("stationary", "clean"):
                sel = x[sub].to_numpy(bool)
                r["modulation_index"][sub] = {
                    "n_clips": int(sel.sum()),
                    "db_vs_c0": float(np.mean(x[o].to_numpy(float)[sel] - c0[o].to_numpy(float)[sel])) if sel.any() else None,
                    "db_vs_clean": float(np.mean(x[o].to_numpy(float)[sel] - x[k2].to_numpy(float)[sel])) if sel.any() else None}
            res["contracts"][c] = r
        out["masks"][m] = res
    return out


def markdown(rep) -> str:
    L = ["# Oracle-mask ceiling per transform (Task 1b)", "",
         f"{rep['summary']['n_clips']} validation clips from `{rep['source']}`. Revision `{rep['provenance']['revision']}`.",
         "A ceiling of each transform under oracle masks, not a prediction of trained quality. Differences are "
         "(contract - C0) under the same mask, with 95 % scene-clustered bootstrap intervals.", ""]
    flags = rep["summary"]["flags"]
    L += ["## Flags for the owner (mean below C0 by more than the D2 margin)", ""] + ([f"- {f}" for f in flags] or ["none"])
    for m, res in rep["summary"]["masks"].items():
        L += ["", f"## Mask `{m}`", "", "C0 means: " + ", ".join(f"{k} {v:.4f}" for k, v in res["c0_means"].items()), "",
              "| contract | " + " | ".join(f"d {k}" for k in METRICS) + " | mod idx vs C0 / clean, stationary (dB) | clean speech (dB) |",
              "|---|" + "---|" * (len(METRICS) + 2)]
        for c, r in res["contracts"].items():
            cells = [f"{r['d_vs_c0'][k]['mean']:+.4f} [{r['d_vs_c0'][k]['lo']:+.4f}, {r['d_vs_c0'][k]['hi']:+.4f}]"
                     + (" **flag**" if r["d_vs_c0"][k]["below_c0_by_margin"] else "") for k in METRICS]
            mi = r["modulation_index"]
            f = lambda s: "n/a" if mi[s]["db_vs_c0"] is None else f"{mi[s]['db_vs_c0']:+.2f} / {mi[s]['db_vs_clean']:+.2f}"  # noqa: E731
            L.append(f"| {c} | " + " | ".join(cells) + f" | {f('stationary')} | {f('clean')} |")
    return "\n".join(L) + "\n"


def select(root, per_bucket):
    from vaani.data.dataset import RenderedDataset
    items = RenderedDataset(root).items
    seen, idx = {}, []
    for i, p in enumerate(items):
        b = p.parent.name
        if per_bucket is None or seen.get(b, 0) < per_bucket:
            seen[b] = seen.get(b, 0) + 1
            idx.append(i)
    return idx


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-root", default="data/eval_r2"); ap.add_argument("--split", default="val")
    ap.add_argument("--per-bucket", type=int); ap.add_argument("--limit", type=int)
    ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--out", default="results_r2/r8_ld/oracle")
    a = ap.parse_args(argv)
    root = Path(a.eval_root) / a.split
    if not root.is_dir():
        print(f"no rendered split at {root}", file=sys.stderr)
        return 2
    from vaani.data.dataset import RenderedDataset
    ds = RenderedDataset(root)
    idx = select(root, a.per_bucket)[: a.limit]
    rows = []
    for n, i in enumerate(idx):
        it = ds[i]
        rows += score_clip(it["meta"], it["mix"].numpy(), it["clean"].numpy())
        if n % 50 == 0:
            print(f"{n + 1}/{len(idx)}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "clips.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    rep = {"schema": "r8_ld_oracle_ceiling_v1", "source": str(root), "contracts": list(CONTRACTS),
           "contract_hashes": {c: ac.get_audio_contract(c).contract_hash for c in CONTRACTS},
           "masks": {"irm": "sqrt(|S|^2/(|S|^2+|X-S|^2)), noisy phase",
                     "ccm": f"S_c/X_c in the |.|^{COMPRESS} compressed domain, |M| <= 1"},
           "provenance": cmp.provenance(["ld_oracle_ceiling.py", *argv]), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "summary": summarise(rows, a.n_boot)}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    (out / "report.md").write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
