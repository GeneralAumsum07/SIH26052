"""Is the model using its complex mask, or is it a magnitude masker in disguise?

For a trained checkpoint (or the pretrained GTCRN), reports per bucket:
  mask_phase   mean |angle| of the applied mask on speech-dominant bins.
               Under ~15 deg means the complex mask is barely doing complex work,
               and the 15 dB SNR target is out of reach below about +4 dB input SNR.
  +oracle phase / +oracle magnitude
               the model's own output with one component replaced by the ideal one.
               Whichever swap moves SNR more is your first-order error term.

Run:
    uv run python scripts/mask_phase_probe.py --system ckpt:runs/vaani_full/best.pt
    uv run python scripts/mask_phase_probe.py --system gtcrn_pretrained
    uv run --with numba python scripts/mask_phase_probe.py --system cascade:runs/r7_e256_wr64_refiner/best.pt \
        --eval-root data/eval_r2 --split val --per-bucket 0 --out results_r2/r7/diag/mask_phase_cascade

Any `vaani.eval.enhance_fn` spec works (ckpt:, cascade:, onnx:, a baseline name). --out writes
<out>.csv (one row per clip) and <out>.json (per-bucket and overall means, oracle-phase headroom).

Run this after every training round. It is the cheapest way to tell whether a loss
change actually bought you phase, rather than just moving the aggregate metric.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import stft as S
from vaani.eval import enhance_fn


def to_c(x):
    a = S.stft(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy()
    return a[..., 0] + 1j * a[..., 1]


def from_c(c, length):
    t = torch.from_numpy(np.stack([c.real, c.imag], -1).astype(np.float32))[None]
    return S.istft(t, length=length)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="ckpt:runs/<name>/best.pt or a baseline name")
    ap.add_argument("--eval-root", default="data/eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-bucket", type=int, default=6, help="clips per bucket; 0 = every clip")
    ap.add_argument("--out", help="write <out>.csv (per clip) and <out>.json (summary)")
    a = ap.parse_args()
    torch.set_num_threads(1)  # one core per diagnostic process; several run side by side
    rows = []

    fn = enhance_fn(a.system, device="cpu")
    ds = RenderedDataset(Path(a.eval_root) / a.split)
    seen, acc, ph = defaultdict(int), defaultdict(list), defaultdict(list)

    for i in range(len(ds)):
        it = ds[i]
        bucket = it["meta"].get("bucket", "?")
        if a.per_bucket > 0 and seen[bucket] >= a.per_bucket:
            continue
        seen[bucket] += 1
        mix, clean = it["mix"].numpy(), it["clean"].numpy()
        L = mix.shape[1]
        est = np.asarray(fn(mix), np.float32)[:L]
        Y, Sp, E = to_c(mix[0]), to_c(clean), to_c(est)
        eps = 1e-10
        M = E / (Y + eps)                 # the mask the model actually applied
        Mi = Sp / (Y + eps)               # the ideal one
        w = np.abs(Sp) > np.percentile(np.abs(Sp), 70)
        ph[bucket].append(float((np.abs(np.angle(M[w])) * 180 / np.pi).mean()))

        variants = {
            "model": est,
            "+oracle phase": from_c(np.abs(M) * np.exp(1j * np.angle(Mi)) * Y, L),
            "+oracle magnitude": from_c(np.clip(np.abs(Mi), 0, 2) * np.exp(1j * np.angle(M)) * Y, L),
        }
        row = dict(id=it["meta"].get("id"), bucket=bucket, mask_phase_deg=ph[bucket][-1])
        for k, v in variants.items():
            v = np.asarray(v, np.float32)[:L]
            acc[(bucket, k)].append((metrics.snr_db(clean, v), metrics.stoi(clean, v),
                                     metrics.pesq_wb(clean, v)))
            tag = {"model": "model", "+oracle phase": "oph", "+oracle magnitude": "omag"}[k]
            row.update({f"{tag}_snr": acc[(bucket, k)][-1][0], f"{tag}_stoi": acc[(bucket, k)][-1][1],
                        f"{tag}_pesq": acc[(bucket, k)][-1][2]})
        rows.append(row)

    print(f"system: {a.system}\n")
    print(f"{'bucket':<26s}{'mask_phase':>11s}" + "".join(
        f"{k:>26s}" for k in ("model", "+oracle phase", "+oracle magnitude")))
    print(f"{'':<26s}{'(deg)':>11s}" + "".join(f"{'SNR    STOI   PESQ':>26s}" for _ in range(3)))
    for bucket in sorted(seen):
        line = f"{bucket:<26s}{np.mean(ph[bucket]):11.1f}"
        for k in ("model", "+oracle phase", "+oracle magnitude"):
            v = np.array(acc[(bucket, k)], float)
            v = v[np.isfinite(v).all(1)]
            line += f"{v[:, 0].mean():10.2f}{v[:, 1].mean():9.3f}{v[:, 2].mean():7.2f}"
        print(line)

    if a.out:
        df = pd.DataFrame(rows)
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out.with_suffix(".csv"), index=False)
        cols = [c for c in df.columns if c not in ("id", "bucket")]

        def summ(g):
            d = {c: float(np.nanmean(g[c])) for c in cols}
            # headroom = what swapping in the ideal phase (or magnitude) would buy over the model's own output
            for t in ("oph", "omag"):
                for mt in ("snr", "stoi", "pesq"):
                    d[f"{t}_gain_{mt}"] = float(np.nanmean(g[f"{t}_{mt}"] - g[f"model_{mt}"]))
            d["n"] = int(len(g))
            return d
        bb = {b: summ(g) for b, g in df.groupby("bucket")}
        ph_b = [v["mask_phase_deg"] for v in bb.values()]
        summary = {"system": a.system, "eval_root": a.eval_root, "split": a.split, "n_clips": int(len(df)),
                   "overall": summ(df), "bucket_mask_phase_deg_range": [float(min(ph_b)), float(max(ph_b))],
                   "per_bucket": bb}
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(f"wrote {out.with_suffix('.csv')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
