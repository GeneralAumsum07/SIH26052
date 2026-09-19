"""What the DSP controller actually does on the frozen eval set.

The ablation matrix reports end-to-end metrics; this reports the controller's own
behaviour, which is what the novelty claim is about.

Run:
    uv run python scripts/diag_controller.py --eval-root data/eval --split test

Reports, per bucket:
  burst_rate   fraction of frames with burst_flag set  (expected >0 in impulsive buckets)
  gate0        fraction of frames with adapt_gate == 0
  jump_p99     99th percentile of feature 0 (log_energy_delta), against the 12 dB threshold
  nhat_noise   corr(n_hat, mix_primary - clean)   <- n_hat tracking the noise (wanted)
  nhat_speech  corr(n_hat, clean)                 <- n_hat tracking the speech (unwanted)
  erle         10log10(||noise||^2 / ||primary - n_hat - clean||^2); negative = harmful
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline


def nc(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="data/eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-bucket", type=int, default=10)
    a = ap.parse_args()

    ds = RenderedDataset(Path(a.eval_root) / a.split)
    seen, rows = defaultdict(int), defaultdict(list)
    for i in range(len(ds)):
        it = ds[i]
        bucket = it["meta"]["bucket"]
        if seen[bucket] >= a.per_bucket:
            continue
        seen[bucket] += 1
        mix, clean = it["mix"].numpy(), it["clean"].numpy()
        r = pipeline.run(mix)
        noise = mix[0] - clean
        resid = mix[0] - r["n_hat"] - clean
        rows[bucket].append((
            r["burst"].mean(),
            (r["gate"] == 0).mean(),
            np.percentile(r["features"][:, 0], 99),
            nc(r["n_hat"], noise),
            nc(r["n_hat"], clean),
            10 * np.log10((noise ** 2).sum() / ((resid ** 2).sum() + 1e-20) + 1e-20),
        ))

    print(f"{'bucket':<26s} {'burst':>7s} {'gate0':>7s} {'jump_p99':>9s} "
          f"{'nhat~noise':>11s} {'nhat~speech':>12s} {'erle_dB':>8s}")
    for bucket in sorted(rows):
        v = np.array(rows[bucket]).mean(0)
        print(f"{bucket:<26s} {v[0]:7.3f} {v[1]:7.3f} {v[2]:9.2f} "
              f"{v[3]:11.3f} {v[4]:12.3f} {v[5]:8.2f}")
    allv = np.array([x for b in rows for x in rows[b]])
    print(f"\nOVERALL burst-frame rate {allv[:, 0].mean():.4f}   "
          f"frames over the 12 dB jump threshold (p99 basis): "
          f"{(allv[:, 2] >= 12).mean():.3f} of clips")


if __name__ == "__main__":
    main()
