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

Plan 2.7 gate (--limiter / --diff-jump-max, ablatable DSP config): also prints burst TPR = share of
impulse onsets in fault_burst_* clips flagged within 3 frames (target > 0.5) and FPR = burst-frame rate
on fault_none clips, i.e. speech + stationary noise with no burst at all (target < 5 %).
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
    ap.add_argument("--limiter", action="store_true", help="2.7b sub-block limiter ahead of the STFT")
    ap.add_argument("--blocking", action="store_true", help="2.8 blocking matrix on the reference path")
    ap.add_argument("--block-margin", type=float, default=0.0, help="2.8: adapt only when prim/ref ratio exceeds the noise ratio by this (dB)")
    ap.add_argument("--headroom", type=float, default=None, help="limiter headroom over the running level (dB)")
    ap.add_argument("--diff-jump-max", type=float, default=None, help="2.7a burst rule threshold (dB); None = legacy level_diff rule")
    a = ap.parse_args()
    lim = a.limiter if a.headroom is None else {"headroom_db": a.headroom}
    ctl = {} if a.diff_jump_max is None else {"diff_jump_max_db": a.diff_jump_max}
    if a.block_margin: ctl["block_margin_db"] = a.block_margin
    dsp_cfg = {"limiter": lim, "blocking": a.blocking, "controller": ctl}
    print(f"dsp_cfg = {dsp_cfg}")

    ds = RenderedDataset(Path(a.eval_root) / a.split)
    seen, rows = defaultdict(int), defaultdict(list)
    hits, onsets, fp_frames, none_frames = 0, 0, 0, 0
    for i in range(len(ds)):
        it = ds[i]
        bucket = it["meta"]["bucket"]
        if seen[bucket] >= a.per_bucket:
            continue
        seen[bucket] += 1
        mix, clean = it["mix"].numpy(), it["clean"].numpy()
        r = pipeline.run(mix, dsp_cfg=dsp_cfg)
        if bucket.startswith("fault_burst"):
            for t in it["meta"].get("impulse_onsets_s", []):
                k = int(t * 16000 / 256); onsets += 1; hits += bool(r["burst"][k:k + 4].any())
        elif bucket.startswith("fault_none"):
            fp_frames += int(r["burst"].sum()); none_frames += len(r["burst"])
        mix = r["mix"]   # limited when the limiter is on: ERLE and n_hat are judged against what the model sees
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
    if onsets:
        print(f"2.7 gate: burst TPR {hits / onsets:.3f} ({hits}/{onsets} onsets, target > 0.5)   "
              f"FPR on fault_none {fp_frames / max(none_frames, 1):.4f} (target < 0.05)")


if __name__ == "__main__":
    main()
