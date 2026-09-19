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

Run this after every training round. It is the cheapest way to tell whether a loss
change actually bought you phase, rather than just moving the aggregate metric.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    ap.add_argument("--per-bucket", type=int, default=6)
    a = ap.parse_args()

    fn = enhance_fn(a.system, device="cpu")
    ds = RenderedDataset(Path(a.eval_root) / a.split)
    seen, acc, ph = defaultdict(int), defaultdict(list), defaultdict(list)

    for i in range(len(ds)):
        it = ds[i]
        bucket = it["meta"].get("bucket", "?")
        if seen[bucket] >= a.per_bucket:
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
        for k, v in variants.items():
            v = np.asarray(v, np.float32)[:L]
            acc[(bucket, k)].append((metrics.snr_db(clean, v), metrics.stoi(clean, v),
                                     metrics.pesq_wb(clean, v)))

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


if __name__ == "__main__":
    main()
