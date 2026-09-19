"""What is the best your architecture could possibly do on YOUR eval set?

Computes, per input-SNR bucket, the output SNR/STOI/PESQ achievable by:
  * an oracle magnitude mask with noisy phase   -- the wall a magnitude masker hits
  * an oracle complex mask restricted to what your decoder can actually emit
    (projected onto the ERB band-split subspace, then tanh-bounded)  -- your true ceiling

Run:
    uv run python scripts/ceiling_analysis.py --eval-root data/eval --split test

Read it like this: if a trained system sits near the magnitude-mask row, more capacity
and more data will not take it past that row -- the remaining headroom is in phase, and
the fix is the complex mask (loss rebalance, deep-filter head), not a bigger encoder.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import stft as S
from vaani.models.gtcrn import ERB


def to_c(x):
    a = S.stft(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy()
    return a[..., 0] + 1j * a[..., 1]


def from_c(c, length):
    t = torch.from_numpy(np.stack([c.real, c.imag], -1).astype(np.float32))[None]
    return S.istft(t, length=length)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="data/eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-bucket", type=int, default=8)
    a = ap.parse_args()

    erb = ERB(65, 64).eval()
    B = erb.ierb_fc.weight.detach().numpy()          # (192, 64) ERB coeffs -> high bins
    P = B @ np.linalg.pinv(B)                        # orthogonal projector onto range(B)

    def erb_project(M):
        """Least-squares projection onto the mask subspace the decoder can emit.
        NB: bs(bm(.)) is NOT that subspace -- it is a badly scaled round trip and will
        make the ceiling look ~9 dB worse than it is."""
        out = M.copy()
        out[65:] = P @ M[65:]
        return out

    ds = RenderedDataset(Path(a.eval_root) / a.split)
    seen, acc = defaultdict(int), defaultdict(list)
    for i in range(len(ds)):
        it = ds[i]
        meta = it["meta"]
        bucket = meta.get("bucket", "?")
        if seen[bucket] >= a.per_bucket:
            continue
        seen[bucket] += 1
        mix, clean = it["mix"].numpy(), it["clean"].numpy()
        y, s = mix[0], clean
        n = y - s
        L = len(y)
        Y, Sp, N = to_c(y), to_c(s), to_c(n)
        eps = 1e-10

        irm = np.sqrt(np.abs(Sp) ** 2 / (np.abs(Sp) ** 2 + np.abs(N) ** 2 + eps)).astype(complex)
        ratio = Sp / (Y + eps)
        c1 = np.clip(ratio.real, -1, 1) + 1j * np.clip(ratio.imag, -1, 1)
        cp = erb_project(c1)
        cp = np.clip(cp.real, -1, 1) + 1j * np.clip(cp.imag, -1, 1)

        for name, est in (("unprocessed", y),
                          ("oracle magnitude mask", from_c(irm * Y, L)),
                          ("architecture ceiling", from_c(cp * Y, L))):
            est = np.asarray(est, np.float32)[:L]
            acc[(bucket, name)].append((metrics.snr_db(s, est), metrics.stoi(s, est),
                                        metrics.pesq_wb(s, est)))

    names = ["unprocessed", "oracle magnitude mask", "architecture ceiling"]
    print(f"{'bucket':<26s}" + "".join(f"{n[:22]:>24s}" for n in names))
    print(f"{'':<26s}" + "".join(f"{'SNR   STOI   PESQ':>24s}" for _ in names))
    for bucket in sorted(seen):
        line = f"{bucket:<26s}"
        for n in names:
            v = np.array(acc[(bucket, n)], float)
            v = v[np.isfinite(v).all(1)]
            line += f"{v[:, 0].mean():9.2f}{v[:, 1].mean():8.3f}{v[:, 2].mean():7.2f}"
        print(line)


if __name__ == "__main__":
    main()
