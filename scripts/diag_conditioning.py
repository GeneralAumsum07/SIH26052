"""Is the DSP conditioning path actually doing anything?

Answers, for a trained VaaniNet checkpoint, the one question the ablation matrix
cannot: does the model use the 18-dim feature vector and the NLMS channel at all?

Run:
    uv run python scripts/diag_conditioning.py --ckpt runs/vaani_full/best.pt --n 120

Interpretation:
  * |dSTOI| for "feats zeroed" close to 0  -> the FiLM conditioning is inert; the
    controller cannot possibly show up in the ablation matrix.
  * |dSTOI| for "n_hat zeroed" close to 0  -> the NLMS channel is inert too, and the
    dual-channel gain is coming from the reference spectrum alone.
  * film_gain_ratio << 1                   -> the FiLM shift is negligible next to the
    activations it is added to.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from pystoi import stoi as _stoi

from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline, stft
from vaani.train import build_model

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-root", default="data/eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=120, help="clips to sample (stratified over buckets)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    assert cfg["model"] == "vaani", "this diagnostic only applies to VaaniNet checkpoints"
    m = build_model("vaani")
    m.load_state_dict(ck["model"])
    m.eval()

    # --- static check: how big is the FiLM shift next to the activations it modifies? ---
    w = m.encoder.film.weight.detach()
    b = m.encoder.film.bias.detach()
    print(f"film.weight  |w|_mean {w.abs().mean():.5f}  |w|_max {w.abs().max():.5f}")
    print(f"film.bias    |b|_mean {b.abs().mean():.5f}  |b|_max {b.abs().max():.5f}")

    ds = RenderedDataset(Path(a.eval_root) / a.split)
    idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)

    variants = ["as trained", "feats zeroed", "n_hat zeroed", "ref zeroed", "feats+n_hat zeroed"]
    acc = {v: [] for v in variants}
    film_ratio = []

    # capture the activation the FiLM shift is added to, to size the shift against it
    grab = {}
    h = m.encoder.en_convs[0].register_forward_hook(lambda _m, _i, o: grab.__setitem__("x", o.detach()))

    with torch.no_grad():
        for i in idx:
            it = ds[int(i)]
            mix = it["mix"].numpy()
            clean = it["clean"].numpy()
            r = pipeline.run(mix, controller_on=cfg["controller_on"])
            x = torch.from_numpy(mix)[None]
            P = stft.stft(x[:, 0])
            R = stft.stft(x[:, 1])
            N = stft.stft(torch.from_numpy(r["n_hat"])[None])
            F = torch.from_numpy(r["features"])[None]
            Z = torch.zeros_like(F)

            for name, spec6, feats in [
                ("as trained", torch.cat([P, R, N], -1), F),
                ("feats zeroed", torch.cat([P, R, N], -1), Z),
                ("n_hat zeroed", torch.cat([P, R, torch.zeros_like(N)], -1), F),
                ("ref zeroed", torch.cat([P, torch.zeros_like(R), N], -1), F),
                ("feats+n_hat zeroed", torch.cat([P, R, torch.zeros_like(N)], -1), Z),
            ]:
                out = m(spec6, feats)
                y = stft.istft(out, length=mix.shape[1])[0].numpy()
                acc[name].append(_stoi(clean, y, SR, extended=False))
                if name == "as trained":
                    shift = m.encoder.film(torch.clamp(F * m.encoder.feat_scale, -3.0, 3.0))
                    film_ratio.append(float(shift.abs().mean() / (grab["x"].abs().mean() + 1e-9)))
    h.remove()

    base = float(np.mean(acc["as trained"]))
    print(f"\nclips: {len(idx)}   checkpoint: {a.ckpt}")
    print(f"film_shift / activation magnitude: {np.mean(film_ratio):.4f}"
          "   (<0.01 means the conditioning is numerically irrelevant)\n")
    print(f"{'variant':<22s} {'STOI':>7s} {'dSTOI':>8s}")
    for v in variants:
        mu = float(np.mean(acc[v]))
        print(f"{v:<22s} {mu:7.4f} {mu - base:+8.4f}")


if __name__ == "__main__":
    main()
