"""Is the DSP conditioning path actually doing anything?

Answers, for a trained VaaniNet checkpoint, the one question the ablation matrix
cannot: does the model use the 18-dim feature vector and the NLMS channel at all?

Run:
    uv run python scripts/diag_conditioning.py --ckpt runs/vaani_full/best.pt --n 120
    uv run --with numba python scripts/diag_conditioning.py --ckpt runs/r7_e256_wr64_refiner/best.pt \
        --eval-root data/eval_r2 --split val --n 0 --out results_r2/r7/diag/conditioning_cascade

A cascade checkpoint (vaani_cascade) is accepted: the ablations act on the frozen first stage's
inputs, and the refiner sees the same spec6, so "ref zeroed" zeroes the reference for both stages.
--out writes <out>.csv (one row per clip x variant: SNR_out, STOI, PESQ-WB) and <out>.json
(per-variant means and paired deltas with a 1000-sample bootstrap CI over clips).
With --out, rows are appended to <out>.partial.csv as each clip finishes, so rerunning the same
command resumes; a clip that killed the previous process (native crash) is skipped and listed
under "skipped_ids" in <out>.json.

Interpretation:
  * |dSTOI| for "feats zeroed" close to 0  -> the FiLM conditioning is inert; the
    controller cannot possibly show up in the ablation matrix.
  * |dSTOI| for "n_hat zeroed" close to 0  -> the NLMS channel is inert too, and the
    dual-channel gain is coming from the reference spectrum alone.
  * film_gain_ratio << 1                   -> the FiLM shift is negligible next to the
    activations it is added to.
  * "coh zeroed" dSTOI = zeroed minus as-trained STOI. A negative value means
    coherence helps; e.g. -0.01 is a 0.01 absolute STOI loss on zeroing. Near zero
    suggests no measured benefit on these clips, not proof the pathway is unused.
"""
import argparse
import faulthandler
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline, stft
from vaani.models import cascade
from vaani.train import build_model

SR = 16000


@contextmanager
def zero_coherence(model):
    """Ablate only the coherence feature, retaining the three input spectra."""
    if not model.use_coh:
        raise ValueError("coherence ablation requires a coh=true checkpoint")

    def zero_channel(_module, inputs):
        # ERB is linear and per channel, so zeroing here also zeros every SFE
        # copy of coherence without recomputing or damaging reference features.
        features = inputs[0].clone()
        assert features.shape[1] == 10, "expected nine spectral channels plus coherence"
        features[:, 9] = 0
        return (features,)

    handle = model.sfe.register_forward_pre_hook(zero_channel)
    try:
        yield
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-root", default="data/eval")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=120, help="clips to sample (stratified over buckets); 0 = every clip")
    ap.add_argument("--out", help="write <out>.csv (per clip x variant) and <out>.json (summary)")
    a = ap.parse_args()
    try:
        faulthandler.enable()  # a native crash (rc=139) otherwise leaves no stack in the job log
    except (AttributeError, OSError, ValueError):  # captured stderr (pytest) has no fileno
        pass
    torch.set_num_threads(1)  # one core per diagnostic process; several run side by side

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    assert cfg["model"] in ("vaani", cascade.MODEL_NAME), "this diagnostic only applies to VaaniNet (or cascade) checkpoints"
    if cfg["model"] == cascade.MODEL_NAME:
        model = cascade.FrozenCascade.from_config(cfg)
        model.load_state_dict(ck["model"])
        m = model.first  # every ablation targets the first stage's inputs and modules
    else:
        model = m = build_model("vaani", model_cfg=cfg.get("model_cfg"))
        m.load_state_dict(ck["model"])
    model.eval()

    # --- static check: how big is the FiLM shift next to the activations it modifies? ---
    film = m.encoder.film
    if film is not None:
        w = film.weight.detach()
        b = film.bias.detach()
        print(f"film.weight  |w|_mean {w.abs().mean():.5f}  |w|_max {w.abs().max():.5f}")
        print(f"film.bias    |b|_mean {b.abs().mean():.5f}  |b|_max {b.abs().max():.5f}")
    else:
        print("FiLM disabled; feature-zeroing variants are expected no-ops.")

    ds = RenderedDataset(Path(a.eval_root) / a.split)
    idx = np.arange(len(ds)) if a.n <= 0 else np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)

    variants = ["as trained", "feats zeroed", "n_hat zeroed", "ref zeroed", "feats+n_hat zeroed"]
    if film is None:  # features only reach the net through FiLM; zeroing them is an exact no-op, so skip the cost
        variants = ["as trained", "n_hat zeroed", "ref zeroed"]
    if m.use_coh:
        variants.append("coh zeroed")
    rows = []
    done, skipped = set(), []
    part = inflight = None
    if a.out:
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        part, inflight = out.with_suffix(".partial.csv"), out.with_suffix(".inflight")
        if part.exists():
            rows = pd.read_csv(part).to_dict("records")
            done = {int(r["clip"]) for r in rows}
        if inflight.exists():  # the previous process died inside this clip; skip it rather than crash again
            skipped = [int(x) for x in inflight.read_text().split()]
            done |= set(skipped)

    # capture the activation the FiLM shift is added to, to size the shift against it
    grab = {}
    h = m.encoder.en_convs[0].register_forward_hook(lambda _m, _i, o: grab.__setitem__("x", o.detach()))

    with torch.no_grad():
        for i in idx:
            if int(i) in done:
                continue
            if inflight is not None:
                inflight.write_text(" ".join(str(x) for x in skipped + [int(i)]))
            it = ds[int(i)]
            mix = it["mix"].numpy()
            clean = it["clean"].numpy()
            r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"))
            x = torch.from_numpy(r["mix"])[None]
            P = stft.stft(x[:, 0])
            R = stft.stft(x[:, 1])
            N = stft.stft(torch.from_numpy(r["n_hat"])[None])
            F = torch.from_numpy(r["features"])[None]
            Z = torch.zeros_like(F)

            meta = it["meta"]
            inputs = [
                ("as trained", torch.cat([P, R, N], -1), F),
                ("feats zeroed", torch.cat([P, R, N], -1), Z),
                ("n_hat zeroed", torch.cat([P, R, torch.zeros_like(N)], -1), F),
                ("ref zeroed", torch.cat([P, torch.zeros_like(R), N], -1), F),
                ("feats+n_hat zeroed", torch.cat([P, R, torch.zeros_like(N)], -1), Z),
            ]
            if m.use_coh:
                inputs.append(("coh zeroed", torch.cat([P, R, N], -1), F))
            inputs = [t for t in inputs if t[0] in variants]
            clip_rows = []
            for name, spec6, feats in inputs:
                with zero_coherence(m) if name == "coh zeroed" else nullcontext():
                    y = model(spec6, feats)
                y = stft.istft(y, length=mix.shape[1])[0].numpy()
                fr = np.nan
                if name == "as trained" and film is not None:
                    shift = m.encoder.film(torch.clamp(F * m.encoder.feat_scale, -3.0, 3.0))
                    fr = float(shift.abs().mean() / (grab["x"].abs().mean() + 1e-9))
                clip_rows.append(dict(clip=int(i), id=meta.get("id"), bucket=meta.get("bucket"), variant=name,
                                      snr_out=metrics.snr_db(clean, y), stoi=metrics.stoi(clean, y),
                                      pesq_wb=metrics.pesq_wb(clean, y), film_ratio=fr))
            rows += clip_rows
            if part is not None:
                pd.DataFrame(clip_rows).to_csv(part, mode="a", header=not part.exists(), index=False)
    h.remove()

    df = pd.DataFrame(rows)
    acc = {v: df.loc[df["variant"] == v, "stoi"].to_numpy() for v in variants}
    film_ratio = df["film_ratio"].dropna().tolist()
    base = float(np.mean(acc["as trained"]))
    print(f"\nclips: {df['clip'].nunique()}   skipped (native crash): {len(skipped)}   checkpoint: {a.ckpt}")
    if film_ratio:
        print(f"film_shift / activation magnitude: {np.mean(film_ratio):.4f}"
              "   (<0.01 means the conditioning is numerically irrelevant)\n")
    print("dSTOI = variant - as trained (negative means zeroing hurts).")
    print(f"{'variant':<22s} {'STOI':>7s} {'dSTOI':>8s}")
    for v in variants:
        mu = float(np.mean(acc[v]))
        print(f"{v:<22s} {mu:7.4f} {mu - base:+8.4f}")

    if a.out:
        df.drop(columns=["clip", "film_ratio"]).to_csv(out.with_suffix(".csv"), index=False)
        wide = df.pivot(index=["bucket", "id"], columns="variant", values=["snr_out", "stoi", "pesq_wb"])
        rng = np.random.default_rng(0)
        boot = rng.integers(0, len(wide), (1000, len(wide)))  # one clip resample shared by every variant and metric
        summary = {"checkpoint": a.ckpt, "model": cfg["model"], "eval_root": a.eval_root, "split": a.split,
                   "n_clips": int(len(wide)), "skipped_ids": [ds[k]["meta"].get("id") for k in skipped],
                   "film_shift_ratio": float(np.mean(film_ratio)) if film_ratio else None,
                   "variants": {}}
        for v in variants:
            summary["variants"][v] = {}
            for k in ("snr_out", "stoi", "pesq_wb"):
                d = (wide[(k, v)] - wide[(k, "as trained")]).to_numpy()
                bm = np.nanmean(d[boot], axis=1)
                summary["variants"][v][k] = {"mean": float(np.nanmean(wide[(k, v)])), "delta": float(np.nanmean(d)),
                                             "delta_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                             "n_finite": int(np.isfinite(d).sum())}
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        part.unlink(missing_ok=True); inflight.unlink(missing_ok=True)
        print(f"wrote {out.with_suffix('.csv')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
