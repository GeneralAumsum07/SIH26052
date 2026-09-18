"""Render frozen val/test sets, bucketed by noise class x input SNR.
Burst buckets also get a 'twin' with the impulse removed (same seed) for
the recovery-time metric. Writes an eval-set hash for run.json.
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from vaani.data import impulses, manifests
from vaani.data.dataset import BUCKET_SNRS, SR, _load
from vaani.data.mixer import MixConfig, mix
from vaani.data.rirs import RirBank

CLASSES = {"stationary": ("stationary", False), "changing": ("changing", False),
           "impulsive": ("changing", True), "impulsive+stationary": ("stationary", True)}


def render_bucket_item(seed: list[int], speech_df, pool_df, n: int, snr: float, burst: bool, bank):
    """One eval clip. Returns (mix, clean, meta, twin_or_None); twin only set for burst buckets."""
    rng = np.random.default_rng(seed)
    x = _load(speech_df.path.iloc[int(rng.integers(len(speech_df)))], n, rng)
    s = np.pad(x, (0, n - len(x)))
    nz = [_load(pool_df.path.iloc[int(rng.integers(len(pool_df)))], n, rng)]
    cfg = MixConfig(snr_range=(snr, snr), p_clean=0.0)
    imp, on = (impulses.generate(np.random.default_rng(seed + [7])) if burst else (None, []))
    if burst:
        imp, on = imp, on["onsets_s"]
    m, c, meta = mix(np.random.default_rng(seed), s, nz, imp, on, bank, cfg)
    twin = None
    if burst:  # identical draw with no impulse
        twin, _, _ = mix(np.random.default_rng(seed), s, nz, None, [], bank, cfg)
    return m, c, meta, twin


def main(a):
    df = pd.concat([manifests.read(p) for p in a.manifests]); df = df[df.split == a.split]
    speech, noise = df[df.kind == "speech"], df[df.kind == "noise"]
    bank = RirBank(a.bank) if Path(a.bank).exists() else None
    root = Path(a.out) / a.split; n = int(a.clip_s * SR)

    for cls, (cont_cls, burst) in CLASSES.items():
        pool = noise[noise.noise_class == cont_cls]
        if pool.empty:
            # silently substituting a different noise class would make the bucket's
            # label lie about its contents, so this must fail loudly instead
            raise ValueError(f"no noise rows with noise_class={cont_cls!r} for bucket {cls!r}")
        for snr in BUCKET_SNRS:
            d = root / f"{cls}_{snr}"; d.mkdir(parents=True, exist_ok=True)
            for i in range(a.per_bucket):
                # stable_hash, not hash(): hash() is process-salted and would break reproducibility
                seed = [a.seed, manifests.stable_hash(cls) % 1000, snr + 100, i]
                m, c, meta, twin = render_bucket_item(seed, speech, pool, n, snr, burst, bank)
                meta["noise_class"] = cls
                sf.write(d / f"{i:04d}.mix.wav", m.T, SR); sf.write(d / f"{i:04d}.clean.wav", c, SR)
                json.dump(meta, open(d / f"{i:04d}.json", "w"))
                if twin is not None:
                    sf.write(d / f"{i:04d}.twin.mix.wav", twin.T, SR)

    # clean bucket: no noise at all
    d = root / "clean_inf"; d.mkdir(exist_ok=True)
    for i in range(a.per_bucket):
        rng = np.random.default_rng([a.seed, 999, i])
        s = np.pad((x := _load(speech.path.iloc[int(rng.integers(len(speech)))], n, rng)), (0, n - len(x)))
        m, c, meta = mix(rng, s, [np.zeros(n, np.float32)], None, [], bank, MixConfig(p_clean=1.0))
        meta["noise_class"] = "clean"
        sf.write(d / f"{i:04d}.mix.wav", m.T, SR); sf.write(d / f"{i:04d}.clean.wav", c, SR)
        json.dump(meta, open(d / f"{i:04d}.json", "w"))

    h = hashlib.sha1()
    for p in sorted(root.rglob("*.json")): h.update(p.read_bytes())
    (root / "EVALSET_HASH").write_text(h.hexdigest()[:12]); print("eval-set hash", h.hexdigest()[:12])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--bank", default="data/rirs/bank.npz")
    ap.add_argument("--per-bucket", type=int, default=40)
    ap.add_argument("--clip-s", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1234)
    main(ap.parse_args())
