"""Train = dynamic mixing (a fresh mixture every item, infinite variety).
Val/test = rendered once, frozen, so numbers across runs are comparable.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from vaani.data import impulses, manifests
from vaani.data.mixer import MixConfig, mix
from vaani.data.rirs import RirBank
from vaani.dsp import pipeline

SR = 16000
BUCKET_SNRS = [-10, -5, 0, 5, 10, 15]
NOISE_CLASSES = ["stationary", "changing", "impulsive", "impulsive+stationary", "clean"]


def _load(path: str, n: int | None, rng) -> np.ndarray:
    info = sf.info(path)
    if n is None or info.frames <= n:
        x, _ = sf.read(path, dtype="float32"); return x
    start = int(rng.integers(0, info.frames - n + 1))
    x, _ = sf.read(path, dtype="float32", start=start, frames=n); return x


class DynamicMixDataset(Dataset):
    def __init__(self, manifest_paths, split, bank_path, cfg: MixConfig, crop_s=4.0, epoch_len=20000, seed=0,
                 with_dsp=False, controller_on=True):
        df = pd.concat([manifests.read(p) for p in manifest_paths])
        df = df[df.split == split]
        self.speech = df[df.kind == "speech"].reset_index(drop=True)
        self.noise = df[df.kind == "noise"].reset_index(drop=True)
        # store the path, not an open RirBank in __init__, so workers open their own handle
        self.bank_path = bank_path
        self._bank = None
        self.cfg, self.n, self.epoch_len, self.seed = cfg, int(crop_s * SR), epoch_len, seed
        # with_dsp: run NLMS+features here so the ~150 ms/clip DSP lands in DataLoader workers, not the trainer
        self.with_dsp, self.controller_on = with_dsp, controller_on
        assert len(self.speech) and len(self.noise), "empty manifest split"

    @property
    def bank(self):
        # lazy-open: mmap'd npz handles don't survive pickling across worker spawn
        if self.bank_path and self._bank is None:
            self._bank = RirBank(self.bank_path)
        return self._bank

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        # epoch rides in the index (see EpochSampler): persistent workers never see attribute changes
        epoch, i = divmod(int(idx), self.epoch_len)
        rng = np.random.default_rng([self.seed, epoch, i])
        s = _load(self.speech.path.iloc[int(rng.integers(len(self.speech)))], self.n, rng)
        s = np.pad(s, (0, self.n - len(s)))
        # noise draw: continuous class(es) plus optionally an impulsive one
        k = int(rng.integers(1, 3))
        cont = self.noise[self.noise.noise_class != "impulsive"]
        rows = [cont.iloc[int(rng.integers(len(cont)))] for _ in range(k)]
        noises = [_load(r.path, self.n, rng) for r in rows]
        noise_class = "stationary" if all(r.noise_class == "stationary" for r in rows) else "changing"
        imp, onsets = None, []
        if rng.random() < 0.5:
            impd = self.noise[self.noise.noise_class == "impulsive"]
            if len(impd) and rng.random() < 0.5:
                imp = _load(impd.path.iloc[int(rng.integers(len(impd)))], 2 * SR, rng); onsets = [0.0]
            else:
                imp, m = impulses.generate(rng); onsets = m["onsets_s"]
            noise_class = "impulsive+stationary" if noise_class == "stationary" else "impulsive"
        mixed, clean, meta = mix(rng, s, noises, imp, onsets, self.bank, self.cfg)
        meta["noise_class"] = "clean" if meta["clean_bucket"] else noise_class
        out = {"mix": torch.from_numpy(mixed), "clean": torch.from_numpy(clean), "meta": meta}
        if self.with_dsp:
            r = pipeline.run(mixed, controller_on=self.controller_on)
            out["n_hat"] = torch.from_numpy(r["n_hat"]); out["feats"] = torch.from_numpy(r["features"])
        return out


class EpochSampler(Sampler):
    """Yields epoch*epoch_len + i so DynamicMixDataset draws fresh mixtures per epoch."""
    def __init__(self, epoch_len: int):
        self.epoch_len, self.epoch = epoch_len, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return self.epoch_len

    def __iter__(self):
        base = self.epoch * self.epoch_len
        return iter(range(base, base + self.epoch_len))


class RenderedDataset(Dataset):
    def __init__(self, root: Path):
        self.items = sorted(p for p in Path(root).glob("*/*.mix.wav") if not p.name.endswith(".twin.mix.wav"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]; stem = p.name[: -len(".mix.wav")]
        x, _ = sf.read(p, dtype="float32"); c, _ = sf.read(p.with_name(stem + ".clean.wav"), dtype="float32")
        meta = json.load(open(p.with_name(stem + ".json")))
        meta.update(bucket=p.parent.name, id=stem)
        twin = p.with_name(stem + ".twin.mix.wav")
        out = {"mix": torch.from_numpy(x.T.copy()), "clean": torch.from_numpy(c), "meta": meta}
        if twin.exists():
            t, _ = sf.read(twin, dtype="float32"); out["twin"] = torch.from_numpy(t.T.copy())
        return out


def collate(batch):
    out = {"mix": torch.stack([b["mix"] for b in batch]), "clean": torch.stack([b["clean"] for b in batch]),
           "meta": [b["meta"] for b in batch]}
    for k in ("twin", "n_hat", "feats"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out
