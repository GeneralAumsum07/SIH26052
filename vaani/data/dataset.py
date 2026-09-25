"""Train = dynamic mixing (a fresh mixture every item, infinite variety).
Val/test = rendered once, frozen, so numbers across runs are comparable.
"""
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import lfilter
from torch.utils.data import Dataset, Sampler

from vaani.data import impulses, manifests
from vaani.data.mixer import MixConfig, mix
from vaani.data.pack import open_pack
from vaani.data.rirs import RirBank
from vaani.data.scenes import ScenePool, sample_scene
from vaani.dsp import pipeline, stft
from vaani.dsp.limiter import Limiter

SR = 16000
BUCKET_SNRS = [-10, -5, 0, 5, 10, 15]
NOISE_CLASSES = ["stationary", "changing", "impulsive", "impulsive+stationary", "clean"]


def _load(path: str, n: int | None, rng, pack=None) -> np.ndarray:
    """Crop one file. `pack` (vaani.data.pack) serves the same bytes from an int16 memmap instead
    of an sf.info open + an sf.read open + a FLAC decode; None falls back to soundfile.

    The rng is drawn exactly once, and only when the file is longer than the crop. The pack must
    not disturb that: a frame count off by one would flip the branch, change the number of draws
    and silently desynchronise the training stream from every run before it. That is why the pack
    index stores each file's exact frame count rather than deriving it from `duration_s`."""
    frames = None if pack is None else pack.frames(path)
    if frames is None:
        frames = sf.info(path).frames
    if n is None or frames <= n:
        x = None if pack is None else pack.read(path)
        if x is None:
            x, _ = sf.read(path, dtype="float32")
        return x
    start = int(rng.integers(0, frames - n + 1))
    x = None if pack is None else pack.read(path, start, n)
    if x is None:
        x, _ = sf.read(path, dtype="float32", start=start, frames=n)
    return x


# --- reference-failure augmentation (spec 6.2; off unless data.ref_corrupt is configured) ---
REF_KINDS = ("dropout", "burst", "delay", "gain", "polarity", "clip", "noise", "lowpass", "leak")
REF_CORRUPT_DEFAULTS = dict(
    p=0.4,                                  # fraction of items given one reference fault (drawn by weights)
    p_absent=0.0,                           # extra fraction with the reference absent the whole crop (validity 0; M9: ~0.15)
    weights=dict(dropout=1.0, burst=1.0, delay=1.0, gain=1.0, polarity=0.5, clip=1.0, noise=1.0, lowpass=1.0, leak=1.5),
    burst_s=(0.1, 1.0), bursts=(1, 3),      # burst dropouts: count and length each
    delay_samples=(1, 48),                  # ADC/channel slip, either sign (48 = 3 ms)
    gain_db=(-18.0, -6.0), gain_up_db=(3.0, 8.0), p_gain_up=0.2,
    clip_frac=(0.05, 0.4),                  # clip level as a fraction of the reference peak
    noise_snr_db=(-10.0, 10.0),             # unrelated noise relative to the reference's own RMS
    lowpass_hz=(300.0, 1500.0), lowpass_db=(-20.0, -6.0),   # strap/obstruction over the reference mic
    leak_db=(-14.0, 1.0), leak_delay=(0, 8),   # talker leakage onto the reference relative to the primary
    leak_near_db=(-4.0, 1.0), p_leak_near=0.5,  # half of it at near-primary level (the web-WAV failure)
)
REF_SEED = 0x5EF   # separate stream: the mixture draws are identical with or without corruption


def ref_corrupt_config(cfg):
    """None/False -> None (off); True or a dict -> REF_CORRUPT_DEFAULTS overridden by it (weights merge per kind)."""
    if not cfg:
        return None
    user = cfg if isinstance(cfg, dict) else {}
    c = {**REF_CORRUPT_DEFAULTS, **user}
    c["weights"] = {**REF_CORRUPT_DEFAULTS["weights"], **user.get("weights", {})}
    unknown = set(c) - set(REF_CORRUPT_DEFAULTS) | set(c["weights"]) - set(REF_KINDS)
    ok_w = all(v >= 0 for v in c["weights"].values()) and sum(c["weights"].values()) > 0
    if unknown or not ok_w or not (0 <= c["p"] and 0 <= c["p_absent"] and c["p"] + c["p_absent"] <= 1):
        raise ValueError(f"bad ref_corrupt config: unknown keys {sorted(unknown)} or p, p_absent outside [0,1]")
    return c


def _shift(x, d):
    """Integer shift, zero fill: d > 0 delays x, d < 0 advances it."""
    y = np.zeros_like(x)
    if d >= 0:
        y[d:] = x[:len(x) - d]
    else:
        y[:d] = x[-d:]
    return y


def apply_ref_fault(rng, mixed, clean, kind, c):
    """One reference fault on a (2, n) mixture. Returns (mixed', per-sample availability, trace dict).
    Availability is the capture path's knowledge: 0 only for dropouts (the channel is absent); every other fault
    leaves a present-but-wrong reference for the reliability map to catch."""
    out = mixed.copy(); ref = out[1]; n = ref.shape[0]; avail = np.ones(n, bool); tr = {"kind": kind}
    if kind == "dropout":
        ref[:] = 0.0; avail[:] = False
    elif kind == "burst":
        spans = []
        for _ in range(int(rng.integers(c["bursts"][0], c["bursts"][1] + 1))):
            ln = int(rng.uniform(*c["burst_s"]) * SR); a = int(rng.integers(0, max(1, n - ln)))
            ref[a:a + ln] = 0.0; avail[a:a + ln] = False; spans.append([a, a + ln])
        tr["spans"] = spans
    elif kind == "delay":
        d = int(rng.integers(c["delay_samples"][0], c["delay_samples"][1] + 1)) * (1 if rng.random() < 0.5 else -1)
        ref[:] = _shift(ref, d); tr["delay"] = d
    elif kind == "gain":
        g = float(rng.uniform(*c["gain_up_db"])) if rng.random() < c["p_gain_up"] else float(rng.uniform(*c["gain_db"]))
        ref *= 10 ** (g / 20); tr["gain_db"] = g
    elif kind == "polarity":
        ref *= -1.0
    elif kind == "clip":
        f = float(rng.uniform(*c["clip_frac"])); lvl = np.abs(ref).max() * f
        ref[:] = np.clip(ref, -lvl, lvl); tr["clip_frac"] = f
    elif kind == "noise":
        snr = float(rng.uniform(*c["noise_snr_db"])); a = float(rng.uniform(0.0, 0.99))
        z = lfilter([1.0], [1.0, -a], rng.standard_normal(n)).astype(np.float32)   # white .. strongly low-tilted
        z *= np.sqrt((ref ** 2).mean() + 1e-12) / (z.std() + 1e-9) * 10 ** (-snr / 20)
        ref += z; tr.update(snr_db=snr, pole=a)
    elif kind == "lowpass":
        fc = float(rng.uniform(*c["lowpass_hz"])); att = float(rng.uniform(*c["lowpass_db"]))
        a = float(np.exp(-2 * np.pi * fc / SR))
        ref[:] = lfilter([1 - a], [1.0, -a], ref).astype(np.float32) * 10 ** (att / 20); tr.update(fc_hz=fc, att_db=att)
    elif kind == "leak":
        near = rng.random() < c["p_leak_near"]
        g = float(rng.uniform(*(c["leak_near_db"] if near else c["leak_db"])))
        d = int(rng.integers(c["leak_delay"][0], c["leak_delay"][1] + 1))
        ref += _shift(clean, d) * 10 ** (g / 20); tr.update(leak_db=g, delay=d)
    else:
        raise ValueError(kind)
    np.clip(ref, -1.0, 1.0, out=ref)   # the reference ADC saturates like any other
    return out.astype(np.float32), avail, tr


def corrupt_reference(rng, mixed, clean, c):
    """Draw whether and how to corrupt this item. Same rng seed -> same trace, whatever model width consumes it."""
    kinds = list(c["weights"]); w = np.asarray([c["weights"][k] for k in kinds], float)
    u = rng.random(); k = int(rng.choice(len(kinds), p=w / w.sum()))   # both drawn always: a fixed-length stream
    if u < c["p_absent"]:
        return apply_ref_fault(rng, mixed, clean, "dropout", c)
    if u >= c["p_absent"] + c["p"] or w[k] <= 0:
        return mixed, np.ones(mixed.shape[1], bool), None
    return apply_ref_fault(rng, mixed, clean, kinds[k], c)


def front_end(mixed, dsp_cfg=None, avail=None):
    """The classical front end without NLMS/features (VaaniFE inputs p/pr/pr_pld): the hop-by-hop limiter and the
    ref_policy zero/ramp, exactly as the first half of pipeline.run. Returns (mix (2,n), frame validity (T,))."""
    dsp_cfg = dsp_cfg or {}
    prim, ref = mixed[0].astype(np.float32), mixed[1].astype(np.float32); n = len(prim)
    if dsp_cfg.get("limiter"):
        lk = dsp_cfg["limiter"]; lim = Limiter(**(lk if isinstance(lk, dict) else {}))
        lp, lr = np.empty_like(prim), np.empty_like(ref)
        for i in range(0, n, stft.HOP):
            lp[i:i + stft.HOP], lr[i:i + stft.HOP] = lim.process_block(prim[i:i + stft.HOP], ref[i:i + stft.HOP])
            lim.engaged = 0
        prim, ref = lp, lr
    av = np.ones(n, bool) if avail is None else np.asarray(avail, bool)
    pol = dsp_cfg.get("ref_policy")
    if pol is not None:
        ref = ref * pipeline.ref_gain(av, pol.get("ramp_frames", pipeline.RAMP_FRAMES))
    return np.stack([prim, ref]).astype(np.float32), pipeline.frame_avail(av, n // stft.HOP + 1)


def load_exclude_groups(path):
    """data.exclude_groups_file (held-out drone/NOISEX/speaker groups of the r8 test set). Accepts a JSON list, or a
    dict whose (nested) values hold the ids; every string is matched against group_id, speaker_id and source_id.
    An absent file -> empty set with a loud warning, since training could then see test-set groups."""
    if not path or not Path(path).exists():
        msg = f"data.exclude_groups_file {path!r} is ABSENT: no held-out groups are excluded from training"
        warnings.warn(msg); print("WARNING: " + msg, flush=True)
        return set()
    j = json.load(open(path, encoding="utf-8"))
    if isinstance(j, dict) and str(j.get("schema", "")).startswith("vaani.heldout_exclude/"):
        # the testset stage's file (configs/data/r8_heldout_exclude.json): its rule is "drop every listed source_id"
        return {s for src in j["sources"].values() for s in src.get("source_ids", [])}
    out = set()

    def walk(v):
        if isinstance(v, str):
            out.add(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
    walk(j)
    return out


def drop_groups(df, ids):
    """Rows whose group_id, speaker_id or source_id is held out."""
    if not ids:
        return df
    hit = np.zeros(len(df), bool)
    for c in ("group_id", "speaker_id", "source_id"):
        if c in df:
            hit |= df[c].astype(str).isin(ids).to_numpy()
    return df[~hit]


class DynamicMixDataset(Dataset):
    def __init__(self, manifest_paths, split, bank_path, cfg: MixConfig, crop_s=4.0, epoch_len=20000, seed=0,
                 with_dsp=False, controller_on=True, dsp_cfg=None, pack_root=None, ref_corrupt=None,
                 fe_inputs=False, exclude_groups_file=None, scene_weights=None):
        df = pd.concat([manifests.read(p) for p in manifest_paths])
        if exclude_groups_file is not None:   # the key present in the config = the r8 recipe: apply it or warn loudly
            df = drop_groups(df, load_exclude_groups(exclude_groups_file))
        df = df[df.split == split]
        self.speech = df[df.kind == "speech"].reset_index(drop=True)
        self.noise = df[df.kind == "noise"].reset_index(drop=True)
        # split once here, not per item: the boolean filter over 8k rows was paid on every __getitem__
        self.cont = self.noise[self.noise.noise_class != "impulsive"].reset_index(drop=True)
        self.impd = self.noise[self.noise.noise_class == "impulsive"].reset_index(drop=True)
        # store the path, not an open RirBank in __init__, so workers open their own handle
        self.bank_path = bank_path
        self._bank = None
        # same reason for the packed corpus: a memmap handle does not survive worker spawn
        self.pack_root = pack_root
        self._pack, self._pack_open = None, False
        self.cfg, self.n, self.epoch_len, self.seed = cfg, int(crop_s * SR), epoch_len, seed
        # with_dsp: run NLMS+features here so the ~150 ms/clip DSP lands in DataLoader workers, not the trainer
        self.with_dsp, self.controller_on, self.dsp_cfg = with_dsp, controller_on, dsp_cfg
        self.ref_corrupt = ref_corrupt_config(ref_corrupt)
        # fe_inputs (VaaniFE without n_hat): limiter/ref_policy front end only, validity always emitted
        self.fe_inputs = bool(fe_inputs) and not with_dsp
        # mixer v2 (M8): scenes draw their noise classes; v1 never builds or touches the pool
        self.scene_pool = ScenePool(self.noise) if cfg.version == 2 else None
        self.scene_weights = scene_weights
        assert len(self.speech) and len(self.cont), "empty manifest split"

    @property
    def bank(self):
        # lazy-open: mmap'd npz handles don't survive pickling across worker spawn
        if self.bank_path and self._bank is None:
            self._bank = RirBank(self.bank_path)
        return self._bank

    @property
    def pack(self):
        if not self._pack_open:
            self._pack, self._pack_open = open_pack(self.pack_root), True
        return self._pack

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        # epoch rides in the index (see EpochSampler): persistent workers never see attribute changes
        epoch, i = divmod(int(idx), self.epoch_len)
        rng = np.random.default_rng([self.seed, epoch, i])
        s = _load(self.speech.path.iloc[int(rng.integers(len(self.speech)))], self.n, rng, self.pack)
        s = np.pad(s, (0, self.n - len(s)))
        if self.scene_pool is not None:
            mixed, clean, meta = self._mix_v2(rng, s)
            return self._finish(epoch, i, mixed, clean, meta)
        # noise draw: continuous class(es) plus optionally an impulsive one
        k = int(rng.integers(1, 3))
        rows = [self.cont.iloc[int(rng.integers(len(self.cont)))] for _ in range(k)]
        noises = [_load(r.path, self.n, rng, self.pack) for r in rows]
        noise_class = "stationary" if all(r.noise_class == "stationary" for r in rows) else "changing"
        imp, onsets = None, []
        if rng.random() < 0.5:
            if len(self.impd) and rng.random() < 0.5:
                imp = _load(self.impd.path.iloc[int(rng.integers(len(self.impd)))], 2 * SR, rng, self.pack)
                # corpus crops start wherever the random offset landed: peak-normalise like generate()
                # does so impulse_peak_db means the same thing, and find the real onsets in the waveform
                imp = imp / (np.abs(imp).max() + 1e-9); onsets = impulses.detect_onsets(imp, SR)
            else:
                # a configured kind list draws here (one rng call either way, so the stream is unchanged when None)
                kind = str(rng.choice(self.cfg.impulse_kinds)) if self.cfg.impulse_kinds else None
                imp, m = impulses.generate(rng, kind=kind); onsets = m["onsets_s"]
            noise_class = "impulsive+stationary" if noise_class == "stationary" else "impulsive"
        mixed, clean, meta = mix(rng, s, noises, imp, onsets, self.bank, self.cfg)
        meta["noise_class"] = "clean" if meta["clean_bucket"] else noise_class
        return self._finish(epoch, i, mixed, clean, meta)

    def _mix_v2(self, rng, s):
        """M8 scene -> rows -> mix_v2. Sources without an eligible row leave the scene, so rows and levels stay aligned."""
        scene = sample_scene(rng, weights=self.scene_weights, crop_s=self.n / SR)
        rows, imp_row = self.scene_pool.draw(rng, scene)
        keep = [k for k, r in enumerate(rows) if r is not None]
        scene["sources"] = [scene["sources"][k] for k in keep]
        noises = [_load(rows[k].path, self.n, rng, self.pack) for k in keep]
        imp, onsets = None, []
        ev = scene.get("event") or {}
        if imp_row is not None:
            imp = _load(imp_row.path, 2 * SR, rng, self.pack)
            imp = imp if imp.ndim == 1 else imp[:, 0]
            imp = imp / (np.abs(imp).max() + 1e-9); onsets = impulses.detect_onsets(imp, SR)
        elif ev.get("fired"):   # the scene's event with no corpus row: the synthetic generator
            kind = str(rng.choice(self.cfg.impulse_kinds)) if self.cfg.impulse_kinds else None
            imp, m = impulses.generate(rng, kind=kind); onsets = m["onsets_s"]
        return mix(rng, s, noises, imp, onsets, self.bank, self.cfg, scene=scene)

    def _finish(self, epoch, i, mixed, clean, meta):
        avail = None
        if self.ref_corrupt is not None:
            mixed, avail, tr = corrupt_reference(np.random.default_rng([self.seed, epoch, i, REF_SEED]), mixed, clean, self.ref_corrupt)
            meta["ref_fault"] = tr
        out = {"mix": torch.from_numpy(mixed), "clean": torch.from_numpy(clean), "meta": meta}
        if self.with_dsp:
            r = pipeline.run(mixed, controller_on=self.controller_on, dsp_cfg=self.dsp_cfg, ref_avail=avail)
            out["n_hat"] = torch.from_numpy(r["n_hat"]); out["feats"] = torch.from_numpy(r["features"])
            out["mix"] = torch.from_numpy(r["mix"])   # the limited signal when the limiter is on
            if avail is not None:   # capture-path label per model frame, with or without a DSP ref_policy
                fa = r["ref_avail"] if "ref_avail" in r else pipeline.frame_avail(avail, r["features"].shape[0])
                out["ref_avail"] = torch.from_numpy(fa)
        elif self.fe_inputs:
            m2, fa = front_end(mixed, self.dsp_cfg, avail)
            out["mix"] = torch.from_numpy(m2); out["ref_avail"] = torch.from_numpy(fa)
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
    for k in ("twin", "n_hat", "feats", "ref_avail"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out
