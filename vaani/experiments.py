"""Reproducible experiment recipes and read-only preflight for the next round.

No function here trains, steps an optimizer, writes a checkpoint or unpacks a RIR
bank. Launching is a separate explicit action in scripts/retraining.py.
"""
import copy
import hashlib
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

from vaani.data.mixer import MixConfig, sample_snr
from vaani.export import _stream_twin, layer_macs
from vaani.models.cascade import FrozenCascade
from vaani.models.vaani_net import VaaniNet
from vaani.train import build_model
from vaani.training_controls import make_schedule_config
from vaani.losses import build_loss
from vaani.data import manifests
from vaani.data.dataset import RenderedDataset


@lru_cache(maxsize=64)
def _manifest_sources(path, modified_ns):
    """Reuse a read-only source check across a pack; mtime invalidates edits."""
    table = manifests.read(path)
    for split in ("train", "val"):
        rows = table[table.split == split]
        # Individual corpora need not contain both kinds; combined manifests do.
        for source in rows.path.unique():
            if not Path(source).is_file():
                raise FileNotFoundError(f"Manifest source missing: {source} ({path})")
    return table


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recipes(base, anchor="runs/vaani_full_r4_ctl/best.pt"):
    """Standalone YAMLs, no inheritance resolution left for tomorrow's machine."""
    control = copy.deepcopy(base)
    control.update(name="r5_continue128", init_from=anchor, epochs=128, resume=True)
    control["val"] = dict(eval_root="data/eval_r2", split="val", dynamic_items=200)
    control["model_cfg"].update(channels=16, noise_floor=False)
    control["early_stopping"] = dict(patience=None, min_delta=0.)
    if Path(anchor).is_file():
        control["init_sha256"] = sha256(anchor)
    result = {control["name"]: control}
    variants = {
        "r5_noise_floor": ("model_cfg", "noise_floor", True),
        "r5_wsnr08": ("loss_cfg", "w_snr", .8),
        "r5_wsnr12": ("loss_cfg", "w_snr", 1.2),
        "r5_snr_cap20": ("loss_cfg", "snr_max_db", 20.),
    }
    for name, (section, key, value) in variants.items():
        cfg = copy.deepcopy(control); cfg["name"] = name; cfg[section][key] = value
        result[name] = cfg
    for sampling in ("triangular_low", "stratified"):
        cfg = copy.deepcopy(control); cfg["name"] = f"r5_snr_{sampling}"
        cfg["data"]["mix"].update(snr_sampling=sampling)
        if sampling == "stratified":
            cfg["data"]["mix"].update(snr_bins=[-10, -5, 0, 5, 15], snr_weights=[.4, .3, .2, .1])
        result[cfg["name"]] = cfg
    for channels in (8, 16, 32):
        for seed in (0, 1):
            cfg = copy.deepcopy(control)
            cfg.update(name=f"r5_width{channels}_s{seed}", seed=seed, init_from=None)
            cfg.pop("init_sha256", None)
            cfg["model_cfg"]["channels"] = channels
            result[cfg["name"]] = cfg
    for hidden in (16, 24, 32, 48):
        for past in (2, 4):
            for seed in (4606, 4607):
                name = f"r5_refiner_h{hidden}_p{past}_s{seed}"
                cfg = dict(name=name, model="vaani_cascade", base_checkpoint=anchor,
                           seed=seed, epochs=32, batch_size=32, num_workers=6, amp=True,
                           resume=True, early_screen_stop=False,
                           refiner_cfg=dict(hidden=hidden, past=past, scale=.25),
                           optim=dict(lr=.001, warmup=200, clip=1., weight_decay=.0001),
                           loss_cfg=copy.deepcopy(control["loss_cfg"]),
                           val=dict(eval_root="data/eval_r2", split="val"))
                if Path(anchor).is_file():
                    cfg["base_checkpoint_sha256"] = sha256(anchor)
                result[name] = cfg
    return result


def preflight(cfg, check_files=True):
    """Check a recipe, construct its model and count arithmetic, without training."""
    name = cfg["name"]
    if not name or Path(name).name != name or any(c in name for c in "/\\:") or name in {".", ".."}:
        raise ValueError("name must be a single run directory name")
    if cfg.get("val", {}).get("split", "val") != "val":
        raise ValueError("training selection requires val")
    if cfg["epochs"] <= 0 or cfg["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    identities = {}
    for key, hash_key in (("init_from", "init_sha256"), ("base_checkpoint", "base_checkpoint_sha256")):
        if cfg.get(key):
            p = Path(cfg[key])
            if check_files and not p.is_file():
                raise FileNotFoundError(p)
            if p.is_file():
                digest = sha256(p); identities[key] = digest
                if cfg.get(hash_key) and cfg[hash_key] != digest:
                    raise ValueError(f"{key} hash differs from the declared recipe")
    if cfg["model"] == "vaani_cascade":
        model, ccfg = FrozenCascade.from_first_stage(cfg["base_checkpoint"], cfg.get("refiner_cfg"))
        mc, data = ccfg.get("model_cfg", {}), ccfg["data"]
    elif cfg["model"] == "vaani":
        mc, data = cfg.get("model_cfg", {}), cfg["data"]
        model = build_model("vaani", cfg.get("init_from"), mc)
    else:
        raise ValueError("retraining pack supports vaani and vaani_cascade")
    mixcfg = MixConfig(**data.get("mix", {}))
    sample_snr(np.random.default_rng(0), mixcfg)
    build_loss(cfg.get("loss", "hybrid"), cfg.get("loss_cfg"))
    for key, value in cfg.get("loss_cfg", {}).items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"loss_cfg.{key} must be finite")
    for key, value in cfg["optim"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"optim.{key} must be finite and nonnegative")
    if check_files:
        required = list(data["manifests"]) + ([data["bank"]] if data.get("bank") else [])
        missing = [p for p in required if not Path(p).is_file()]
        val_root = Path(cfg["val"].get("eval_root", "data/eval_r2")) / "val"
        if not any(val_root.glob("*/*.mix.wav")):
            missing.append(str(val_root))
        if missing:
            raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
        tables = [_manifest_sources(str(Path(p).resolve()), Path(p).stat().st_mtime_ns) for p in data["manifests"]]
        for split in ("train", "val"):
            speech = sum(int(((t.split == split) & (t.kind == "speech")).sum()) for t in tables)
            noise = sum(int(((t.split == split) & (t.kind == "noise") & (t.noise_class != "impulsive")).sum()) for t in tables)
            if not speech or not noise:
                raise ValueError(f"Empty speech/continuous-noise sources in {split}")
        for mix_path in RenderedDataset(val_root).items:
            for suffix in (".clean.wav", ".json"):
                paired = mix_path.with_name(mix_path.name.replace(".mix.wav", suffix))
                if not paired.is_file():
                    raise FileNotFoundError(paired)
    model.eval(); stream, caches, names, _ = _stream_twin(model, mc)
    costs = layer_macs(stream, (torch.zeros(1, 257, 1, 6), torch.zeros(1, 1, 18), *caches))
    steps = (data.get("epoch_len", 20000) + cfg["batch_size"] - 1) // cfg["batch_size"]
    schedule = make_schedule_config(cfg["epochs"], steps, cfg["optim"].get("warmup", 200), cfg.get("max_steps"))
    return dict(name=name, model=cfg["model"], checkpoints=identities, schedule=schedule,
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                matrix_macs_per_frame=sum(costs.values()), matrix_mmacs_per_second=sum(costs.values()) * 62.5 / 1e6,
                cache_shapes={n: list(c.shape) for n, c in zip(names[2:], caches)},
                existing_run=(Path(cfg.get("runs_dir", "runs")) / name).exists(),
                checked_files=check_files, training_launched=False)
