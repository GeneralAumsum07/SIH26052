"""Train several configs on ONE batch stream.

Why: the step is ~144 ms and roughly 130 ms of it is CPU building the batch (FLAC decode, RIR
convolution, the NLMS + 18-feature front end). The model step is ~5-15 ms of GPU. So producing
the batch once and stepping every model on it turns an N-run sweep into ~one run of dataloading.
Measured shape of the win: the 16-config refiner grid is ~10 h sequential and well under an hour
shared; a 32/64/128/256 epoch sweep is 480 epochs of data sequentially and 256 shared.

    uv run python -m vaani.train_multi configs/retraining/r6_ctl64.yaml ... --tag r6

Correctness, which is the whole reason this file is careful:

* Sharing is only valid when every config would have produced the SAME batches. `assert_shared`
  refuses otherwise. That guard is not paranoia - the r6 corpus arms differ by exactly one
  manifest line, and silently feeding them one stream would make the arms identical and the
  comparison meaningless while still producing plausible numbers.
* Models that legitimately share a stream are compared PAIRWISE (same data, same order), which
  lowers variance and is the right way to read an epoch-budget or architecture effect. Say so
  when reporting.
* Seed variants sharing a stream vary in initialisation only, not in data. Report that as
  init-seed variance, not as run-to-run variance.

Artifacts per run are byte-for-byte what `vaani.train` writes (best.pt, last.pt, run.json), so
evaluation, the matrix and the tier protocols need no changes.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vaani import losses, runtime
from vaani.data import manifests
from vaani.data.dataset import DynamicMixDataset, EpochSampler, collate
from vaani.data.mixer import MixConfig
from vaani.train import (MAX_BAD_STEPS, _abs, _git_hash, _save, build_model, build_param_groups,
                         clip_groups, prepare_batch, validate)
from vaani.training_controls import (cosine_lr_multiplier, make_schedule_config,
                                     should_stop_for_patience, validate_resume_schedule,
                                     verify_checkpoint_hash)

# Fields that decide what a batch contains. Two configs may share a loader only if all of these
# agree; everything else (epochs, optim, model_cfg, loss_cfg, seed-of-initialisation, name) is
# free to differ, which is the point of the sweep.
STREAM_KEYS = ("model", "loss", "controller_on", "batch_size", "seed", "dsp")
STREAM_DATA_KEYS = ("manifests", "bank", "crop_s", "epoch_len", "mix", "pack")


def stream_signature(cfg: dict) -> dict:
    d = cfg.get("data", {})
    return {k: cfg.get(k) for k in STREAM_KEYS} | {f"data.{k}": d.get(k) for k in STREAM_DATA_KEYS}


def assert_shared(cfgs: list[dict]) -> dict:
    """Refuse to share a stream across configs that would have seen different data."""
    sigs = [stream_signature(c) for c in cfgs]
    for cfg, sig in zip(cfgs[1:], sigs[1:]):
        if sig != sigs[0]:
            diff = {k: (sigs[0].get(k), sig.get(k)) for k in set(sigs[0]) | set(sig) if sigs[0].get(k) != sig.get(k)}
            raise SystemExit(
                f"{cfg['name']} does not share a batch stream with {cfgs[0]['name']}: {diff}\n"
                "Configs that differ in their data must be trained separately - sharing a loader here "
                "would feed them identical batches and silently void the comparison.")
    return sigs[0]


class Run:
    """One config's model, optimiser, schedule and bookkeeping, stepped on the shared batch."""

    def __init__(self, cfg, device, steps_per_epoch, runs_dir=None):
        self.cfg, self.device = cfg, device
        verify_checkpoint_hash(_abs(cfg.get("init_from")), cfg.get("init_sha256"))
        self.dir = Path(runs_dir or cfg.get("runs_dir", "runs")) / cfg["name"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(self.dir)
        torch.manual_seed(cfg["seed"])          # initialisation seed; the data seed is shared
        self.model = build_model(cfg["model"], cfg.get("init_from"), cfg.get("model_cfg")).to(device)
        self.params = sum(p.numel() for p in self.model.parameters())
        self.groups = build_param_groups(self.model, cfg["optim"])
        self.opt = torch.optim.AdamW(self.groups, weight_decay=cfg["optim"].get("weight_decay", 1e-4))
        self.total = cfg["max_steps"] if cfg.get("max_steps") else cfg["epochs"] * steps_per_epoch
        self.warm = cfg["optim"].get("warmup", 500)
        self.schedule = make_schedule_config(cfg["epochs"], steps_per_epoch, self.warm, cfg.get("max_steps"))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s, w=self.warm, t=self.total: cosine_lr_multiplier(s, w, t))
        sp = cfg["loss"] == "speech_preservation"
        lc = cfg.get("loss_cfg", {})
        self.loss_fn = losses.SpeechPreservationLoss(**lc) if sp else losses.HybridLoss(**lc)
        self.burst_w = self.loss_fn.burst_weight if sp else 1.0
        self.step, self.best, self.start_epoch, self.history = 0, -1.0, 0, []
        self.bad = self.skipped = 0
        self.clamp_sum, self.clamp_batches = 0.0, 0
        self._resume()

    @property
    def epochs(self):
        return self.cfg["epochs"]

    def active(self, epoch) -> bool:
        if epoch < self.start_epoch or epoch >= self.epochs:
            return False
        stop = self.cfg.get("early_stopping") or {}
        return not should_stop_for_patience(self.history, stop.get("patience"), stop.get("min_delta", 0.))

    def _resume(self):
        last = self.dir / "last.pt"
        if not last.exists():
            return
        ck = torch.load(last, map_location=self.device, weights_only=False)
        saved = ck.get("schedule") or self.schedule
        validate_resume_schedule(saved, self.schedule)
        self.model.load_state_dict(ck["model"])
        self.step, self.best = ck["step"], ck.get("best", self.best)
        self.history = ck.get("history", [])
        self.start_epoch = ck.get("epoch", -1) + 1
        if "optim" in ck:
            self.opt.load_state_dict(ck["optim"]); self.sched.load_state_dict(ck["sched"])
        print(f"  resumed {self.cfg['name']} at epoch {self.start_epoch}, step {self.step}", flush=True)

    def train_step(self, inputs, target, fw, is_clean, use_amp):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = self.model(*inputs)
        loss = self.loss_fn(pred.float(), target, fw, is_clean)
        if self.loss_fn.w_snr:
            self.clamp_sum += float(self.loss_fn.last_snr_clamp_fraction); self.clamp_batches += 1
        if not torch.isfinite(loss):
            self.bad += 1; self.skipped += 1
            if self.bad >= MAX_BAD_STEPS:
                raise RuntimeError(f"{self.cfg['name']}: {self.bad} consecutive non-finite losses at step {self.step}")
            return
        self.bad = 0
        self.opt.zero_grad(set_to_none=True); loss.backward()
        clip_groups(self.groups, self.cfg["optim"].get("clip", 5.0))
        self.opt.step(); self.sched.step(); self.step += 1
        if self.step % 20 == 0:
            self.tb.add_scalar("train/loss", loss.item(), self.step)
            self.tb.add_scalar("train/lr", self.sched.get_last_lr()[0], self.step)

    def end_epoch(self, epoch, vdl):
        v = validate(self.model, vdl, self.cfg, self.device)
        self.tb.add_scalar("val/stoi", v, self.step)
        self.history.append(dict(epoch=epoch, step=self.step, val_stoi=v, lr=self.sched.get_last_lr()[0],
                                 snr_clamp_fraction=self.clamp_sum / max(self.clamp_batches, 1),
                                 val_metrics=getattr(vdl, "_last_val_metrics", {"stoi": v})))
        self.clamp_sum, self.clamp_batches = 0.0, 0
        if v > self.best:
            self.best = v
            _save({"model": self.model.state_dict(), "config": self.cfg, "step": self.step}, self.dir / "best.pt")
        _save({"model": self.model.state_dict(), "config": self.cfg, "step": self.step, "epoch": epoch,
               "best": self.best, "optim": self.opt.state_dict(), "sched": self.sched.state_dict(),
               "schedule": self.schedule, "history": self.history}, self.dir / "last.pt")
        self.write_run_json()
        return v

    def write_run_json(self, **extra):
        cfg, vc = self.cfg, self.cfg.get("val", {})
        ev = _abs(vc.get("eval_root", "data/eval")) / vc.get("split", "val") / "EVALSET_HASH"
        info = dict(name=cfg["name"], config=cfg, git_sha=_git_hash(),
                    config_hash=hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                    manifest_hash=manifests.content_hash(cfg["data"]["manifests"]),
                    evalset_hash=ev.read_text().strip() if ev.exists() else "none",
                    init_from=str(_abs(cfg.get("init_from"))) if cfg.get("init_from") else None,
                    params=self.params, seed=cfg["seed"], torch=torch.__version__, cuda=torch.version.cuda,
                    best_metric="stoi_frozen_val_screen" if vc.get("eval_root") else "stoi_dynamic_val",
                    best_val_stoi=self.best, steps=self.step, skipped_steps=self.skipped,
                    schedule=self.schedule, history=self.history,
                    shared_loader=True,   # the comparison against its siblings is PAIRED
                    **extra)
        json.dump(info, open(self.dir / "run.json", "w"), indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--max-epochs", type=int, default=None, help="cap for a smoke test")
    a = ap.parse_args(argv)

    cfgs = [yaml.safe_load(open(p)) for p in a.configs]
    if len({c["name"] for c in cfgs}) != len(cfgs):
        raise SystemExit("duplicate run names")
    assert_shared(cfgs)
    base = cfgs[0]
    device = torch.device(base.get("device", "cuda"))
    runtime.tune_backends(device)
    np.random.seed(base["seed"])

    d = base["data"]; mixcfg = MixConfig(**d.get("mix", {}))
    dsk = dict(with_dsp=base["model"] == "vaani", controller_on=base["controller_on"],
               dsp_cfg=base.get("dsp"), pack_root=d.get("pack", "data/pack"))
    ds = DynamicMixDataset(d["manifests"], "train", d.get("bank"), mixcfg, d.get("crop_s", 4.0),
                           d.get("epoch_len", 20000), base["seed"], **dsk)
    vds = DynamicMixDataset(d["manifests"], "val", d.get("bank"), mixcfg, d.get("crop_s", 4.0),
                            base.get("val", {}).get("dynamic_items", 200), base["seed"] + 1, **dsk)
    nw = runtime.resolve_workers(base.get("num_workers", "auto"))
    lk = runtime.loader_kwargs(nw, device)
    sampler = EpochSampler(len(ds))
    dl = DataLoader(ds, base["batch_size"], sampler=sampler, collate_fn=collate, **lk)
    vdl = DataLoader(vds, base["batch_size"], collate_fn=collate, **lk)

    runs = [Run(c, device, len(dl), a.runs_dir) for c in cfgs]
    use_amp = bool(base.get("amp", True)) and device.type == "cuda"
    horizon = max(r.epochs for r in runs)
    if a.max_epochs:
        horizon = min(horizon, a.max_epochs)
    t0 = time.time()
    print(f"runtime: {runtime.describe()} num_workers={nw}")
    print(f"shared stream: {len(runs)} runs, horizon {horizon} epochs, {len(dl)} steps/epoch")
    for r in runs:
        print(f"  {r.cfg['name']}: {r.epochs} epochs, {r.params} params, from {r.start_epoch}")

    for epoch in range(horizon):
        live = [r for r in runs if r.active(epoch)]
        if not live:
            break
        sampler.set_epoch(epoch)
        for batch in dl:
            # the expensive part, paid once for every model in `live`
            inputs, target, fw, is_clean = prepare_batch(batch, base["model"], device, live[0].burst_w)
            for r in live:
                r.train_step(inputs, target, fw, is_clean, use_amp)
        line = []
        for r in live:
            line.append(f"{r.cfg['name']} {r.end_epoch(epoch, vdl):.4f}")
        print(f"epoch {epoch} ({time.time() - t0:.0f}s) " + "  ".join(line), flush=True)
        for r in runs:
            if r.active(epoch) and epoch + 1 >= r.epochs:
                print(f"  {r.cfg['name']} finished its {r.epochs}-epoch budget", flush=True)

    for r in runs:
        df_norm = float(sum(p.detach().norm() ** 2 for n, p in r.model.named_parameters()
                            if n.startswith("df.")) ** 0.5)
        r.write_run_json(end=time.time(), wall_s=time.time() - t0, df_norm=df_norm)
        r.tb.close()
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
