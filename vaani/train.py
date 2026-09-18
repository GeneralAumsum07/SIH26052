"""Config-driven trainer. One YAML == one ablation row.

Inputs per model:
  gtcrn : spec (B,257,T,2) of the primary channel only
  vaani : spec6 (B,257,T,6) [prim, ref, n_hat] + feats (B,T,18)
The DSP pipeline (NLMS + features + controller) runs on CPU inside the
DataLoader workers via DynamicMixDataset(with_dsp=True); prepare_batch only
moves tensors to the device and takes STFTs there.
"""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path

import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pystoi import stoi

from vaani import losses
from vaani.data import manifests
from vaani.data.dataset import DynamicMixDataset, EpochSampler, collate
from vaani.data.mixer import MixConfig
from vaani.dsp import stft
from vaani.models.gtcrn import GTCRN
from vaani.models.vaani_net import VaaniNet

SR = 16000
ROOT = Path(__file__).resolve().parents[1]  # repo root, so configs work from any cwd
MAX_BAD_STEPS = 20


def _abs(p):
    return None if p is None else (Path(p) if Path(p).is_absolute() else ROOT / p)


def build_model(name, init_from=None):
    init_from = _abs(init_from)
    if name == "gtcrn":
        m = GTCRN()
        if init_from:
            m.load_state_dict(torch.load(init_from, map_location="cpu", weights_only=True)["model"])
        return m
    if name == "vaani":
        return VaaniNet.from_pretrained_gtcrn(init_from) if init_from else VaaniNet()
    raise ValueError(name)


def frame_weights_from_meta(metas, n_frames, burst_weight=3.0, half_window_s=0.15):
    """(B,T) loss weights: burst_weight on frames within +-half_window_s of an impulse onset."""
    w = torch.ones(len(metas), n_frames)
    hw = int(half_window_s * SR / stft.HOP)
    for b, m in enumerate(metas):
        for on in m.get("impulse_onsets_s", []):
            k = int(on * SR / stft.HOP)
            w[b, max(0, k - hw): k + hw] = burst_weight
    return w


def prepare_batch(batch, model_name, device, burst_weight=1.0):
    """Batch (CPU, from collate) -> (model inputs, target spec, frame weights, is_clean) on device."""
    mix, clean, metas = batch["mix"].to(device), batch["clean"].to(device), batch["meta"]
    target = stft.stft(clean)  # STFTs on device: cheaper than CPU + transfer of the wider spec
    fw = frame_weights_from_meta(metas, target.shape[2], burst_weight)
    is_clean = torch.tensor([bool(m.get("clean_bucket", False)) for m in metas])
    if model_name == "gtcrn":
        inputs = (stft.stft(mix[:, 0]),)
    else:
        # n_hat/feats were computed in the dataset workers, which own controller_on
        n_hat = batch["n_hat"].to(device)
        spec6 = torch.cat([stft.stft(mix[:, 0]), stft.stft(mix[:, 1]), stft.stft(n_hat)], dim=-1)
        inputs = (spec6, batch["feats"].to(device))
    return inputs, target, fw.to(device), is_clean.to(device)


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        return "nogit"


def _save(state, path):
    tmp = path.with_suffix(".tmp")  # write-then-rename so a crash never leaves a truncated .pt
    torch.save(state, tmp); os.replace(tmp, path)


@torch.no_grad()
def validate(model, dl, cfg, device):
    model.eval(); scores = []
    for batch in dl:
        inputs, target, _, _ = prepare_batch(batch, cfg["model"], device)
        pred = model(*inputs).float()
        y = stft.istft(pred, length=batch["clean"].shape[-1]).cpu().numpy()
        for b in range(y.shape[0]):
            scores.append(stoi(batch["clean"][b].numpy(), y[b], SR, extended=False))
    model.train(); return float(np.mean(scores))


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda"))
    run_dir = Path(cfg.get("runs_dir", "runs")) / cfg["name"]; run_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(run_dir)
    t_start = time.time()

    d = cfg["data"]; mixcfg = MixConfig(**d.get("mix", {}))
    with_dsp = cfg["model"] == "vaani"  # gtcrn never needs n_hat/feats, skip the 150 ms/clip
    dsk = dict(with_dsp=with_dsp, controller_on=cfg["controller_on"])
    ds = DynamicMixDataset(d["manifests"], "train", d.get("bank"), mixcfg, d.get("crop_s", 4.0),
                           d.get("epoch_len", 20000), cfg["seed"], **dsk)
    vds = DynamicMixDataset(d["manifests"], "val", d.get("bank"), mixcfg, d.get("crop_s", 4.0),
                            cfg.get("val", {}).get("dynamic_items", 200), cfg["seed"] + 1, **dsk)
    nw = cfg.get("num_workers", 4)
    # Windows spawns workers: persistent_workers avoids re-importing numba/JIT every epoch;
    # the epoch therefore travels in the sampler's indices, not in dataset attributes
    sampler = EpochSampler(len(ds))
    dl = DataLoader(ds, cfg["batch_size"], sampler=sampler, collate_fn=collate, num_workers=nw, persistent_workers=nw > 0)
    vdl = DataLoader(vds, cfg["batch_size"], collate_fn=collate, num_workers=nw, persistent_workers=nw > 0)

    model = build_model(cfg["model"], cfg.get("init_from")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # only the FiLM projection gets lr_new; the zero-init conv slices share tensors with
    # pretrained weights so they train at the base lr
    new_params = [p for n, p in model.named_parameters() if "film" in n]
    old_params = [p for n, p in model.named_parameters() if "film" not in n]
    groups = [{"params": old_params, "lr": cfg["optim"]["lr"]}]
    if new_params:
        groups.append({"params": new_params, "lr": cfg["optim"].get("lr_new", cfg["optim"]["lr"])})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = cfg["max_steps"] if cfg.get("max_steps") else cfg["epochs"] * len(dl)
    warm = cfg["optim"].get("warmup", 500)
    # float(): a numpy scalar in the scheduler state would break weights_only resume
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: float(min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total))))
    sp = cfg["loss"] == "speech_preservation"
    loss_fn = losses.SpeechPreservationLoss() if sp else losses.HybridLoss()
    burst_w = loss_fn.burst_weight if sp else 1.0
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    step, best, start_epoch = 0, -1.0, 0
    last = run_dir / "last.pt"
    if cfg.get("resume", True) and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"]); step, best = ck["step"], ck.get("best", best)
        start_epoch = ck.get("epoch", -1) + 1
        if "optim" in ck:
            opt.load_state_dict(ck["optim"]); sched.load_state_dict(ck["sched"])
        print(f"resumed {last} at step {step}, epoch {start_epoch}")

    evalset_hash = ROOT / "data/eval/val/EVALSET_HASH"
    run_info = dict(name=cfg["name"], config=cfg, git_sha=_git_hash(),
                    config_hash=hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                    manifest_hash=manifests.content_hash(d["manifests"]),
                    evalset_hash=evalset_hash.read_text().strip() if evalset_hash.exists() else "none",
                    init_from=str(_abs(cfg.get("init_from"))) if cfg.get("init_from") else None,
                    params=n_params, seed=cfg["seed"], torch=torch.__version__, cuda=torch.version.cuda,
                    amp=use_amp, start=t_start, best_metric="stoi_dynamic_val", best_val_stoi=best,
                    steps=step, wall_s=0.0, skipped_steps=0)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)

    bad, skipped = 0, 0
    for epoch in range(start_epoch, cfg["epochs"]):
        sampler.set_epoch(epoch)
        for batch in dl:
            inputs, target, fw, is_clean = prepare_batch(batch, cfg["model"], device, burst_w)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(*inputs)
            loss = loss_fn(pred.float(), target, fw, is_clean)  # loss/iSTFT stay fp32
            if not torch.isfinite(loss):
                bad += 1; skipped += 1
                if bad >= MAX_BAD_STEPS:
                    raise RuntimeError(f"{bad} consecutive non-finite losses at step {step}")
                continue
            bad = 0
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"].get("clip", 5.0))
            opt.step(); sched.step(); step += 1
            if step % 20 == 0:
                tb.add_scalar("train/loss", loss.item(), step); tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        v = validate(model, vdl, cfg, device); tb.add_scalar("val/stoi", v, step)
        if v > best:
            best = v; _save({"model": model.state_dict(), "config": cfg, "step": step}, run_dir / "best.pt")
        _save({"model": model.state_dict(), "config": cfg, "step": step, "epoch": epoch, "best": best,
               "optim": opt.state_dict(), "sched": sched.state_dict()}, last)
        print(f"epoch {epoch} step {step} val_stoi {v:.4f} best {best:.4f} skipped {skipped}")
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break
    run_info.update(end=time.time(), wall_s=time.time() - t_start, best_val_stoi=best, steps=step, skipped_steps=skipped)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)
    tb.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); main(ap.parse_args().config)
