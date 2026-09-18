"""Config-driven trainer. One YAML == one ablation row.

Inputs per model:
  gtcrn : spec (B,257,T,2) of the primary channel only
  vaani : spec6 (B,257,T,6) [prim, ref, n_hat] + feats (B,T,18)
The DSP pipeline (NLMS + features + controller) runs on CPU inside the
DataLoader workers via DynamicMixDataset(with_dsp=True); prepare_batch only
does STFTs and moves tensors to the device.
"""
import argparse, hashlib, json, subprocess, time
from pathlib import Path

import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pystoi import stoi

from vaani import losses
from vaani.data import manifests
from vaani.data.dataset import DynamicMixDataset, collate
from vaani.data.mixer import MixConfig
from vaani.dsp import stft
from vaani.models.gtcrn import GTCRN
from vaani.models.vaani_net import VaaniNet

SR = 16000


def build_model(name, init_from=None):
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


def prepare_batch(batch, model_name, controller_on, device, burst_weight=1.0):
    """Batch (CPU, from collate) -> (model inputs, target spec, frame weights, is_clean) on device."""
    mix, clean, metas = batch["mix"], batch["clean"], batch["meta"]
    target = stft.stft(clean)
    fw = frame_weights_from_meta(metas, target.shape[2], burst_weight)
    is_clean = torch.tensor([bool(m.get("clean_bucket", False)) for m in metas])
    if model_name == "gtcrn":
        inputs = (stft.stft(mix[:, 0]).to(device),)
    else:
        # n_hat/feats were computed in the dataset workers; controller_on is baked in there
        spec6 = torch.cat([stft.stft(mix[:, 0]), stft.stft(mix[:, 1]), stft.stft(batch["n_hat"])], dim=-1)
        inputs = (spec6.to(device), batch["feats"].to(device))
    return inputs, target.to(device), fw.to(device), is_clean.to(device)


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def _save(model, cfg, step, path):
    torch.save({"model": model.state_dict(), "config": cfg, "step": step}, path)


@torch.no_grad()
def validate(model, dl, cfg, device):
    model.eval(); scores = []
    for batch in dl:
        inputs, target, _, _ = prepare_batch(batch, cfg["model"], cfg["controller_on"], device)
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
    # Windows spawns workers: persistent_workers avoids re-importing numba/JIT every epoch
    dl = DataLoader(ds, cfg["batch_size"], shuffle=False, collate_fn=collate, num_workers=nw, persistent_workers=nw > 0)
    vdl = DataLoader(vds, cfg["batch_size"], collate_fn=collate, num_workers=nw, persistent_workers=nw > 0)

    model = build_model(cfg["model"], cfg.get("init_from")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # new layers (zero-init FiLM) get a higher LR than pretrained ones
    new_params = [p for n, p in model.named_parameters() if "film" in n]
    old_params = [p for n, p in model.named_parameters() if "film" not in n]
    groups = [{"params": old_params, "lr": cfg["optim"]["lr"]}]
    if new_params:
        groups.append({"params": new_params, "lr": cfg["optim"].get("lr_new", cfg["optim"]["lr"])})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = cfg["max_steps"] if cfg.get("max_steps") else cfg["epochs"] * len(dl)
    warm = cfg["optim"].get("warmup", 500)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    sp = cfg["loss"] == "speech_preservation"
    loss_fn = losses.SpeechPreservationLoss() if sp else losses.HybridLoss()
    burst_w = 3.0 if sp else 1.0
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    evalset_hash = Path("data/eval/val/EVALSET_HASH")
    run_info = dict(name=cfg["name"], config=cfg, git_sha=_git_hash(),
                    config_hash=hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                    manifest_hash=manifests.content_hash(d["manifests"]),
                    evalset_hash=evalset_hash.read_text().strip() if evalset_hash.exists() else "none",
                    params=n_params, seed=cfg["seed"], torch=torch.__version__, cuda=torch.version.cuda,
                    amp=use_amp, start=t_start, best_val_stoi=None, steps=0, wall_s=0.0)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)

    step, best = 0, -1.0
    for epoch in range(cfg["epochs"]):
        ds.set_epoch(epoch)
        for batch in dl:
            inputs, target, fw, is_clean = prepare_batch(batch, cfg["model"], cfg["controller_on"], device, burst_w)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(*inputs)
            loss = loss_fn(pred.float(), target, fw, is_clean)  # loss/iSTFT stay fp32
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"].get("clip", 5.0))
            opt.step(); sched.step(); step += 1
            if step % 20 == 0:
                tb.add_scalar("train/loss", loss.item(), step); tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        v = validate(model, vdl, cfg, device); tb.add_scalar("val/stoi", v, step)
        _save(model, cfg, step, run_dir / "last.pt")
        if v > best:
            best = v; _save(model, cfg, step, run_dir / "best.pt")
        print(f"epoch {epoch} step {step} val_stoi {v:.4f} best {best:.4f}")
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break
    run_info.update(end=time.time(), wall_s=time.time() - t_start, best_val_stoi=best, steps=step)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)
    tb.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); main(ap.parse_args().config)
