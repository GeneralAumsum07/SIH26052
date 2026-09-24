"""Train the Tier 4.6 residual refiner on top of a frozen first stage (plan §8).

Data comes from the anchor checkpoint's own embedded config: same manifests, mixer, RIR bank, controller and DSP
settings, train split only. Stage one runs under no_grad in eval mode inside FrozenCascade; only refiner parameters
reach the optimiser. Per-epoch model selection is a deterministic validation screen (first four sorted items of
every bucket of data/eval_r2/val, anchor scored once on the same items): eligible epochs keep dSTOI >= -0.003 and the
winner maximises dSNR_out + 5 * dPESQ, earliest epoch on ties. Nothing here ever touches the test split.
"""
import hashlib, json, time
from pathlib import Path

import numpy as np, torch, yaml
from torch.utils.data import DataLoader

from vaani import losses, metrics, runtime
from vaani.data.dataset import DynamicMixDataset, EpochSampler, RenderedDataset, collate
from vaani.data.mixer import MixConfig
from vaani.dsp import pipeline, stft
from vaani.models.cascade import FrozenCascade
from vaani.train import _save, prepare_batch
from vaani.training_controls import cosine_lr_multiplier, verify_checkpoint_hash

LOSS_CFG = dict(w_complex=50, w_mag=50, p=0.5, w_snr=0.2, snr_max_db=30)
STOI_TOL, PESQ_W, EARLY_EPOCH, EARLY_SNR, EARLY_PESQ = 0.003, 5.0, 2, 0.1, 0.01
SCREEN_PER_BUCKET = 4
SCREEN_WORKERS = 32   # CPU processes for the screen's DSP and metrics; $VAANI_SCREEN_WORKERS caps it on a shared box


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _cfg_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def build_train_data(first_cfg, seed, batch_size, num_workers, split="train"):
    """The anchor's exact data recipe. `split` is exposed only so a test can prove it is 'train'."""
    assert split == "train", "the refiner trains on train sources only"
    d = first_cfg["data"]
    # pack_root as vaani.train passes it: without it this trainer decodes FLAC per item and ignores the
    # packed int16 memmap the bootstrap built. The pack is verified bit-identical to the FLACs, so this
    # is a speed difference only - but the refiner reads the same corpus as its backbone, and there is
    # no reason for the second stage to read it the slow way.
    ds = DynamicMixDataset(d["manifests"], split, d.get("bank"), MixConfig(**d.get("mix", {})), d.get("crop_s", 4.0),
                           d.get("epoch_len", 20000), seed, with_dsp=True, controller_on=first_cfg["controller_on"],
                           dsp_cfg=first_cfg.get("dsp"), pack_root=d.get("pack", "data/pack"), ref_corrupt=d.get("ref_corrupt"))
    sampler = EpochSampler(len(ds))
    # num_workers may be "auto" - vaani.train resolves it through runtime, and this trainer has to do
    # the same or a config that works for the backbone dies here with a TypeError comparing str to int.
    nw = runtime.resolve_workers(num_workers)
    return ds, sampler, DataLoader(ds, batch_size, sampler=sampler, collate_fn=collate, num_workers=nw, persistent_workers=nw > 0)


def screen_items(eval_root, split):
    """First SCREEN_PER_BUCKET items of each bucket in sorted order: deterministic, small, every bucket represented."""
    if split != "val":
        raise ValueError("model selection must use val, never test")
    ds = RenderedDataset(Path(eval_root) / split); per = {}
    for i, p in enumerate(ds.items):
        per.setdefault(p.parent.name, [])
        if len(per[p.parent.name]) < SCREEN_PER_BUCKET: per[p.parent.name].append(i)
    if not per:
        raise ValueError(f"No validation items in {eval_root}/{split}")
    return ds, [i for b in sorted(per) for i in per[b]]


_pool = None
_screen_cache = {}


def _screen_pool():
    """One persistent spawn pool for the CPU halves of the screen (DSP front end, PESQ/STOI); children never touch
    CUDA. Thread caps keep 32 workers inside the box's pid cgroup (see rirs.build_bank)."""
    global _pool
    if _pool is None:
        import os
        from multiprocessing import get_context
        caps = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
        saved = {k: os.environ.get(k) for k in caps}; os.environ.update(caps)
        try: _pool = get_context("spawn").Pool(int(os.environ.get("VAANI_SCREEN_WORKERS", SCREEN_WORKERS)))
        finally:
            for k, v in saved.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
    return _pool


def _dsp_item(args):
    mix, clean, controller_on, dsp_cfg = args
    r = pipeline.run(mix, controller_on=controller_on, dsp_cfg=dsp_cfg)
    return {"clean": clean, "mix": r["mix"], "n_hat": r["n_hat"], "features": r["features"], "n": mix.shape[1]}


def _metric_item(args):
    clean, y = args
    return (metrics.snr_db(clean, y), metrics.stoi(clean, y), metrics.pesq_wb(clean, y))


@torch.no_grad()
def score_items(model, ds, idx, first_cfg, device):
    """Per-item (snr_out, stoi, pesq_wb) through the anchor's DSP pipeline, exactly as vaani.eval runs a checkpoint.
    The DSP front end does not depend on the model, so it is computed once per (dataset, idx) and reused every
    epoch; the model forward runs here on `device`; PESQ/STOI fan out to the pool. Same numbers as the serial loop."""
    key = (ds, tuple(idx), _cfg_hash({"dsp": first_cfg.get("dsp"), "controller_on": first_cfg["controller_on"]}))
    if key not in _screen_cache:
        items = [ds[i] for i in idx]
        _screen_cache[key] = _screen_pool().map(_dsp_item, [(it["mix"].numpy(), it["clean"].numpy(), first_cfg["controller_on"],
                                                             first_cfg.get("dsp")) for it in items], chunksize=4)
    was_training = model.training; model.eval(); ys = []
    for r in _screen_cache[key]:
        x = torch.from_numpy(r["mix"])[None].to(device)
        spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None].to(device))], -1)
        ys.append(stft.istft(model(spec6, torch.from_numpy(r["features"])[None].to(device)).float(), length=r["n"])[0].cpu().numpy())
    model.train(was_training)
    out = _screen_pool().map(_metric_item, [(r["clean"], y) for r, y in zip(_screen_cache[key], ys)], chunksize=4)
    return np.asarray(out, float)


def select_epoch(history):
    """history: list of (epoch, d_snr, d_stoi, d_pesq) means. Returns the winning epoch or None (no eligible epoch)."""
    ok = [(e, ds, dst, dp) for e, ds, dst, dp in history if dst >= -STOI_TOL - 1e-9]
    if not ok: return None
    ok.sort(key=lambda t: (-(t[1] + PESQ_W * t[3]), t[0]))   # highest score, earliest epoch on ties
    return ok[0][0]


def should_stop_early(history):
    """After EARLY_EPOCH epochs, stop unless some eligible epoch reached +EARLY_SNR dB or +EARLY_PESQ."""
    if len(history) < EARLY_EPOCH: return False
    return not any(dst >= -STOI_TOL - 1e-9 and (ds >= EARLY_SNR or dp >= EARLY_PESQ) for _, ds, dst, dp in history)


def main(config_path, max_steps=None):
    cfg = yaml.safe_load(open(config_path)); torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    verify_checkpoint_hash(cfg["base_checkpoint"], cfg.get("base_checkpoint_sha256"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(cfg.get("runs_dir", "runs")) / cfg["name"]; run_dir.mkdir(parents=True, exist_ok=True)
    model, ccfg = FrozenCascade.from_first_stage(cfg["base_checkpoint"], refiner_cfg=cfg.get("refiner_cfg")); model.to(device).train()
    first_cfg = ccfg["first_stage"]["config"]; anchor_sha = ccfg["first_stage"]["sha256"]
    ccfg["refiner_train"] = cfg   # the cascade checkpoint records how its refiner was trained
    ds, sampler, dl = build_train_data(first_cfg, cfg["seed"], cfg["batch_size"], cfg.get("num_workers", 6))
    params = [p for p in model.parameters() if p.requires_grad]
    assert all(n.startswith("refiner.") for n, p in model.named_parameters() if p.requires_grad)
    o = cfg["optim"]; opt = torch.optim.AdamW(params, lr=o["lr"], weight_decay=o.get("weight_decay", 1e-4))
    total = max_steps or cfg["epochs"] * len(dl); warm = o.get("warmup", 200)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr_multiplier(s, warm, total))
    loss_fn = losses.build_loss("hybrid", {**LOSS_CFG, **cfg.get("loss_cfg", {})})
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    vds, vidx = screen_items(cfg["val"]["eval_root"], cfg["val"]["split"])
    anchor_scores = score_items(model.first, vds, vidx, first_cfg, device)   # zero-init cascade == first stage; score the stage itself
    step, start_epoch, history = 0, 0, []
    last = run_dir / "last.pt"
    if cfg.get("resume", True) and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        if ck.get("anchor_sha256") != anchor_sha or ck.get("train_cfg_hash") != _cfg_hash(cfg):
            raise RuntimeError(f"{last} was trained against another anchor or config; refusing to resume")
        if ck.get("schedule_total_steps", total) != total:
            raise RuntimeError("Refiner resume changed its schedule budget; start a new run")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optim"]); sched.load_state_dict(ck["sched"])
        step, start_epoch, history = ck["step"], ck["epoch"] + 1, [tuple(h) for h in ck["history"]]
        print(f"resumed {last} at step {step}, epoch {start_epoch}")

    t0 = time.time(); info = dict(name=cfg["name"], config=cfg, anchor_sha256=anchor_sha, start=t0, screen_items=len(vidx),
                                  anchor_screen=anchor_scores.mean(0).tolist(), params=sum(p.numel() for p in params))
    clamp_history = ck.get("clamp_history", []) if cfg.get("resume", True) and last.exists() else []
    bad_steps = 0
    for epoch in range(start_epoch, cfg["epochs"]):
        sampler.set_epoch(epoch); done = False
        clamp_sum, clamp_batches = 0., 0
        for batch in dl:
            inputs, target, _, is_clean = prepare_batch(batch, "vaani", device)   # (spec6, feats[, ref_avail])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                z = model(*inputs)
            z = z.float(); loss = loss_fn(z, target)
            if is_clean.any():   # clean items are supervised to the clean target, never to the noisy input
                loss = loss + (z[is_clean] - target[is_clean]).abs().mean()
            if not torch.isfinite(loss):
                bad_steps += 1
                if bad_steps >= 20:
                    raise RuntimeError("20 consecutive non-finite refiner losses")
                continue
            bad_steps = 0
            clamp_sum += float(loss_fn.last_snr_clamp_fraction); clamp_batches += 1
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, o.get("clip", 1.0)); opt.step(); sched.step(); step += 1
            if max_steps and step >= max_steps: done = True; break
        s = score_items(model, vds, vidx, first_cfg, device) - anchor_scores
        history.append((epoch, float(s[:, 0].mean()), float(s[:, 1].mean()), float(s[:, 2].mean())))
        clamp_history.append(dict(epoch=epoch, snr_clamp_fraction=clamp_sum / max(clamp_batches, 1)))
        print(f"epoch {epoch} step {step} dSNR {history[-1][1]:+.3f} dSTOI {history[-1][2]:+.4f} dPESQ {history[-1][3]:+.3f}")
        _save({"model": model.state_dict(), "config": ccfg, "step": step, "epoch": epoch, "history": history,
               "anchor_sha256": anchor_sha, "train_cfg_hash": _cfg_hash(cfg), "optim": opt.state_dict(), "sched": sched.state_dict(),
               "schedule_total_steps": total, "clamp_history": clamp_history}, last)
        _save({"model": model.state_dict(), "config": ccfg, "step": step, "epoch": epoch, "anchor_sha256": anchor_sha}, run_dir / f"epoch{epoch:02d}.pt")
        if select_epoch(history) == epoch:
            _save({"model": model.state_dict(), "config": ccfg, "step": step, "epoch": epoch, "anchor_sha256": anchor_sha}, run_dir / "best.pt")
        if done: break
        if cfg.get("early_screen_stop", True) and len(history) == EARLY_EPOCH and should_stop_early(history):
            print(f"stopping after {EARLY_EPOCH} epochs: no eligible epoch reached +{EARLY_SNR} dB or +{EARLY_PESQ} PESQ on the screen"); break
    sel = select_epoch(history)
    info.update(end=time.time(), wall_s=time.time() - t0, steps=step, history=history, selected_epoch=sel,
                stopped_early=len(history) < cfg["epochs"] and not max_steps, clamp_history=clamp_history)
    json.dump(info, open(run_dir / "run.json", "w"), indent=2)
    if sel is None: print("no epoch met the STOI condition; best.pt not written")
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(); main(a.config, a.max_steps)
