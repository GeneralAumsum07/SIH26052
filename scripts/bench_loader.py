"""Training loader throughput (plan 11.6 "measure items/s with v2 and numba before renting").

Builds the train DynamicMixDataset exactly as vaani.train.main does for each config (mixer version, DSP or the
VaaniFE front end, reference faults, held-out exclusion) and times a DataLoader at several worker counts.
The first batch is excluded (worker spawn, numba JIT, file-handle warm-up). Items/s per worker is what sizes the
rental's vCPU count: loader items/s must reach the GPU's step rate x batch size.

usage: uv run --with numba python scripts/bench_loader.py --out results_r2/r8/loader_bench.json \
           --configs configs/retraining/r8_fe_mini.yaml configs/retraining/r8_refvalid_v2.yaml \
           configs/retraining/r7_e256_wr64.yaml --workers 1 2 3 --batches 6
"""
import argparse, json, os, platform, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vaani.data.dataset import DynamicMixDataset, EpochSampler, collate  # noqa: E402
from vaani.data.mixer import MixConfig  # noqa: E402
from vaani import train  # noqa: E402


def build_dataset(cfg):
    """The train dataset with train.main's keyword wiring (kept in step with it)."""
    d = cfg["data"]
    dsk = dict(with_dsp=train.needs_dsp(cfg), controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"),
               pack_root=d.get("pack", "data/pack"), ref_corrupt=d.get("ref_corrupt"))
    if cfg["model"] == "vaani_fe":
        dsk["fe_inputs"] = True
    for k in ("exclude_groups_file", "scene_weights"):
        if k in d:
            dsk[k] = str(train._abs(d[k])) if k == "exclude_groups_file" and d[k] else d[k]
    return DynamicMixDataset(d["manifests"], "train", d.get("bank"), MixConfig(**d.get("mix", {})), d.get("crop_s", 4.0),
                             d.get("epoch_len", 20000), cfg["seed"], **dsk)


def bench(ds, workers, batch, batches):
    sampler = EpochSampler(len(ds))
    kw = dict(num_workers=workers, persistent_workers=workers > 0) if workers else {}
    dl = DataLoader(ds, batch, sampler=sampler, collate_fn=collate, **kw)
    it = iter(dl); t0 = time.perf_counter(); next(it); t1 = time.perf_counter()
    for _ in range(batches):
        next(it)
    t2 = time.perf_counter()
    del it, dl
    n = batches * batch
    return dict(workers=workers, items=n, first_batch_s=round(t1 - t0, 2), wall_s=round(t2 - t1, 2),
                items_per_s=round(n / (t2 - t1), 2), items_per_s_per_worker=round(n / (t2 - t1) / max(workers, 1), 2))


def step_time(cfg, batch, steps, warm):
    """Median / mean seconds per optimiser step on one fixed batch; the first `warm` steps (cuDNN autotune) dropped."""
    from vaani import losses
    dev = torch.device("cuda")
    cfg = dict(cfg); cfg["data"] = dict(cfg["data"], epoch_len=batch)
    ds = build_dataset(cfg); b = collate([ds[i] for i in range(batch)])
    model = train.build_model(cfg["model"], None, cfg.get("model_cfg")).to(dev).train()
    lc = cfg.get("loss_cfg", {})
    loss_fn = (losses.build_loss("fe", lc) if cfg["loss"] == "fe" else losses.HybridLoss(**lc)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    inputs, target, fw, is_clean = train.prepare_batch(b, cfg["model"], dev)
    torch.cuda.reset_peak_memory_stats(); ts = []
    for k in range(warm + steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(*inputs)
        loss = loss_fn(pred.float(), target, fw, is_clean)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize()
        if k >= warm:
            ts.append(time.perf_counter() - t0)
    ts.sort(); med = ts[len(ts) // 2]
    return dict(config=None, model=cfg["model"], model_cfg=cfg.get("model_cfg"), loss=cfg["loss"],
                params=sum(p.numel() for p in model.parameters()), batch=batch, crop_s=cfg["data"].get("crop_s", 4.0),
                steps_timed=steps, warmup_steps=warm, step_s_median=round(med, 4), step_s_mean=round(sum(ts) / len(ts), 4),
                items_per_s_gpu=round(batch / med, 1), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                gpu=torch.cuda.get_device_name(0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True); ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--out", required=True); ap.add_argument("--label", default="smoke: shared, loaded laptop; not reportable")
    ap.add_argument("--step-time", action="store_true"); ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args(argv)
    cmd = "uv run --with numba python scripts/bench_loader.py " + " ".join(argv or sys.argv[1:])
    if a.step_time:
        rows = []
        for c in a.configs:
            r = step_time(yaml.safe_load(open(c)), a.batch, a.steps, 5); r["config"] = c
            print(json.dumps(r), flush=True); rows.append(r)
        out = dict(label=a.label, command=cmd, torch=torch.__version__, amp="bf16 autocast, fp32 loss", rows=rows)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=2))
        return
    try:
        import numba; nb = numba.__version__
    except ImportError:
        nb = None
    rows = []
    for c in a.configs:
        cfg = yaml.safe_load(open(c)); ds = build_dataset(cfg)
        for w in a.workers:
            r = dict(config=c, model=cfg["model"], mix_version=int(cfg["data"].get("mix", {}).get("version", 1)),
                     dsp=train.needs_dsp(cfg), **bench(ds, w, a.batch, a.batches))
            print(json.dumps(r), flush=True); rows.append(r)
    out = dict(label=a.label, command=cmd,
               numba=nb, torch=torch.__version__, cpu_count=os.cpu_count(), platform=platform.platform(),
               crop_s=4.0, batch=a.batch, rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
