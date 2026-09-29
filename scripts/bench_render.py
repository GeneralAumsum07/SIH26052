"""CPU mixer vs GPU renderer (perf.numerics.render cpu|gpu): per-item costs and end-to-end loader throughput.

Per item, single process: the CPU path's ds[i]; the GPU path's CPU half (ds.recipe) and its render + finish
(mixer_gpu.render_and_finish, as the batch server runs it). End to end: a DataLoader at --workers per path, the GPU
path rendering in the main process exactly as vaani/train.py and the stream server do.

usage: python scripts/bench_render.py --config configs/retraining/r8_ld_fe_mini.yaml --workers 40 --out results_r2/r8_ld/perf/render_bench.json
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.bench_loader import build_dataset  # noqa: E402
from vaani.data import mixer_gpu  # noqa: E402
from vaani.data.dataset import EpochBatchSampler, collate  # noqa: E402


def per_item(ds, n, device):
    t0 = time.perf_counter()
    for i in range(n):
        ds[i]
    cpu = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    recs = [ds.recipe(i) for i in range(n)]
    recipe = (time.perf_counter() - t0) / n
    mixer_gpu.render_and_finish(ds, recs[:2], device); torch.cuda.synchronize()   # CUDA context + kernel warm-up
    t0 = time.perf_counter()
    mixer_gpu.render_and_finish(ds, recs, device); torch.cuda.synchronize()
    render = (time.perf_counter() - t0) / n
    return dict(cpu_item_s=cpu, gpu_recipe_s=recipe, gpu_render_finish_s=render)


def end_to_end(ds, path, workers, batch, batches, device):
    bs = EpochBatchSampler(len(ds), batch, 0, 50)
    lk = dict(num_workers=workers, persistent_workers=False, prefetch_factor=2)
    if path == "gpu":
        dl = DataLoader(mixer_gpu.RecipeDataset(ds), batch_sampler=bs, collate_fn=mixer_gpu.collate_recipes, **lk)
        it = (mixer_gpu.render_and_finish(ds, r, device) for r in dl)
    else:
        it = iter(DataLoader(ds, batch_sampler=bs, collate_fn=collate, **lk))
    for _ in range(2 * workers + 1):   # spawn, JIT, and drain the prefetch buffer: time the steady state only
        next(it)
    t0 = time.perf_counter()
    for _ in range(batches):
        next(it)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dict(path=path, workers=workers, batches=batches, items_per_s=round(batch * batches / dt, 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--items", type=int, default=64)
    ap.add_argument("--workers", type=int, default=40); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--batches", type=int, default=30); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    ds = build_dataset(cfg); dev = torch.device("cuda")
    rep = dict(config=a.config, gpu=torch.cuda.get_device_name(0), per_item=per_item(ds, a.items, dev),
               end_to_end=[end_to_end(ds, p, a.workers, a.batch, a.batches, dev) for p in ("cpu", "gpu")])
    pi = rep["per_item"]
    rep["cpu_cost_ratio"] = round(pi["cpu_item_s"] / pi["gpu_recipe_s"], 2)   # CPU seconds saved per item by the GPU path
    rep["render_ceiling_items_per_s"] = round(1 / pi["gpu_render_finish_s"], 1)   # one serial renderer per stream
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
