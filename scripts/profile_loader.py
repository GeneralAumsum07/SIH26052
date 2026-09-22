"""Fifteen minutes of measurement, to be run on the box before the long jobs start.

    uv run python scripts/profile_loader.py configs/retraining/r6_ctl64.yaml
    uv run python scripts/profile_loader.py configs/retraining/r6_ctl64.yaml --workers 24 48 72 96

Three questions, in the order they matter:

1. Is this box CPU-bound, as every plan since 2026-09-22 assumes? `--sweep` times batches at
   several worker counts. If throughput stops rising well before the core count, something other
   than the dataloader is the limit and the shared-loader plan needs rethinking.
2. Where does a single item's time go? `--profile` runs cProfile over N items and prints the top
   cumulative entries. The prediction on record is `pipeline.run` first, then FLAC decode.
3. Does the pack help? `--compare-pack` times the loader with and without it.

Nothing here trains; it is safe to run against any config.
"""
import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani import runtime                                          # noqa: E402
from vaani.data.dataset import DynamicMixDataset, EpochSampler, collate  # noqa: E402
from vaani.data.mixer import MixConfig                             # noqa: E402


def build(cfg, pack_root, epoch_len=None):
    d = cfg["data"]
    return DynamicMixDataset(
        d["manifests"], "train", d.get("bank"), MixConfig(**d.get("mix", {})),
        d.get("crop_s", 4.0), epoch_len or d.get("epoch_len", 20000), cfg["seed"],
        with_dsp=cfg["model"] == "vaani", controller_on=cfg["controller_on"],
        dsp_cfg=cfg.get("dsp"), pack_root=pack_root)


def profile_items(cfg, n, pack_root):
    ds = build(cfg, pack_root)
    ds[0]                                   # pay numba JIT and the lazy opens outside the timing
    pr = cProfile.Profile(); pr.enable()
    for i in range(1, n + 1):
        ds[i]
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(22)
    print(buf.getvalue())


def time_items(cfg, n, pack_root):
    ds = build(cfg, pack_root)
    ds[0]
    t = time.perf_counter()
    for i in range(1, n + 1):
        ds[i]
    dt = time.perf_counter() - t
    return dt / n


def sweep(cfg, workers, batches):
    from torch.utils.data import DataLoader
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for nw in workers:
        ds = build(cfg, cfg["data"].get("pack", "data/pack"))
        dl = DataLoader(ds, cfg["batch_size"], sampler=EpochSampler(len(ds)), collate_fn=collate,
                        **runtime.loader_kwargs(nw, device))
        it = iter(dl); next(it)             # warm the workers before timing
        t = time.perf_counter()
        got = 0
        for _ in range(batches):
            try:
                next(it); got += 1
            except StopIteration:
                break
        dt = time.perf_counter() - t
        per = dt / max(got, 1)
        steps_per_epoch = len(ds) // cfg["batch_size"]
        rows.append((nw, per, per * steps_per_epoch / 60))
        print(f"  workers={nw:>3}  {per * 1000:7.1f} ms/batch  ->  {rows[-1][2]:5.2f} min/epoch", flush=True)
        del it, dl, ds
    best = min(rows, key=lambda r: r[1])
    print(f"\nknee: workers={best[0]} at {best[1] * 1000:.1f} ms/batch ({best[2]:.2f} min/epoch)")
    print("Set VAANI_WORKERS to that, or leave num_workers: auto if it is already close.")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--workers", type=int, nargs="*", default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--compare-pack", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    everything = not (a.profile or a.sweep or a.compare_pack)

    print(f"box: {runtime.describe()}")
    cores = runtime.cpu_count()
    workers = a.workers or sorted({max(2, cores // 4), max(2, cores // 2), max(2, cores - 8), max(2, cores - 2)})

    if everything or a.compare_pack:
        print("\n== per-item cost, packed vs unpacked ==")
        packed = time_items(cfg, min(a.items, 100), cfg["data"].get("pack", "data/pack"))
        plain = time_items(cfg, min(a.items, 100), None)
        print(f"  packed   {packed * 1000:7.1f} ms/item")
        print(f"  soundfile{plain * 1000:7.1f} ms/item   ->  pack saves {100 * (1 - packed / plain):.0f}%")
        if packed >= plain:
            print("  NOTE: no saving - check data/pack exists and covers these manifests")

    if everything or a.profile:
        print(f"\n== where one item's time goes (cProfile, {a.items} items) ==")
        print("   prediction on record: pipeline.run first, FLAC decode second")
        profile_items(cfg, a.items, cfg["data"].get("pack", "data/pack"))

    if everything or a.sweep:
        print(f"\n== worker sweep ({a.batches} batches each) ==")
        sweep(cfg, workers, a.batches)
        print("\nWhile the real run is going, confirm the CPU-bound assumption:")
        print("  nvidia-smi dmon -s u -d 5      # sustained utilisation above ~60 % falsifies it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
