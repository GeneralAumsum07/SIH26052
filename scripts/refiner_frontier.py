"""Validation-only conditional screening and quality/matrix-MAC frontier tables.

No training. Sweep explicitly reads validation audio; summarize consumes existing
per-item evaluation CSVs and their named checkpoint files. Neither selects a test
winner. Estimated matrix MACs must be accompanied by target-device timings later.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline, stft
from vaani.export import _load_batch_model, _stream_twin, layer_macs
from vaani.experiments import sha256
from vaani.models.cascade import FrozenCascade
from vaani.models.conditional_refiner import ConditionalRefinerRuntime


def pareto_flags(rows):
    """SNR-vs-MAC Pareto flag only; STOI/PESQ remain explicit guardrail columns."""
    return [bool(np.isfinite(a["snr_out"])) and not any(b["matrix_mmacs_per_second"] <= a["matrix_mmacs_per_second"]
                    and b["snr_out"] >= a["snr_out"]
                    and (b["matrix_mmacs_per_second"] < a["matrix_mmacs_per_second"] or b["snr_out"] > a["snr_out"])
                    for b in rows) for a in rows]


def summarize(items):
    df = pd.DataFrame(items)
    nominal_conditions = ~df.clipped & ~df.ref_dropout & df.fault.isna()
    masks = {
        "nominal": nominal_conditions & df.snr_in.isin([0, 5, 10]),
        "severe_stationary": nominal_conditions & df.noise_class.str.contains("stationary", na=False) & (df.snr_in <= 0),
        "transient": df.fault.fillna("").str.startswith("fault_burst"),
        "clean": df.noise_class.eq("clean") & df.fault.isna(),
    }
    rows = []
    for envelope, mask in masks.items():
        cells = []
        for system, g in df[mask].groupby("system"):
            cell = dict(envelope=envelope, system=system, n=len(g),
                        matrix_mmacs_per_second=float(g.matrix_mmacs_per_second.iloc[0]))
            for m in ("snr_out", "stoi", "pesq_wb"):
                v = g[m].to_numpy(float); v = v[np.isfinite(v)]
                cell[m] = float(v.mean()) if len(v) else float("nan")
                cell[f"n_{m}"] = len(v)
            cells.append(cell)
        for cell, flag in zip(cells, pareto_flags(cells)):
            cell["pareto_snr"] = flag; rows.append(cell)
    return rows


@torch.no_grad()
def sweep(a):
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True); cfg = ck["config"]
    m = FrozenCascade.from_config(cfg).eval(); m.load_state_dict(ck["model"])
    policies = {"bypass": dict(mode="bypass"), "always": dict(mode="always")}
    policies.update({f"snr_{t:g}": dict(snr_threshold_db=t, speech_threshold=a.speech_threshold,
                                     reliability_threshold=a.reliability_threshold) for t in a.thresholds})
    runtimes = {name: ConditionalRefinerRuntime(m, **kw) for name, kw in policies.items()}
    stream, caches, _, _ = _stream_twin(m, cfg.get("model_cfg", {}))
    costs = layer_macs(stream, (torch.zeros(1, 257, 1, 6), torch.zeros(1, 1, 18), *caches))
    first_macs = sum(v for k, v in costs.items() if k.startswith("first."))
    ds = RenderedDataset(Path(a.eval_root) / a.split)
    if not ds.items:
        raise ValueError("No rendered items")
    seen, rows = {}, []
    for i, path in enumerate(ds.items):
        bucket = path.parent.name
        if a.per_bucket is not None and seen.get(bucket, 0) >= a.per_bucket:
            continue
        seen[bucket] = seen.get(bucket, 0) + 1
        item = ds[i]; meta = item["meta"]; mix, clean = item["mix"].numpy(), item["clean"].numpy()
        r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"))
        x = torch.from_numpy(r["mix"])[None]
        spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1)
        feats = torch.from_numpy(r["features"])[None]
        y = m.first(spec6, feats)
        for name, runtime in runtimes.items():
            z = runtime.refine(spec6, feats, y, torch.from_numpy(r["reliability"]))
            est = stft.istft(z, length=len(clean))[0].numpy()
            rows.append(dict(system=name, id=meta["id"], bucket=bucket, noise_class=meta["noise_class"],
                             snr_in=meta["snr_db"], clipped=meta["clipped"], ref_dropout=meta["ref_dropout"], fault=meta.get("fault"),
                             snr_out=metrics.snr_db(clean, est), stoi=metrics.stoi(clean, est), pesq_wb=metrics.pesq_wb(clean, est)))
    # Global frame-weighted cost is deliberately reused across the quality
    # envelopes; it is not represented as an envelope-specific activation rate.
    stats = {name: r.stats for name, r in runtimes.items()}
    for row in rows:
        row["matrix_mmacs_per_second"] = (first_macs + stats[row["system"]]["avg_matrix_macs_per_frame"]) * 62.5 / 1e6
    return rows, dict(checkpoint=a.checkpoint, checkpoint_sha256=sha256(a.checkpoint),
                      eval_root=a.eval_root, split=a.split, per_bucket=a.per_bucket, policies=policies, stats=stats,
                      cost_scope="global frame-weighted Conv/Linear/GRU matrix MAC estimate, includes c0 on skipped frames; not measured runtime")


def existing(a):
    rows, inputs, reference_keys = [], [], None
    paths = [p for pattern in a.csvs for p in sorted(glob.glob(pattern)) if ".partial" not in Path(p).name]
    if not paths:
        raise ValueError("No final CSVs matched")
    for p in paths:
        df = pd.read_csv(p, dtype={"id": str})
        systems = df.system.unique()
        if len(systems) != 1 or not str(systems[0]).startswith(("ckpt:", "cascade:")):
            raise ValueError(f"{p}: expected exactly one checkpoint system")
        if df.duplicated(["bucket", "id"]).any():
            raise ValueError(f"{p}: duplicate observations")
        keys = set(zip(df.bucket, df.id))
        if reference_keys is not None and keys != reference_keys:
            raise ValueError("Frontier CSVs must score exactly the same item keys")
        reference_keys = keys
        checkpoint = str(systems[0]).split(":", 1)[1]
        m, mc = _load_batch_model(checkpoint); s, caches, _, _ = _stream_twin(m, mc)
        costs = layer_macs(s, (torch.zeros(1, 257, 1, 6), torch.zeros(1, 1, 18), *caches))
        df["matrix_mmacs_per_second"] = sum(costs.values()) * 62.5 / 1e6
        rows.extend(df.to_dict("records"))
        inputs.append(dict(csv=p, csv_sha256=sha256(p), checkpoint=checkpoint, checkpoint_sha256=sha256(checkpoint)))
    return rows, dict(inputs=inputs, cost_scope="always-on matrix MAC estimate; caller must supply matched validation CSVs")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--checkpoint", required=True)
    sw.add_argument("--eval-root", default="data/eval_r2")
    sw.add_argument("--split", choices=["val", "test"], default="val")
    sw.add_argument("--allow-test", action="store_true")
    sw.add_argument("--thresholds", nargs="+", type=float, default=[6., 12., 18., 24.])
    sw.add_argument("--speech-threshold", type=float, default=.05)
    sw.add_argument("--reliability-threshold", type=float, default=None)
    sw.add_argument("--per-bucket", type=int, default=None)
    su = sub.add_parser("summarize"); su.add_argument("csvs", nargs="+")
    for parser in (sw, su):
        parser.add_argument("--out", required=True, help="output directory for items/frontier/provenance")
    a = ap.parse_args(argv)
    if a.command == "sweep":
        if a.split == "test" and not a.allow_test:
            ap.error("test diagnostics require --allow-test")
        if a.per_bucket is not None and a.per_bucket <= 0:
            ap.error("--per-bucket must be positive")
    torch.set_num_threads(1)
    rows, metadata = sweep(a) if a.command == "sweep" else existing(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "items.csv", index=False)
    pd.DataFrame(summarize(rows)).to_csv(out / "frontier.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}; no training launched")


if __name__ == "__main__":
    main()
