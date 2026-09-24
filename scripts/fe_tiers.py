"""Orin-tier projection work (plan 11.6): export every VaaniFE tier UNTRAINED (seeded random weights)
to runs/fe_tiers/ (git-ignored), run the G2 gate on each, time one ORT CPU thread, and write
results_r2/fe_tiers/tiers.json. r7's shipping graph is gated alongside as the comparison row.

  python scripts/fe_tiers.py [--seed 0] [--hops 500]

Random weights measure cost, never quality. Timing is laptop ORT CPU, 1 thread, on a loaded
shared machine: non-reportable until an idle-machine re-run.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import graph_gate  # noqa: E402
from vaani import export as E  # noqa: E402
from vaani.models import vaani_fe as V  # noqa: E402

TIMING_LABEL = "laptop ORT CPU 1 thread, loaded shared machine: smoke, non-reportable"


def run(seed=0, hops=500, out_dir=ROOT / "runs/fe_tiers", res=ROOT / "results_r2/fe_tiers/tiers.json"):
    import onnxruntime as ort
    import torch
    torch.set_num_threads(1)
    rows = []
    for tier in V.TIERS:
        arch = f"configs/arch/vaani_fe_{tier}.yaml"
        cfg = yaml.safe_load((ROOT / arch).read_text(encoding="utf-8"))
        m = E.fe_untrained(cfg, seed)
        rep = E.export_fe(m, out_dir / f"{tier}.onnx", seed=seed)
        g = graph_gate.gate(rep["onnx"], E.fe_untrained(cfg, seed), seed=seed)
        t = E.fe_timing(rep["folded"], m, hops=hops, seed=seed)
        s = g["structure"]
        rows.append({"tier": tier, "status": V.TIER_STATUS[tier], "arch": arch, "seed": seed,
                     "weights": "untrained, torch.manual_seed(seed) default init",
                     "c1_c2_f_k_l": "/".join(str(m.cfg[x]) for x in ("c1", "c2", "f", "k", "l")),
                     "params": rep["params"], "params_training_form": rep["params_training_form"],
                     "mac_per_hop": rep["mac_per_hop"], "mmac_per_s": rep["mmac_per_s"],
                     "state_bytes": rep["state_bytes"], "raw_nodes": s["raw_nodes"], "folded_nodes": s["folded_nodes"],
                     "layout_nodes": s["layout_nodes"], "layout_share": s["layout_share"], "ops": s["ops"],
                     "loop_nodes": s["loop_nodes"], "scatternd": s["scatternd"], "symbolic_dims": s["symbolic_dims"],
                     "shape_range_nodes": s["shape_range_nodes"],
                     "parity_ort_vs_torch_max_abs": g["parity"]["ort_vs_torch_max_abs"],
                     "stream_vs_offline_max_abs": g["parity"]["stream_vs_offline_max_abs"],
                     "g2_pass": g["pass"], "g2_failed": g["failed"],
                     "ort_cpu_1t_mean_ms": round(t["ms_per_hop_mean"], 4), "ort_cpu_1t_p99_ms": round(t["ms_per_hop_p99"], 4),
                     "timing_label": TIMING_LABEL, "onnx_sha256": rep["onnx_sha256"], "folded_sha256": rep["folded_sha256"]})
        print(json.dumps({k: rows[-1][k] for k in ("tier", "params", "mmac_per_s", "folded_nodes", "g2_pass",
                                                     "ort_cpu_1t_mean_ms", "ort_cpu_1t_p99_ms")}), flush=True)
    r7 = ROOT / "deploy/r7/cascade.onnx"
    comparison = None
    if r7.exists():
        g = graph_gate.gate(r7)
        s = g["structure"]
        comparison = {"onnx": "deploy/r7/cascade.onnx", "status": "trained, shipping (control and fallback)",
                      "raw_nodes": s["raw_nodes"], "folded_nodes": s["folded_nodes"], "layout_share": s["layout_share"],
                      "loop_nodes": s["loop_nodes"], "gru_nodes": s["ops"].get("GRU", 0), "scatternd": s["scatternd"],
                      "symbolic_dims": s["symbolic_dims"], "g2_pass": g["pass"], "g2_failed": g["failed"],
                      "parity": "not_run: r7 has no VaaniFE twin (its own parity is vaani/export.py parity_and_timing)"}
    out = {"generated_by": "python scripts/fe_tiers.py --seed %d --hops %d" % (seed, hops),
           "scope": "every row but mini is a projection: untrained, no quality claim; mini is to be trained in r8 "
                    "and these are its untrained-architecture costs",
           "g2_limits": {"max_nodes": graph_gate.MAX_NODES, "max_layout_share": graph_gate.MAX_LAYOUT_SHARE,
                         "parity_tol": E.FE_PARITY_TOL, "fold_level": "basic"},
           "mac_rule": "vaani.models.vaani_fe.count_macs: dense Conv/ConvTranspose/Linear/GRU matrices + attention QK^T and AV",
           "timing_label": TIMING_LABEL, "platform": platform.platform(), "processor": platform.processor(),
           "torch": torch.__version__, "onnxruntime": ort.__version__, "tiers": rows, "r7_comparison": comparison}
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hops", type=int, default=500)
    a = ap.parse_args()
    o = run(a.seed, a.hops)
    sys.exit(0 if all(r["g2_pass"] for r in o["tiers"]) else 1)
