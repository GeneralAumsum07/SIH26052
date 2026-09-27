"""Orin-tier projection work (plan 11.6): export every VaaniFE tier UNTRAINED (seeded random weights)
to runs/fe_tiers/ (git-ignored), run the G2 gate on each, time one ORT CPU thread, and write
results_r2/fe_tiers/tiers.json. r7's shipping graph is gated alongside as the comparison row.

  python scripts/fe_tiers.py [--seed 0] [--hops 500]
  python scripts/fe_tiers.py --low-delay      # low-delay plan Task 5: results_r2/fe_tiers/tiers_ld.json

--low-delay exports every tier under each shortlisted low-delay contract (Arm A's three supports and Arm B), Mini-P18
under each Arm A contract, Mini-P32 under Arm B and Arm R's network under each Arm A contract. Each graph carries
its contract in metadata_props and is gated and stamped separately (some share shapes or weights).

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
from vaani import audio_contract as ac  # noqa: E402
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


def low_delay_networks():
    """(name, model_cfg) of every low-delay step graph Task 5 exports."""
    nets = []
    for tier in V.TIERS:
        base = yaml.safe_load((ROOT / f"configs/arch/vaani_fe_{tier}.yaml").read_text(encoding="utf-8"))
        base = dict(base.get("model_cfg", base))
        for cid in (*ac.ARM_A_IDS, ac.ARM_B_ID):
            nets.append((f"{tier}@{cid}", {**base, "audio_contract": cid}))
    for cid in ac.ARM_A_IDS:
        nets.append((f"mini_p18@{cid}", {"tier": "mini", "audio_contract": cid, **V.MINI_P["p18"]}))
    nets.append((f"mini_p32@{ac.ARM_B_ID}", {"tier": "mini", "audio_contract": ac.ARM_B_ID, **V.MINI_P["p32"]}))
    for cid in ac.ARM_A_IDS:
        nets.append((f"arm_r@{cid}", {"tier": "mini", "audio_contract": cid, "df_bins": 96, "df_lags": (0, 3, 5)}))
    return nets


def run_low_delay(seed=0, hops=500, out_dir=ROOT / "runs/fe_tiers_ld", res=ROOT / "results_r2/fe_tiers/tiers_ld.json",
                  only=None):
    import onnxruntime as ort
    import torch
    torch.set_num_threads(1)
    rows = []
    for name, mc in low_delay_networks():
        if only is not None and name not in only:
            continue
        m = E.fe_untrained(mc, seed)
        path = out_dir / (name.replace("@", "__") + ".onnx")
        rep = E.export_fe(m, path, seed=seed)
        g = graph_gate.gate(rep["onnx"], E.fe_untrained(mc, seed), seed=seed)
        t = E.fe_timing(rep["folded"], m, hops=hops, seed=seed)
        s, p = g["structure"], g["parity"]
        rows.append({"network": name, "audio_contract": rep["audio_contract"], "audio_contract_hash": rep["audio_contract_hash"],
                     "profile": rep["metadata"].get(ac.META_PROFILE), "model_cfg": mc, "seed": seed,
                     "weights": "untrained, torch.manual_seed(seed) default init",
                     "params": rep["params"], "mac_per_hop": rep["mac_per_hop"], "mmac_per_s": rep["mmac_per_s"],
                     "state_bytes": rep["state_bytes"], "df_cache_bytes": rep["df_cache_bytes"],
                     "folded_nodes": s["folded_nodes"], "layout_share": s["layout_share"], "loop_nodes": s["loop_nodes"],
                     "scatternd": s["scatternd"], "symbolic_dims": s["symbolic_dims"],
                     "parity_ort_vs_torch_max_abs": p["ort_vs_torch_max_abs"], "parity_max_rel": p["ort_vs_torch_max_rel"],
                     "state_max_abs": p["state_max_abs"], "state_max_rel": p["state_max_rel"],
                     "stream_vs_offline_max_abs": p["stream_vs_offline_max_abs"], "reset_max_abs": p["reset_max_abs"],
                     "interleaved_max_abs": p["interleaved_max_abs"], "contract_check": g["audio_contract"],
                     "g2_pass": g["pass"], "g2_failed": g["failed"],
                     "ort_cpu_1t_mean_ms": round(t["ms_per_hop_mean"], 4), "ort_cpu_1t_p99_ms": round(t["ms_per_hop_p99"], 4),
                     "deadline_ms": ac.get_audio_contract(rep["audio_contract"]).deadline_ms,
                     "timing_label": TIMING_LABEL, "onnx_sha256": rep["onnx_sha256"], "folded_sha256": rep["folded_sha256"]})
        print(json.dumps({k: rows[-1][k] for k in ("network", "mmac_per_s", "folded_nodes", "layout_share", "g2_pass",
                                                     "ort_cpu_1t_mean_ms")}), flush=True)
    out = {"generated_by": "python scripts/fe_tiers.py --low-delay --seed %d --hops %d" % (seed, hops),
           "scope": "untrained cost and graph-structure rows; no quality claim",
           "g2_limits": {"max_nodes": graph_gate.MAX_NODES, "max_layout_share": graph_gate.MAX_LAYOUT_SHARE,
                         "parity_tol": E.FE_PARITY_TOL, "fold_level": "basic"},
           "timing_label": TIMING_LABEL, "platform": platform.platform(), "torch": torch.__version__,
           "onnxruntime": ort.__version__, "providers": ["CPUExecutionProvider"], "networks": rows}
    if only is None:
        res.parent.mkdir(parents=True, exist_ok=True)
        res.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hops", type=int, default=500)
    ap.add_argument("--low-delay", action="store_true", help="export the low-delay networks (Task 5)")
    a = ap.parse_args()
    if a.low_delay:
        o = run_low_delay(a.seed, a.hops)
        sys.exit(0 if all(r["g2_pass"] for r in o["networks"]) else 1)
    o = run(a.seed, a.hops)
    sys.exit(0 if all(r["g2_pass"] for r in o["tiers"]) else 1)
