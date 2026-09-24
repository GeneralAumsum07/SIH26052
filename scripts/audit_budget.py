"""Mini budget audit (spec 6.2): entries and matrix MMAC/s of r7, the reference-validity Mini, and untrained wider
ref_validity backbones as projection data. Counting is vaani.export.layer_macs on the streaming cascade, the same hooks
as docs/research/2026-09-24/profile_scaling.py (dense Conv/Linear/GRU MACs incl. the fixed ERB matrices; no bias, norm,
activation, elementwise, DSP or STFT). Costs only: random weights, no quality claim.

Budget (spec 6.2, 10 % over r7): at most 60,000 total entries (every parameter, frozen ERB banks included) and
at most 90.706 matrix MMAC/s for core + reliability extension + refiner.

usage: .venv/Scripts/python.exe scripts/audit_budget.py [--out results_r2/r8/budget]
"""
import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani.export import layer_macs
from vaani.models.cascade import StreamCascade, init_cascade_caches

BUDGET_ENTRIES, BUDGET_MMACS = 60_000, 90.706
HOPS_PER_S = 62.5   # 16 kHz / 256
R7_REFINER = ROOT / "results_r2/runs/r7_e256_wr64_refiner/best.pt"


def _refiner_cfg():
    if R7_REFINER.exists():
        cfg = torch.load(R7_REFINER, map_location="cpu", weights_only=True)["config"]
        return dict(cfg.get("refiner_cfg") or {}), cfg.get("model_cfg")
    return {"hidden": 16, "past": 2}, None


def audit(name, mc, rc, trained):
    torch.manual_seed(20260924)
    model = StreamCascade(mc, rc).eval()
    caches = init_cascade_caches(first_model_cfg=mc, refiner_cfg=rc)
    spec, feats = torch.randn(1, 257, 1, 6) * .1, torch.zeros(1, 1, 18)
    costs = layer_macs(model, (spec, feats, *caches))
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    macs = sum(costs.values()); mmacs = macs * HOPS_PER_S / 1e6
    return {"name": name, "trained_weights_exist": trained, "model_cfg": mc, "refiner_cfg": rc,
            "total_entries": params, "learnable_entries": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "buffer_entries": buffers, "state_entries": sum(c.numel() for c in caches),
            "ref_extension_entries": sum(p.numel() for n, p in model.named_parameters() if "ref_conv" in n),
            "matrix_macs_per_hop": macs, "matrix_mmac_per_second": round(mmacs, 3),
            "ref_extension_macs_per_hop": sum(v for k, v in costs.items() if "ref_conv" in k),
            "within_budget": params <= BUDGET_ENTRIES and mmacs <= BUDGET_MMACS}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results_r2/r8/budget")
    a = ap.parse_args(argv)
    rc, r7mc = _refiner_cfg()
    base = r7mc or {"channels": 16, "film": False, "coh": True, "df_order": 3, "noise_floor": False}
    rows = [audit("r7 (shipping)", dict(base), rc, True),
            audit("refvalid Mini (C16 + ref_validity + r7 refiner)", dict(base, ref_validity=True), rc, False)]
    for w in (32, 64, 96):   # projection only: untrained, and warm start cannot widen, so these would train from scratch
        rows.append(audit(f"C{w} + ref_validity (untrained projection)", dict(base, channels=w, ref_validity=True), rc, False))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cmd = "python scripts/audit_budget.py" + (" " + " ".join(argv) if argv else "")
    out.with_suffix(".json").write_text(json.dumps({"budget": {"total_entries": BUDGET_ENTRIES, "matrix_mmac_per_second": BUDGET_MMACS},
                                                    "counting": "vaani.export.layer_macs on StreamCascade, one hop", "command": cmd,
                                                    "rows": rows}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mini budget audit", "", f"Source: `budget.json`. Command: `{cmd}` (CPU, deterministic; counts, not timings).", "",
             f"Budget (spec 6.2): total entries <= {BUDGET_ENTRIES:,}, matrix MMAC/s <= {BUDGET_MMACS}.", "",
             "| system | total entries | learnable | ref extension entries | MMAC/s | ref extension MAC/hop | within budget |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['total_entries']:,} | {r['learnable_entries']:,} | {r['ref_extension_entries']:,} | "
                     f"{r['matrix_mmac_per_second']:.3f} | {r['ref_extension_macs_per_hop']:,} | {'yes' if r['within_budget'] else 'NO'} |")
    lines += ["", "Total entries count every parameter, the frozen ERB banks included; BN running buffers and the streaming",
              "state are listed separately in the JSON. C32/C64/C96 rows are untrained projections of cost only."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
