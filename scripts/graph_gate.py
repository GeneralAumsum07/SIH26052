"""G2 graph gate (plan 11.3 design rules, 11.8): structural budget for a streaming ONNX step graph,
plus FP32 ORT-vs-torch parity and streaming == offline when a VaaniFE torch twin is given.

The graph is first folded with ORT's provider-independent basic level (constant folding, Conv+BN
fusion, redundant-node removal); every count is on the folded graph. Checks:
  nodes            folded node count < 250
  loops            Loop/Scan/If/GRU/LSTM/RNN nodes == 0 (TensorRT lowers them to loops: no CUDA graph)
  scatternd        ScatterND == 0 (caches must be Slice+Concat)
  symbolic_dims    0 tensor dims without a static value (inputs, outputs, inferred value_info)
  shape_ops        Shape/Range nodes == 0 after folding
  layout_share     layout ops < 30% of nodes; layout = LAYOUT_OPS below (pure data movement)
  parity           ORT-vs-torch carried-state max abs spectral error <= 1e-5 (twin only)
  stream_offline   torch step vs torch offline forward <= 1e-5 (twin only)
Without a twin, parity/stream_offline are reported "not_run" and do not fail the gate.

  python scripts/graph_gate.py runs/fe_tiers/mini.onnx --tier mini --seed 0 --json out.json
  python scripts/graph_gate.py deploy/r7/cascade.onnx          # documented comparison; expected FAIL
Exit code 1 when any check fails.
"""
import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani import export as E  # noqa: E402

MAX_NODES = 250
MAX_LAYOUT_SHARE = 0.30
LOOP_OPS = {"Loop", "Scan", "If", "GRU", "LSTM", "RNN"}
SHAPE_OPS = {"Shape", "Range"}
LAYOUT_OPS = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "Expand", "Concat", "Slice", "Gather", "Split"}


def _symbolic_dims(model):
    """Count dims with no static value over graph inputs, outputs and inferred value_info."""
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:  # inference failure leaves value_info empty; I/O still counted
        pass
    g, n, names = model.graph, 0, []
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            n += 1; names.append(vi.name); continue
        bad = sum(1 for d in t.shape.dim if not d.HasField("dim_value"))
        if bad:
            n += bad; names.append(vi.name)
    return n, names[:10]


def structure(onnx_path, level="basic"):
    with tempfile.TemporaryDirectory() as td:
        folded = E.fe_fold(onnx_path, Path(td) / "folded.onnx", level)
        model = onnx.load(str(folded))
    ops = collections.Counter(nd.op_type for nd in model.graph.node)
    nodes = sum(ops.values())
    layout = sum(v for k, v in ops.items() if k in LAYOUT_OPS)
    sym, sym_names = _symbolic_dims(model)
    raw_nodes = len(onnx.load(str(onnx_path)).graph.node)
    return {"raw_nodes": raw_nodes, "folded_nodes": nodes, "ops": dict(sorted(ops.items())),
            "loop_nodes": sum(ops[k] for k in LOOP_OPS), "scatternd": ops.get("ScatterND", 0),
            "shape_range_nodes": sum(ops[k] for k in SHAPE_OPS), "layout_nodes": layout,
            "layout_share": round(layout / max(nodes, 1), 4), "symbolic_dims": sym,
            "symbolic_examples": sym_names, "fold_level": level}


def gate(onnx_path, twin=None, level="basic", streams=3, hops=200, seed=0):
    s = structure(onnx_path, level)
    checks = {"nodes": s["folded_nodes"] < MAX_NODES, "loops": s["loop_nodes"] == 0,
              "scatternd": s["scatternd"] == 0, "symbolic_dims": s["symbolic_dims"] == 0,
              "shape_ops": s["shape_range_nodes"] == 0, "layout_share": s["layout_share"] < MAX_LAYOUT_SHARE}
    rep = {"onnx": Path(onnx_path).as_posix(), "structure": s,
           "limits": {"max_nodes": MAX_NODES, "max_layout_share": MAX_LAYOUT_SHARE,
                      "parity_tol": E.FE_PARITY_TOL, "layout_ops": sorted(LAYOUT_OPS), "loop_ops": sorted(LOOP_OPS)}}
    if twin is not None:
        with tempfile.TemporaryDirectory() as td:  # parity on the folded graph, i.e. what ships
            p = E.fe_parity(twin, E.fe_fold(onnx_path, Path(td) / "folded.onnx", level), streams, hops, seed)
        rep["parity"] = p
        checks["parity"] = p["ort_vs_torch_max_abs"] <= E.FE_PARITY_TOL
        checks["stream_offline"] = p["stream_vs_offline_max_abs"] <= E.FE_PARITY_TOL
    else:
        rep["parity"] = "not_run: no torch twin"
    rep["checks"] = checks
    rep["failed"] = sorted(k for k, ok in checks.items() if not ok)
    rep["pass"] = not rep["failed"]
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    tw = ap.add_mutually_exclusive_group()
    tw.add_argument("--tier", help="torch twin: seeded untrained VaaniFE tier (use the export's seed)")
    tw.add_argument("--arch", help="torch twin: seeded untrained VaaniFE from configs/arch/*.yaml")
    tw.add_argument("--ckpt", help="torch twin: trained VaaniFE checkpoint from vaani/train.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="basic", choices=["basic", "extended"])
    ap.add_argument("--hops", type=int, default=200)
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    twin = None
    if a.tier:
        twin = E.fe_untrained(a.tier, a.seed)
    elif a.arch:
        import yaml
        twin = E.fe_untrained(yaml.safe_load(Path(a.arch).read_text(encoding="utf-8")), a.seed)
    elif a.ckpt:
        twin = E.fe_load(a.ckpt)
    rep = gate(a.onnx, twin, a.level, a.streams, a.hops, a.seed)
    txt = json.dumps(rep, indent=2)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
