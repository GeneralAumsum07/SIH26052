"""INT8 dynamic quantization of an exported streaming graph, measured rather than asserted.

Clause 14b of SIH26052 names quantization. This module does the conversion and reports what
it actually cost, in the same three currencies the deployment contract already uses: graph
size, ORT single-core latency per 16 ms hop, and numerical agreement with the FP32 graph.

Two honest caveats belong with every number this produces:

1. **Random-input agreement is not an audio result.** `max_abs_err` here is measured on the
   same synthetic 10-second stream as `export.parity_and_timing`, which tells you the graph
   is wired correctly and roughly how far INT8 moved it. The quality cost that matters is
   SNR/STOI/PESQ on the frozen eval split -- run `vaani.eval --system onnx:<graph>@<ckpt>`
   for the FP32 and INT8 graphs and diff the report rows.
2. **This is a complex-valued mask network.** Dynamic quantization degrades phase-sensitive
   behaviour more readily than it does magnitude-only enhancement, because the error lands on
   the real and imaginary parts independently and the mask's phase is their ratio. Measure it;
   a bad result is still a reportable result.

Dynamic (weight-only, activations quantized on the fly) rather than static, because static
calibration would have to be frozen against a representative audio distribution and then
defended as part of the deployment contract. Dynamic needs no calibration set, so the graph
stays a pure function of the checkpoint.

    uv run python -m vaani.quantize deploy/tier46/cascade.onnx \
        --ckpt results_r2/runs/vaani_tier46_refiner/best.pt \
        --out deploy/tier46/cascade.int8.onnx --report-json deploy/tier46/int8_report.json
"""
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from vaani import export


# Ops ORT's dynamic path has an integer kernel for. Everything else -- notably GRU, which is
# most of this model's recurrence -- stays float32. A leftover entry here means the converter
# skipped a node it could have taken, which is the first thing to check if the size or latency
# result disappoints; it is not by itself a failed conversion.
DYNAMIC_QUANTIZABLE = ("Conv", "MatMul", "Gemm", "LSTM", "Attention", "EmbedLayerNormalization")


def graph_stats(onnx_path):
    """Node op_type counts, node count and initializer bytes.

    The initializer total is the one that explains a surprising size result: weights are only
    part of a graph's bytes, and the rest is node protobuf (names, shapes, attributes). On a
    50 K-parameter model that overhead dominates, so trading weight bytes for extra nodes can
    make the file *larger*. The report carries the numbers so the claim is checkable.
    """
    g = onnx.load(str(onnx_path)).graph
    return {"op_counts": dict(Counter(n.op_type for n in g.node)),
            "nodes": len(g.node),
            "initializers": len(g.initializer),
            "initializer_bytes": int(sum(numpy_helper.to_array(i).nbytes for i in g.initializer))}


def quantize(fp32_path, out_path, per_channel=False):
    """Weight-only INT8 dynamic quantization. Returns the written path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8, per_channel=per_channel)
    return out_path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _measure(paths, spec, feats, repeats):
    """Streaming output and best-of-`repeats` latency for each graph, measured interleaved.

    Interleaving matters: measuring one graph to completion and then the other lets a burst of
    background load land entirely on one of them. Measured A-then-B on this laptop, the same
    FP32 graph read 1.36 ms in one process and 0.90 ms in the next -- a 50 % swing that would
    have been silently attributed to INT8. Round-robin plus best-of-N keeps the comparison
    honest: the minimum is the right statistic because contention only ever adds time.
    """
    sessions = [export.load_session(p) for p in paths]
    best = [None] * len(paths)
    outs = [None] * len(paths)
    for _ in range(repeats):
        for i, sess in enumerate(sessions):
            outs[i], times = export.stream_onnx(sess, spec, feats)
            stats = export.timing_stats(times)
            if best[i] is None or stats["ms_per_frame_mean"] < best[i]["ms_per_frame_mean"]:
                best[i] = stats
    return outs, best


def report(fp32_path, int8_path, ckpt_path=None, seconds=10, repeats=3):
    """Size / latency / agreement for the FP32 and INT8 graphs under one protocol."""
    spec_t, feats_t = export.random_stream_inputs(seconds)
    spec, feats = spec_t.numpy(), feats_t.numpy()
    (fp32_out, int8_out), (fp32_time, int8_time) = _measure([fp32_path, int8_path], spec, feats, repeats)

    fp32_g, int8_g = graph_stats(fp32_path), graph_stats(int8_path)
    int8_ops = int8_g["op_counts"]
    fp32_bytes, int8_bytes = Path(fp32_path).stat().st_size, Path(int8_path).stat().st_size
    # Scale-relative, because an absolute delta on a spectrum whose own scale is arbitrary
    # says nothing; the FP32 output's max magnitude is the only defensible denominator.
    scale = float(np.abs(fp32_out).max())
    r = {"fp32": {"onnx": Path(fp32_path).as_posix(), "onnx_sha256": _sha256(fp32_path),
                  "onnx_bytes": fp32_bytes, **fp32_g, **fp32_time},
         "int8": {"onnx": Path(int8_path).as_posix(), "onnx_sha256": _sha256(int8_path),
                  "onnx_bytes": int8_bytes, **int8_g, **int8_time},
         "size_ratio": int8_bytes / fp32_bytes,
         "size_reduction_percent": 100.0 * (1 - int8_bytes / fp32_bytes),
         "latency_ratio": int8_time["ms_per_frame_mean"] / fp32_time["ms_per_frame_mean"],
         "quantizable_float_ops_remaining": {k: v for k, v in int8_ops.items() if k in DYNAMIC_QUANTIZABLE},
         "max_abs_err_vs_fp32_graph": float(np.abs(int8_out - fp32_out).max()),
         "relative_err_vs_fp32_graph": float(np.abs(int8_out - fp32_out).max() / scale) if scale else float("nan"),
         "seconds": seconds, "timed_frames": spec.shape[2] - 1, "repeats": repeats,
         "intra_op_num_threads": 1, "provider": "CPUExecutionProvider",
         "platform": platform.platform(), "processor": platform.processor(),
         "onnxruntime": ort.__version__, "onnx": onnx.__version__, "torch": torch.__version__,
         "scope": "ORT model only on a synthetic random stream; excludes DSP, STFT, iSTFT and audio I/O. "
                  "Numerical agreement here is a wiring check, not a speech-quality result: for that, "
                  "evaluate both graphs with vaani.eval --system onnx:<graph>@<ckpt> on the frozen split.",
         "method": "onnxruntime.quantization.quantize_dynamic, QInt8 weights, no calibration set"}
    if ckpt_path is not None:
        v, mc = export._load_batch_model(ckpt_path)
        with torch.no_grad():
            ref = v(spec_t, feats_t).numpy()
        r["checkpoint"] = Path(ckpt_path).as_posix()
        r["checkpoint_sha256"] = _sha256(ckpt_path)
        r["fp32"]["max_abs_err_vs_torch"] = float(np.abs(fp32_out - ref).max())
        r["int8"]["max_abs_err_vs_torch"] = float(np.abs(int8_out - ref).max())
    return r


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fp32_onnx")
    ap.add_argument("--out", default=None, help="INT8 graph path (default: <fp32>.int8.onnx alongside it)")
    ap.add_argument("--ckpt", default=None, help="also report each graph's agreement with the batch PyTorch model")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats; the fastest mean wins (background load only adds)")
    ap.add_argument("--per-channel", action="store_true", help="per-channel weight scales: more accurate, slightly larger")
    ap.add_argument("--report-json")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.fp32_onnx).with_suffix(".int8.onnx")
    quantize(a.fp32_onnx, out, per_channel=a.per_channel)
    r = report(a.fp32_onnx, out, a.ckpt, a.seconds, a.repeats)
    r["per_channel"] = a.per_channel
    if a.report_json:
        Path(a.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report_json).write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
