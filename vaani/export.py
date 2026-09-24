"""Streaming VaaniNet -> ONNX with explicit state, plus parity and a CPU
timing proxy. The Pi measurement belongs to the embedded lead; this number
only tells us whether we are in the right order of magnitude."""
import time
import warnings
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from vaani.models.modules.convert import convert_to_stream
from vaani.models import cascade
from vaani.models.vaani_net import StreamVaaniNet, VaaniNet, init_caches


IN_NAMES = ["spec6", "feats", "conv_cache", "tra_cache", "inter_cache", "df_cache", "coh_cache"]
OUT_NAMES = ["spec_out", "conv_cache_out", "tra_cache_out", "inter_cache_out", "df_cache_out", "coh_cache_out"]


def layer_macs(model, inputs):
    """Dense Conv/Linear/GRU matrix MAC estimate for these inputs, not runtime.

    Includes fixed ERB Linear weights and padded convolution positions. Excludes
    bias, normalization, activations, elementwise arithmetic and DSP/STFT. Counting
    the streaming twin avoids charging the refiner for recomputing cached frames.
    """
    counts, handles = {}, []

    def hook(name):
        def count(m, args, out):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                n = out.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size)
            elif isinstance(m, torch.nn.ConvTranspose2d):
                n = args[0].numel() * (m.out_channels // m.groups) * math.prod(m.kernel_size)
            elif isinstance(m, torch.nn.Linear):
                n = out.numel() * m.in_features
            else:  # each GRU weight matrix is used once per sequence step and batch item
                steps = args[0].numel() // m.input_size
                n = steps * sum(p.numel() for k, p in m.named_parameters() if k.startswith("weight_"))
            counts[name] = counts.get(name, 0) + int(n)
        return count

    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear, torch.nn.GRU)):
            handles.append(m.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def deployment_report(ckpt_path, onnx_path, seconds=10):
    """Bind timing, graph identity and arithmetic estimates to the same checkpoint."""
    v, mc = _load_batch_model(ckpt_path)
    s, caches, _, _ = _stream_twin(v, mc)
    costs = layer_macs(s, (torch.zeros(1, 257, 1, 6), torch.zeros(1, 1, 18), *caches))
    stages = {"first_stage": v.first, "refiner": v.refiner} if isinstance(v, cascade.FrozenCascade) else {"first_stage": v}
    arithmetic = {}
    for name, m in stages.items():
        prefix = "first." if name == "first_stage" else "refiner."
        macs = sum(n for k, n in costs.items() if len(stages) == 1 or k.startswith(prefix))
        arithmetic[name] = {"parameters": sum(p.numel() for p in m.parameters()),
                            "trainable_parameters": sum(p.numel() for p in m.parameters() if p.requires_grad),
                            "matrix_macs_per_frame": macs, "matrix_mmacs_per_second": macs * 62.5 / 1e6}
    ckpt_path, onnx_path = Path(ckpt_path), Path(onnx_path)
    return {**parity_and_timing(ckpt_path, onnx_path, seconds),
            "checkpoint": ckpt_path.as_posix(), "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
            "onnx": onnx_path.as_posix(), "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "onnx_bytes": onnx_path.stat().st_size, "seconds": seconds, "timed_frames": int(seconds * 62.5) - 1,
            "intra_op_num_threads": 1, "provider": "CPUExecutionProvider", "platform": platform.platform(),
            "processor": platform.processor(), "torch": torch.__version__, "onnxruntime": ort.__version__,
            "scope": "ORT model only; excludes DSP, STFT, iSTFT and audio I/O; frame zero excluded from timing, included in parity",
            "arithmetic_scope": "Dense Conv/Linear/GRU matrix MAC estimate, including fixed ERB and padding; excludes bias, normalization, activations, elementwise ops, DSP and STFT; not runtime",
            "stages": arithmetic, "layer_matrix_macs": costs}


def _load_batch_model(ckpt_path):
    """Load a checkpoint into its batch module: VaaniNet for 'vaani', FrozenCascade for 'vaani_cascade'."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_kind = ck["config"]["model"]; mc = ck["config"].get("model_cfg", {})
    if model_kind == "vaani":
        v = VaaniNet(**mc).eval()
    elif model_kind == cascade.MODEL_NAME:
        v = cascade.FrozenCascade.from_config(ck["config"]).eval()
    else:
        raise NotImplementedError(f"export supports config['model'] in ('vaani', 'vaani_cascade'), got {model_kind!r}")
    v.load_state_dict(ck["model"])
    return v, mc


def _stream_twin(v, mc):
    """Streaming module + its cache tuple + ONNX names; the cascade appends refine_cache to the existing signature."""
    in_names, out_names = list(IN_NAMES), list(OUT_NAMES)
    if mc.get("noise_floor", False):
        in_names += ["noise_cache"]; out_names += ["noise_cache_out"]
    if isinstance(v, cascade.FrozenCascade):
        s = cascade.StreamCascade(mc, v.refiner_cfg).eval()
        convert_to_stream(s.first, v.first); s.refiner.load_state_dict(v.refiner.state_dict())
        return s, cascade.init_cascade_caches(first_model_cfg=mc, refiner_cfg=v.refiner_cfg), in_names + ["refine_cache"], out_names + ["refine_cache_out"]
    s = StreamVaaniNet(**mc).eval()
    convert_to_stream(s, v)  # load_state_dict fails: stream conv wrappers nest keys one level deeper
    return s, init_caches(channels=mc.get("channels", 16), noise_floor=mc.get("noise_floor", False)), in_names, out_names


def export(ckpt_path, out_path):
    v, mc = _load_batch_model(ckpt_path)
    s, caches, in_names, out_names = _stream_twin(v, mc)
    spec = torch.zeros(1, 257, 1, 6); f = torch.zeros(1, 1, 18)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # Legacy TorchScript exporter warns on the GRU bool trace and Slice folding; shapes
        # are static batch-1 so the trace is exact, as proven by parity_and_timing.
        warnings.simplefilter("ignore")
        torch.onnx.export(s, (spec, f, *caches), str(out_path), opset_version=17,
                           input_names=in_names, output_names=out_names, dynamo=False)
    return Path(out_path)


def load_session(onnx_path, threads=1):
    """One ORT CPU session at the embedded single-core budget."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads  # matches the embedded single-core budget
    opts.log_severity_level = 3  # silence unused-initializer warnings at the source
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def zero_caches(sess):
    """Cache names and zero initial values read off the graph, not off a checkpoint.

    The contract says every cache is zero at stream start, so the graph's own declared
    shapes are sufficient: this works for a quantized or pruned graph, and for widths and
    optional caches this code has never seen. Inputs 0/1 are spec6 and feats.
    """
    names, values = [], []
    for spec in sess.get_inputs()[2:]:
        if any(not isinstance(d, int) for d in spec.shape):
            raise ValueError(f"cache input {spec.name!r} has a dynamic shape {spec.shape}; export is static batch-1")
        names.append(spec.name); values.append(np.zeros(spec.shape, np.float32))
    return names, values


def stream_onnx(sess, spec, feats, cache_names=None, caches=None):
    """Frame-by-frame ORT run carrying caches forward, exactly as the embedded loop must.

    spec (1,257,T,6) and feats (1,T,18) are float32 numpy. Returns the concatenated output
    spectrum (1,257,T,2) and the per-frame milliseconds with frame zero dropped -- it is a
    session/allocator warm-up and skews timing, but its output is still in the returned
    spectrum so parity covers it.
    """
    if cache_names is None or caches is None:
        cache_names, caches = zero_caches(sess)
    outs, times = [], []
    for t in range(spec.shape[2]):
        inp = {"spec6": spec[:, :, t:t + 1], "feats": feats[:, t:t + 1], **dict(zip(cache_names, caches))}
        t0 = time.perf_counter()
        o = sess.run(None, inp)
        if t > 0:
            times.append((time.perf_counter() - t0) * 1000)
        outs.append(o[0]); caches = o[1:]
    return np.concatenate(outs, axis=2), times


def random_stream_inputs(seconds=10, seed=0):
    """The shared 10-second random-input timing protocol; torch tensors so the batch reference can consume them."""
    T = int(seconds * 16000 / 256)
    if T < 2:
        raise ValueError("Timing requires at least two frames (one warm-up and one measured)")
    torch.manual_seed(seed)
    return torch.randn(1, 257, T, 6) * 0.1, torch.randn(1, T, 18)


def timing_stats(times):
    return {"ms_per_frame_mean": float(np.mean(times)), "ms_per_frame_p99": float(np.percentile(times, 99))}


def parity_and_timing(ckpt_path, onnx_path, seconds=10):
    v, mc = _load_batch_model(ckpt_path)
    spec, f = random_stream_inputs(seconds)
    with torch.no_grad():
        ref = v(spec, f).numpy()

    sess = load_session(onnx_path)
    _, caches, in_names, _ = _stream_twin(v, mc)
    got, times = stream_onnx(sess, spec.numpy(), f.numpy(), in_names[2:], [c.numpy() for c in caches])
    return {"max_abs_err": float(np.abs(got - ref).max()), **timing_stats(times)}


SHIPPING_CKPT = "results_r2/runs/r7_e256_wr64_refiner/best.pt"  # tracked r7 cascade; embeds its backbone
SHIPPING_ONNX = "deploy/r7/cascade.onnx"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=SHIPPING_CKPT); ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--report-json", help="Save timing, checkpoint/ONNX hashes and per-stage parameter/MAC estimates")
    ap.add_argument("--overwrite", action="store_true", help="Allow the default --out to replace an existing graph")
    a = ap.parse_args()
    if a.out is None:   # the cascade never overwrites the shipped first-stage graph
        kind = torch.load(a.ckpt, map_location="cpu", weights_only=True)["config"]["model"]
        a.out = SHIPPING_ONNX if kind == cascade.MODEL_NAME else "deploy/model.onnx"
        # a re-export on another torch differs in the last ulp of folded weights, so never replace a sha-pinned graph silently
        if Path(a.out).exists() and not a.overwrite:
            ap.error(f"{a.out} exists; pass --out <path> or --overwrite")
    p = export(a.ckpt, a.out)
    r = deployment_report(a.ckpt, p, a.seconds) if a.report_json else parity_and_timing(a.ckpt, p, a.seconds)
    if a.report_json:
        Path(a.report_json).write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(r, indent=2))
