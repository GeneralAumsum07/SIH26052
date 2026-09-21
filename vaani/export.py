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


def parity_and_timing(ckpt_path, onnx_path, seconds=10):
    v, mc = _load_batch_model(ckpt_path)
    T = int(seconds * 16000 / 256)
    if T < 2:
        raise ValueError("Timing requires at least two frames (one warm-up and one measured)")
    torch.manual_seed(0)
    spec = torch.randn(1, 257, T, 6) * 0.1
    f = torch.randn(1, T, 18)
    with torch.no_grad():
        ref = v(spec, f).numpy()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # matches the embedded single-core budget
    opts.log_severity_level = 3  # silence unused-initializer warnings at the source
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    _, caches, in_names, _ = _stream_twin(v, mc)
    caches = [c.numpy() for c in caches]
    outs, times = [], []
    for t in range(T):
        inp = {"spec6": spec[:, :, t:t + 1].numpy(), "feats": f[:, t:t + 1].numpy(), **dict(zip(in_names[2:], caches))}
        t0 = time.perf_counter()
        o = sess.run(None, inp)
        if t > 0:  # frame 0 is a warm-up (session/allocator warm-up skews timing); still used for parity
            times.append((time.perf_counter() - t0) * 1000)
        outs.append(o[0]); caches = o[1:]

    got = np.concatenate(outs, axis=2)
    return {"max_abs_err": float(np.abs(got - ref).max()),
            "ms_per_frame_mean": float(np.mean(times)),
            "ms_per_frame_p99": float(np.percentile(times, 99))}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", default=None)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--report-json", help="Save timing, checkpoint/ONNX hashes and per-stage parameter/MAC estimates")
    a = ap.parse_args()
    if a.out is None:   # the cascade never overwrites the shipped first-stage graph
        kind = torch.load(a.ckpt, map_location="cpu", weights_only=True)["config"]["model"]
        a.out = "deploy/tier46/cascade.onnx" if kind == cascade.MODEL_NAME else "deploy/model.onnx"
    p = export(a.ckpt, a.out)
    r = deployment_report(a.ckpt, p, a.seconds) if a.report_json else parity_and_timing(a.ckpt, p, a.seconds)
    if a.report_json:
        Path(a.report_json).write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(r, indent=2))
