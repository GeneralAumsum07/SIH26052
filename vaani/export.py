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


# ---- VaaniFE (plan 11.3/11.4): step-graph export, ORT folding, carried-state parity ------------
FE_OPSET = 17
FE_PARITY_TOL = 1e-5  # spec 8: FP32 max abs spectral error on the bounded parity corpus


def fe_untrained(arch, seed=0):
    """Seeded random-weight VaaniFE for projection exports; arch is a tier name or an arch dict."""
    from vaani.models import vaani_fe
    torch.manual_seed(seed)  # same seed -> same weights, so a gate can rebuild the torch twin
    m = vaani_fe.build(arch) if isinstance(arch, str) else vaani_fe.from_arch(arch)
    return m.eval()


def fe_load(ckpt_path):
    """Trained VaaniFE from a train.py checkpoint ({'model': state_dict, 'config': {'model_cfg': ...}})."""
    from vaani.models import vaani_fe
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m = vaani_fe.from_arch(ck["config"].get("model_cfg", {}))
    m.load_state_dict(ck["model"])
    return m.eval()


def fe_fold(onnx_path, out_path, level="basic"):
    """Save ORT's offline-optimised graph. 'basic' is provider-independent (constant folding, Conv+BN
    fusion, redundant-node removal); 'extended' may add CPU-only contrib ops, so it is not portable."""
    so = ort.SessionOptions()
    so.graph_optimization_level = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                   "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED}[level]
    so.optimized_model_filepath = str(out_path)
    so.log_severity_level = 3
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    return Path(out_path)


def fe_parity_corpus(model, streams=3, hops=200, seed=0):
    """Bounded random carried-state corpus: randn*0.1 spectra (the r7 protocol scale), validity
    held in random runs of 10-60 hops so both present and absent reference states carry over."""
    g = np.random.default_rng(seed)
    out = []
    for _ in range(streams):
        spec = (g.standard_normal((1, 257, hops, model.n_raw)) * 0.1).astype(np.float32)
        valid, t = np.ones((1, hops), np.float32), 0
        while t < hops:
            run = int(g.integers(10, 61)); valid[:, t:t + run] = float(g.random() > 0.3); t += run
        out.append((spec, valid))
    return out


def fe_stream_ort(sess, model, spec, valid):
    """Carry the flat state through ORT hop by hop. spec (1,257,T,n) offline layout, valid (1,T);
    returns (1,257,T,2) and the final state."""
    state = np.zeros((1, model.state_size), np.float32)
    outs = []
    for t in range(spec.shape[2]):
        feeds = {"spec": np.ascontiguousarray(spec[:, :, t].transpose(0, 2, 1)), "state": state}
        if model.uses_ref:
            feeds["valid"] = valid[:, t:t + 1]
        o, state = sess.run(["spec_out", "state_out"], feeds)
        outs.append(o.transpose(0, 2, 1)[:, :, None])  # (1,2,257) -> (1,257,1,2)
    return np.concatenate(outs, 2), state


def fe_parity(model, onnx_path, streams=3, hops=200, seed=0):
    """ORT vs torch step (carried state) and torch step vs torch offline, on the same corpus."""
    from vaani.models import vaani_fe
    sess = load_session(onnx_path)
    err, rel, state_err, off_err = 0.0, 0.0, 0.0, 0.0
    for spec, valid in fe_parity_corpus(model, streams, hops, seed):
        got, st_ort = fe_stream_ort(sess, model, spec, valid)
        s, v = torch.from_numpy(spec), torch.from_numpy(valid)
        with torch.no_grad():
            st, ref = model.init_state(1), []
            for t in range(s.shape[2]):
                o, st = model.step(vaani_fe.frame_to_step(s[:, :, t:t + 1]), v[:, t:t + 1], st)
                ref.append(vaani_fe.step_to_frame(o))
            ref = torch.cat(ref, 2).numpy()
            off = model(s, None, v).numpy()
        d = np.abs(got - ref)
        err, off_err = max(err, float(d.max())), max(off_err, float(np.abs(ref - off).max()))
        rel = max(rel, float(d.max() / max(np.abs(ref).max(), 1e-12)))
        state_err = max(state_err, float(np.abs(st_ort - st.numpy()).max()))
    return {"ort_vs_torch_max_abs": err, "ort_vs_torch_max_rel": rel, "state_max_abs": state_err,
            "stream_vs_offline_max_abs": off_err, "streams": streams, "hops_per_stream": hops, "seed": seed,
            "corpus": "randn*0.1 raw spectra, validity in random 10-60 hop runs (p_valid 0.7)",
            "pass": err <= FE_PARITY_TOL and off_err <= FE_PARITY_TOL}


def fe_timing(onnx_path, model, hops=500, warm=50, seed=0):
    """1-thread ORT CPU per-hop step time with carried state; model only (no DSP/STFT/I/O)."""
    sess = load_session(onnx_path, threads=1)
    spec, valid = fe_parity_corpus(model, 1, hops + warm, seed)[0]
    state, times = np.zeros((1, model.state_size), np.float32), []
    for t in range(hops + warm):
        feeds = {"spec": np.ascontiguousarray(spec[:, :, t].transpose(0, 2, 1)), "state": state}
        if model.uses_ref:
            feeds["valid"] = valid[:, t:t + 1]
        t0 = time.perf_counter()
        _, state = sess.run(["spec_out", "state_out"], feeds)
        if t >= warm:
            times.append((time.perf_counter() - t0) * 1000)
    return {"ms_per_hop_mean": float(np.mean(times)), "ms_per_hop_p99": float(np.percentile(times, 99)),
            "hops": hops, "warmup": warm, "intra_op_num_threads": 1, "provider": "CPUExecutionProvider"}


def export_fe(model, out_path, folded_path=None, level="basic", parity=True, streams=3, hops=200, seed=0):
    """Export VaaniFE.step (opset 17, static batch one, flat state, named I/O), save ORT's folded graph
    and check ORT-vs-torch parity over carried-state hops. Returns a report dict."""
    from vaani.models.vaani_fe import StepGraph, summary
    model = model.eval()
    wrap = StepGraph(model).eval()
    ins, outs = wrap.io_names()
    out_path = Path(out_path)
    folded_path = Path(folded_path) if folded_path else out_path.with_name(out_path.stem + ".folded.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tracer warnings on static batch-one shapes; parity below is the proof
        torch.onnx.export(wrap, wrap.example_inputs(seed), str(out_path), opset_version=FE_OPSET,
                          input_names=ins, output_names=outs, dynamo=False, do_constant_folding=True)
    fe_fold(out_path, folded_path, level)
    rep = {"onnx": out_path.as_posix(), "folded": folded_path.as_posix(), "fold_level": level, "opset": FE_OPSET,
           "inputs": ins, "outputs": outs, "onnx_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
           "folded_sha256": hashlib.sha256(folded_path.read_bytes()).hexdigest(), **summary(model),
           "torch": torch.__version__, "onnxruntime": ort.__version__}
    if parity:
        rep["parity"] = fe_parity(model, folded_path, streams, hops, seed)
    return rep


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
