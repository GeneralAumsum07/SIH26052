"""Streaming VaaniNet -> ONNX with explicit state, plus parity and a CPU
timing proxy. The Pi measurement belongs to the embedded lead; this number
only tells us whether we are in the right order of magnitude."""
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from vaani.models.modules.convert import convert_to_stream
from vaani.models import cascade
from vaani.models.vaani_net import StreamVaaniNet, VaaniNet, init_caches


IN_NAMES = ["spec6", "feats", "conv_cache", "tra_cache", "inter_cache", "df_cache", "coh_cache"]
OUT_NAMES = ["spec_out", "conv_cache_out", "tra_cache_out", "inter_cache_out", "df_cache_out", "coh_cache_out"]


def _load_batch_model(ckpt_path):
    """Load a checkpoint into its batch module: VaaniNet for 'vaani', FrozenCascade for 'vaani_cascade'."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_kind = ck["config"]["model"]; mc = ck["config"].get("model_cfg", {})
    if model_kind == "vaani":
        v = VaaniNet(**mc).eval()
    elif model_kind == cascade.MODEL_NAME:
        v = cascade.FrozenCascade(mc).eval()
    else:
        raise NotImplementedError(f"export supports config['model'] in ('vaani', 'vaani_cascade'), got {model_kind!r}")
    v.load_state_dict(ck["model"])
    return v, mc


def _stream_twin(v, mc):
    """Streaming module + its cache tuple + ONNX names; the cascade appends refine_cache to the existing signature."""
    if isinstance(v, cascade.FrozenCascade):
        s = cascade.StreamCascade(mc).eval()
        convert_to_stream(s.first, v.first); s.refiner.load_state_dict(v.refiner.state_dict())
        return s, cascade.init_cascade_caches(), IN_NAMES + ["refine_cache"], OUT_NAMES + ["refine_cache_out"]
    s = StreamVaaniNet(**mc).eval()
    convert_to_stream(s, v)  # load_state_dict fails: stream conv wrappers nest keys one level deeper
    return s, init_caches(), IN_NAMES, OUT_NAMES


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
    a = ap.parse_args()
    if a.out is None:   # the cascade never overwrites the shipped first-stage graph
        kind = torch.load(a.ckpt, map_location="cpu", weights_only=True)["config"]["model"]
        a.out = "deploy/tier46/cascade.onnx" if kind == cascade.MODEL_NAME else "deploy/model.onnx"
    p = export(a.ckpt, a.out)
    print(parity_and_timing(a.ckpt, p))
