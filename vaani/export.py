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
from vaani.models.vaani_net import StreamVaaniNet, VaaniNet, init_caches


def _load_batch_model(ckpt_path):
    """Load checkpoint into the batch VaaniNet; only 'vaani' configs are exportable."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_kind = ck["config"]["model"]
    if model_kind != "vaani":
        raise NotImplementedError(f"export only supports config['model'] == 'vaani', got {model_kind!r}")
    v = VaaniNet().eval()
    v.load_state_dict(ck["model"])
    return v


def export(ckpt_path, out_path):
    v = _load_batch_model(ckpt_path)
    s = StreamVaaniNet().eval()
    convert_to_stream(s, v)  # load_state_dict fails: stream conv wrappers nest keys one level deeper
    spec = torch.zeros(1, 257, 1, 6); f = torch.zeros(1, 1, 18); caches = init_caches()
    with warnings.catch_warnings():
        # Legacy TorchScript exporter warns on the GRU bool trace and Slice folding; shapes
        # are static batch-1 so the trace is exact, as proven by parity_and_timing.
        warnings.simplefilter("ignore")
        torch.onnx.export(s, (spec, f, *caches), str(out_path), opset_version=17,
                           input_names=["spec6", "feats", "conv_cache", "tra_cache", "inter_cache"],
                           output_names=["spec_out", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
                           dynamo=False)
    return Path(out_path)


def parity_and_timing(ckpt_path, onnx_path, seconds=10):
    v = _load_batch_model(ckpt_path)
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

    caches = [c.numpy() for c in init_caches()]
    outs, times = [], []
    for t in range(T):
        inp = {"spec6": spec[:, :, t:t + 1].numpy(), "feats": f[:, t:t + 1].numpy(),
               "conv_cache": caches[0], "tra_cache": caches[1], "inter_cache": caches[2]}
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
    ap.add_argument("ckpt"); ap.add_argument("--out", default="deploy/model.onnx")
    a = ap.parse_args()
    p = export(a.ckpt, a.out)
    print(parity_and_timing(a.ckpt, p))
