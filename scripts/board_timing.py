"""Tier 3.1: per-frame timing on the actual target board (Zero 2 W) - model (ONNX Runtime, 1 thread) plus the
numpy DSP front end, against the 16 ms hop. Needs only numpy + onnxruntime (no torch: 512 MB board), so
copy the repo, `pip install numpy onnxruntime`, and run:

    python scripts/board_timing.py deploy/model.onnx --seconds 30 --out deploy/board_timing.json

The DSP number is the Python reference (`vaani.dsp.pipeline.run`, pure-Python NLMS when numba is absent),
which is an upper bound on the C port; the model number is the real ORT cost. p99 < 12 ms -> stay on the
Zero 2 W (plan 3.2); 12-16 -> stay, no on-device speech-preservation head; > 16 -> faster board.
"""
import argparse, json, platform, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani.dsp import pipeline, stft   # noqa: E402  (numpy-only after the lazy torch import)

HOP_MS = stft.HOP / 16.0


def _board_model():
    # Raspberry Pi OS exposes the exact board here (e.g. "Raspberry Pi 5 Model B Rev 1.0"); absent elsewhere
    try: return Path("/proc/device-tree/model").read_text().rstrip(chr(0)).strip()
    except OSError: return platform.node()


def time_model(onnx_path, n_frames, threads=1):
    import onnxruntime as ort
    opts = ort.SessionOptions(); opts.intra_op_num_threads = threads; opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    # cache shapes come from the model itself so this stays valid if init_caches changes
    ins = {i.name: i for i in sess.get_inputs()}
    caches = {n: np.zeros([d if isinstance(d, int) else 1 for d in ins[n].shape], np.float32)
              for n in ins if n.endswith("_cache")}
    rng = np.random.default_rng(0); times = []
    for t in range(n_frames):
        inp = {"spec6": (rng.standard_normal((1, 257, 1, 6)) * 0.1).astype(np.float32),
               "feats": rng.standard_normal((1, 1, 18)).astype(np.float32), **caches}
        t0 = time.perf_counter(); o = sess.run(None, inp); dt = (time.perf_counter() - t0) * 1000
        if t > 0: times.append(dt)   # frame 0 is allocator warm-up
        caches = dict(zip([n for n in ins if n.endswith("_cache")], o[1:]))
    return np.array(times)


def time_dsp(seconds, reps=3):
    # the reference runs whole-clip; per-frame cost = wall / frames, taken over a few reps for the best (steady) run
    rng = np.random.default_rng(0)
    mix = (rng.standard_normal((2, int(seconds * 16000))) * 0.05).astype(np.float32)
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); out = pipeline.run(mix, dsp_cfg={"limiter": True}); best = min(best, time.perf_counter() - t0)
    return best * 1000 / out["features"].shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?", default="deploy/model.onnx")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    n = int(a.seconds * 16000 / stft.HOP)
    m = time_model(a.onnx, n, a.threads)
    d = time_dsp(a.seconds)
    from vaani.dsp import nlms; nlms_kind = "numba" if nlms._HAVE_NUMBA else "pure-python"
    r = {"host": platform.machine(), "board": _board_model(), "platform": platform.platform(), "python": platform.python_version(),
         "onnx": str(a.onnx), "frames": int(len(m)), "threads": a.threads,
         "model_ms_mean": float(m.mean()), "model_ms_p99": float(np.percentile(m, 99)), "model_ms_max": float(m.max()),
         "dsp_ms_per_frame": float(d), "dsp_nlms_kernel": nlms_kind,
         "total_ms_p99": float(np.percentile(m, 99) + d), "hop_ms": HOP_MS,
         "rtf": float((m.mean() + d) / HOP_MS)}
    # the pure-Python NLMS is the golden reference, not the port: an order of magnitude slower than numba/C on the
    # same core, so it must not drive the board call. Then the decision rests on the model alone and says so.
    budget = r["total_ms_p99"] if nlms_kind == "numba" else r["model_ms_p99"]
    r["decision_basis"] = "model + dsp (numba kernel)" if nlms_kind == "numba" else "model only; dsp reference kernel excluded"
    r["board_decision_3_2"] = ("stay" if budget < 12 else
                               "stay, no on-device speech-preservation head" if budget < 16 else "faster board")
    print(json.dumps(r, indent=2))
    if a.out: Path(a.out).write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
