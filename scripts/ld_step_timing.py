"""Gate 0a Python step timing for low-delay VaaniFE step graphs (low-delay plan Task 0). Torch-free: NumPy and ONNX
Runtime through vaani.backend.FeOrtBackend, so it runs on the 512 MB board. The C++ loop (native/vaani_ld
vld_step_bench) is the primary timing; this loop is the Python comparison the plan asks for.

    python scripts/ld_step_timing.py runs/fe_tiers_ld/mini_p18__vaanife_ld_asym512_h96_s160_v1.onnx \
        --seconds 600 --out results_r2/r8_ld/gate0/py_step_mini_p18.json

Each graph is paced at its contract's hop rate (166.67 steps/s at H = 96, 125 at H = 128) on ORT CPU with one
thread, after a warmup, for each input (random, silent, very-low-level). An FFT/OLA stub stands in for the transform:
one 512-point rfft of the analysis frame and one irfft plus overlap-add per hop. The step and the whole stubbed hop
are timed separately; maximum, p99.9 and p99 are exact over every hop. Python cannot set FPCR.FZ, so there is no FZ
axis here (the ORT denormal option only acts on x86). Late hops: the stubbed hop's wake-to-finish time exceeding H.
"""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani import audio_contract as ac  # noqa: E402
from vaani.backend import FeOrtBackend  # noqa: E402

INPUTS = ("random", "silent", "lowlevel")
LEVEL = {"random": 0.1, "silent": 0.0, "lowlevel": 1e-6}     # waveform RMS before the transform stub


def _stats(ms: np.ndarray) -> dict:
    ms = np.asarray(ms, np.float64)
    return {"n": int(ms.size), "max_ms": float(ms.max()), "p999_ms": float(np.quantile(ms, 0.999)),
            "p99_ms": float(np.quantile(ms, 0.99)), "mean_ms": float(ms.mean())}


def _board() -> str:
    try: return Path("/proc/device-tree/model").read_text().rstrip(chr(0)).strip()
    except OSError: return platform.node()


def _temp_c():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError): return None


class TransformStub:
    """The FFT/OLA cost of one hop at K = 512: the analysis rfft of a windowed frame and the synthesis irfft plus a
    hop-length overlap-add. Buffers are preallocated; it is a cost stand-in, not the low-delay transform."""

    def __init__(self, k: int, hop: int, rng: np.random.Generator):
        self.k, self.hop = k, hop
        self.win = np.hanning(k).astype(np.float32)
        self.frame = np.zeros(k, np.float32)
        self.ola = np.zeros(k, np.float32)
        self.rng = rng

    def analysis(self, level: float) -> np.ndarray:
        self.frame[:-self.hop] = self.frame[self.hop:]
        self.frame[-self.hop:] = level * self.rng.standard_normal(self.hop).astype(np.float32)
        return np.fft.rfft(self.frame * self.win)

    def synthesis(self, y: np.ndarray) -> np.ndarray:
        x = np.fft.irfft(y, self.k).astype(np.float32)
        self.ola[:-self.hop] = self.ola[self.hop:]
        self.ola[-self.hop:] = 0.0
        self.ola += x * self.win
        return self.ola[:self.hop]


def time_graph(onnx_path, seconds: float, inputs=INPUTS, warmup: int = 500, threads: int = 1, paced: bool = True,
               seed: int = 0) -> dict:
    be = FeOrtBackend(onnx_path, threads=threads)
    c = be.audio_contract
    hop_s = c.hop / c.sr
    hops = max(1, int(round(seconds / hop_s)))
    runs = []
    for inp in inputs:
        rng = np.random.default_rng(seed)
        stub = TransformStub(512, c.hop, rng)
        state = be.new_state()
        spec6 = np.zeros((1, 257, 1, 6), np.float32)
        step_ms = np.empty(hops); hop_ms = np.empty(hops); late = 0
        t_start = _temp_c()
        for i in range(warmup + hops):
            if i == warmup:
                state = be.reset(state)
                t0 = time.perf_counter()
            if paced and i >= warmup:                     # wake at hop i's capture-complete instant
                due = t0 + (i - warmup) * hop_s
                while True:
                    d = due - time.perf_counter()
                    if d <= 0: break
                    time.sleep(d if d > 5e-4 else 0)
            w = time.perf_counter()
            for ch in range(be.n_raw // 2):               # one analysis per raw input plane (primary, reference)
                X = stub.analysis(LEVEL[inp])
                spec6[0, :, 0, 2 * ch] = X.real; spec6[0, :, 0, 2 * ch + 1] = X.imag
            s0 = time.perf_counter()
            y = be.step(spec6, None, state, 1.0)
            s1 = time.perf_counter()
            stub.synthesis(y[0, :, 0, 0] + 1j * y[0, :, 0, 1])
            e = time.perf_counter()
            if i >= warmup:
                j = i - warmup
                step_ms[j] = 1e3 * (s1 - s0); hop_ms[j] = 1e3 * (e - w)
                late += (e - w) > hop_s
        runs.append({"input": inp, "fz": None, "paced": paced, "hops": hops, "warmup": warmup,
                     "step": _stats(step_ms), "whole_hop": _stats(hop_ms), "late_hops": int(late),
                     "temp_c_start": t_start, "temp_c_end": _temp_c()})
    p = Path(onnx_path)
    return {"onnx": str(p), "onnx_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "audio_contract": c.audio_contract_id, "profile_id": be.profile_id, "state_floats": be.state_floats,
            "hop_ms": 1e3 * hop_s, "threads": threads, "runs": runs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--seconds", type=float, default=600.0, help="per input; the plan asks for >= 10 minutes")
    ap.add_argument("--inputs", default=",".join(INPUTS))
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--unpaced", action="store_true", help="back-to-back steps (cost only, not a Gate 0a figure)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    inputs = tuple(s for s in a.inputs.split(",") if s)
    bad = set(inputs) - set(INPUTS)
    if bad:
        ap.error(f"unknown inputs {sorted(bad)}")
    import onnxruntime as ort
    rep = {"tool": "scripts/ld_step_timing.py", "board": _board(), "machine": platform.machine(),
           "platform": platform.platform(), "python": platform.python_version(), "onnxruntime": ort.__version__,
           "seconds_per_input": a.seconds, "reportable": a.seconds >= 600 and not a.unpaced,
           "graphs": [time_graph(p, a.seconds, inputs, a.warmup, a.threads, not a.unpaced, a.seed) for p in a.onnx]}
    s = json.dumps(rep, indent=2)
    print(s)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(s + "\n", encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()
