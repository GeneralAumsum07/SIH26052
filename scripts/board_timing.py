"""Tier 3.1: per-hop timing on the actual target board - a thin wrapper over scripts/hop_benchmark.py (the complete
hop: limiter, blocking, NLMS, features, controller, STFT, ORT model, iSTFT) on ORT CPU, 1 thread, against the 16 ms
hop. Needs only numpy + onnxruntime (+ numba for the real NLMS cost; no torch: 512 MB board):

    python scripts/board_timing.py deploy/r7/cascade.onnx --seconds 30 --out deploy/board_timing.json

p99 < 12 ms -> stay on the board (plan 3.2); 12-16 -> stay, no on-device speech-preservation head; > 16 -> faster
board. Without numba the NLMS is the pure-Python golden reference, an order of magnitude slower than the numba/C
port, so the decision then rests on the model stage alone and says so.
"""
import argparse, json, platform, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hop_benchmark  # noqa: E402

HOP_MS = 16.0


def _board_model():
    # Raspberry Pi OS exposes the exact board here (e.g. "Raspberry Pi 5 Model B Rev 1.0"); absent elsewhere
    try: return Path("/proc/device-tree/model").read_text().rstrip(chr(0)).strip()
    except OSError: return platform.node()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="?", default=str(hop_benchmark.R7_ONNX))
    ap.add_argument("--config", default=str(hop_benchmark.R7_CONFIG))
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--resample48", action="store_true", help="include the 48 kHz resamplers (I2S capture path)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    full = hop_benchmark.run(["r7"], ["ort-cpu"], a.seconds, threads=a.threads, resample48=a.resample48,
                             config=a.config, onnx=None if Path(a.onnx).resolve() == hop_benchmark.R7_ONNX.resolve() else a.onnx,
                             label="board_timing", argv=["board_timing.py"] + list(sys.argv[1:] if argv is None else argv))
    row = full["rows"][0]; w = row["warm"]
    nlms_kind = full["versions"]["nlms_kernel"]
    r = {"host": platform.machine(), "board": _board_model(), "platform": platform.platform(),
         "python": platform.python_version(), "onnx": row["onnx"], "onnx_sha256": row["onnx_sha256"],
         "frames": w["hop"]["n"], "threads": a.threads, "dsp_nlms_kernel": nlms_kind,
         "model_ms_mean": w["model"]["mean_ms"], "model_ms_p99": w["model"]["p99_ms"], "model_ms_max": w["model"]["max_ms"],
         "hop_ms_mean": w["hop"]["mean_ms"], "hop_ms_p99": w["hop"]["p99_ms"], "hop_ms_max": w["hop"]["max_ms"],
         "hop_deadline_misses": w["hop"]["deadline_misses"], "hop_ms": HOP_MS,
         "rtf": w["hop"]["mean_ms"] / HOP_MS, "hop_benchmark": full}
    budget = r["hop_ms_p99"] if nlms_kind == "numba" else r["model_ms_p99"]
    r["decision_basis"] = "complete hop (numba kernel)" if nlms_kind == "numba" else "model only; dsp reference kernel excluded"
    r["board_decision_3_2"] = ("stay" if budget < 12 else
                               "stay, no on-device speech-preservation head" if budget < 16 else "faster board")
    s = json.dumps(r, indent=2)
    print(s)
    if a.out: Path(a.out).write_text(s + "\n")
    return r


if __name__ == "__main__":
    main()
