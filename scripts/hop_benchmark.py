"""Complete-hop benchmark: the whole per-16 ms-hop path of the deployed system, stage by stage, per backend and model.

Every hop runs `vaani.live.StreamEngine.process` exactly as the live loop does (limiter, blocking matrix, NLMS
(numba kernel when installed), 18 features, controller, STFT, model step, iSTFT + overlap-add), optionally wrapped in
the 48 kHz streaming resamplers (--resample48: 3:1 decimation before, 1:3 interpolation after). Nothing is left out
of "hop": the number is what a capture loop pays per period, minus ALSA I/O.

    # board (numpy + onnxruntime only, no torch): the shipped r7 graph on ORT CPU
    python scripts/hop_benchmark.py --models r7 --backends ort-cpu --seconds 30 --out hop.json

    # dev machine: r7 plus untrained C16/C32/C64/C96 cascades on every backend present
    python scripts/hop_benchmark.py --models r7,C16,C32,C64,C96 --backends ort-cpu,torch-cpu,torch-cuda,ort-cuda

Models. r7 = deploy/r7/cascade.onnx (ORT) and its cascade checkpoint (Torch twin). C<w> = an UNTRAINED cascade at
first-stage width w with the profile settings of docs/research/2026-09-24/profile_scaling.py (film off, coh on,
df_order 3, no noise floor, refiner hidden 16 / past 2), written as a scratch checkpoint and exported with
`vaani.export.export` into --scratch: cost only, never quality. Every row runs behind r7's DSP configuration.

Cold vs warm. "cold" is the first --cold hops of a fresh engine (numba compile if not cached, ORT/torch first-run
allocation, cuDNN autotune); "warm" is every hop after --warm discarded ones. Per stage and for the whole hop:
mean / p50 / p95 / p99 / max, plus deadline misses against 16 ms. The JSON records full provenance (git commit and
dirty state, host, CPU, library versions, NLMS kernel, graph sha256, command line). Numbers from a shared or loaded
machine are not reportable; --label says which it was.

ORT-CUDA / TensorRT rows go through OrtBackend's I/O-binding path, which has not been run on this machine's CPU-only
onnxruntime: such a row is skipped (and says so) when the provider is absent.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani import backend as bk   # noqa: E402  (numpy-only at import; torch is imported only for torch/C<w> rows)
from vaani import live            # noqa: E402

HOP = live.HOP
R7_ONNX = ROOT / "deploy" / "r7" / "cascade.onnx"
R7_CONFIG = ROOT / "deploy" / "r7" / "model_config.json"
R7_CKPT = ROOT / "results_r2" / "runs" / "r7_e256_wr64_refiner" / "best.pt"
PROFILE = {"film": False, "coh": True, "df_order": 3, "noise_floor": False}   # profile_scaling.py
REFINER = {"hidden": 16, "past": 2}
STAGES = ("decimate", "limiter", "blocking", "nlms", "stft", "features", "controller", "model", "istft", "interpolate")
QS = (50, 95, 99)


def stats(x, deadline_ms=None) -> dict:
    x = np.asarray(x, np.float64)
    if not len(x):
        return {"n": 0}
    r = {"n": int(len(x)), "mean_ms": float(x.mean()), **{f"p{q}_ms": float(np.percentile(x, q)) for q in QS},
         "max_ms": float(x.max())}
    if deadline_ms is not None:
        r["deadline_ms"] = deadline_ms; r["deadline_misses"] = int((x > deadline_ms).sum())
    return r


def test_signal(seconds: float, seed: int = 0) -> np.ndarray:
    """Speech-like primary (gated harmonics), correlated noise on both mics, periodic loud bursts so the limiter,
    blocking matrix and controller burst paths all run. (2, n) float32 at 16 kHz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * live.SR); t = np.arange(n) / live.SR
    v = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 12)) * 0.08 * (np.sin(2 * np.pi * 2.5 * t) > 0)
    noise = rng.standard_normal(n) * 0.03
    prim, ref = v + noise, 0.3 * v + np.roll(noise, 3)
    for s in range(live.SR // 2, n - 400, 2 * live.SR):
        b = rng.standard_normal(400); prim[s:s + 400] += b; ref[s:s + 400] += b
    return np.stack([prim, ref]).astype(np.float32)


def git_provenance() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    return {"commit": g("rev-parse", "HEAD"), "dirty_files": len((g("status", "--porcelain") or "").splitlines())}


def untrained_checkpoint(width: int, scratch: Path, dsp_cfg: dict, seed: int) -> Path:
    """Scratch FrozenCascade checkpoint at `width` (random weights, cost only)."""
    import torch
    from vaani.models import cascade
    torch.manual_seed(seed)
    mc = {"channels": width, **PROFILE}
    m = cascade.FrozenCascade(mc, REFINER).eval()
    ck = scratch / f"C{width}_untrained.pt"
    torch.save({"model": m.state_dict(), "step": 0,
                "config": {"model": cascade.MODEL_NAME, "model_cfg": mc, "refiner_cfg": dict(REFINER),
                           "controller_on": dsp_cfg["controller_on"], "dsp": dsp_cfg["dsp"],
                           "note": "untrained, hop_benchmark cost measurement only"}}, ck)
    return ck


def model_sources(name: str, scratch: Path, dsp_cfg: dict, seed: int, need_torch: bool) -> dict:
    """-> {"onnx": path or None, "ckpt": path or None, "trained": bool}. ONNX for a C<w> row via vaani.export.export."""
    if name == "r7":
        return {"onnx": R7_ONNX, "ckpt": R7_CKPT if R7_CKPT.exists() else None, "trained": True}
    if not name.upper().startswith("C"):
        raise SystemExit(f"unknown model {name!r}: r7 or C<width>")
    from vaani import export
    ck = untrained_checkpoint(int(name[1:]), scratch, dsp_cfg, seed)
    onnx = scratch / f"{name.upper()}_untrained.onnx"
    if not onnx.exists():
        export.export(ck, onnx)
    return {"onnx": onnx, "ckpt": ck, "trained": False}


def make_backend(kind: str, src: dict, threads: int, profile_id: str):
    """-> (backend, None) or (None, reason it was skipped)."""
    if kind == "ort-cpu":
        return bk.OrtBackend(src["onnx"], threads=threads, profile_id=profile_id), None
    if kind in ("ort-cuda", "ort-trt"):
        import onnxruntime as ort
        prov = "CUDAExecutionProvider" if kind == "ort-cuda" else "TensorrtExecutionProvider"
        if prov not in ort.get_available_providers():
            return None, f"{prov} not in this onnxruntime ({ort.get_available_providers()})"
        return bk.OrtBackend(src["onnx"], threads=threads, providers=[prov, "CPUExecutionProvider"],
                             profile_id=profile_id), None
    if kind in ("torch-cpu", "torch-cuda"):
        import torch
        dev = kind.split("-")[1]
        if dev == "cuda" and not torch.cuda.is_available():
            return None, "torch.cuda.is_available() is False"
        if src["ckpt"] is None:
            return None, "no checkpoint for the Torch twin"
        torch.set_num_threads(threads)
        return bk.TorchBackend.from_checkpoint(src["ckpt"], device=dev, profile_id=profile_id), None
    raise SystemExit(f"unknown backend {kind!r}")


def bench_one(backend, dsp_cfg: dict, mix: np.ndarray, warm: int, cold: int, resample48: bool) -> dict:
    """Run the whole clip hop by hop through a fresh engine; per-hop stage times -> cold / warm stats."""
    eng = live.StreamEngine(None, dsp_cfg["controller_on"], dsp_cfg["dsp"], backend=backend, stage_timing=True)
    dec = live.Decimate3(2) if resample48 else None
    itp = live.Interpolate3(1) if resample48 else None
    x = mix
    if resample48:           # the benchmark input is 16 kHz; a 48 kHz capture is its 3x zero-order stand-in
        x = np.repeat(mix, 3, axis=1)
    blk = HOP * (3 if resample48 else 1)
    rows = []
    for j in range(x.shape[1] // blk):
        b = x[:, j * blk:(j + 1) * blk]
        t0 = time.perf_counter()
        b16 = dec(b) if dec else b
        t1 = time.perf_counter()
        y = eng.process(b16[0], b16[1])
        t2 = time.perf_counter()
        if itp:
            itp(y[None])
        t3 = time.perf_counter()
        st = dict(eng.last["stages"])
        st["decimate"] = (t1 - t0) * 1000 if dec else 0.0
        st["interpolate"] = (t3 - t2) * 1000 if itp else 0.0
        st["hop"] = (t3 - t0) * 1000
        rows.append(st)
    keys = [s for s in STAGES if resample48 or s not in ("decimate", "interpolate")] + ["hop"]

    def block(rs):
        return {k: stats([r[k] for r in rs], bk.DEADLINE_MS if k in ("hop", "model") else None) for k in keys}
    return {"hops": len(rows), "cold": block(rows[:cold]), "warm": block(rows[warm:]),
            "first_hop_ms": rows[0]["hop"] if rows else None, "backend_tested_here": bool(getattr(backend, "tested", True))}


def versions(torch_used: bool) -> dict:
    import onnxruntime as ort
    from vaani.dsp import nlms
    v = {"python": platform.python_version(), "numpy": np.__version__, "onnxruntime": ort.__version__,
         "ort_providers": ort.get_available_providers(), "nlms_kernel": "numba" if nlms._HAVE_NUMBA else "pure-python"}
    try:
        import numba
        v["numba"] = numba.__version__
    except ImportError:
        v["numba"] = None
    if torch_used:
        import torch
        v["torch"] = torch.__version__
        v["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return v


def run(models=("r7",), backends=("ort-cpu",), seconds=10.0, warm=50, cold=10, threads=1, resample48=False,
        scratch=None, config=R7_CONFIG, onnx=None, seed=0, label="", argv=None) -> dict:
    dsp_cfg = live.load_model_config(config)
    scratch = Path(scratch or (ROOT / "runs" / "hop_benchmark_scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    mix = test_signal(seconds, seed)
    torch_used = any(b.startswith("torch") for b in backends) or any(m != "r7" for m in models)
    out = {"schema": "vaani.hop_benchmark/1", "label": label, "command": argv,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_provenance(),
           "host": {"machine": platform.machine(), "platform": platform.platform(), "processor": platform.processor(),
                    "node": platform.node()},
           "versions": versions(torch_used), "config": str(Path(config).as_posix()),
           "controller_on": dsp_cfg["controller_on"], "dsp": dsp_cfg["dsp"],
           "signal": {"seconds": seconds, "seed": seed, "kind": "hop_benchmark.test_signal (synthetic)"},
           "warm_discard": warm, "cold_hops": cold, "threads": threads, "resample48": resample48,
           "deadline_ms": bk.DEADLINE_MS, "rows": [], "skipped": []}
    for m in models:
        src = model_sources(m, scratch, dsp_cfg, seed, torch_used)
        if m == "r7" and onnx is not None:
            src["onnx"] = Path(onnx)
        elif m == "r7":                          # the shipped graph must be the one its DSP config names
            live.verify_onnx(src["onnx"], dsp_cfg.get("onnx_sha256"))
        for b in backends:
            backend, why = make_backend(b, src, threads, f"{m}")
            if backend is None:
                out["skipped"].append({"model": m, "backend": b, "reason": why}); print(f"skip {m}/{b}: {why}", file=sys.stderr)
                continue
            r = bench_one(backend, dsp_cfg, mix, warm, cold, resample48)
            r.update(model=m, backend=b, trained=src["trained"],
                     onnx=str(Path(src["onnx"]).as_posix()) if b.startswith("ort") else None,
                     onnx_sha256=bk.file_sha256(src["onnx"]) if b.startswith("ort") else None,
                     ckpt=str(Path(src["ckpt"]).as_posix()) if b.startswith("torch") and src["ckpt"] else None)
            out["rows"].append(r)
            w = r["warm"]["hop"]
            print(f"{m:4s} {b:10s} warm hop p50 {w.get('p50_ms', float('nan')):6.2f} p99 {w.get('p99_ms', float('nan')):6.2f} "
                  f"max {w.get('max_ms', float('nan')):6.2f} ms, misses {w.get('deadline_misses')}/{w.get('n')}; "
                  f"model p99 {r['warm']['model'].get('p99_ms', float('nan')):.2f}; first hop {r['first_hop_ms']:.1f} ms",
                  file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="r7", help="comma list: r7, C16, C32, C64, C96 (C<w> are untrained)")
    ap.add_argument("--backends", default="ort-cpu", help="comma list: ort-cpu, torch-cpu, torch-cuda, ort-cuda, ort-trt")
    ap.add_argument("--onnx", default=None, help="override the r7 graph (default deploy/r7/cascade.onnx)")
    ap.add_argument("--config", default=str(R7_CONFIG), help="model_config.json whose DSP settings every row runs behind")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--warm", type=int, default=50, help="hops discarded before the warm statistics")
    ap.add_argument("--cold", type=int, default=10, help="first hops reported as cold")
    ap.add_argument("--threads", type=int, default=1, help="ORT intra-op / torch threads (contract budget: 1)")
    ap.add_argument("--resample48", action="store_true", help="include the 48 kHz streaming resamplers in the hop")
    ap.add_argument("--scratch", default=None, help="where untrained C<w> checkpoints/graphs go (default runs/hop_benchmark_scratch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="", help="e.g. 'smoke, shared loaded machine: non-reportable'")
    ap.add_argument("--out", default=None, help="JSON output path")
    a = ap.parse_args(argv)
    r = run([m.strip() for m in a.models.split(",") if m.strip()], [b.strip() for b in a.backends.split(",") if b.strip()],
            a.seconds, a.warm, a.cold, a.threads, a.resample48, a.scratch, a.config, a.onnx, a.seed, a.label,
            argv=[Path(sys.argv[0]).name] + list(sys.argv[1:] if argv is None else argv))
    s = json.dumps(r, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(s + "\n", encoding="utf-8")
    else:
        print(s)
    return r


if __name__ == "__main__":
    main()
