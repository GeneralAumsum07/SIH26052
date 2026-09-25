"""M6 receiver-radius smoke: per-room cost and RIR statistics at 0.3 / 0.06 / 0.05 m on bank_r3's own armoured rooms.
Run from the repo root: .venv/Scripts/python.exe results_r2/r8/banks/bench_radius.py --rooms 5 --workers 5"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SR = 16000
MAX_LEN = 16000   # bank_r3 --max-len-s 1.0


def bank_r3_params(n=5000, seed=0, n_noise=3, frac=0.2):
    # the exact draw sequence build_bank made for bank_r3 (legacy stream, seed 0)
    from vaani.data import rirs
    rng = rirs.bank_rng(seed)
    arm = np.zeros(n, bool); arm[:int(round(n * frac))] = True; rng.shuffle(arm)
    return arm, [rirs.draw_room_params(rng, n_noise, bool(a)) for a in arm]


def schroeder_t(h, lo, hi):
    # T-lo..hi fitted on the backward-integrated decay, extrapolated to 60 dB
    e = np.cumsum(h[::-1].astype(np.float64) ** 2)[::-1]; e = 10 * np.log10(e / e[0] + 1e-30)
    i0, i1 = np.argmax(e <= -lo), np.argmax(e <= -hi)
    if i1 <= i0: return float("nan")
    t = np.arange(i0, i1) / SR; k = np.polyfit(t, e[i0:i1], 1)[0]
    return float(-60.0 / k)


def stats(pair):
    # per-mic DRR / late energy, and the inter-mic similarity of the late tail the ray tracer supplies
    out = {}
    env = []
    for m in range(2):
        h = pair[m].astype(np.float64); pk = int(np.argmax(np.abs(h)))
        d = int(pk + 0.0025 * SR); late0 = int(0.05 * SR)
        e_dir, e_rest, e_late = (h[:d] ** 2).sum(), (h[d:] ** 2).sum(), (h[late0:] ** 2).sum()
        out[f"m{m}_drr_db"] = float(10 * np.log10(e_dir / max(e_rest, 1e-30)))
        out[f"m{m}_late_db"] = float(10 * np.log10(max(e_late, 1e-30)))
        out[f"m{m}_t20"] = schroeder_t(h, 5, 25); out[f"m{m}_t30"] = schroeder_t(h, 5, 35)
        w = int(0.01 * SR); seg = h[late0:late0 + (int(0.5 * SR) // w) * w].reshape(-1, w)
        env.append(10 * np.log10((seg ** 2).mean(1) + 1e-30))
    out["late_env_corr"] = float(np.corrcoef(env[0], env[1])[0, 1])       # 10 ms energy envelope, 50-550 ms
    out["late_env_rms_diff_db"] = float(np.sqrt(np.mean((env[0] - env[1]) ** 2)))
    a, b = pair[0, int(0.05 * SR):].astype(np.float64), pair[1, int(0.05 * SR):].astype(np.float64)
    out["late_wave_corr"] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))
    return out


def run(task):
    idx, rr, prm = task
    from vaani.data import rirs
    t = time.perf_counter()
    s = rirs.simulate_from_params(prm, max_len=MAX_LEN, receiver_radius=rr)
    dt = time.perf_counter() - t
    rec = {"idx": idx, "radius": rr, "sec": dt, "rt60_drawn": prm["rt60"], "dims": [float(x) for x in prm["dims"]]}
    rec.update({f"speech_{k}": v for k, v in stats(s["speech"]).items()})
    rec.update({f"noise0_{k}": v for k, v in stats(s["noise"][0]).items()})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=5)
    ap.add_argument("--plain", type=int, default=20, help="non-armoured rooms to time (legacy settings)")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.3, 0.06, 0.05])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results_r2/r8/banks/bench_radius.json")
    a = ap.parse_args()
    arm, params = bank_r3_params()
    ai = [int(i) for i in np.flatnonzero(arm)[:a.rooms]]
    pi = [int(i) for i in np.flatnonzero(~arm)[:a.plain]]
    # the re-derived draws must reproduce the stored bank_r3 rooms (non-armoured ISM is deterministic on one machine)
    from vaani.data import rirs
    sp = np.load("data/rirs/bank_r3.speech.npy", mmap_mode="r"); nz = np.load("data/rirs/bank_r3.noise.npy", mmap_mode="r")
    rt = np.load("data/rirs/bank_r3.rt60.npy")
    check = []
    for i in pi[:3]:
        s = rirs.simulate_from_params(params[i], max_len=MAX_LEN)
        check.append({"idx": i, "speech_maxdiff": float(np.abs(s["speech"] - sp[i]).max()),
                      "noise_maxdiff": float(np.abs(s["noise"] - nz[i]).max()), "rt60_equal": bool(np.float32(s["rt60"]) == rt[i])})
    # every drawn RT60 (armoured rooms included) against the stored column
    check.append({"all_rt60_equal": bool(np.array_equal(np.array([p["rt60"] for p in params], np.float32), rt)),
                  "armoured_equal": bool(np.array_equal(arm, np.load("data/rirs/bank_r3.npz")["armoured"]))})
    stored = []
    for i in ai:   # the realisation actually stored in bank_r3 (legacy radius) for the same rooms
        rec = {"idx": i}; rec.update({f"speech_{k}": v for k, v in stats(np.asarray(sp[i])).items()})
        rec.update({f"noise0_{k}": v for k, v in stats(np.asarray(nz[i, 0])).items()}); stored.append(rec)
    os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")})
    from multiprocessing import get_context
    tasks = [(i, rr, params[i]) for rr in a.radii for i in ai] + [(i, None, params[i]) for i in pi]
    t0 = time.perf_counter()
    with get_context("spawn").Pool(a.workers) as pool:
        recs = pool.map(run, tasks, chunksize=1)
    wall = time.perf_counter() - t0
    res = {"note": "SMOKE, non-reportable timing: shared 16-core laptop with other jobs running",
           "workers": a.workers, "wall_s": wall, "rederive_check": check, "stored_bank_r3": stored,
           "armoured": [r for r in recs if r["radius"] is not None], "plain": [r for r in recs if r["radius"] is None]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for rr in a.radii:
        rs = [r for r in recs if r["radius"] == rr]
        print(f"r={rr}: sec {np.mean([r['sec'] for r in rs]):.1f}  t30 {np.nanmean([r['speech_m1_t30'] for r in rs]):.3f}"
              f"  late_env_corr {np.mean([r['noise0_late_env_corr'] for r in rs]):.3f}")
    print("plain sec", np.mean([r["sec"] for r in recs if r["radius"] is None]), "check", check)


if __name__ == "__main__":
    main()
