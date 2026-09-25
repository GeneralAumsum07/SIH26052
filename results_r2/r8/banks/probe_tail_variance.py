"""Run-to-run spread of one armoured room (bank_r3 index 7) at a fixed radius: pra's ray-traced tail is a random
sequence from the global np.random, so repeated simulations of the same room differ. Run from the repo root:
.venv/Scripts/python.exe results_r2/r8/banks/probe_tail_variance.py --radius 0.3 --reps 3"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, ".")
from bench_radius import bank_r3_params  # noqa: E402
from vaani.data import rirs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=7)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results_r2/r8/banks/probe_tail_variance.json")
    a = ap.parse_args()
    _, params = bank_r3_params()
    out = []
    for rep in range(a.reps):
        s = rirs.simulate_from_params(params[a.idx], max_len=16000, receiver_radius=a.radius)
        for name, h in (("speech_ref", s["speech"][1]), ("noise0_primary", s["noise"][0][0])):
            e = h.astype(np.float64) ** 2
            out.append({"rep": rep, "rir": name,
                        "late_from_50ms_db": float(10 * np.log10(e[800:].sum())),
                        "late_from_100ms_db": float(10 * np.log10(e[1600:].sum())),
                        "top1pct_share_of_late": float(np.sort(e[800:])[-len(e[800:]) // 100:].sum() / e[800:].sum()),
                        "env_100ms_db": [float(10 * np.log10(e[i:i + 1600].sum() + 1e-30)) for i in range(0, 16000, 1600)]})
            print(out[-1]["rep"], name, round(out[-1]["late_from_50ms_db"], 2), round(out[-1]["late_from_100ms_db"], 2))
    Path(a.out).write_text(json.dumps({"idx": a.idx, "radius": a.radius, "runs": out}, indent=1))


if __name__ == "__main__":
    main()
