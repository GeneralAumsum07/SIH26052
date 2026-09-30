"""Score enhanced WAVs brought back from the Pi against their clean references (laptop; needs the training venv).

    python scripts/score_pi_outputs.py pi_results/audio --clips r8_runs_final/pi_bundle/clips --out pi_results/scores.csv

Expects <model>/<clip>.enh.wav under the first argument and <clip>.mix.wav + <clip>.clean.wav under --clips.
Clips without a clean reference (live recordings) are listed with input/output level only.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani import metrics  # noqa: E402


def db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="folder holding <model>/<clip>.enh.wav")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted(Path(a.audio).glob("*/*.enh.wav")):
        clip = f.name[: -len(".enh.wav")]
        y, sr = sf.read(f, dtype="float32")
        y = y[:, 0] if y.ndim > 1 else y
        row = {"model": f.parent.name, "clip": clip, "out_dbfs": round(db(y), 2)}
        mixp, cleanp = Path(a.clips) / f"{clip}.mix.wav", Path(a.clips) / f"{clip}.clean.wav"
        if mixp.exists() and cleanp.exists() and sr == 16000:
            prim = sf.read(mixp, dtype="float32")[0][:, 0]
            clean = sf.read(cleanp, dtype="float32")[0]
            n = min(len(y), len(clean), len(prim)); y, clean, prim = y[:n], clean[:n], prim[:n]
            row.update(snr_in=round(metrics.snr_db(clean, prim), 2), snr_out=round(metrics.snr_db(clean, y), 2),
                       stoi_in=round(metrics.stoi(clean, prim), 3), stoi_out=round(metrics.stoi(clean, y), 3),
                       pesq_in=round(metrics.pesq_wb(clean, prim), 2), pesq_out=round(metrics.pesq_wb(clean, y), 2))
        rows.append(row)
    keys = ["model", "clip", "snr_in", "snr_out", "stoi_in", "stoi_out", "pesq_in", "pesq_out", "out_dbfs"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={r.get(k, '-')}" for k in keys))
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
