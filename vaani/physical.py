"""Physical-test scoring: real two-mic recordings through the board path, with or without a clean reference.

    python -m vaani.physical --dir data/manifests/physical_test --out results_r2/real/physical_scores.csv

Reads `<dir>/physical.csv` (docs/physical_test_schema.md), runs every clip through `vaani.live.StreamEngine` with
deploy/r7/cascade.onnx and its model_config.json - the same per-hop engine the Pi runs, not the offline torch path -
and writes one row per clip. Every row gets DNSMOS P.835 SIG/BAK/OVRL of input and output plus a level-based
attenuation proxy; rows with `has_clean` also get SNR/STOI/PESQ against `clips/<id>.clean.wav`.

The attenuation proxy is reference-free and crude: 20 ms frames whose input energy is within 30 dB of the clip's
99th-percentile frame count as active (speech or loud noise alike), and a frame is "attenuated" when the output is
more than 20 dB below the input there. A high fraction means whole stretches of the input vanished, which is what
the Pi web-WAV run showed; it cannot tell removed noise from removed speech.
"""
from __future__ import annotations

import argparse
import csv
import sys
from math import gcd
from pathlib import Path

import numpy as np

from vaani import live

REPO = Path(__file__).resolve().parents[1]
ONNX = REPO / "deploy/r7/cascade.onnx"
CONFIG = REPO / "deploy/r7/model_config.json"
HOP = live.HOP
SR = live.SR
CSV_COLUMNS = ["id", "speaker", "language", "condition", "transcript", "has_clean"]
CONDITIONS = {"engine", "engine+burst", "env_change", "ref_fault", "quiet"}
FRAME = 320            # 20 ms proxy frames
ACTIVE_DB = 30.0       # active = within this many dB of the clip's p99 frame energy
ATTEN_DB = 20.0        # attenuated = output this many dB below input
OUT_COLUMNS = ["id", "speaker", "language", "condition", "has_clean", "seconds",
               "dnsmos_in_sig", "dnsmos_in_bak", "dnsmos_in_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
               "active_frames", "atten20_frac", "mean_atten_db",
               "gate_mean", "burst_frac", "limiter_frac",
               "snr_in", "snr_out", "stoi_in", "stoi", "pesq_in", "pesq"]


def _truthy(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n", ""):
        return False
    raise ValueError(f"has_clean must be 0/1/true/false, got {v!r}")


def load_physical_csv(path) -> list[dict]:
    """Rows of physical.csv with `wav` and `clean` (None unless has_clean) paths under `<csv dir>/clips/`."""
    path = Path(path)
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        cols = [c.strip() for c in (rd.fieldnames or [])]
        missing = [c for c in CSV_COLUMNS if c not in cols]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; need {CSV_COLUMNS}")
        rows, seen = [], set()
        for i, r in enumerate(rd, start=2):
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k is not None}
            if not r["id"]:
                raise ValueError(f"{path}:{i}: empty id")
            if r["id"] in seen:
                raise ValueError(f"{path}:{i}: duplicate id {r['id']!r}")
            if r["condition"] not in CONDITIONS:
                raise ValueError(f"{path}:{i}: condition {r['condition']!r} not in {sorted(CONDITIONS)}")
            seen.add(r["id"])
            r["has_clean"] = _truthy(r["has_clean"])
            r["wav"] = path.parent / "clips" / f"{r['id']}.wav"
            r["clean"] = path.parent / "clips" / f"{r['id']}.clean.wav" if r["has_clean"] else None
            rows.append(r)
    return rows


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """(channels, n) at any rate -> 16 kHz (polyphase; the schema asks for 16 kHz, this only forgives 44.1/48)."""
    if sr == SR:
        return np.asarray(x, np.float32)
    from scipy.signal import resample_poly
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g, axis=-1).astype(np.float32)


def load_clip(row: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """(2, T) primary+reference at 16 kHz, and the (T,) clean primary or None."""
    x, sr = live.read_wav(row["wav"])
    if x.shape[0] != 2:
        raise ValueError(f"{row['wav']}: need 2 channels (primary, reference), got {x.shape[0]}")
    x = to_16k(x, sr)
    clean = None
    if row.get("clean") is not None:
        c, csr = live.read_wav(row["clean"])
        clean = to_16k(c, csr)[0]
        if abs(len(clean) - x.shape[1]) > 1:           # resampling may differ by one sample; more is a bad file
            raise ValueError(f"{row['clean']}: {len(clean)} samples vs {x.shape[1]} in the noisy clip")
        n = min(len(clean), x.shape[1]); clean, x = clean[:n], x[:, :n]
    return x, clean


def run_engine(mix: np.ndarray, onnx=ONNX, config=CONFIG, threads: int = 1, dsp: dict | None = None,
               trace: bool = False, ref_valid: bool = True, guards=None):
    """(2, T) at 16 kHz through a fresh StreamEngine hop by hop -> ((T,) output aligned to the input, diagnostics).

    The input is zero-padded by one hop so the engine's one-hop lag is flushed; the first output hop (the left
    context) is dropped. `dsp` overrides the config's DSP block (probe use only; None = the shipping config).
    trace=True also returns every hop's `eng.last` dict under diag["trace"] (validity-flag latency, guards).
    ref_valid=False is the reference-zeroed construction at validity 0: passed to process() when the engine takes a
    validity input (VaaniFE) or runs the trained reference policy (dsp.ref_policy: its DSP half treats the hop as
    absent); r7 has neither, so it keeps its default call and the caller's zeroed reference.
    guards: StreamEngine's `guards=` (None = off, the r7 default path)."""
    cfg = live.load_model_config(config)
    kw = {"guards": guards} if guards else {}
    eng = live.StreamEngine(onnx, cfg["controller_on"], cfg["dsp"] if dsp is None else dsp, threads=threads, **kw)
    pkw = {"ref_valid": False} if not ref_valid and (getattr(eng, "takes_valid", False) or getattr(eng, "pol", None) is not None) else {}
    mix = np.asarray(mix, np.float32)
    T = mix.shape[1]
    n = -(-T // HOP) + 1
    x = np.pad(mix, ((0, 0), (0, n * HOP - T)))
    ys, gate, burst, lim, tr = [], [], [], [], []
    for j in range(n):
        ys.append(eng.process(x[0, j * HOP:(j + 1) * HOP], x[1, j * HOP:(j + 1) * HOP], **pkw))
        d = eng.last or {}                             # diagnostics only; NaN if a refactor drops a key
        gate.append(d.get("gate", np.nan)); burst.append(d.get("burst", np.nan)); lim.append(d.get("limiter", np.nan))
        if trace:
            tr.append({k: v for k, v in d.items() if k != "stages"})
    y = np.concatenate(ys)[HOP:HOP + T]
    mean = lambda v: float(np.mean(np.asarray(v, float)))
    diag = {"gate_mean": mean(gate), "burst_frac": mean(burst), "limiter_frac": mean(lim)}
    if trace:
        diag["trace"] = tr                             # tr[j] was computed on input hop j (starts at j*HOP)
    return y, diag


_TAKES_VALID: dict = {}


def engine_takes_valid(onnx=ONNX, config=CONFIG) -> bool:
    """Whether the StreamEngine for this ONNX has a validity input (so ref_valid=False reaches the model)."""
    key = (str(onnx), str(config))
    if key not in _TAKES_VALID:
        cfg = live.load_model_config(config)
        _TAKES_VALID[key] = bool(live.StreamEngine(onnx, cfg["controller_on"], cfg["dsp"]).takes_valid)
    return _TAKES_VALID[key]


def _frames_db(v: np.ndarray, n: int = FRAME) -> np.ndarray:
    k = len(v) // n
    return 10 * np.log10((v[:k * n].reshape(k, n).astype(np.float64) ** 2).mean(1) + 1e-12)


def attenuation_proxy(x_in: np.ndarray, y: np.ndarray) -> dict:
    """Reference-free level proxy (see module doc): active frames, fraction attenuated >20 dB, mean attenuation."""
    n = min(len(x_in), len(y))
    ein, eout = _frames_db(x_in[:n]), _frames_db(y[:n])
    if len(ein) == 0:
        return {"active_frames": 0, "atten20_frac": float("nan"), "mean_atten_db": float("nan")}
    active = ein > np.percentile(ein, 99) - ACTIVE_DB
    k = int(active.sum())
    return {"active_frames": k,
            "atten20_frac": float(((eout < ein - ATTEN_DB) & active).sum() / max(k, 1)),
            "mean_atten_db": float((ein - eout)[active].mean()) if k else float("nan")}


_DN = None


def dnsmos(x: np.ndarray) -> dict:
    global _DN
    if _DN is None:
        from vaani.dnsmos import DNSMOS
        _DN = DNSMOS()
    return _DN(x)


def score(prim: np.ndarray, y: np.ndarray, clean: np.ndarray | None = None) -> dict:
    """Metrics for one output against its primary input (and the clean primary when there is one)."""
    di, do = dnsmos(prim), dnsmos(y)
    r = {"dnsmos_in_sig": di["sig"], "dnsmos_in_bak": di["bak"], "dnsmos_in_ovrl": di["ovrl"],
         "dnsmos_sig": do["sig"], "dnsmos_bak": do["bak"], "dnsmos_ovrl": do["ovrl"], **attenuation_proxy(prim, y)}
    nan = float("nan")
    r.update(snr_in=nan, snr_out=nan, stoi_in=nan, stoi=nan, pesq_in=nan, pesq=nan)
    if clean is not None:
        from vaani import metrics
        r.update(snr_in=metrics.snr_db(clean, prim), snr_out=metrics.snr_db(clean, y),
                 stoi_in=metrics.stoi(clean, prim), stoi=metrics.stoi(clean, y),
                 pesq_in=metrics.pesq_wb(clean, prim), pesq=metrics.pesq_wb(clean, y))
    return r


def score_dir(csv_path, out_csv, onnx=ONNX, config=CONFIG, threads: int = 1, save_wav=None) -> list[dict]:
    rows = load_physical_csv(csv_path)
    out = []
    for r in rows:
        mix, clean = load_clip(r)
        y, diag = run_engine(mix, onnx, config, threads)
        if save_wav:
            Path(save_wav).mkdir(parents=True, exist_ok=True)
            live.write_wav(Path(save_wav) / f"{r['id']}.enh.wav", y, SR)
        out.append({"id": r["id"], "speaker": r["speaker"], "language": r["language"], "condition": r["condition"],
                    "has_clean": int(r["has_clean"]), "seconds": mix.shape[1] / SR, **diag,
                    **score(mix[0], y, clean)})
        print(f"{r['id']}: OVRL {out[-1]['dnsmos_in_ovrl']:.2f} -> {out[-1]['dnsmos_ovrl']:.2f}, "
              f"atten>20dB {out[-1]['atten20_frac']:.2f}", file=sys.stderr, flush=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for o in out:
            w.writerow({k: (f"{o[k]:.4f}" if isinstance(o[k], float) else o[k]) for k in OUT_COLUMNS})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="data/manifests/physical_test", help="folder holding physical.csv and clips/")
    ap.add_argument("--csv", help="default: <dir>/physical.csv")
    ap.add_argument("--out", required=True, help="per-clip scores CSV")
    ap.add_argument("--onnx", default=str(ONNX))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--save-wav", help="also write <id>.enh.wav here")
    a = ap.parse_args(argv)
    score_dir(a.csv or Path(a.dir) / "physical.csv", a.out, a.onnx, a.config, a.threads, a.save_wav)


if __name__ == "__main__":
    main()
