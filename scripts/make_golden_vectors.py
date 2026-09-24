"""Golden vectors for the DSP port. Deterministic inputs -> expected outputs.

Short (0.5 s) clips + compressed npz so this stays committed under deploy/
(the DSP lead ports features.py/controller.py/nlms.py to C and diffs
against these) instead of the git-ignored /data/ tree.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import soundfile as sf

from vaani.dsp import pipeline

R7_CKPT, R7_ONNX = "results_r2/runs/r7_e256_wr64_refiner/best.pt", "deploy/r7/cascade.onnx"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None  # a tarball without .git still regenerates; provenance says so


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--checkpoint", help=f"Read controller/DSP configuration from the deployed checkpoint (shipping: {R7_CKPT})")
ap.add_argument("--onnx", help=f"Record the graph this checkpoint was exported to (shipping: {R7_ONNX})")
ap.add_argument("--out", help="Separate output directory; legacy vectors are preserved by default")
args = ap.parse_args()
cfg = {}
if args.checkpoint:
    import torch
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"]
    cfg = {"controller_on": ck.get("controller_on", True), "dsp": ck.get("dsp", {}),
           "checkpoint": Path(args.checkpoint).as_posix(), "checkpoint_sha256": _sha(args.checkpoint)}
    if args.onnx:
        cfg.update(onnx=Path(args.onnx).as_posix(), onnx_sha256=_sha(args.onnx))
    status = _git("status", "--porcelain", "--", "vaani/dsp")
    # dirty counts only the code that computes the arrays; the inputs are the committed wavs themselves
    cfg.update(git_sha=_git("rev-parse", "HEAD"), git_dirty_dsp=None if status is None else bool(status))
out = Path(args.out or ("deploy/dsp_reference/vectors_cascade" if args.checkpoint else "deploy/dsp_reference/vectors"))
out.mkdir(parents=True, exist_ok=True)
if cfg:
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
rng = np.random.default_rng(42)

SR = 16000
DUR = 0.5  # seconds - kept short so the committed vectors stay well under budget
n = int(SR * DUR)
t = np.arange(n) / SR

cases = {}
voiced = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
noise = rng.standard_normal(n).astype(np.float32) * 0.05
cases["speech_plus_noise"] = np.stack([voiced + noise, np.roll(voiced, 5) * 0.3 + np.roll(noise, 2)])

imp = np.zeros(n, np.float32)
imp_len = min(800, n)
imp[: imp_len] = np.exp(-np.arange(imp_len) / 100) * rng.standard_normal(imp_len)
imp = np.roll(imp, n // 4)  # place the impulse mid-clip
cases["burst"] = np.stack([voiced * 0.2 + imp, np.roll(voiced, 5) * 0.06 + imp * 0.9])

drop = cases["speech_plus_noise"].copy()
drop[1, n // 2:] *= 0.01
cases["ref_dropout"] = drop

if cfg:
    # Constant speech from frame zero seeds the controller's energy floor too
    # high to open the blocking adaptation gate. A quiet lead-in followed by
    # near-mouth speech exercises that path without triggering a far-field burst.
    onset_rng = np.random.default_rng(77)
    floor = onset_rng.normal(0, 0.002, n).astype(np.float32)
    speech = onset_rng.normal(0, 0.1, n).astype(np.float32)
    speech[:int(0.15 * SR)] = 0
    cases["speech_onset"] = np.stack([floor + speech, floor + 0.15 * np.roll(speech, 5)])

for name, x in cases.items():
    sf.write(out / f"{name}.wav", x.T, SR, subtype="FLOAT")  # PCM16 quantised (and clipped the burst) so the npz never matched the wav
    r = pipeline.run(x, controller_on=cfg.get("controller_on", True), dsp_cfg=cfg.get("dsp"))
    np.savez_compressed(out / f"{name}.npz", **r)
    print(name, "burst frames:", int(r["burst"].sum()))
