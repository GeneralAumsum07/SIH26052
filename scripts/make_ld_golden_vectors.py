"""Low-delay golden vectors for the native runtime (low-delay plan Task 6). Deterministic inputs -> per-stage tensors.

  python scripts/make_ld_golden_vectors.py [--out deploy/dsp_reference/vectors_ld] [--seed 0]

For every shortlisted contract (Arm A's three supports with Mini-P18, Arm B with Mini-P32) the script exports the
seeded untrained network's step graph (vaani.export.export_fe: contract-stamped, folded) to <contract>/model.onnx and
streams each case through the ORT graph with vaani.low_delay_live.LowDelayStreamEngine's components, recording every
hop's stage tensors:
  frontend_in      (hops, 2, H) raw primary/reference, available (hops, H)
  limiter_in       (hops, 2, H) after reference zeroing (and the finite fallback of a non-finite primary)
  limiter_out      (hops, 2, H) the r8 limiter alone (compiled kernel, as every r8 arm)
  frontend_out     (hops, 2, H) after the reconnect ramp: the samples analysis sees; frame_valid (hops,)
  analysis         (hops, 257, 4) primary re/im, reference re/im of the last K samples
  step_out         (hops, 257, 2) the neural step's output spectrum; state (hops, S) after each step
  synthesis        (hops, H) released samples; the aligned output is concat(synthesis)[release_lead:][:N]
The last hop is the flush hop (known zeros, frontend bypassed). To keep the committed set small, the full stage
tensors (limiter_in .. step_out) are kept for the STAGE_CASE only, at hops `stage_hops` (startup, steady state and
the flush), and its state at `state_hops`; every case keeps its inputs, frame validity, discontinuity flags, synthesis, output and the state
after the last hop (final_state). The stage-by-stage replay is asserted equal to the
engine's own run() output and within 1e-5 of the offline route, so the vectors cannot drift from the reference.

Resampler vectors (resampler/<id>.npz, R0/R1/R2): 16 kHz -> 48 kHz interpolation and 48 -> 16 kHz decimation of a
fixed signal in uneven blocks, with the coefficients and their SHA-256.

Clips are 0.5 s so the committed vectors stay small. The graphs are untrained (seeded): they pin arithmetic, not
quality. manifest.json records every file's SHA-256, the contract hashes and the generating commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani import audio_contract as ac  # noqa: E402
from vaani import backend as bk  # noqa: E402
from vaani import resampler as rs  # noqa: E402
from vaani.dsp import pipeline  # noqa: E402
from vaani.low_delay_live import LowDelayStreamEngine  # noqa: E402

SR = 16000
N = 8000                                   # 0.5 s
DSP = {"limiter": True, "limiter_kernel": "numba", "ref_policy": {"absent": "freeze", "ramp_samples": 3072}}
NETS = [(cid, "p18") for cid in ac.ARM_A_IDS] + [(ac.ARM_B_ID, "p32")]
TOL = 1e-5
STAGE_CASE = "speech_plus_noise"
STAGE_KEYS = ("limiter_in", "limiter_out", "frontend_out", "analysis", "step_out")


def stage_hops(n_hops: int) -> np.ndarray:
    """Hops whose stage tensors are kept: the first 32 (startup, history fill) and the last 4 (end, flush)."""
    return np.unique(np.r_[np.arange(min(32, n_hops)), np.arange(max(0, n_hops - 4), n_hops)])


def state_hops(n_hops: int) -> np.ndarray:
    """Hops whose recurrent state is kept: the first 4, hops 15 and 31, and the last 4."""
    return np.unique(np.r_[np.arange(min(4, n_hops)), [h for h in (15, 31) if h < n_hops],
                           np.arange(max(0, n_hops - 4), n_hops)])


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cases(seed=0):
    """name -> (primary, reference, available) at 16 kHz, N samples."""
    g = np.random.default_rng(seed)
    t = np.arange(N) / SR
    voiced = (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    noise = (g.standard_normal(N) * 0.05).astype(np.float32)
    ref = (0.8 * noise + g.standard_normal(N).astype(np.float32) * 0.01).astype(np.float32)
    on = np.ones(N, bool)
    out = {"speech_plus_noise": (voiced + noise, ref, on)}
    gap = on.copy(); gap[2000:3500] = False; gap[5000:5003] = False
    out["ref_dropout"] = (voiced + noise, ref, gap)
    loud = (voiced + noise).copy(); loud[3000:3400] *= 12.0
    out["limiter_burst"] = (loud, ref * 4.0, on)
    nf = (voiced + noise).copy(); nf[4100] = np.nan; nf[4101] = np.inf
    rf = ref.copy(); rf[6000] = np.nan
    out["nonfinite"] = (nf, rf, on)
    out["silence"] = (np.zeros(N, np.float32), np.zeros(N, np.float32), on)
    out["low_level"] = ((voiced + noise) * 1e-5, ref * 1e-5, on)
    return out


def replay(eng: LowDelayStreamEngine, prim, ref, avail):
    """Stream one clip hop by hop through the engine's components, recording every stage; the flush hop last."""
    c, H = eng.c, eng.hop
    eng.reset()
    lim = pipeline.make_limiter(eng.dsp)
    hops = -(-N // H)
    pad = hops * H - N
    P = np.pad(prim, (0, pad)); R = np.pad(ref, (0, pad)); A = np.pad(avail, (0, pad), constant_values=True)
    rec = {k: [] for k in ("frontend_in", "available", "limiter_in", "limiter_out", "frontend_out", "frame_valid",
                           "analysis", "step_out", "state", "synthesis", "discontinuity")}
    for j in range(hops + 1):
        flush = j == hops
        s = slice(j * H, (j + 1) * H)
        p, r, av = (np.zeros(H, np.float32), np.zeros(H, np.float32), np.ones(H, bool)) if flush else (P[s], R[s], A[s])
        rec["frontend_in"].append(np.stack([p, r])); rec["available"].append(av)
        # the limiter alone, on the same zeroed inputs the frontend gives it
        av_eff = av & np.isfinite(r)
        li = np.stack([np.where(np.isfinite(p), p, 0), np.where(av_eff, np.where(np.isfinite(r), r, 0), 0)]).astype(np.float32)
        rec["limiter_in"].append(li)
        if flush:
            rec["limiter_out"].append(li)
            y = eng.flush()
            fo, disc = np.zeros((2, H), np.float32), False
        else:
            lp, lr = lim.process_block(li[0].copy(), li[1].copy()); lim.engaged = 0
            rec["limiter_out"].append(np.stack([lp, lr]))
            y = eng.process(p, r, av)
            fo, disc = None, eng.last["discontinuity"]
        # read the stage outputs back from the engine's state after the hop
        hist_p, hist_r = eng.an_p.hist, eng.an_r.hist
        frame = np.stack([hist_p[-H:], hist_r[-H:]]) if fo is None else fo
        rec["frontend_out"].append(frame.astype(np.float32))
        rec["frame_valid"].append(np.float32(eng.validity.last_bad < eng.validity.sample - c.k))
        a_p = np.fft.rfft(np.concatenate([_prev(eng, "p"), hist_p[-H:]]) * eng.an_p.a)
        a_r = np.fft.rfft(np.concatenate([_prev(eng, "r"), hist_r[-H:]]) * eng.an_r.a)
        rec["analysis"].append(np.stack([a_p.real, a_p.imag, a_r.real, a_r.imag], -1).astype(np.float32))
        rec["step_out"].append(eng._last_out.copy())
        rec["state"].append(eng.backend._host(eng.state.caches["state"])[0].copy())
        rec["synthesis"].append(y.astype(np.float32))
        rec["discontinuity"].append(bool(disc))
        _keep_prev(eng)
    out = {k: np.asarray(v) for k, v in rec.items()}
    y = np.concatenate(rec["synthesis"])
    out["output"] = y[c.release_lead:c.release_lead + N]
    return out


# the analysis frame needs the K - H samples that preceded this hop: tracked beside the engine
_PREV = {}


def _prev(eng, ch):
    return _PREV.get((id(eng), ch), np.zeros(eng.c.k - eng.c.hop, np.float32))


def _keep_prev(eng):
    _PREV[(id(eng), "p")] = eng.an_p.hist.copy()
    _PREV[(id(eng), "r")] = eng.an_r.hist.copy()


class _Recording:
    """Backend wrapper that keeps the last step output (the engine returns only synthesized samples)."""

    def __init__(self, b, eng_ref):
        self.b, self.eng_ref = b, eng_ref

    def __getattr__(self, k):
        return getattr(self.b, k)

    def step(self, spec6, feats, state, valid=1.0):
        out = self.b.step(spec6, feats, state, valid)
        self.eng_ref[0]._last_out = out[0, :, 0, :].copy()
        return out

    def reset(self, state):
        r = self.b.reset(state)
        self.eng_ref[0]._last_out = np.zeros((257, 2), np.float32)   # a bypassed hop: no neural output
        return r


def resampler_vectors(out: Path, seed=0):
    g = np.random.default_rng(seed + 1)
    x16 = (g.standard_normal((2, 1600)) * 0.1).astype(np.float32)
    x48 = (g.standard_normal((2, 4800)) * 0.1).astype(np.float32)
    blocks = [96, 128, 7, 1, 300, 1068]                          # uneven 16 kHz block sizes, sum 1600
    files = {}
    for rid in rs.IDS:
        pair = rs.load(rid)
        itp, dec = pair.interpolator(2), pair.decimator(2)
        yi, yd, a = [], [], 0
        for b in blocks:
            yi.append(itp(x16[:, a:a + b])); yd.append(dec(x48[:, 3 * a:3 * (a + b)])); a += b
        p = out / "resampler" / f"{rid}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, h=pair.h, x16=x16, x48=x48, blocks=np.asarray(blocks), interpolated=np.concatenate(yi, 1),
                            decimated=np.concatenate(yd, 1))
        files[rid] = {"file": p.relative_to(out).as_posix(), "sha256": _sha(p), "coef_sha256": pair.sha256,
                      "delays": pair.delays}
    return files


def main(argv=None):
    import torch
    from vaani import export as E
    from vaani.enhance_low_delay import enhance_low_delay
    from vaani.models import vaani_fe as V
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deploy/dsp_reference/vectors_ld"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = {"generated_by": "python scripts/make_ld_golden_vectors.py --seed %d" % a.seed, "git_sha": _git("rev-parse", "HEAD"),
           "dsp": DSP, "clip_samples": N, "sr": SR, "tolerance": TOL, "contracts": {}}
    inputs = cases(a.seed)
    for cid, tiling in NETS:
        c = ac.get_audio_contract(cid)
        d = out / cid
        d.mkdir(exist_ok=True)
        model = E.fe_untrained({"tier": "mini", "audio_contract": cid, **V.MINI_P[tiling]}, a.seed)
        rep = E.export_fe(model, d / "model_raw.onnx", d / "model.onnx", streams=2, hops=60, seed=a.seed)
        (d / "model_raw.onnx").unlink()
        if not rep["parity"]["pass"]:
            raise SystemExit(f"{cid}: export parity failed {rep['parity']['failed']}")
        holder = [None]
        eng = LowDelayStreamEngine(c, _Recording(bk.FeOrtBackend(d / "model.onnx", audio_contract=cid), holder), DSP)
        holder[0] = eng
        entry = {"contract_hash": c.contract_hash, "network": f"mini_{tiling}", "seed": a.seed,
                 "model": {"file": f"{cid}/model.onnx", "sha256": _sha(d / "model.onnx")},
                 "state_floats": model.state_size, "hop": c.hop, "support": c.support, "release_lead": c.release_lead,
                 "cases": {}}
        for name, (p, r, av) in inputs.items():
            _PREV.clear()
            rec = replay(eng, p, r, av)
            ref_eng = LowDelayStreamEngine(c, bk.FeOrtBackend(d / "model.onnx", audio_contract=cid), DSP)
            y_eng = ref_eng.run(p, r, av)
            if not np.array_equal(rec["output"], y_eng, equal_nan=True):
                raise SystemExit(f"{cid}/{name}: stage replay differs from LowDelayStreamEngine.run")
            if np.isfinite(p).all():                               # the offline route rejects nothing but NaN input
                y_off, _ = enhance_low_delay(np.stack([p, r]), av, model, c, dsp_cfg=DSP)
                err = float(np.abs(y_off - y_eng).max())
                if err > TOL:
                    raise SystemExit(f"{cid}/{name}: stream vs offline {err:.3g} > {TOL}")
            else:
                err = None
            del rec["output"]                                          # = synthesis shifted by release_lead
            rec["final_state"] = rec["state"][-1]
            if name == STAGE_CASE:
                keep = stage_hops(len(rec["synthesis"]))
                rec["stage_hops"] = keep
                for k in STAGE_KEYS:
                    rec[k] = rec[k][keep]
                rec["state_hops"] = state_hops(len(rec["synthesis"]))
                rec["state"] = rec["state"][rec["state_hops"]]
            else:
                for k in (*STAGE_KEYS, "state"):
                    del rec[k]
            f = d / f"{name}.npz"
            np.savez_compressed(f, **rec)
            entry["cases"][name] = {"file": f.relative_to(out).as_posix(), "sha256": _sha(f), "hops": int(len(rec["synthesis"])),
                                    "stream_vs_offline_max_abs": err}
            print(f"{cid} {name}: {len(rec['synthesis'])} hops, stream vs offline {err}", flush=True)
        man["contracts"][cid] = entry
    man["resampler"] = resampler_vectors(out, a.seed)
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
