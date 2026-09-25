"""Validate an RIR bank against a reference bank: loader + mmap sidecars, mixer v2 room draws per scene, RT60 /
energy / inter-mic statistics per room class, room overlap with other banks, sha256 of every file.
Run from the repo root:
.venv/Scripts/python.exe results_r2/r8/banks/validate_bank.py --bank data/rirs/bank_r8.npz --ref data/rirs/bank_r3.npz \
    --overlap data/rirs/bank.npz data/rirs/bank_r3.npz data/rirs/bank_eval_r8.npz"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from vaani.data import mixer, rirs, scenes  # noqa: E402

SR = 16000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def t30(h):
    # Schroeder backward integral, -5..-35 dB fit extrapolated to 60 dB; nan when the RIR never decays 35 dB
    e = np.cumsum(h[::-1] ** 2)[::-1]
    if e[0] <= 0: return float("nan")
    e = 10 * np.log10(e / e[0] + 1e-30); i0, i1 = int(np.argmax(e <= -5)), int(np.argmax(e <= -35))
    if i1 <= i0: return float("nan")
    return float(-60.0 / np.polyfit(np.arange(i0, i1) / SR, e[i0:i1], 1)[0])


def room_stats(sp, nz):
    """sp (2, L) speech pair, nz (n_noise, 2, L); per-room scalars."""
    n0 = nz[0].astype(np.float64); s = sp.astype(np.float64)
    e_s = (s ** 2).sum(1); late0 = int(0.1 * SR)
    pk = int(np.argmax(np.abs(s[0]))); d = pk + int(0.0025 * SR)
    w = int(0.01 * SR); L = n0.shape[1]; m = ((min(L, int(0.6 * SR)) - late0) // w) * w
    env = [10 * np.log10((n0[c, late0:late0 + m].reshape(-1, w) ** 2).mean(1) + 1e-30) for c in range(2)] if m > 0 else None
    a, b = n0[0, late0:], n0[1, late0:]
    return {
        "t30_noise_primary_s": t30(n0[0]), "t30_noise_ref_s": t30(n0[1]),
        "late100_rel_noise_primary_db": float(10 * np.log10((n0[0, late0:] ** 2).sum() / max((n0[0] ** 2).sum(), 1e-30) + 1e-30)),
        "drr_speech_primary_db": float(10 * np.log10((s[0, :d] ** 2).sum() / max((s[0, d:] ** 2).sum(), 1e-30))),
        "speech_ild_db": float(10 * np.log10(e_s[0] / max(e_s[1], 1e-30))),
        "late_env_corr_noise": float(np.corrcoef(env[0], env[1])[0, 1]) if env is not None else float("nan"),
        "late_env_rms_diff_noise_db": float(np.sqrt(np.mean((env[0] - env[1]) ** 2))) if env is not None else float("nan"),
        "late_wave_corr_noise": float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30)),
        "max_abs": float(max(np.abs(sp).max(), np.abs(nz).max())), "finite": bool(np.isfinite(sp).all() and np.isfinite(nz).all()),
        "zero_rir": bool((np.abs(sp).sum(-1) == 0).any() or (np.abs(nz).sum(-1) == 0).any()),
    }


def bank_stats(path: Path, name: str, every: int = 1):
    b = rirs.RirBank(path)
    with np.load(path) as z:
        arm = np.asarray(z["armoured"], bool) if "armoured" in z.files else np.zeros(len(b), bool)
        meta = {k: (z[k].item() if z[k].ndim == 0 else None) for k in z.files if k not in ("speech", "noise", "rt60", "armoured")}
    rows = []
    for i in range(0, len(b), every):
        r = room_stats(np.asarray(b.speech[i]), np.asarray(b.noise[i])); r.update(idx=i, armoured=bool(arm[i]), rt60_drawn=float(b.rt60[i]))
        rows.append(r)
    summ = {"bank": name, "n": len(b), "armoured": int(arm.sum()), "armoured_frac": float(arm.mean()),
            "shapes": {"speech": list(b.speech.shape), "noise": list(b.noise.shape)}, "meta": {k: str(v) for k, v in meta.items()},
            "mmap": all(isinstance(x, np.memmap) for x in (b.speech, b.noise, b.rt60)),
            "all_finite": all(r["finite"] for r in rows), "any_zero_rir": sum(r["zero_rir"] for r in rows),
            "max_abs": max(r["max_abs"] for r in rows)}
    return b, arm, rows, summ


def table(rows, name):
    out = []
    keys = [k for k in rows[0] if k not in ("idx", "armoured", "finite", "zero_rir", "max_abs")]
    for cls, sel in (("all", lambda r: True), ("armoured", lambda r: r["armoured"]), ("plain", lambda r: not r["armoured"])):
        rs = [r for r in rows if sel(r)]
        if not rs: continue
        for k in keys:
            v = np.array([r[k] for r in rs], float); v = v[np.isfinite(v)]
            out.append({"bank": name, "class": cls, "metric": k, "n": len(v),
                        "median": float(np.median(v)) if len(v) else float("nan"),
                        "p10": float(np.percentile(v, 10)) if len(v) else float("nan"),
                        "p90": float(np.percentile(v, 90)) if len(v) else float("nan")})
    return out


class Spy:
    """Records what the mixer asked the bank for and what kind of room it got back."""
    def __init__(self, bank, arm):
        self.bank, self.rt_arm, self.log = bank, set(np.asarray(bank.rt60)[arm].tolist()), []

    def sample(self, rng, armoured=None):
        r = self.bank.sample(rng, armoured=armoured)
        self.log.append((armoured, r["rt60"] in self.rt_arm)); return r


def mixer_draws(bank, arm, n=40):
    t = np.arange(4 * SR) / SR
    speech = (0.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
    res = {}
    for name in ("apc", "command_post", "patrol"):
        spy = Spy(bank, arm); paths = []
        for seed in range(n):
            rng = np.random.default_rng(seed)
            sc = scenes.sample_scene(rng, name, crop_s=4.0)
            noises = [rng.standard_normal(3 * SR).astype(np.float32) * 0.01 for _ in sc["sources"]]
            m, c, meta = mixer.mix(rng, speech, noises, None, [], spy, mixer.MixConfig(version=2, p_clean=0.0), scene=sc)
            assert np.isfinite(m).all()
            paths.append(meta["path"])
        res[name] = {"items": n, "room_path": paths.count("room"), "armoured_rooms_served": sum(g for _, g in spy.log),
                     "plain_rooms_served": sum(not g for _, g in spy.log)}
    return res


def overlap(bank, other: Path):
    # rooms are continuous draws, so an exact rt60 match that also has an identical speech RIR is the same room
    o = rirs.RirBank(other); rt_o = {float(v): i for i, v in enumerate(np.asarray(o.rt60))}
    hits = [(i, rt_o[float(v)]) for i, v in enumerate(np.asarray(bank.rt60)) if float(v) in rt_o]
    L = min(bank.speech.shape[-1], o.speech.shape[-1])
    same = sum(np.array_equal(bank.speech[i][:, :L], o.speech[j][:, :L]) for i, j in hits)
    return {"other": str(other), "rt60_matches": len(hits), "identical_speech_rir": int(same)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True); ap.add_argument("--ref", default=None)
    ap.add_argument("--overlap", nargs="*", default=[]); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--mixer-items", type=int, default=40)
    ap.add_argument("--out", default="results_r2/r8/banks/validate")
    a = ap.parse_args()
    out = Path(a.out); res, rows_all = {}, []
    for tag, p in (("bank", a.bank), ("ref", a.ref)):
        if p is None: continue
        p = Path(p); b, arm, rows, summ = bank_stats(p, p.stem, a.every)
        summ["sha256"] = {f.name: sha256(f) for f in [p] + [p.with_name(f"{p.stem}.{k}.npy") for k in rirs.RirBank.KEYS]}
        summ["mixer_v2"] = mixer_draws(b, arm, a.mixer_items)
        if tag == "bank":
            summ["overlap"] = [overlap(b, Path(o)) for o in a.overlap if Path(o).resolve() != p.resolve()]
        res[tag] = summ; rows_all += table(rows, p.stem)
        print(json.dumps({k: v for k, v in summ.items() if k != "sha256"}, indent=1), flush=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0])); w.writeheader(); w.writerows(rows_all)
    for r in rows_all:
        if r["class"] != "all": print(f"{r['bank']:>14} {r['class']:>8} {r['metric']:>30} med {r['median']:8.3f}  p10 {r['p10']:8.3f}  p90 {r['p90']:8.3f}")


if __name__ == "__main__":
    main()
