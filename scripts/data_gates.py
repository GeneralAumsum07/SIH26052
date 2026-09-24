"""G1 data gate (plan 11.2 / 11.8): is the inter-mic level difference (ILD) alone a speech-vs-noise shortcut?

Per STFT bin of the mixer output, ILD = 10 log10(|P|^2 / |R|^2). Bins whose local SNR (clean vs primary - clean)
is above +10 dB are speech bins, below -10 dB noise bins; AUC = P(ILD of a speech bin > ILD of a noise bin).
Same construction as the diag lane's shortcut.py part 1 (v1: 0.975 parametric, 0.897 room). Gate: v2 AUC <= 0.75 on
both paths. Also: the reference-speech level histogram against M2 and the SPL calibration round trip.

usage: python scripts/data_gates.py [--versions 1 2] [--items 24] [--out results_r2/r8/data_gates]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly, stft as sstft

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vaani.data import calib, scenes   # noqa: E402
from vaani.data.mixer import V2_DEFAULTS, MixConfig, mix, mix_v2_ref_draw, speech_active_power   # noqa: E402

SR, NS = 16000, 6 * 16000
SPEECH_MANIFESTS = ("librispeech_100h.parquet", "cv_hi.parquet")
# v1 noise plan: the training corpora r7 saw (drone kept for v1; v2 excludes it via scenes.V2_EXCLUDED_CORPORA)
V1_NOISE = ("mad.parquet", "dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet", "demand.parquet",
            "esc50.parquet", "drone.parquet", "dns_datasets_fullband.noise_fullband.audioset_000.tar.parquet")
V2_NOISE = V1_NOISE[:4] + V1_NOISE[5:]
M2_BINS = [-30.0, -20.0, -16.9, -8.5, -6.0, 3.0, 10.0]


def load16(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g, axis=0).astype(np.float32)
    return x


def auc(pos, neg):
    """P(pos > neg), rank-based with average ranks for ties (mono items give many exact ties)."""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def speech_clip(rng, man):
    """6 s of concatenated utterances (at most 4 s each, 0.3 s gaps) at -26 dBFS speech-active rms, as shortcut.py."""
    parts, n = [], 0
    while n < NS:
        x = load16(man.path.iloc[int(rng.integers(len(man)))])[: 4 * SR, 0]
        parts += [x, np.zeros(int(0.3 * SR), np.float32)]; n += len(x) + int(0.3 * SR)
    s = np.concatenate(parts)[:NS]
    return (s * (10 ** (-26 / 20) / (np.sqrt(speech_active_power(s)) + 1e-9))).astype(np.float32)


def load_noise(row):
    x = load16(str(row["path"]).replace("\\", "/"))
    return x if x.shape[1] == 2 else x[:, 0]


def bins_ild(m, clean):
    """(speech-bin ILDs, noise-bin ILDs) for one item; bins more than 50 dB under the item's peak are dropped."""
    P, R, C = (sstft(v, nperseg=512, noverlap=256)[2] for v in (m[0], m[1], clean))
    pe, re_, ce = np.abs(P) ** 2 + 1e-12, np.abs(R) ** 2 + 1e-12, np.abs(C) ** 2 + 1e-12
    loc = 10 * np.log10(ce / (np.abs(P - C) ** 2 + 1e-12)); ild = 10 * np.log10(pe / re_)
    live = 10 * np.log10(pe) > 10 * np.log10(pe.max()) - 50
    return ild[live & (loc > 10)], ild[live & (loc < -10)]


def run_path(version, path, n_items, seed, bank, speech_men, noise_df, pool, v2_over, scene_over=None):
    sp_all, nz_all, rsg, modes, snrs, scenes_used = [], [], [], [], [], []
    for i in range(n_items):
        rng = np.random.default_rng(seed + i)
        s = speech_clip(rng, speech_men[i % len(speech_men)])
        if version == 1:
            noise = [load_noise(noise_df.iloc[int(rng.integers(len(noise_df)))])]
            cfg = MixConfig(p_room=1.0 if path == "room" else 0.0, p_clean=0.0, p_clip=0.0, p_ref_dropout=0.0,
                            p_wind=0.0, speech_rms_db=(-26.0, -26.0))
            m, clean, meta = mix(rng, s, noise, None, [], bank if path == "room" else None, cfg)
        else:
            cfg = MixConfig(version=2, p_room=1.0, p_clean=0.0, v2={"path": path, **v2_over})
            sc = scenes.sample_scene(rng, crop_s=NS / SR, **(scene_over or {}))
            rows, _ = pool.draw(rng, sc)
            noise = [load_noise(r) for r in rows if r is not None]
            sc["sources"] = [src for src, r in zip(sc["sources"], rows) if r is not None]
            m, clean, meta = mix(rng, s, noise, None, [], bank if path == "room" else None, cfg, scene=sc)
            modes.append(meta["ref_mode"]); snrs.append(meta["snr_db"]); scenes_used.append(meta["scene"])
        assert np.isfinite(m).all() and np.isfinite(clean).all(), f"non-finite v{version} {path} item {i}"
        rsg.append(meta["ref_speech_gain_db"])
        a, b = bins_ild(m, clean); sp_all.append(a); nz_all.append(b)
    sp, nz = np.concatenate(sp_all), np.concatenate(nz_all)
    rsg = np.asarray(rsg, float)
    out = {"items": n_items, "n_speech_bins": int(len(sp)), "n_noise_bins": int(len(nz)),
           "auc_ild": auc(sp, nz),
           "speech_ild_p10_p50_p90": [float(v) for v in np.percentile(sp, [10, 50, 90])],
           "noise_ild_p10_p50_p90": [float(v) for v in np.percentile(nz, [10, 50, 90])],
           "ref_speech_gain_db_min_med_max": [float(rsg.min()), float(np.median(rsg)), float(rsg.max())],
           "ref_speech_gain_hist": {"edges": M2_BINS, "counts": np.histogram(np.clip(rsg, -30, 10), M2_BINS)[0].tolist()},
           "share_ref_gain_-6_to_+3": float(((rsg >= -6) & (rsg <= 3)).mean())}
    if version == 2:
        out["ref_modes"] = {k: int(v) for k, v in zip(*np.unique(modes, return_counts=True))}
        out["scenes"] = {k: int(v) for k, v in zip(*np.unique(scenes_used, return_counts=True))}
        out["snr_db_p10_p50_p90"] = [float(v) for v in np.percentile(snrs, [10, 50, 90])]
    return out


def m2_histogram(n=20000, seed=0, v2_over=None):
    """The M2 reference-gain draw alone (no audio): mode shares and the -6..+3 dB share the mixer produces."""
    rng = np.random.default_rng(seed)
    p = {**V2_DEFAULTS, **(v2_over or {})}
    d = [mix_v2_ref_draw(rng, p) for _ in range(n)]
    g = np.asarray([x[1] for x in d]); modes = [x[0] for x in d]
    return {"n": n, "modes": {k: float(v / n) for k, v in zip(*np.unique(modes, return_counts=True))},
            "share_-6_to_+3": float(((g >= -6) & (g <= 3)).mean()), "min": float(g.min()), "max": float(g.max()),
            "hist": {"edges": M2_BINS, "counts": np.histogram(np.clip(g, -30, 10), M2_BINS)[0].tolist()}}


def spl_round_trip():
    spl = np.linspace(40, 123, 84)
    x = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR).astype(np.float64)
    err = [abs(calib.float_rms_db_to_spl(calib.rms_db(calib.scale_to_spl(x, s, "rms"))) - s) for s in spl]
    sine94 = calib.scale_to_spl(x, 94.0, "rms")
    return {"max_abs_err_db": float(max(err)),
            "sine_94dB_peak_dbfs": float(20 * np.log10(np.abs(sine94).max())),
            "full_scale_peak_db_spl": float(calib.HARD_CLIP_PEAK_DB_SPL),
            "pass": bool(max(err) < 0.01 and abs(20 * np.log10(np.abs(sine94).max()) + 26.0) < 0.05)}


def bench(n_items, seed, bank, speech_men, man_dir, split, crop_s=4.0):
    """Mixer items/s on in-memory audio (file reads excluded, one process): v1 at its training defaults, v2 with scenes.
    Smoke only on a shared machine; the numba flag says whether the Markov chain ran jitted."""
    import time
    from vaani.data import mixer as M
    n = int(crop_s * SR)
    rng0 = np.random.default_rng(seed)
    sp = [speech_clip(rng0, speech_men[i % len(speech_men)])[:n] for i in range(8)]
    nd = pd.concat([pd.read_parquet(man_dir / f) for f in V2_NOISE if (man_dir / f).exists()], ignore_index=True)
    nd = nd[(nd.split == split) & (nd.noise_class != "impulsive")].reset_index(drop=True)
    pool = scenes.ScenePool(nd); cache = {}

    def audio(r):
        if r["path"] not in cache:
            cache[r["path"]] = load_noise(r)
        return cache[r["path"]]
    jobs = []
    for i in range(n_items):   # draws and file reads happen here, outside the timed loops
        rng = np.random.default_rng(seed + 1000 + i)
        sc = scenes.sample_scene(rng, crop_s=crop_s)
        rows, _ = pool.draw(rng, sc)
        sc["sources"] = [s_ for s_, r in zip(sc["sources"], rows) if r is not None]
        jobs.append((sc, [audio(r) for r in rows if r is not None], audio(nd.iloc[int(rng.integers(len(nd)))])))
    out = {"items": n_items, "crop_s": crop_s, "numba": M._njit is not None, "bank": bank is not None,
           "label": "smoke: shared, loaded machine; not reportable"}
    for v in (1, 2):
        cfg = MixConfig() if v == 1 else MixConfig(version=2)
        m1 = mix(np.random.default_rng(0), sp[0], [jobs[0][2]], None, [], bank, cfg, **({} if v == 1 else {"scene": jobs[0][0]}))   # warm-up
        t0 = time.perf_counter()
        for i, (sc, nz, nz1) in enumerate(jobs):
            rng = np.random.default_rng(seed + i)
            if v == 1:
                mix(rng, sp[i % len(sp)], [nz1], None, [], bank, cfg)
            else:
                mix(rng, sp[i % len(sp)], nz, None, [], bank, cfg, scene=dict(sc, sources=[dict(x) for x in sc["sources"]]))
        dt = time.perf_counter() - t0
        out[f"v{v}_items_per_s"] = float(n_items / dt)
        out[f"v{v}_ms_per_item"] = float(1000 * dt / n_items)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--split", default="train")
    ap.add_argument("--manifest-dir", default=str(REPO / "data/manifests"))
    ap.add_argument("--bank", default=str(REPO / "data/rirs/bank_r3.npz"))
    ap.add_argument("--v2", default="{}", help="JSON overrides for the mix.v2 block")
    ap.add_argument("--scene", default="{}", help="JSON overrides for scenes.sample_scene (p_near, near_spl_rel)")
    ap.add_argument("--paths", nargs="+", default=["param", "room"])
    ap.add_argument("--out", default=str(REPO / "results_r2/r8/data_gates"))
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--bench", type=int, default=0, help="N > 0: time N mixer items per version instead of the gate")
    ap.add_argument("--bench-out", default=str(REPO / "results_r2/r8/mixer_bench.json"))
    a = ap.parse_args(argv)
    from vaani.data.rirs import RirBank
    man_dir = Path(a.manifest_dir)
    speech = [pd.read_parquet(man_dir / f).query("split == @a.split").reset_index(drop=True) for f in SPEECH_MANIFESTS]
    bank = RirBank(Path(a.bank)) if Path(a.bank).exists() else None
    v2_over, scene_over = json.loads(a.v2), json.loads(a.scene)
    if a.bench:
        res = bench(a.bench, a.seed, bank, speech, man_dir, a.split)
        res["command"] = "python scripts/data_gates.py --bench " + str(a.bench)
        Path(a.bench_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.bench_out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
        return res
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for v in a.versions:
        files = V1_NOISE if v == 1 else V2_NOISE
        nd = pd.concat([pd.read_parquet(man_dir / f) for f in files if (man_dir / f).exists()], ignore_index=True)
        nd = nd[(nd.split == a.split) & (nd.noise_class != "impulsive")].reset_index(drop=True)
        pool = scenes.ScenePool(nd) if v == 2 else None
        res = {"version": v, "split": a.split, "seed": a.seed, "bank": Path(a.bank).name if bank else None,
               "v2_overrides": v2_over if v == 2 else None, "scene_overrides": scene_over if v == 2 else None}
        for path in a.paths:
            if path == "room" and bank is None:
                res[path] = "TBD: no RIR bank at " + a.bank; continue
            res[path] = run_path(v, path, a.items, a.seed, bank, speech, nd, pool, v2_over, scene_over)
            print(f"v{v} {path}: auc_ild={res[path]['auc_ild']:.3f}", flush=True)
        if v == 2:
            res["m2_draw"] = m2_histogram(v2_over=v2_over)
            aucs = [res[p]["auc_ild"] for p in a.paths if isinstance(res[p], dict)]
            res["gate_auc_le"] = a.gate
            res["gate_pass"] = bool(aucs and max(aucs) <= a.gate and res["m2_draw"]["share_-6_to_+3"] >= 0.25 - 0.01)
        res["spl_round_trip"] = spl_round_trip()
        (out_dir / f"v{v}.json").write_text(json.dumps(res, indent=1))
        results[v] = res
    return results


if __name__ == "__main__":
    main()
