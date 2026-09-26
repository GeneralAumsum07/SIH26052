"""G1 data gate (plan 11.2 / 11.8): is the inter-mic level difference (ILD) alone a speech-vs-noise shortcut?

Per STFT bin of the mixer output, ILD = 10 log10(|P|^2 / |R|^2). Bins whose local SNR (clean vs primary - clean)
is above +10 dB are speech bins, below -10 dB noise bins; AUC = P(ILD of a speech bin > ILD of a noise bin).
Same construction as the diag lane's shortcut.py part 1 (v1: 0.975 parametric, 0.897 room). Gate: v2 AUC <= 0.75 on
both paths. Also: the reference-speech level histogram against M2 and the SPL calibration round trip.

Per item (on unless --no-save-items) the speech-bin and noise-bin ILDs are kept as 0.1 dB histograms, the noise bins
split by the noise component that dominates them on the primary (bed, near, point, wind), so the pooled AUC can be
bootstrapped over items (--bootstrap B) and broken down by scene, reference mode, near-field source, wind and M2 bucket
without re-rendering (--from-items DIR recomputes both from a saved run).

usage: python scripts/data_gates.py [--versions 1 2] [--items 24] [--out results_r2/r8/data_gates] [--bootstrap 2000]
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
from vaani.data.mixer import V2_DEFAULTS, MixConfig, mix, mix_v2, mix_v2_ref_draw, speech_active_power   # noqa: E402

SR, NS = 16000, 6 * 16000
SPEECH_MANIFESTS = ("librispeech_100h.parquet", "cv_hi.parquet")
# v1 noise plan: the training corpora r7 saw (drone kept for v1; v2 excludes it via scenes.V2_EXCLUDED_CORPORA)
V1_NOISE = ("mad.parquet", "dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet", "demand.parquet",
            "esc50.parquet", "drone.parquet", "dns_datasets_fullband.noise_fullband.audioset_000.tar.parquet")
V2_NOISE = V1_NOISE[:4] + V1_NOISE[5:]
M2_BINS = [-30.0, -20.0, -16.9, -8.5, -6.0, 3.0, 10.0]
ILD_EDGES = np.linspace(-60.0, 60.0, 1201)   # 0.1 dB bins; the JSON keeps the rank AUC next to the binned one
COMPS = ("bed", "near", "point", "wind")      # noise-bin label: the component with the most primary power in that bin


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


def ild_hist(v):
    return np.histogram(np.clip(v, ILD_EDGES[0], ILD_EDGES[-1] - 1e-6), ILD_EDGES)[0].astype(np.float32)


def hist_auc(S, N):
    """AUC from binned counts (last axis): a speech bin beats every noise bin below its bin and ties (half) within it."""
    S = np.asarray(S, np.float64); N = np.asarray(N, np.float64)
    below = np.cumsum(N, axis=-1) - N
    a = (S * (below + 0.5 * N)).sum(-1) / (S.sum(-1) * N.sum(-1) + 1e-30)
    return float(a) if np.ndim(a) == 0 else a


def noise_labels(trace, n):
    """Per STFT bin of the primary, the index into COMPS of the dominant noise component (after the linear front end)."""
    parts = {c: np.zeros(n, np.float32) for c in COMPS}
    for role, pair in trace.get("noise", []):
        parts[role if role in parts else "point"] += pair[0]
    if trace.get("wind") is not None:
        parts["wind"] += trace["wind"][0]
    if trace.get("front_end"):
        g = np.asarray(trace["gains"])[:1]
        parts = {c: calib.front_end_linear(v[None], g)[0] for c, v in parts.items()}
    pw = np.stack([np.abs(sstft(parts[c], nperseg=512, noverlap=256)[2]) ** 2 for c in COMPS])
    return np.argmax(pw, axis=0)


def noise_bin_labels(m, clean, lab):
    """The dominant-component label of every noise bin, with the same bin selection as bins_ild."""
    P, C = (sstft(v, nperseg=512, noverlap=256)[2] for v in (m[0], clean))
    pe, ce = np.abs(P) ** 2 + 1e-12, np.abs(C) ** 2 + 1e-12
    loc = 10 * np.log10(ce / (np.abs(P - C) ** 2 + 1e-12))
    live = 10 * np.log10(pe) > 10 * np.log10(pe.max()) - 50
    return lab[live & (loc < -10)]


def item_rng(seed, i, legacy=False):
    """Item i's generator. Legacy seed + i made runs overlap (seed 101 item 101 == seed 202 item 0), so a "fresh" seed
    was not disjoint from the tuning seeds; [seed, i] spawns independent streams per (seed, item)."""
    return np.random.default_rng(seed + i if legacy else [seed, i])


def run_path(version, path, n_items, seed, bank, speech_men, noise_df, pool, v2_over, scene_over=None, items=None,
             legacy_seeds=False):
    """items: a list to receive one record per item (scene, mode, near/wind, M2 bucket) plus its ILD histograms."""
    sp_all, nz_all, rsg, modes, snrs, scenes_used = [], [], [], [], [], []
    for i in range(n_items):
        rng = item_rng(seed, i, legacy_seeds)
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
            tr = {}   # mix() routes version 2 to mix_v2 with these same arguments; the trace does not touch the rng
            m, clean, meta = mix_v2(rng, s, noise, None, [], bank if path == "room" else None, cfg, scene=sc, trace=tr)
            modes.append(meta["ref_mode"]); snrs.append(meta["snr_db"]); scenes_used.append(meta["scene"])
        assert np.isfinite(m).all() and np.isfinite(clean).all(), f"non-finite v{version} {path} item {i}"
        rsg.append(meta["ref_speech_gain_db"])
        a, b = bins_ild(m, clean); sp_all.append(a); nz_all.append(b)
        if items is not None:
            # v1 has one noise source: every noise bin is "bed"
            lab = noise_bin_labels(m, clean, noise_labels(tr, m.shape[1])) if version == 2 else np.zeros(len(b), int)
            near = [c for c in meta.get("noise_sources", []) if c["role"] == "near"]
            g = meta["ref_speech_gain_db"]
            items.append(dict(
                i=i, seed=seed + i if legacy_seeds else f"{seed}:{i}", scene=meta.get("scene", "v1"), ref_mode=meta.get("ref_mode", "v1"),
                near="none" if not near else ("unspatialised" if near[0]["ild_db"] is None else "pos" if near[0]["ild_db"] > 0 else "neg"),
                wind="yes" if meta.get("wind_mps", 0) > 0 else "no",
                m2_bucket="-6..+3" if -6 <= g <= 3 else "physical", ref_speech_gain_db=float(g),
                snr_db=float(meta["snr_db"] if version == 2 else (meta.get("snr_achieved_db") or np.nan)),
                clipped=bool(meta.get("clipped")), n_speech=int(len(a)), n_noise=int(len(b)),
                auc_item=auc(a, b) if len(a) and len(b) else np.nan,
                _S=ild_hist(a), _N=np.stack([ild_hist(b[lab == k]) for k in range(len(COMPS))])))
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


GROUP_KEYS = ("scene", "ref_mode", "near", "wind", "m2_bucket", "clipped")


def breakdown(recs, S, N) -> dict:
    """Which items and components make ILD separable. Per grouping: the within-group AUC (that group's speech and
    noise bins only), its share of all noise bins, and the pooled AUC with that group's items left out.
    by_component: all speech bins against the noise bins one component dominates, and the pooled AUC without them."""
    Nt = N.sum(1); St = S.sum(0); tot = max(float(Nt.sum()), 1.0)
    out = {"pooled_hist_auc": hist_auc(St, Nt.sum(0)), "by_component": {}}
    for k, c in enumerate(COMPS):
        nk = N[:, k].sum(0)
        out["by_component"][c] = {"noise_bin_share": float(nk.sum() / tot),
                                  "auc_vs_this": hist_auc(St, nk) if nk.sum() else None,
                                  "auc_without_this": hist_auc(St, Nt.sum(0) - nk) if (Nt.sum(0) - nk).sum() else None}
    for key in GROUP_KEYS:
        vals = np.asarray([str(r[key]) for r in recs]); res = {}
        for v in sorted(set(vals)):
            sel = vals == v
            res[v] = {"items": int(sel.sum()), "noise_bin_share": float(Nt[sel].sum() / tot),
                      "auc_within": hist_auc(S[sel].sum(0), Nt[sel].sum(0)) if Nt[sel].sum() and S[sel].sum() else None,
                      "auc_leave_out": hist_auc(S[~sel].sum(0), Nt[~sel].sum(0)) if (~sel).any() else None}
        out["by_" + key] = res
    return out


def bootstrap(S, N, b: int, gate: float = 0.75, seed: int = 0) -> dict:
    """Percentile CI of the pooled AUC over items (items resampled with replacement; bins stay with their item)."""
    rng = np.random.default_rng(seed)
    Nt = N.sum(1).astype(np.float64); S = S.astype(np.float64)
    w = rng.multinomial(len(S), np.full(len(S), 1 / len(S)), size=b).astype(np.float64)
    a = hist_auc(w @ S, w @ Nt)
    return {"b": b, "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "ci95": [float(v) for v in np.percentile(a, [2.5, 97.5])], "share_le_gate": float((a <= gate).mean())}


def save_items(out_dir, tag, recs):
    """items_<tag>.csv (one row per item) and items_<tag>.npz (S: items x bins, N: items x COMPS x bins)."""
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs]).to_csv(
        Path(out_dir) / f"items_{tag}.csv", index=False)
    np.savez_compressed(Path(out_dir) / f"items_{tag}.npz", S=np.stack([r["_S"] for r in recs]),
                        N=np.stack([r["_N"] for r in recs]), edges=ILD_EDGES, comps=np.asarray(COMPS))


def load_items(out_dir, tag):
    z = np.load(Path(out_dir) / f"items_{tag}.npz")
    return pd.read_csv(Path(out_dir) / f"items_{tag}.csv").to_dict("records"), z["S"], z["N"]


def item_stats(recs, S, N, b: int, gate: float) -> dict:
    out = {"breakdown": breakdown(recs, S, N)}
    if b:
        out["bootstrap"] = bootstrap(S, N, b, gate)
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


# --- level-chain trace (results_r2/r8/calib): where the v2 peaks come from, per scene -------------------------------
TRACE_SCENES = ("helicopter", "apc", "firefight", "artillery", "patrol", "command_post")
# the r8 training noise pools (mad_v2, not the folder-grouped mad) plus the gunshot corpora the scene events draw from
TRACE_NOISE = ("mad_v2.parquet", "dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet", "demand.parquet",
               "esc50.parquet", "dns_datasets_fullband.noise_fullband.audioset_000.tar.parquet", "gunshots.parquet",
               "cadre.parquet")
PARTS = ("speech", "bed", "near", "point", "wind", "impulse")
AOP_FRAME = 320   # 20 ms: the AOP is a sine level, so it is tested on short-frame rms, not on the instantaneous peak
LF_HZ = 60.0


def _spl(db_float):
    return float(db_float + calib.SPL_TO_FLOAT_RMS_DB)


def _peak_spl(x):
    return _spl(20 * np.log10(np.abs(x).max(initial=0.0) + 1e-12))


def _lf_share(x, hz=LF_HZ):
    X = np.abs(np.fft.rfft(np.asarray(x, np.float64))) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(X[f < hz].sum() / (X.sum() + 1e-30))


def _lowpass(x, hz=LF_HZ):
    X = np.fft.rfft(np.asarray(x, np.float64)); X[np.fft.rfftfreq(len(x), 1 / SR) >= hz] = 0
    return np.fft.irfft(X, len(x))


def _frame_spl(x2):
    n = x2.shape[-1] // AOP_FRAME * AOP_FRAME
    f = x2[..., :n].reshape(*x2.shape[:-1], -1, AOP_FRAME).astype(np.float64)
    return 10 * np.log10((f ** 2).mean(-1) + 1e-30) + calib.SPL_TO_FLOAT_RMS_DB


def trace_parts(tr, n):
    """Acoustic (2, n) pair per PARTS entry from a mix_v2 trace (zeros where absent)."""
    parts = {k: np.zeros((2, n), np.float32) for k in PARTS}
    parts["speech"] = tr["speech"].astype(np.float32)
    for role, pair in tr.get("noise", []):
        parts[role if role in parts else "point"] += pair
    if tr.get("wind") is not None:
        parts["wind"] += tr["wind"]
    if tr.get("impulse") is not None:
        parts["impulse"] += tr["impulse"]
    return parts


def trace_item(tr, meta, scene) -> dict:
    """Level chain of one v2 item: per part and in total, A- and Z-weighted Leq, peak, crest (primary, after the linear
    front end), the knee / AOP / rails shares, the part that is largest at each railed sample, and two counterfactuals:
    the item without each part, and the saturator fed the pre-HPF signal (what H2 would look like)."""
    lin = tr["lin"]; n = lin.shape[1]; g = np.asarray(tr["gains"])
    ac = trace_parts(tr, n)
    fe = (lambda x: calib.front_end_linear(x, g)) if tr.get("front_end") else (lambda x: x.astype(np.float32))
    lp = {k: fe(v) for k, v in ac.items()}
    knee = calib.soft_knee(); a = np.abs(lin)
    r = {"scene": meta["scene"], "effort": meta["effort"], "lombard": meta["lombard"], "path": meta["path"],
         "speech_spl_set": float(scene["speech_spl"]), "snr_db": float(meta["snr_db"]), "wind_mps": float(meta["wind_mps"]),
         "wind_spl_set": meta.get("wind_spl_db"), "event_fired": bool((scene.get("event") or {}).get("fired", False)),
         "event_peak_set": (scene.get("event") or {}).get("peak_spl"), "impulse_source": meta.get("impulse_source")}
    for s_ in scene["sources"]:
        r.setdefault(f"{s_['role']}_spl_set", float(s_["spl"]))
    tot = lin[0]
    r.update(total_A=_spl(calib.a_weighted_rms_db(tot)), total_Z=_spl(calib.rms_db(tot)), total_peak=_peak_spl(lin),
             total_lf_share_pre=_lf_share(sum(v[0] for v in ac.values())))
    r["total_crest"] = _peak_spl(tot) - r["total_Z"]
    fr = _frame_spl(lin)
    r.update(past_knee=bool(a.max() > knee), knee_sample_frac=float((a > knee).mean()),
             past_aop=bool((fr > calib.AOP_DB_SPL).any()), aop_frame_frac=float((fr > calib.AOP_DB_SPL).mean()),
             past_rails=bool(a.max() >= 1.0), rails_sample_frac=float((a >= 1.0).mean()),
             overloaded_meta=bool(meta["overloaded"]), clipped_meta=bool(meta["clipped"]))
    for k in PARTS:
        x = lp[k][0]
        if not np.any(x):
            continue
        r[f"{k}_A"] = _spl(calib.a_weighted_rms_db(x)); r[f"{k}_Z"] = _spl(calib.rms_db(x)); r[f"{k}_peak"] = _peak_spl(x)
        r[f"{k}_crest"] = r[f"{k}_peak"] - r[f"{k}_Z"]
        r[f"{k}_lf_share_pre"] = _lf_share(ac[k][0]); r[f"{k}_lf_peak_pre"] = _peak_spl(_lowpass(ac[k][0]))
        # counterfactual: the item without this part, still linear front end, rails test
        r[f"rails_frac_without_{k}"] = float((np.abs(lin - lp[k]) >= 1.0).mean())
    rail = a >= 1.0
    if rail.any():
        stack = np.stack([np.abs(lp[k]) for k in PARTS])                  # parts x 2 x n
        dom = np.argmax(stack[:, rail], axis=0)
        for j, k in enumerate(PARTS):
            r[f"rails_dom_{k}"] = float((dom == j).mean())
    pre = sum(v for v in ac.values()) * (10 ** (g / 20))[:, None].astype(np.float32)
    r["rails_frac_if_hpf_after"] = float((np.abs(pre) >= 1.0).mean())
    r["peak_if_hpf_after"] = _peak_spl(pre)
    r["past_rails_if_hpf_after"] = bool(np.abs(pre).max() >= 1.0)
    # counterfactuals on the same parts: (H1) beds/points scaled on unweighted rms instead of dBA; (near split) bed + near
    # holding the drawn bed level; (H5) a mic with full scale 10 dB higher (130 dB AOP class)
    zs = {k: (10 ** ((calib.a_weighted_rms_db(ac[k][0]) - calib.rms_db(ac[k][0])) / 20) if np.any(ac[k][0]) else 1.0)
          for k in ("bed", "near", "point")}
    lz = fe(sum(v * np.float32(zs.get(k, 1.0)) for k, v in ac.items()))
    r["past_rails_if_Z"] = bool(np.abs(lz).max() >= 1.0); r["peak_if_Z"] = _peak_spl(lz)
    if "near_spl_set" in r and np.any(ac["near"]):
        cut = np.float32(10 ** (-10 * np.log10(1 + 10 ** ((r["near_spl_set"] - r["bed_spl_set"]) / 10)) / 20))
        ls = fe(sum(v * (cut if k in ("bed", "near") else np.float32(1.0)) for k, v in ac.items()))
    else:
        ls = lin
    r["past_rails_if_near_split"] = bool(np.abs(ls).max() >= 1.0); r["peak_if_near_split"] = _peak_spl(ls)
    r["past_rails_if_fs130"] = bool(a.max() >= 10 ** (10 / 20))
    # distortion the saturator adds, re the linear signal (primary), per curve
    for cv in ("knee105", "tanh120"):
        e = np.clip(calib.saturate_curve(lin[0], cv), -1, 1) - lin[0]
        r[f"sat_err_db_{cv}"] = float(10 * np.log10((e.astype(np.float64) ** 2).sum() / ((lin[0].astype(np.float64) ** 2).sum() + 1e-30) + 1e-30))
    return r


def level_trace(n_items, seed, split, man_dir, bank, out_dir, scene_names=TRACE_SCENES, v2_over=None):
    """Render v2 scene items in memory exactly as the r8 test render does (render_eval_sets.render_scene_item), from
    the split's pools, with mix swapped for mix_v2(trace=...). Writes level_trace_items.csv and level_trace.json."""
    sys.path.insert(0, str(REPO / "scripts"))
    import render_eval_sets as R
    from vaani.data import mixer as M
    df = pd.concat([pd.read_parquet(man_dir / f) for f in (*SPEECH_MANIFESTS, *TRACE_NOISE) if (man_dir / f).exists()],
                   ignore_index=True)
    df = df[df.split == split]
    speech = df[df.kind == "speech"].reset_index(drop=True)
    pool = scenes.ScenePool(df[df.kind == "noise"])
    cfg = MixConfig(version=2, p_clean=0.0, v2={**R.R8_V2, **(v2_over or {})})
    recs = []
    orig = R.mix
    try:
        for si, name in enumerate(scene_names):
            for i in range(n_items):
                box = {}

                def traced(*a_, **k_):
                    box["tr"] = {}
                    out = M.mix_v2(*a_, trace=box["tr"], **k_)
                    box["scene"] = k_.get("scene")
                    return out
                R.mix = traced
                m, c, meta, _ = R.render_scene_item([seed, si, i], speech, pool, NS, name, bank, cfg=cfg)
                rec = trace_item(box["tr"], meta, box["scene"]); rec["i"] = i
                recs.append(rec)
            print(f"trace {name}: {n_items} items", flush=True)
    finally:
        R.mix = orig
    t = pd.DataFrame(recs)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t.to_csv(out_dir / "level_trace_items.csv", index=False)
    summ = {"items_per_scene": n_items, "seed": seed, "split": split, "crop_s": NS / SR, "knee_float": calib.soft_knee(),
            "knee_peak_db_spl": _spl(20 * np.log10(calib.soft_knee())), "aop_db_spl": calib.AOP_DB_SPL,
            "rails_peak_db_spl": calib.HARD_CLIP_PEAK_DB_SPL, "aop_frame_s": AOP_FRAME / SR, "v2_overrides": v2_over or {},
            "scenes": {}}
    for name, g_ in t.groupby("scene", sort=False):
        d = {"items": int(len(g_))}
        for col in ("past_knee", "past_aop", "past_rails", "overloaded_meta", "clipped_meta", "event_fired",
                    "past_rails_if_hpf_after", "past_rails_if_Z", "past_rails_if_near_split", "past_rails_if_fs130"):
            d[f"share_{col}"] = float(g_[col].mean())
        for col in ("knee_sample_frac", "aop_frame_frac", "rails_sample_frac", "rails_frac_if_hpf_after"):
            d[f"mean_{col}"] = float(g_[col].mean())
        for col in [c for c in t.columns if any(c.endswith(s_) for s_ in ("_A", "_Z", "_peak", "_crest", "_lf_share_pre",
                                                                          "_lf_peak_pre", "_set"))] + [
                    "snr_db", "peak_if_hpf_after", "peak_if_Z", "peak_if_near_split", "sat_err_db_knee105", "sat_err_db_tanh120"]:
            if col.startswith("past_") or g_[col].dtype == bool:   # flags are summarised as shares above
                continue
            v = pd.to_numeric(g_[col], errors="coerce").dropna().astype(float)
            if len(v):
                d[col] = [round(float(q), 2) for q in np.percentile(v, [10, 50, 90])] + [int(len(v))]
        dom = {k: float(g_[f"rails_dom_{k}"].fillna(0).mul(g_["rails_sample_frac"]).sum()) for k in PARTS if f"rails_dom_{k}" in g_}
        tot = sum(dom.values())
        d["railed_samples_by_largest_part"] = {k: round(v / tot, 4) for k, v in dom.items()} if tot else {}
        d["share_items_railed_without"] = {k: float((g_[f"rails_frac_without_{k}"].fillna(g_["rails_sample_frac"]) > 0).mean())
                                           for k in PARTS if f"rails_frac_without_{k}" in g_}
        d["share_items_part_alone_past_rails"] = {k: float((pd.to_numeric(g_[f"{k}_peak"], errors="coerce") >= calib.HARD_CLIP_PEAK_DB_SPL).mean())
                                                  for k in PARTS if f"{k}_peak" in g_}
        summ["scenes"][name] = d
    (out_dir / "level_trace.json").write_text(json.dumps(summ, indent=1))
    return summ


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
    ap.add_argument("--no-save-items", dest="save_items", action="store_false", help="skip items_*.csv/npz and the breakdown")
    ap.add_argument("--bootstrap", type=int, default=0, help="B > 0: bootstrap CI of the pooled AUC over items")
    ap.add_argument("--from-items", nargs="+", default=None,
                    help="DIR [DIR ...]: recompute breakdown and bootstrap from saved runs' items_* (several dirs are pooled)")
    ap.add_argument("--pooled-out", default=None, help="with several --from-items dirs: where item_stats.json goes")
    ap.add_argument("--legacy-seeds", action="store_true", help="item rng = seed + i (runs before 2026-09-25 15:00)")
    ap.add_argument("--calib", type=int, default=0, help="N > 0: level-chain trace, N items per scene, into --out")
    ap.add_argument("--calib-scenes", nargs="+", default=list(TRACE_SCENES))
    a = ap.parse_args(argv)
    if a.from_items:
        res = {}
        for f in sorted(Path(a.from_items[0]).glob("items_*.npz")):
            tag = f.stem[len("items_"):]
            parts = [load_items(d, tag) for d in a.from_items]
            recs = [r for p_ in parts for r in p_[0]]
            S, N = np.concatenate([p_[1] for p_ in parts]), np.concatenate([p_[2] for p_ in parts])
            res[tag] = {"dirs": [str(d) for d in a.from_items], "items": len(recs), "pooled_auc": hist_auc(S.sum(0), N.sum((0, 1))),
                        **item_stats(recs, S, N, a.bootstrap, a.gate)}
        out = Path(a.pooled_out or a.from_items[0]); out.mkdir(parents=True, exist_ok=True)
        (out / "item_stats.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: [v["pooled_auc"], v.get("bootstrap", {}).get("ci95")] for k, v in res.items()}, indent=1))
        print(json.dumps({k: v.get("bootstrap") for k, v in res.items()}, indent=1))
        return res
    from vaani.data.rirs import RirBank
    man_dir = Path(a.manifest_dir)
    speech = [pd.read_parquet(man_dir / f).query("split == @a.split").reset_index(drop=True) for f in SPEECH_MANIFESTS]
    bank = RirBank(Path(a.bank)) if Path(a.bank).exists() else None
    v2_over, scene_over = json.loads(a.v2), json.loads(a.scene)
    if a.calib:
        res = level_trace(a.calib, a.seed, a.split, man_dir, bank, a.out, a.calib_scenes, v2_over)
        print(json.dumps({k: {q: v[q] for q in ("share_past_knee", "share_past_aop", "share_past_rails")}
                          for k, v in res["scenes"].items()}, indent=1))
        return res
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
        res = {"version": v, "split": a.split, "seed": a.seed, "seed_scheme": "seed+i" if a.legacy_seeds else "[seed, i]", "bank": Path(a.bank).name if bank else None,
               "v2_overrides": v2_over if v == 2 else None, "scene_overrides": scene_over if v == 2 else None}
        for path in a.paths:
            if path == "room" and bank is None:
                res[path] = "TBD: no RIR bank at " + a.bank; continue
            recs = [] if a.save_items else None
            res[path] = run_path(v, path, a.items, a.seed, bank, speech, nd, pool, v2_over, scene_over, items=recs,
                                 legacy_seeds=a.legacy_seeds)
            print(f"v{v} {path}: auc_ild={res[path]['auc_ild']:.3f}", flush=True)
            if recs:
                tag = f"v{v}_{path}"; save_items(out_dir, tag, recs)
                S = np.stack([r["_S"] for r in recs]); N = np.stack([r["_N"] for r in recs])
                res[path]["items_file"] = f"items_{tag}.npz"
                res[path].update(item_stats(recs, S, N, a.bootstrap, a.gate))
                bs = res[path].get("bootstrap")
                print(f"  hist auc {res[path]['breakdown']['pooled_hist_auc']:.4f}" + (f", ci95 {bs['ci95']}" if bs else ""),
                      flush=True)
        if v == 2:
            res["m2_draw"] = m2_histogram(v2_over=v2_over)
            aucs = [res[p]["auc_ild"] for p in a.paths if isinstance(res[p], dict)]
            res["gate_auc_le"] = a.gate
            res["gate_pass"] = bool(aucs and max(aucs) <= a.gate and res["m2_draw"]["share_-6_to_+3"] >= 0.25)   # plan M2: at least 25 % of draws in -6..+3 dB
        res["spl_round_trip"] = spl_round_trip()
        (out_dir / f"v{v}.json").write_text(json.dumps(res, indent=1))
        results[v] = res
    return results


if __name__ == "__main__":
    main()
