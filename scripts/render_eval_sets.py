"""Render frozen val/test sets, bucketed by noise class x input SNR.
Burst buckets also get a 'twin' with the impulse removed (same seed) for
the recovery-time metric. Impulses come from the synthetic generator or
from recorded impulsive clips in the manifest ("recorded_*" buckets). In
eval_r2 those recorded clips are ESC-50 household transients (can opening,
mouse click, keyboard, fireworks, footsteps, clock tick), and every mixture
is synthetic: clean speech plus noise through a simulated room, not a
recording made in noise. --faults adds reliability-fault buckets (clipping,
reference-mic faults, over-range bursts) at fixed SNRs; report.py keeps
them out of the nominal envelope. --defence renders the PS defence-noise set
instead (DEFENCE below: recorded gunshots, Friedlander blasts, MAD
helicopter/vehicle, ESC-50 siren) with r7's training mix block. --r8-test
renders the pre-registered r8 test set (plan 11.2 G6, ex-B3; results_r2/r8/testset/PROTOCOL.md): v1 nominal,
defence, held-out drone/NOISEX, EARS loud, faults and mixer-v2 scenes, from the held-out groups that
--write-heldout records in configs/data/r8_heldout_exclude.json. Writes an eval-set hash for run.json.
"""
import argparse, csv, hashlib, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import lfilter

from vaani.data import impulses, manifests, rirs, scenes
from vaani.data.dataset import BUCKET_SNRS, SR, _load
from vaani.data.mixer import MixConfig, concat_xfade, fit_xfade, mix
from vaani.data.rirs import RirBank

# bucket -> (continuous noise class, impulse source); None = no burst
CLASSES = {"stationary": ("stationary", None), "changing": ("changing", None),
           "impulsive": ("changing", "synthetic"), "impulsive+stationary": ("stationary", "synthetic"),
           "recorded_impulsive": ("changing", "corpus"), "recorded_impulsive+stationary": ("stationary", "corpus")}
FAULT_SNRS = [0, 5]

# defence category -> (continuous bed, impulse source). Blasts use vaani/data/blast.py physics v2: small arms over
# 1-300 m, artillery 0.1-20 kg TNT over 30-3000 m; the level at the mic is still set by impulse_peak_db below. Transient categories sit on the MAD stationary bed
# (vehicle, helicopter, fighter test rows) so the input SNR means the same thing as in the bed-only rows.
DEFENCE = {"gunshot": ("mad_stationary", "gunshot"),
           "blast_small_arms": ("mad_stationary", "blast:small_arms"),
           "blast_artillery": ("mad_stationary", "blast:artillery"),
           "helicopter": ("mad:helicopter", None), "vehicle": ("mad:vehicle", None), "siren": ("esc50:siren", None)}
# noise_class of the bed-only categories: sources.MAD_CLASS_MAP puts helicopter and vehicle in stationary;
# the two ESC-50 siren test clips are labelled changing in the manifest
CLASS_OF_BED = {"mad:helicopter": "stationary", "mad:vehicle": "stationary", "esc50:siren": "changing"}
# r7's training mix block (configs/retraining/r7_e256_wr64.yaml data.mix) minus impulse_kinds, which the category fixes
DEFENCE_MIX = {"impulse_peak_db": (15.0, 45.0), "impulse_room": True, "overload_softclip": True,
               "speech_rms_db": (-32.0, -18.0)}
GUNSHOT_CORPORA = ("gunshots", "cadre")


def _impulse(seed, source, imp_df):
    """(waveform, onsets_s, source label) for a burst bucket."""
    rng = np.random.default_rng(seed + [7])
    if source == "synthetic":
        imp, m = impulses.generate(rng)
        return imp, m["onsets_s"], f"synthetic:{m['kind']}"
    if source.startswith("blast:"):
        # physics v2 (same-sign ground bounce, ISO 9613-1 absorption over the drawn range), not r7's v1 training draw
        imp, m = impulses.generate(rng, kind="blast", blast_kind=source.split(":", 1)[1], physics="v2")
        return imp, m["onsets_s"], f"synthetic:blast:{m['blast_kind']}"
    if source == "gunshot":
        return _peak_window(imp_df.iloc[int(rng.integers(len(imp_df)))])
    row = imp_df.iloc[int(rng.integers(len(imp_df)))]
    imp = _load(row.path, 2 * SR, rng)
    # same treatment as DynamicMixDataset: peak-normalise so impulse_peak_db is comparable, onsets from the waveform
    imp = (imp / (np.abs(imp).max() + 1e-9)).astype(np.float32)
    return imp, impulses.detect_onsets(imp, SR), str(row.get("source_id", row.path))


def _peak_window(row, win_s=2.0, pre_s=0.25):
    """A recorded shot cropped to win_s around its loudest sample: a random crop can miss the shot, and
    peak-normalising a crop without it would turn background hiss into a 45 dB 'transient'."""
    x, _ = sf.read(row.path, dtype="float32")
    x = x if x.ndim == 1 else x.mean(axis=1)
    a = max(0, int(np.argmax(np.abs(x))) - int(pre_s * SR))
    imp = x[a:a + int(win_s * SR)]
    imp = (imp / (np.abs(imp).max() + 1e-9)).astype(np.float32)
    return imp, impulses.detect_onsets(imp, SR), str(row.get("source_id", row.path))


def render_bucket_item(seed: list[int], speech_df, pool_df, n: int, snr: float, impulse, bank, imp_df=None, cfg=None,
                       tag_noise=False):
    """One eval clip. impulse is None | "synthetic" | "corpus" | "gunshot" | "blast:<kind>". Returns
    (mix, clean, meta, twin_or_None). tag_noise records the bed's source_id; off so older sets keep their meta bytes."""
    rng = np.random.default_rng(seed)
    sp = speech_df.iloc[int(rng.integers(len(speech_df)))]
    x = _load(sp.path, n, rng)
    s = np.pad(x, (0, n - len(x)))
    j = int(rng.integers(len(pool_df)))
    nz = [_load(pool_df.path.iloc[j], n, rng)]
    cfg = cfg or MixConfig(snr_range=(snr, snr), p_clean=0.0)
    imp, on, src = _impulse(seed, impulse, imp_df) if impulse else (None, [], None)
    m, c, meta = mix(np.random.default_rng(seed), s, nz, imp, on, bank, cfg)
    meta["impulse_source"] = src
    meta["speech_source"] = str(sp.get("source_id", sp.path))  # report needs the corpus to know whether English WER applies
    if tag_noise:
        meta["noise_source"] = str(pool_df.iloc[j].get("source_id", pool_df.path.iloc[j]))
    twin = None
    if impulse:  # identical draw with no impulse, scaled like the burst clip so only the impulse differs
        twin, _, _ = mix(np.random.default_rng(seed), s, nz, None, [], bank, cfg, norm_gain=meta["norm_gain"])
    return m, c, meta, twin


# --- reliability faults: applied to the observed channels only, the clean target is untouched ---
def _clip_primary(m, frac, ctx):
    # level fixed by the first call so the burst clip and its twin are clipped identically
    lvl = ctx.setdefault("lvl", float(np.abs(m[0]).max() * frac))
    out = m.copy(); out[0] = np.clip(m[0], -lvl, lvl); return out


def _ref_dropout(m, seconds, ctx):
    out = m.copy(); a = m.shape[1] // 3
    out[1, a:min(m.shape[1], a + int(seconds * SR))] *= np.float32(10 ** (-40 / 20)); return out


def _ref_obstructed(m, _, ctx):
    """Hand or fabric over the reference mic: muffled and attenuated, not silent.
    Unity-gain one-pole near 200 Hz measures about -6 dB broadband on clean speech. The frozen eval_r2 render used
    a x3.0 makeup (+3 dB, i.e. muffled but louder); that set is kept as is, later renders get the attenuated form."""
    out = m.copy(); out[1] = lfilter([0.08], [1.0, -0.92], m[1]).astype(np.float32); return out


def _ref_gain(m, db, ctx):
    out = m.copy(); out[1] = m[1] * np.float32(10 ** (db / 20)); return out


def _ref_desync(m, samples, ctx):
    """Clock slip / wiring delay between the two capture channels: the reference arrives late, the gap is silence.
    (The frozen eval_r2 render used np.roll, which wrapped the last 64 samples to the front.)"""
    out = m.copy(); out[1] = np.concatenate([np.zeros(int(samples), np.float32), m[1, :-int(samples)]]); return out


def _identity(m, _, ctx):
    return m


# name -> (mixer overrides, post-hoc degradation, its argument)
FAULTS = {
    "fault_none":           ({}, _identity, None),
    "fault_clip_mild":      ({}, _clip_primary, 0.60),
    "fault_clip_hard":      ({}, _clip_primary, 0.25),
    "fault_refdrop_long":   ({}, _ref_dropout, 1.50),
    "fault_refobstruct":    ({}, _ref_obstructed, None),
    "fault_refgain_-12dB":  ({}, _ref_gain, -12.0),
    "fault_refdesync":      ({}, _ref_desync, 64),
    # loud bursts: above the r1/r2 training range (peaks capped at +12 dB re speech RMS) but inside r3's [15, 45] dB draw,
    # so for r3 these are in-distribution transient buckets, not generalisation tests
    "fault_burst_p24dB":    ({"impulse_peak_db": (24.0, 24.0)}, _identity, None),
    "fault_burst_p36dB":    ({"impulse_peak_db": (36.0, 36.0)}, _identity, None),
    "fault_burst_overload": ({"impulse_peak_db": (36.0, 36.0)}, _clip_primary, 0.25),
}


def render_fault_item(seed, speech_df, pool_df, n, snr, fault, bank):
    """One fault clip: same speech/noise/room as every other fault at this seed, so the fault is the only difference."""
    over, fn, arg = FAULTS[fault]
    # random mixer faults off: the fault under test must be the only degradation present
    cfg = MixConfig(snr_range=(snr, snr), p_clean=0.0, p_clip=0.0, p_ref_dropout=0.0, **over)
    burst = "synthetic" if "burst" in fault else None
    m, c, meta, twin = render_bucket_item(seed, speech_df, pool_df, n, snr, burst, bank, cfg=cfg)
    ctx = {}
    m = fn(m, arg, ctx)
    if twin is not None:
        twin = fn(twin, arg, ctx)
    meta.update(fault=fault, clipped="clip" in fault or "overload" in fault, ref_dropout="refdrop" in fault)
    return m, c, meta, twin


def _write(d, i, m, c, meta, twin):
    sf.write(d / f"{i:04d}.mix.wav", m.T, SR, subtype="FLOAT"); sf.write(d / f"{i:04d}.clean.wav", c, SR, subtype="FLOAT")
    json.dump(meta, open(d / f"{i:04d}.json", "w"))
    if twin is not None:
        sf.write(d / f"{i:04d}.twin.mix.wav", twin.T, SR, subtype="FLOAT")


def guard_frozen(a):
    """Refuse to re-render over a frozen eval set. Every published number is relative to a specific
    EVALSET_HASH, so overwriting one silently invalidates the whole results tree; its presence is the
    marker that a set was completed and scored. A partially written set has no hash and may be resumed."""
    stamp = Path(a.out) / a.split / "EVALSET_HASH"
    if stamp.exists() and not getattr(a, "force", False):
        raise SystemExit(f"{stamp} exists: {stamp.parent} is a frozen eval set and results reference its hash "
                         f"({stamp.read_text().strip()}). Render a new set to a different --out, or pass --force "
                         f"to overwrite it deliberately.")


def guard_bank(a):
    """A missing bank used to fall back silently to the parametric path, so a wrong path rendered a different set."""
    if getattr(a, "no_bank", False):
        return None
    if not Path(a.bank).exists():
        raise SystemExit(f"RIR bank {a.bank} not found. Build it (scripts/make_rir_bank.py) or pass --no-bank "
                         f"to render every item on the parametric path deliberately.")
    return a.bank


def _source_class(df):
    # source_id is "<corpus>:<class>/<clip>" for MAD and ESC-50
    return df.source_id.str.split(":").str[1].str.split("/").str[0]


def defence_pools(noise):
    """(beds, shots) for DEFENCE from one split's noise rows. An empty pool fails loudly, never substitutes."""
    mad, esc = noise[noise.corpus == "mad"], noise[noise.corpus == "esc50"]
    beds = {"mad_stationary": mad[mad.noise_class == "stationary"],
            "mad:helicopter": mad[_source_class(mad) == "helicopter"],
            "mad:vehicle": mad[_source_class(mad) == "vehicle"],
            "esc50:siren": esc[_source_class(esc) == "siren"]}
    shots = noise[noise.corpus.isin(GUNSHOT_CORPORA)]
    for k, v in {**beds, "gunshot": shots}.items():
        if v.empty:
            raise ValueError(f"no noise rows for defence pool {k!r}")
    return beds, shots


def _stamp(root):
    h = hashlib.sha1()
    for p in sorted(root.rglob("*.json")): h.update(p.read_bytes())
    (root / "EVALSET_HASH").write_text(h.hexdigest()[:12]); print("eval-set hash", h.hexdigest()[:12])


INDEX_COLS = ["bucket", "id", "category", "snr_db", "noise_class", "noise_source", "impulse_source", "impulse_peak_db",
              "overloaded", "clipped", "ref_dropout", "path", "speech_source"]


def main_defence(a, bank):
    """DEFENCE categories x BUCKET_SNRS from one split's rows, at r7's training transient levels; index.csv per item."""
    df = pd.concat([manifests.read(p) for p in a.manifests]); df = df[df.split == a.split]
    speech, noise = df[df.kind == "speech"], df[df.kind == "noise"]
    beds, shots = defence_pools(noise)
    root = Path(a.out) / a.split; n = int(a.clip_s * SR); rows = []
    for cat, (bed, impulse) in DEFENCE.items():
        for snr in BUCKET_SNRS:
            d = root / f"{cat}_{snr}"; d.mkdir(parents=True, exist_ok=True)
            cfg = MixConfig(snr_range=(snr, snr), p_clean=0.0, **DEFENCE_MIX)
            for i in range(a.per_bucket):
                seed = [a.seed, manifests.stable_hash(cat) % 1000, snr + 100, i]
                m, c, meta, twin = render_bucket_item(seed, speech, beds[bed], n, snr, impulse, bank, imp_df=shots,
                                                      cfg=cfg, tag_noise=True)
                meta["category"] = cat
                meta["noise_class"] = "impulsive+stationary" if impulse else CLASS_OF_BED[bed]
                _write(d, i, m, c, meta, twin)
                rows.append({**{k: meta.get(k) for k in INDEX_COLS}, "bucket": d.name, "id": f"{i:04d}"})
    with open(root / "index.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS); w.writeheader(); w.writerows(rows)
    _stamp(root)


# --- r8 test set (plan 11.2 G6, ex-B3): held-out groups, then one pre-registered render ---
HELDOUT_SCHEMA = "vaani.heldout_exclude/1"
# NOISEX-92 is one long file per noise type: hold out whole files, each with a same-type sibling left in training
NOISEX_HELDOUT = {"noisex92:buccaneer2": "buccaneer1", "noisex92:m109": "leopard", "noisex92:destroyerengine": "destroyerops"}
DRONE_HOLDOUT_MOD = 5   # stable_hash(recording) % 5 == 0: about 20 % of recordings
R8_V1_CORPORA = ("esc50", "mad")   # plus every dns_* corpus: eval_r2's nominal pools, MAD swapped for the video-grouped cut
R8_SPEECH_CORPORA = ("librispeech", "cv_hi")
R8_SUBSET_SIZES = {"v1": 20, "defence": 16, "heldout": 16, "loud": 16, "fault": 16, "v2": 48}
# test scenes are physical draws: the out-of-physics reference tail is a training augmentation (ablation 3b), not a condition
R8_V2 = {"tail_share": 0.0}
R8_INDEX_COLS = INDEX_COLS + ["subset", "scene"]


def drone_recording(source_id: str) -> str:
    """DroneAudioDataset rows are 1 s chunks "<recording>_NNN_"; the recording is the unit that must not straddle splits."""
    return re.sub(r"_\d+_?$", "", source_id)


def _ears_style(source_id: str) -> str:
    return source_id.split("/", 1)[1]


def heldout_spec(man_dir="data/manifests") -> dict:
    """The r8 held-out groups, per source. Matching is on source_id, which survives manifest re-cuts (mad vs mad_v2)."""
    md = Path(man_dir)
    drone = manifests.read(md / "drone.parquet"); noisex = manifests.read(md / "noisex92.parquet")
    ears = manifests.read(md / "ears.parquet"); mad = manifests.read(md / "mad_v2.parquet")
    drone = drone.assign(rec=drone.source_id.map(drone_recording))
    recs = sorted(r for r in drone.rec.unique() if manifests.stable_hash(r) % DRONE_HOLDOUT_MOD == 0)
    d_rows = drone[drone.rec.isin(recs)]
    missing = sorted(set(NOISEX_HELDOUT) - set(noisex.source_id))
    if missing:
        raise ValueError(f"NOISEX held-out files not in the manifest: {missing}")
    n_rows = noisex[noisex.source_id.isin(list(NOISEX_HELDOUT))]
    spk = min(sorted(ears[ears.split == "train"].group_id.unique()), key=manifests.stable_hash)
    e_rows = ears[ears.group_id == spk]
    m_rows = mad[mad.split == "test"]

    def entry(rows, **kw):
        return {**kw, "n_rows": int(len(rows)), "hours": round(float(rows.duration_s.sum()) / 3600, 4),
                "source_ids": sorted(rows.source_id)}
    return {
        "schema": HELDOUT_SCHEMA,
        "apply": ("r8 training and val pools drop every manifest row whose source_id is listed under any source below. "
                  "These rows exist only for data/eval_r8_test (results_r2/r8/testset/PROTOCOL.md)."),
        "generated_by": f"python scripts/render_eval_sets.py --write-heldout configs/data/r8_heldout_exclude.json --manifest-dir {man_dir}",
        "sources": {
            "drone": entry(d_rows, manifest="drone.parquet", group="recording: source_id minus its trailing _NNN_ chunk index",
                           rule=f"hold out recordings with manifests.stable_hash(recording) % {DRONE_HOLDOUT_MOD} == 0",
                           caveat=("inferred: adjacent recordings of one family may share a flight session, and the mixed_* "
                                   "families may reuse clean drone audio; independence holds at the recording-file level only"),
                           group_ids=recs),
            "noisex92": entry(n_rows, manifest="noisex92.parquet", group="file (one recording per noise type)",
                              rule="hold out three defence recordings whose same-type sibling stays in training: "
                                   + ", ".join(f"{k.split(':')[1]} (sibling {v})" for k, v in NOISEX_HELDOUT.items()),
                              group_ids=sorted(n_rows.group_id)),
            "ears": entry(e_rows, manifest="ears.parquet", group="speaker",
                          rule="hold out the whole train-split speaker with the smallest manifests.stable_hash(group_id); "
                               "the test set uses its *_loud clips (the val speaker stays in val)",
                          group_ids=[spk], test_clips=sorted(e_rows.source_id[e_rows.source_id.map(_ears_style).str.endswith("_loud")])),
            "mad": entry(m_rows, manifest="mad_v2.parquet", group="YouTube video (mad-yt-<id>)",
                         rule=("the video-grouped, speech-filtered mad_v2 test split; the older folder-grouped mad.parquet "
                               "put some of these videos in train, so they are excluded from any MAD manifest by source_id"),
                         group_ids=sorted(m_rows.group_id.unique())),
        },
    }


def read_heldout(path) -> tuple[dict, set]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema") != HELDOUT_SCHEMA:
        raise ValueError(f"{path}: schema {spec.get('schema')!r} != {HELDOUT_SCHEMA!r}")
    return spec, {s for v in spec["sources"].values() for s in v["source_ids"]}


def _r8_seed(a, subset, name, snr, i):
    return [a.seed, manifests.stable_hash(subset) % 1000, manifests.stable_hash(name) % 1000, int(snr) + 100, i]


def _drone_loader(rows):
    """Whole held-out recordings (their chunks joined in chunk order), drawn uniformly per recording, not per chunk."""
    rows = rows.assign(rec=rows.source_id.map(drone_recording)).sort_values("source_id")
    recs = {r: list(g.path) for r, g in rows.groupby("rec", sort=True)}
    names = sorted(recs)

    def load(rng, n):
        r = names[int(rng.integers(len(names)))]
        x = concat_xfade([sf.read(p, dtype="float32")[0] for p in recs[r]], int(0.05 * SR))
        return fit_xfade(x, n, rng, int(0.05 * SR)), r
    return load, recs


def _file_loader(rows):
    def load(rng, n):
        r = rows.iloc[int(rng.integers(len(rows)))]
        return _load(r.path, n, rng), str(r.source_id)
    return load


def render_heldout_item(seed, speech_df, load_noise, n, snr, bank, cfg=None):
    """Bed-only v1 clip whose noise comes from a loader (a held-out recording) rather than a manifest row."""
    rng = np.random.default_rng(seed)
    sp = speech_df.iloc[int(rng.integers(len(speech_df)))]
    x = _load(sp.path, n, rng); s = np.pad(x, (0, n - len(x)))
    nz, src = load_noise(rng, n)
    cfg = cfg or MixConfig(snr_range=(snr, snr), p_clean=0.0)
    m, c, meta = mix(np.random.default_rng(seed), s, [np.pad(nz, (0, n - len(nz)))], None, [], bank, cfg)
    meta.update(impulse_source=None, speech_source=str(sp.get("source_id", sp.path)), noise_source=src)
    return m, c, meta, None


def _scene_pool(noise, drone_rows):
    """ScenePool over the test noise rows; scenes.noise_tags gives drone rows no tag (v2 training drops the corpus), so
    the held-out recordings are indexed under "drone" here, one group per recording."""
    df = pd.concat([noise, drone_rows]).reset_index(drop=True)
    pool = scenes.ScenePool(df)
    is_d = (df.corpus == "drone").to_numpy()
    if is_d.any():
        rec = df.source_id.map(drone_recording).to_numpy()
        pool.index["drone"] = [np.flatnonzero(is_d & (rec == r)) for r in sorted(set(rec[is_d]))]
    return pool


def render_scene_item(seed, speech_df, pool, n, scene_name, bank, drone_recs=None, cfg=None):
    """One mixer-v2 scene clip: scene levels set the SNR (an output), no twin. drone_recs maps a drone recording to
    its chunk paths so a drawn drone row plays its whole recording instead of a looped 1 s chunk."""
    rng = np.random.default_rng(seed)
    sp = speech_df.iloc[int(rng.integers(len(speech_df)))]
    x = _load(sp.path, n, rng); s = np.pad(x, (0, n - len(x)))
    scene = scenes.sample_scene(rng, name=scene_name, crop_s=n / SR)
    rows, imp_row = pool.draw(rng, scene)
    noises = []
    for r in rows:
        if r is None:
            noises.append(np.zeros(n, np.float32))
        elif r.corpus == "drone" and drone_recs:
            noises.append(concat_xfade([sf.read(p, dtype="float32")[0] for p in drone_recs[drone_recording(r.source_id)]],
                                       int(0.05 * SR)))
        else:
            noises.append(_load(r.path, n, rng))
    imp, on, src = None, [], None
    ev = scene.get("event")
    if ev is not None and ev.get("fired"):
        if imp_row is None:   # no recorded impulsive row for the event's tags: physics-v2 synthetic blast
            kind = "artillery" if scene_name == "artillery" else "small_arms"
            imp, mt = impulses.generate(rng, kind="blast", blast_kind=kind, physics="v2")
            on, src = mt["onsets_s"], f"synthetic:blast:{mt['blast_kind']}"
        elif imp_row.corpus in GUNSHOT_CORPORA:
            imp, on, src = _peak_window(imp_row)
        else:
            imp = _load(imp_row.path, 2 * SR, rng); imp = (imp / (np.abs(imp).max() + 1e-9)).astype(np.float32)
            on, src = impulses.detect_onsets(imp, SR), str(imp_row.source_id)
    cfg = cfg or MixConfig(version=2, p_clean=0.0, v2=dict(R8_V2))
    m, c, meta = mix(rng, s, noises, imp, on, bank, cfg, scene=scene)
    meta.update(impulse_source=src, speech_source=str(sp.get("source_id", sp.path)), scene_params=scene,
                noise_source="+".join(x["source_id"] for x in scene["sources"] if x.get("source_id")) or None)
    return m, c, meta, None


def r8_pools(df, held_ids, spec):
    """Every pool of the r8 test set, from test-split rows plus the held-out rows the spec names. Fails loudly on a
    MAD row that is not in the video-grouped cut (the folder-grouped manifest leaks videos into train)."""
    # held-out drone/NOISEX/EARS rows keep their train/val split labels: the spec, not the split column, selects them
    allmad = df[df.corpus == "mad"]
    test = df[df.split == "test"]
    mad = test[test.corpus == "mad"]
    stray = sorted(set(mad.source_id) - set(spec["sources"]["mad"]["source_ids"]))
    if stray or not allmad.group_id.str.startswith("mad-yt-").any():   # the folder-grouped cut has no mad-yt- group
        raise ValueError(f"{len(stray)} MAD test rows outside the held-out mad_v2 split, or folder-grouped MAD rows: "
                         f"pass data/manifests/mad_v2.parquet, not mad.parquet")
    if allmad.source_id.duplicated().any():
        raise ValueError("MAD rows from more than one manifest: pass data/manifests/mad_v2.parquet only")
    speech = test[(test.kind == "speech") & test.corpus.isin(R8_SPEECH_CORPORA)]
    noise = test[test.kind == "noise"]
    v1 = noise[noise.corpus.isin(R8_V1_CORPORA) | noise.corpus.str.startswith("dns_")]
    held = df[df.source_id.isin(held_ids)].drop_duplicates("source_id")
    drone = held[held.corpus == "drone"]; noisex = held[held.corpus == "noisex92"]
    loud = held[held.source_id.isin(spec["sources"]["ears"]["test_clips"])]
    for k, v in {"speech": speech, "v1 noise": v1, "drone": drone, "noisex92": noisex, "ears loud": loud}.items():
        if v.empty:
            raise ValueError(f"no rows for r8 pool {k!r}")
    return dict(speech=speech, noise=noise, v1=v1, drone=drone, noisex=noisex, loud=loud)


def main_r8(a, bank):
    """The pre-registered r8 test set: labelled subsets in one split root, index.csv per item (category = subset/name)."""
    spec, held_ids = read_heldout(a.heldout)
    df = pd.concat([manifests.read(p) for p in a.manifests])
    P = r8_pools(df, held_ids, spec)
    # --r8-size shrinks every subset to one count (tests, smoke); the pre-registered set uses R8_SUBSET_SIZES
    sizes = {k: (getattr(a, "r8_size", None) or v) for k, v in R8_SUBSET_SIZES.items()}
    root = Path(a.out) / a.split; n = int(a.clip_s * SR); rows = []

    def emit(subset, name, bucket, i, item, scene=None):
        m, c, meta, twin = item
        meta["category"] = f"{subset}/{name}"; meta["subset"] = subset
        d = root / bucket; d.mkdir(parents=True, exist_ok=True)
        _write(d, i, m, c, meta, twin)
        rows.append({**{k: meta.get(k) for k in INDEX_COLS}, "bucket": bucket, "id": f"{i:04d}", "subset": subset,
                     "scene": scene})

    # v1 nominal: eval_r2's classes and mix defaults on fresh seeds and the video-grouped MAD
    impd = P["v1"][P["v1"].noise_class == "impulsive"]
    for cls, (cont, impulse) in CLASSES.items():
        pool = P["v1"][P["v1"].noise_class == cont]
        for snr in BUCKET_SNRS:
            for i in range(sizes["v1"]):
                it = render_bucket_item(_r8_seed(a, "v1", cls, snr, i), P["speech"], pool, n, snr, impulse, bank,
                                        imp_df=impd, tag_noise=True)
                it[2]["noise_class"] = cls
                emit("v1", cls, f"{cls}_{snr}", i, it)
    for i in range(sizes["v1"]):
        rng = np.random.default_rng(_r8_seed(a, "v1", "clean", 0, i))
        sp = P["speech"].iloc[int(rng.integers(len(P["speech"])))]
        s = np.pad((x := _load(sp.path, n, rng)), (0, n - len(x)))
        m, c, meta = mix(rng, s, [np.zeros(n, np.float32)], None, [], bank, MixConfig(p_clean=1.0))
        meta.update(noise_class="clean", speech_source=str(sp.get("source_id", sp.path)))
        emit("v1", "clean", "clean_inf", i, (m, c, meta, None))

    # defence: plan A3 categories at r7's training transient levels, physics-v2 blasts
    beds, shots = defence_pools(P["noise"])
    for cat, (bed, impulse) in DEFENCE.items():
        for snr in BUCKET_SNRS:
            cfg = MixConfig(snr_range=(snr, snr), p_clean=0.0, **DEFENCE_MIX)
            for i in range(sizes["defence"]):
                it = render_bucket_item(_r8_seed(a, "defence", cat, snr, i), P["speech"], beds[bed], n, snr, impulse,
                                        bank, imp_df=shots, cfg=cfg, tag_noise=True)
                it[2]["noise_class"] = "impulsive+stationary" if impulse else CLASS_OF_BED[bed]
                emit("defence", cat, f"{cat}_{snr}", i, it)

    # held-out recordings (never in any training pool): drone, NOISEX-92
    d_load, d_recs = _drone_loader(P["drone"])
    for name, load, cls in (("drone", d_load, "stationary"), ("noisex92", _file_loader(P["noisex"]), "stationary")):
        for snr in BUCKET_SNRS:
            for i in range(sizes["heldout"]):
                it = render_heldout_item(_r8_seed(a, "heldout", name, snr, i), P["speech"], load, n, snr, bank)
                it[2]["noise_class"] = cls
                emit("heldout", name, f"heldout_{name}_{snr}", i, it)

    # loud speech from the held-out EARS speaker over the MAD stationary bed (Lombard GRID is not on disk)
    for snr in BUCKET_SNRS:
        for i in range(sizes["loud"]):
            it = render_bucket_item(_r8_seed(a, "loud", "ears_loud", snr, i), P["loud"], beds["mad_stationary"], n, snr,
                                    None, bank, tag_noise=True)
            it[2]["noise_class"] = "stationary"
            emit("loud", "ears_loud", f"loud_ears_{snr}", i, it)

    # reliability faults, as eval_r2 test: one seed per (snr, i) shared by all faults
    st = P["v1"][P["v1"].noise_class == "stationary"]
    for fault in FAULTS:
        for snr in FAULT_SNRS:
            for i in range(sizes["fault"]):
                it = render_fault_item(_r8_seed(a, "fault", "shared", snr, i), P["speech"], st, n, snr, fault, bank)
                it[2]["noise_class"] = "impulsive+stationary" if "burst" in fault else "stationary"
                emit("fault", fault, f"{fault}_{snr}", i, it)

    # mixer-v2 scenes (M1-M4, M7, M10, M11): SPL-calibrated, SNR is an output; windy_ridge is the gusty-wind bucket
    pool = _scene_pool(P["noise"], P["drone"])
    for name in scenes.SCENE_WEIGHTS:
        for i in range(sizes["v2"]):
            it = render_scene_item(_r8_seed(a, "v2", name, 0, i), P["speech"], pool, n, name, bank, drone_recs=d_recs)
            emit("v2", name, f"v2_{name}", i, it, scene=name)

    with open(root / "index.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=R8_INDEX_COLS); w.writeheader(); w.writerows(rows)
    _stamp(root)


def ensure_eval_bank(path, n, seed, workers):
    """Eval-only RIR bank (mixer2 M6 seed namespace): its rooms share no draw stream with any training bank. Room mode
    is image-source only, so the M6 receiver radius (ray tracer) does not apply; no armoured entries (RirBank.sample
    cannot select them, so they would leak into every room scene)."""
    if Path(path).exists():
        return
    rirs.build_bank(Path(path), n=n, seed=seed, seed_namespace="eval", workers=workers)
    print("wrote eval bank", path)


def main(a):
    guard_frozen(a)   # before the manifests are read: a guard that fires after minutes of work is not a guard
    if getattr(a, "build_eval_bank", None):
        ensure_eval_bank(a.bank, a.build_eval_bank, a.bank_seed, a.bank_workers)
    bank_path = guard_bank(a)
    bank = RirBank(bank_path) if bank_path else None
    if getattr(a, "r8_test", False):
        return main_r8(a, bank)
    if getattr(a, "defence", False):
        return main_defence(a, bank)
    classes = CLASSES if not a.classes else {k: CLASSES[k] for k in a.classes}
    df = pd.concat([manifests.read(p) for p in a.manifests]); df = df[df.split == a.split]
    speech, noise = df[df.kind == "speech"], df[df.kind == "noise"]
    root = Path(a.out) / a.split; n = int(a.clip_s * SR)
    impd = noise[noise.noise_class == "impulsive"]

    for cls, (cont_cls, impulse) in classes.items():
        pool = noise[noise.noise_class == cont_cls]
        if pool.empty or (impulse == "corpus" and impd.empty):
            # silently substituting a different noise class would make the bucket's
            # label lie about its contents, so this must fail loudly instead
            raise ValueError(f"no noise rows for bucket {cls!r} (needs {cont_cls!r}{' + impulsive' if impulse == 'corpus' else ''})")
        for snr in BUCKET_SNRS:
            d = root / f"{cls}_{snr}"; d.mkdir(parents=True, exist_ok=True)
            for i in range(a.per_bucket):
                # stable_hash, not hash(): hash() is process-salted and would break reproducibility
                seed = [a.seed, manifests.stable_hash(cls) % 1000, snr + 100, i]
                m, c, meta, twin = render_bucket_item(seed, speech, pool, n, snr, impulse, bank, imp_df=impd)
                meta["noise_class"] = cls
                _write(d, i, m, c, meta, twin)

    if a.faults:
        pool = noise[noise.noise_class == "stationary"]
        for fault in FAULTS:
            for snr in FAULT_SNRS:
                d = root / f"{fault}_{snr}"; d.mkdir(parents=True, exist_ok=True)
                for i in range(a.per_bucket):
                    # one seed per (snr, i) shared by all faults: identical speech/noise/room, the fault is the only variable
                    seed = [a.seed, 777, snr + 100, i]
                    m, c, meta, twin = render_fault_item(seed, speech, pool, n, snr, fault, bank)
                    meta["noise_class"] = "impulsive+stationary" if "burst" in fault else "stationary"
                    _write(d, i, m, c, meta, twin)

    # clean bucket: no noise at all
    d = root / "clean_inf"; d.mkdir(exist_ok=True)
    for i in range(a.per_bucket):
        rng = np.random.default_rng([a.seed, 999, i])
        sp = speech.iloc[int(rng.integers(len(speech)))]
        s = np.pad((x := _load(sp.path, n, rng)), (0, n - len(x)))
        m, c, meta = mix(rng, s, [np.zeros(n, np.float32)], None, [], bank, MixConfig(p_clean=1.0))
        meta["noise_class"] = "clean"; meta["speech_source"] = str(sp.get("source_id", sp.path))
        _write(d, i, m, c, meta, None)

    # EVALSET_HASH is a WITHIN-platform integrity check, not a cross-platform identity proof. It
    # digests the metadata JSON as text, and the JSON carries measured floats (snr_achieved_db,
    # ref_speech_gain_db) whose last bit depends on the numpy/BLAS build. Verified 2026-09-23:
    # the same manifests and bank rendered on Windows and on Linux give different hashes
    # (17a9414959bb vs aa96a28a9955) while being the same set - 0 of 2280 items differ in any
    # selection field, the rendered audio is bit-identical (maxdiff 0 on int16 samples), and the
    # only deltas are those two fields at <= 1.9e-06, far below the int16 quantisation step.
    # To decide whether two renders are the same set, compare selections and audio, not this hash.
    _stamp(root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=None, help="required except with --write-heldout")
    ap.add_argument("--split", choices=["val", "test"], default=None, help="required except with --write-heldout")
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--bank", default="data/rirs/bank.npz")
    ap.add_argument("--per-bucket", type=int, default=40)
    ap.add_argument("--clip-s", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--faults", action="store_true", help="also render the reliability-fault buckets")
    ap.add_argument("--classes", nargs="+", choices=sorted(CLASSES), default=None,
                    help="render only these bucket classes (default: all). A held-out set built from one "
                         "corpus may legitimately have no rows of some class, and the impulse-bearing "
                         "buckets draw from training corpora, which a generalisation set must not do.")
    ap.add_argument("--no-bank", action="store_true",
                    help="render without a RIR bank (parametric path only); otherwise a missing --bank is an error")
    ap.add_argument("--defence", action="store_true",
                    help="render the DEFENCE categories (gunshot, blast, MAD helicopter/vehicle, siren) instead of CLASSES")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an already-frozen eval set (one carrying EVALSET_HASH); invalidates every result that cites its hash")
    ap.add_argument("--r8-test", action="store_true",
                    help="render the pre-registered r8 test set (results_r2/r8/testset/PROTOCOL.md) instead of CLASSES; needs --heldout")
    ap.add_argument("--heldout", default="configs/data/r8_heldout_exclude.json", help="held-out spec that --r8-test draws from")
    ap.add_argument("--r8-size", type=int, default=None, help="one per-bucket count for every r8 subset (smoke only)")
    ap.add_argument("--write-heldout", default=None, metavar="PATH",
                    help="write the r8 held-out spec (drone, NOISEX-92, EARS, MAD groups) to PATH and exit")
    ap.add_argument("--manifest-dir", default="data/manifests", help="where --write-heldout reads the manifests")
    ap.add_argument("--build-eval-bank", type=int, default=None, metavar="N",
                    help="build --bank as an N-room eval-only bank (rirs seed namespace 'eval') when it does not exist")
    ap.add_argument("--bank-seed", type=int, default=0, help="seed of --build-eval-bank")
    ap.add_argument("--bank-workers", type=int, default=3, help="simulation processes of --build-eval-bank")
    a = ap.parse_args()
    if a.write_heldout:
        Path(a.write_heldout).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write_heldout).write_text(json.dumps(heldout_spec(a.manifest_dir), indent=1) + "\n", encoding="utf-8")
        raise SystemExit(f"wrote {a.write_heldout}")
    if a.split is None:
        ap.error("--split is required")
    if not a.manifests:
        ap.error("--manifests is required")
    main(a)
