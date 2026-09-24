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
helicopter/vehicle, ESC-50 siren) with r7's training mix block. Writes an
eval-set hash for run.json.
"""
import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import lfilter

from vaani.data import impulses, manifests
from vaani.data.dataset import BUCKET_SNRS, SR, _load
from vaani.data.mixer import MixConfig, mix
from vaani.data.rirs import RirBank

# bucket -> (continuous noise class, impulse source); None = no burst
CLASSES = {"stationary": ("stationary", None), "changing": ("changing", None),
           "impulsive": ("changing", "synthetic"), "impulsive+stationary": ("stationary", "synthetic"),
           "recorded_impulsive": ("changing", "corpus"), "recorded_impulsive+stationary": ("stationary", "corpus")}
FAULT_SNRS = [0, 5]

# defence category -> (continuous bed, impulse source). Transient categories sit on the MAD stationary bed
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
        imp, m = impulses.generate(rng, kind="blast", blast_kind=source.split(":", 1)[1])
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


def main(a):
    guard_frozen(a)   # before the manifests are read: a guard that fires after minutes of work is not a guard
    bank_path = guard_bank(a)
    bank = RirBank(bank_path) if bank_path else None
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
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
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
    main(ap.parse_args())
