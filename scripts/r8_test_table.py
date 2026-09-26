#!/usr/bin/env python3
"""G6 table for the pre-registered r8 test set (results_r2/r8/testset/PROTOCOL.md, "Report layout").

    python scripts/r8_test_table.py            # the three PROTOCOL CSVs -> results_r2/r8/testset/table.md + table_cells.csv

Reads the three per-item CSVs that PROTOCOL.md's scoring commands write (`vaani.eval` on data/eval_r8_test_b, the
selected r8 checkpoint, r7 and raw) and writes the pre-registered layout:
- headline rows per subset (v1-nominal envelope, v1, defence, heldout, the Lombard/loud bucket, v2 scenes; fault apart);
- per category x input SNR: mean [95% clustered-bootstrap CI] of SNR_out / STOI / PESQ / DNSMOS OVRL, the per-clip
  all-three pass rate `pass3`, and PASS / ~ / FAIL against the PS targets (15 dB / 0.85 / 2.5, strict);
- r8 minus r7 and r8 minus raw, paired per item, clustered the same way;
- v2 per scene (SNR is a mixer output there) with the snr_in quantiles alongside, and fixed snr_in bands;
- the fault subset separately (as vaani/report.py), and recovery_s where the CSV has it;
- `pesq_nan` beside every PESQ column and the pesq 0.0.4 footnote under every PESQ table (Rachit, 2026-09-26).

Test root (Rachit, 2026-09-26): data/eval_r8_test_b (EVALSET_HASH 5bfda53eacbf). data/eval_r8_test (ed024af085a2) is
superseded, frozen and unscored: a root with that hash, or a CSV whose v2 rows carry that render's snr_in, is refused
unless --allow-superseded (the table is then labelled as not the pre-registered test).

Cluster (PROTOCOL): `impulse_source` for the gunshot category, `noise_source` otherwise; an item whose cluster key is
empty (fault/*, v1/clean: no noise bed) is its own cluster. Resamples: 1,000, seed 0 (vaani.report.cluster_ci).
Written before any r8 test score existed and tested only on synthetic CSVs (tests/test_r8_test_table.py).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vaani.report import TARGETS, cluster_ci, nominal_mask, pass_all  # noqa: E402

TESTSET = "results_r2/r8/testset"
DEFAULTS = {"r8": f"{TESTSET}/r8_selected.csv", "r7": f"{TESTSET}/r7_e256_wr64_cascade.csv", "raw": f"{TESTSET}/raw.csv"}
ROLES = ("r8", "r7", "raw")
CONTRASTS = (("r8", "r7"), ("r8", "raw"))
EVALSET_HASH = "5bfda53eacbf"     # test root B (Rachit, 2026-09-26); a set scored on anything else is refused
EVAL_ROOT = "data/eval_r8_test_b/test"
SUPERSEDED = {"ed024af085a2": "data/eval_r8_test/test"}     # the original render: frozen, never scored
# sha256[:12] of the sorted "bucket,id,snr_in:.3f" lines of the v2 rows (clip meta snr_db == index.csv snr_db): the renders
# share every (bucket, id) and differ in snr_in on v2 rows only, so this names the root a CSV was scored on
V2_DIGEST = {"5bfda53eacbf": "8a4f3615fdb7", "ed024af085a2": "bac5e7e0cc92"}
N_ITEMS = 2308
METRICS = [("snr_out", "SNR_out dB", 2), ("stoi", "STOI", 3), ("pesq_wb", "PESQ", 2), ("dnsmos_ovrl", "OVRL", 2)]
DELTAS = [("snr_out", "dSNR_out dB", 2), ("stoi", "dSTOI", 3), ("pesq_wb", "dPESQ", 2), ("dnsmos_ovrl", "dOVRL", 2),
          ("pass3", "dpass3", 2)]
SUBSETS = ["v1", "defence", "heldout", "loud", "v2", "fault"]
LABEL = {"v1": "v1 subset (mixer v1 defaults, as eval_r2 test)", "defence": "defence (A3 categories, physics-v2 blasts)",
         "heldout": "held-out drone / NOISEX-92 beds", "loud": "Lombard/loud bucket (EARS p004 loud clips)",
         "v2": "v2 mixer scenes (SNR is an output)", "fault": "reliability faults (outside the nominal envelope)"}
ORDER = {"v1": ["stationary", "changing", "impulsive", "impulsive+stationary", "recorded_impulsive",
                "recorded_impulsive+stationary", "clean"],
         "defence": ["gunshot", "blast_small_arms", "blast_artillery", "helicopter", "vehicle", "siren"],
         "heldout": ["drone", "noisex92"],
         "v2": ["patrol", "command_post", "drone", "artillery", "windy_ridge", "firefight", "apc", "helicopter"]}
# v2 snr_in bands, fixed before scoring (descriptive only; PROTOCOL registers the per-scene rows)
V2_BANDS = [(-np.inf, -5.0), (-5.0, 5.0), (5.0, 15.0), (15.0, np.inf)]
V2_QUANTILES = (0.1, 0.5, 0.9)
# decision 6 (Rachit, 2026-09-26): accept the garbage reads, count the NaNs, footnote every PESQ table
PESQ_NOTE = ("PESQ note (every PESQ column marked †): pesq 0.0.4 (the ITU P.862 reference C code) reads before the start "
             "of a heap buffer in `utterance_split` on some noise-dominated inputs (results_r2/r8/native_crash/README.md). "
             "Where that read faults, the isolated child of `vaani.metrics.pesq_wb` dies and the item scores NaN: it is "
             "counted in `pesq_nan`, left out of the PESQ mean, CI and mark, and counts as not passing in `pass3`. Where "
             "it does not fault, PESQ returns a value computed from out-of-bounds memory, which cannot be detected per "
             "item. Measured rate: 2 of 2,280 raw noisy eval_r2_relabel/test inputs (0.09 %) under ASan "
             "(results_r2/r8/native_crash/asan/sweep_relabel_test.tsv); the rate on model outputs was not measured.")
PESQ_FOOT = ("† PESQ: `pesq_nan` = items whose isolated PESQ child failed (NaN, outside the mean); about 0.09 % of raw "
             "inputs read garbage undetectably (PESQ note, top).")


def _bool(s):
    """clipped/ref_dropout arrive as bools, 'True'/'False' strings or blanks; blank means False."""
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)


def load(paths, expect_items=N_ITEMS, allow_partial=False):
    """{role: csv} -> one long frame with role, subset, name, snr, cluster and pass3; refuses inconsistent sets."""
    frames, keys = [], {}
    for role, p in paths.items():
        p = Path(p)
        if ".partial" in p.name:
            raise ValueError(f"{p}: a partial snapshot, not a final result")
        df = pd.read_csv(p, dtype={"id": str})
        if "category" not in df or df.category.isna().any():
            raise ValueError(f"{p}: every row needs `category` (vaani.eval copies it from index.csv)")
        if df.duplicated(["bucket", "id"]).any():
            raise ValueError(f"{p}: duplicate (bucket, id) rows; supply each final result once")
        if df.system.nunique() != 1:
            raise ValueError(f"{p}: {df.system.nunique()} systems in one file; one system per CSV")
        if expect_items and len(df) != expect_items and not allow_partial:
            raise ValueError(f"{p}: {len(df)} rows, expected {expect_items} (rerun the scoring with --resume)")
        keys[role] = set(zip(df.bucket, df.id))
        frames.append(df.assign(role=role))
    ks = list(keys.values())
    if any(k != ks[0] for k in ks[1:]) and not allow_partial:
        raise ValueError("the CSVs score different (bucket, id) sets; paired deltas need the same items")
    df = pd.concat(frames, ignore_index=True)
    for c in ("noise_source", "impulse_source", "fault", "recovery_s"):
        if c not in df:
            df[c] = np.nan
    for m, _, _ in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce") if m in df else np.nan
    df["snr_in"] = pd.to_numeric(df.snr_in, errors="coerce")
    df["clipped"], df["ref_dropout"] = _bool(df.clipped), _bool(df.ref_dropout)
    df["subset"] = df.category.str.split("/", n=1).str[0]
    df["name"] = df.category.str.split("/", n=1).str[1].fillna("")
    df["snr"] = np.where(df.subset == "v2", np.nan, df.snr_in.round(1))   # v2: continuous, no SNR cells
    key = np.where(df.name == "gunshot", df.impulse_source.fillna("").astype(str), df.noise_source.fillna("").astype(str))
    df["cluster"] = np.where(key == "", "item:" + df.bucket.astype(str) + "|" + df.id.astype(str), key)
    df["pass3"] = pass_all(df).astype(float)       # a NaN metric (failed clip) fails
    df["failed"] = df.snr_out.isna()
    df["pesq_nan"] = df.pesq_wb.isna() & ~df.failed     # the PESQ child alone failed; a failed clip is counted apart
    return df


def v2_digest(g):
    """V2_DIGEST key of one system's rows: which render its v2 snr_in values come from."""
    v = g[g.subset == "v2"]
    lines = sorted(f"{b},{i},{s:.3f}" for b, i, s in zip(v.bucket.astype(str), v.id.astype(str), v.snr_in))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]


def check_render(df, allow_superseded=False, allow_partial=False):
    """Note for the header; refuses a CSV scored on the superseded root (or on neither known render)."""
    notes = []
    for r in [x for x in ROLES if x in set(df.role)]:
        d = v2_digest(df[df.role == r]); root = next((h for h, v in V2_DIGEST.items() if v == d), None)
        if root == EVALSET_HASH:
            continue
        if root in SUPERSEDED:
            if not allow_superseded:
                raise ValueError(f"{r}: CSV scored on the superseded root {root} ({SUPERSEDED[root]}); the r8 test root "
                                 f"is {EVALSET_HASH} ({EVAL_ROOT}). --allow-superseded overrides (labelled)")
            notes.append(f"**`{r}` was scored on the SUPERSEDED root {root} ({SUPERSEDED[root]}; --allow-superseded): "
                         "not the pre-registered test.**")
        elif allow_partial:
            notes.append(f"`{r}`: v2 rows match no known render (digest {d}; partial draft, render not verified).")
        else:
            raise ValueError(f"{r}: v2 snr_in digest {d} matches no known render {V2_DIGEST}")
    return " ".join(notes) or f"Every CSV's v2 snr_in matches the `{EVALSET_HASH}` render."


def _v1_nominal(g):
    """vaani.report's nominal envelope inside v1: unclipped, no ref dropout, no fault, input SNR 0/5/10 dB."""
    v = g[g.subset == "v1"]
    return v[nominal_mask(v.assign(fault=v.fault.where(v.fault.notna() & (v.fault.astype(str) != ""), np.nan)))]


def stats(g, n_boot):
    out = {m: cluster_ci(g[m], g.cluster, n=n_boot) if g[m].notna().any() else (np.nan,) * 3 for m, _, _ in METRICS}
    out["pass3"] = cluster_ci(g.pass3, g.cluster, n=n_boot) if len(g) else (np.nan,) * 3
    return out


def mark(m, t):
    """PASS = lower 95% bound above target; ~ = mean above, bound not; FAIL = mean not above."""
    mean, lo, _ = t
    if m not in TARGETS or not np.isfinite(mean):
        return ""
    return "PASS" if lo > TARGETS[m] else ("~" if mean > TARGETS[m] else "FAIL")


def verdict(st):
    """All three PS targets on one row: PASS if every bound clears, ~ if every mean does, else FAIL."""
    ms = [mark(m, st[m]) for m in TARGETS]
    if "" in ms:
        return "n/a"
    return "PASS" if all(x == "PASS" for x in ms) else ("~" if "FAIL" not in ms else "FAIL")


def fmt(t, d=2):
    return "n/a" if not np.isfinite(t[0]) else f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def snr_label(v):
    return "clean" if np.isinf(v) else f"{v:g}"


def paired(df, a, b):
    """Per item a - b for every delta metric, on the items both scored; cluster and cell keys from a's rows."""
    cols = [m for m, _, _ in DELTAS]
    A = df[df.role == a].set_index(["bucket", "id"]); B = df[df.role == b].set_index(["bucket", "id"])
    j = A[cols + ["pesq_nan", "cluster", "subset", "name", "snr", "snr_in", "clipped", "ref_dropout", "fault"]].join(
        B[cols + ["pesq_nan"]], rsuffix="_b", how="inner")
    for m in cols:
        j[m] = j[m] - j[f"{m}_b"]
    j["pesq_nan"] = j.pesq_nan | j.pesq_nan_b          # dPESQ is NaN when either side's child failed
    return j.drop(columns=[f"{m}_b" for m in cols + ["pesq_nan"]]).reset_index()


def _heads(spec):
    """Column heads: PESQ marked with the footnote dagger and followed by a pesq_nan count column."""
    out = []
    for m, h, _ in spec:
        out += [h + "†", "pesq_nan"] if m == "pesq_wb" else [h]
    return out


class Table:
    def __init__(self, df, n_boot=1000):
        self.df, self.n_boot, self.L, self.cells = df, n_boot, [], []
        self.roles = [r for r in ROLES if r in set(df.role)]
        self.pairs = {(a, b): paired(df, a, b) for a, b in CONTRASTS if a in self.roles and b in self.roles}

    def _cell(self, section, subset, category, snr, system, st, n, pesq_nan=None):
        for m, t in st.items():
            self.cells.append({"section": section, "subset": subset, "category": category, "snr_in": snr,
                               "system": system, "metric": m, "mean": t[0], "lo": t[1], "hi": t[2], "n": n})
        if pesq_nan is not None:
            self.cells.append({"section": section, "subset": subset, "category": category, "snr_in": snr,
                               "system": system, "metric": "pesq_nan", "mean": pesq_nan, "lo": np.nan, "hi": np.nan,
                               "n": n})

    def _rows(self, section, head, groups):
        """groups: [(label cells, subset, category, snr, frame of all roles)] -> one row per role."""
        self.L += [f"| {head} | system | n | " + " | ".join(_heads(METRICS)) + " | pass3 | targets |",
                   "|---" * (head.count("|") + len(METRICS) + 6) + "|"]
        for lab, subset, cat, snr, g in groups:
            for r in self.roles:
                x = g[g.role == r]
                if x.empty:
                    continue
                k = int(x.pesq_nan.sum()); st = stats(x, self.n_boot)
                self._cell(section, subset, cat, snr, r, st, len(x), k)
                cs = [fmt(st[m], d) + (f" {mark(m, st[m])}" if m in TARGETS else "") for m, _, d in METRICS]
                cs.insert(3, str(k))
                self.L.append(f"| {lab} | {r} | {len(x)} | " + " | ".join(cs) + f" | {fmt(st['pass3'])} | {verdict(st)} |")
        self.L += ["", PESQ_FOOT, ""]

    def _deltas(self, section, head, groups):
        """groups: [(label, subset, category, snr, mask fn on a paired frame)]."""
        for (a, b), j in self.pairs.items():
            self.L += [f"{a} minus {b}, paired per item:", "",
                       f"| {head} | n | " + " | ".join(_heads(DELTAS)) + " |",
                       "|---" * (head.count("|") + len(DELTAS) + 3) + "|"]
            for lab, subset, cat, snr, sel in groups:
                x = sel(j)
                if x.empty:
                    continue
                st = {m: cluster_ci(x[m], x.cluster, n=self.n_boot) for m, _, _ in DELTAS}
                k = int(x.pesq_nan.sum()); self._cell(section, subset, cat, snr, f"{a}-{b}", st, len(x), k)
                cs = [fmt(st[m], d) for m, _, d in DELTAS]; cs.insert(3, str(k))
                self.L.append(f"| {lab} | {len(x)} | " + " | ".join(cs) + " |")
            self.L += ["", PESQ_FOOT, ""]

    def headline(self):
        df = self.df
        subs = [("v1-nominal envelope", "v1", "nominal", _v1_nominal)]
        subs += [(LABEL[s], s, "all", (lambda s: lambda g: g[g.subset == s])(s)) for s in SUBSETS if s in set(df.subset)]
        self.L += ["## Headline: every subset pooled", "",
                   "The v1-nominal envelope is vaani/report.py's (unclipped, no reference dropout, no fault, input SNR "
                   "0/5/10 dB) inside the v1 subset: the same construction as the eval_r2 nominal row, not the same items. "
                   "Fault is outside every envelope and is listed last only for completeness.", ""]
        self._rows("headline", "subset", [(lab, s, c, np.nan, f(df)) for lab, s, c, f in subs])
        self._deltas("headline", "subset", [(lab, s, c, np.nan, f) for lab, s, c, f in subs])

    def per_cell(self, subset):
        df = self.df[self.df.subset == subset]
        if df.empty:
            return
        cats = [c for c in ORDER.get(subset, []) if c in set(df.name)] + sorted(set(df.name) - set(ORDER.get(subset, [])))
        snrs = sorted(df.snr.dropna().unique(), key=lambda v: (np.isinf(v), v))
        self.L += [f"## {LABEL[subset]}: per category x input SNR", ""]
        if subset == "fault":
            self.L += ["Fault buckets share seeds with fault_none: read each fault against fault_none, not against the "
                       "nominal rows.", ""]
        if subset == "loud":
            self.L += ["One held-out speaker's 12 loud clips are reused across the 96 items; the noise-bed clustering "
                       "does not resample speech, so these CIs are optimistic (inferred).", ""]
        groups = [(f"{c} | {snr_label(s)}", subset, f"{subset}/{c}", s, df[(df.name == c) & (df.snr == s)])
                  for c in cats for s in snrs if ((df.name == c) & (df.snr == s)).any()]
        self._rows("cell", "category | SNR_in", groups)
        self._deltas("cell", "category | SNR_in", [(lab, sub, cat, s, (lambda c, s: lambda j: j[
            (j.subset == subset) & (j.name == c) & (j.snr == s)])(cat.split("/", 1)[1], s))
            for lab, sub, cat, s, _ in groups])

    def v2(self):
        df = self.df[self.df.subset == "v2"]
        if df.empty:
            return
        scenes = [c for c in ORDER["v2"] if c in set(df.name)] + sorted(set(df.name) - set(ORDER["v2"]))
        ref = df[df.role == self.roles[0]]
        self.L += ["## v2 mixer scenes: per scene", "",
                   "SNR is an output of the SPL-calibrated mixer, not a controlled variable: each scene row carries "
                   "its snr_in quantiles (from the scored rows) and the share of items whose input passed the 123 dB "
                   "rails (`clipped`). PS-target marks are shown but a scene is not an SNR tier.", "",
                   "| scene | n | snr_in q10 / q50 / q90 dB | clipped share |", "|---|---|---|---|"]
        for c in scenes:
            x = ref[ref.name == c]; q = x.snr_in.quantile(list(V2_QUANTILES)).to_numpy()
            self.L.append(f"| {c} | {len(x)} | " + " / ".join(f"{v:.1f}" for v in q) + f" | {x.clipped.mean():.2f} |")
            for p, v in zip(V2_QUANTILES, q):
                self.cells.append({"section": "v2_snr_in", "subset": "v2", "category": f"v2/{c}", "snr_in": np.nan,
                                   "system": "index", "metric": f"snr_in_q{int(p * 100)}", "mean": v, "lo": np.nan,
                                   "hi": np.nan, "n": len(x)})
        self.L.append("")
        groups = [(c, "v2", f"v2/{c}", np.nan, df[df.name == c]) for c in scenes]
        self._rows("v2_scene", "scene", groups)
        self._deltas("v2_scene", "scene", [(c, "v2", f"v2/{c}", np.nan, (lambda c: lambda j: j[
            (j.subset == "v2") & (j.name == c)])(c)) for c in scenes])
        self.L += ["### v2, all scenes pooled by fixed snr_in band (descriptive; bands fixed before scoring)", ""]
        bands = [(f"[{lo:g}, {hi:g}) dB", lo, hi) for lo, hi in V2_BANDS]
        self._rows("v2_band", "snr_in band", [(lab, "v2", f"v2/band{lo:g}", lo,
                                                df[(df.snr_in >= lo) & (df.snr_in < hi)]) for lab, lo, hi in bands])
        self._deltas("v2_band", "snr_in band", [(lab, "v2", f"v2/band{lo:g}", lo, (lambda lo, hi: lambda j: j[
            (j.subset == "v2") & (j.snr_in >= lo) & (j.snr_in < hi)])(lo, hi)) for lab, lo, hi in bands])

    def recovery(self):
        df = self.df[pd.to_numeric(self.df.recovery_s, errors="coerce").notna()]
        if df.empty:
            return
        self.L += ["## Recovery time after an impulse (recovery_s; inf = never recovered)", "",
                   "| subset | system | n | median s | p90 s | never |", "|---|---|---|---|---|---|"]
        for s in [x for x in SUBSETS if x in set(df.subset)]:
            for r in self.roles:
                v = pd.to_numeric(df[(df.subset == s) & (df.role == r)].recovery_s, errors="coerce")
                if len(v):
                    self.L.append(f"| {s} | {r} | {len(v)} | {v.median():.3f} | {v.quantile(0.9):.3f} | "
                                  f"{int(np.isinf(v).sum())} |")
        self.L.append("")


def build(df, n_boot=1000, hash_note=""):
    t = Table(df, n_boot)
    specs = {r: df[df.role == r].system.iloc[0] for r in t.roles}
    t.L += ["# r8 test set (G6): the single post-selection scoring", "",
            f"Generated by `scripts/r8_test_table.py` from the per-item CSVs of PROTOCOL.md. {hash_note}", "",
            "Systems: " + "; ".join(f"`{r}` = `{specs[r]}` ({int((df.role == r).sum())} rows, "
                                    f"{int((df.failed & (df.role == r)).sum())} failed clips, "
                                    f"{int((df.pesq_nan & (df.role == r)).sum())} pesq_nan)" for r in t.roles) + ".", "",
            PESQ_NOTE, "",
            "PS targets (SIH26052): SNR_out > 15 dB, STOI > 0.85, PESQ > 2.5, strict. Cells: mean [95% CI]. Mark: PASS = "
            "lower bound above target, ~ = mean above but bound not, FAIL = mean not above. `targets` = all three at "
            "once on the row (PASS / ~ / FAIL). `pass3` = share of clips meeting all three at once; a failed clip "
            "(NaN metric) counts as not passing.", "",
            f"CIs: clustered bootstrap ({n_boot} resamples, seed 0, vaani.report.cluster_ci). Cluster = `impulse_source` "
            "for gunshot, `noise_source` otherwise (PROTOCOL.md); an item with no noise bed (fault/*, v1/clean) is its "
            "own cluster. Deltas are paired per (bucket, id) and clustered the same way. CIs resample items of one "
            "checkpoint; they do not measure training-seed variance.", ""]
    t.headline()
    for s in ("v1", "defence", "heldout", "loud"):
        t.per_cell(s)
    t.v2()
    t.per_cell("fault")
    t.recovery()
    t.L += ["## Known limits (PROTOCOL.md, stated before scoring)", "",
            "- Every mixture is synthetic. The loud bucket is one EARS speaker; the Lombard effect in v2 is the alpha "
            "tilt only (no F0 shift).",
            "- Siren has two ESC-50 test recordings and NOISEX-92 three held-out files: their CIs are wide.",
            "- v2 `clipped` = input past the 123 dB SPL rails; the v2 clean target is pre-saturation, so v2 scores "
            "include saturation loss by design.",
            "- PESQ garbage reads (PESQ note, top; accepted, Rachit 2026-09-26): about 0.09 % of raw inputs, not "
            "detectable per item; `pesq_nan` counts only the reads that faulted.", ""]
    return "\n".join(t.L), pd.DataFrame(t.cells)


def check_hash(eval_root, allow_superseded=False):
    """Note for the header; raises if the set on disk is not the pre-registered one."""
    h = Path(eval_root) / "EVALSET_HASH"
    if not h.exists():
        return f"Eval set hash not checked here ({h} absent); PROTOCOL registers `{EVALSET_HASH}`."
    got = h.read_text().strip()
    if got in SUPERSEDED and allow_superseded:
        return f"**Eval set EVALSET_HASH `{got}`: the SUPERSEDED root (--allow-superseded), not PROTOCOL's.**"
    if got in SUPERSEDED:
        raise ValueError(f"{h} = {got!r}: the superseded root ({SUPERSEDED[got]}); the r8 test root is {EVALSET_HASH!r} "
                         f"({EVAL_ROOT}). --allow-superseded overrides (labelled)")
    if got != EVALSET_HASH:
        raise ValueError(f"{h} = {got!r}, PROTOCOL registers {EVALSET_HASH!r}")
    return f"Eval set `{EVAL_ROOT}`, EVALSET_HASH `{got}` (matches PROTOCOL)."


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for r in ROLES:
        ap.add_argument(f"--{r}", default=str(REPO / DEFAULTS[r]), help=f"per-item CSV of {r} (default {DEFAULTS[r]})")
    ap.add_argument("--eval-root", default=str(REPO / EVAL_ROOT))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--expect-items", type=int, default=N_ITEMS)
    ap.add_argument("--allow-partial", action="store_true", help="accept short or mismatched CSVs (drafts only)")
    ap.add_argument("--allow-superseded", action="store_true",
                    help=f"accept the superseded root {', '.join(SUPERSEDED)} (never for the G6 table; labelled)")
    ap.add_argument("--out", default=str(REPO / TESTSET / "table.md"))
    ap.add_argument("--cells", help="long-format CSV of every number in the table (default: <out stem>_cells.csv)")
    a = ap.parse_args(argv)
    df = load({r: getattr(a, r) for r in ROLES}, a.expect_items, a.allow_partial)
    note = check_hash(a.eval_root, a.allow_superseded) + " " + check_render(df, a.allow_superseded, a.allow_partial)
    md, cells = build(df, a.boot, note)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    cells.to_csv(a.cells or out.with_name(out.stem + "_cells.csv"), index=False)
    print("wrote", out)
    return md, cells


if __name__ == "__main__":
    main()
