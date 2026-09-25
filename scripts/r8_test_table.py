#!/usr/bin/env python3
"""G6 table for the pre-registered r8 test set (results_r2/r8/testset/PROTOCOL.md, "Report layout").

    python scripts/r8_test_table.py            # the three PROTOCOL CSVs -> results_r2/r8/testset/table.md + table_cells.csv

Reads the three per-item CSVs that PROTOCOL.md's scoring commands write (`vaani.eval` on data/eval_r8_test, the
selected r8 checkpoint, r7 and raw) and writes the pre-registered layout:
- headline rows per subset (v1-nominal envelope, v1, defence, heldout, the Lombard/loud bucket, v2 scenes; fault apart);
- per category x input SNR: mean [95% clustered-bootstrap CI] of SNR_out / STOI / PESQ / DNSMOS OVRL, the per-clip
  all-three pass rate `pass3`, and PASS / ~ / FAIL against the PS targets (15 dB / 0.85 / 2.5, strict);
- r8 minus r7 and r8 minus raw, paired per item, clustered the same way;
- v2 per scene (SNR is a mixer output there) with the snr_in quantiles alongside, and fixed snr_in bands;
- the fault subset separately (as vaani/report.py), and recovery_s where the CSV has it.

Cluster (PROTOCOL): `impulse_source` for the gunshot category, `noise_source` otherwise; an item whose cluster key is
empty (fault/*, v1/clean: no noise bed) is its own cluster. Resamples: 1,000, seed 0 (vaani.report.cluster_ci).
Written before any r8 test score existed and tested only on synthetic CSVs (tests/test_r8_test_table.py).
"""
import argparse
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
EVALSET_HASH = "ed024af085a2"     # PROTOCOL.md; a CSV set scored on anything else is refused
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
    return df


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
    j = A[cols + ["cluster", "subset", "name", "snr", "snr_in", "clipped", "ref_dropout", "fault"]].join(
        B[cols], rsuffix="_b", how="inner")
    for m in cols:
        j[m] = j[m] - j[f"{m}_b"]
    return j.drop(columns=[f"{m}_b" for m in cols]).reset_index()


class Table:
    def __init__(self, df, n_boot=1000):
        self.df, self.n_boot, self.L, self.cells = df, n_boot, [], []
        self.roles = [r for r in ROLES if r in set(df.role)]
        self.pairs = {(a, b): paired(df, a, b) for a, b in CONTRASTS if a in self.roles and b in self.roles}

    def _cell(self, section, subset, category, snr, system, st, n):
        for m, t in st.items():
            self.cells.append({"section": section, "subset": subset, "category": category, "snr_in": snr,
                               "system": system, "metric": m, "mean": t[0], "lo": t[1], "hi": t[2], "n": n})

    def _rows(self, section, head, groups):
        """groups: [(label cells, subset, category, snr, frame of all roles)] -> one row per role."""
        self.L += [f"| {head} | system | n | " + " | ".join(h for _, h, _ in METRICS) + " | pass3 | targets |",
                   "|---" * (head.count("|") + len(METRICS) + 5) + "|"]
        for lab, subset, cat, snr, g in groups:
            for r in self.roles:
                x = g[g.role == r]
                if x.empty:
                    continue
                st = stats(x, self.n_boot); self._cell(section, subset, cat, snr, r, st, len(x))
                cs = [fmt(st[m], d) + (f" {mark(m, st[m])}" if m in TARGETS else "") for m, _, d in METRICS]
                self.L.append(f"| {lab} | {r} | {len(x)} | " + " | ".join(cs) + f" | {fmt(st['pass3'])} | {verdict(st)} |")
        self.L.append("")

    def _deltas(self, section, head, groups):
        """groups: [(label, subset, category, snr, mask fn on a paired frame)]."""
        for (a, b), j in self.pairs.items():
            self.L += [f"{a} minus {b}, paired per item:", "",
                       f"| {head} | n | " + " | ".join(h for _, h, _ in DELTAS) + " |",
                       "|---" * (head.count("|") + len(DELTAS) + 2) + "|"]
            for lab, subset, cat, snr, sel in groups:
                x = sel(j)
                if x.empty:
                    continue
                st = {m: cluster_ci(x[m], x.cluster, n=self.n_boot) for m, _, _ in DELTAS}
                self._cell(section, subset, cat, snr, f"{a}-{b}", st, len(x))
                self.L.append(f"| {lab} | {len(x)} | " + " | ".join(fmt(st[m], d) for m, _, d in DELTAS) + " |")
            self.L.append("")

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
                                    f"{int((df.failed & (df.role == r)).sum())} failed clips)" for r in t.roles) + ".", "",
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
            "include saturation loss by design.", ""]
    return "\n".join(t.L), pd.DataFrame(t.cells)


def check_hash(eval_root):
    """Note for the header; raises if the set on disk is not the pre-registered one."""
    h = Path(eval_root) / "EVALSET_HASH"
    if not h.exists():
        return f"Eval set hash not checked here ({h} absent); PROTOCOL registers `{EVALSET_HASH}`."
    got = h.read_text().strip()
    if got != EVALSET_HASH:
        raise ValueError(f"{h} = {got!r}, PROTOCOL registers {EVALSET_HASH!r}")
    return f"Eval set `data/eval_r8_test/test`, EVALSET_HASH `{got}` (matches PROTOCOL)."


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for r in ROLES:
        ap.add_argument(f"--{r}", default=str(REPO / DEFAULTS[r]), help=f"per-item CSV of {r} (default {DEFAULTS[r]})")
    ap.add_argument("--eval-root", default=str(REPO / "data/eval_r8_test/test"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--expect-items", type=int, default=N_ITEMS)
    ap.add_argument("--allow-partial", action="store_true", help="accept short or mismatched CSVs (drafts only)")
    ap.add_argument("--out", default=str(REPO / TESTSET / "table.md"))
    ap.add_argument("--cells", help="long-format CSV of every number in the table (default: <out stem>_cells.csv)")
    a = ap.parse_args(argv)
    df = load({r: getattr(a, r) for r in ROLES}, a.expect_items, a.allow_partial)
    md, cells = build(df, a.boot, check_hash(a.eval_root))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    cells.to_csv(a.cells or out.with_name(out.stem + "_cells.csv"), index=False)
    print("wrote", out)
    return md, cells


if __name__ == "__main__":
    main()
