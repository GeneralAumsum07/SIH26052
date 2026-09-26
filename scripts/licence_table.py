"""Licence provenance for every corpus, generated from the manifests rather than maintained by hand.

A DRDO reviewer is entitled to ask what the system trains on and whether that material could follow
the design into a fielded product. The answer has to come from the manifests, because they are what
training actually reads -- a hand-maintained table drifts the moment a recipe changes, and a drifted
licence table is worse than none.

The commercial-use column is the only judgement here, and it is deliberately coarse: `no` for anything
carrying NC, `unclear` where the redistribution terms are genuinely unresolved, `yes` only for permissive
terms we can name. Where it says unclear it means unclear -- that is a finding, not an omission.

    uv run python scripts/licence_table.py --recipe configs/retraining/r7_e256_wr64.yaml --out docs/licences.md

Several recipes render one section each (the shipping r7 recipe and the r8 retrain):

    uv run python scripts/licence_table.py --recipe configs/retraining/r7_e256_wr64.yaml \\
        --recipe configs/retraining/r8_fe_mini.yaml --out docs/licences.md

A recipe manifest that is not scanned here (the r8 corpora only the training box downloads) takes its licence from
the dataset registry (configs/data/r8_datasets.yaml) and shows its size as a box-scan TBD.
"""
import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

# Substring -> (commercial use, short name). Order matters: NC is checked before the permissive CC forms,
# because "CC BY-NC 4.0" contains "CC BY".
RULES = [
    ("scanner keeps CC0 and CC BY", ("yes", "CC0 / CC BY per clip (NC and Sampling+ clips dropped)")),
    ("conflict: Zenodo field CC BY 4.0, record text CC BY-SA", ("yes, share-alike", "CC BY-SA (sources conflict)")),
    ("CC BY-NC", ("no", "CC BY-NC")),
    ("/licenses/by-nc", ("no", "CC BY-NC")),   # FSD50K rows record the licence URL, not its name
    ("publicdomain/zero", ("yes", "CC0")),
    ("/licenses/by/", ("yes", "CC BY")),
    ("CC0", ("yes", "CC0")),
    ("CC BY-SA", ("yes, share-alike", "CC BY-SA 4.0")),
    ("CC BY", ("yes", "CC BY 4.0")),
    ("NIJ", ("unclear", "US DOJ/NIJ award output")),
    ("DroneAudioDataset", ("unclear", "citation-on-use")),
    ("NOISEX-92", ("unclear", "SPIB redistribution terms unstated")),
    ("MAD", ("no", "YouTube-sourced")),
    ("DNS-4", ("unclear", "per-clip, see DNS README")),
]
REGISTRY = "configs/data/r8_datasets.yaml"
# recorded licences that disagree between sources (docs/impl/2026-09-24/research/datasets_final.md); the table keeps
# the stricter reading
CONFLICTS = {
    "mad": "Kaggle metadata says CC BY-SA 4.0, the authors' README CC BY 4.0; the audio itself is YouTube-sourced, so "
           "the table records it as non-commercial either way.",
    "demand": "the Zenodo licence field says CC BY 4.0, the record text CC BY-SA 3.0 (reported, not re-read), and the "
              "scanner records CC BY-SA 4.0; the table keeps share-alike.",
}
HEAD = ["| corpus | kind | clips | hours | licence | commercial use |", "|---|---|---:|---:|---|---|"]
TRANSFER = ("The research prototype trains on corpora including non-commercial-licensed and "
            "unclear-licence material. A fielded version would retrain on licensed or "
            "government-collected data; the pipeline is corpus-agnostic and the manifest layer makes "
            "the substitution mechanical -- a recipe is a list of manifest paths, and nothing in the "
            "model, the DSP front end or the training loop is tied to a particular corpus.")
DISCLAIMER = "This is a licence statement, not a legal opinion, and `unclear` rows are unresolved rather than cleared."


def classify(licence: str):
    for needle, verdict in RULES:
        if needle in licence:
            return verdict
    return ("unclear", licence)


def recipe_manifests(path):
    """The manifests the deployed recipe actually trains on, so the table cannot claim a corpus is in
    use when the winning config dropped it."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    node = cfg.get("data", cfg)
    return {Path(m).name for m in node.get("manifests", [])}


def registry_licences(path=REGISTRY):
    """{manifest file name: (corpus, licence)} from the dataset registry, for manifests not scanned here."""
    p = Path(path)
    if not p.exists():
        return {}
    reg = yaml.safe_load(p.read_text(encoding="utf-8"))
    out = {}
    for block in ("datasets", "optional"):
        for name, v in (reg.get(block) or {}).items():
            mans = v.get("manifest") or []
            for m in [mans] if isinstance(mans, str) else mans:
                out[Path(m).name] = (name, str(v.get("licence", "")))
    return out


def collect(manifest_dir, in_recipe, registry=None):
    """One row per (manifest, corpus, kind, licence). A recipe manifest absent from manifest_dir gets one registry row
    with clips/hours unknown (it is scanned on the training box only)."""
    rows = []
    have = {f.name for f in Path(manifest_dir).glob("*.parquet")}
    for name in sorted(set(in_recipe) - have):
        corpus, licence = (registry or {}).get(name, (Path(name).stem, "not in the registry"))
        commercial, short = classify(licence)
        rows.append({"corpus": corpus, "kind": "box scan", "clips": None, "hours": None, "licence": short,
                     "commercial": commercial, "in_recipe": True, "manifest": name})
    for f in sorted(Path(manifest_dir).glob("*.parquet")):
        cols = set(pq.read_schema(f).names)
        if not {"corpus", "kind", "licence", "duration_s"} <= cols:   # an audit table (mad_speech_contamination), not a corpus
            continue
        df = pd.read_parquet(f, columns=["corpus", "kind", "licence", "duration_s"])
        for (corpus, kind, licence), g in df.groupby(["corpus", "kind", "licence"], dropna=False):
            commercial, short = classify(str(licence))
            rows.append({"corpus": corpus, "kind": kind, "clips": len(g),
                         "hours": g.duration_s.sum() / 3600.0, "licence": short,
                         "commercial": commercial, "in_recipe": f.name in in_recipe,
                         "manifest": f.name})
    return pd.DataFrame(rows)


def _row(r):
    known = r.clips is not None and r.clips == r.clips   # NaN once pandas mixes None into an int column
    size = f"{int(r.clips):,} | {r.hours:,.1f}" if known else "TBD: box scan | TBD: box scan"
    return f"| {r.corpus} | {r.kind} | {size} | {r.licence} | {r.commercial} |"


def render(df, recipe):
    used = df[df.in_recipe]
    blocked = sorted(set(used[used.commercial == "no"].corpus))
    unclear = sorted(set(used[used.commercial == "unclear"].corpus))

    out = ["# Corpus licences",
           "",
           f"Generated from `data/manifests/*.parquet` by `scripts/licence_table.py`. Training recipe: "
           f"`{recipe}`. Regenerate rather than edit.",
           "",
           "## In the deployed recipe",
           ""] + HEAD
    out += [_row(r) for r in used.sort_values(["kind", "corpus"]).itertuples()]
    out += ["", "## Downloaded but not in the deployed recipe", ""] + HEAD
    out += [_row(r) for r in df[~df.in_recipe].sort_values(["kind", "corpus"]).itertuples()]
    out += ["", "## What this means for transfer", "",
            f"Non-commercial material in the deployed recipe: **{', '.join(blocked) if blocked else 'none'}**. "
            f"Unresolved redistribution terms: **{', '.join(unclear) if unclear else 'none'}**.", "",
            TRANSFER, "", DISCLAIMER]
    return "\n".join(out) + "\n"


def render_many(sections):
    """sections: [(recipe path, collect() frame)]. One table per recipe, then what none of them reads."""
    out = ["# Corpus licences", "",
           "Generated from `data/manifests/*.parquet` by `scripts/licence_table.py`. Recipes: "
           + ", ".join(f"`{r}`" for r, _ in sections) + ". Regenerate rather than edit. Rows of kind `box scan` "
           f"are manifests only the training box scans; their licence comes from `{REGISTRY}`.", ""]
    used = set()
    for recipe, df in sections:
        u = df[df.in_recipe]; used |= set(u.manifest)
        blocked = sorted(set(u[u.commercial == "no"].corpus))
        unclear = sorted(set(u[u.commercial == "unclear"].corpus))
        out += [f"## In `{recipe}`", ""] + HEAD + [_row(r) for r in u.sort_values(["kind", "corpus"]).itertuples()]
        out += ["", f"Non-commercial material: **{', '.join(blocked) or 'none'}**. "
                    f"Unresolved redistribution terms: **{', '.join(unclear) or 'none'}**.", ""]
    rest = sections[0][1]
    rest = rest[~rest.manifest.isin(used)]
    out += ["## Downloaded but in none of these recipes", ""] + HEAD
    out += [_row(r) for r in rest.sort_values(["kind", "corpus"]).itertuples()]
    out += ["", "## Recorded licence conflicts", ""] + [f"- {k}: {v}" for k, v in sorted(CONFLICTS.items())]
    out += ["", "## What this means for transfer", "", TRANSFER, "", DISCLAIMER]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--recipe", action="append", help="repeatable; default configs/retraining/r7_e256_wr64.yaml")
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--out", default="docs/licences.md")
    a = ap.parse_args()
    recipes = a.recipe or ["configs/retraining/r7_e256_wr64.yaml"]
    reg = registry_licences(a.registry)
    if len(recipes) == 1:
        text = render(collect(a.manifests, recipe_manifests(recipes[0]), reg), recipes[0])
    else:
        text = render_many([(r, collect(a.manifests, recipe_manifests(r), reg)) for r in recipes])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
