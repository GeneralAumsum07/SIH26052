"""Licence provenance for every corpus, generated from the manifests rather than maintained by hand.

A DRDO reviewer is entitled to ask what the system trains on and whether that material could follow
the design into a fielded product. The answer has to come from the manifests, because they are what
training actually reads -- a hand-maintained table drifts the moment a recipe changes, and a drifted
licence table is worse than none.

The commercial-use column is the only judgement here, and it is deliberately coarse: `no` for anything
carrying NC, `unclear` where the redistribution terms are genuinely unresolved, `yes` only for permissive
terms we can name. Where it says unclear it means unclear -- that is a finding, not an omission.

    uv run python scripts/licence_table.py --recipe configs/retraining/r5_continue128.yaml --out docs/licences.md
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

# Substring -> (commercial use, short name). Order matters: NC is checked before the permissive CC forms,
# because "CC BY-NC 4.0" contains "CC BY".
RULES = [
    ("CC BY-NC", ("no", "CC BY-NC")),
    ("CC0", ("yes", "CC0")),
    ("CC BY-SA", ("yes, share-alike", "CC BY-SA 4.0")),
    ("CC BY", ("yes", "CC BY 4.0")),
    ("NIJ", ("unclear", "US DOJ/NIJ award output")),
    ("DroneAudioDataset", ("unclear", "citation-on-use")),
    ("NOISEX-92", ("unclear", "SPIB redistribution terms unstated")),
    ("MAD", ("no", "YouTube-sourced")),
    ("DNS-4", ("unclear", "per-clip, see DNS README")),
]


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


def collect(manifest_dir, in_recipe):
    rows = []
    for f in sorted(Path(manifest_dir).glob("*.parquet")):
        df = pd.read_parquet(f, columns=["corpus", "kind", "licence", "duration_s"])
        for (corpus, kind, licence), g in df.groupby(["corpus", "kind", "licence"], dropna=False):
            commercial, short = classify(str(licence))
            rows.append({"corpus": corpus, "kind": kind, "clips": len(g),
                         "hours": g.duration_s.sum() / 3600.0, "licence": short,
                         "commercial": commercial, "in_recipe": f.name in in_recipe,
                         "manifest": f.name})
    return pd.DataFrame(rows)


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
           "",
           "| corpus | kind | clips | hours | licence | commercial use |",
           "|---|---|---:|---:|---|---|"]
    for r in used.sort_values(["kind", "corpus"]).itertuples():
        out.append(f"| {r.corpus} | {r.kind} | {r.clips:,} | {r.hours:,.1f} | {r.licence} | {r.commercial} |")

    out += ["", "## Downloaded but not in the deployed recipe", "",
            "| corpus | kind | clips | hours | licence | commercial use |",
            "|---|---|---:|---:|---|---|"]
    for r in df[~df.in_recipe].sort_values(["kind", "corpus"]).itertuples():
        out.append(f"| {r.corpus} | {r.kind} | {r.clips:,} | {r.hours:,.1f} | {r.licence} | {r.commercial} |")

    out += ["", "## What this means for transfer", "",
            f"Non-commercial material in the deployed recipe: **{', '.join(blocked) if blocked else 'none'}**. "
            f"Unresolved redistribution terms: **{', '.join(unclear) if unclear else 'none'}**.", "",
            "The research prototype trains on corpora including non-commercial-licensed and "
            "unclear-licence material. A fielded version would retrain on licensed or "
            "government-collected data; the pipeline is corpus-agnostic and the manifest layer makes "
            "the substitution mechanical -- a recipe is a list of manifest paths, and nothing in the "
            "model, the DSP front end or the training loop is tied to a particular corpus.", "",
            "This is a licence statement, not a legal opinion, and `unclear` rows are unresolved "
            "rather than cleared."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--recipe", default="configs/retraining/r5_continue128.yaml")
    ap.add_argument("--out", default="docs/licences.md")
    a = ap.parse_args()

    df = collect(a.manifests, recipe_manifests(a.recipe))
    text = render(df, a.recipe)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
