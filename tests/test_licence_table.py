"""The licence table is evidence for a transfer claim, so its classifier must not quietly mislabel.
The dangerous direction is a false `yes`: calling restricted material commercially usable."""
import pandas as pd
import pytest
import yaml

from scripts import licence_table as lt


@pytest.mark.parametrize("licence,expected", [
    ("CC BY-NC 4.0", "no"),                       # must not be caught by the "CC BY" rule
    ("ESC-50 (CC BY-NC per clip, see meta)", "no"),
    ("CC0", "yes"),
    ("CC BY 4.0", "yes"),
    ("CC BY-SA 4.0", "yes, share-alike"),
    ("MAD (YouTube-sourced; see repo)", "no"),
    ("NOISEX-92 (DRA Malvern 1992, via SPIB; redistribution terms unclear)", "unclear"),
    ("DNS-4 archive noise_fullband (see DNS README per-clip licences)", "unclear"),
    ("something nobody has classified", "unclear"),   # unknown is unclear, never yes
])
def test_classification(licence, expected):
    assert lt.classify(licence)[0] == expected


def test_nc_is_checked_before_permissive_cc():
    # "CC BY-NC 4.0" contains "CC BY"; rule order is what stops a false permissive verdict
    assert lt.classify("CC BY-NC 4.0")[0] == "no"


def test_recipe_manifests_reads_the_training_list(tmp_path):
    cfg = tmp_path / "r.yaml"
    cfg.write_text(yaml.safe_dump({"data": {"manifests": ["data/manifests/a.parquet", "data/manifests/b.parquet"]}}),
                   encoding="utf-8")
    assert lt.recipe_manifests(cfg) == {"a.parquet", "b.parquet"}


def test_in_recipe_split_is_driven_by_the_config(tmp_path):
    # a corpus downloaded but dropped from the winning recipe must not be reported as in use
    for name, corpus in [("used.parquet", "used"), ("dropped.parquet", "dropped")]:
        pd.DataFrame({"corpus": [corpus], "kind": ["noise"], "licence": ["CC0"], "duration_s": [3600.0]}
                     ).to_parquet(tmp_path / name)
    df = lt.collect(tmp_path, {"used.parquet"})
    assert set(df[df.in_recipe].corpus) == {"used"}
    assert set(df[~df.in_recipe].corpus) == {"dropped"}


def test_render_names_restricted_corpora_in_the_recipe(tmp_path):
    pd.DataFrame({"corpus": ["ears"], "kind": ["speech"], "licence": ["CC BY-NC 4.0"], "duration_s": [3600.0]}
                 ).to_parquet(tmp_path / "ears.parquet")
    text = lt.render(lt.collect(tmp_path, {"ears.parquet"}), "r.yaml")
    assert "Non-commercial material in the deployed recipe: **ears**" in text
