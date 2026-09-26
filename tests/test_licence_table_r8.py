"""The licence table over the r8 recipe: box-only manifests take the registry's licence with a TBD size, FSD50K's
licence URLs classify without a false yes, audit parquets are skipped, and each recipe gets its own section."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import licence_table as lt

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("licence,expected", [
    ("http://creativecommons.org/licenses/by-nc/3.0/", "no"),
    ("http://creativecommons.org/licenses/by/3.0/", "yes"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "yes"),
    ("https://creativecommons.org/licenses/sampling+/1.0/", "unclear"),
    ("per clip; the scanner keeps CC0 and CC BY only", "yes"),
    ("conflict: Zenodo field CC BY 4.0, record text CC BY-SA 3.0", "yes, share-alike"),
])
def test_r8_licence_strings(licence, expected):
    assert lt.classify(licence)[0] == expected


def test_real_registry_names_every_r8_manifest_licence():
    reg = lt.registry_licences(REPO / lt.REGISTRY)
    mans = lt.recipe_manifests(REPO / "configs/retraining/r8_fe_mini.yaml")
    assert mans <= set(reg), sorted(mans - set(reg))
    assert reg["demand_pairs.parquet"][0] == "demand" and lt.classify(reg["fsd50k.parquet"][1])[0] == "yes"


def test_box_only_manifest_gets_a_tbd_row_and_audit_tables_are_skipped(tmp_path):
    pd.DataFrame({"corpus": ["ears"], "kind": ["speech"], "licence": ["CC BY-NC 4.0"], "duration_s": [3600.0]}
                 ).to_parquet(tmp_path / "ears.parquet")
    pd.DataFrame({"source_id": ["x"], "duration_s": [1.0]}).to_parquet(tmp_path / "audit.parquet")
    df = lt.collect(tmp_path, {"ears.parquet", "c3gd.parquet"}, {"c3gd.parquet": ("c3gd", "CC BY 4.0")})
    assert set(df.manifest) == {"ears.parquet", "c3gd.parquet"}
    text = lt.render_many([("r8.yaml", df)])
    assert "| c3gd | box scan | TBD: box scan | TBD: box scan | CC BY 4.0 | yes |" in text
    assert "Non-commercial material: **ears**" in text and "## Recorded licence conflicts" in text


def test_each_recipe_gets_its_own_section(tmp_path):
    for name, corpus in [("a.parquet", "a"), ("b.parquet", "b"), ("c.parquet", "c")]:
        pd.DataFrame({"corpus": [corpus], "kind": ["noise"], "licence": ["CC0"], "duration_s": [3600.0]}
                     ).to_parquet(tmp_path / name)
    text = lt.render_many([("r7.yaml", lt.collect(tmp_path, {"a.parquet"})), ("r8.yaml", lt.collect(tmp_path, {"b.parquet"}))])
    r7, r8 = text.split("## In `r7.yaml`")[1].split("## In `r8.yaml`")
    assert "| a |" in r7 and "| b |" not in r7 and "| b |" in r8.split("## Downloaded")[0]
    assert "| c |" in text.split("## Downloaded but in none of these recipes")[1]


def test_generated_doc_covers_the_r8_recipe():
    text = (REPO / "docs/licences.md").read_text(encoding="utf-8")
    assert "## In `configs/retraining/r8_fe_mini.yaml`" in text and "## In `configs/retraining/r7_e256_wr64.yaml`" in text
    for c in ("avq_drone", "c3gd", "fsd50k", "lombard_grid"):
        assert f"| {c} |" in text, c
