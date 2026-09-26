"""scripts/heldout_freesound.py on a synthetic tree: it lists the train/val chunks of a test noise recording (and the
same Freesound id in another corpus), keeps the render_eval_sets sources, and --check catches a new scan."""
import json

import pandas as pd
import yaml

from scripts import heldout_freesound as hf
from vaani.data.manifests import COLUMNS


def _man(path, rows):
    base = dict(corpus="dns_noise", kind="noise", speaker_id="", path="x", duration_s=10.0, licence="t", sha1="",
                noise_class="stationary")
    pd.DataFrame([dict(base, group_id=r[0], **dict(zip(("source_id", "split"), r))) for r in rows], columns=COLUMNS).to_parquet(path)


def _tree(tmp_path):
    (tmp_path / "configs/retraining").mkdir(parents=True); (tmp_path / "configs/data").mkdir(parents=True)
    (tmp_path / "data/manifests").mkdir(parents=True); (tmp_path / "data/eval_r8_test/test").mkdir(parents=True)
    mans = ["data/manifests/dns.parquet", "data/manifests/esc50.parquet", "data/manifests/fsd50k.parquet"]
    for r in hf.RECIPES:
        (tmp_path / r).write_text(yaml.safe_dump({"data": {"manifests": mans}}), encoding="utf-8")
    _man(tmp_path / mans[0], [("dnsn:door_Freesound_validated_11_0", "test"), ("dnsn:door_Freesound_validated_11_1", "train"),
                              ("dnsn:door_Freesound_validated_11_2", "val"), ("dnsn:door_Freesound_validated_12_0", "train")])
    _man(tmp_path / mans[1], [("esc50:door_wood_knock/3-11-A-30", "train"), ("esc50:rain/1-13-A-10", "train")])
    pd.DataFrame({"noise_source": ["dnsn:door_Freesound_validated_11_0"], "impulse_source": [None],
                  "speech_source": ["ls:1"]}).to_csv(tmp_path / hf.INDEX, index=False)
    (tmp_path / "configs/data/r8_heldout_exclude.json").write_text(json.dumps(
        {"schema": "vaani.heldout_exclude/1", "sources": {"drone": {"manifest": "drone.parquet", "source_ids": ["d:1"]}}}))
    assert hf.main(["--root", str(tmp_path), "--write-sources"]) == 0
    return tmp_path, mans


def test_siblings_are_listed_and_other_sources_kept(tmp_path):
    root, mans = _tree(tmp_path)
    assert hf.main(["--root", str(root)]) == 0
    s = json.loads((root / "configs/data/r8_heldout_exclude.json").read_text())["sources"]
    assert s["drone"]["source_ids"] == ["d:1"]
    assert s["freesound_dns"]["source_ids"] == ["dnsn:door_Freesound_validated_11_1", "dnsn:door_Freesound_validated_11_2"]
    assert s["freesound_esc50"]["source_ids"] == ["esc50:door_wood_knock/3-11-A-30"]   # same upload, other corpus
    assert "freesound_fsd50k" not in s   # not scanned here
    assert hf.main(["--root", str(root), "--check"]) == 0


def test_check_fails_once_a_new_freesound_manifest_is_scanned(tmp_path):
    root, mans = _tree(tmp_path)
    hf.main(["--root", str(root)])
    _man(root / mans[2], [("fsd50k:Door/11", "train"), ("fsd50k:Door/14", "train")])
    assert hf.main(["--root", str(root), "--check"]) == 1
    hf.main(["--root", str(root)])
    s = json.loads((root / "configs/data/r8_heldout_exclude.json").read_text())["sources"]
    assert s["freesound_fsd50k"]["source_ids"] == ["fsd50k:Door/11"] and hf.main(["--root", str(root), "--check"]) == 0


def test_composite_v2_noise_sources_are_split(tmp_path):
    # a v2 scene joins its sources with "+": every component's Freesound id counts, not only the last one
    root, mans = _tree(tmp_path)
    pd.DataFrame({"noise_source": ["esc50:rain/1-13-A-10+dnsn:door_Freesound_validated_11_0"], "impulse_source": [None],
                  "speech_source": ["ls:1"]}).to_csv(root / hf.INDEX, index=False)
    assert hf.main(["--root", str(root), "--check"]) == 1   # the committed list no longer matches the index
    hf.main(["--root", str(root), "--write-sources"]); hf.main(["--root", str(root)])
    s = json.loads((root / "configs/data/r8_heldout_exclude.json").read_text())["sources"]
    assert s["freesound_dns"]["source_ids"] == ["dnsn:door_Freesound_validated_11_1", "dnsn:door_Freesound_validated_11_2"]
    assert s["freesound_esc50"]["source_ids"] == ["esc50:door_wood_knock/3-11-A-30"]


def test_the_committed_list_stands_in_for_the_index_on_the_box(tmp_path):
    root, mans = _tree(tmp_path)
    hf.main(["--root", str(root)])
    (root / hf.INDEX).unlink()   # the box has no data/eval_r8_test
    assert hf.main(["--root", str(root), "--check"]) == 0
    _man(root / mans[2], [("fsd50k:Door/11", "train")])
    assert hf.main(["--root", str(root), "--check"]) == 1
    hf.main(["--root", str(root)])
    s = json.loads((root / "configs/data/r8_heldout_exclude.json").read_text())["sources"]
    assert s["freesound_fsd50k"]["source_ids"] == ["fsd50k:Door/11"]
