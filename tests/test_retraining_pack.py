import yaml
import pytest
from pathlib import Path

from vaani.experiments import recipes, preflight


def test_pack_has_matched_controls_and_all_refiner_cells():
    base = yaml.safe_load(Path("configs/exp/vaani_full_r4_ctl.yaml").read_text())
    pack = recipes(base)
    assert len(pack) == 29
    refiners = [c for c in pack.values() if c["model"] == "vaani_cascade"]
    assert len(refiners) == 16
    assert {(c["refiner_cfg"]["hidden"], c["refiner_cfg"]["past"], c["seed"]) for c in refiners} == {
        (h, p, s) for h in (16, 24, 32, 48) for p in (2, 4) for s in (4606, 4607)}
    for c in pack.values():
        assert c["val"]["split"] == "val"
        if c["name"].startswith("r5_width"):
            assert c["init_from"] is None and c["epochs"] == 128
    assert pack["r5_continue128"]["data"] == pack["r5_noise_floor"]["data"]


def test_preflight_does_not_create_a_run(tmp_path):
    base = yaml.safe_load(Path("configs/exp/vaani_full_r4_ctl.yaml").read_text())
    cfg = recipes(base)["r5_width8_s0"]
    cfg["runs_dir"] = str(tmp_path / "runs")
    r = preflight(cfg, check_files=False)
    assert not r["training_launched"] and not Path(cfg["runs_dir"]).exists()
    assert r["schedule"]["total_steps"] == 128 * 625
    assert r["cache_shapes"]["conv_cache"] == [2, 1, 8, 16, 33]
    assert r["matrix_macs_per_frame"] > 0


def test_preflight_refuses_test_selection():
    with pytest.raises(ValueError, match="val"):
        preflight(dict(name="safe", val=dict(split="test")))


def test_preflight_cli_cannot_launch_subprocess(tmp_path, monkeypatch):
    from scripts import retraining
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight tried to launch a subprocess")
    monkeypatch.setattr(retraining.subprocess, "run", forbidden)
    base = yaml.safe_load(Path("configs/exp/vaani_full_r4_ctl.yaml").read_text())
    cfg = recipes(base)["r5_width8_s0"]
    cfg["runs_dir"] = str(tmp_path / "runs")
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert retraining.main(["preflight", str(path), "--skip-file-checks"]) == 0
    assert not Path(cfg["runs_dir"]).exists()
