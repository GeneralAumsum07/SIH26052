"""The r6 corpus arms are only interpretable if the corpus is the single variable. Wave 4 bundled
four changes and its null could not be attributed to any of them; these tests stop that recurring."""
from pathlib import Path

import pytest
import yaml

CTL = "configs/retraining/r6_ctl64.yaml"
ARMS = {"configs/retraining/r6_demand64.yaml": "data/manifests/demand.parquet",
        "configs/retraining/r6_wham64.yaml": "data/manifests/wham.parquet"}


def load(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


@pytest.mark.parametrize("arm,added", ARMS.items())
def test_arm_differs_from_control_only_in_its_corpus(arm, added):
    c, a = load(CTL), load(arm)
    assert {k for k in set(c) | set(a) if c.get(k) != a.get(k)} == {"name", "data"}
    assert {k for k in set(c["data"]) | set(a["data"]) if c["data"].get(k) != a["data"].get(k)} == {"manifests"}
    assert a["data"]["manifests"] == c["data"]["manifests"] + [added]


@pytest.mark.parametrize("cfg", [CTL, *ARMS])
def test_arms_are_fresh_runs_at_the_registered_budget(cfg):
    # a warm start would confound the corpus effect with whatever the init checkpoint already learned
    c = load(cfg)
    assert "init_from" not in c
    assert c["epochs"] == 64
    assert c["seed"] == load(CTL)["seed"]


def test_the_two_arms_are_not_bundled():
    demand, wham = load("configs/retraining/r6_demand64.yaml"), load("configs/retraining/r6_wham64.yaml")
    assert "data/manifests/wham.parquet" not in demand["data"]["manifests"]
    assert "data/manifests/demand.parquet" not in wham["data"]["manifests"]


def test_the_generalisation_corpus_never_enters_a_training_recipe():
    # vehicle_interior is the held-out set; a recipe naming it would silently void the claim
    for p in Path("configs").rglob("*.yaml"):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        node = cfg.get("data", cfg)
        mans = node.get("manifests", []) if isinstance(node, dict) else []
        assert not any("vehicle_interior" in str(m) for m in mans), f"{p} trains on the held-out corpus"


SWEEP = [f"configs/retraining/r6_e{e}.yaml" for e in (32, 128, 256)]


@pytest.mark.parametrize("cfg", [CTL, *ARMS, *SWEEP])
def test_a_fresh_run_declares_no_checkpoint_hash(cfg):
    """`verify_checkpoint_hash` raises when a hash is declared and no init_from is set, so a stale
    init_sha256 on a fresh recipe kills the run on its first line. All three arms shipped that way."""
    c = load(cfg)
    if c.get("init_from") is None:
        assert "init_sha256" not in c, f"{cfg}: init_sha256 without init_from aborts training at startup"


@pytest.mark.parametrize("cfg", [CTL, *ARMS, *SWEEP])
def test_workers_follow_the_box(cfg):
    """A constant baked into a config wastes most of a 64-core box; "auto" sizes from the machine."""
    assert load(cfg).get("num_workers") == "auto"


def test_the_epoch_sweep_shares_one_batch_stream_with_the_control():
    """r6_ctl64 IS the 64-epoch point, so the four budgets train on 256 epochs of dataloading
    rather than 480, and the budget comparison is paired."""
    from vaani.train_multi import stream_signature
    cfgs = [load(p) for p in (SWEEP[0], CTL, SWEEP[1], SWEEP[2])]
    assert [c["epochs"] for c in cfgs] == [32, 64, 128, 256]
    assert all(stream_signature(c) == stream_signature(cfgs[0]) for c in cfgs)


def test_a_corpus_arm_never_shares_a_stream_with_the_sweep():
    from vaani.train_multi import assert_shared
    with pytest.raises(SystemExit):
        assert_shared([load(CTL), load("configs/retraining/r6_wham64.yaml")])
