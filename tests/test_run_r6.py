"""run_r6.sh: the protocol gates the session, training precedes scoring, and the arms stay separate.

The ordering here is the experiment's validity, not housekeeping. A run that scores before the
protocol is committed produces numbers that cannot be claimed, and one that bundles the arms
reproduces the wave-4 mistake that made this session necessary.
"""
import os
import shutil
import subprocess

import pytest

from tests.test_run_optimization import ROOT

STUB = '''#!/usr/bin/env bash
echo "$*" >> calls.txt
case "$*" in
  *fetch_data*) ;;
  *crest_audit*) ;;
  *render_eval_sets*)
    mkdir -p data/eval_gen/test; echo abc123 > data/eval_gen/test/EVALSET_HASH ;;
  *check_heldout*) ;;
  *vaani.train*)
    cfg=$(echo "$*" | tr ' ' '\\n' | grep configs/retraining | head -1)
    n=$(basename "$cfg" .yaml); mkdir -p "runs/$n"; touch "runs/$n/best.pt" ;;
  *vaani.eval*)
    out=$(echo "$*" | sed 's/.*--out //; s/ .*//'); mkdir -p "$(dirname "$out")"; touch "$out" ;;
  *licence_table*) ;;
esac
exit 0
'''


def _sandbox(tmp_path, protocol_committed=True):
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts" / "run_r6.sh", tmp_path / "scripts" / "run_r6.sh")
    (tmp_path / "results_r2/generalisation").mkdir(parents=True)
    (tmp_path / "results_r2/generalisation/PROTOCOL.md").write_text("registered")
    (tmp_path / "data/eval_r2/test").mkdir(parents=True)
    (tmp_path / "data/eval_r2/test/EVALSET_HASH").write_text("eda217ab2a38")
    (tmp_path / "data/manifests").mkdir(parents=True)
    for m in ("vehicle_interior", "wham"):
        (tmp_path / f"data/manifests/{m}.parquet").touch()
    (tmp_path / "results_r2/runs/vaani_tier46_refiner").mkdir(parents=True)
    (tmp_path / "results_r2/runs/vaani_tier46_refiner/best.pt").touch()

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    if protocol_committed:
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "register"],
                       cwd=tmp_path, check=True)
    else:
        subprocess.run(["git", "rm", "--cached", "-q", "results_r2/generalisation/PROTOCOL.md"],
                       cwd=tmp_path, check=True)

    stub = tmp_path / "uv"
    stub.write_text(STUB, encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    return tmp_path


def _run(tmp_path):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    return subprocess.run([bash, "-c", 'export PATH="$PWD:$PATH"; bash scripts/run_r6.sh'],
                          cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True)


def _calls(tmp_path):
    return (tmp_path / "calls.txt").read_text().splitlines()


def test_an_uncommitted_protocol_stops_the_session(tmp_path):
    # registration that is not committed is not registration: it can be edited after seeing the numbers
    _sandbox(tmp_path, protocol_committed=False)
    r = _run(tmp_path)
    assert r.returncode != 0
    assert not (tmp_path / "calls.txt").exists()


def test_a_missing_protocol_stops_the_session(tmp_path):
    _sandbox(tmp_path)
    (tmp_path / "results_r2/generalisation/PROTOCOL.md").unlink()
    r = _run(tmp_path)
    assert r.returncode != 0


def test_disjointness_is_proven_before_any_training(tmp_path):
    _sandbox(tmp_path)
    assert _run(tmp_path).returncode == 0
    calls = _calls(tmp_path)
    check = next(i for i, c in enumerate(calls) if "check_heldout" in c)
    train = next(i for i, c in enumerate(calls) if "vaani.train" in c)
    assert check < train


def test_each_arm_is_trained_before_it_is_scored(tmp_path):
    _sandbox(tmp_path)
    assert _run(tmp_path).returncode == 0
    calls = _calls(tmp_path)
    for arm in ("r6_ctl64", "r6_demand64", "r6_wham64"):
        trained = next(i for i, c in enumerate(calls) if "vaani.train" in c and arm in c)
        scored = next(i for i, c in enumerate(calls) if "vaani.eval" in c and arm in c)
        assert trained < scored


def test_the_arms_are_trained_as_separate_runs(tmp_path):
    _sandbox(tmp_path)
    _run(tmp_path)
    trains = [c for c in _calls(tmp_path) if "vaani.train" in c]
    assert len(trains) == 3, "control plus two arms, never bundled"


def test_finished_work_is_not_repeated(tmp_path):
    # GPU hours: an interrupted session must resume, not restart
    _sandbox(tmp_path)
    _run(tmp_path)
    first = len(_calls(tmp_path))
    (tmp_path / "calls.txt").unlink()
    _run(tmp_path)
    second = _calls(tmp_path)
    assert not any("vaani.train" in c for c in second)
    assert not any("vaani.eval" in c for c in second)
    assert first > len(second)


def test_the_generalisation_set_is_scored_for_the_deployed_system(tmp_path):
    _sandbox(tmp_path)
    _run(tmp_path)
    assert any("tier46_gen" in c for c in _calls(tmp_path))
    assert (tmp_path / "results_r2/r6/DONE").exists()
