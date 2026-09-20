"""Exercise orchestration failures without renting GPU time or touching real results."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


@pytest.mark.parametrize("round", ["3", "3b", "3c"])
@pytest.mark.parametrize("failure", ["rir", "train", "hash", "lock", "eval", "report", "none"])
def test_round3_completion_requires_success(tmp_path, failure, round):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "scripts/run_round.sh", tmp_path / "scripts/run_round.sh")
    (tmp_path / "runs").mkdir()
    (tmp_path / "results_r2").mkdir()
    frozen = tmp_path / "data/eval_r2/test"
    frozen.mkdir(parents=True)
    (frozen / "EVALSET_HASH").write_text("mismatch" if failure == "hash" else "eda217ab2a38")
    # The stub exposes each failure boundary; training still creates the directory as the real CLI does.
    stub = tmp_path / "uv"
    stub.write_text('''#!/usr/bin/env bash
[[ "$*" == "run --all-extras "* ]] || exit 2
case "$*" in
  *RirBank*)
    [ "$ROUND_TEST_FAILURE" != rir ] || exit 1
    touch rir_ready ;;
  *vaani.train*)
    [ -f rir_ready ] || exit 2
    [ "$OMP_NUM_THREADS:$MKL_NUM_THREADS" = 1:1 ] || exit 2
    name=${@: -1}; name=${name##*/}; name=${name%.yaml}
    mkdir -p "runs/$name"
    [ "$ROUND_TEST_FAILURE" != train ] || exit 1 ;;
  *verify_eval_set.py*)
    # the real gate recomputes the digest; the stub only needs to fail when the fixture says the set is wrong
    [[ "$*" == *"data/eval_r2/test eda217ab2a38" ]] || exit 2
    [ "$(cat data/eval_r2/test/EVALSET_HASH)" = eda217ab2a38 ] || exit 1 ;;
  *vaani.eval*)
    [ -f eval_lock ] || exit 2
    [[ "$*" == *"--asr --asr-device cuda --dnsmos"* ]] || exit 2
    [ "$ROUND_TEST_FAILURE" != eval ] || exit 1 ;;
  *vaani.report*) [ "$ROUND_TEST_FAILURE" != report ] || exit 1 ;;
esac
exit 0
''', encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    # Git Bash lacks flock; verify acquisition/failure ordering here. Production
    # uses the host's kernel lock, which remains held until the runner exits.
    lock = tmp_path / "flock"
    lock.write_text('''#!/usr/bin/env bash
[ "$ROUND_TEST_FAILURE" != lock ] || exit 1
touch eval_lock
''', encoding="utf-8", newline="\n")
    lock.chmod(0o755)
    env = dict(os.environ, ROUND_TEST_FAILURE=failure)
    # Let bash construct PATH: Windows separators are not POSIX separators.
    result = subprocess.run([bash, "-c", f'export PATH="$PWD:$PATH"; bash scripts/run_round.sh {round}'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (failure == "none"), result.stderr
    assert (tmp_path / f"results_r2/ROUND{round.upper()}_DONE").exists() == (failure == "none")
    if failure == "none":
        expected = {"3": {"vaani_full_r3", "vaani_no_controller_r3", "vaani_full_r3_nodsp"},
                    "3b": {"vaani_full_r3_s1", "vaani_full_r3_s2", "vaani_no_controller_r3_s1",
                           "vaani_no_controller_r3_s2", "vaani_full_r3_df1", "vaani_full_r3_df1_s1", "vaani_full_r3_df1_s2"},
                    "3c": {"vaani_full_r3_e32", "vaani_full_r3_wsnr0", "vaani_full_r3_wsnr04"}}[round]
        assert {p.parent.name for p in (tmp_path / "runs").glob("*/DONE")} == expected


@pytest.mark.parametrize("base,suffix,seed,order", [
    ("vaani_full_r3", "s1", 1, 3), ("vaani_full_r3", "s2", 2, 3),
    ("vaani_no_controller_r3", "s1", 1, 3), ("vaani_no_controller_r3", "s2", 2, 3),
    ("vaani_full_r3", "df1", 0, 1), ("vaani_full_r3", "df1_s1", 1, 1), ("vaani_full_r3", "df1_s2", 2, 1),
])
def test_round3b_configs_change_only_intended_axes(base, suffix, seed, order):
    root = Path(__file__).parents[1] / "configs/exp"
    expected = yaml.safe_load((root / f"{base}.yaml").read_text())
    name = f"{base}_{suffix}"
    expected.update(name=name, seed=seed, num_workers=6)
    expected["model_cfg"]["df_order"] = order
    assert yaml.safe_load((root / f"{name}.yaml").read_text()) == expected


@pytest.mark.parametrize("suffix,key,value", [("e32", "epochs", 32), ("wsnr0", "w_snr", 0.0), ("wsnr04", "w_snr", 0.4)])
def test_round3c_configs_change_one_knob(suffix, key, value):
    root = Path(__file__).parents[1] / "configs/exp"
    expected = yaml.safe_load((root / "vaani_full_r3.yaml").read_text())
    expected.update(name=f"vaani_full_r3_{suffix}", num_workers=6)
    (expected if key == "epochs" else expected["loss_cfg"])[key] = value
    assert yaml.safe_load((root / f"vaani_full_r3_{suffix}.yaml").read_text()) == expected
