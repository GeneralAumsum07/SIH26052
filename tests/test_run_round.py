"""Exercise orchestration failures without renting GPU time or touching real results."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("failure", ["train", "hash", "eval", "report", "none"])
def test_round3_completion_requires_success(tmp_path, failure):
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
case "$*" in
  *vaani.train*)
    name=${@: -1}; name=${name##*/}; name=${name%.yaml}
    mkdir -p "runs/$name"
    [ "$ROUND_TEST_FAILURE" != train ] || exit 1 ;;
  *vaani.eval*) [ "$ROUND_TEST_FAILURE" != eval ] || exit 1 ;;
  *vaani.report*) [ "$ROUND_TEST_FAILURE" != report ] || exit 1 ;;
esac
exit 0
''', encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    env = dict(os.environ, ROUND_TEST_FAILURE=failure)
    # Let bash construct PATH: Windows separators are not POSIX separators.
    result = subprocess.run([bash, "-c", 'export PATH="$PWD:$PATH"; bash scripts/run_round.sh 3'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (failure == "none"), result.stderr
    assert (tmp_path / "results_r2/ROUND3_DONE").exists() == (failure == "none")
