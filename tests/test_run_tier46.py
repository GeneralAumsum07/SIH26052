"""run_tier46.sh: the bank is materialised before the trainer, the trainer sees the thread caps, DONE only on success."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("failure", ["rir", "train", "none"])
def test_tier46_launcher(tmp_path, failure):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "scripts/run_tier46.sh", tmp_path / "scripts/run_tier46.sh")
    (tmp_path / "cfg.yaml").write_text("name: t46\n")
    stub = tmp_path / "uv"
    stub.write_text('''#!/usr/bin/env bash
[[ "$*" == "run --all-extras "* ]] || exit 2
case "$*" in
  *'["name"]'*) echo t46 ;;
  *RirBank*)
    [ "$T46_TEST_FAILURE" != rir ] || exit 1
    touch rir_ready ;;
  *vaani.train_refiner*)
    [ -f rir_ready ] || exit 2
    [ "$OMP_NUM_THREADS:$MKL_NUM_THREADS:$OPENBLAS_NUM_THREADS:$NUMBA_NUM_THREADS" = 1:1:1:1 ] || exit 2
    mkdir -p runs/t46
    [ "$T46_TEST_FAILURE" != train ] || exit 1 ;;
esac
exit 0
''', encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    env = dict(os.environ, T46_TEST_FAILURE=failure)
    result = subprocess.run([bash, "-c", 'export PATH="$PWD:$PATH"; bash scripts/run_tier46.sh cfg.yaml'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == (failure == "none"), result.stderr
    assert (tmp_path / "runs/t46/DONE").exists() == (failure == "none")
