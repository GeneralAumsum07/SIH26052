"""run_optimization.sh: preconditions fail loudly, stages run in order, existing CSVs are not re-evaluated."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

# Records each uv invocation so the test can assert ordering, then fakes the artefact that
# invocation was supposed to produce. Evals are 15 minutes each in reality.
STUB = '''#!/usr/bin/env bash
echo "$*" >> calls.txt
case "$*" in
  *vaani.quantize*) ;;
  *vaani.prune*)
    for p in p10 p20 p30 p40 p50; do mkdir -p "runs/prune/$p"; touch "runs/prune/$p/best.pt"; done ;;
  *vaani.eval*)
    out=$(echo "$*" | sed 's/.*--out //; s/ .*//'); mkdir -p "$(dirname "$out")"; touch "$out" ;;
  *optimization_report*) ;;
esac
exit 0
'''


def _sandbox(tmp_path, with_checkpoint=True, with_evalset=True):
    (tmp_path / "scripts").mkdir()
    for s in ("run_optimization.sh",):
        shutil.copyfile(ROOT / "scripts" / s, tmp_path / "scripts" / s)
    if with_checkpoint:
        (tmp_path / "results_r2/runs/vaani_tier46_refiner").mkdir(parents=True)
        (tmp_path / "results_r2/runs/vaani_tier46_refiner/best.pt").touch()
    if with_evalset:
        (tmp_path / "data/eval_r2/test").mkdir(parents=True)
        (tmp_path / "data/eval_r2/test/EVALSET_HASH").write_text("test")
    (tmp_path / "deploy/tier46").mkdir(parents=True)
    (tmp_path / "deploy/tier46/cascade.onnx").touch()
    (tmp_path / "results_r2").mkdir(exist_ok=True)
    (tmp_path / "results_r2/vaani_tier46_refiner.csv").touch()
    stub = tmp_path / "uv"
    stub.write_text(STUB, encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    return stub


def _run(tmp_path):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    return subprocess.run([bash, "-c", 'export PATH="$PWD:$PATH"; bash scripts/run_optimization.sh'],
                          cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True)


@pytest.mark.parametrize("missing", ["checkpoint", "evalset"])
def test_missing_inputs_stop_before_any_work(tmp_path, missing):
    # Silently producing an empty optimization.md would be worse than failing: the tables are
    # evidence for a requirements claim.
    _sandbox(tmp_path, with_checkpoint=missing != "checkpoint", with_evalset=missing != "evalset")
    r = _run(tmp_path)
    assert r.returncode != 0
    assert not (tmp_path / "calls.txt").exists()


def test_stages_run_in_order_and_done_is_last(tmp_path):
    _sandbox(tmp_path)
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    kinds = [next(k for k in ("quantize", "prune", "eval", "optimization_report") if k in c) for c in calls]
    assert kinds.index("quantize") < kinds.index("eval")          # nothing is scored before it exists
    assert kinds.index("prune") < len(kinds) - 1
    assert kinds[-1] == "optimization_report"                     # tables are built from finished CSVs
    assert (tmp_path / "results_r2/optim/DONE").exists()


def test_existing_csvs_are_not_re_evaluated(tmp_path):
    # A 1.5 h run must be resumable: the eval stage is the expensive part and each CSV is final.
    _sandbox(tmp_path)
    (tmp_path / "results_r2/optim").mkdir(parents=True)
    for name in ("onnx_cascade", "onnx_cascade_int8", "prune_p10"):
        (tmp_path / f"results_r2/optim/{name}.csv").touch()
    _run(tmp_path)
    evals = [c for c in (tmp_path / "calls.txt").read_text().splitlines() if "vaani.eval" in c]
    assert not any(n in c for c in evals for n in ("onnx_cascade.csv", "onnx_cascade_int8.csv", "prune_p10.csv"))
    assert any("prune_p20.csv" in c for c in evals)
