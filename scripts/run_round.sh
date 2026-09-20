#!/usr/bin/env bash
# Ablation matrix. Usage: run_round.sh [1|2|3|3b]. Re-runnable: train resumes
# from last.pt, evals skip existing CSVs. Round 2 needs the DNS manifest from fetch_data --dns-shards.
# Round 3 (plan 2.6): three runs, scored on the frozen eval_r2 test split next to the r1/r2 rows.
set -euo pipefail
cd "$(dirname "$0")/.."
round=${1:-1}; sfx=""; [ "$round" = 2 ] && sfx=_r2
if [ "$round" = 3 ] || [ "$round" = 3b ]; then
  # These concurrent waves target the GPU host, not the laptop. Wave 3b holds
  # seeds 1/2 and the df_order=1 control; launch it only after the timing check.
  R3="vaani_full_r3 vaani_no_controller_r3 vaani_full_r3_nodsp"
  marker=ROUND3_DONE
  if [ "$round" = 3b ]; then
    R3="vaani_full_r3_s1 vaani_full_r3_s2 vaani_no_controller_r3_s1 vaani_no_controller_r3_s2 vaani_full_r3_df1"
    marker=ROUND3B_DONE
  fi
  mkdir -p runs results_r2
  # Each loader inherits these limits; otherwise every worker can claim the
  # host's full OpenMP/MKL pool, overwhelming the concurrent training jobs.
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  # RirBank uses shared .tmp.npy names while unpacking. Materialize each bank
  # in one process before children start, so all loaders only open mmap files.
  uv run --all-extras python -c 'import sys, yaml; from vaani.data.rirs import RirBank; banks = {yaml.safe_load(open("configs/exp/" + r + ".yaml"))["data"]["bank"] for r in sys.argv[1:]}; [RirBank(p) for p in sorted(banks)]' $R3
  pids=()
  for r in $R3; do
    [ -f runs/$r/DONE ] || { uv run --all-extras python -u -m vaani.train configs/exp/$r.yaml >> runs/$r.log 2>&1 && touch runs/$r/DONE; } &
    pids+=("$!")
  done
  # Bare wait hides failed children; finish every sibling before deciding whether evaluation is safe.
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [ "$failed" = 0 ] || { echo "round$round training failed; inspect runs/*_r3*.log" >&2; exit 1; }
  # Training can proceed during a frozen-set transfer, but mixed-hash matrix rows cannot. The hash is
  # recomputed from the files present: a partial copy carries a valid EVALSET_HASH file and must still fail.
  uv run --all-extras python scripts/verify_eval_set.py data/eval_r2/test eda217ab2a38 || {
    echo "eval_r2 test split incomplete or not the frozen set; restore it before evaluation" >&2; exit 1;
  }
  # Both waves share the matrix glob. Hold one Linux advisory lock across all
  # eval writes and report generation so neither reads the other's partial CSV.
  exec 9>results_r2/.eval_report.lock
  flock 9
  for r in $R3; do
    [ -f results_r2/$r.csv ] || uv run --all-extras python -m vaani.eval --system ckpt:runs/$r/best.pt --split test --eval-root data/eval_r2 --out results_r2/$r.csv --asr --asr-device cuda --dnsmos > results_r2/eval_$r.log 2>&1
  done
  uv run --all-extras python -m vaani.report results_r2/*.csv --asr-ref results_r2/asr/clean.csv --out results_r2/matrix.md > results_r2/report.log 2>&1
  echo "round$round done $(date)" > results_r2/$marker; exit 0
fi
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do r=$r$sfx
  [ -f runs/$r/DONE ] || { uv run python -u -m vaani.train configs/exp/$r.yaml >> runs/$r.log 2>&1 && touch runs/$r/DONE; }
done
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do r=$r$sfx
  [ -f results/$r.csv ] || uv run python -m vaani.eval --system ckpt:runs/$r/best.pt --split test --out results/$r.csv --asr --asr-device cuda > results/eval_$r.log 2>&1
done
[ -f results/asr/clean.csv ] || uv run python scripts/asr_clean_reference.py --asr-device cuda > results/asr/clean.log 2>&1
uv run python -m vaani.report results/*.csv --asr-ref results/asr/clean.csv --out results/matrix.md > results/report.log 2>&1
echo "round$round done $(date)" > results/ROUND${round}_DONE
