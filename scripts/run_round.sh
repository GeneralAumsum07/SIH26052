#!/usr/bin/env bash
# Ablation matrix, sequential on one GPU. Usage: run_round.sh [1|2|3]. Re-runnable: train resumes
# from last.pt, evals skip existing CSVs. Round 2 needs the DNS manifest from fetch_data --dns-shards.
# Round 3 (plan 2.6): three runs, scored on the frozen eval_r2 test split next to the r1/r2 rows.
set -u
cd "$(dirname "$0")/.."
round=${1:-1}; sfx=""; [ "$round" = 2 ] && sfx=_r2
if [ "$round" = 3 ]; then
  for r in vaani_full_r3 vaani_no_controller_r3 vaani_full_r3_nodsp; do
    [ -f runs/$r/DONE ] || { uv run python -u -m vaani.train configs/exp/$r.yaml >> runs/$r.log 2>&1 && touch runs/$r/DONE; }
  done
  for r in vaani_full_r3 vaani_no_controller_r3 vaani_full_r3_nodsp; do
    [ -f results_r2/$r.csv ] || uv run python -m vaani.eval --system ckpt:runs/$r/best.pt --split test --eval-root data/eval_r2 --out results_r2/$r.csv --asr --asr-device cuda > results_r2/eval_$r.log 2>&1
  done
  uv run python -m vaani.report results_r2/*.csv --asr-ref results_r2/asr/clean.csv --out results_r2/matrix.md > results_r2/report.log 2>&1
  echo "round3 done $(date)" > results_r2/ROUND3_DONE; exit 0
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
