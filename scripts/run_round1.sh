#!/usr/bin/env bash
# Round-1 matrix, sequential on one GPU. Re-runnable: train resumes from last.pt, evals skip existing CSVs.
set -u
cd "$(dirname "$0")/.."
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do
  [ -f runs/$r/DONE ] || { uv run python -u -m vaani.train configs/exp/$r.yaml >> runs/$r.log 2>&1 && touch runs/$r/DONE; }
done
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do
  [ -f results/$r.csv ] || uv run python -m vaani.eval --system ckpt:runs/$r/best.pt --split test --out results/$r.csv --asr > results/eval_$r.log 2>&1
done
uv run python -m vaani.report results/*.csv --out results/matrix.md > results/report.log 2>&1
echo "round1 done $(date)" > results/ROUND1_DONE
