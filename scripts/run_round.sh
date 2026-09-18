#!/usr/bin/env bash
# Ablation matrix, sequential on one GPU. Usage: run_round.sh [1|2]. Re-runnable: train resumes
# from last.pt, evals skip existing CSVs. Round 2 needs the DNS manifest from fetch_data --dns-shards.
set -u
cd "$(dirname "$0")/.."
round=${1:-1}; sfx=""; [ "$round" = 2 ] && sfx=_r2
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do r=$r$sfx
  [ -f runs/$r/DONE ] || { uv run python -u -m vaani.train configs/exp/$r.yaml >> runs/$r.log 2>&1 && touch runs/$r/DONE; }
done
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do r=$r$sfx
  [ -f results/$r.csv ] || uv run python -m vaani.eval --system ckpt:runs/$r/best.pt --split test --out results/$r.csv --asr > results/eval_$r.log 2>&1
done
uv run python -m vaani.report results/*.csv --out results/matrix.md > results/report.log 2>&1
echo "round$round done $(date)" > results/ROUND${round}_DONE
