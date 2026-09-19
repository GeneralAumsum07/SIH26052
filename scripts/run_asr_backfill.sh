#!/usr/bin/env bash
# Re-evaluate every round-1 system with ASR on the frozen test set (earlier CSVs predate faster-whisper).
set -u
cd "$(dirname "$0")/.."
for s in raw nlms_only gtcrn_pretrained; do
  uv run python -m vaani.eval --system $s --split test --out results/$s.csv --asr --asr-device cuda > results/eval_$s.log 2>&1
done
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do
  uv run python -m vaani.eval --system ckpt:runs/$r/best.pt --split test --out results/$r.csv --asr --asr-device cuda > results/eval_$r.log 2>&1
done
uv run python -m vaani.report results/*.csv --asr-ref results/asr/clean.csv --out results/matrix.md > results/report.log 2>&1
