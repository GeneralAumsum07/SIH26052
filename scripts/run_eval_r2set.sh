#!/usr/bin/env bash
# Round-2 test split (rendered from all five manifests) evaluated for every system.
# Kept apart from data/eval and results/ so the frozen round-1 numbers stay the reference.
set -u
cd "$(dirname "$0")/.."
M="data/manifests/librispeech.parquet data/manifests/esc50.parquet data/manifests/cv_hi.parquet data/manifests/mad.parquet data/manifests/dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet"
[ -f data/eval_r2/test/EVALSET_HASH ] || uv run python scripts/render_eval_sets.py --manifests $M --split test --out data/eval_r2 > data/eval_r2_render.log 2>&1 || exit 1
mkdir -p results_r2/asr
for s in raw nlms_only gtcrn_pretrained; do
  [ -f results_r2/$s.csv ] || uv run python -m vaani.eval --system $s --split test --eval-root data/eval_r2 --out results_r2/$s.csv --asr --asr-device cuda > results_r2/eval_$s.log 2>&1
done
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do for x in "" _r2; do r2=$r$x
  [ -f results_r2/$r2.csv ] || uv run python -m vaani.eval --system ckpt:runs/$r2/best.pt --split test --eval-root data/eval_r2 --out results_r2/$r2.csv --asr --asr-device cuda > results_r2/eval_$r2.log 2>&1
done; done
[ -f results_r2/asr/clean.csv ] || uv run python scripts/asr_clean_reference.py --asr-device cuda --eval-root data/eval_r2 --out results_r2/asr/clean.csv > results_r2/asr/clean.log 2>&1
uv run python -m vaani.report results_r2/*.csv --asr-ref results_r2/asr/clean.csv --out results_r2/matrix.md > results_r2/report.log 2>&1
echo "r2set done $(date)" > results_r2/DONE
