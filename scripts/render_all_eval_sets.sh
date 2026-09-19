#!/usr/bin/env bash
# Re-render both frozen eval sets after the bucket redesign (recorded-impulse + fault buckets).
# data/eval    : round-1 corpora (LibriSpeech + ESC-50), val feeds training, test carries faults.
# data/eval_r2 : all five manifests, the headline test split; val added for the r3 retrain.
set -eu
cd "$(dirname "$0")/.."
M1="data/manifests/librispeech.parquet data/manifests/esc50.parquet"
M2="$M1 data/manifests/cv_hi.parquet data/manifests/mad.parquet data/manifests/dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet"
uv run python scripts/render_eval_sets.py --manifests $M1 --split val  --out data/eval
uv run python scripts/render_eval_sets.py --manifests $M1 --split test --out data/eval    --faults
uv run python scripts/render_eval_sets.py --manifests $M2 --split val  --out data/eval_r2
uv run python scripts/render_eval_sets.py --manifests $M2 --split test --out data/eval_r2 --faults
echo "render done $(date)"
