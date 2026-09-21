#!/usr/bin/env bash
# Tier 4.6 refiner training on the GPU box. Usage: run_tier46.sh [config] (default configs/exp/vaani_tier46_refiner.yaml).
# A box-local copy of the config may raise num_workers (256 vCPU host: 32) and point val.eval_root at the 148-item
# screen subset; the per-item RNG keeps both result-neutral. Re-runnable: the trainer resumes from last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=${1:-configs/exp/vaani_tier46_refiner.yaml}
name=$(uv run --all-extras python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["name"])' "$cfg")
# Loader workers and the screen pool inherit these; without them every process claims the host's whole BLAS/numba
# pool and the box's pid cgroup (7680, threads count) runs out.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# RirBank unpacks .npz -> .npy on first open under shared temp names; do it once here so 32 loaders only mmap.
uv run --all-extras python -c 'import sys, torch, yaml; from vaani.data.rirs import RirBank; c = yaml.safe_load(open(sys.argv[1])); RirBank(torch.load(c["base_checkpoint"], weights_only=True, map_location="cpu")["config"]["data"]["bank"])' "$cfg"
mkdir -p runs
if [ ! -f runs/$name/DONE ]; then
  uv run --all-extras python -u -m vaani.train_refiner "$cfg" >> runs/$name.log 2>&1 || { echo "refiner failed, see runs/$name.log" >&2; exit 1; }
  touch runs/$name/DONE
fi
echo "tier46 refiner done: runs/$name (best.pt, run.json). Next: screen_tier46.py refiner --checkpoint runs/$name/best.pt on the full val split."
