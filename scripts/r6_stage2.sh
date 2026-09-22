#!/bin/bash
# r6 stage-2 supervisor.
#
# The ResidualRefiner sits on a FROZEN backbone (FrozenCascade sets requires_grad_(False) and keeps
# .first in eval permanently), so it cannot be trained jointly with the base - it is a second stage by
# construction. This launches each backbone's refiner the moment that backbone finishes, so stage 2
# overlaps with the backbones still running instead of queueing behind all of them.
#
# The refiner configs carry no data: section on purpose - train_refiner takes the recipe from the base
# checkpoint's own config, so each arm's refiner trains on exactly the corpus its backbone did.
cd /workspace/SIH26052
R=results_r2/r6
SLOG=/workspace/supervisor.log
log(){ echo "$(date -u +%H:%M:%S) $*" >> "$SLOG"; }

wait_marker(){  # $1 marker, $2 file
  while ! grep -q "$1" "$2" 2>/dev/null; do sleep 20; done
  log "saw $1"
}

run_backbone(){  # name gpu workers
  local n=$1 g=$2 w=$3
  if [ -f "runs/$n/best.pt" ] && grep -q "BACKBONE_${n}_DONE" "$R/train_$n.log" 2>/dev/null; then
    log "backbone $n already complete"; return 0
  fi
  log "START backbone $n gpu=$g workers=$w"
  CUDA_VISIBLE_DEVICES=$g VAANI_WORKERS=$w uv run python -m vaani.train "configs/retraining/$n.yaml" > "$R/train_$n.log" 2>&1
  local rc=$?
  echo "BACKBONE_${n}_DONE_rc$rc" >> "$R/train_$n.log"
  log "END backbone $n rc=$rc"
  return $rc
}

run_refiner(){  # backbone-name gpu workers
  local n=$1 g=$2 w=$3
  if [ -f "runs/${n}_refiner/best.pt" ] && grep -q "REFINER_${n}_DONE" "$R/train_${n}_refiner.log" 2>/dev/null; then
    log "refiner $n already complete"; return 0
  fi
  if [ ! -f "runs/$n/best.pt" ]; then log "SKIP refiner $n: runs/$n/best.pt missing"; return 1; fi
  log "START refiner $n gpu=$g workers=$w (base runs/$n/best.pt)"
  CUDA_VISIBLE_DEVICES=$g VAANI_WORKERS=$w uv run python -m vaani.train_refiner "configs/retraining/${n}_refiner.yaml" > "$R/train_${n}_refiner.log" 2>&1
  local rc=$?
  echo "REFINER_${n}_DONE_rc$rc" >> "$R/train_${n}_refiner.log"
  log "END refiner $n rc=$rc"
}

log "=== supervisor start ==="

# arms are already running; refine each as it lands
( wait_marker ARMD_EXIT_ "$R/train_r6_demand64.log"; run_refiner r6_demand64 1 16 ) &
( wait_marker ARMW_EXIT_ "$R/train_r6_wham64.log";  run_refiner r6_wham64  1 16 ) &

# the two short backbones wait for the arms to free capacity, then run and refine
( wait_marker ARMD_EXIT_ "$R/train_r6_demand64.log"
  wait_marker ARMW_EXIT_ "$R/train_r6_wham64.log"
  run_backbone r6_e32 1 24 && run_refiner r6_e32 1 16 ) &

( wait_marker ARMD_EXIT_ "$R/train_r6_demand64.log"
  wait_marker ARMW_EXIT_ "$R/train_r6_wham64.log"
  run_backbone r6_ctl64 0 24 && run_refiner r6_ctl64 0 16 ) &

# the two long backbones are already running
( wait_marker E128_EXIT_ "$R/train_r6_e128.log"; run_refiner r6_e128 1 20 ) &
( wait_marker E256_EXIT_ "$R/train_r6_e256.log"; run_refiner r6_e256 0 24 ) &

wait
log "=== supervisor done ==="
