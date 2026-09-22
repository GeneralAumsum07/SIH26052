#!/bin/bash
# Resume the two short backbones once the box has room, then refine them.
#
# r6_e32 and r6_ctl64 were paused at epoch 5 (both have last.pt; their configs set resume: true) because
# six concurrent trainers had the box at load 132 on 128 cores and e256 - the critical path - had slowed
# from 2.6 to 3.9 min/epoch. Neither of these two is on the critical path: e256 finishes last whatever
# they do, so running them later costs nothing and running them now costs e256 hours.
#
# They restart when e128 and both arm refiners are done, which is when the box has capacity again.
cd /workspace/SIH26052
R=results_r2/r6
log(){ echo "$(date -u +%H:%M:%S) $*" >> /workspace/resume.log; }

done_at(){ # $1 log, $2 budget -> 0 when the run reached its last epoch
  [ -f "$1" ] || return 1
  local n
  n=$(grep -c "^epoch" "$1" 2>/dev/null || echo 0)
  [ "$n" -ge "$2" ]
}

log "=== waiting for e128 (128) + both arm refiners (32) ==="
while true; do
  if done_at $R/train_r6_e128.log 128 \
     && done_at $R/train_r6_demand64_refiner.log 32 \
     && done_at $R/train_r6_wham64_refiner.log 32; then
    break
  fi
  sleep 60
done
log "capacity free; resuming the short backbones"

run(){ # name gpu workers budget
  local n=$1 g=$2 w=$3 b=$4
  if done_at $R/train_$n.log "$b"; then log "$n already complete"; return 0; fi
  log "START backbone $n gpu=$g w=$w"
  CUDA_VISIBLE_DEVICES=$g VAANI_WORKERS=$w uv run python -m vaani.train "configs/retraining/$n.yaml" >> "$R/train_$n.log" 2>&1
  log "END backbone $n rc=$?"
  if [ -f "runs/$n/best.pt" ]; then
    log "START refiner $n"
    CUDA_VISIBLE_DEVICES=$g VAANI_WORKERS=12 uv run python -m vaani.train_refiner "configs/retraining/${n}_refiner.yaml" > "$R/train_${n}_refiner.log" 2>&1
    log "END refiner $n rc=$?"
  else
    log "SKIP refiner $n: no best.pt"
  fi
}

( run r6_e32   1 20 32 ) &
( run r6_ctl64 0 20 64 ) &
wait
log "=== done ==="
