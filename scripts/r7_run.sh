#!/bin/bash
# r7: one 64-epoch warm restart of the r6_e256 backbone (configs/retraining/r7_e256_wr64.yaml:
# init_from runs/r6_e256/best.pt, sha256-pinned; everything else byte-identical to r6_e256), then its
# 2,498-param refiner on the frozen result, then scoring on both frozen sets.
#
# Strictly sequential, because each stage consumes the previous one's finished checkpoint. Every
# stage is idempotent: rerunning this script after a crash skips what is complete and resumes the rest
# (train/train_refiner resume from their own last.pt). "Complete" is judged the way r6_evals.py
# learned to: a run is done when its log shows the last epoch of its budget (best.pt alone proves
# nothing), and a csv is done when it has one row per item (a killed eval leaves a short csv).
set -u
cd /workspace/SIH26052
R=results_r2/r7; mkdir -p "$R"
LOG=/workspace/r7.log
GPU=${GPU:-1}
EVAL_WORKERS=${EVAL_WORKERS:-12}
N=r7_e256_wr64
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }
last_epoch(){ tr '\r' '\n' < "$1" 2>/dev/null | grep -aoE '^epoch [0-9]+' | awk '{print $2}' | sort -n | tail -1; }
finished(){ local e; e=$(last_epoch "$1"); [ -n "$e" ] && [ "$e" -ge $(($2 - 1)) ]; }
rows(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }
want(){ echo $(( $(find "$1/test" -name '*.json' | wc -l) + 1 )); }

log "=== r7 start gpu=$GPU ==="

if finished "$R/train_$N.log" 64; then log "backbone $N already complete"; else
  log "START backbone $N"
  CUDA_VISIBLE_DEVICES=$GPU VAANI_WORKERS=32 uv run python -m vaani.train "configs/retraining/$N.yaml" >> "$R/train_$N.log" 2>&1
  log "END backbone $N rc=$?"
  finished "$R/train_$N.log" 64 || { log "FAIL backbone $N did not reach epoch 63"; exit 1; }
fi

if finished "$R/train_${N}_refiner.log" 32; then log "refiner $N already complete"; else
  log "START refiner $N"
  CUDA_VISIBLE_DEVICES=$GPU VAANI_WORKERS=24 uv run python -m vaani.train_refiner "configs/retraining/${N}_refiner.yaml" >> "$R/train_${N}_refiner.log" 2>&1
  log "END refiner $N rc=$?"
  finished "$R/train_${N}_refiner.log" 32 || { log "FAIL refiner $N did not reach epoch 31"; exit 1; }
fi

# cascade (the deployable system) and backbone, on both sets, side by side: eval is CPU-bound
pids=()
for spec in "cascade:runs/${N}_refiner/best.pt|${N}_cascade" "ckpt:runs/$N/best.pt|$N"; do
  sys=${spec%%|*}; name=${spec##*|}
  for set in "eval_r2|data/eval_r2" "gen|data/eval_gen"; do
    tag=${set%%|*}; root=${set##*|}; csv="$R/${name}_${tag}.csv"; w=$(want "$root")
    if [ "$(rows "$csv")" -ge "$w" ]; then log "eval ${name}_$tag already complete"; continue; fi
    [ -f "$csv" ] && mv -f "$csv" "$csv.partial"
    log "START eval ${name}_$tag ($sys)"
    ( CUDA_VISIBLE_DEVICES=$GPU uv run python -m vaani.eval --system "$sys" --split test --eval-root "$root" \
        --workers "$EVAL_WORKERS" --dnsmos --out "$csv" > "$R/eval_${name}_$tag.log" 2>&1
      r=$(rows "$csv")
      if [ "$r" -ge "$w" ]; then log "END eval ${name}_$tag rows=$r"; else log "FAIL eval ${name}_$tag rows=$r want=$w"; fi ) &
    pids+=($!)
  done
done
wait "${pids[@]}"
log "=== r7 done ==="
