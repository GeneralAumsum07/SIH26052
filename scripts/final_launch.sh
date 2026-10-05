#!/usr/bin/env bash
# Final runs (2026-10-05): r8_final_mid on GPU 0, r8_final_large_plus on GPU 1. After scripts/r8_box_setup.sh (no --launch):
#   bash scripts/final_launch.sh                                       # open-ended: full FULL_STEPS schedule, runs until stopped
#   TRAIN_END="2026-10-05 23:30" bash scripts/final_launch.sh          # time-boxed: sized to end by TRAIN_END (IST)
#   bash scripts/final_launch.sh status | export | stop                # export works mid-run (checkpoints are write-then-rename)
# 1. smoke: both configs train 2 short epochs in parallel; TensorBoard wall times give s/step and val overhead per epoch
# 2. sizing: TRAIN_END set -> max_steps so steps + val fit before it (5 % margin); unset -> FULL_STEPS
# 3. launch: the tiers that passed, in tmux "final"; resumable (resume: true), relaunched up to 20 times on a crash
# Env: TRAIN_END, FULL_STEPS (200000), RUNS (= runs/final), VAANI_SCREEN_WORKERS (4), GPU_MID (0), GPU_LP (1).
set -uo pipefail
cd "$(dirname "$0")/.."
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
PY="${PY:-.venv/bin/python}"; RUNS="${RUNS:-runs/final}"; mkdir -p "$RUNS/logs"
export VAANI_SCREEN_WORKERS="${VAANI_SCREEN_WORKERS:-4}"
GPU_MID="${GPU_MID:-0}"; GPU_LP="${GPU_LP:-1}"; FULL_STEPS="${FULL_STEPS:-200000}"
read -ra NAMES <<< "${TIERS:-mid large_plus}"; declare -A GPU=([mid]=$GPU_MID [large_plus]=$GPU_LP)
# val screen clips per forward: whole val clips beside the CUDA graph pool; Large+ at 16 ran out of VRAM on the laptop
declare -A SB=([mid]=8 [large_plus]=4)
say() { echo "$(date '+%F %T') [final] $*" | tee -a "$RUNS/final.log"; }

# OMP/BLAS pools size to nproc, not the CFS quota (pids.max blew up on the 2026-09-29 box): cap them
caps="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4"
ncpus() { local n q p; n=$(nproc)
  if read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max && [ "$q" != max ] && [ $((q / p)) -lt "$n" ]; then n=$((q / p)); fi; echo "$n"; }
# loader workers per run: half the quota after trainers + screens, and ~1 GB RSS each must fit in 70 % of MemAvailable
workers() { local c m; c=$(( ($(ncpus) - 8 - 2 * VAANI_SCREEN_WORKERS) / 2 ))
  m=$(awk '/MemAvailable/ {print int($2 / 1048576 * 0.7 / 2)}' /proc/meminfo)
  [ -n "$m" ] && [ "$m" -lt "$c" ] && c=$m; [ "$c" -lt 2 ] && c=2; echo "$c"; }

# derived config: cfg_out <tier> <out> <python dict updates...>
cfg_out() { "$PY" - "$1" "$2" "$3" <<'EOF'
import ast, os, sys, yaml
t, out, upd = sys.argv[1], sys.argv[2], ast.literal_eval(sys.argv[3])
c = yaml.safe_load(open(f"{os.environ.get('CFG_DIR', 'configs/retraining')}/r8_final_{t}.yaml"))
for k, v in upd.items():
    d = c
    *path, last = k.split(".")
    for p in path: d = d[p]
    d[last] = v
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
EOF
}

# per-tier config overrides found by the smoke, kept on disk so a rerun of launch reuses them
ovr() { cat "$RUNS/ovr_$1" 2>/dev/null; }

smoke() {  # 2 epochs x 200 steps; composite every 2: epoch 0's val point is plain, the final one is composite
  # Mid on the laptop sometimes went non-finite from step 0 under compile + CUDA graph (eager was finite), so a failed
  # tier is retried: tries 1-2 as configured, 3 without the CUDA graph, 4 eager (no compile either). A tier that passes
  # leaves smoke_ok_<tier>; one that never passes is dropped and the other still launches
  local t try i pids pending=("${NAMES[@]}") next
  for t in "${pending[@]}"; do rm -f "$RUNS/ovr_$t" "$RUNS/smoke_ok_$t"; done
  for try in 1 2 3 4; do
    [ ${#pending[@]} = 0 ] && break
    pids=()
    for t in "${pending[@]}"; do
      [ $try = 3 ] && echo "'perf.numerics.cuda_graph': False, " > "$RUNS/ovr_$t"
      [ $try = 4 ] && echo "'perf.numerics.cuda_graph': False, 'perf.numerics.compile': False, " > "$RUNS/ovr_$t"
      cfg_out $t "$RUNS/smoke_$t.yaml" "{$(ovr $t)'name': 'smoke_$t', 'runs_dir': '$RUNS/smoke', 'resume': False, 'epochs': 2, 'max_steps': 400, 'log_every': 20, 'data.epoch_len': 6400, 'val.composite.every': 2}"
      rm -rf "$RUNS/smoke/smoke_$t"
      say "smoke $t on GPU ${GPU[$t]} ($(workers) workers, try $try$( [ $try -ge 3 ] && echo " overrides: $(ovr $t)"))"
      env $caps VAANI_SCREEN_BATCH=${SB[$t]} CUDA_VISIBLE_DEVICES=${GPU[$t]} VAANI_WORKERS=$(workers) $PY -m vaani.train "$RUNS/smoke_$t.yaml" > "$RUNS/logs/smoke_$t.try$try.log" 2>&1 &
      pids+=($!)
    done
    next=(); i=0
    for t in "${pending[@]}"; do
      if wait "${pids[$i]}" && grep -q '"end"' "$RUNS/smoke/smoke_$t/run.json" 2>/dev/null; then
        touch "$RUNS/smoke_ok_$t"; say "smoke $t passed (try $try)"
      else
        say "smoke $t failed (try $try): $(grep -m1 -E 'Error|non-finite' "$RUNS/logs/smoke_$t.try$try.log")"; next+=("$t")
      fi
      i=$((i + 1))
    done
    pending=("${next[@]}")
  done
  [ ${#pending[@]} = 0 ] || say "SMOKE FAILED for ${pending[*]} after 4 tries (logs $RUNS/logs/smoke_*): not launched"
}

datacheck() {  # every input the configs read, before spending 10 minutes on a smoke
  "$PY" - "${NAMES[@]}" <<'EOF2'
import os, sys, yaml
miss = []
for t in sys.argv[1:]:
    c = yaml.safe_load(open(f"{os.environ.get('CFG_DIR', 'configs/retraining')}/r8_final_{t}.yaml")); d = c["data"]
    miss += [p for p in [*d["manifests"], d["bank"], d["exclude_groups_file"], f"{c['val']['eval_root']}/{c['val']['split']}"]
             if not os.path.exists(p)]
if miss:
    sys.exit("MISSING (datasets still arriving? tail -f runs/box_setup/datasets.log): " + ", ".join(sorted(set(miss))))
print("data ok")
EOF2
}

size() {  # prints "<tier> <max_steps> <epochs> <s_per_step> <val_s_per_epoch> <startup_s>" from the smoke's TB wall times
  "$PY" - "$RUNS" "$TRAIN_END" "${NAMES[@]}" <<'EOF'
import glob, json, math, sys, time
from datetime import datetime, timedelta, timezone
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
IST = timezone(timedelta(hours=5, minutes=30))          # TRAIN_END is IST whatever the box's clock zone is
runs, end = sys.argv[1], datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M").replace(tzinfo=IST).timestamp()
avail = (end - time.time()) * 0.95                     # 5 % margin for the last val point and the final save
for t in sys.argv[3:]:
    d = f"{runs}/smoke/smoke_{t}"
    ev = {}
    for f in glob.glob(f"{d}/events.*"):
        a = EventAccumulator(f, size_guidance={"scalars": 0}); a.Reload()
        if "train/loss" in a.Tags()["scalars"]:
            ev.update({s.step: s.wall_time for s in a.Scalars("train/loss")})
    rj = json.load(open(f"{d}/run.json"))
    # steady state: 100 -> 200 (epoch 0) and 240 -> 400 (epoch 1); 200 -> 220 and 400 -> end hold one val point each
    sps = ((ev[200] - ev[100]) + (ev[400] - ev[240])) / 260
    plain, comp = ev[220] - ev[200] - 20 * sps, rj["end"] - ev[400]
    val = plain + (comp - plain) / 4                    # the real runs screen composite every 4th val point
    startup = ev[20] - rj["start"] - 20 * sps            # loader start + torch.compile + graph capture
    spe = 625                                            # epoch_len 20000 / batch 32
    steps = int((avail - startup) / (sps + val / spe))
    steps -= steps % spe or 0
    print(t, steps, math.ceil(steps / spe), round(sps, 4), round(val, 1), round(startup, 1))
EOF
}

launch() {
  command -v tmux > /dev/null || { echo "tmux missing (apt-get install -y tmux): the runs live in tmux session final"; exit 2; }
  say "box: $(ncpus) cpus, $(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' ' '), TRAIN_END ${TRAIN_END:-none, open-ended} (now $(date -u -d @$(( $(date +%s) + 19800 )) '+%F %T') IST)"
  datacheck || exit 1
  local all=("${NAMES[@]}") t todo=()
  for t in "${all[@]}"; do [ -f "$RUNS/smoke_ok_$t" ] || todo+=("$t"); done
  if [ ${#todo[@]} -gt 0 ]; then NAMES=("${todo[@]}"); smoke; fi
  NAMES=(); for t in "${all[@]}"; do [ -f "$RUNS/smoke_ok_$t" ] && NAMES+=("$t"); done
  [ ${#NAMES[@]} -gt 0 ] || { say "nothing passed the smoke: not launching"; exit 1; }
  local line t steps ep sps val st
  if [ -n "${TRAIN_END:-}" ]; then size > "$RUNS/sizing.txt" || { say "SIZING FAILED"; exit 1; }
  else  # open-ended: the full recipe schedule (Arm B Mini's 200k); nothing here ever stops a run, only `stop` does
    : > "$RUNS/sizing.txt"; for t in "${NAMES[@]}"; do echo "$t $FULL_STEPS $(( (FULL_STEPS + 624) / 625 )) - - -" >> "$RUNS/sizing.txt"; done
  fi
  while read -r t steps ep sps val st; do
    if [ "$sps" = - ]; then say "$t: open-ended, max_steps $steps ($ep epochs); runs until scripts/final_launch.sh stop"
    else say "$t: $sps s/step, $val s val/epoch, $st s startup -> max_steps $steps ($ep epochs)"; fi
    [ "$steps" -gt 0 ] || { say "no time left for $t"; continue; }
    cfg_out $t "$RUNS/r8_final_$t.yaml" "{$(ovr $t)'runs_dir': '$RUNS', 'max_steps': $steps, 'epochs': $ep}"
    tmux has-session -t final 2>/dev/null || tmux new-session -d -s final -n idle || { say "LAUNCH FAILED: tmux new-session"; exit 1; }
    tmux new-window -t final -n "$t" "ulimit -n \$(ulimit -Hn); for i in \$(seq 20); do env $caps VAANI_SCREEN_BATCH=${SB[$t]} CUDA_VISIBLE_DEVICES=${GPU[$t]} VAANI_WORKERS=$(workers) $PY -m vaani.train $RUNS/r8_final_$t.yaml >> $RUNS/logs/$t.log 2>&1 && grep -q '\"end\"' $RUNS/r8_final_$t/run.json && { date > $RUNS/r8_final_$t/DONE; break; }; sleep 20; done; exec bash" || { say "LAUNCH FAILED for $t: tmux new-window"; exit 1; }
    say "launched $t on GPU ${GPU[$t]} (tmux final:$t, log $RUNS/logs/$t.log)"
  done < "$RUNS/sizing.txt"
}

status() {
  local t; for t in "${NAMES[@]}"; do
    echo "== $t $( [ -f "$RUNS/r8_final_$t/DONE" ] && echo DONE)"; grep -E '^epoch' "$RUNS/logs/$t.log" 2>/dev/null | tail -2
    "$PY" -c "
import json; r=json.load(open('$RUNS/r8_final_$t/run.json')); print('steps', r.get('steps'), 'best', r.get('best_metric'), r.get('best_key'))" 2>/dev/null
  done
}

export_() {  # ONNX + folded graph + model_config.json of each run's best.pt (last.pt if no best yet), as the Pi bundle does
  local t ck; for t in "${NAMES[@]}"; do
    ck="$RUNS/r8_final_$t/best.pt"; [ -f "$ck" ] || ck="$RUNS/r8_final_$t/last.pt"; [ -f "$ck" ] || continue
    mkdir -p "$RUNS/export/$t"
    $PY -c "from vaani import export as E, live as L
r = E.export_fe(E.fe_load('$ck'), '$RUNS/export/$t/final_$t.onnx')
L.write_model_config('$ck', '$RUNS/export/$t/model_config.json', '$RUNS/export/$t/final_$t.folded.onnx')
print('$t', '$ck', r['params'], r['mmac_per_s'])" && say "exported $t from $ck"
  done
}

stop() {  # the only thing that ends an open-ended run; safe mid-step (atomic checkpoints), `launch` again resumes from last.pt
  tmux kill-session -t final 2>/dev/null && say "stopped: tmux session final killed" || echo "no tmux session final"
}

case "${1:-launch}" in launch) launch;; status) status;; export) export_;; stop) stop;; smoke) smoke;; size) size;; datacheck) datacheck;; *) sed -n '2,9p' "$0"; exit 2;; esac
