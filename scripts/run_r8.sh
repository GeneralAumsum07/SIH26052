#!/usr/bin/env bash
# r8 schedule on a 2-GPU box (plan 11.6, configs/retraining/R8_RUNBOOK.md): one resumable queue per GPU, in tmux.
#   bash scripts/run_r8.sh start [all|pilots|full]   launch both queues in tmux session "r8" (default: all)
#   bash scripts/run_r8.sh status                    every queued run: DONE / RUNNING / FAILED / DROPPED / PENDING
#   bash scripts/run_r8.sh next [0|1]                the next run each queue would start
#   bash scripts/run_r8.sh workers                   loader workers per queue (after the memory cap)
#   bash scripts/run_r8.sh go-full                   release the full runs after the pilots (winning settings copied)
#   bash scripts/run_r8.sh _queue <gpu> <phase>      the queue itself (what tmux runs)
# Low-delay queue (low-delay plan Task 8 / Section 3.10; names registered below, resolved by scripts/r8_ld_queue.py):
#   bash scripts/run_r8.sh ld-start [all|pilots|full]   LD_GPUS x LD_SLOTS lanes, the scorer and the batch servers
#   bash scripts/run_r8.sh ld-plan [phase]              every queued name -> config, network, contract, run dir
#   bash scripts/run_r8.sh ld-status                    DONE / RUNNING / PENDING / WAITING / NOT_ELIGIBLE / STOPPED ...
#   bash scripts/run_r8.sh ld-decide stage1 arm_a|arm_b | stage2 none|<ld_s2 stems>   record, stop ruled-out runs
#   bash scripts/run_r8.sh ld-go-full                   authorize the full runs (readiness evidence still required)
#   bash scripts/run_r8.sh _ld_lane <gpu> <slot> <phase>
# Resumable: a run with runs/<name>/DONE is skipped; any other run is relaunched with the same command, and
# train.py resumes it from runs/<name>/last.pt (resume: true). Rerun "start" after a reboot; nothing is lost.
# Full runs refuse to start unless the G1 result ($G1_JSON) has gate_pass true; pilots too, unless
# ALLOW_PILOTS_WITHOUT_G1=1. "all" runs the pilots, then waits for "go-full" (plan: the winning pilot settings are
# copied into the full configs first); FULL_GO=1 skips that wait.
# Env: DRY_RUN=1 (print, run nothing), PILOT_HOURS (12: no pilot starts later than this after its queue began;
# the rest are logged as dropped), WORKERS_GPU0/WORKERS_GPU1 (default (nproc - RESERVE_CPUS) / 2, lowered so both
# queues' loaders and screens fit in MemAvailable at VAANI_WORKER_RSS_GB per process; MEMINFO for tests),
# VAANI_SCREEN_WORKERS (8), MAX_TRIES (3), TRAIN_CMD, RUNS_DIR (runs), G1_JSON, NO_TMUX=1 (background, no tmux).
set -uo pipefail
cd "$(dirname "$0")/.."

RUNS_DIR="${RUNS_DIR:-runs}"; Q="$RUNS_DIR/r8_queue"
G1_JSON="${G1_JSON:-results_r2/r8/data_gates/box_g1/v2.json}"
# the synced venv directly: no uv resolve per launch, and numba + torch_pesq come from `uv sync --all-extras`
if [ -z "${TRAIN_CMD:-}" ]; then
  if [ -x .venv/bin/python ]; then TRAIN_CMD=".venv/bin/python -m vaani.train"
  else TRAIN_CMD="uv run --extra fast --extra train python -m vaani.train"; fi
fi
PILOT_HOURS="${PILOT_HOURS:-12}"; MAX_TRIES="${MAX_TRIES:-3}"; RESERVE_CPUS="${RESERVE_CPUS:-4}"
MEMINFO="${MEMINFO:-/proc/meminfo}"   # no MemAvailable there = no memory cap
# INTERIM per-worker memory (GB) until results_r2/r8/loader_rss/ measures it: a private copy of what one v2 worker loads,
# bank_r8 sidecars 0.640 speech + 1.920 noise (memmapped, counted as private) + dataset 0.019 (laptop pickle) = 2.58 -> 3
WORKER_RSS_GB_DEFAULT=3
export VAANI_SCREEN_WORKERS="${VAANI_SCREEN_WORKERS:-8}"   # two runs' composite screens must not starve the loaders
A=configs/retraining/r8_ablations
# D4 (low-delay plan): the r8 product is the low-delay Mini, so the legacy queue keeps only C0 (ab1_fe_mini_s0/s1 and
# the full r8_fe_mini). The refvalid arms, the 32 ms ab2-ab6 pilots and the opt-in ab7_bank_r3 leave the queue; their
# configurations stay in the repository. Both queues key run directories by name, so C0 is never trained twice.
PILOTS_0=(ab1_fe_mini_s0 ab1_fe_mini_s1)
PILOTS_1=()
FULL_0=(r8_fe_mini); FULL_1=()
# ---- low-delay registry (Section 3.10 priorities; arms.json holds each name's class, wave and stop rules) ----------
LD_P1=(ab1_fe_mini_s0 ab1_fe_mini_s1 ld_a_s0 ld_a_s1 ld_b_s0 ld_b_s1)              # Arm B: only when Gate 0a pilots it
LD_P2=(ld_s2_overparam ld_s2_gru_default ld_s2_mrstft05 ld_s2_warmup480 ld_s2_native)   # Stage 2 on Arm A, seed 0
LD_P3=(ab1_fe_mini_s2 ab1_fe_mini_s3 ab1_fe_mini_s4 ld_a_s2 ld_a_s3 ld_a_s4
       ld_conf_s0 ld_conf_s1 ld_conf_s2 ld_conf_s3 ld_conf_s4)                     # ld_conf_*: wave 2 (Stage-2 decision)
LD_P4=(ld_r_s0 ld_r_s1 ld_p4_tail00 ld_p4_tail10 ld_p4_refdrop00 ld_p4_refdrop30 ld_p4_bounded
       ld_p4_kappa1 ld_p4_kappa2 ld_p4_kappa4)                                      # Arm R at the lowest pilot priority
LD_FULL=(r8_fe_mini r8_ld_fe_mini r8_ld_fe_mini_overparam r8_ld_fe_mini_armb r8_ld_fe_mini_conf)   # D4 + D8 (speculative)
LD_GPUS="${LD_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)}"; [ "${LD_GPUS:-0}" -ge 1 ] 2>/dev/null || LD_GPUS=1
LD_SLOTS="${LD_SLOTS:-1}"                  # concurrent runs per GPU: the first hour's concurrency scan sets it
LD_SCORER="${LD_SCORER:-async}"            # perf.ops, bit-exact: one scorer per box ...
LD_STREAM="${LD_STREAM:-shared}"           # ... and one batch server per rendered stream (local rendering if absent)
LD_MPS_PCT="${LD_MPS_PCT:-}"               # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for pilots (full runs uncapped)
export GATE0_JSON="${GATE0_JSON:-results_r2/r8_ld/gate0/eligibility.json}"
export LD_READY_JSON="${LD_READY_JSON:-$RUNS_DIR/r8_queue/preflight_ld.json}"

PY="${PY:-}"
[ -n "$PY" ] || for c in .venv/bin/python .venv/Scripts/python.exe python3 python; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi; done

cfg_of() { case "$1" in r8_*) echo "configs/retraining/$1.yaml";; *) echo "$A/$1.yaml";; esac; }
queue_of() {  # $1 gpu, $2 phase -> names
  local g=$1 ph=$2; local -n p="PILOTS_$g" f="FULL_$g"
  case "$ph" in pilots) echo "${p[@]:-}";; full) echo "${f[@]:-}";; all) echo "${p[@]:-} ${f[@]:-}";; esac
}
# training writes $RUNS_DIR/<config name> (train.py); markers stay in $RUNS_DIR/<queued name>
train_dir_of() {  # $1 queued name, $2 config
  local nm; nm=$(sed -n 's/^name: *//p' "$2" 2>/dev/null | head -1); echo "$RUNS_DIR/${nm:-$1}"
}
is_full() { case "$1" in r8_*) return 0;; *) return 1;; esac; }
state_of() {  # DONE / RUNNING / FAILED / DROPPED / PENDING
  local n=$1 d="$RUNS_DIR/$1"
  if [ -f "$d/DONE" ]; then echo DONE
  elif [ -f "$Q/running.$n" ] && kill -0 "$(cat "$Q/running.$n")" 2>/dev/null; then echo RUNNING
  elif [ -f "$d/FAILED" ]; then echo FAILED
  elif grep -qx "$n" "$Q/dropped.txt" 2>/dev/null; then echo DROPPED
  else echo PENDING; fi
}
g1_pass() {
  [ -f "$G1_JSON" ] || { echo "G1: $G1_JSON missing"; return 1; }
  "$PY" - "$G1_JSON" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); items = min((j.get(k) or {}).get("items", 0) for k in ("param", "room"))
ok = j.get("gate_pass") is True and items >= 200
print(f"G1: gate_pass {j.get('gate_pass')}, {items} items, seed {j.get('seed')}, bank {j.get('bank')}")
sys.exit(0 if ok else 1)
EOF
}
workers() {  # per-queue loader workers: two concurrent runs split the cores, and all their processes must fit in RAM
  local g=$1 v; v=$(eval echo "\${WORKERS_GPU$g:-}")
  [ -n "$v" ] && { echo "$v"; return; }
  local n; n=$(nproc 2>/dev/null || echo 8); v=$(( (n - RESERVE_CPUS) / 2 )); [ "$v" -lt 2 ] && v=2
  # each queue holds 2 persistent loaders of v workers (train + val, runtime.loader_kwargs) + the screen pool
  local ma m r="${VAANI_WORKER_RSS_GB:-$WORKER_RSS_GB_DEFAULT}"
  if ! awk -v r="$r" 'BEGIN { exit !(r ~ /^[0-9]*\.?[0-9]+$/ && r + 0 > 0) }'; then   # 0 or junk must not lift the cap
    echo "workers gpu$g: VAANI_WORKER_RSS_GB='$r' is not a positive number; using $WORKER_RSS_GB_DEFAULT" >&2
    r=$WORKER_RSS_GB_DEFAULT
  fi
  ma=$(awk '/^MemAvailable:/ {print $2; exit}' "$MEMINFO" 2>/dev/null)
  if [ -n "$ma" ]; then
    m=$(awk -v kb="$ma" -v r="$r" -v s="$VAANI_SCREEN_WORKERS" \
      'BEGIN { w = int((kb * 1024 / 1e9 / 2 / r - s) / 2); print (w < 1 ? 1 : w) }')
    if [ "$m" -lt "$v" ]; then
      echo "workers gpu$g: capped $v -> $m by memory (MemAvailable $(awk -v kb="$ma" 'BEGIN { printf "%.1f", kb * 1024 / 1e9 }') GB, $r GB per worker," \
        "2 loaders x workers + $VAANI_SCREEN_WORKERS screen processes per queue, 2 queues)" >&2
      v=$m
    fi
  fi
  echo "$v"
}
log() { echo "$(date '+%F %T') [gpu$1] $2" | tee -a "$Q/queue.log"; }

run_queue() {  # $1 gpu, $2 phase
  local g=$1 ph=$2 n c w tries rc
  mkdir -p "$Q/logs"
  # a dry run must not start the PILOT_HOURS clock
  [ -f "$Q/gpu$g.started" ] || [ "${DRY_RUN:-0}" = 1 ] || date +%s > "$Q/gpu$g.started"
  local t0; t0=$(cat "$Q/gpu$g.started" 2>/dev/null || date +%s)
  local g1ok=0; g1_pass >/dev/null && g1ok=1   # once per queue for the pilots; each full run re-reads it
  for n in $(queue_of "$g" "$ph"); do
    c=$(cfg_of "$n"); [ "$(state_of "$n")" = DONE ] && continue
    [ "$(state_of "$n")" = DROPPED ] && continue
    [ -f "$c" ] || { log "$g" "$n: $c missing, skipped"; continue; }
    if is_full "$n"; then
      if [ "$ph" = all ] && [ "${FULL_GO:-0}" != 1 ]; then   # the pilots' winners go into the full configs first
        until [ -f "$Q/full_go" ] || [ "${DRY_RUN:-0}" = 1 ]; do
          log "$g" "pilots done; waiting for 'run_r8.sh go-full' before $n"; sleep 600; done
      fi
      g1_pass || { log "$g" "REFUSED $n: the G1 gate has not passed on this box"; return 1; }
    else
      if [ "${ALLOW_PILOTS_WITHOUT_G1:-0}" != 1 ] && [ $g1ok != 1 ]; then
        log "$g" "REFUSED $n: the G1 gate has not passed ($G1_JSON)"; return 1; fi
      if [ $(( $(date +%s) - t0 )) -gt $(( PILOT_HOURS * 3600 )) ] && [ "$(state_of "$n")" = PENDING ]; then
        echo "$n" >> "$Q/dropped.txt"; log "$g" "DROPPED $n: past PILOT_HOURS=$PILOT_HOURS"; continue; fi
    fi
    w=$(workers "$g" 2>"$Q/workers$g.msg")   # a memory cap goes into queue.log beside the run it throttles
    [ -s "$Q/workers$g.msg" ] && log "$g" "$(cat "$Q/workers$g.msg")"
    local cmd="CUDA_VISIBLE_DEVICES=$g VAANI_WORKERS=$w VAANI_SCREEN_WORKERS=$VAANI_SCREEN_WORKERS $TRAIN_CMD $c"
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY gpu$g: $cmd"; continue; fi
    rm -f "$RUNS_DIR/$n/FAILED"; tries=0; rc=1
    while [ $tries -lt "$MAX_TRIES" ]; do
      tries=$((tries + 1)); log "$g" "start $n (try $tries/$MAX_TRIES): $cmd"
      echo $$ > "$Q/running.$n"
      CUDA_VISIBLE_DEVICES=$g VAANI_WORKERS=$w $TRAIN_CMD "$c" >> "$Q/logs/$n.log" 2>&1; rc=$?
      rm -f "$Q/running.$n"
      if [ $rc = 0 ] && grep -q '"end": ' "$(train_dir_of "$n" "$c")/run.json" 2>/dev/null; then
        mkdir -p "$RUNS_DIR/$n"; date '+%F %T' > "$RUNS_DIR/$n/DONE"; log "$g" "DONE $n"; break; fi
      log "$g" "$n exited rc=$rc (log $Q/logs/$n.log); relaunching resumes from last.pt"; sleep 30
    done
    [ -f "$RUNS_DIR/$n/DONE" ] || { mkdir -p "$RUNS_DIR/$n"; echo "rc=$rc after $tries tries" > "$RUNS_DIR/$n/FAILED"; log "$g" "FAILED $n"; }
  done
  log "$g" "queue $ph finished"
}

# ---- low-delay queue ------------------------------------------------------------------------------------------
LDQ() { LD_NAMES="${LD_P1[*]} ${LD_P2[*]} ${LD_P3[*]} ${LD_P4[*]} ${LD_FULL[*]}" RUNS_DIR="$RUNS_DIR" \
        "$PY" scripts/r8_ld_queue.py "$@" --gpus "$LD_GPUS" --slots "$LD_SLOTS"; }
ld_lane() {  # $1 gpu, $2 slot, $3 phase: claim the highest-priority runnable job, run it, repeat
  local g=$1 sl=$2 ph=$3 line n c cn prio w ni full spec tries rc ops env
  mkdir -p "$Q/logs"
  while true; do
    line=$(LDQ next --phase "$ph" --gpu "$g" 2>>"$Q/ld_lane$g.$sl.err") || {
      log "$g" "LD QUEUE ERROR: $(tail -1 "$Q/ld_lane$g.$sl.err")"; return 3; }
    [ -n "$line" ] || break
    IFS=$'\t' read -r n c cn prio w ni full spec <<< "$line"
    ops="{\"scorer\":\"$LD_SCORER\",\"stream\":\"$LD_STREAM\",\"priority\":$prio}"
    env="CUDA_VISIBLE_DEVICES=$g VAANI_WORKERS=$w VAANI_SCREEN_WORKERS=$VAANI_SCREEN_WORKERS"
    env="$env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    [ -n "$LD_MPS_PCT" ] && [ "$full" != True ] && env="$env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$LD_MPS_PCT"
    if [ "${DRY_RUN:-0}" = 1 ]; then
      echo "DRY gpu$g.$sl P$prio: $env VAANI_PERF_OPS='$ops' nice -n $ni $TRAIN_CMD $c"; continue; fi
    tries=0; rc=1
    while [ $tries -lt "$MAX_TRIES" ]; do
      tries=$((tries + 1)); log "$g" "start $n (lane $sl, P$prio, try $tries/$MAX_TRIES, speculative $spec): $c"
      # shellcheck disable=SC2086
      ( echo "$BASHPID" > "$Q/running.$n"; exec env $env VAANI_PERF_OPS="$ops" nice -n "$ni" $TRAIN_CMD "$c" ) \
        >> "$Q/logs/$n.log" 2>&1; rc=$?
      rm -f -- "${Q:?}/running.${n:?}"
      [ -f "$RUNS_DIR/$n/STOPPED" ] && { log "$g" "STOPPED $n (a decision ruled it out)"; break; }
      if [ $rc = 0 ] && grep -q '"end": ' "$RUNS_DIR/$cn/run.json" 2>/dev/null; then
        mkdir -p "$RUNS_DIR/$n"; date '+%F %T' > "$RUNS_DIR/$n/DONE"; log "$g" "DONE $n"; break; fi
      log "$g" "$n exited rc=$rc; relaunching resumes from last.pt"; sleep 30
    done
    if [ ! -f "$RUNS_DIR/$n/DONE" ] && [ ! -f "$RUNS_DIR/$n/STOPPED" ]; then
      mkdir -p "$RUNS_DIR/$n"; echo "rc=$rc after $tries tries" > "$RUNS_DIR/$n/FAILED"; log "$g" "FAILED $n"; fi
    rmdir "$Q/claims/$n" 2>/dev/null
  done
  log "$g" "ld lane $sl ($ph) finished"
}
ld_start() {  # $1 phase
  local ph=$1 g sl key c name n lane prio wave st net con td rest dirs=() pr=()
  LDQ plan --phase "$ph" > "$Q/ld_plan.txt" || { cat "$Q/ld_plan.txt"; return 3; }   # a missing config fails here
  cat "$Q/ld_plan.txt"
  if [ "${DRY_RUN:-0}" != 1 ] && ! grep -q '^# Gate 0a complete' "$Q/ld_plan.txt"; then
    echo "REFUSED: the Gate 0a record is not complete ($GATE0_JSON): no low-delay compute before Gate 0a"; return 1; fi
  mkdir -p "$Q/claims"
  for n in $(ls "$Q/claims"); do   # a claim without a live run is stale (reboot, or a dry run)
    if ! { [ -f "$Q/running.$n" ] && kill -0 "$(cat "$Q/running.$n")" 2>/dev/null; }; then rmdir "$Q/claims/$n"; fi
  done
  if [ "$LD_STREAM" = shared ]; then   # one batch server per rendered stream
    while IFS=$'\t' read -r key c name; do
      [ -n "$key" ] || continue
      if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY stream $key: $PY -m vaani.data.stream_server --config $c"
      else tmux new-window -t r8ld -n "s$key" "$PY -m vaani.data.stream_server --config $c; exec bash"; fi
    done < <(LDQ streams --phase "$ph")
  fi
  if [ "$LD_SCORER" = async ]; then     # one scorer per box; its queue order follows the priority classes
    while IFS=$'\t' read -r lane prio wave st n c net con td rest; do
      case "$lane" in gpu*) dirs+=("$td"); pr+=("$td=${prio#P}");; esac
    done < "$Q/ld_plan.txt"
    if [ "${#dirs[@]}" -gt 0 ]; then
      if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY scorer: $PY scripts/r8_scorer.py --runs ${dirs[*]} --priority ${pr[*]}"
      else tmux new-window -t r8ld -n scorer "$PY scripts/r8_scorer.py --runs ${dirs[*]} --priority ${pr[*]}; exec bash"; fi
    fi
  fi
  for sl in $(seq 0 $((LD_SLOTS - 1))); do for g in $(seq 0 $((LD_GPUS - 1))); do
    if [ "${DRY_RUN:-0}" = 1 ]; then ld_lane "$g" "$sl" "$ph"
    else tmux new-window -t r8ld -n "g$g.$sl" "bash scripts/run_r8.sh _ld_lane $g $sl $ph; exec bash"; fi
  done; done
  if [ "${DRY_RUN:-0}" = 1 ]; then for n in $(ls "$Q/claims"); do rmdir "$Q/claims/$n"; done; fi
}

cmd="${1:-status}"; shift || true
mkdir -p "$Q"
case "$cmd" in
  ld-start)
    ph="${1:-all}"; case "$ph" in all|pilots|full) ;; *) echo "phase must be all|pilots|full"; exit 2;; esac
    if [ "$ph" != pilots ] || [ "${ALLOW_PILOTS_WITHOUT_G1:-0}" != 1 ]; then
      g1_pass || { echo "REFUSED: run the G1 gate on this box first (scripts/r8_box_setup.sh does)"; exit 1; }; fi
    if [ "${DRY_RUN:-0}" != 1 ]; then   # refuse before any session exists (ld_start re-checks on its own plan)
      LDQ plan --phase "$ph" 2>&1 | grep -q '^# Gate 0a complete' || {
        echo "REFUSED: the Gate 0a record is not complete ($GATE0_JSON): no low-delay compute before Gate 0a"; exit 1; }
      command -v tmux >/dev/null || { echo "tmux missing (apt-get install -y tmux)"; exit 1; }
      tmux has-session -t r8ld 2>/dev/null && { echo "tmux session r8ld already exists: tmux attach -t r8ld"; exit 1; }
      tmux new-session -d -s r8ld -n ctl "bash scripts/run_r8.sh ld-status; exec bash"
    fi
    ld_start "$ph"; rc=$?
    [ $rc = 0 ] && [ "${DRY_RUN:-0}" != 1 ] && echo "started: tmux attach -t r8ld   (bash scripts/run_r8.sh ld-status)"
    exit $rc;;
  _ld_lane) ld_lane "$1" "$2" "$3";;
  ld-plan) LDQ plan --phase "${1:-all}";;
  ld-status) LDQ status;;
  ld-go-full) touch "$Q/ld_full_go"; echo "low-delay full runs authorized (readiness evidence still required: $LD_READY_JSON)";;
  ld-decide)
    stop=$(LDQ decide "$1" "$2") || exit $?
    for n in $stop; do
      mkdir -p "$RUNS_DIR/$n"; echo "$1=$2" > "$RUNS_DIR/$n/STOPPED"
      if [ -f "$Q/running.$n" ]; then
        pkill -TERM -P "$(cat "$Q/running.$n")" 2>/dev/null; kill -TERM "$(cat "$Q/running.$n")" 2>/dev/null; fi
      echo "stopped $n ($1=$2)"
    done;;
  start)
    ph="${1:-all}"; case "$ph" in all|pilots|full) ;; *) echo "phase must be all|pilots|full"; exit 2;; esac
    if [ "$ph" = full ] || [ "${ALLOW_PILOTS_WITHOUT_G1:-0}" != 1 ]; then
      g1_pass || { echo "REFUSED: run the G1 gate on this box first (scripts/r8_box_setup.sh does)"; exit 1; }; fi
    if [ "${DRY_RUN:-0}" = 1 ]; then
      for g in 0 1; do echo "gpu$g workers $(workers $g): $(queue_of $g "$ph")"; run_queue $g "$ph"; done; exit 0; fi
    if [ "${NO_TMUX:-0}" = 1 ]; then
      for g in 0 1; do nohup bash "$0" _queue $g "$ph" >> "$Q/gpu$g.out" 2>&1 & done; echo "queues started (no tmux)"; exit 0; fi
    command -v tmux >/dev/null || { echo "tmux missing (apt-get install -y tmux) or set NO_TMUX=1"; exit 1; }
    tmux has-session -t r8 2>/dev/null && { echo "tmux session r8 already exists: tmux attach -t r8"; exit 1; }
    tmux new-session -d -s r8 -n gpu0 "bash scripts/run_r8.sh _queue 0 $ph; exec bash"
    tmux new-window -t r8 -n gpu1 "bash scripts/run_r8.sh _queue 1 $ph; exec bash"
    echo "started: tmux attach -t r8   (status: bash scripts/run_r8.sh status)";;
  _queue) run_queue "$1" "$2";;
  workers) for g in 0 1; do echo "gpu$g workers $(workers $g)"; done;;
  go-full) touch "$Q/full_go"; echo "full runs released";;
  status)
    for g in 0 1; do
      echo "== gpu$g (workers $(workers $g))"
      for n in $(queue_of $g all); do
        s=$(state_of "$n"); last=""
        [ -f "$Q/logs/$n.log" ] && last=$(grep -E '^epoch [0-9]+ step' "$Q/logs/$n.log" | tail -1)
        printf '  %-8s %-20s %s\n' "$s" "$n" "$last"
      done
    done
    g1_pass || true;;
  next)
    for g in ${1:-0 1}; do
      for n in $(queue_of $g all); do s=$(state_of "$n")
        if [ "$s" = PENDING ] || [ "$s" = RUNNING ] || [ "$s" = FAILED ]; then echo "gpu$g $s $n $(cfg_of "$n")"; break; fi
      done
    done;;
  *) sed -n '2,9p' "$0"; exit 2;;
esac
