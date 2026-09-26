#!/usr/bin/env bash
# r8 schedule on a 2-GPU box (plan 11.6, configs/retraining/R8_RUNBOOK.md): one resumable queue per GPU, in tmux.
#   bash scripts/run_r8.sh start [all|pilots|full]   launch both queues in tmux session "r8" (default: all)
#   bash scripts/run_r8.sh status                    every queued run: DONE / RUNNING / FAILED / DROPPED / PENDING
#   bash scripts/run_r8.sh next [0|1]                the next run each queue would start
#   bash scripts/run_r8.sh workers                   loader workers per queue (after the memory cap)
#   bash scripts/run_r8.sh go-full                   release the full runs after the pilots (winning settings copied)
#   bash scripts/run_r8.sh _queue <gpu> <phase>      the queue itself (what tmux runs)
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
# priority order (plan 11.6: 1, 3b, 2, 3, 4, 6); ablation 2 leads with pr_nhat, the NLMS read Rachit asked for
PILOTS_0=(ab1_fe_mini_s0 ab1_fe_mini_s1 ab2_pr_nhat_s0 ab2_pr_nhat_s1 ab2_p_s0 ab2_p_s1 ab2_pr_pld_s0 ab2_pr_pld_s1
          ab4_bounded_df0 ab4_unbounded_df3 ab4_bounded_df3)
PILOTS_1=(ab1_refvalid_s0 ab1_refvalid_s1 ab3b_tail00 ab3b_tail10 ab3_refdrop00_s0 ab3_refdrop00_s1
          ab3_refdrop30_s0 ab3_refdrop30_s1 ab6_kappa1 ab6_kappa2 ab6_kappa4)
# opt-in bank arm (gen_r8_configs.py --bank-arm): queued last beside its baseline ab1_fe_mini_s0 only once generated
[ -f "$A/ab7_bank_r3.yaml" ] && PILOTS_0+=(ab7_bank_r3)
FULL_0=(r8_fe_mini); FULL_1=(r8_refvalid_v2)

PY="${PY:-}"
[ -n "$PY" ] || for c in .venv/bin/python .venv/Scripts/python.exe python3 python; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi; done

cfg_of() { case "$1" in r8_*) echo "configs/retraining/$1.yaml";; *) echo "$A/$1.yaml";; esac; }
queue_of() {  # $1 gpu, $2 phase -> names
  local g=$1 ph=$2; local -n p="PILOTS_$g" f="FULL_$g"
  case "$ph" in pilots) echo "${p[@]}";; full) echo "${f[@]}";; all) echo "${p[@]} ${f[@]}";; esac
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
      if [ $rc = 0 ] && grep -q '"end": ' "$RUNS_DIR/$n/run.json" 2>/dev/null; then
        mkdir -p "$RUNS_DIR/$n"; date '+%F %T' > "$RUNS_DIR/$n/DONE"; log "$g" "DONE $n"; break; fi
      log "$g" "$n exited rc=$rc (log $Q/logs/$n.log); relaunching resumes from last.pt"; sleep 30
    done
    [ -f "$RUNS_DIR/$n/DONE" ] || { mkdir -p "$RUNS_DIR/$n"; echo "rc=$rc after $tries tries" > "$RUNS_DIR/$n/FAILED"; log "$g" "FAILED $n"; }
  done
  log "$g" "queue $ph finished"
}

cmd="${1:-status}"; shift || true
mkdir -p "$Q"
case "$cmd" in
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
