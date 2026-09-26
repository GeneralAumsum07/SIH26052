#!/usr/bin/env bash
# Bootstrap the rented 2-GPU box for r8 (configs/retraining/R8_RUNBOOK.md) and, with --launch, start the queues the
# moment every input is present and the G1 gate has passed here. Run from the repo checkout, inside tmux:
#   bash scripts/r8_box_setup.sh --env-check      # which credentials are missing; nothing else (seconds)
#   bash scripts/r8_box_setup.sh --dry-run        # env check + every stage it would run; downloads/installs nothing
#   bash scripts/r8_box_setup.sh [--launch]       # the bootstrap; --launch ends with `scripts/run_r8.sh start all`
# Idempotent and resumable: each finished stage leaves runs/box_setup/<stage>.ok, background stages keep a pid file and
# are joined, never relaunched, while alive. Rerun the same line after a dropped session. Log: runs/box_setup.log.
# Credentials come only from the environment (names in scripts/r8_preflight.py BOX_ENV and configs/data/
# r8_datasets.yaml `credentials`); values are never printed or written.
# Env: GPUS (2), FETCH_PARALLEL (8), G1_SEED (202: the laptop's one-shot confirmation seed, so the box number is
# directly comparable), G1_ITEMS (200), SKIP_TESTS=1, SKIP_PACK=1, SKIP_BENCH=1, PREFLIGHT_SMOKE=N (N train steps
# per config in preflight), ALLOW_MISSING_ENV=1 (continue past missing credentials), PY (python for the env check).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=run; LAUNCH=0
for a in "$@"; do case "$a" in
  --env-check) MODE=env;; --dry-run) MODE=dry;; --launch) LAUNCH=1;;
  *) sed -n '2,15p' "$0"; exit 2;; esac; done
S=runs/box_setup; mkdir -p "$S"
GPUS="${GPUS:-2}"; FETCH_PARALLEL="${FETCH_PARALLEL:-8}"; G1_SEED="${G1_SEED:-202}"; G1_ITEMS="${G1_ITEMS:-200}"
G1_JSON=results_r2/r8/data_gates/box_g1/v2.json
VAL=data/eval_r2/val; VAL_HASH=b5f7a4d43bee
MIRROR_DL=data/mirror_dl
[ "$MODE" = run ] && exec > >(tee -a runs/box_setup.log) 2>&1
say() { echo "$(date '+%F %T') [box] $*"; }
die() { say "ERROR: $*"; exit 1; }
done_() { [ -f "$S/$1.ok" ]; }
mark() { date '+%F %T' > "$S/$1.ok"; }
# run a stage once: stage <name> <command...>; in --dry-run only print it
stage() {
  local n=$1; shift
  if done_ "$n"; then say "skip $n (done $(cat "$S/$n.ok"))"; return 0; fi
  if [ "$MODE" = dry ]; then say "DRY $n: $*"; return 0; fi
  say "stage $n"; "$@"; mark "$n"
}
# background stage: bg <name> <function>; joined later with join_bg <name>
bg() {
  local n=$1 f=$2
  if done_ "$n"; then say "skip $n (done)"; return 0; fi
  if [ -f "$S/$n.pid" ] && kill -0 "$(cat "$S/$n.pid")" 2>/dev/null; then say "$n already running (pid $(cat "$S/$n.pid"))"; return 0; fi
  if [ "$MODE" = dry ]; then say "DRY $n (background): $f"; return 0; fi
  say "start $n in the background (log $S/$n.log)"; rm -f "$S/$n.failed"
  ( "$f" > "$S/$n.log" 2>&1 && mark "$n" || { echo "rc=$?" > "$S/$n.failed"; exit 1; } ) &
  echo $! > "$S/$n.pid"
}
join_bg() {  # $1 name, [$2 marker to wait for instead of <name>.ok]
  local n=$1 m=${2:-$1}
  [ "$MODE" = dry ] && return 0
  while ! done_ "$m"; do
    [ -f "$S/$n.failed" ] && { tail -30 "$S/$n.log"; die "$n failed ($(cat "$S/$n.failed")); fix and rerun"; }
    if ! { [ -f "$S/$n.pid" ] && kill -0 "$(cat "$S/$n.pid")" 2>/dev/null; } && ! done_ "$m"; then
      # a previous invocation's job died with its shell, or never started: the caller relaunches it
      return 1; fi
    sleep 20
  done
}

# --- 0. what the box is (recorded every run; the runbook's sizing rule reads nproc and RAM) ----------------------------
specs() {
  say "nproc $(nproc)  pids.max $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo n/a)"
  free -g 2>/dev/null | sed -n '1,2p' || true; df -h . | tail -1
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || say "WARN: no nvidia-smi"
}
if [ "$MODE" = run ]; then specs | tee -a "$S/specs.txt"; elif [ "$MODE" = dry ]; then say "DRY specs: nproc, free, df, nvidia-smi"; fi

# --- 1. tooling: seconds; everything after needs uv --------------------------------------------------------------------
tools() {
  local c miss=0 pk="aria2 tmux rsync zip lbzip2 libsndfile1 build-essential"
  for c in aria2c tmux rsync zip lbzip2 cc; do command -v $c >/dev/null || miss=1; done
  if [ $miss = 1 ]; then
    [ "$(id -u)" = 0 ] || die "tools missing and not root: apt-get install -y $pk"
    apt-get update -qq && apt-get install -y -qq $pk > /dev/null
  fi   # build-essential: pesq builds from sdist; zip: FSD50K's split-zip join; lbzip2: the DNS .tar.bz2 shards
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
}
[ "$MODE" = run ] && { tools; export PATH="$HOME/.local/bin:$PATH"; }

# --- 2. credentials, before anything long -----------------------------------------------------------------------------
envcheck() {
  local py="${PY:-}"
  if [ -z "$py" ] && [ -x .venv/bin/python ]; then py=.venv/bin/python; fi
  if [ -n "$py" ]; then "$py" scripts/r8_preflight.py --env-check
  else uv run --no-project --quiet --python 3.12 --with pyyaml python scripts/r8_preflight.py --env-check; fi
}
set +e; envcheck; ENV_RC=$?; set -e
if [ "$MODE" = env ]; then exit $ENV_RC; fi
if [ $ENV_RC != 0 ] && [ "$MODE" = run ] && [ "${ALLOW_MISSING_ENV:-0}" != 1 ]; then
  die "credentials above are MISSING: export them (runbook 'Before renting') or rerun with ALLOW_MISSING_ENV=1"; fi

# --- 3. environment: uv sync from the lock (torch cu128 from download.pytorch.org, numba, torch-pesq, faster-whisper) --
sync() { uv sync --all-extras --frozen --python 3.12; }
if [ "$MODE" = dry ]; then say "DRY sync: uv sync --all-extras --frozen --python 3.12"; else say "uv sync"; sync; fi
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY="${PY:-python3}"; fi   # dry runs off the box keep PY
HF=.venv/bin/hf
gpucheck() {   # a matmul per device: an sm_120 card on a too-old driver or torch fails here, not 20 minutes into a run
  "$PY" - "$GPUS" <<'EOF'
import sys, torch
need = int(sys.argv[1]); n = torch.cuda.device_count()
print(f"torch {torch.__version__} cuda {torch.version.cuda}: {n} device(s)")
for i in range(n):
    x = torch.randn(512, 512, device=f"cuda:{i}"); float((x @ x).sum())
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)} sm_{''.join(map(str, torch.cuda.get_device_capability(i)))} ok")
import numba, torch_pesq   # noqa: F401 - hard requirements of the r8 runs
sys.exit(0 if n >= need else f"need {need} GPUs, torch sees {n}")
EOF
}
if [ "$MODE" = dry ]; then say "DRY gpucheck: $GPUS devices + numba + torch_pesq"; else gpucheck; fi

# --- 4. RIR banks by sha256 from the release (background; never regenerated: a rebuilt bank is a different file) -----
banks() {
  "$PY" scripts/r8_preflight.py --bank-plan > "$S/bank_plan.txt"
  local f h a need
  while read -r f h a need; do
    if [ -f "$f" ] && echo "$h  $f" | sha256sum -c --quiet - 2>/dev/null; then echo "ok   $f"; continue; fi
    if [ -z "${RIR_BANK_URL:-}" ]; then
      [ "$need" = need ] && { echo "FAIL $f: RIR_BANK_URL unset and the configs train on it"; return 1; }
      echo "WARN $f: RIR_BANK_URL unset (not trained on)"; continue; fi
    rm -f "$f"
    if aria2c -c -x8 -s8 -k 10M --max-tries=5 --console-log-level=warn -d "$(dirname "$f")" -o "$(basename "$f")" "$RIR_BANK_URL/$a" \
        && echo "$h  $f" | sha256sum -c -; then echo "ok   $f (fetched)"; continue; fi
    rm -f "$f"
    case "$f" in *.npy) echo "WARN $f: not fetched; RirBank rebuilds sidecars on first load (preflight checks the hash)";;
      *) [ "$need" = need ] && { echo "FAIL $f"; return 1; }; echo "WARN $f: not fetched (not trained on)";; esac
  done < "$S/bank_plan.txt"
}
if [ ! -f configs/data/r8_banks.json ]; then
  say "WARNING WARNING: configs/data/r8_banks.json is missing - fetching only bank_r3 by the remote_setup.sh hash (bank_r8, which the r8 configs train, has no hash then: preflight FAILs)"; fi
bg banks banks

# --- 5. laptop-only artefacts from the private HF mirror (scripts/r8_mirror_stage.sh) ----------------------------------
mirror_install() {   # idempotent; rerun after the scan so a scanner cannot leave its own mad_v2 in place
  [ -f "$MIRROR_DL/SHA256SUMS" ] || return 0
  if [ ! -f "$VAL/EVALSET_HASH" ]; then tar -C data -xf "$MIRROR_DL/eval_r2_val.tar" || return 1; fi
  mkdir -p data/manifests data/mirror/manifests_laptop
  cp -p "$MIRROR_DL"/manifests/*.parquet data/manifests/ || return 1
  [ -d "$MIRROR_DL/manifests_laptop" ] && cp -p "$MIRROR_DL"/manifests_laptop/*.parquet data/mirror/manifests_laptop/ || true
  [ -f "$MIRROR_DL/field/abcd.wav" ] && mkdir -p data/field && cp -p "$MIRROR_DL/field/abcd.wav" data/field/ || true   # G4 --web-wav
}
mirror() {
  [ -n "${MIRROR_HF_REPO:-}" ] || { echo "MIRROR_HF_REPO unset: no val set, no mad_v2"; return 1; }
  "$HF" download "$MIRROR_HF_REPO" --repo-type dataset --local-dir "$MIRROR_DL" || return 1   # HF_TOKEN from the env
  ( cd "$MIRROR_DL" && sha256sum -c --quiet SHA256SUMS ) || { echo "mirror SHA256SUMS mismatch"; return 1; }
  mirror_install || return 1
  "$PY" scripts/verify_eval_set.py "$VAL" "$VAL_HASH"
}
bg mirror mirror

# --- 6. datasets straight onto the box, the first queued jobs' (and G1's) first ----------------------------------------
for f in scripts/r8_datasets.py configs/data/r8_datasets.yaml; do
  [ -f $f ] || { [ "$MODE" = dry ] && say "DRY WARN: $f missing (the dataset stage would stop here)"     || die "$f is missing (the dataset fetcher and its table): pull the latest main"; }; done
ORDER=""; [ -f configs/data/r8_datasets.yaml ] && ORDER=$("$PY" scripts/r8_preflight.py --fetch-order)
FIRST=$(echo "$ORDER" | sed -n 's/^FIRST=//p'); REST=$(echo "$ORDER" | sed -n 's/^REST=//p')
say "datasets first: ${FIRST:-<none>}"; say "datasets after: ${REST:-<none>}"
datasets() {
  local D="$PY scripts/r8_datasets.py"
  if [ -n "$FIRST" ] && [ ! -f "$S/datasets_first.ok" ]; then
    $D fetch --only "$FIRST" --parallel "$FETCH_PARALLEL" && $D scan --only "$FIRST" && $D verify --only "$FIRST" || return 1
    date '+%F %T' > "$S/datasets_first.ok"; fi
  if [ -n "$REST" ]; then
    $D fetch --only "$REST" --parallel "$FETCH_PARALLEL" && $D scan --only "$REST" && $D verify --only "$REST" || return 1; fi
}
bg datasets datasets

# --- 7. join what the gate and the first jobs need ---------------------------------------------------------------------
for j in banks mirror; do
  until join_bg $j; do say "$j was not running; relaunching"; bg $j $j; done; done
[ -n "$FIRST" ] && until join_bg datasets datasets_first; do say "datasets was not running; relaunching"; bg datasets datasets; done
[ "$MODE" = run ] && mirror_install   # after the scan: the laptop's VAD-filtered MAD list wins
# one process writes any missing .npy sidecars, before G1 and the loaders each decompress a ~2.6 GB copy at once
sidecars() {
  awk '$4 == "need" && $1 ~ /\.npz$/ {print $1}' "$S/bank_plan.txt" | while read -r f; do
    "$PY" -c "import sys; from vaani.data.rirs import RirBank; b = RirBank(sys.argv[1]); print('sidecars ok', sys.argv[1], len(b.rt60))" "$f" || return 1
  done
}
stage sidecars sidecars

# --- 7b. hold out the Freesound siblings of r8 test noise; FSD50K is scanned only here, so the file can change here ----
heldout() {
  "$PY" scripts/heldout_freesound.py --check && return 0
  "$PY" scripts/heldout_freesound.py && "$PY" scripts/heldout_freesound.py --check || return 1
  cp -p configs/data/r8_heldout_exclude.json "$S/r8_heldout_exclude.box.json"
  say "configs/data/r8_heldout_exclude.json changed on this box: copy $S/r8_heldout_exclude.box.json back and commit it"
}
stage heldout heldout

# --- 8. tests that guard silent data corruption, then the packed corpus (speed only; verified bit-identical) ----------
tests() { VAANI_R8_BOX=1 "$PY" -m pytest -q -p no:cacheprovider tests/test_r8_box.py tests/test_losses.py tests/test_pack.py \
  tests/test_mixer_v2.py tests/test_data_gates.py tests/test_golden_vectors.py tests/test_heldout_disjoint.py \
  tests/test_heldout_freesound.py tests/test_r8_configs.py tests/test_scenes_r8.py tests/test_dropout_parity.py; }
[ "${SKIP_TESTS:-0}" = 1 ] || stage tests tests
pack() { "$PY" scripts/pack_corpus.py --manifests 'data/manifests/*.parquet' --out data/pack; }
[ "${SKIP_PACK:-0}" = 1 ] || stage pack pack

# --- 9. G1 on this box, on the full configs' own mix.v2 block and bank: no full run starts without it ------------------
g1() {
  local cmd; cmd=$(PY="$PY" "$PY" scripts/r8_preflight.py --g1-cmd --g1-seed "$G1_SEED" --g1-items "$G1_ITEMS" --g1 "$G1_JSON")
  echo "$cmd" > "$S/g1_cmd.txt"; say "G1: $cmd"
  eval "$cmd"
  "$PY" -c "import json,sys; j=json.load(open('$G1_JSON')); print({k: (j[k] or {}).get('auc_ild') for k in ('param','room')}, 'gate_pass', j['gate_pass']); sys.exit(0 if j['gate_pass'] is True else 1)" \
    || die "G1 FAILED on this box ($G1_JSON): the full runs will refuse to start. Do not override; report it."
}
stage g1 g1

# --- 10. loader/step bench (sizes VAANI_WORKERS), then preflight ------------------------------------------------------
kids() {  # every descendant pid of $1 (breadth first)
  ps -e -o pid=,ppid= | awk -v r="$1" '{ c[$2] = c[$2] " " $1 }
    END { q = r; while (q != "") { n = split(q, a, " "); q = ""; for (i = 1; i <= n; i++) if (a[i] in c) { printf "%s", c[a[i]]; q = q c[a[i]] } } }'
}
memwatch() {  # $1 pid, $2 log: every 5 s MemAvailable, and Rss / Pss / private (USS) of $1 and its descendants
  local root=$1 log=$2 p t
  echo "# root $root; lines: <t> mem <MemAvailable kB> | <t> proc <pid> <ppid> <rss kB> <pss kB> <private kB> <comm>" > "$log"
  while kill -0 "$root" 2>/dev/null; do
    t=$(date +%s); echo "$t mem $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" >> "$log"
    for p in $root $(kids "$root"); do
      awk -v t="$t" -v p="$p" -v pp="$(awk '{print $4}' /proc/$p/stat 2>/dev/null)" -v c="$(cat /proc/$p/comm 2>/dev/null)" \
        '/^Rss:/ {r=$2} /^Pss:/ {s=$2} /^Private_(Clean|Dirty):/ {v+=$2}
         END { if (r != "") print t, "proc", p, pp, r, s + 0, v + 0, c }' /proc/$p/smaps_rollup 2>/dev/null >> "$log"
    done
    sleep 5
  done
}
watched() {  # watched <log> <command...>: run it with a memwatch beside it; its exit code is the command's
  local log=$1 rc=0 bp mw; shift
  "$@" & bp=$!
  memwatch "$bp" "$log" & mw=$!
  wait "$bp" || rc=$?
  wait "$mw" 2>/dev/null || true
  return $rc
}
bench() {  # the memory logs size run_r8.sh's VAANI_WORKER_RSS_GB (per-worker memory) against this box's MemAvailable
  watched "$S/bench_mem_loader.log" "$PY" scripts/bench_loader.py --out results_r2/r8/loader_bench_box.json --workers 8 16 24 32 --batches 20 \
    --label "rental box $(nproc) vCPU" --configs configs/retraining/r8_fe_mini.yaml configs/retraining/r8_refvalid_v2.yaml
  CUDA_VISIBLE_DEVICES=0 watched "$S/bench_mem_step.log" "$PY" scripts/bench_loader.py --step-time --out results_r2/r8/step_time_box.json \
    --label "rental box GPU0" --configs configs/retraining/r8_fe_mini.yaml configs/retraining/r8_refvalid_v2.yaml
  "$PY" scripts/r8_preflight.py --mem-summary "$S/bench_mem_loader.log" "$S/bench_mem_step.log" --mem-out results_r2/r8/loader_mem_box.json
}
[ "${SKIP_BENCH:-0}" = 1 ] || stage bench bench
PF=("$PY" scripts/r8_preflight.py --sample 50 --gpus "$GPUS" --g1 "$G1_JSON")
[ -n "${PREFLIGHT_SMOKE:-}" ] && PF+=(--smoke "$PREFLIGHT_SMOKE")
if [ "$MODE" = dry ]; then say "DRY preflight: ${PF[*]}"; else "${PF[@]}" || die "preflight FAILED (runs/preflight.json)"; fi

# --- 11. launch ----------------------------------------------------------------------------------------------------
if [ $LAUNCH = 1 ]; then
  if [ "$MODE" = dry ]; then say "DRY launch: bash scripts/run_r8.sh start all"; else bash scripts/run_r8.sh start all; fi
else say "ready: bash scripts/run_r8.sh start all   (status: bash scripts/run_r8.sh status)"; fi
[ -n "$REST" ] && [ "$MODE" = run ] && ! done_ datasets && say "remaining datasets still arriving: tail -f $S/datasets.log"
exit 0
