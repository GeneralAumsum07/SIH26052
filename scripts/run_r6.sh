#!/usr/bin/env bash
# r6 session driver: the order of work from the final-round data audit, on the training box.
#
# Ordering is not cosmetic. The protocol is registered before anything is scored; disjointness is
# proven before the generalisation set is used; the two corpus arms run separately so a null can be
# attributed. Each stage skips if its artefact exists, so an interrupted session resumes instead of
# repeating GPU hours.
#
# Downloads are NOT done here - fetch the corpora onto the box first, then run this.
#
#   bash scripts/run_r6.sh
#
# Env: OUT, WORKERS. The epoch budget lives in the configs, since the arms must not differ in it.
set -euo pipefail

OUT="${OUT:-results_r2/r6}"
WORKERS="${WORKERS:-4}"          # 8 workers twice killed a pool worker in the twin bucket; 4 is the safe default
GEN="data/eval_gen"
PROTOCOL="results_r2/generalisation/PROTOCOL.md"
mkdir -p "$OUT"

# --- preconditions: fail before spending anything -------------------------------------------------
[ -f "$PROTOCOL" ] || { echo "missing $PROTOCOL: the protocol must be registered before scoring"; exit 1; }
# git ls-files reads the INDEX, so `git add` alone would satisfy it. Registration means the protocol
# is in history BEFORE the results exist, so ask history, and record the commit so the ordering is auditable.
PROTOCOL_COMMIT="$(git log -1 --format=%H -- "$PROTOCOL")"
[ -n "$PROTOCOL_COMMIT" ] || {
  echo "$PROTOCOL is not committed. Registration only means something if its commit precedes the results."; exit 1; }
git diff --quiet HEAD -- "$PROTOCOL" || {
  echo "$PROTOCOL has uncommitted edits. Commit them before scoring, or the registered text is not the text used."; exit 1; }
printf '%s\n' "$PROTOCOL_COMMIT" > "$OUT/PROTOCOL_COMMIT"
echo "protocol registered at $PROTOCOL_COMMIT ($(git log -1 --format=%cI "$PROTOCOL_COMMIT"))"
[ -f data/eval_r2/test/EVALSET_HASH ] || { echo "data/eval_r2 missing: the comparison baseline is required"; exit 1; }

# The whole test suite runs here, not on a laptop: 35 of the test files need torch, and the only
# environment with the right torch is this box after `uv sync`. Training changes that break a test
# are far cheaper to find in this 60 s than in a 4 h run, so this is a precondition, not a courtesy.
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  echo "== test suite (the first place torch-dependent tests can run) =="
  uv run pytest -q -x --timeout=300 > "$OUT/pytest.log" 2>&1 || {
    echo "tests failed; see $OUT/pytest.log" >&2; tail -30 "$OUT/pytest.log" >&2; exit 1; }
  tail -1 "$OUT/pytest.log"
fi

echo "== scan the new corpora into manifests =="
# A full rescan re-reads and re-hashes every clip of every source and the scanners are sequential, so it
# costs about an hour of wall clock with both GPUs idle. On a box whose bootstrap already built the
# manifests there is nothing for it to find. RESCAN=0 skips it; the default is unchanged.
if [ "${RESCAN:-1}" = 1 ]; then
  uv run python scripts/fetch_data.py --config configs/data/round1.yaml 2>&1 | tail -5
else
  echo "skip rescan (RESCAN=0); manifests present: $(ls data/manifests/*.parquet | wc -l)"
fi

echo "== crest audit: do not take a corpus's label on trust (MAD taught that at 13 dB) =="
# crest_audit takes paths and prints a table; the converted 16 kHz FLACs are what training reads
for m in wham vehicle_interior; do
  d="data/raw/$m"
  [ -d "$d" ] || { echo "skip crest $m (not scanned)"; continue; }
  [ -s "$OUT/crest_$m.log" ] && { echo "skip crest $m (done)"; continue; }
  uv run python scripts/crest_audit.py "$d" --by-parent > "$OUT/crest_$m.log" 2>&1 \
    || echo "crest audit failed for $m; see $OUT/crest_$m.log"
  tail -20 "$OUT/crest_$m.log"
done

echo "== render the held-out generalisation set =="
# render_eval_sets refuses to overwrite a frozen set, so eval_r2 cannot be damaged from here
if [ -f "$GEN/test/EVALSET_HASH" ]; then
  echo "skip render (eval_gen already frozen: $(cat "$GEN/test/EVALSET_HASH"))"
elif [ -f data/manifests/vehicle_interior.parquet ]; then
  uv run python scripts/render_eval_sets.py \
    --manifests data/manifests/librispeech_100h.parquet data/manifests/cv_hi.parquet \
                data/manifests/vehicle_interior.parquet \
    --split test --out "$GEN" --per-bucket 40 > "$OUT/render_gen.log" 2>&1
else
  echo "vehicle_interior not downloaded; generalisation set not rendered"
fi

echo "== prove the held-out corpus is held out, before it is used =="
if [ -f data/manifests/vehicle_interior.parquet ]; then
  uv run python scripts/check_heldout.py --heldout data/manifests/vehicle_interior.parquet \
    --recipes configs/retraining/r6_ctl64.yaml configs/retraining/r6_demand64.yaml \
              configs/retraining/r6_wham64.yaml configs/retraining/r5_continue128.yaml
fi

# --- training: the arms differ in their data, so each needs its own loader -------------------------
# They deliberately do NOT go through vaani.train_multi: that shares one batch stream, which is only
# valid for configs whose data is identical. Feeding the arms a shared stream would make them see the
# same batches and silently void the very comparison they exist for (train_multi refuses, by design).
#
# ARMS_PARALLEL=1 (default) runs them in sequence. Set it to 3 on a box with cores to spare: total CPU
# work is the same, but the arms then finish together under the same contention, which is what a paired
# comparison wants, and one arm dying no longer hides behind another still running.
ARMS_PARALLEL="${ARMS_PARALLEL:-1}"
CORES=$(nproc)
train () {  # $1 = config name under configs/retraining
  if [ -f "runs/$1/best.pt" ]; then echo "skip train $1 (best.pt exists)"; return; fi
  echo "== train $1 =="
  # the 64-epoch budget is registered in the config itself, not passed here: the arms must not differ
  VAANI_WORKERS="${VAANI_WORKERS:-$(( (CORES - 2) / ARMS_PARALLEL ))}" \
    uv run python -m vaani.train "configs/retraining/$1.yaml" > "$OUT/train_$1.log" 2>&1
}

eval_system () {  # $1 = system spec, $2 = basename, $3 = eval root
  [ -f "$OUT/$2.csv" ] && { echo "skip eval $2 (csv exists)"; return; }
  uv run python -m vaani.eval --system "$1" --split test --eval-root "$3" \
    --workers "$WORKERS" --dnsmos --out "$OUT/$2.csv" > "$OUT/eval_$2.log" 2>&1
}

# The epoch sweep and the control are ONE shared-stream run: r6_ctl64 is the 64-epoch point, so
# 32/64/128/256 cost 256 epochs of dataloading instead of 480, the budgets are compared pairwise,
# and the corpus arms get their control for free. SWEEP=0 skips it and trains the control alone.
SWEEP="${SWEEP:-1}"
if [ "$SWEEP" = 1 ] && [ ! -f runs/r6_ctl64/best.pt ]; then
  echo "== epoch sweep + control on one shared batch stream =="
  uv run python -m vaani.train_multi \
    configs/retraining/r6_e32.yaml configs/retraining/r6_ctl64.yaml \
    configs/retraining/r6_e128.yaml configs/retraining/r6_e256.yaml \
    > "$OUT/train_sweep.log" 2>&1
fi

ARMS="r6_ctl64 r6_demand64"
[ -f data/manifests/wham.parquet ] && ARMS="$ARMS r6_wham64" || echo "skip r6_wham64 (wham not downloaded)"
if [ "$ARMS_PARALLEL" -gt 1 ]; then
  pids=()
  for n in $ARMS; do train "$n" & pids+=("$!"); done
  failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [ "$failed" = 0 ] || { echo "an arm failed; see $OUT/train_*.log" >&2; exit 1; }
else
  for n in $ARMS; do train "$n"; done
fi

echo "== score every arm on eval_r2, and on the generalisation set =="
for n in r6_ctl64 r6_demand64 r6_wham64 r6_e32 r6_e128 r6_e256; do
  [ -f "runs/$n/best.pt" ] || continue
  eval_system "ckpt:runs/$n/best.pt" "$n" data/eval_r2
  if [ -f "$GEN/test/EVALSET_HASH" ]; then eval_system "ckpt:runs/$n/best.pt" "${n}_gen" "$GEN"; fi
done

# the deployed system on the generalisation set: the headline number the protocol registers
if [ -f "$GEN/test/EVALSET_HASH" ]; then
  eval_system "cascade:results_r2/runs/vaani_tier46_refiner/best.pt" "tier46_gen" "$GEN"
fi

echo "== regenerate the licence table: WHAM! is CC BY-NC and changes the transfer story =="
uv run python scripts/licence_table.py --out docs/licences.md > /dev/null

date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/DONE"
echo "r6 complete -> $OUT"
