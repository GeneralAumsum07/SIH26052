#!/usr/bin/env bash
# Clauses 14b/14c of SIH26052: quantization and pruning, measured on the frozen round-2 test
# split rather than asserted. Produces results_r2/optim/ and deploy/tier46/int8*.json.
#
# Sequential by design: every number here is a single-core CPU latency or a CPU-bound eval, so
# running two stages at once would make each of them slower and the comparison meaningless.
# Roughly 15 min per eval on the dev laptop (2280 items, 8 workers), so budget about 1.5 h.
#
# ASR is deliberately off. WER answers "is the speech still intelligible to a recogniser", which
# is a system-level question; the optimization question is the size/latency/quality trade, and
# --asr would triple the wall clock for a column nothing here reads.
set -eu
cd "$(dirname "$0")/.."

CKPT=${CKPT:-results_r2/runs/vaani_tier46_refiner/best.pt}
FP32=${FP32:-deploy/tier46/cascade.onnx}
OUT=results_r2/optim
mkdir -p "$OUT"

[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT (see deploy/CONTRACT.md for provenance)"; exit 1; }
[ -f data/eval_r2/test/EVALSET_HASH ] || { echo "render data/eval_r2 first: scripts/render_all_eval_sets.sh"; exit 1; }

eval_system () {  # $1 = system spec, $2 = output basename
  [ -f "$OUT/$2.csv" ] && { echo "skip $2 (csv exists)"; return; }
  uv run python -m vaani.eval --system "$1" --split test --eval-root data/eval_r2 \
    --workers "${WORKERS:-8}" --dnsmos --out "$OUT/$2.csv" > "$OUT/eval_$2.log" 2>&1
}

echo "== 14b: INT8 dynamic quantization =="
# Per-tensor and per-channel weight scales are both reported: per-channel is the accuracy-first
# default in most guides, and showing it did not rescue the result forecloses the obvious question.
uv run python -m vaani.quantize "$FP32" --ckpt "$CKPT" --repeats 5 \
  --out deploy/tier46/cascade.int8.onnx --report-json deploy/tier46/int8_report.json > /dev/null
uv run python -m vaani.quantize "$FP32" --ckpt "$CKPT" --repeats 5 --per-channel \
  --out deploy/tier46/cascade.int8_pc.onnx --report-json deploy/tier46/int8_perchannel_report.json > /dev/null

# The FP32 graph is evaluated too, and is not redundant with the checkpoint row in the matrix:
# it isolates the quantization delta from any ONNX-vs-PyTorch difference. (Measured at 1.3e-8,
# i.e. none -- but that is a result, not an assumption.)
eval_system "onnx:$FP32@$CKPT" onnx_cascade
eval_system "onnx:deploy/tier46/cascade.int8.onnx@$CKPT" onnx_cascade_int8

echo "== 14c: global magnitude pruning sweep =="
uv run python -m vaani.prune "$CKPT" --sparsity 0.0 0.1 0.2 0.3 0.4 0.5 \
  --out-dir runs/prune --report-json "$OUT/prune_sparsity.json"
# p00 is verified byte-identical in output to the source checkpoint, so the 0 % row of the sweep
# is results_r2/vaani_tier46_refiner.csv and is not re-evaluated.
for p in p10 p20 p30 p40 p50; do
  eval_system "cascade:runs/prune/$p/best.pt" "prune_$p"
done

echo "== tables =="
uv run python scripts/optimization_report.py --out "$OUT/optimization.md"
echo "optimization done $(date)" > "$OUT/DONE"
