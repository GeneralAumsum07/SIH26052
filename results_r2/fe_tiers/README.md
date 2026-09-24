# VaaniFE tiers: untrained cost and G2 graph-gate results

Plan 11.3 / 11.6 ("Orin-tier projection work") and 11.8 G2. Every tier here is exported with
**random (seeded) weights**: these rows measure graph structure and cost, never quality.

| Tier | Status | C1/C2/F/K/L | Params (deploy / training form) | MMAC/s | FP32 state | Folded nodes | Layout share | ORT vs torch max abs | G2 |
|---|---|---|---|---|---|---|---|---|---|
| mini | **to be trained in r8** | 32/24/16/2/1 | 29,274 / 29,914 | 69.76 | 3,072 B | 116 | 0.259 | 1.6e-8 | pass |
| mid | projection, untrained | 48/40/32/3/2 | 107,610 / 109,146 | 320.384 | 15,360 B | 158 | 0.260 | 5.6e-9 | pass |
| large | projection, untrained | 80/64/48/4/2 | 322,194 / 324,754 | 1,156.608 | 49,152 B | 193 | 0.264 | 1.3e-8 | pass |
| large_plus | projection, untrained | 96/72/48/4/3 | 500,042 / 504,266 | 1,827.84 | 55,296 B | 200 | 0.260 | 4.5e-8 | pass |
| *r7 deploy/r7/cascade.onnx (comparison)* | trained, shipping | GTCRN cascade | - | - | - | 539 | 0.586 | not run | **fail** |

Source: `tiers.json` (this directory), produced by

    .venv/Scripts/python.exe scripts/fe_tiers.py --seed 0 --hops 500

The ONNX files go to `runs/fe_tiers/` (git-ignored); their sha256 are in `tiers.json`.

- **Params:** "deploy" is the exported step graph (BatchNorm folded into the 1x1 convs) and
  reproduces the prototype counts exactly; "training form" counts the unfolded BN affine and
  running-stat entries (`norm: bn`, the default). The Mini is under the spec budget of 60,000
  entries and 90.706 MMAC/s in both forms (asserted in `tests/test_vaani_fe.py`).
- **MMAC/s:** `vaani.models.vaani_fe.count_macs` (dense Conv/ConvTranspose/Linear/GRU matrices plus
  attention QK^T and AV) at 62.5 hops/s (16 kHz, hop 256).
- **State:** the only state is the time GRUs' hidden state, K x F x C2 FP32 (`df_taps: 0`).
- **G2 limits** (`scripts/graph_gate.py`): folded (ORT basic level) node count < 250;
  Loop/Scan/If/GRU/LSTM/RNN = 0; ScatterND = 0; 0 symbolic dims; 0 Shape/Range nodes;
  layout ops (Transpose, Reshape, Squeeze, Unsqueeze, Flatten, Expand, Concat, Slice, Gather,
  Split) < 30% of nodes; ORT-vs-torch FP32 parity <= 1e-5 over a seeded corpus of carried-state
  hops; streaming step == offline forward <= 1e-5. Per-op counts are in `tiers.json`.
- **r7 fails G2 as expected:** 14 GRU nodes, 18 ScatterND, 539 folded nodes, layout share 0.586
  (`failed: layout_share, loops, nodes, scatternd`). r7 stays the shipping control and fallback;
  this row only documents why a new graph family was needed for Orin backends.

## Timing (non-reportable)

`ort_cpu_1t_mean_ms` / `ort_cpu_1t_p99_ms` in `tiers.json` (mini 0.39 / 1.10 ms, mid 1.03 / 13.8,
large 1.57 / 4.46, large_plus 2.10 / 3.23) were taken on the laptop, ORT CPU EP, 1 thread, while
about nine other jobs shared the machine. They are **smoke numbers, not reportable** (the mid p99
is visibly contention noise). The reportable laptop numbers are the idle-machine measurements in
plan 11.3 (Mini 0.29 / 0.64 ms, Mid 0.44 / 0.62 ms, Large 1.05 / 1.61 ms) until this script is
re-run on an idle machine. None of these are Orin or Pi numbers.

TBD: Orin (TensorRT / CUDA EP) and Pi 5 timings for each tier; they need the boards.
