# results_r2/generalisation

`PROTOCOL.md` is the pre-registration and is not edited after scoring.

| file | what | produced by (repo root) |
|---|---|---|
| `raw_gen.csv` | raw passthrough (no enhancement) on eval_gen test; the paired-vs-raw baseline in `per_grid.md` | `CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system raw --split test --eval-root data/eval_gen --out results_r2/generalisation/raw_gen.csv --workers 1` (520 rows) |
| `per_grid.md` | r7 cascade on eval_gen split by grid (stationary = registered, changing = amendment), paired by bucket vs raw, per-bucket gap to eval_r2 | `uv run python scripts/r7_breakdown.py` |

r7 cascade CSV: `results_r2/r7/r7_e256_wr64_cascade_gen.csv`.

Headline (from `per_grid.md`, nominal 0/5/10 dB unclipped): stationary (registered) n=102 SNR_out 14.295 /
STOI 0.932 / PESQ-WB 2.490, all-three pass 40.2%; changing (amendment) n=99 18.370 / 0.970 / 3.153, 70.7%;
pooled n=201 16.302 / 0.951 / 2.817. The registered stationary grid misses the 15 dB and 2.5 targets on its own.
