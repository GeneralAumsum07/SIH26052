# results_r2/r7

r7 is the shipping control: backbone `results_r2/runs/r7_e256_wr64/best.pt`, cascade
`results_r2/runs/r7_e256_wr64_refiner/best.pt`, deployed as `deploy/r7/cascade.onnx`.

| file | what | produced by (repo root) |
|---|---|---|
| `r7_e256_wr64_cascade_eval_r2.csv`, `r7_e256_wr64_eval_r2.csv` | per-clip scores, eval_r2 test (2280 clips) | `vaani.eval` (see the matching `eval_*.log`) |
| `r7_e256_wr64_cascade_gen.csv`, `r7_e256_wr64_gen.csv` | per-clip scores, eval_gen test | `vaani.eval` (see the matching `eval_*.log`) |
| `raw_eval_r2_relabel.csv` | raw passthrough (no enhancement) on the same render as the r7 eval_r2 CSVs; the paired-vs-raw baseline in `breakdown.md` | `CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system raw --split test --eval-root data/eval_r2_relabel --out results_r2/r7/raw_eval_r2_relabel.csv --workers 0` |
| `breakdown.md` | headline rows with row and scene-clustered CIs, per-clip all-three pass rate, class x input-SNR table, fault buckets, paired delta vs raw | `uv run python scripts/r7_breakdown.py` |
| `diag/` | conditioning ablation and mask-phase probe on VAL | see `diag/README.md` |

`raw_eval_r2_relabel.csv`: 2280 rows, no NaN metric, bucket/id/snr_in/noise_class identical to the r7
cascade CSV. The scoring process segfaulted twice at clip 2061 of 2280 (no Python traceback) and completed
on the third identical attempt, so the crash is not deterministic in the clip. `results_r2/raw.csv` is the
earlier render and is not paired with the current r7 CSVs.

Headline (from `breakdown.md`): nominal n=617 SNR_out 14.864 / STOI 0.917 / PESQ-WB 2.462; all-three pass
35.5% nominal, 24.1% full split, 3.8% transients.
