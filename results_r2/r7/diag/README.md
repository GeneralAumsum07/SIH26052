# r7 diagnostics on VAL (plan item A2)

Split: `data/eval_r2` **val**, all 1480 clips (37 buckets x 40). Selection-grade only; test is not touched.
Systems: cascade `results_r2/runs/r7_e256_wr64_refiner/best.pt`, backbone `results_r2/runs/r7_e256_wr64/best.pt`.
Numbers below are copied from the JSON files beside this README. Timings are not reported.

## Commands (repo root, CPU)

```
export CUDA_VISIBLE_DEVICES=-1
uv run --with numba python scripts/diag_conditioning.py --ckpt results_r2/runs/r7_e256_wr64_refiner/best.pt --eval-root data/eval_r2 --split val --n 0 --out results_r2/r7/diag/conditioning_cascade
uv run --with numba python scripts/diag_conditioning.py --ckpt results_r2/runs/r7_e256_wr64/best.pt --eval-root data/eval_r2 --split val --n 0 --out results_r2/r7/diag/conditioning_backbone
uv run --with numba python scripts/mask_phase_probe.py --system cascade:results_r2/runs/r7_e256_wr64_refiner/best.pt --eval-root data/eval_r2 --split val --per-bucket 0 --out results_r2/r7/diag/mask_phase_cascade
uv run --with numba python scripts/mask_phase_probe.py --system ckpt:results_r2/runs/r7_e256_wr64/best.pt --eval-root data/eval_r2 --split val --per-bucket 0 --out results_r2/r7/diag/mask_phase_backbone
```

The first conditioning run segfaulted (rc=139, no traceback) after 38 min; the script now resumes from
`<out>.partial.csv` and skips a clip that killed the previous process (`skipped_ids` in the JSON).
Cascade: 1480 clips, none skipped. Backbone: attempt 1 segfaulted (rc=139) and the resumed run skipped one
clip, `recorded_impulsive+stationary_-10` id `0009`, so n=1479. The rows written before that crash lost
their id zero-padding (fixed in f497921); `conditioning_backbone.csv` ids were re-padded to 4 digits after
the run, values untouched. The mask-phase runs did not crash (n=1480 each).

## Conditioning ablation (`conditioning_{cascade,backbone}.json`)

Paired delta = variant minus as-trained on the same clip, 1000-sample bootstrap CI over clips.
FiLM is off in r7, so the "feats zeroed" variant does not apply (film_shift_ratio = null).

| system | variant | SNR_out mean (delta [95% CI]) | STOI mean (delta [95% CI]) | PESQ-WB mean (delta [95% CI]) |
|---|---|---|---|---|
| cascade (n=1480) | as trained | 14.580 | 0.8990 | 2.400 |
| cascade (n=1480) | n_hat zeroed | 14.550 (-0.030 [-0.051, -0.008]) | 0.8986 (-0.0003 [-0.0005, -0.0002]) | 2.395 (-0.004 [-0.006, -0.002]) |
| cascade (n=1480) | ref zeroed | 7.486 (-7.094 [-7.340, -6.853]) | 0.7865 (-0.1125 [-0.1170, -0.1081]) | 1.542 (-0.858 [-0.885, -0.830]) |
| cascade (n=1480) | coh zeroed | 14.414 (-0.166 [-0.214, -0.121]) | 0.8975 (-0.0015 [-0.0018, -0.0012]) | 2.382 (-0.018 [-0.024, -0.012]) |
| backbone (n=1479) | as trained | 13.992 | 0.8912 | 2.301 |
| backbone (n=1479) | n_hat zeroed | 13.952 (-0.040 [-0.058, -0.018]) | 0.8907 (-0.0005 [-0.0006, -0.0003]) | 2.295 (-0.006 [-0.008, -0.004]) |
| backbone (n=1479) | ref zeroed | 7.603 (-6.388 [-6.640, -6.124]) | 0.7852 (-0.1060 [-0.1101, -0.1018]) | 1.550 (-0.751 [-0.778, -0.725]) |
| backbone (n=1479) | coh zeroed | 13.772 (-0.220 [-0.277, -0.162]) | 0.8896 (-0.0016 [-0.0019, -0.0014]) | 2.280 (-0.021 [-0.027, -0.015]) |

Reading: the reference-mic spectrum carries the model. Zeroing it costs about 7 dB SNR_out, 0.11 STOI and
0.86 PESQ-WB on the cascade (backbone: 6.4 dB, 0.11, 0.75). The NLMS noise estimate n_hat and the coherence feature are nearly inert.
Zeroing n_hat costs 0.03 dB SNR_out and zeroing coherence 0.17 dB. Both CIs exclude zero, but both are
two orders of magnitude smaller than the reference effect. Inferred: the dual-channel gain comes from the
reference spectrum alone, not from the NLMS path. This is the motivation for the trained reference-absent
mode and for checking reference validity in r8.

## Mask phase and oracle swaps (`mask_phase_{cascade,backbone}.json`)

`mask_phase_deg` is the mean |angle| of the applied mask on speech-dominant bins. `+oracle phase` and
`+oracle magnitude` replace one component of the model's output with the ideal one.

| system | mask phase (deg), overall | per-bucket range | model SNR_out | +oracle phase gain (dB) | +oracle magnitude gain (dB) | +oracle phase gain STOI / PESQ | +oracle magnitude gain STOI / PESQ |
|---|---|---|---|---|---|---|---|
| cascade | 21.73 | 1.45 to 37.68 | 14.580 | +2.164 | +4.176 | +0.021 / +0.269 | +0.080 / +1.493 |
| backbone | 15.26 | 0.78 to 27.29 | 13.983 | +2.289 | +3.593 | +0.022 / +0.290 | +0.083 / +1.468 |

Reading: the mask does complex work (overall above the 15 deg "magnitude masker" line; phase grows as input
SNR falls, up to 37.7 deg at impulsive+stationary -10 dB on the cascade). Magnitude error is the
larger first-order term: an oracle magnitude buys about 1.9x (cascade) and 1.6x (backbone) the SNR_out of an
oracle phase, and 5x the PESQ-WB. The refiner raises mask phase use (15.3 to 21.7 deg) and SNR_out (+0.60 dB)
while leaving oracle-phase headroom about the same (2.16 vs 2.29 dB). Inferred: r8 capacity is better spent
on magnitude estimation at low input SNR than on a phase-only branch.

Per-clip rows: `mask_phase_{cascade,backbone}.csv`, `conditioning_{cascade,backbone}.csv`.
