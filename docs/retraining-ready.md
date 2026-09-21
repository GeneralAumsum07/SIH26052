# Severe-bucket retraining pack

Prepared 21 September 2026. No training or production evaluation sweep was launched.
The 29 standalone recipes in `configs/retraining/` pass the read-only preflight on
this machine. `results_r2/retraining/preflight.json` records checkpoint hashes,
parameter counts, cache shapes, schedule budgets and matrix MAC estimates.
Repeat preflight on the training host; local readiness is not a guarantee of GPU
memory fit or throughput there.

## Tomorrow's commands

Run from the repository root with the project environment activated (on Windows,
use `.venv/Scripts/python.exe` in place of `python`). These commands are provided
for a future launch; they were **not executed** during preparation.

```powershell
python -m scripts.retraining preflight "configs/retraining/*.yaml" --out results_r2/retraining/preflight.json
python -m scripts.retraining launch configs/retraining/r5_continue128.yaml configs/retraining/r5_noise_floor.yaml
python -m scripts.retraining launch "configs/retraining/r5_refiner_*.yaml"
python -m scripts.retraining launch "configs/retraining/r5_wsnr*.yaml" "configs/retraining/r5_snr_*.yaml"
python -m scripts.retraining launch "configs/retraining/r5_width*.yaml"
```

Launch validates the entire requested batch first, then runs jobs serially with
thread caps and initializes the RIR bank before spawning data workers. No automatic
launch, scheduler, or parallel GPU allocation is installed. Model checkpoints go to
the recipe's named directory under `runs/`.

| Family | Runs | Initialization and budget |
|---|---:|---|
| Continuation control, floor features | 2 | Same pinned r4_ctl anchor; 128 additional epochs each |
| SNR weight 0.8/1.2, clamp 20, triangular/stratified draws | 5 | Same anchor, 128 additional epochs |
| First-stage channels 8/16/32, seeds 0/1 | 6 | All from scratch, 128 epochs; width16 is the matched control |
| Refiner hidden16/24/32/48 × past2/4 × seeds4606/4607 | 16 | Frozen pinned r4_ctl anchor, 32 epochs, full budget |

The continuation is a declared warm restart with a fresh optimizer and an 80,000-step
cosine budget; it is not a claim of 128 cumulative lifetime epochs. Resuming a named
run restores optimizer time and refuses a changed cosine budget or critical recipe.
To extend a finished experiment, use a new name and `init_from`.

The refiner grid deliberately pins the **current** r4_ctl checkpoint so every cell
shares the same first stage. It does not automatically switch to tomorrow's best
continuation. To repeat it on that winner, generate a separate pack with a new
`--anchor` and rename its runs before launch; preserve the old grid for comparison.
Noise-floor features add zero-initialized input slices to the same-width warm start.
Width changes cannot warm-start and are compared only to the matched scratch control.

## Validation and diagnostics

Epoch selection uses a deterministic validation screen (first four sorted items per
bucket), never test. First-stage history records STOI/SNR/PESQ, LR and loss-clamp
fraction; the refiner records its existing anchor-relative selection history plus
clamp history. Inspect full validation before choosing a winner. The grid disables
the old two-epoch refiner screen stop to give every cell the same training budget.
Optional first-stage patience is off by default. A cosine plateau does not prove a
capacity limit, and fresh mixtures do not eliminate corpus overfitting.

Oracle analysis and clamp auditing, when ready to run:

```powershell
python -m scripts.ceiling_analysis --eval-root data/eval_r2 --system ckpt:runs/vaani_full_r4_ctl/best.pt --items-out results_r2/retraining/oracle-items.csv --aggregate-out results_r2/retraining/oracle.json
python -m scripts.refiner_frontier sweep --checkpoint runs/vaani_tier46_refiner/best.pt --eval-root data/eval_r2 --thresholds 6 12 18 24 --out results_r2/retraining/conditional
```

Both default to all validation items; `--per-bucket N` explicitly labels a smaller
diagnostic subset. Test access requires `--split test --allow-test`. Oracle rows
include raw, system, IRM, IAM, clean phase with system magnitude, unrestricted
complex mask, and an ERB-projected complex-mask diagnostic. The ERB projection is
not a universal architecture ceiling. IRM/IAM are reference-informed comparisons,
not proofs of an optimal waveform-metric bound. Silent mixture bins cannot recover
nonzero clean energy through any finite multiplicative mask.

`mean_clamp_at_20` and `mean_clamp_at_30` in each oracle bucket report binding
fractions using the training loss's SNR epsilon convention. Other loss terms still
provide gradients when the absolute-SNR term saturates. These whole-clip diagnostics
complement training-crop clamp histories; they are not identical populations.

For width/refiner quality-versus-cost tables, evaluate each checkpoint on the same
validation items using `python -m vaani.eval` and then:

```powershell
python -m vaani.eval --system ckpt:runs/r5_width16_s0/best.pt --split val --eval-root data/eval_r2 --out results_r2/retraining/val-width16-s0.csv
python -m scripts.refiner_frontier summarize "results_r2/retraining/val-*.csv" --out results_r2/retraining/frontier
```

The summarizer rejects duplicate or mismatched item keys and requires one checkpoint
system per CSV. Keep all inputs on the same validation split and dataset version.
It emits nominal, severe stationary, transient and clean rows with finite metric
counts and an SNR-versus-matrix-MAC Pareto flag; STOI/PESQ remain explicit guardrails.
MACs exclude DSP, normalization and elementwise arithmetic and are **not timings**.
Conditional costs use global frame-weighted activation, not per-envelope timing.

## Conditional runtime and deployment

The host runtime always advances the cheap c0 history, even when c1/c2 are skipped,
so reactivation never reads stale context. Bypass returns the first-stage spectrum
exactly. The policy uses estimated local SNR and DSP speech presence; optionally
low reliability forces refinement. Thresholds require validation and do not promise
clean-speech transparency or a particular speedup. `crossfade` is a constant residual
blend, not temporal smoothing.

For the ordinary evaluator, write a YAML with `base_checkpoint` and `conditional`
keys (e.g. `conditional: {snr_threshold_db: 12, speech_threshold: 0.05}`), then use
`conditional:path/to/policy.yaml` as the system. ONNX export remains **always-on**;
the host's conditional control flow is not an exported ONNX branch.

New checkpoint fields: `model_cfg.channels`, `model_cfg.noise_floor`, optional floor
rates, and `refiner_cfg: {hidden, past, scale}`. Defaults preserve old checkpoint
keys and streaming behavior. Floor models append `noise_cache` after the existing
five first-stage caches; cascades append `refine_cache` last. Use the exported graph's
named shapes or preflight output rather than hard-coded legacy cache dimensions.

## Verification boundary

Final focused suite: **100 passed, 5 deselected**. Read-only pack preflight:
**29 ready, 0 blocked**. No `runs/r5*` directories were created.

Verified default/nondefault batch-stream parity, causality, zero-slice warm starts,
nondefault ONNX parity, refiner cache transitions, sampling, scheduling, clamp math,
recipe generation, preflight and diagnostic output using focused tests. Trainer-main
smoke tests were intentionally excluded to honor the no-training request. No model
quality improvement, convergence, GPU throughput or hardware speedup is claimed.
The independent agent review did not complete because of usage limits; final
integration checks were performed locally.
