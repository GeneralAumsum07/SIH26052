# Severe-bucket retraining readiness

## Scope and design

Implement the architecture and experiment infrastructure in the supplied severe-bucket
review. No training job, production evaluation sweep, or deployment is launched.
Existing checkpoint behavior and default model state dict keys must remain compatible.
Experiments are hypotheses: a plateau under cosine decay does not prove a capacity
limit, and dynamic mixtures do not remove overfitting risk.

Opt-in model knobs: `model_cfg.channels` (8/16/32), `model_cfg.noise_floor` (two
causal power-domain channels: log floor and log posterior ratio), `refiner_cfg`
(`hidden`, `past`, `scale`). Preserve the default first-stage five caches; floor
models append a noise-floor cache. Cascade adds the configured refiner cache.
Conditional refinement is a host-side policy: compute/cache c0 every frame, skip
c1/c2 on easy frames; do not trace a Python branch as a conditional ONNX graph.
Always-on ONNX remains supported and explicit. Selection/sweeps default to val.

## Task 1: Architecture and integration (primary agent)

- [ ] Add configurable first-stage width with matching streaming caches and
  strict same-width warm start; widened feature inputs are zero-initialized.
- [ ] Add causal floor features with batch/stream parity and no future leakage.
- [ ] Parameterize refiner and checkpoint config, export, cache shapes and training
  construction; preserve old checkpoints with absent refiner config.
- [ ] Implement conditional host runtime with fresh hidden history on skipped
  frames, optional residual blend, activation/cost counters, and evaluation path.
- [ ] Test default compatibility, nondefault streaming/ONNX parity, gradients,
  causality, invalid configurations, and conditional cache transitions.
- [ ] Commit architecture and related tests without attribution.

## Task 2: Training controls (independent subtask)

- [ ] Add deterministic opt-in uniform/low-triangular/stratified SNR sampling;
  preserve the exact RNG draw for the legacy uniform default.
- [ ] Make full-budget cosine schedule explicit and resume-safe; persist epoch
  validation history and optional patience stopping without claiming convergence.
- [ ] Log SNR-clamp binding fraction and allow refiner loss configuration.
- [ ] Test controls with isolated functions/synthetic tensors only; no trainer launch.
- [ ] Commit training control changes after integration review.

## Task 3: Oracle diagnostic (independent subtask)

- [ ] Extend per-bucket oracle analysis to raw, chosen trained system, IRM,
  IAM, oracle phase with system magnitude, and unrestricted oracle complex mask.
- [ ] Keep restricted ERB projection labelled diagnostic, not a universal ceiling.
- [ ] Emit item-level CSV and aggregate output; default val, explicit test opt-in,
  no silently truncated sample count, robust zero bins and identity tests.
- [ ] Commit diagnostic code/tests after review.

## Task 4: Experiment pack and verification

- [ ] Generate standalone YAMLs for 128-epoch continuation/control, floor input,
  SNR strata, w_snr 0.8/1.2, width 0.5/1/2 and refiner 16/24/32/48 x past 2/4 x
  two seeds. Width candidates start from scratch with a matched scratch control.
- [ ] Add a dry-run/preflight entry point that checks config, dependencies, hashes,
  model shapes and matrix MAC costs without training or touching run state.
- [ ] Add conditional threshold sweep and parameter/MAC quality frontier reporting.
- [ ] Document exact tomorrow commands and experiment ordering, unresolved empirical
  questions, checkpoint paths and no-training verification results.
- [ ] Run focused tests and one final independent review, fix findings, commit.

## Rulings

- The user's explicit build request authorizes implementation; additional design
  approval is unnecessary. Keep all new behavior opt-in and reversible.
- Reference-fault retraining from the previous review is a separate deferred
  direction; this scope implements the newly supplied document's features.
- No claimed clean-speech transparency or speedup until measured. Conditional
  refinement cannot infer a clean label; thresholds need validation.
