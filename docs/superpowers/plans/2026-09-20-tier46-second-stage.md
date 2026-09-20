<!-- Authored by Astra (Codex, medium effort) on 2026-09-20 21:40 IST from the brief in the session scratchpad; executed by Claude. -->

# Tier 4.6 Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task-by-task. Write each task’s tests first, demonstrate the expected failure, implement, verify, then commit only that task’s files. Do not push.

**Goal:** Test whether inexpensive residual suppression or a small learned refinement stage improves the frozen e32/r4 system enough to justify deployment, while preserving intelligibility and the 16 ms processing deadline.

**Architecture:** Preserve the existing DSP front end, 512/256 STFT, and first-stage checkpoint. First test a conservative decision-directed Wiener post-filter after the complete network output, including deep-filter taps. Independently prepare a tiny causal complex residual refiner; train it only after board timing permits additional compute. Defer asymmetric windows beyond the submission.

**Tech stack:** Existing uv/PyTorch/NumPy/ONNX Runtime stack, ONNX opset 17, float32 deployment, 16 kHz audio, one ORT thread.

**References:** `docs/superpowers/plans/2026-09-20-review-fixes.md`, `docs/superpowers/specs/2026-09-18-vaani-training-stack-design.md`, `.superpowers/sdd/2026-09-18-vaani-training-stack/progress.md`, and `deploy/CONTRACT.md`.

This planning session changed no files. The working tree already contains changes to `results_r2/matrix.md` and additional result artifacts; preserve them.

## 1. Recommendation and scope

| Priority | Decision | Projected quality change relative to frozen first stage | Confirmation or kill measurement |
|---|---|---|---|
| 1 | Build one conservative DSP post-filter experiment | **Estimate:** +0.2–0.7 dB SNR_out, 0–0.08 PESQ; STOI may decline | Validation selection followed by the frozen-test gate below; stop if useful gains require excessive speech attenuation |
| 2 | Prepare, then conditionally train a tiny complex residual refiner | **Estimate:** +0.4–1.2 dB SNR_out, +0.05–0.20 PESQ; zero improvement remains plausible | Actual-board timing before training; validation screen; one frozen-test evaluation |
| 3 | Drop asymmetric windows from the September delivery | **Estimate:** no direct quality gain; a future redesign might reduce algorithmic latency, with unknown quality cost | Reconsider only after deployment is stable and measured latency is an identified problem |

These estimates are engineering hypotheses, not measured results or guarantees. They are **not additive**. This plan does not assume either experiment closes the approximately 1.1–1.3 dB / 0.2 PESQ nominal gap.

Do not initially combine the post-filter and refiner. The refiner is an alternative second stage, trained directly against the frozen first-stage output. This gives two attributable experiments and avoids another tuning dimension.

### Evidence governing these choices

- The ledger measures reference-mic importance at approximately −0.061 STOI when removed; preserve its input.
- The same diagnostics find `n_hat` and the coherence-map input nearly inert. Do not assume they provide a reliable residual-noise estimate.
- Deep-filter taps contribute approximately 0.1 dB despite increased learning rate. Another tap-order or learning-rate sweep is low priority.
- Oracle magnitude/phase replacements demonstrate headroom, not attainable gains for these implementations.
- The sampled “architecture ceiling” script is a diagnostic construction, not proof of a global mathematical ceiling.
- A real-valued post-filter preserves first-stage phase. It cannot solve the measured phase gap.
- The current board script reports model p99 plus a best-run average DSP cost. That is a timing proxy, not the p99 of the complete frame loop.

### Latency clarification

`deploy/CONTRACT.md` derives **32 ms total algorithmic latency**, comprising a 16 ms hop plus 16 ms lookahead. Do not describe the current contract as 32 ms plus another 16 ms.

Keep that framing unchanged. Both proposed second stages consume the current spectral frame before the existing synthesis operation, adding computation but no additional frame lookahead.

## 2. Frozen artifacts and acceptance policy

### Freeze the first stage

Use `runs/vaani_full_r3_e32/best.pt` as the default anchor. Do not silently substitute an in-progress r4 checkpoint.

When r4 and its same-data control finish:

1. Require complete frozen-test CSVs before describing their results.
2. Compare e32, r4, and r4 control on one identical validation set.
3. Choose a replacement anchor using validation, before Tier 4.6 parameter selection.
4. If no eligible r4 artifact is available when selection starts, retain e32 throughout this experiment.

Validation selection rule: retain checkpoints with nominal STOI no more than 0.003 below e32 and PESQ no more than 0.01 below e32; choose highest nominal SNR_out, breaking ties by PESQ, then checkpoint path. This is a declared engineering rule, not an established statistical threshold.

Record checkpoint SHA256, embedded config, git revision, validation digest, and frozen-test digest in `results_r2/tier46/anchor.json`. Copy the chosen checkpoint to `runs/tier46_anchor/best.pt`, verifying byte equality. Never overwrite that anchor during an experiment.

### Data integrity

The frozen test split is:

```text
data/eval_r2/test
2280 items
metadata digest: eda217ab2a38
nominal slice: 617 items
```

Use the existing nominal definition exactly:

```python
nominal = (
    ~df.clipped
    & ~df.ref_dropout
    & df.fault.isna()
    & df.snr_in.isin([0, 5, 10])
)
```

`fault_none` remains outside nominal because that is the current reporting convention.

The existing verifier checks metadata hashes and file presence, not WAV contents. Supplement it with a SHA256 manifest of every mix, clean, twin, and metadata file; validate WAV sample rate, channel count, matching lengths, and finite samples. Infer required twins from impulse metadata, so a bucket missing **all** twins cannot escape validation.

Do not render or modify the frozen test split.

### Selection and kill gates

Tune only on `data/eval_r2/val`, after freezing its current files and verifying that its source split is validation. If unavailable, stop selection; do not substitute test data. The historical test split has already been inspected repeatedly, so describe these as frozen benchmark results, not a newly untouched holdout.

Choose at most one post-filter setting and one refiner checkpoint on validation. Pre-register both before evaluating either on test.

For each candidate, require:

| Gate | Requirement on frozen test |
|---|---|
| Completeness | Exactly the same 2280 unique `(bucket,id)` keys as anchor; no failed/nonfinite SNR_out, STOI, or PESQ |
| Post-filter utility | Nominal ΔSNR_out ≥ +0.25 dB **and** ΔPESQ ≥ +0.02 |
| Refiner utility | Nominal ΔSNR_out ≥ +0.50 dB **and** ΔPESQ ≥ +0.05 |
| Paired uncertainty | Lower confidence bound for nominal ΔSNR_out and ΔPESQ > 0 |
| Intelligibility | Nominal STOI > 0.85; mean ΔSTOI ≥ −0.003; lower confidence bound > −0.005 |
| Per-bucket protection | Every bucket: mean ΔSTOI ≥ −0.01 and ΔPESQ ≥ −0.05 |
| Severe/burst protection | Each of the severe and `fault_burst_*` aggregates: ΔSNR_out ≥ −0.2 dB |
| Recovery | No additional unrecovered events; finite-pair median recovery increase ≤ one hop, 0.016 s |
| Deployment | Actual-board complete-loop p99 < 12 ms, plus physical runtime stability check |

Use 10,000 paired bootstrap resamples with seed 4606. Cluster by item ID, retaining all bucket variants of an ID together; report the small number of independent seed clusters. Because two candidates are allowed, use **97.5% two-sided intervals** for their primary delta gates. Keep the existing report’s ordinary 95% descriptive intervals separately labelled.

A failed gate kills that candidate for this release. Do not retune on test and try again.

“Targets met” is a separate claim requiring nominal means strictly above **15 dB / 0.85 / 2.5**. Passing the improvement gate alone is not target success. Preserve per-noise-class operating envelopes and burst results.

## 3. Task 1 — Freeze artifacts and implement the paired gate

**Files**

- Create `scripts/tier46_protocol.py`
- Create `vaani/tier46_gate.py`
- Create `tests/test_tier46_protocol.py`
- Create `tests/test_tier46_gate.py`
- Modify `vaani/report.py`
- Modify `tests/test_report.py`

**Interfaces**

```python
# scripts/tier46_protocol.py
freeze(eval_root, anchor_candidates, out_dir) -> dict

# vaani/tier46_gate.py
compare(anchor_csv, candidate_csv, protocol, kind) -> dict
# kind: "postfilter" | "refiner"
# Result contains counts, paired deltas/CIs, each gate, and overall pass.
```

- [ ] Write tests rejecting missing rows, duplicate keys, changed WAV bytes, all-missing twins, and NaN metrics.
- [ ] Write tests showing pairing is independent of CSV order and that `inf` recovery remains a failure.
- [ ] Add synthetic gate fixtures exercising equality at strict thresholds and a candidate that improves SNR but fails STOI.
- [ ] Run:

```powershell
uv run pytest tests/test_tier46_protocol.py tests/test_tier46_gate.py tests/test_report.py -q
```

Expected first result: new interfaces/tests fail.

- [ ] Implement the integrity checks and gate above. Add `--protocol` to `vaani.report` so the report includes hashes and anchor identity without changing old invocations.
- [ ] Expose these new commands:

```powershell
uv run python scripts/verify_eval_set.py data/eval_r2/test eda217ab2a38
uv run python scripts/tier46_protocol.py freeze --eval-root data/eval_r2 --anchor runs/vaani_full_r3_e32/best.pt --out results_r2/tier46
```

`freeze` must exit nonzero on a mismatch, copy the anchor without replacing an existing unequal file, and write manifests atomically.

- [ ] Rerun the focused tests and commit the six source/test files:

```powershell
git add scripts/tier46_protocol.py vaani/tier46_gate.py tests/test_tier46_protocol.py tests/test_tier46_gate.py vaani/report.py tests/test_report.py
git commit -m "test: freeze Tier 4.6 evaluation protocol and paired gates"
```

**Time estimate:** 2–3 laptop hours. No GPU.

## 4. Task 2 — Implement the conservative post-filter

**Files**

- Create `vaani/dsp/postfilter.py`
- Create `configs/exp/tier46_postfilter.yaml`
- Create `tests/test_postfilter.py`
- Create `tests/test_eval_tier46.py`
- Modify `vaani/eval.py`

### Algorithm

Use a decision-directed Wiener gain with a causal rolling-minimum residual-output PSD estimate. This is deliberately simpler than MMSE-LSA: no exponential-integral approximation or new runtime dependency.

Input \(Y_{t,f}\) is the **complete first-stage enhanced spectrum**, after the current-frame mask and past-frame deep-filter contributions. All PSD estimates refer to this output domain.

Fixed constants:

```yaml
alpha_power: 0.8
alpha_noise: 0.9
alpha_dd: 0.95
min_frames: 64
warmup_frames: 16
noise_bias: 1.5
gain_floor: 0.8
epsilon: 1.0e-10
```

Define the implementation precisely:

```python
power = abs(Y) ** 2
smooth = power if first_frame else 0.8 * smooth + 0.2 * power
ring.push(smooth)                       # 64 past/current frames, no future frames
floor_psd = minimum_over_valid_ring()
candidate_noise = 1.5 * floor_psd
noise = candidate_noise if first_frame else (
    0.9 * noise + 0.1 * candidate_noise
)
noise = maximum(noise, 1e-10)

gamma = minimum(power / noise, 1000.0)
xi = 0.95 * previous_output_power / noise + 0.05 * maximum(gamma - 1, 0)
gain = clip(xi / (1 + xi), gain_floor, 1.0)
gain = maximum(gain_floor, frequency_smooth_1_2_1(gain))
# Frequency smoothing uses replicated edge bins and weights [0.25, 0.5, 0.25].
gain = where(gain > previous_gain,
             gain,                     # Restore speech immediately.
             0.8 * previous_gain + 0.2 * gain)
if frame_index < 16:
    gain = ones_like(gain)
Z = gain * Y
previous_output_power = abs(Z) ** 2
previous_gain = gain
```

Initialize previous power to zero, previous gain to one, and valid-ring count to zero. Silence produces zero output without NaNs. Apply gains equally to real and imaginary components.

**Known limitation:** sustained speech may contaminate the minimum estimate, and changing residual noise may be tracked slowly. The high gain floor and acceptance gates address this conservatively; they do not make the estimator reliable by assumption.

### Interface and integration

```python
class ResidualPostFilter:
    def __init__(self, **config): ...
    def reset(self) -> None: ...
    def process_frame(self, spectrum: np.ndarray) -> np.ndarray: ...
    # complex64 (257,) -> complex64 (257,)
```

Add `post:<yaml-path>` parsing to `enhance_fn`. Its YAML contains `base_checkpoint` and `postfilter`.

Refactor the checkpoint closure sufficiently to obtain its output spectrum before `istft`. Apply the filter frame by frame, then perform **one** iSTFT. Do not re-run the DSP front end or analyze a synthesized intermediate waveform.

Construct/reset post-filter state inside each `f(mix)` call, including no-burst twins. Existing `ckpt:` behavior must remain unchanged.

### Tests first

```python
def test_unity_floor_is_identity():
    rng = np.random.default_rng(46)
    y = (rng.normal(size=(257, 80))
         + 1j * rng.normal(size=(257, 80))).astype(np.complex64)
    pf = ResidualPostFilter(gain_floor=1.0)
    z = np.stack([pf.process_frame(y[:, t]) for t in range(80)], axis=1)
    np.testing.assert_array_equal(z, y)
```

Also test:

- Zero input and finite outputs across startup.
- Gain bounds and phase preservation on nonzero bins.
- Prefix outputs unchanged when future frames change.
- Reset reproduces a fresh instance.
- Different clip processing orders produce identical per-clip outputs.
- Chunking into 1, 7, and 23 frames preserves the same state sequence.
- A new speech onset restores gain immediately.
- Unity-filter checkpoint waveform matches the existing path within `1e-6`.

- [ ] Run failing tests:

```powershell
uv run pytest tests/test_postfilter.py tests/test_eval_tier46.py -q
```

- [ ] Implement, rerun, then commit the five listed files.

**Time estimate:** 3–4 laptop hours. No GPU.

## 5. Task 3 — Select one post-filter on validation

**Files**

- Create `scripts/screen_tier46.py`
- Create `tests/test_screen_tier46.py`
- Modify `scripts/tier46_protocol.py`

Keep the search bounded to four combinations:

```text
gain_floor ∈ {0.70, 0.85}
noise_bias ∈ {1.0, 1.5}
```

All other constants remain fixed. These are experimental settings, not measured optima.

Cache only validation first-stage spectra for this screen, keyed by anchor hash, validation audio digest, DSP config, and STFT definition. Do not cache states across clips.

Choose the candidate with highest nominal ΔSNR_out among settings passing the post-filter utility and protection thresholds on validation; break ties by PESQ and then higher gain floor. If none passes, stop the post-filter experiment without evaluating it on test.

Tests must reject a cache from another checkpoint/split and verify deterministic selection.

- [ ] Run tests before implementation, then implement and rerun:

```powershell
uv run pytest tests/test_screen_tier46.py -q
uv run python scripts/screen_tier46.py postfilter --protocol results_r2/tier46/anchor.json --split val --out results_r2/tier46/postfilter_screen
```

The new screen command writes a locked selected config:

```text
results_r2/tier46/postfilter.selected.yaml
```

- [ ] Evaluate the selected candidate once, after preregistration:

```powershell
uv run python -m vaani.eval --system ckpt:runs/tier46_anchor/best.pt --eval-root data/eval_r2 --split test --workers 8 --out results_r2/tier46/anchor.csv
uv run python -m vaani.eval --system post:results_r2/tier46/postfilter.selected.yaml --eval-root data/eval_r2 --split test --workers 8 --out results_r2/tier46/postfilter.csv
uv run python scripts/tier46_protocol.py gate --kind postfilter --anchor results_r2/tier46/anchor.csv --candidate results_r2/tier46/postfilter.csv --protocol results_r2/tier46/anchor.json --out results_r2/tier46/postfilter_gate.json
```

The gate command exits nonzero on failure. CSV existence never implies completion.

- [ ] Commit the three source/test files.

**Time estimate:** 1–2 hours implementation; **estimate:** 1–4 hours CPU evaluation, depending on validation size and worker throughput. Measure the first 40 clips to update the estimate. Do not spend GPU time on this experiment.

## 6. Task 4 — Implement the tiny residual refiner and stream export

**Files**

- Create `vaani/models/residual_refiner.py`
- Create `vaani/models/cascade.py`
- Create `tests/test_residual_refiner.py`
- Create `tests/test_cascade.py`
- Modify `vaani/export.py`
- Modify `tests/test_export.py`
- Modify `vaani/eval.py`

### Architecture

The first stage is frozen VaaniNet. The second stage sees:

- \(Y\): first-stage enhanced complex spectrum.
- \(P\): primary spectrum actually supplied to VaaniNet, after configured limiting.
- \(R\): reference spectrum actually supplied to VaaniNet.
- \(D=P-Y\): removed complex residual, **not labelled as pure noise**.

For each bin/frame:

\[
s=\sqrt{|P|^2+|Y|^2+10^{-8}}
\]

Build eight real channels:

```text
Re(Y)/s, Im(Y)/s,
Re(D)/s, Im(D)/s,
Re(R)/s, Im(R)/s,
log(1+|P|), log(1+|Y|)
```

Clip normalized real/imaginary channels to `[-10,10]`.

Use three convolutions:

```python
h0 = relu(conv_1x1(features, in_channels=8, out_channels=16))
h1 = relu(conv_3x3(causal_pad(h0, past_frames=2, freq_bins=1),
                  in_channels=16, out_channels=16))
delta = tanh(conv_1x1(h1, in_channels=16, out_channels=2))
Z = Y + 0.25 * s * delta
```

Zero-initialize the final convolution so the initial cascade is exactly the first stage. Set the correction’s imaginary DC and Nyquist bins to zero.

The refiner has 2,498 parameters with biases. Its convolutions require approximately 0.63 million MACs per frame, derived from the specified shapes; this is **not a runtime measurement**. Report both total parameters and trainable parameters.

Do not widen GTCRN, modify vendored files, or apply the existing checkpoint twice to a distribution it was not trained to process.

### Interfaces and state

```python
class ResidualRefiner(nn.Module):
    def forward(self, primary, reference, enhanced): ...
    # Each (B,257,T,2); returns same shape.

    def step(self, primary, reference, enhanced, refine_cache): ...
    # Each spectrum (1,257,1,2)
    # refine_cache (1,16,2,257), oldest frame first
    # Returns enhanced spectrum and updated refine_cache.

class FrozenCascade(nn.Module):
    def forward(self, spec6, feats): ...
    # Frozen first-stage model followed by refiner.
```

Override `train(mode)` so the first stage always remains in `.eval()` with `requires_grad=False`; otherwise inherited BatchNorm statistics would change the allegedly frozen model.

The exported graph contains the existing streaming VaaniNet followed by the refiner:

```text
existing seven inputs + refine_cache
existing six outputs + refine_cache_out
```

Retain every existing cache name and shape. Use `convert_to_stream` for the first stage; load refiner weights directly.

Add `cascade:<checkpoint-path>` to `enhance_fn`. Embed both stages’ state dictionaries and the first-stage config/hash in the cascade checkpoint so evaluation does not depend on an external movable checkpoint.

Extend export dispatch to recognize `config["model"] == "vaani_cascade"` and produce `deploy/tier46/cascade.onnx`, leaving `deploy/model.onnx` untouched.

### Tests first

- Zero initialization reproduces the first stage.
- Refiner parameters receive finite nonzero gradients.
- First-stage parameters **and buffers** are byte-identical after two optimization steps.
- Nonzero refiner weights give batch/frame parity below `1e-4`.
- Random future frames do not change prefix outputs.
- Separate streams and reset states cannot contaminate each other.
- Silence, large finite inputs, and reference dropout remain finite.
- ONNX parity covers nonzero refiner weights and nonzero deep-filter taps; identity initialization must not conceal a broken cache.

```powershell
uv run pytest tests/test_residual_refiner.py tests/test_cascade.py tests/test_export.py tests/test_eval_tier46.py -q
```

- [ ] Write tests, observe failure, implement, rerun, and commit the seven files.

**Time estimate:** 4–6 laptop hours, including export. No training GPU yet.

## 7. Task 5 — Replace proxy timing with the complete frame measurement

**Files**

- Create `vaani/dsp/streaming.py`
- Create `vaani/stream_runtime.py`
- Create `tests/test_stream_runtime.py`
- Create `tests/test_board_timing.py`
- Modify `scripts/board_timing.py`
- Modify `tests/test_pipeline.py`

### Runtime interfaces

```python
class StreamingDSP:
    def __init__(self, controller_on, dsp_cfg): ...
    def reset(self): ...
    def push(self, stereo_hop): ...
    def flush(self, total_samples): ...
    # Produces aligned spec6/features using the checkpoint's DSP config.

class StreamEnhancer:
    def reset(self): ...
    def push(self, stereo_hop: np.ndarray) -> np.ndarray: ...
    def flush(self, total_samples: int) -> np.ndarray: ...
```

Move the existing per-block operations into reusable streaming state without changing their order:

1. Shared limiter on new stereo samples.
2. Blocking matrix using previous speech-adaptation decision.
3. NLMS using previous adaptation gate.
4. Aligned primary/reference/noise STFT frames.
5. Features and controller.
6. ORT first stage, optional refiner, optional DSP post-filter.
7. Inverse FFT, overlap-add normalization, emitted hop.

For this experiment, permit **either** refiner **or** post-filter, not both.

Reproduce startup reflect padding and final reflect padding, including odd-length clips. Emit only samples with finalized overlap-add contributions. Test the observed sample delay against the documented framing; resolve discrepancies before claiming unchanged latency.

Keep the legacy whole-clip pipeline available as the independent parity reference.

### Timing corrections

Add to `scripts/board_timing.py`:

```text
--checkpoint-config PATH
--postfilter PATH
--input PATH
--warmup-frames 100
--duration-minutes 10
```

The checkpoint-config file is JSON exported from the frozen checkpoint; the board must not require torch to read it.

Measure wall time around the **entire** `push()` operation. Include feature construction, FFTs, cache transfers, ORT calls, post-processing, and synthesis. Report mean, p95, p99, max, counts above 12 and 16 ms, RTF, cold-start timing, board identity, kernel type, and artifact hashes.

Map ONNX state outputs by explicit names, not dictionary ordering. Set both ORT intra-op and inter-op thread counts to one.

### Tests and commands

Tests first: legacy DSP parity, output waveform parity below `1e-4`, arbitrary input chunk boundaries, tail handling, correct checkpoint DSP config, named cache feedback, and injected slow-stage timing classification.

```powershell
uv run pytest tests/test_stream_runtime.py tests/test_board_timing.py tests/test_pipeline.py tests/test_stft.py -q
```

On the actual board, using the new CLI:

```bash
python scripts/board_timing.py deploy/tier46/anchor.onnx --checkpoint-config deploy/tier46/anchor_config.json --seconds 30 --threads 1 --out deploy/tier46/board_anchor.json
python scripts/board_timing.py deploy/tier46/cascade.onnx --checkpoint-config deploy/tier46/anchor_config.json --seconds 30 --threads 1 --out deploy/tier46/board_cascade.json
```

Time the untrained refiner graph before GPU training; its fixed architecture establishes compute cost. Retest the final trained artifact.

### Board gate

- **p99 < 12 ms:** eligible for a longer stability test.
- **12–16 ms:** no Tier 4.6 deployment; retain the faster accepted configuration.
- **≥16 ms:** reject on that board; evaluate Pi 4/5 if available.
- No actual board: implementation and CPU validation may continue, but cascade GPU training and deployment approval remain pending.
- Pure-Python DSP timing is diagnostic. A model-only result cannot certify the full deadline. A failed Python implementation does not prove the C port will fail; retime the actual port before making that claim.

Before shipping, run the actual runtime under audio I/O for at least ten minutes: require no underruns and no recorded frame deadline misses. The timing script alone does not prove audio-I/O stability.

- [ ] Commit the six files after tests pass.

**Time estimate:** 4–6 implementation hours plus **estimate:** 1–2 board hours. Hardware availability is an external dependency.

## 8. Task 6 — Train the frozen-stage cascade

Run only after the board gate passes.

**Files**

- Create `vaani/train_refiner.py`
- Create `configs/exp/vaani_tier46_refiner.yaml`
- Create `tests/test_train_refiner.py`
- Modify `scripts/screen_tier46.py`

Use the existing `DynamicMixDataset`, `EpochSampler`, `collate`, and `prepare_batch(..., "vaani", ...)`. Create train data from the anchor checkpoint’s exact manifests, mixer, bank, controller, and DSP config.

Run stage one on the GPU under `torch.no_grad()` and in `.eval()`. Train only the refiner. Do not write a second data mixer or cache a fixed training set.

Configuration:

```yaml
name: vaani_tier46_refiner
model: vaani_cascade
base_checkpoint: runs/tier46_anchor/best.pt
seed: 4606
epochs: 8
batch_size: 32
num_workers: 6
amp: true
optim:
  lr: 0.001
  warmup: 200
  clip: 1.0
  weight_decay: 0.0001
val:
  eval_root: data/eval_r2
  split: val
```

Inherit `epoch_len=20000` and `crop_s=4.0`. Use AdamW and the existing warmup/cosine formula.

Loss:

```python
loss = HybridLoss(
    w_complex=50, w_mag=50, p=0.5, w_snr=0.2, snr_max_db=30
)(Z.float(), target.float())

if is_clean.any():
    loss += (Z[is_clean] - target[is_clean]).abs().mean()
```

Clean supervision uses the clean target, not the noisy input. Keep loss and iSTFT in float32.

Use a deterministic validation screen: first four sorted items per bucket. Evaluate anchor on exactly those items. At each epoch, require ΔSTOI ≥ −0.003; among eligible epochs maximize:

```text
score = ΔSNR_out + 5 × ΔPESQ
```

Tie-break by earlier epoch. At epoch 2, stop if no screened checkpoint reaches either +0.1 dB SNR_out or +0.01 PESQ without violating STOI. Do not evaluate test at this stage.

After eight epochs, evaluate only the selected checkpoint on full validation. Require the refiner utility/protection thresholds before preregistering it for test. No seed sweep or automatic longer run.

Tests first:

- Two-step CPU training updates only refiner weights.
- `train()` and validation cannot alter first-stage BatchNorm buffers.
- Data comes exclusively from train sources.
- Validation ordering and selection are deterministic.
- Resume checks anchor/config hashes and restores optimizer/scheduler state.
- Checkpoint is self-contained and exportable.

```powershell
uv run pytest tests/test_train_refiner.py tests/test_cascade.py -q
```

GPU-box commands, run from the verified checkout after r4 completes:

```bash
uv run python scripts/verify_eval_set.py data/eval_r2/test eda217ab2a38
uv run python -m vaani.train_refiner configs/exp/vaani_tier46_refiner.yaml
uv run python scripts/screen_tier46.py refiner --protocol results_r2/tier46/anchor.json --checkpoint runs/vaani_tier46_refiner/best.pt --split val --out results_r2/tier46/refiner_validation
```

Then, only if validation passes:

```powershell
uv run python -m vaani.eval --system cascade:runs/vaani_tier46_refiner/best.pt --eval-root data/eval_r2 --split test --workers 8 --out results_r2/tier46/refiner.csv
uv run python scripts/tier46_protocol.py gate --kind refiner --anchor results_r2/tier46/anchor.csv --candidate results_r2/tier46/refiner.csv --protocol results_r2/tier46/anchor.json --out results_r2/tier46/refiner_gate.json
uv run python scripts/mask_phase_probe.py --system cascade:runs/vaani_tier46_refiner/best.pt --eval-root data/eval_r2 --split test --per-bucket 6
```

The phase probe explains the result; larger phase angles are not independently an acceptance criterion.

- [ ] Commit the four source/config/test files.

**Time estimate:** 3–4 implementation hours; **estimate:** 1–3 GPU hours plus 1–3 CPU evaluation hours. Measure the first epoch and enforce the eight-epoch cap. Do not interrupt r4 or launch overlapping jobs without a verified resource budget.

## 9. Task 7 — Golden vectors, deployment handoff, and final report

Only package candidates that pass quality and timing gates.

**Files**

- Modify `scripts/make_golden_vectors.py`
- Modify `tests/test_golden_vectors.py`
- Modify `deploy/CONTRACT.md`
- Modify `deploy/dsp_reference/README.md`
- Create `deploy/tier46/CONTRACT.md`
- Create versioned vectors under `deploy/dsp_reference/vectors/tier46/`
- Update `.superpowers/sdd/2026-09-18-vaani-training-stack/progress.md`

Make vector generation accept:

```text
--checkpoint
--postfilter
--out-dir
```

Preserve existing legacy vectors. The current generator/test replay uses default DSP configuration; Tier 4.6 vectors must embed and replay the selected checkpoint’s actual r3/r4 DSP config.

Generate deterministic cases covering:

- Speech plus noise and stationary residual noise.
- A burst after at least one second of tracker history.
- Reference dropout.
- Silence and clean speech.
- Non-hop-aligned length and reset/restart.

Store input WAVs as float32, plus intermediate `mix`, `n_hat`, features, gates, first-stage spectrum, final spectrum, output waveform, and every post-filter/refiner state. Generate expected outputs by rereading the written WAVs.

For the C port:

- Post-filter stays outside ONNX; port the equations and float32 state directly.
- Refiner remains inside the combined ONNX graph; the port carries one additional cache.
- Keep 512/256 framing and normalization unchanged.
- Require waveform/spectral parity `<1e-4`, feature/noise-estimate parity `<1e-4`, and exact discrete gate/burst decisions.
- Existing vectors remain valid for their original configuration.

Tests first, then generation and verification:

```powershell
uv run pytest tests/test_golden_vectors.py tests/test_export.py tests/test_stream_runtime.py -q
uv run python -m vaani.export runs/tier46_anchor/best.pt --out deploy/tier46/anchor.onnx
uv run python -m vaani.export runs/vaani_tier46_refiner/best.pt --out deploy/tier46/cascade.onnx
uv run python scripts/make_golden_vectors.py --checkpoint runs/tier46_anchor/best.pt --postfilter results_r2/tier46/postfilter.selected.yaml --out-dir deploy/dsp_reference/vectors/tier46
```

Run only commands for accepted artifacts. For an accepted cascade, pass its checkpoint and omit `--postfilter`.

Produce the report using explicit CSV paths for the anchor and accepted/evaluated candidates:

```powershell
uv run python -m vaani.report results_r2/tier46/anchor.csv results_r2/tier46/postfilter.csv results_r2/tier46/refiner.csv --protocol results_r2/tier46/anchor.json --out results_r2/tier46/matrix.md
uv run pytest -q
```

Remove a CSV argument if that experiment was killed before test evaluation. Do not glob partial results into the report.

Record both rejected and accepted experiments, their thresholds, actual measured values, board identity, latency convention, and limitations. Do not overwrite the existing headline matrix or `deploy/model.onnx` as part of experiment packaging.

- [ ] Commit only the named task files and generated Tier 4.6 vectors, with no attribution trailer. Do not push.

**Time estimate:** 3–4 hours, plus embedded-team port verification.

## 10. Asymmetric windows: deferred decision

Do not modify `vaani/dsp/stft.py` for this release.

Simply truncating the synthesis sqrt-Hann window is not an acceptable implementation. An analysis/synthesis pair must satisfy normalized overlap-add reconstruction:

\[
\sum_m w_a[n-mH]\,w_s[n-mH]\neq0
\]

throughout the emitted region, with the corresponding normalization and a documented causal emission schedule.

A shorter synthesis window alone does not establish lower latency: the current analysis still needs future samples. A real redesign would require:

- Analysis/synthesis alignment and reconstruction proofs/tests.
- Consistent frame alignment for controller features and NLMS spectra.
- Startup, tail, impulse-delay, and batch/stream tests.
- Retraining or at least explicit checkpoint/window compatibility testing.
- Revalidated exports, versioned golden vectors, and coordinated C-port changes.

**Estimate:** 2–4 engineering days plus retraining and port validation, with uncertain quality. That competes directly with the ten-day deadline.

Reopen only after deployment is stable, with an independently specified latency target. A future candidate must demonstrate measured sample-delay reduction and pass the same frozen-test intelligibility/protection gates; no quality or latency improvement is promised here.

## 11. Schedule and stop rules

| Date | Work | Compute |
|---|---|---|
| 20–21 Sep | Freeze protocol; implement post-filter and validation screen | Laptop CPU |
| 21–22 Sep | Implement refiner/export and complete-loop timing | Laptop; actual board |
| 22–23 Sep | One gated refiner training run; selected candidate test evaluations | GPU box, CPU metrics |
| 23–24 Sep | Golden vectors, port verification, report | Laptop and board |
| 25 Sep | Freeze architecture; retain only accepted artifacts | No new experiments |
| 26–29 Sep | Deployment stability, evidence, submission integration | Board |
| 30 Sep | Buffer | — |

If board timing remains unavailable, finish the post-filter evidence and document the refiner as prepared but untrained. If neither candidate passes, retain the anchor and report the negative result.

Do not:

- Change the frozen test split, redefine nominal, or suppress failed metric rows.
- Treat removed residual, NLMS output, or coherence as oracle noise.
- Tune thresholds against test failures.
- Run VaaniNet twice without training the second stage for that input distribution.
- Add a synthesis/reanalysis boundary between stages.
- Combine post-filter and refiner without a separate preregistered experiment.
- Infer Pi performance from laptop timing.
- Trade away STOI merely because SNR improves.
- Claim target success from SI-SDR, SNR improvement, or a selected easy bucket.
- Push, publish, overwrite shared artifacts, or include self-attribution.

The deliverable is an auditable decision: an accepted quality improvement with measured board feasibility, or a documented rejection that preserves the existing deployment.
