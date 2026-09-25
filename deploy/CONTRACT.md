# VAANI deployment contract (r7 cascade)

This is the per-frame contract an embedded port must reproduce. It describes the **r7 cascade**, the
shipping system. The previous candidate (tier46) and the historical first-stage export are in the
appendices; they share r7's signature but not its weights.

## Shipping artifacts

| Artifact | Path | Bytes | sha256 |
|---|---|---:|---|
| Streaming graph | `deploy/r7/cascade.onnx` | 474,599 | `e67a2c42a2fd53ec4d7e6c4dfbf9cef920c249e499a509a2b40af401c5cefe3c` |
| Cascade checkpoint (embeds the backbone) | `results_r2/runs/r7_e256_wr64_refiner/best.pt` | 294,044 | `121f0c3d15936b810573dc089c597e67fe67f6d09e3d75c2469746decefe2083` |
| Backbone checkpoint | `results_r2/runs/r7_e256_wr64/best.pt` | 278,522 | `0bea98181cf7846d64fdd14e6fc16ec9d45c0028d5feb08f76279f5947b81106` |
| DSP configuration the weights were trained behind | `deploy/r7/model_config.json` | – | – |
| Parity, timing and MAC record | `deploy/r7/cascade_parity_timing.json` | – | – |
| Golden vectors for this DSP configuration | `deploy/dsp_reference/vectors_cascade/` | – | bound to both hashes above in its `config.json` |

Both checkpoints and their `run.json` are tracked in git, so a clone can re-export the graph (see
[Export and reproduction](#export-and-reproduction)). The graph does not encode the DSP
configuration: a port must read `deploy/r7/model_config.json` (limiter on, blocking matrix on,
controller on with `diff_jump_max_db` 3.0 and `block_margin_db` 10.0; `model_cfg` channels 16,
`df_order` 3, coherence on, FiLM off, noise floor off).

Live runtime: `vaani/live.py::StreamEngine` runs this contract one hop at a time (limiter, blocking
matrix, NLMS, features, controller, graph, overlap-add) with numpy and onnxruntime only, and
`tests/test_live.py` holds it to the offline eval path within 1e-5. `scripts/capture_loop.py` wraps
it for ALSA capture/playback and for WAV files.

## Audio and framing

- 16 kHz, 2 channels (0 = primary near-mouth, 1 = reference), float32 in [-1, 1].
- STFT: n_fft 512, hop 256, window = sqrt(periodic Hann 512), center=True (reflect pad 256).
- One model call per hop: 62.5 hops/s.

## Latency contract

| Quantity | Value | Kind |
|---|---:|---|
| Hop (processing deadline per model call) | 16 ms | arithmetic from the framing |
| Algorithmic delay | **32 ms** | arithmetic: one 16 ms hop plus the 16 ms half-window the centred STFT has not yet received |
| 48 kHz path resampler group delay | **+4 ms** total (48→16 kHz and 16→48 kHz together) | arithmetic from the FIR in `vaani/live.py` |
| Audio buffering and compute | not included above | – |
| End-to-end (microphone to earphone) | **not measured** | TBD: measured on the Pi 5 or the Orin with a loopback impulse |

The algorithmic delay is 32 ms, not 16 ms. Changing the window or hop (for example a 256/128 or
asymmetric-window low-delay profile) is a separately trained and evaluated profile, not a deployment
switch.

Complete-hop cost: `scripts/hop_benchmark.py` times the whole hop as `StreamEngine` runs it (limiter,
blocking matrix, NLMS, features, controller, STFT, model, iSTFT, optional 48 kHz resampling), per
stage, e.g. `python scripts/hop_benchmark.py --models r7,C16,C32,C64,C96 --backends ort-cpu,torch-cpu,torch-cuda`.
`scripts/board_timing.py --seconds 30 --resample48 --out deploy/board_timing.json` wraps it for the
board. Only smoke runs on a loaded laptop exist; no complete-hop result is committed or reportable,
and no board run has been made. TBD: a complete-hop run on an idle machine and on the Pi 5.

## Per frame, in order

All steps below are enabled for r7 by `deploy/r7/model_config.json`.

0. Sub-block limiter on the 256 new samples of BOTH mics, `vaani/dsp/limiter.py`: 2 ms sub-blocks,
   one shared gain, instant attack, 50 ms release, ceiling = 26 dB over a 500 ms RMS tracker that
   is frozen while engaged; engages only when the sub-block primary/reference ratio is within 4 dB
   of the tracked noise ratio (far-field). No added latency: the whole hop is limited before it
   reaches the NLMS, the STFT and the model. Whether the limiter engaged in this hop or the
   previous one is an input to step 3.
1. Blocking matrix, `vaani/dsp/blocking.py`: the same NLMS kernel with the roles swapped (16 taps,
   mu 0.01, predicts the reference from the primary), adapted with gate = previous frame's
   `speech_adapt` (controller: speech-presence > 0.5 AND primary >= `block_margin_db` (10) over its
   own slow energy floor AND no burst/overload/dropout), never reset. `ref_b = ref - h_s * prim`
   replaces `ref` as the noise-path NLMS input.
2. NLMS block on the 256 new samples with gate = previous frame's `adapt_gate` (64 taps, mu 0.05,
   eps 1e-6) -> `n_hat` block. See `deploy/dsp_reference/` and `vaani/dsp/nlms.py`.
3. Features (18, order below) from the current 512-sample primary/reference frames and their
   spectra, `vaani/dsp/features.py`.
4. Controller -> `adapt_gate`, `burst_flag`, `reliability`, `vaani/dsp/controller.py`. Thresholds:
   jump 12 dB, hold 4 frames, ramp 12 frames, speech-freeze 0.5. With `diff_jump_max_db` = 3.0 the
   level-difference test is the *differential jump*: the same 4 ms onset test run on the reference
   frame, subtracted from the primary's (feature 0); burst = onset >= 12 dB AND diff <= 3 dB, OR the
   limiter engaged (step 0). The differential jump is not one of the 18 features.
5. ONNX call (signature below). The refiner runs inside the same call after the first stage;
   `spec_out` is the refined spectrum.
6. iSTFT overlap-add with the same sqrt-Hann window, on `spec_out`.

For the *no-controller* configuration (not r7): gate = 1, feats = zeros.

## ONNX signature (`deploy/r7/cascade.onnx`, opset 17, static shapes, batch 1)

| Input | Shape | Output |
|---|---|---|
| `spec6` = [prim_re, prim_im, ref_re, ref_im, nhat_re, nhat_im] | (1,257,1,6) | `spec_out` (1,257,1,2), enhanced primary re/im |
| `feats` | (1,1,18) | – |
| `conv_cache` | (2,1,16,16,33) | `conv_cache_out` |
| `tra_cache` | (2,3,1,1,16) | `tra_cache_out` |
| `inter_cache` | (2,1,33,16) | `inter_cache_out` |
| `df_cache` | (1,257,3,2) | `df_cache_out` |
| `coh_cache` | (1,4,257) | `coh_cache_out` |
| `refine_cache` | (1,16,2,257) | `refine_cache_out` |

All float32. All six caches are zero-initialised at stream start and each `*_out` is fed back
unchanged as the next frame's input. The graph has 1,756 nodes, including 14 `GRU` and 18
`ScatterND` nodes (read from the file with `onnx`); both matter for a TensorRT port.

### Cache meaning

Per-stream (batch 1) state, zeroed once at stream start (`vaani/models/vaani_net.py::init_caches`),
never between frames of the same stream:
- `conv_cache (2,1,16,16,33)`: dim0 {encoder, decoder}; per-block receptive-field history for the
  three dilated GTConvBlocks.
- `tra_cache (2,3,1,1,16)`: dim0 {encoder, decoder}, dim1 the 3 GTConvBlocks; hidden state of their
  temporal recurrence.
- `inter_cache (2,1,33,16)`: dim0 {dpgrnn1, dpgrnn2}; inter-frame GRU hidden state.
- `df_cache (1,257,3,2)`: the previous three primary spectra (re/im), newest at index 0; with
  `df_order` 3 the first two slots feed the deep-filter taps.
- `coh_cache (1,4,257)`: EMA cross-spectra of primary vs reference (Pxx, Pyy, Re Pxy, Im Pxy),
  alpha 0.9 per frame.
- `refine_cache (1,16,2,257)`: the refiner's previous two hidden frames, oldest first.

## Feature order

log_energy_delta, spectral_flux, peak_to_rms, clip_frac_primary, clip_frac_reference,
speech_presence, coherence_b0..b7, level_diff_db, ref_dropout, nlms_health, prev_gate

(18 total; exact order and definitions in `vaani/dsp/features.py::FEATURE_NAMES`.)

## Stream state and telemetry

A stream is represented as `{profile_id, sample_counter, channel_validity, discontinuity_flags,
state}`, where `state` carries the caches plus a schema version, their shapes and dtypes, and a hash
of the preprocessing configuration. Rules:

- Width or profile changes happen only at an explicit reset/reinitialisation boundary. Hidden state
  is never resized or reused across widths or profiles.
- A runtime reset must not replay stale audio.
- No future-time information enters normalisation, masks, coherence, controller inputs or hidden
  states.
- Each hop reports timing, queue occupancy, clipping, missing samples and fallback events.

`vaani/backend.py` implements this (tests: `tests/test_stream_contract.py`, `tests/test_backend.py`):

- `StreamState`: `schema_version` (`SCHEMA_VERSION = 1`), `profile_id`, `config_hash`, `caches`,
  `sample_counter`, `channel_validity` (primary, reference) and `discontinuity_flags`.
  `StreamState.save` / `load` write an `.npz` whose JSON header spells out every cache's name, shape
  and dtype; another schema version or a header/array mismatch is refused.
- Discontinuity bits: `DISC_GAP = 1` (input samples dropped by the overload queue or a capture
  xrun), `DISC_BYPASS = 2` (hops passed through raw by the overload bypass), `DISC_RESET = 4`
  (mid-stream reset), `DISC_REF_DROPOUT = 8` (the reference was invalid for at least one hop).
- `Backend`: `new_state(config_hash)`, `reset(state)`, `step(spec6, feats, state, valid=1.0)`,
  `to_host(state)`, `from_host(state)`. `check_state` refuses a state whose profile or cache shapes
  differ from the backend's. Backends: `OrtBackend`, `TorchBackend` (r7) and `FeOrtBackend`,
  `FeTorchBackend` (VaaniFE). The GPU execution-provider paths are written but untested.
- Telemetry `summary()` keys: `steps`, `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `max_ms`,
  `deadline_ms`, `deadline_misses`, `fallback_events` (a count per event name).
- Graph hash: `StreamEngine` checks the graph against `onnx_sha256` in `model_config.json` on load
  and refuses a mismatch unless `allow_hash_mismatch` (`--allow-hash-mismatch` in
  `scripts/capture_loop.py`) is set.
- Overload: `vaani.live.BoundedHopQueue` drops the oldest hops when full (`--queue-hops 8`); once
  the backlog reaches `--bypass-depth 4` the loop passes the raw primary through for that hop.
  Both set the discontinuity bits above.
- Reference dropout: the StreamEngine docstring in `vaani/live.py`. It zeroes the reference and
  freezes NLMS and blocking adaptation, then ramps the reference back over 16 hops (256 ms). This is
  runtime safety only: r7 was not trained for it, and its quality during a dropout is not measured.

Runtime guards (`vaani/guards.py`, `tests/test_guards.py`), opt-in (`--guards`, off by default):

- Reference informativeness: |corr(P, R)| > 0.97 and |ILD| < 1 dB, held for 0.5 s, marks the
  reference uninformative (validity 0). It recovers after 0.5 s of informative hops. Events:
  `ref_uninformative` / `ref_informative`.
- Never-vanish: output more than 25 dB below the input for more than 0.5 s of VAD-speech hops
  crossfades to the fallback. Events: `never_vanish` / `never_vanish_release`.

G4 field acceptance (`scripts/field_accept.py`) scores its validity-flag latency criterion from
`eng.last[key]` for `key` in (`ref_informative`, `ref_validity`, `validity`), where falsy means
uninformative. An r8 runtime must expose one of these keys for that criterion to be scored.

## Parity and timing (r7)

`deploy/r7/cascade_parity_timing.json`, 10-second random-input stream, 624 timed frames (frame zero
excluded from timing, included in parity), one intra-op thread, CPUExecutionProvider:

- Maximum absolute error against batch PyTorch: **1.11e-6** (tolerance < 1e-4).
- Model time: **1.135 ms mean / 1.960 ms p99** per 16 ms hop, on a **cloud x86 Linux core with
  ORT 1.25**, not the development laptop or a board.
- On the Windows development laptop (AMD64 Family 25 Model 117, ORT 1.30, one thread, 2,000 timed
  hops): 1.034 ms mean, 1.732 ms p99 (laptop). Source: a local-only profile JSON
  (`docs/research/2026-09-24/scaling_profile.json`, not tracked), so this pair cannot be checked
  from a clone. TBD: re-time on an idle machine and commit the JSON.

These are model-only numbers. They exclude DSP, STFT/iSTFT, resampling and audio I/O, so they do
not establish end-to-end latency or board feasibility. Re-time on the Pi 5 and, when available, the
Orin.

A port passes when its `max_abs_err` against the PyTorch stream reference is < 1e-4.

### Arithmetic budget (from the same JSON)

| Component | Parameters | Matrix MAC/frame | Matrix MMAC/s at 62.5 frames/s |
|---|---:|---:|---:|
| First stage (frozen during refiner training) | 50,249 | 686,112 | 42.882 |
| Residual refiner | 2,498 | 633,248 | 39.578 |
| Cascade | 52,747 | 1,319,360 | 82.460 |

MACs are a dense Conv/Linear/GRU matrix-operation estimate from one streaming call, including the
fixed ERB transforms and padded positions; biases, normalisation, activations, elementwise ops and
DSP/STFT are excluded. It is not a runtime measurement. The refiner's 16-channel 3x3 convolution
runs across all 257 bins (592,128 MAC/frame in that layer alone): a small parameter count does not
mean low compute.

## Export and reproduction

Run from the repository root:

```bash
# re-export the shipping graph and its parity/timing record (the defaulted output is refused without --overwrite)
uv run python -m vaani.export results_r2/runs/r7_e256_wr64_refiner/best.pt --out deploy/r7/cascade.onnx --seconds 10 --report-json deploy/r7/cascade_parity_timing.json --overwrite
# regenerate the golden vectors for r7's DSP configuration, bound to the checkpoint and graph hashes
uv run python scripts/make_golden_vectors.py --checkpoint results_r2/runs/r7_e256_wr64_refiner/best.pt --onnx deploy/r7/cascade.onnx
uv run pytest tests/test_export.py tests/test_golden_vectors.py -q
```

`vaani/export.py::export` builds the streaming twin, loads weights via `convert_to_stream` (a plain
`load_state_dict` does not work: the stream conv wrappers nest keys one level deeper) and traces one
frame with `torch.onnx.export(..., opset_version=17, dynamo=False)` at the static shapes above.
`parity_and_timing` runs the ORT session frame by frame against the batch model, carrying caches
exactly as the embedded loop must. A re-export's timing will differ from the recorded one; its
parity must stay under the tolerance.

Reproducibility, measured: re-exporting r7 with torch 2.11 gives the same topology (all 1,756 nodes
byte-equal) and parity 6.7e-7 against PyTorch. It is **not** byte-identical to
`deploy/r7/cascade.onnx` (exported with torch 2.14): 23 folded Conv initializers differ by up to
9.5e-7, and shipped vs re-exported outputs differ by up to 5.1e-7. `deploy/r7/cascade.onnx` stays
the sha-pinned artifact.

Board install: `bash scripts/pi_setup.sh [--timing]` (`deploy/PI_SETUP.md`). It creates a Python
3.12 `.venv-board` that is PEP 668-safe, installs `requirements-deploy.txt` binary-only (numpy +
onnxruntime + numba, no torch), checks that torch is not importable and verifies r7's sha256.
The aarch64 wheels are unverified on a board. TBD: the Pi 5 OS image and a first board run.

## Golden vectors

`deploy/dsp_reference/vectors_cascade/<case>.wav` and `<case>.npz` cover r7's DSP configuration
(limiter, blocking matrix, differential-jump controller); `config.json` binds it to the checkpoint
and graph SHA256. Match `n_hat`, `features` and limited `mix` within 1e-4, and `gate`, `burst` and
`reliability` exactly. Replay tests also require the preserved legacy r1/r2 set in
`deploy/dsp_reference/vectors/`.

## Not covered here

Output crossfade/bypass on low reliability, overrun handling beyond the stream flags above, and
radio interfacing are the DSP/embedded leads' responsibility. This contract is for transmit-path
speech enhancement; it is not an ear-side ANC loop.

## Future profiles (r8, VaaniFE)

New checkpoints may use a different backbone (the VaaniFE family), a reference-validity input and
different cache names and shapes. A port must read the exported input names and dimensions rather
than reuse r7's allocations, and must zero every cache at a new stream. VaaniFE graphs are built
without `GRU`, `Loop` or `ScatterND` nodes (one-step GRUs as Gemm cells, Slice+Concat caches).
Orin-tier figures for that family are projections from counts, graph checks and laptop timing; no
Orin latency, power or quality is measured.

### VaaniFE step graph (`vaani/models/vaani_fe.py`, `vaani/export.py::export_fe`)

| Name | Shape | Meaning |
|---|---|---|
| `spec` (in) | (1, n_raw, 257) | the first n_raw channels of the engine's `[P re, P im, R re, R im, n_hat re, n_hat im]` frame, channels-first; n_raw is 4 for the default inputs `pr` |
| `valid` (in) | (1, 1) | this frame's reference validity; absent when the inputs are `p` (mono) |
| `state` (in) | (1, S) float32 | the only cache: K blocks of F·C2 GRU hidden, plus (df_taps − 1)·128 deep-filter frames when df_taps > 0 |
| `spec_out` (out) | (1, 2, 257) | enhanced raw STFT frame [re, im] |
| `state_out` (out) | (1, S) | next state |

- Mini: S = 2 · 16 · 24 = 768 floats (3,072 B). Profile id `vaani_fe-<tier>`. A state from another
  tier or profile is refused, never resized. `StreamState` and `SCHEMA_VERSION` are unchanged.
- Validity per frame: 1 iff the reference was valid on both hops of the frame **and** the runtime
  guards trust it; otherwise 0, with the reference zeroed. The reconnect ramp is 16 hops, or
  `dsp.ref_policy.ramp_frames` (12) for a validity model.
- `model_config.json`: `{"kind": "vaani_fe", "model": "vaani_fe", "profile", "controller_on", "dsp",
  "model_cfg", "onnx", "onnx_sha256"}`, written by `vaani.live.write_fe_model_config`, or by
  `write_model_config` from a `model: vaani_fe` checkpoint. r7's config is unchanged (`kind`
  defaults to `cascade`).
- Export: a checkpoint from `vaani/train.py` goes through `export.fe_load` + `export_fe` (opset 17,
  static batch one). This writes `<name>.onnx` and ORT's folded `<name>.folded.onnx`, then checks
  ORT-vs-torch parity over carried-state hops (tolerance 1e-5). `best.pt` records `weights`
  (`raw` or `ema`) and `selection`.
- Run live: `python scripts/capture_loop.py --onnx <tier>.folded.onnx --config <dir>/model_config.json --in-wav mix.wav --out-wav out.wav`.
  The backend is picked from the graph's input names.
- Tests: `tests/test_fe_stream.py` covers parity, state round trip, interleaved streams, the
  validity rule, the mono path at validity 0, and the guards driving validity 0.

G2 graph gate (`scripts/graph_gate.py`; limits: folded nodes < 250, no Loop/Scan/If/GRU/LSTM/RNN,
no ScatterND, no symbolic dims or Shape/Range, layout ops < 30 %, parity ≤ 1e-5). On untrained
seeded exports (`results_r2/fe_tiers/README.md`), every tier passes: Mini 116 folded nodes, Mid
158, Large 193, Large+ 200, layout share 0.259–0.264, parity ≤ 4.5e-8. r7 fails (539 nodes,
14 GRU, 18 ScatterND, layout share 0.586). The r8 Mini smoke export also passes (116 folded nodes, `configs/retraining/R8_RUNBOOK.md`). This is
a laptop graph check, not a TensorRT or Orin result.

### Reference-validity r7 variant (the r8 fallback)

With `model_cfg.ref_validity: true`, the r7 architecture takes a reference-availability input
`ref_avail` in {0, 1}: (B, T) for the batch model, (B, 1) as a keyword to `StreamVaaniNet`. `None`
means present. The DSP `ref_policy` (default off) zeroes the reference and freezes the NLMS for
unavailable samples, then ramps back over 12 frames. The ONNX export does **not** yet expose
`ref_avail` as a graph input. TBD: add it before this variant can ship. All of this is default-off,
and r7's outputs are unchanged. Budget: 53,147 entries, 85.621 MMAC/s (`results_r2/r8/budget.md`).

Existing opt-in retraining flags on the r7 architecture: `model_cfg.channels`,
`model_cfg.noise_floor` and `refiner_cfg.hidden/past/scale`. Widths change the convolution and
recurrent cache shapes; floor-enabled models append `noise_cache` (1,2,257: power floor and
initialised flag) after `coh_cache`, before the cascade's final `refine_cache` (1,hidden,past,257).
The conditional refiner is a separate Python host runtime that advances c0 history on skipped frames
and conditionally computes c1/c2; ONNX export is always-on. Its MAC activation estimates do not
establish target-device latency or clean-speech transparency.

---

## Appendix A: tier46 cascade (previous candidate, not shipping)

`results_r2/runs/vaani_tier46_refiner/best.pt` (sha256
`932b086a2842eb88f4232b087fd99f8769bd102135e6b887dd6ecf38ab5aa2f6`): frozen `vaani_full_r4_ctl`
first stage plus residual refiner, exported to `deploy/tier46/cascade.onnx` (474,599 bytes, sha256
`6d1b58e7…`). It has the same signature, cache shapes, parameter count and MAC budget as r7, with
different weights. The `.onnx` and `.pt` files under `deploy/tier46/` are not tracked; its JSON
reports are.

Laptop measurement (`deploy/tier46/trained_cascade_timing.json`; Windows 11, AMD64 Family 25 Model
117, ORT 1.30.0 CPU, one thread, 624 timed frames): max_abs_err **1.505e-6**; **0.999 ms mean /
1.696 ms p99** per hop (laptop, model only).

### INT8 quantization: measured on tier46, not adopted

Dynamic INT8 (`onnxruntime.quantization.quantize_dynamic`, QInt8 weights, no calibration set),
measured on the same laptop, interleaved, best of five 10-second runs. The FP32 row differs slightly
between the two reports because each is its own timing run.

| Graph (report) | Bytes | Nodes | Initializer bytes | ms/frame mean | ms/frame p99 |
|---|---:|---:|---:|---:|---:|
| `cascade.onnx` FP32 (`int8_report.json`) | 474,599 | 1,756 | 210,108 | 0.906 | 1.152 |
| `cascade.int8.onnx` (`int8_report.json`) | 567,969 | 1,906 | 157,133 | 1.319 | 1.626 |
| `cascade.onnx` FP32 (`int8_perchannel_report.json`) | 474,599 | 1,756 | 210,108 | 0.902 | 1.156 |
| `cascade.int8_pc.onnx` per-channel (`int8_perchannel_report.json`) | 569,805 | 1,906 | 158,913 | 1.310 | 1.582 |

INT8 is **19.7 % larger and 1.45x slower** (per-channel: 20.1 % larger, 1.45x slower). Two
structural reasons, specific to a model this small:

- Weights are only 210 KB of a 474 KB graph; the rest is node protobuf. Quantization cut weight
  bytes to about 157 KB but added 150 nodes, and the node overhead exceeded the saving.
- ORT's dynamic path has no integer kernel for `GRU`, which is most of this model's recurrence, so
  the inserted `DynamicQuantizeLinear` nodes are added work on top of an unchanged float recurrence.
  One `MatMul` was left unquantized.

The graph-level error against FP32 is 0.076 max abs (0.27 relative), a wiring check rather than a
quality result; quality on the frozen split is in `results_r2/optim/optimization.md`. These numbers
describe this graph on ORT 1.30 CPU, not quantization in general, and r7 was not re-quantized
(inferred to behave the same, since it shares the graph structure).

## Appendix B: historical first-stage export

`deploy/model.onnx` (generated locally, not tracked) is a first-stage-only graph from the round-3 checkpoint
`results_r2/runs/vaani_full_r3_dflr/best.pt` (`df_order` 3 + coherence, exported 2026-09-20). On the dev
laptop, 10-second random input, one thread: max_abs_err 1.0e-6, 1.16 ms mean / 1.78 ms p99, 423 KB
(laptop). The round-1 `results_r2/runs/vaani_full/best.pt` export measured 1.2e-6 / 1.86 ms / 3.25 ms /
424 KB on the same laptop. It has no `refine_cache` and does not describe the cascade.
