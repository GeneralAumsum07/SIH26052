# VAANI deployment contract (trained Tier 4.6 cascade)

> **Shipping model changed (23 Sep 2026).** The system to ship is now the r7 cascade, `runs/r7_e256_wr64_refiner/best.pt` (see the README). It has the same architecture, parameter count, ONNX signature and cache shapes as the tier46 cascade described below, so this contract applies to it unchanged. Its export is `deploy/r7/cascade.onnx` (474,599 bytes, sha256 `e67a2c42a2fd53ec`), with the DSP configuration the weights were trained behind in `deploy/r7/model_config.json` (the graph does not encode it). Parity against batch PyTorch: max_abs_err **1.11e-6** (tolerance 1e-4); model time 1.14 ms mean / 1.96 ms p99 per 16 ms hop on a cloud x86 core (`deploy/r7/cascade_parity_timing.json`) - not the dev laptop, so not directly comparable with the tier46 figures below; re-time on the laptop and the Pi. The measurements below remain the tier46 graph's.

Live runtime: `vaani/live.py::StreamEngine` runs this contract one hop at a time (limiter, blocking matrix, NLMS, features, controller, graph, overlap-add) with numpy + onnxruntime only, and `tests/test_live.py` holds it to the offline eval path within 1e-5. `scripts/capture_loop.py` wraps it for ALSA capture/playback and for WAV files.

The evaluated deployment candidate is `runs/vaani_tier46_refiner/best.pt`: frozen
`vaani_full_r4_ctl` first stage plus residual refiner. It exports to
`deploy/tier46/cascade.onnx`. The older `deploy/model.onnx` is a first-stage-only
target; its historical measurements below do not describe the cascade.

Audio: 16 kHz, 2 channels (0 = primary near-mouth, 1 = reference), float32 [-1,1].
STFT: n_fft 512, hop 256, window = sqrt(periodic Hann 512), center=True (reflect pad 256).
One model call per hop (16 ms).

Latency (derived, not measured): one hop (16 ms) + the STFT window's lookahead. With
center=True the window straddles the hop, so the last half-window (16 ms) is not yet
available -- this is arithmetic from the STFT framing above, not a runtime measurement.
Total: 16 ms + 16 ms = 32 ms.

## Per frame, in order
0. (round-3 checkpoints, `dsp.limiter` in the checkpoint config; steps 0 and the blocking step are r3-only) Sub-block limiter on the 256 new samples of BOTH mics, `vaani/dsp/limiter.py`: 2 ms sub-blocks, one shared gain, instant attack, 50 ms release, ceiling = 26 dB over a 500 ms RMS tracker that is frozen while engaged, engages only when the sub-block primary/reference ratio is within 4 dB of the tracked noise ratio (far-field). No added latency: the whole hop is limited before it reaches the NLMS, the STFT and the model. Whether the limiter engaged in this hop or the previous one is an input to step 3.
1. (round-3 checkpoints, `dsp.blocking`) Blocking matrix, `vaani/dsp/blocking.py`: the same NLMS kernel with the roles swapped (16 taps, mu 0.01, predicts the reference from the primary), adapted with gate = previous frame's `speech_adapt` (controller: speech-presence > 0.5 AND primary >= `block_margin_db` (10) over its own slow energy floor AND no burst/overload/dropout), never reset. `ref_b = ref - h_s * prim` replaces `ref` as the noise-path NLMS input. With the controller off it stays at zero (inert).
1. NLMS block on the 256 new samples with gate = previous frame's `adapt_gate` (64 taps, mu 0.05, eps 1e-6) -> `n_hat` block. See `dsp_reference/` and `vaani/dsp/nlms.py`.
2. Features (18, order below) from the current 512-sample primary/reference frames and their spectra. `vaani/dsp/features.py`.
3. Controller -> `adapt_gate`, `burst_flag`, `reliability`. `vaani/dsp/controller.py`. Thresholds: jump 12 dB, level-diff <= 3 dB, hold 4 frames, ramp 12 frames, speech-freeze 0.5 (0.6 before 2026-09-20; see `dsp_reference/README.md`). Round-3 checkpoints (`dsp.controller.diff_jump_max_db` = 3.0 in the checkpoint config) replace the level-diff test with the *differential jump*: the same 4 ms onset test run on the reference frame, subtracted from the primary's (feature 0); burst = onset >= 12 dB AND diff <= 3 dB, OR the limiter engaged (step 0). The differential jump is not one of the 18 features.
4. ONNX `model.onnx`: inputs `spec6 (1,257,1,6)` = [prim_re, prim_im, ref_re, ref_im, nhat_re, nhat_im], `feats (1,1,18)`, `conv_cache (2,1,16,16,33)`, `tra_cache (2,3,1,1,16)`, `inter_cache (2,1,33,16)`, `df_cache (1,257,3,2)`, `coh_cache (1,4,257)`, all float32, all zero-initialised at stream start; outputs `spec_out (1,257,1,2)` (enhanced primary re/im) + the five caches, same shapes, updated. Feed caches back unchanged into the next frame's call. (v1 had three caches; `df_cache`/`coh_cache` were added 2026-09-20 for the round-3 architecture and are present, and passed through, for every exported checkpoint.)
5. For `tier46/cascade.onnx`, the refiner runs inside the same ONNX call after the first stage.
   It adds input `refine_cache (1,16,2,257)` and output `refine_cache_out` to the signature above;
   all six caches are zero-initialised at stream start and fed back on every frame. `spec_out`
   is the refined spectrum. The cache stores the previous two refiner hidden frames, oldest first.
6. iSTFT overlap-add with the same sqrt-Hann window, on `spec_out`.

For the *no-controller* configuration: gate = 1, feats = zeros.

## Feature order
log_energy_delta, spectral_flux, peak_to_rms, clip_frac_primary, clip_frac_reference, speech_presence, coherence_b0..b7, level_diff_db, ref_dropout, nlms_health, prev_gate

(18 total; exact order and definitions in `vaani/dsp/features.py::FEATURE_NAMES`.)

## First-stage cache shapes and zero-init
All five caches are per-stream (batch=1) state carried frame-to-frame; zero-initialise once
at stream start (`vaani/models/vaani_net.py::init_caches`), never between frames of the
same stream:
- `conv_cache (2,1,16,16,33)` -- dim0 indexes {encoder, decoder}; per-block receptive-field
  history for the three dilated GTConvBlocks.
- `tra_cache (2,3,1,1,16)` -- dim0 {encoder, decoder}, dim1 indexes the 3 GTConvBlocks; hidden
  state for their internal temporal recurrence.
- `inter_cache (2,1,33,16)` -- dim0 indexes {dpgrnn1, dpgrnn2}; inter-frame GRU hidden state.
- `df_cache (1,257,3,2)` -- the previous three primary spectra (re/im), newest at index 0. The model
  shifts the current primary frame in on every call; a round-3 checkpoint (`model_cfg.df_order` = 3)
  reads the first `df_order-1` slots for its deep-filter taps, earlier checkpoints only pass it through.
- `coh_cache (1,4,257)` -- EMA cross-spectra of primary vs reference (Pxx, Pyy, Re Pxy, Im Pxy), alpha
  0.9 per frame, used when `model_cfg.coh` is on; otherwise passed through unchanged.

## Architecture flags (`model_cfg` in the training config, stored in the checkpoint)
`df_order` (1 = complex ratio mask, 3 = per-bin complex FIR over the current and two past frames),
`film` (18-feature FiLM shift on the first encoder layer; off in round 3), `coh` (coherence map as a
10th input channel). `vaani/export.py` reads them from the checkpoint; the ONNX signature does not change.

## Export and parity: historical first-stage measurement
`vaani/export.py::export(ckpt_path, out_path)` builds the streaming twin (`StreamVaaniNet`),
loads weights via `convert_to_stream` (a plain `load_state_dict` does not work -- the stream
conv wrappers nest keys one level deeper), and traces one frame with
`torch.onnx.export(..., opset_version=17, dynamo=False)` at the shapes above (batch 1, static).

`vaani/export.py::parity_and_timing(ckpt_path, onnx_path, seconds)` runs the ONNX Runtime
session frame-by-frame (CPU, `intra_op_num_threads=1`) against the batch `VaaniNet` doing the
same, carrying caches forward exactly as the embedded loop must. On the historical round-3
checkpoint (`runs/vaani_full_r3_dflr/best.pt`, `df_order` 3 + coherence, exported to
`deploy/model.onnx` on 2026-09-20), a 10-second random-input run measured (CPU, ONNX Runtime,
`intra_op_num_threads=1`, dev laptop, frame 0 excluded from timing as a warm-up frame but still
included in the parity check):
- `max_abs_err` = 1.0e-6 (tolerance: < 1e-4)
- `ms_per_frame_mean` = 1.16 ms
- `ms_per_frame_p99` = 1.78 ms
- ONNX file size: 423 KB
(The round-1 `runs/vaani_full/best.pt` export measured 1.2e-6 / 1.86 ms / 3.25 ms / 424 KB on
the same laptop; the extra df/coh work is invisible next to the GRU stack.)

These are measured single-core desktop-CPU/ORT numbers from `parity_and_timing` run on the
dev laptop, not a Pi measurement -- they only establish that the model is well inside the
16 ms/hop budget in principle (unlike the latency line above, which is arithmetic, not
measured). The
embedded lead must re-run `parity_and_timing`-equivalent timing on the actual target.

A port passes when its `max_abs_err` against the PyTorch stream reference is < 1e-4.

## Trained cascade measurement and arithmetic budget

Measured 2026-09-21 on the development laptop (Windows 11, AMD64 Family 25 Model
117, ONNX Runtime 1.30.0 CPU provider, one intra-op thread), using the same
10-second random-input protocol, 625 frames, frame zero excluded from timing:

- Maximum absolute error against batch PyTorch: **1.505e-6**, tolerance <1e-4.
- Model time: **0.999 ms mean / 1.696 ms p99** per 16 ms hop.
- Graph size: **474,599 bytes**.

These are model-only desktop measurements, excluding DSP, STFT/iSTFT and audio
I/O. They do not establish end-to-end latency or target-board feasibility.
Timing varies across runs; `tier46/trained_cascade_timing.json` records the
checkpoint/graph SHA256, environment, measurement and layer arithmetic counts.

| component | total parameters | trainable during refiner training | matrix MAC/frame | matrix MMAC/s at 62.5 frames/s |
|---|---:|---:|---:|---:|
| Frozen r4_ctl first stage | 50,249 | 0 | 686,112 | 42.882 |
| Residual refiner | 2,498 | 2,498 | 633,248 | 39.578 |
| Cascade | 52,747 | 2,498 | 1,319,360 | 82.460 |

MACs are a derived dense Conv/Linear/GRU matrix-operation estimate from one
streaming call, including fixed ERB transforms and padded positions. Biases,
normalization, activations, elementwise operations and DSP/STFT are excluded.
This explicit convention replaces the review's unexplained 38.7 MMAC/s estimate
for stage one; it is not a runtime measurement or an ONNX kernel instruction count.
The refiner's 16-channel 3x3 convolution runs across all 257 bins: 592,128
MAC/frame in that layer alone. Small parameter count does not mean low compute.

## INT8 quantization: measured and not adopted

`deploy/tier46/cascade.int8.onnx` exists and keeps this contract's input/output names and
static shapes, so the embedded loop can feed it unchanged. **It is not the deployment
target.** Measured on the same laptop and protocol as the section above (interleaved,
best of five 10-second runs, `deploy/tier46/int8_report.json`):

| graph | bytes | nodes | initializer bytes | ms/frame mean | ms/frame p99 |
|---|---:|---:|---:|---:|---:|
| `cascade.onnx` (FP32) | 474,599 | 1,756 | 210,108 | 0.908 | 1.212 |
| `cascade.int8.onnx` | 567,969 | 1,906 | 157,133 | 1.318 | 1.603 |
| `cascade.int8_pc.onnx` (per-channel) | 569,805 | 1,906 | 157,133 | 1.305 | 1.586 |

INT8 dynamic quantization is **19.7 % larger and 1.45x slower** here. Two structural
reasons, both specific to a model this small and worth knowing before repeating the
experiment:

- Weights are only 210 KB of a 474 KB graph; the rest is node protobuf. Quantization cut
  weight bytes to 157 KB but added 150 nodes, and the node overhead exceeded the saving.
- ORT's dynamic path has no integer kernel for `GRU`, which is most of this model's
  recurrence, so the 29 inserted `DynamicQuantizeLinear` nodes are added work on top of an
  unchanged float recurrence. Only one `MatMul` was left unquantized.

Quality cost on the frozen split is in `results_r2/optim/optimization.md`; the accuracy
question is moot for deployment given the size and latency results, but it is measured
rather than assumed because a complex-valued mask network is more phase-sensitive to
weight quantization than a magnitude-only one.

An embedded lead re-running this on a target where the weights *do* dominate the memory
budget, or on a runtime with an integer GRU kernel, should expect a different answer:
these numbers describe this graph on ORT 1.30 CPU, not quantization in general.

## Artifact generation and availability

Run from the repository root with the locally retained trained checkpoint:

```bash
uv run python -m vaani.export runs/vaani_tier46_refiner/best.pt --out deploy/tier46/cascade.onnx --seconds 10 --report-json deploy/tier46/trained_cascade_timing.json
```

`runs/` checkpoints and generated ONNX files are intentionally ignored. A clean
clone can reproduce the matrix from included CSVs, but cannot reproduce this
trained export without the checkpoint. No public checkpoint download is currently
specified. Obtain the exact checkpoint from the experiment owner and verify
SHA256 `932b086a2842eb88f4232b087fd99f8769bd102135e6b887dd6ecf38ab5aa2f6`.
The checkpoint embeds both stages; no separate anchor is required for export.
Existing `cascade_untrained.*` files are scratch identity-cascade artifacts and
must not be substituted for the trained graph.

## Golden vectors

`dsp_reference/vectors_cascade/<case>.wav` and `<case>.npz` cover the trained
cascade's DSP configuration: limiter, blocking matrix, differential-jump
controller. `config.json` binds that configuration to the checkpoint SHA256.
Match `n_hat`, `features` and limited `mix` within 1e-4, and `gate`, `burst` and
`reliability` exactly. Replay tests require both this set and the preserved
legacy r1/r2 set in `dsp_reference/vectors/`.

```bash
uv run python scripts/make_golden_vectors.py --checkpoint runs/vaani_tier46_refiner/best.pt
uv run pytest tests/test_golden_vectors.py -q
```

## Not covered here
Output crossfade/bypass on low reliability, overrun handling, and radio interfacing are the DSP/embedded leads' responsibility.

## Opt-in retraining architectures

The trained Tier 4.6 artifact above retains its original contract. New checkpoints
may set `model_cfg.channels`, `model_cfg.noise_floor`, and `refiner_cfg.hidden/past/scale`.
Widths change convolution/recurrent cache shapes; floor-enabled models append
`noise_cache` (1,2,257: power floor and initialized flag) after `coh_cache`, before
the cascade's final `refine_cache` (1,hidden,past,257). Zero every cache at a new
stream. Read the exported named dimensions rather than reusing legacy allocations.

The conditional refiner is a separate Python host runtime that advances c0 history
on skipped frames and conditionally computes c1/c2. ONNX export is always-on.
Its MAC activation estimates do not establish target-device latency or clean-speech
transparency.
