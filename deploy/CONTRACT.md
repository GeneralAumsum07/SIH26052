# VAANI deployment contract (v1)

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
5. iSTFT overlap-add with the same sqrt-Hann window, on `spec_out`.

For the *no-controller* configuration: gate = 1, feats = zeros.

## Feature order
log_energy_delta, spectral_flux, peak_to_rms, clip_frac_primary, clip_frac_reference, speech_presence, coherence_b0..b7, level_diff_db, ref_dropout, nlms_health, prev_gate

(18 total; exact order and definitions in `vaani/dsp/features.py::FEATURE_NAMES`.)

## Cache shapes and zero-init
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

## Export and parity
`vaani/export.py::export(ckpt_path, out_path)` builds the streaming twin (`StreamVaaniNet`),
loads weights via `convert_to_stream` (a plain `load_state_dict` does not work -- the stream
conv wrappers nest keys one level deeper), and traces one frame with
`torch.onnx.export(..., opset_version=17, dynamo=False)` at the shapes above (batch 1, static).

`vaani/export.py::parity_and_timing(ckpt_path, onnx_path, seconds)` runs the ONNX Runtime
session frame-by-frame (CPU, `intra_op_num_threads=1`) against the batch `VaaniNet` doing the
same, carrying caches forward exactly as the embedded loop must. On the round-1 trained
checkpoint (`runs/vaani_full/best.pt`, exported to `deploy/model.onnx`), a 10-second
random-input run measured (CPU, ONNX Runtime, `intra_op_num_threads=1`, dev laptop, frame 0
excluded from timing as a warm-up frame but still included in the parity check):
- `max_abs_err` = 1.2e-6 (tolerance: < 1e-4)
- `ms_per_frame_mean` = 1.86 ms
- `ms_per_frame_p99` = 3.25 ms
- ONNX file size: 424 KB

These are measured single-core desktop-CPU/ORT numbers from `parity_and_timing` run on the
dev laptop, not a Pi measurement -- they only establish that the model is well inside the
16 ms/hop budget in principle (unlike the latency line above, which is arithmetic, not
measured). The
embedded lead must re-run `parity_and_timing`-equivalent timing on the actual target.

A port passes when its `max_abs_err` against the PyTorch stream reference is < 1e-4.

## Golden vectors
`dsp_reference/vectors/<case>.wav` (stereo input) and `<case>.npz` (n_hat, features, gate, burst, reliability). A port passes when n_hat matches to 1e-4 and gate/burst match exactly. The current vectors are for the r1/r2 DSP (no limiter, level-diff rule); they are regenerated with the r3 `dsp` block when an r3 checkpoint becomes the shipped model, and the port then also has to match the limited `mix`.

## Not covered here
Output crossfade/bypass on low reliability, overrun handling, and radio interfacing are the DSP/embedded leads' responsibility.
