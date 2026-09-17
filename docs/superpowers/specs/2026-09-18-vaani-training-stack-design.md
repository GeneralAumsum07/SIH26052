# VAANI training stack — design spec

Date: 2026-09-18
Problem statement: SIH26052 (DRDO) — AI/ML speech enhancement for communication under
stationary, changing and impulsive noise. Targets: SNR > 15 dB, STOI > 0.85, PESQ > 2.5.
Owner: Rachit (ML lead — owns the whole training stack end-to-end).
Upstream plan: `~/.codex/plans/.../PLAN.md` (Astra). This spec covers only the ML/data
workstream; hardware, embedded runtime and listening tests belong to other team members
but their interfaces are defined here.

Deadline: 30 September 2026 (12 days from this spec).

---

## 1. Goal and scope

Deliver, from this repo alone:

1. A reproducible dataset pipeline producing **synchronised two-channel** (primary +
   reference mic) noisy/clean speech mixtures covering continuous, changing and
   impulsive noise, with speaker- and noise-disjoint splits.
2. A **DSP reference implementation** (NLMS noise estimator + impulse/reliability
   controller) in Python with golden test vectors, to be ported by the DSP lead.
3. A trained **dual-channel GTCRN-derived model (VaaniNet)** conditioned on the DSP
   features, plus every baseline in the plan's ablation matrix.
4. An **evaluation harness** reporting SNR, STOI, PESQ per noise class × input-SNR
   bucket, plus recovery time and supporting ASR word accuracy.
5. An **ONNX export** of the streaming model with a written deployment contract.

Out of scope: embedded runtime, physical recordings, listening tests, contact-sensor
experiment, quantisation (gated later step), the output crossfade/bypass logic on the
device (the DSP lead owns it; this repo only produces the input-side features).

## 2. Decisions locked during brainstorming

| Decision | Choice |
|---|---|
| Ownership | ML lead owns data + DSP reference + model + eval + export |
| Compute | Local RTX 5060 Laptop (8 GB, Blackwell sm_120), Python 3.13, one experiment at a time |
| Data ambition | Round 1: lean subset, training starts day 1. Round 2: full DNS-5 clean/noise shards streamed in the background and folded in |
| Two-channel synthesis | Blend: ~60 % `pyroomacoustics` room simulation, ~40 % parametric model, per sample |
| Controller features | Reference Python implementation lives here and is the spec for the embedded port |
| Architecture | Own package `vaani/`, GTCRN vendored as a module, YAML-per-experiment, dynamic mixing for train, frozen rendered val/test |

## 3. Repository layout

```
SIH_2026/
  pyproject.toml                # uv-managed; torch pinned to a cu128 build (sm_120)
  vaani/
    data/
      sources.py                # per-corpus adapters -> manifest rows
      manifests.py              # parquet manifest schema, hashing, split assignment
      splits.py                 # speaker/noise-disjoint 80/10/10 by hash
      rirs.py                   # pyroomacoustics RIR bank generation
      mixer.py                  # two-channel dynamic mixer (room + parametric paths)
      impulses.py               # synthetic impulse generator
      dataset.py                # torch Dataset: dynamic (train) / rendered (val,test)
    dsp/
      nlms.py                   # guarded NLMS noise estimator
      features.py               # per-frame impulse & reliability features
      controller.py             # gating logic -> adapt_gate, burst_flag, reliability
      stft.py                   # shared 512/256 Hann STFT, identical to model's
    models/
      gtcrn.py                  # vendored upstream GTCRN (unchanged)
      gtcrn_stream.py           # vendored streaming GTCRN (unchanged)
      vaani_net.py              # dual-channel + FiLM-conditioned variant
      baselines.py              # raw, NLMS-only, RNNoise wrappers
    losses.py
    train.py
    eval.py
    report.py
    export.py
  configs/
    data/                       # source lists, mixer params, bucket definitions
    exp/                        # one YAML per ablation row
  scripts/
    fetch_data.py               # resumable downloads; round-1 lean, round-2 DNS shards
    render_eval_sets.py         # seeded val/test rendering
    make_golden_vectors.py      # DSP reference test vectors
  tests/
  deploy/                       # export output: model.onnx, dsp_reference/, CONTRACT.md
  data/                         # git-ignored: raw/, manifests/, rirs/, eval/
  runs/                         # git-ignored: checkpoints, tensorboard, run.json
  docs/superpowers/specs/
```

Git: initialised locally, no remote. Commits are made by Rachit only when asked.

## 4. Data

### 4.1 Sources

| Corpus | Role | Round | Notes |
|---|---|---|---|
| LibriSpeech `train-clean-100` | Clean English speech | 1 (~20 h subset), full in 2 | CC BY 4.0 |
| Common Voice Hindi (validated) | Clean Hindi speech | 1 | CC0. Gate: keep clips with VAD-estimated SNR ≥ 30 dB |
| MAD — Military Audio Dataset | Noise (engines, rotors, gunfire, vehicles) | 1 | Record exact downloaded version and usable count; exclude clips containing speech where detectable |
| Synthetic impulses (`impulses.py`) | Impulsive noise | 1 | Exponentially decaying bursts, click trains, gated wideband noise; never depend solely on MAD gunshot count |
| DNS-5 read-speech + noise shards | Clean speech + noise | 2 | Streamed in background; component licences recorded per shard |

All audio is resampled to 16 kHz mono float32, stored as FLAC under `data/raw/<corpus>/`.
Every file gets a manifest row: `source_id, corpus, speaker_id, orig_path, duration_s,
licence, split, sha1`.

AudioSet and FSD50K are not dependencies (per plan).

### 4.2 Splits

Assigned **before** mixing. `split = hash(speaker_id or noise_source_id) mod 10 → {0-7: train, 8: val, 9: test}`.
MAD excerpts cut from the same source video share a `noise_source_id` and therefore a split.
Round-2 DNS shards are split by the same rule so folding them in cannot leak.

### 4.3 Two-channel mixer

Per training sample (dynamic, in DataLoader workers):

1. Pick a clean utterance, 1–3 noise clips, and with p=0.5 an impulse event (synthetic or MAD impulsive class).
2. **Room path (p≈0.6)** — draw an RIR set from a pre-generated bank (`data/rirs/`, ~5 000 rooms): shoebox 2.5–6 m sides, RT60 0.1–0.5 s, mouth source 2.5 cm from primary and 14 cm from reference mic (mics 12 cm apart on a rigid mount), noise sources far-field at random azimuth ≥ 0.8 m. Speech and each noise source convolve with their own RIR pair, giving physically consistent leakage and coherence.
3. **Parametric path (p≈0.4)** — reference speech = primary speech × gain U(−20, −8) dB, fractional delay U(0.1, 0.5) ms; each noise source goes through an independent random 3-tap FIR per channel with near-unity level.
4. **Common augmentations** (both paths): mic gain mismatch U(−3, +3) dB, per-channel random 1st-order tilt (±3 dB), clipping p=0.1 (primary hard-clipped at a random level), reference dropout p=0.05 (reference gain −40 dB for a random 0.2–1 s span), wind/handling low-frequency noise p=0.15.
5. **SNR** target U(−10, +15) dB, defined on the **primary** channel, speech power measured over the speech-active region (energy VAD on the clean signal). Noise is scaled to hit the target; impulse event level is drawn separately from a peak-level range and recorded as `impulse_peak_db` in sample metadata.
6. **Clean bucket** p=0.05: no noise at all (target is identity) so over-processing is penalised.
7. Output: `mix (2, T)`, `clean_primary (1, T)`, metadata dict.

Rule: two channels are never a duplicated mono mixture. A test asserts the inter-channel
speech-level difference and noise coherence are within the designed ranges.

### 4.4 Rendered evaluation sets

`scripts/render_eval_sets.py` renders `val` (~1 h) and `test` (~2 h) once with fixed
seeds into `data/eval/<split>/<bucket>/`. Buckets = noise class
{stationary, changing, impulsive, impulsive+stationary, clean} × input SNR
{−10, −5, 0, +5, +10, +15}. Each burst-bucket clip has a **twin** rendered with the
same seed and the impulse removed, for the recovery-time metric. Eval-set hash is
recorded in every `run.json`.

### 4.5 Physical test hook

`data/manifests/physical_test/README.md` defines the schema (two-channel wav, 16 kHz,
transcript, speaker, condition) so team recordings can be dropped in and evaluated by
the same `eval.py`. Not produced by this workstream.

## 5. DSP reference (`vaani/dsp/`)

Frame-synchronous with the model: 16 kHz, 512-sample Hann window, 256 hop. All
stateful classes expose `reset()` and `process_frame()` and are written in plain
NumPy so they port to C without hidden library semantics.

### 5.1 `nlms.py` — guarded NLMS noise estimator
- Time-domain NLMS, reference → primary, 64 taps, step µ = 0.05 × `adapt_gate`,
  regularisation ε = 1e-6.
- Output is the **noise estimate** `n_hat` (filter output), delivered to the model as
  a feature. It is never subtracted from the primary before the network.
- Update frozen (`adapt_gate → 0`) on high speech-presence, burst, overload or
  reference dropout; re-enabled with a linear ramp over 200 ms after the condition clears.

### 5.2 `features.py` — per-frame feature vector (order is part of the contract)
1. log energy delta (primary)
2. spectral flux (primary)
3. peak-to-RMS ratio (primary)
4. clipping fraction, primary
5. clipping fraction, reference
6. speech-presence estimate (primary/reference energy ratio, smoothed)
7–14. inter-channel magnitude-squared coherence in 8 ERB-spaced bands
15. inter-channel level difference (dB)
16. reference-dropout flag
17. NLMS health: normalised residual energy
18. `adapt_gate` (from controller, previous frame)

### 5.3 `controller.py`
- Burst detector: energy jump ≥ 12 dB over 2 frames **and** inter-channel level
  difference ≤ 3 dB (impulses hit both mics similarly; near-mouth speech does not).
  Hysteresis: hold 4 frames, release when energy returns within 6 dB of pre-burst.
- `reliability ∈ [0,1]` = product of (1 − clip fraction), (1 − dropout), coherence
  sanity, NLMS health.
- `adapt_gate` = 0 during burst / overload / dropout / high speech-presence, else ramped 1.
- Consonant safety: a synthetic test injects 20 ms wideband transients at −10 dB
  relative to voicing on the primary only and asserts `burst_flag` stays false.

### 5.4 Golden vectors
`scripts/make_golden_vectors.py` writes `deploy/dsp_reference/vectors/*.npz`
(input stereo wav → expected `n_hat`, features, gate) for the port's parity test.

## 6. Models (`vaani/models/`)

### 6.1 Vendored GTCRN
`gtcrn.py` and `gtcrn_stream.py` copied verbatim from
`github.com/Xiaobin-Rong/gtcrn` with the DNS3 pretrained checkpoint under
`vaani/models/checkpoints/`. Parity test: batch vs streaming output on the same clip
agrees to 1e-4.

### 6.2 VaaniNet
- Input spectral channels: 6 = (primary, reference, `n_hat`) × (real, imag), after
  GTCRN's ERB compression per channel.
- Conditioning: the 18-dim feature vector → Linear(18, C) → added as a per-frame bias
  (FiLM-style, shift only) to the output of the first encoder block.
- Everything downstream unchanged: grouped temporal convolutions, DPGRNN, decoder,
  complex mask.
- Mask applied to the **primary** spectrum only.
- Streaming state = GTCRN caches + GRU hidden; NLMS taps and controller hysteresis live
  outside the network. No added look-ahead.
- Initialisation: shared weights from the DNS3 checkpoint; new input projection and
  FiLM layers from scratch.
- Budget: ≤ 60 K parameters; MAC increase confined to the first encoder block.

### 6.3 Ablation matrix (all evaluated on the same frozen `test` set)

| Row | Config |
|---|---|
| raw | primary channel unprocessed |
| nlms_only | primary − NLMS output (classical) |
| rnnoise | upstream RNNoise via Python bindings, 48 kHz round-trip, labelled |
| gtcrn_pretrained | vendored DNS3 checkpoint, single channel |
| gtcrn_finetuned | same architecture fine-tuned on our mixtures, single channel |
| vaani_no_controller | VaaniNet, `adapt_gate ≡ 1`, feature vector zeroed |
| vaani_full | VaaniNet with DSP features and controller |
| vaani_full_sp | vaani_full + speech-preservation loss |

DeepFilterNet and H-GTCRN are run once as external quality references with their
sample-rate/channel/training-data differences labelled in the report. They are not
tuned.

## 7. Losses (`losses.py`)

- **Base**: the upstream GTCRN loss (compressed complex spectral MSE, magnitude
  compression p = 0.3) + SI-SNR term, weights as upstream. Used unchanged for
  `gtcrn_finetuned`, `vaani_no_controller`, `vaani_full` so comparisons are
  apples-to-apples.
- **Speech-preservation** (`vaani_full_sp` only): frame-weight multiplier of 3 on
  frames within ±150 ms of an impulse event, and on the clean bucket an additional
  L1 term between output and input spectra. Kept as a separate ablation row.

## 8. Training (`train.py`)

- One YAML in `configs/exp/` per ablation row: model, data blend, loss, schedule, seed.
- PyTorch, AMP bf16, AdamW (lr 5e-4 fine-tune / 1e-3 from-scratch layers), cosine
  schedule with 500-step warm-up, grad-clip 5, 4-second crops, batch ≈ 32 (tuned to
  fit 8 GB).
- Dynamic mixing in DataLoader workers (CPU); RIR bank pre-generated so per-sample
  cost is convolution only.
- Checkpoint on best val STOI; also keep last.
- Every run writes `runs/<name>/run.json`: git hash, config hash, data-manifest hash,
  eval-set hash, seed, torch/cuda versions, start/end time.
- TensorBoard scalars: losses, val STOI/PESQ/SI-SDR per epoch, throughput.
- Round 2: same configs with `data.blend` pointing at manifests that include DNS shards;
  warm-start from the round-1 checkpoint.

## 9. Evaluation (`eval.py`, `report.py`)

- Input: checkpoint or baseline name + eval split. Output: one CSV row per clip.
- Metrics per clip: SI-SDR; **SNR** defined explicitly as
  `10·log10(‖s‖² / ‖ŝ − s‖²)` against the clean primary reference (distortion counts
  as error; not conflated with SI-SDR); STOI; PESQ wideband (ITU-T P.862.2 via the
  `pesq` package at 16 kHz — mode and rate printed in the report header, with the
  note that P.862 is withdrawn in favour of P.863); noise class, input-SNR bucket,
  impulse flag, clip flag, dropout flag.
- **Recovery time** (burst buckets): speech-envelope error vs. the no-burst twin;
  recovery = first time after the burst at which the error stays below threshold for
  200 ms. Reported as a distribution and failure count. No short-window PESQ.
- **Word accuracy** (supporting only): `faster-whisper` small, offline, on the test
  set; WER per bucket. Not in the deployed path.
- `report.py` aggregates to the ablation-matrix markdown table, per bucket, with clip
  counts and 95 % bootstrap CIs. The "nominal envelope" (unclipped 0/+5/+10 dB) and
  "severe envelope" rows are reported separately, as the plan requires.

## 10. Export and deployment contract (`export.py`, `deploy/`)

- Streaming VaaniNet → ONNX opset 17, explicit state tensors as inputs/outputs,
  batch 1, one frame per call.
- Parity check: ONNX Runtime CPU vs PyTorch streaming ≤ 1e-4 on 10 s of test audio.
- Per-frame timing on this laptop CPU (proxy only; the embedded lead measures on the
  Pi).
- `deploy/CONTRACT.md`: 16 kHz, window 512 / hop 256, input tensor layout, feature
  vector order (§5.2), state tensor names and shapes, NLMS parameters, controller
  thresholds, golden-vector locations.
- Quantisation: not in this spec; a later, gated step.

## 11. Tests (`tests/`, pytest)

- `test_mixer.py`: achieved SNR within ±0.5 dB of target; channels not identical;
  speech-level difference and coherence in designed ranges; clean bucket is identity.
- `test_splits.py`: no speaker or noise source appears in two splits.
- `test_nlms.py`: converges on a synthetic reference-only noise case.
- `test_controller.py`: burst detected on synthetic impulse; not tripped by consonant
  transients; gate ramps back over 200 ms.
- `test_models.py`: GTCRN batch/stream parity; VaaniNet parameter count ≤ 60 K;
  output shapes.
- `test_train_smoke.py`: 2 optimisation steps on a tiny synthetic manifest, end-to-end,
  < 60 s on CPU.
- `test_export.py`: ONNX parity.

## 12. Environment

- `uv` project; Python 3.13; `torch`/`torchaudio` from the cu128 index (required for
  sm_120); `pyroomacoustics`, `pesq`, `pystoi`, `soundfile`, `pyarrow`, `onnx`,
  `onnxruntime`, `faster-whisper`, `tensorboard`, `pytest`.
- If a dependency lacks a Python 3.13 wheel, the fallback is a `uv`-managed 3.12
  interpreter for this project; this is recorded in `pyproject.toml` if it happens.

## 13. Sequencing (12 days)

| Days | Deliverable |
|---|---|
| 1–2 | Environment, scaffold, round-1 data fetch in background, manifests, splits, mixer + tests, RIR bank |
| 3 | Rendered val/test; `gtcrn_pretrained` and `raw` rows evaluated; `gtcrn_finetuned` run started |
| 4–6 | DSP reference + golden vectors; VaaniNet; `vaani_no_controller` vs `vaani_full` |
| 7–9 | DNS round-2 shards folded in; `vaani_full_sp`; full matrix incl. RNNoise/NLMS rows; external references |
| 10–11 | Export, contract, hand-off to DSP/embedded leads; report with CIs |
| 12 | Buffer / rerun anything invalidated |

Any change to the frozen model after final evaluation reruns the affected rows.

## 14. Risks and open questions

- **Torch on sm_120 / Python 3.13**: a wheel mismatch could cost a day. Mitigation:
  verify `torch.cuda.is_available()` and a matmul on day 1 before anything else; fall
  back to Python 3.12 via `uv`.
- **MAD download form/version**: the repo points to a packaged download; exact clip
  count is unknown until fetched and is recorded, not assumed.
- **DNS-5 fetch size and time**: unknown for this connection; round 2 is explicitly
  best-effort and the report states which shards made it in.
- **RNNoise Python bindings on Windows**: may need a build; if it fails, RNNoise is run
  via its CLI on WSL or dropped with a note — it is a baseline, not a deliverable.
- **Common Voice Hindi cleanliness**: the 30 dB gate may leave few clips; the report
  states the retained count.
- **Question for the team (not blocking)**: which device the DSP lead ports to first
  affects only the timing proxy, not this spec.
