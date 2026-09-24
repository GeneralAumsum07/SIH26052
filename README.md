# VAANI — dual-mic speech enhancement

**SIH26052 · DRDO** — real-time speech enhancement for a two-microphone headset under stationary,
changing and impulsive noise.

A time-domain NLMS front end on the reference mic feeds a GTCRN-derived network and a residual
refiner. The shipping cascade has **52,747 parameters** and an estimated **82.460 matrix MMAC/s**
(see [the counting convention and deployment measurements](deploy/CONTRACT.md)). The controller
gates DSP adaptation; graceful single-channel fallback under reference faults is not implemented.
VAANI cleans the **transmitted** voice; it is not ear-side active noise cancellation (see
[What "ANC" means here](#what-anc-means-here)).

| | |
|---|---|
| **Shipping model** | r7 cascade, exported as `deploy/r7/cascade.onnx` (sha256 `e67a2c42…`) |
| **Checkpoints** | cascade `results_r2/runs/r7_e256_wr64_refiner/best.pt` (sha256 `121f0c3d…`), backbone `results_r2/runs/r7_e256_wr64/best.pt` (sha256 `0bea9818…`); full hashes in [`deploy/CONTRACT.md`](deploy/CONTRACT.md) |
| **Lineage** | trained from scratch on our own data; no external pretrained weights |
| **Size** | 50,249 (backbone) + 2,498 (refiner) = 52,747 parameters |
| **Latency** | 32 ms algorithmic (16 ms hop + 16 ms STFT lookahead); compute deadline 16 ms per hop |
| **Edge target** | NVIDIA Jetson AGX Orin 64GB (nothing measured on it yet); Raspberry Pi 5 is the development board |
| **Report targets** | SNR_out > 15 dB · STOI > 0.85 · PESQ > 2.5 |

SNR_out is the **absolute** output SNR, not the SNR improvement:

```math
\mathrm{SNR}_{\text{out}} = 10\log_{10}\frac{\lVert s\rVert^{2}}{\lVert \hat{s}-s\rVert^{2}}
```

where $`s`$ is the clean primary-mic speech and $`\hat{s}`$ the enhanced output, so any distortion
counts as error.

**Contents:** [Results](#results) · [What "ANC" means here](#what-anc-means-here) ·
[Known limitations](#known-limitations) · [Setup](#setup) · [Data and eval sets](#data-and-eval-sets) ·
[Train and evaluate](#train-and-evaluate) · [Deploy and run live](#deploy-and-run-live) ·
[Scalable model family and AGX Orin target](#scalable-model-family-and-agx-orin-target) ·
[Layout](#layout)

---

## Results

Every score below comes from **synthetic mixtures** made by the project's own mixer
(`vaani/data/mixer.py`); no real noisy recording is scored yet. eval_r2 numbers are on its current
render; eval_gen is a noise corpus never used in training. Brackets are 95 % bootstrap intervals
over evaluation items; **bold** marks a target cleared on the interval. Rows without brackets are
point estimates.

| r7 cascade | n | SNR_out (dB) | STOI | PESQ-WB |
|---|---|---|---|---|
| eval_r2 nominal (input 0/5/10 dB, no clip or fault) | 617 | 14.864 [14.565, 15.183] | **0.917** [0.911, 0.922] | 2.462 [2.412, 2.511] |
| eval_r2 full test split | 2,280 | 12.78 | 0.870 | 2.138 |
| eval_gen, registered stationary grid | 102 | 14.295 | 0.932 | 2.490 |
| eval_gen, changing grid (added to the protocol later) | 99 | 18.37 | 0.970 | 3.153 |
| loud transients, input 0/5 dB (synthetic bursts at +24 / +36 dB peak re speech RMS, plus a clipped-overload bucket) | 240 | 10.866 | 0.848 | 1.797 |
| matched no-burst control, input 0/5 dB (`fault_none`) | 80 | 12.771 | 0.882 | 2.103 |

<!-- TBD(diag): per-grid eval_gen intervals and the committed table/command that splits eval_gen by grid; clustered-bootstrap CIs for the headline rows -->

- **eval_r2 nominal:** STOI clears its target on the interval; SNR and PESQ miss at the point
  estimate.
- **eval_gen:** the pre-registered stationary grid (`results_r2/generalisation/PROTOCOL.md`)
  misses SNR and PESQ. Only the changing grid, added after registration, clears all three. The
  pooled 201-clip means (16.302 / 0.951 / 2.817) mix the two grids and are not the registered
  analysis.
- **Transients:** all three targets fail. Against the matched no-burst control at the same input
  SNRs, the bursts themselves cost 1.9 dB SNR_out, 0.034 STOI and 0.31 PESQ; the rest of the gap to
  the nominal row is the lower input SNR.
- **Per clip, not per mean:** only **35.5 %** of nominal clips meet all three targets at once,
  **24.1 %** of the full test split, and **3.8 %** of the transient clips.
  <!-- TBD(diag): committed per-clip pass-rate table and command (source: results_r2/r7/r7_e256_wr64_cascade_eval_r2.csv) -->
- **What "impulsive" means in the nominal rows.** The nominal envelope's impulsive classes (416 of
  the 617 clips) carry transients at **−6 to +12 dB peak re speech RMS** (median +3.3 dB), with no
  room path and no soft-clip. Training draws **15-45 dB** peaks with both on. The recorded impulses
  in the `recorded_impulsive` buckets are ESC-50 **household transients** (can opening, mouse
  clicks, keyboard typing, fireworks, footsteps, clock ticks), not gunfire.
- One refiner seed; the intervals describe evaluation-item variation, not training-seed
  uncertainty.

Details:

- **Lineage.** A backbone trained from scratch on our own data for 256 epochs (`r6_e256`),
  warm-restarted for one further 64-epoch cycle (`r7_e256_wr64`), with the 2,498-parameter
  residual refiner trained on top of the frozen result.
- **Operating envelope** (based on means): starts at input SNR +5 dB for changing noise and for
  impulsive noise at −6 to +12 dB peaks, and +10 dB for stationary noise. It says nothing about
  loud transients, which fail all three targets at both tested input SNRs (0 and 5 dB).
- **Against the previous candidate**, the tier46 cascade (14.753 / 0.915 / 2.473 nominal;
  16.023 / 0.950 / 2.835 pooled eval_gen), paired per clip: SNR_out **+0.110 dB [+0.061, +0.163]**
  on eval_r2 and **+0.278 dB [+0.185, +0.388]** on pooled eval_gen; PESQ **−0.012 [−0.021, −0.001]**
  and **−0.019 [−0.036, −0.002]**; STOI level. The warm restart itself added +0.095 dB to the
  backbone and +0.038 dB [+0.018, +0.060] to the cascade over `r6_e256`. r7 was scored on eval_gen
  without the checkpoint registration the protocol asks for.
- **Reference gain loss is unresolved.** At −12 dB reference gain the cascade's 6.862 dB is below
  the single-channel baseline's 8.858 dB (`gtcrn_finetuned`).
- **Measured limitations.** The controller did not improve nominal quality across three r3 seeds.
  Removing limiter/blocking DSP improved the single tested r3 ablation (earlier render), and wider
  wave-4 data did not outperform the same-data control. The fresh r6/r7 runs kept limiter and
  blocking anyway, and no r6/r7-era ablation has re-tested them.

> **Which eval render.** The eval_r2 numbers above are from the current render of eval_r2, re-made
> from the crest-audit relabelled manifests (EVALSET_HASH `17a9414959bb` on Windows, `aa96a28a9955`
> on Linux: the same audio, the hash digests float text), which every score in
> [results_r2/r6/](results_r2/r6/) and [results_r2/r7/](results_r2/r7/) uses. The
> [matrix](results_r2/matrix.md), which separates point estimates from interval-supported passes,
> was measured on the earlier frozen render and is kept for its within-render ablation comparisons.
> The two renders differ at 606 of the 617 nominal items, so scores are comparable within one
> render and not across them.

Further reading:

- [Requirements traceability](docs/requirements-traceability.md) maps every clause of SIH26052 to
  the file and measurement that answers it, including the clauses that are not met (board
  deployment, a microphone prototype, power, radio integration, several named defence noises) and
  the two that are met with negative results (quantization, pruning).
- [Corpus licences](docs/licences.md) are generated from the manifests: three corpora in the
  deployed recipe are non-commercial and three more have unresolved terms, which constrains a
  transfer claim but not the research result.

---

## What "ANC" means here

The problem statement's title says "adaptive noise cancellation (ANC)". VAANI reads that as
**transmit-path speech enhancement**: it cleans the wearer's voice before it is sent, which is what
the statement's SNR, STOI and PESQ targets and its mask-estimation clause measure. It does **not**
cancel noise at the wearer's ear.

Ear-side ANC would be a separate subsystem, and none of it exists here:

- a reference mic outside the earcup and an error mic inside it, placed for the acoustic path;
- identification of the secondary path (speaker to error mic), and tracking it as the fit changes;
- an FxLMS-family controller running at a far lower latency than a 16 ms hop;
- a closed-loop stability analysis and margins;
- acoustic measurement of the attenuation actually delivered at the ear.

Timing results for the communications model say nothing about hearing protection.

---

## Known limitations

- **No real defence recording is scored.** No real gunfire, artillery, helicopter, siren or drone
  recording is in any scored set. <!-- TBD(defence): defence-noise eval set (Zenodo gunshot test rows, Friedlander blast, MAD helicopter/vehicle, 15-45 dB peaks) and its per-category result -->
- **Every score comes from synthetic mixtures.** The two-channel physics of every eval item come
  from the same mixer that made the training data. <!-- TBD(real): first real-recording row (DNSMOS, reference-free) -->
- **The NLMS alone is worse than passthrough:** 1.115 dB SNR_out for `nlms_only` against 1.974 dB
  for raw input (`results_r2/matrix.md`, earlier render).
- **Real two-channel audio can erase speech.** A two-channel web recording whose "reference"
  channel carried the talker at primary level lost 79 % of its active frames through r7; with the
  reference zeroed it lost 14 %. The *inferred* cause is that r7 relies on the level difference
  between the mics to tell speech from noise; this is under investigation.
  <!-- TBD(refvalid): confirmed diagnosis with clean-reference tests, and the reference-validity Mini result -->
- **Reference failure loses to a mono model:** at −12 dB reference gain the cascade scores
  6.862 dB against 8.858 dB for the single-channel `gtcrn_finetuned`.
- **The demo loop plays the wearer's own voice back to them.** `scripts/capture_loop.py` sends the
  enhanced primary to both ears; it should go to a far-end or radio path.
  <!-- TBD(runtime): output routing fix -->
- **The eval_r2 test split informed development.** r7 was launched after r6 scored 14.83 dB on it
  (`configs/retraining/r7_e256_wr64.yaml`), so it is not an untouched held-out set.

---

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Torch comes from the cu128 index (Blackwell GPUs
need it); CPU-only machines still install and run.

```bash
uv sync --all-extras
uv run pytest -q     # CUDA tests skip when no GPU; VAANI_REQUIRE_CUDA=1 makes them fail instead
```

- `pesq` ships as a vendored Windows wheel in `wheels/`; on Linux/macOS uv builds it from PyPI,
  which needs a C compiler.
- For a rented GPU host, `scripts/remote_setup.sh` does the sync; code travels as a `git bundle`
  and data as rsync (the host is not persistent).
- The target board does **not** use this environment — see [Deploy and run live](#deploy-and-run-live).

---

## Data and eval sets

```bash
uv run python scripts/fetch_data.py                 # downloads what it can, writes data/manifests/*.parquet
uv run python scripts/fetch_data.py --only demand   # rescan one source (a full rescan takes ~1 h)
```

`scripts/render_eval_sets.py` needs `--manifests` and `--split` (run bare, it errors), and it
refuses to overwrite a frozen set (`data/eval`, `data/eval_r2*`, `data/eval_gen`).
`scripts/render_all_eval_sets.sh` holds the original recipe for `data/eval` and `data/eval_r2`.

<!-- TBD(defence): exact eval_r2 and eval_gen render commands (manifests, split, --faults, --bank, seeds) that reproduce EVALSET_HASH 17a9414959bb / aa96a28a9955 and the eval_gen hash -->

---

### Sources

See `configs/data/round1.yaml` for URLs and licences.

| Role | Corpora |
|---|---|
| Speech | LibriSpeech, Common Voice Hindi, EARS |
| Noise | ESC-50, NOISEX-92, MAD, DNS-5 noise shards, the Zenodo 7004819 gunshot set (Kabealo et al. 2023) and DroneAudioDataset |
| Two-mic noise | **DEMAND** — a 16-mic grid; channels 1 and 9 are 11.9 cm apart, matching the rig's 12 cm spacing, so their stereo rows are used verbatim on both mics. Not in the r7 recipe |
| Gunshots | **Cadre Forensics** — Zoom H4N stereo, NIJ 2016-DN-BX-0183, registration required. Not in the r7 recipe, and never rendered for evaluation |

Two of them need helper scripts because the hosts sit behind a login or serve odd rates:

```bash
scripts/fetch_cadre.sh     # Box shared links from the Cadre download page (log in first)
scripts/fetch_demand.sh    # Zenodo 1227121; SCAFE only exists at 48 kHz and is resampled at scan time
```

---

### Artillery and gunshot transients are synthesised, deliberately

The public artillery recordings we could obtain are YouTube-sourced. `scripts/crest_audit.py`
measures MAD's `shelling` class at **12.3 dB** event crest, against **13 dB** for ordinary speech:
after loudness normalisation and lossy coding the transient is simply gone, so `MAD_CLASS_MAP`
labels it `changing` rather than `impulsive`.

Impulsive training material is therefore generated from the **Friedlander blast wave**
(`vaani/data/blast.py`):

```math
p(t) = P_0 \left(1 - \frac{t}{T}\right) e^{-t/T}, \qquad t \ge 0
```

| Symbol | Meaning |
|---|---|
| $`P_0`$ | peak overpressure, reached effectively instantaneously |
| $`T`$ | positive-phase duration: the zero crossing at $`t = T`$ is followed by a negative (rarefaction) phase. Roughly 0.15–0.6 ms for small arms at close range, several ms for artillery |

On top of that waveform:

1. **Oversampling.** It is synthesised at 192 kHz and then decimated, so the near-instantaneous
   rise does not alias at 16 kHz.
2. **Ground reflection.** An inverted, attenuated copy arrives 1–9 ms later, giving real gunshot
   recordings their characteristic doublet.
3. **Distance.** A distance-dependent first-order low-pass: high frequencies are absorbed faster,
   so a distant shot is duller and has a lower crest.

The result measures **~26.5 dB** event crest, against **15.2 dB** for the previous synthetic burst.
Every impulse corpus must pass the crest gate before an adapter is written for it. See
[requirements traceability §3.1](docs/requirements-traceability.md).

The r7 recipe draws impulses from `blast, blast, burst, click_train` at 15-45 dB peaks, with the
room path and soft-clip on. The blast generator is used in training only: no eval set contains it
yet. <!-- TBD(defence): blast bucket in the defence eval set -->

---

### Splits

Manifests split by source recording (speaker / recording group), drop byte-identical files so
nothing appears in two splits, and store posix paths so a manifest built on Windows loads on Linux.

---

## Train and evaluate

```bash
uv run python -m vaani.train configs/exp/vaani_full_r3_e32.yaml
uv run python -m vaani.eval --system vaani_full_r3_e32 --split test --eval-root data/eval_r2 \
    --workers 8 --asr --asr-device cuda --dnsmos
uv run python -m vaani.report "results_r2/*.csv" --out results_r2/matrix.md \
    --asr-ref results_r2/asr/clean.csv --protocol results_r2/tier46_v2/anchor.json
```

- `scripts/run_round.sh [1|2|3|3b|3c|3d|4]` runs a whole ablation wave and drops a `ROUND*_DONE`
  marker; it resumes from `last.pt` and skips evals whose CSV exists.
- Round 1 scored on `data/eval` (`results/`); every later round scores on the frozen
  `data/eval_r2` test split (`results_r2/`, 2280 items) so rows stay comparable across waves.
  **The directory suffix names the eval set, not the training round.**
- Per-system CSVs, `matrix.md` and the clean test ASR reference are versioned report inputs. The
  report records the ASR reference hash, excludes `.partial*.csv` snapshots and rejects duplicate
  evaluation/reference keys. Eval logs, other ASR dumps and run markers are ignored.
- WER appears in `matrix.md` for systems scored on the earlier render only. The r7 CSVs have an
  empty `asr_text` column, so no WER is reported for r7.
- Val STOI printed during training is **not** comparable across seeds or across runs with different
  noise pools (val is rendered from the run's own pool); only the test-split matrix is.

---

### Loss

`HybridLoss` (`vaani/losses.py`) works on power-law **compressed** spectra. Writing
$`S = |S|\,e^{j\angle S}`$ for a clean STFT and $`\hat{S}`$ for the estimate:

```math
\tilde{S} = |S|^{p}\, e^{j\angle S}
```

```math
\mathcal{L} \;=\;
w_c\Big[\mathrm{MSE}\big(\mathrm{Re}\hat{\tilde S},\mathrm{Re}\tilde S\big)
      + \mathrm{MSE}\big(\mathrm{Im}\hat{\tilde S},\mathrm{Im}\tilde S\big)\Big]
\;+\; w_m\,\mathrm{MSE}\big(|\hat S|^{p},\,|S|^{p}\big)
\;+\; \mathcal{L}_{\text{SI-SNR}}
\;-\; w_{\text{snr}}\,\min\!\big(\mathrm{SNR}_{\text{out}},\,30\ \mathrm{dB}\big)
```

- **Compressed-magnitude term.** Compressing by $`|S|^p`$ before the spectral MSE is a
  **perceptual weighting**: power-law compression approximates the compressive loudness response of
  human hearing, which is why it is the standard spectral loss in the DNS Challenge baselines and in
  GTCRN. $`p = 0.5`$ here, against the upstream default of 0.3, weights quiet spectral detail more
  heavily. It is not a PESQ or PMSQE surrogate.
- **Absolute-SNR term.** SI-SNR is scale-blind and the target is absolute, so the absolute
  $`\mathrm{SNR}_{\text{out}}`$ is added, clamped at 30 dB.
- **Weights.** The r6/r7 configs use $`w_c = 50`$, $`w_m = 50`$, $`p = 0.5`$,
  $`w_{\text{snr}} = 0.2`$.
- **Speech-preservation variant** (`SpeechPreservationLoss`, not used by r7). Adds an L1 term to
  the clean target on clean-bucket items.

---

### Optimization (ONNX, INT8, pruning)

```bash
scripts/run_optimization.sh          # ~1.5 h: quantize, prune, evaluate each on the frozen split
```

Writes `results_r2/optim/optimization.md`. Both INT8 dynamic quantization and global magnitude
pruning were implemented, measured on the tier46 cascade and **rejected on the evidence**. At
52,747 parameters the graph's bytes are mostly node protobuf rather than weights, so INT8 makes the
file larger (+19.7 %) and slower (×1.45; `deploy/tier46/int8_report.json`), and there is too
little learned capacity for pruning to give up.

`vaani.eval --system onnx:<graph>@<ckpt>` scores an exported graph directly, so an optimized export
is measured in SNR/STOI/PESQ rather than only in bytes.

---

### Comparators

| System | What it is |
|---|---|
| `gtcrn_pretrained` / `gtcrn_finetuned` | the parent architecture |
| `nlms_only`, `raw` | DSP-only and passthrough floors |
| `deepfilternet3` | mono 48 kHz, ~45× the parameter budget. Lives in an isolated `.venv-dfn` (py3.11, `deepfilternet==0.5.6`, `torch==2.0.1` cpu, `soundfile`) because its pins conflict with the main env; `scripts/dfn_worker.py` loads it once per eval process |

The external baselines were scored on the earlier render only, and `gtcrn_finetuned` had 12+6
training epochs against 256+64 for VAANI. DeepFilterNet3 beats VAANI on DNSMOS (OVRL 2.851 against
2.702 for tier46, `results_r2/matrix.md`), while tier46 leads on SNR_out, STOI and PESQ there.

---

## Deploy and run live

The board runs the exported ONNX graph with **no deep-learning framework**: numpy, onnxruntime and
numba only ([`requirements-deploy.txt`](requirements-deploy.txt)).

```bash
pip install -r requirements-deploy.txt                     # on the board (aarch64); not requirements.txt
python scripts/board_timing.py deploy/r7/cascade.onnx --seconds 30 --out deploy/board_timing.json
```

`board_timing.py`'s `total_ms_p99` adds a whole-clip DSP average to the model's p99 and times no
STFT, iSTFT, resampling or I/O, so it is not a per-hop cost.
<!-- TBD(runtime): complete-hop benchmark command (scripts/hop_benchmark.py) and where its result lives -->
<!-- TBD(hygiene): Pi 5 venv step for PEP 668 and pinned aarch64 requirements (deploy/PI_SETUP.md) -->

`scripts/capture_loop.py` runs the whole system live, one 16 ms hop at a time. It captures both mics
through ALSA as one interleaved stream on one clock, runs `vaani.live.StreamEngine`, and currently
plays the enhanced primary to both ears. That is the wearer's own voice, so for a real demo it
belongs on a far-end or radio path (see [Known limitations](#known-limitations)):

```bash
# live: ICS-43434 pair on the Pi's I2S bus, USB sound card for the headphones
python scripts/capture_loop.py --device hw:0,0 --out-device plughw:1,0

# no microphones yet: feed a recorded pair through the ALSA loopback card
python scripts/capture_loop.py --device hw:Loopback,1,0 --format S16_LE --out-device plughw:1,0

# any OS, no audio hardware: the same block-by-block code path on a WAV file
python scripts/capture_loop.py --in-wav mix.wav --out-wav enhanced.wav
```

- **Enter** toggles enhanced ↔ bypass for A/B demos. `--record-dir` keeps the capture and the
  output so a live session can be scored afterwards.
- **numba is required on the board.** Without it the NLMS runs as the pure-Python reference, which
  alone costs more than the 16 ms hop on a loaded laptop, so the live loop refuses to start unless
  forced with `--allow-slow-dsp`.
- The 48 kHz path adds 4 ms of resampler group delay on top of the 32 ms algorithmic latency,
  before audio buffering and compute. End-to-end mic-to-ear latency has not been measured.
- `deploy/r7/model_config.json` carries the DSP configuration the weights were trained behind,
  because the graph does not encode it and the board cannot read a checkpoint without torch.
- `tests/test_live.py` holds the streaming engine to the offline eval path within 1e-5, so a live
  run is the same system the eval scores.
- The per-frame contract (inputs, cache shapes, order of operations) is in
  [`deploy/CONTRACT.md`](deploy/CONTRACT.md).

---

## Scalable model family and AGX Orin target

On 24 Sep 2026 the deployment target was confirmed as the **NVIDIA Jetson AGX Orin 64GB**, the
board the problem statement names ("Jetson AGX Orin or similar"). There is no Orin hardware access,
and nothing has been measured on one. The **Raspberry Pi 5** is the development board on which
the r7 graph has run.

**The chosen family is VaaniFE**, a dual-mic RNNFormer-style network adapted from FastEnhancer.
Only its smallest tier, the **Pi-Mini, is trained**, in the r8 retrain (not yet run). The Orin
tiers are **projections**: architecture definitions with parameter and MAC counts, graph checks on
untrained exports and laptop timing. No Orin tier is trained, and none carries a latency, power or
quality claim.

| Tier | Status | C1/C2/F/K/L | Params | MMAC/s | ONNX nodes (folded) | Laptop ORT CPU mean / p99 (ms) |
|---|---|---|---:|---:|---:|---:|
| Pi-Mini | trained in r8 (pending) | 32/24/16/2/1 | TBD | TBD | TBD | TBD |
| Orin-Mid | projection, not trained | 48/40/32/3/2 | TBD | TBD | TBD | TBD |
| Orin-Large | projection, not trained | 80/64/48/4/2 | TBD | TBD | TBD | TBD |
| Orin-Large+ | projection, not trained | 96/72/48/4/3 | TBD | TBD | TBD | TBD |
| r7 cascade (shipping) | trained | GTCRN C16 + refiner | 52,747 | 82.460 | – | 1.034 / 1.732 |

<!-- TBD(fe): committed tier table (params, MMAC/s, FP32 state bytes, folded node count, laptop ORT CPU 1-thread mean/p99) from scripts/fe_tiers.py with its results_r2/fe_tiers/ JSON and command; knob definitions from configs/arch/ -->
<!-- TBD(fe): G2 graph-gate results per tier from scripts/graph_gate.py (folded nodes < 250, 0 loops, 0 ScatterND, static shapes, layout ops < 30 %, Gemm-cell GRUs, ORT-vs-torch parity <= 1e-5) -->

Why VaaniFE rather than a wider r7:

- **Width scaling of r7 is dispatch-bound on a CPU.** Its folded graph has the same node count at
  every width, so MACs grow much faster than time. The r7 width profiles below were instantiated
  and counted as **measurements only**; they are not the scaling path, and only C16 (r7's own
  shape) is trained.
- **Graph rules for the GPU path.** VaaniFE exports one-step GRUs as Gemm cells, has no
  frequency-axis recurrence, and uses Slice+Concat caches (no ScatterND) with static shapes.
  TensorRT lowers ONNX GRU nodes to loops, which block CUDA-graph capture; r7's graph has 14 GRU
  and 18 ScatterND nodes.
- **Quality scaling is untested on our data.** The only evidence that quality grows with tier size
  is FastEnhancer's published curve on VoiceBank-DEMAND. Whether the larger tiers help on defence
  noise is unknown without training one.
- **Fallback.** If the VaaniFE Pi-Mini fails its gates, the reference-validity r7-shaped Mini
  (C16) is the fallback, and r7 stays the shipping control either way. The Orin projection would
  then have no family validated at Mini size.

r7 width profiles (untrained except C16; measurements, not the chosen family):

| Profile | Total entries | Learned entries | Matrix MMAC/s | FP32 explicit state (bytes) |
|---|---:|---:|---:|---:|
| C16 (r7, trained) | 52,747 | 28,171 | 82.460 | 115,368 |
| C32 (untrained) | 109,883 | 85,307 | 147.278 | 187,560 |
| C64 (untrained) | 318,619 | 294,043 | 379.586 | 331,944 |
| C96 (untrained) | 653,307 | 628,731 | 748.790 | 476,328 |

- "Total entries" includes the fixed ERB matrices (24,576 entries), which are not learned. MACs
  count dense Conv/Linear/GRU matrix work with padding, using the same convention as
  `deploy/CONTRACT.md`; they exclude activations, normalization, elementwise ops, DSP and FFT. State
  plus weights is not total runtime RAM.
- The refiner is 2,498 entries but 39.578 of the 82.460 MMAC/s, because it runs densely over all
  257 bins. A small parameter count does not mean small compute.
- **Laptop model timing (r7).** The trained r7 graph on ONNX Runtime 1.30.0, CPUExecutionProvider,
  one thread, on the Windows development laptop (AMD64 Family 25 Model 117), 100 warm-up plus 2,000
  timed hops on synthetic inputs with carried state: **mean 1.034 ms, median 0.981 ms, p99
  1.732 ms, max 2.447 ms** per 16 ms hop. This is model-only (no DSP, FFT or audio I/O), and it is
  neither real-audio validation nor a Pi or Orin result. The cloud-x86 figure in
  `deploy/r7/cascade_parity_timing.json` (1.135 ms mean, ORT 1.25) is a different machine and
  software stack. <!-- TBD(refvalid): committed source (scripts/audit_budget.py output under results_r2/r8/) for the r7 width-profile counts and this laptop timing; the original profile JSON is local-only -->
- <!-- TBD(runtime): complete-hop benchmark (DSP + STFT + model + iSTFT + resampling) per profile from scripts/hop_benchmark.py, and the backend API --> The complete-hop cost has not been measured yet.
- <!-- TBD(refvalid): budget audit of the reference-validity Mini against the proposed ceiling of 60,000 total entries and 90.706 matrix MMAC/s (scripts/audit_budget.py) --> The reference-validity Mini's budget has not been audited yet.

The claim this supports, and no more:

> Our current trained baseline has 52,747 total parameter entries and a stateful streaming
> implementation. The next Mini, the smallest tier of the VaaniFE family, validates sensor-failure
> handling and the deployment contract. AGX Orin 64GB is the selected deployment target. Larger
> tiers are projections from counts, graph checks and laptop timing; no Orin latency, power or
> larger-model quality is measured, and no larger model is trained.

Every Orin figure in this repository is a projection or a laptop measurement. More compute does not
by itself fix the reference-fault and transient failures above.

---

## Layout

| Path | Contents |
|---|---|
| `vaani/dsp/` | NLMS, frame features, controller; `deploy/dsp_reference/vectors/` holds float32 golden vectors that `tests/test_golden_vectors.py` replays |
| `vaani/data/` | manifests, sources (one `scan_*` per corpus), mixer, impulse synthesis, datasets |
| `vaani/models/` | VaaniNet, the GTCRN baseline, comparator wrappers |
| `vaani/eval.py`, `vaani/report.py`, `vaani/metrics.py` | bucketed metrics with bootstrap CIs |
| `vaani/export.py`, `vaani/quantize.py`, `vaani/prune.py` | streaming ONNX export, INT8 dynamic quantization and magnitude pruning, each with its own measurement report |
| `vaani/live.py` | the streaming runtime: DSP front end, ONNX graph and overlap-add, one hop at a time |
| `deploy/r7/` | the shipping graph, its DSP config and its parity/timing record |
| `results_r2/runs/r7_e256_wr64*/` | the r7 backbone and cascade checkpoints the shipping graph was exported from |
| `results_r2/r7/` | r7 per-clip result CSVs on eval_r2 and eval_gen |
| `configs/exp/`, `configs/retraining/` | one yaml per run |
| `configs/data/` | corpus URLs, licences, splits |
| `scripts/` | fetchers, round runner, live capture loop, board timing, diagnostics (`ceiling_analysis.py`, `mask_phase_probe.py`, `diag_*.py`) |
| `docs/` | requirements traceability, corpus licences, physical test schema |
