# VAANI — dual-mic speech enhancement

**SIH26052 · DRDO** — real-time speech enhancement for a two-microphone headset under stationary,
changing and impulsive noise.

A time-domain NLMS front end on the reference mic feeds a GTCRN-derived network and a residual
refiner. The shipping cascade has **52,747 parameters** and an estimated **82.460 matrix MMAC/s**
(see [the counting convention and deployment measurements](deploy/CONTRACT.md)). The controller
gates DSP adaptation; graceful single-channel fallback under reference faults is not implemented.

| | |
|---|---|
| **Shipping model** | r7 cascade — `runs/r7_e256_wr64_refiner/best.pt`, exported as `deploy/r7/cascade.onnx` |
| **Lineage** | trained from scratch on our own data; no external pretrained weights |
| **Size** | 50,249 (backbone) + 2,498 (refiner) = 52,747 parameters |
| **Latency** | 32 ms algorithmic (16 ms hop + 16 ms STFT lookahead) |
| **Report targets** | SNR_out > 15 dB · STOI > 0.85 · PESQ > 2.5 |

SNR_out is the **absolute** output SNR, not the SNR improvement:

```math
\mathrm{SNR}_{\text{out}} = 10\log_{10}\frac{\lVert s\rVert^{2}}{\lVert \hat{s}-s\rVert^{2}}
```

where $`s`$ is the clean primary-mic speech and $`\hat{s}`$ the enhanced output, so any distortion
counts as error.

**Contents:** [Results](#results) · [Setup](#setup) · [Data and eval sets](#data-and-eval-sets) ·
[Train and evaluate](#train-and-evaluate) · [Deploy and run live](#deploy-and-run-live) ·
[Layout](#layout)

---

## Results

All numbers are on the current render of eval_r2 (nominal envelope, 617 clips) and on eval_gen (a
noise corpus never used in training, 201 nominal clips). Brackets are 95 % bootstrap intervals over
evaluation items; **bold** marks a target cleared on the interval.

| r7 cascade | SNR_out (dB) | STOI | PESQ-WB |
|---|---|---|---|
| eval_r2 | 14.864 [14.565, 15.183] | **0.917** [0.911, 0.922] | 2.462 [2.412, 2.511] |
| eval_gen | **16.302** [15.732, 16.934] | **0.951** [0.944, 0.957] | **2.817** [2.724, 2.915] |
| transient-present (240 clips) | 10.866 | 0.848 | 1.797 |

On eval_r2, STOI clears its target on the interval; SNR and PESQ miss at the point estimate. On
eval_gen all three clear on the interval. On transient-present clips all three fail. This is one
refiner seed; the intervals describe evaluation-item variation, not training-seed uncertainty.

Details:

- **Lineage.** A backbone trained from scratch on our own data for 256 epochs (`r6_e256`),
  warm-restarted for one further 64-epoch cycle (`r7_e256_wr64`), with the 2,498-parameter
  residual refiner trained on top of the frozen result.
- **Operating envelope** (based on means): starts at input SNR +5 dB for changing/impulsive noise
  and +10 dB for stationary noise.
- **Against the previous candidate**, the tier46 cascade (14.753 / 0.915 / 2.473 nominal;
  16.023 / 0.950 / 2.835 on eval_gen), paired per clip: SNR_out **+0.110 dB [+0.061, +0.163]** on
  eval_r2 and **+0.278 dB [+0.185, +0.388]** on eval_gen; PESQ **−0.012 [−0.021, −0.001]** and
  **−0.019 [−0.036, −0.002]**; STOI level. The warm restart itself added +0.095 dB to the backbone
  and +0.038 dB [+0.018, +0.060] to the cascade over `r6_e256`.
- **Reference gain loss is unresolved.** At −12 dB reference gain the cascade's 6.862 dB is below
  the single-channel baseline's 8.858 dB (`gtcrn_finetuned`).
- **Measured limitations.** The controller did not improve nominal quality across three r3 seeds;
  removing limiter/blocking DSP improved the single tested ablation, and wider wave-4 data did not
  outperform the same-data control. These are measured limitations, not grounds to remove
  components from an already-trained checkpoint.

> **Which eval render.** The numbers above are from the current render of eval_r2, re-made from the
> crest-audit relabelled manifests (EVALSET_HASH `17a9414959bb` on Windows, `aa96a28a9955` on Linux:
> the same audio, the hash digests float text), which every score in
> [results_r2/r6/](results_r2/r6/) uses. The [matrix](results_r2/matrix.md), which separates point
> estimates from interval-supported passes, was measured on the earlier frozen render and is kept for
> its within-render ablation comparisons. The two renders differ at 606 of the 617 nominal items, so
> scores are comparable within one render and not across them.

Further reading:

- [Requirements traceability](docs/requirements-traceability.md) maps every clause of SIH26052 to
  the file and measurement that answers it, including the two clauses that are not met (board
  deployment and a microphone prototype) and the two that are met with negative results
  (quantization, pruning).
- [Corpus licences](docs/licences.md) are generated from the manifests: three corpora in the
  deployed recipe are non-commercial and three more have unresolved terms, which constrains a
  transfer claim but not the research result.

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
uv run python scripts/render_eval_sets.py           # frozen val/test buckets under data/eval*
```

### Sources

See `configs/data/round1.yaml` for URLs and licences.

| Role | Corpora |
|---|---|
| Speech | LibriSpeech, Common Voice Hindi, EARS |
| Noise | ESC-50, NOISEX-92, MAD, DNS-5 noise shards, the Kaggle gunshot and drone sets |
| Two-mic noise | **DEMAND** — a 16-mic grid; channels 1 and 9 are 11.9 cm apart, matching the rig's 12 cm spacing, so their stereo rows are used verbatim on both mics |
| Gunshots | **Cadre Forensics** — Zoom H4N stereo, NIJ 2016-DN-BX-0183, registration required |

Two of them need helper scripts because the hosts sit behind a login or serve odd rates:

```bash
scripts/fetch_cadre.sh     # Box shared links from the Cadre download page (log in first)
scripts/fetch_demand.sh    # Zenodo 1227121; SCAFE only exists at 48 kHz and is resampled at scan time
```

### Artillery and gunshot transients are synthesised, deliberately

The public artillery recordings we could obtain are YouTube-sourced. `scripts/crest_audit.py`
measures MAD's `shelling` class at **12.3 dB** event crest, against **13 dB** for ordinary speech:
after loudness normalisation and lossy coding the transient is simply gone, so `MAD_CLASS_MAP`
labels it `changing` rather than `impulsive`.

Impulsive material is therefore generated from the **Friedlander blast wave**
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
- Val STOI printed during training is **not** comparable across seeds or across runs with different
  noise pools (val is rendered from the run's own pool); only the test-split matrix is.

### Loss

`HybridLoss` (`vaani/losses.py`) works on power-law **compressed** spectra. Writing
$`S = |S|\,e^{j\angle S}`$ for a clean STFT and $`\hat{S}`$ for the estimate:

```math
\tilde{S} = |S|^{p}\, e^{j\angle S}
```

```math
\mathcal{L} \;=\;
w_c\Big[\mathrm{MSE}\big(\operatorname{Re}\hat{\tilde S},\operatorname{Re}\tilde S\big)
      + \mathrm{MSE}\big(\operatorname{Im}\hat{\tilde S},\operatorname{Im}\tilde S\big)\Big]
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
- **Speech-preservation variant.** Adds an L1 term to the clean target on clean-bucket items.

### Optimization (ONNX, INT8, pruning)

```bash
scripts/run_optimization.sh          # ~1.5 h: quantize, prune, evaluate each on the frozen split
```

Writes `results_r2/optim/optimization.md`. Both INT8 dynamic quantization and global magnitude
pruning were implemented, measured and **rejected on the evidence**. At 52,747 parameters the
graph's bytes are mostly node protobuf rather than weights, so INT8 makes the file larger (+19.7 %)
and slower (×1.45), and there is too little learned capacity for pruning to give up.

`vaani.eval --system onnx:<graph>@<ckpt>` scores an exported graph directly, so an optimized export
is measured in SNR/STOI/PESQ rather than only in bytes.

### Comparators

| System | What it is |
|---|---|
| `gtcrn_pretrained` / `gtcrn_finetuned` | the parent architecture |
| `nlms_only`, `raw` | DSP-only and passthrough floors |
| `deepfilternet3` | mono 48 kHz, ~45× the parameter budget. Lives in an isolated `.venv-dfn` (py3.11, `deepfilternet==0.5.6`, `torch==2.0.1` cpu, `soundfile`) because its pins conflict with the main env; `scripts/dfn_worker.py` loads it once per eval process |

---

## Deploy and run live

The board runs the exported ONNX graph with **no deep-learning framework**: numpy, onnxruntime and
numba only ([`requirements-deploy.txt`](requirements-deploy.txt)).

```bash
pip install -r requirements-deploy.txt                     # on the board (aarch64); not requirements.txt
python scripts/board_timing.py deploy/r7/cascade.onnx --seconds 30 --out deploy/board_timing.json
```

`scripts/capture_loop.py` runs the whole system live, one 16 ms hop at a time. It captures both mics
through ALSA as one interleaved stream on one clock, runs `vaani.live.StreamEngine`, and plays the
enhanced primary to both ears:

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
- **numba is required on the board.** Without it the NLMS runs as the pure-Python reference at
  ~10.5 ms of the 16 ms hop on a desktop core (0.07 ms with numba), so the live loop refuses to
  start unless forced with `--allow-slow-dsp`.
- `deploy/r7/model_config.json` carries the DSP configuration the weights were trained behind,
  because the graph does not encode it and the board cannot read a checkpoint without torch.
- `tests/test_live.py` holds the streaming engine to the offline eval path within 1e-5, so a live
  run is the same system the eval scores.
- The per-frame contract (inputs, cache shapes, order of operations) is in
  [`deploy/CONTRACT.md`](deploy/CONTRACT.md).

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
| `configs/exp/`, `configs/retraining/` | one yaml per run |
| `configs/data/` | corpus URLs, licences, splits |
| `scripts/` | fetchers, round runner, live capture loop, board timing, diagnostics (`ceiling_analysis.py`, `mask_phase_probe.py`, `diag_*.py`) |
| `docs/` | requirements traceability, corpus licences, physical test schema |
