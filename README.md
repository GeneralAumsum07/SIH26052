# VAANI - dual-mic speech enhancement (SIH26052, DRDO)

Real-time speech enhancement for a two-microphone headset: a time-domain NLMS
front end on the reference mic, a GTCRN-derived network and a residual refiner.
The evaluated cascade has 52,747 parameters and an estimated 82.460 matrix
MMAC/s; see [the counting convention and deployment measurements](deploy/CONTRACT.md).
The controller gates DSP adaptation; graceful single-channel fallback under
reference faults is not implemented. Report targets: SNR_out > 15 dB, STOI > 0.85,
PESQ > 2.5. SNR_out is absolute output SNR, not SNR improvement.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Torch comes from the cu128
index (Blackwell GPUs need it); CPU-only machines still install and run.

```bash
uv sync --all-extras
uv run pytest -q                      # CUDA tests skip when no GPU; VAANI_REQUIRE_CUDA=1 makes them fail instead
```

`pesq` ships as a vendored Windows wheel in `wheels/`; on Linux/macOS uv builds
it from PyPI, which needs a C compiler.

For a rented GPU host, `scripts/remote_setup.sh` does the sync; code travels as
a `git bundle` and data as rsync (the host is not persistent).

## Data and eval sets

```bash
uv run python scripts/fetch_data.py                 # downloads what it can, writes data/manifests/*.parquet
uv run python scripts/fetch_data.py --only demand   # rescan one source (a full rescan takes ~1 h)
uv run python scripts/render_eval_sets.py           # frozen val/test buckets under data/eval*
```

Sources (see `configs/data/round1.yaml` for URLs and licences): LibriSpeech,
Common Voice Hindi, EARS, ESC-50, NOISEX-92, MAD, DNS-5 noise shards, the
Kaggle gunshot and drone sets, **DEMAND** (16-mic grid; channels 1 and 9 are
11.9 cm apart, matching the rig's 12 cm spacing, so their stereo rows are used
verbatim on both mics) and **Cadre Forensics** gunshots (Zoom H4N stereo, NIJ
2016-DN-BX-0183, registration required). Two of them need helper scripts
because the hosts sit behind a login or serve odd rates:

```bash
scripts/fetch_cadre.sh     # Box shared links from the Cadre download page (log in first)
scripts/fetch_demand.sh    # Zenodo 1227121; SCAFE only exists at 48 kHz and is resampled at scan time
```

**Artillery and gunshot transients are synthesised, deliberately.** The public
artillery recordings we could obtain are YouTube-sourced: `scripts/crest_audit.py`
measures MAD's `shelling` class at 12.3 dB event crest against 13 dB for ordinary
speech, so after loudness normalisation and lossy coding the transient is simply
gone, and `MAD_CLASS_MAP` labels it `changing` rather than `impulsive`. Impulsive
material is therefore generated from the Friedlander blast wave,
p(t) = P0(1 - t/T)e^(-t/T), oversampled at 192 kHz with a ground reflection and a
distance-dependent low-pass (`vaani/data/blast.py`); it measures ~26.5 dB event
crest against 15.2 dB for the previous synthetic burst. Every impulse corpus must
pass the crest gate before an adapter is written for it. See
[requirements traceability §3.1](docs/requirements-traceability.md).

Manifests split by source recording (speaker / recording group), drop
byte-identical files so nothing appears in two splits, and store posix paths so
a manifest built on Windows loads on Linux.

## Train and evaluate

```bash
uv run python -m vaani.train configs/exp/vaani_full_r3_e32.yaml
uv run python -m vaani.eval --system vaani_full_r3_e32 --split test --eval-root data/eval_r2 --workers 8 --asr --asr-device cuda --dnsmos
uv run python -m vaani.report "results_r2/*.csv" --out results_r2/matrix.md --asr-ref results_r2/asr/clean.csv --protocol results_r2/tier46_v2/anchor.json
```

`scripts/run_round.sh [1|2|3|3b|3c|3d|4]` runs a whole ablation wave and drops a
`ROUND*_DONE` marker; it resumes from `last.pt` and skips evals whose CSV exists.
Round 1 scored on `data/eval` (`results/`); every later round scores on the
frozen `data/eval_r2` test split (`results_r2/`, 2280 items) so rows stay
comparable across waves. The directory suffix names the **eval set**, not the
training round.

Per-system CSVs, `matrix.md` and the clean test ASR reference are versioned report
inputs. The report records the ASR reference hash, excludes `.partial*.csv`
snapshots and rejects duplicate evaluation/reference keys. Eval logs, other ASR
dumps and run markers are ignored. Val STOI printed during training is **not**
comparable across seeds or across runs with different noise pools (val is
rendered from the run's own pool); only the test-split matrix is.

### Loss

`HybridLoss` (`vaani/losses.py`) combines SI-SNR (plus an absolute-SNR term, since
SI-SNR is scale-blind and the target is absolute), L2 on the complex parts and the
magnitude, L1 to the clean target in the speech-preservation variant, and a
**perceptually weighted compressed-magnitude term**: both target and estimate are
compressed by `mag ** p` before the spectral MSE. Power-law magnitude compression
approximates the compressive loudness response of human hearing, which is why it is
the standard spectral loss in the DNS Challenge baselines and in GTCRN; `p = 0.5`
here against the upstream default of 0.3 weights quiet spectral detail more heavily.
It is a perceptual weighting, not a PESQ or PMSQE surrogate.

### Optimization (ONNX, INT8, pruning)

```bash
scripts/run_optimization.sh          # ~1.5 h: quantize, prune, evaluate each on the frozen split
```

Writes `results_r2/optim/optimization.md`. Both INT8 dynamic quantization and global
magnitude pruning were implemented, measured and **rejected on the evidence**: at
52,747 parameters the graph's bytes are mostly node protobuf rather than weights, so
INT8 makes the file larger (+19.7 %) and slower (x1.45), and there is too little
learned capacity for pruning to give up. `vaani.eval --system onnx:<graph>@<ckpt>`
scores an exported graph directly, so an optimized export is measured in SNR/STOI/PESQ
rather than only in bytes.

### Comparators

- `gtcrn_pretrained` / `gtcrn_finetuned` - the parent architecture.
- `nlms_only`, `raw` - DSP-only and passthrough floors.
- `deepfilternet3` - mono 48 kHz, ~45x the parameter budget. Lives in an
  isolated `.venv-dfn` (py3.11, `deepfilternet==0.5.6`, `torch==2.0.1` cpu,
  `soundfile`) because its pins conflict with the main env;
  `scripts/dfn_worker.py` loads it once per eval process.

### Where things stand

[Corpus licences](docs/licences.md) are generated from the manifests: three corpora in the deployed
recipe are non-commercial and three more have unresolved terms, which constrains a transfer claim but
not the research result.

[Requirements traceability](docs/requirements-traceability.md) maps every clause of
SIH26052 to the file and measurement that answers it, including the two clauses that
are not met (board deployment and a microphone prototype) and the two that are met
with negative results (quantization, pruning).

The [matrix](results_r2/matrix.md) separates point estimates from interval-supported
passes. It was measured on the earlier frozen render of eval_r2 and is kept for its
within-render ablation comparisons. The numbers below are from the current render of eval_r2, re-made from the crest-audit relabelled manifests (EVALSET_HASH `17a9414959bb` on Windows, `aa96a28a9955` on Linux: the same audio, the hash digests float text),
which every score in [results_r2/r6/](results_r2/r6/) uses. The two renders differ at 606 of the
617 nominal items, so scores are comparable within one render and not across them.

The selected cascade's operating envelope, based on means, starts at
input SNR +5 dB for changing/impulsive noise and +10 dB for stationary noise.
On the nominal envelope (617 clips) the tier46 cascade scores **14.753 dB [14.447,15.081]**,
**0.915 STOI [0.909,0.920]**, **2.473 PESQ [2.424,2.524]**, and the r6_e256 cascade (trained
from scratch) scores **14.826 dB [14.532,15.151]**, **0.916 STOI [0.911,0.922]**,
**2.447 PESQ [2.398,2.498]**. STOI clears its target on the interval; SNR and PESQ miss
at the point estimate. On the generalisation set (eval_gen, a noise corpus never used in
training, 201 nominal clips) both clear all three targets on the interval: tier46
**16.023 dB [15.431,16.655] / 0.950 / 2.835 [2.744,2.931]**, r6_e256 cascade
**16.236 dB [15.661,16.887] / 0.950 / 2.761 [2.669,2.854]**. Each is one refiner seed;
the intervals describe evaluation-item variation, not training-seed uncertainty.

On transient-present clips (240) the tier46 cascade scores **10.932 dB / 0.848 / 1.837**,
failing all three targets. Reference gain loss is also unresolved: at -12 dB reference gain,
the cascade's 6.291 dB is below the single-channel baseline's 8.858 dB (gtcrn_finetuned).

The controller did not improve nominal quality across three r3 seeds; removing
limiter/blocking DSP improved the single tested ablation, and wider wave-4 data
did not outperform the same-data control. These are measured limitations, not
grounds to remove components from an already-trained checkpoint. See the
[review resolution and deferred work](docs/adversarial-review-133-resolution.md).

## Layout

- `vaani/dsp/` - NLMS, frame features, controller; `deploy/dsp_reference/vectors/` holds
  float32 golden vectors that `tests/test_golden_vectors.py` replays
- `vaani/data/` - manifests, sources (one `scan_*` per corpus), mixer, impulse synthesis, datasets
- `vaani/models/` - VaaniNet, the GTCRN baseline, comparator wrappers
- `vaani/eval.py`, `vaani/report.py`, `vaani/metrics.py` - bucketed metrics with bootstrap CIs
- `vaani/export.py`, `vaani/quantize.py`, `vaani/prune.py` - streaming ONNX export, INT8 dynamic
  quantization and magnitude pruning, each with its own measurement report
- `configs/exp/` - one yaml per ablation run; `configs/data/` - corpus URLs, licences, splits
- `scripts/` - fetchers, round runner, diagnostics (`ceiling_analysis.py`, `mask_phase_probe.py`, `diag_*.py`)
- `docs/superpowers/` - design spec and implementation plans
