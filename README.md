# VAANI - dual-mic speech enhancement (SIH26052, DRDO)

Real-time speech enhancement for a two-microphone headset: a time-domain NLMS
front end on the reference mic, a small GTCRN-derived network (VaaniNet, ~50k
params) conditioned on DSP features, and a controller that handles bursts and
reference faults. Targets: SNR gain > 15 dB, STOI > 0.85, PESQ > 2.5.

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

Manifests split by source recording (speaker / recording group), drop
byte-identical files so nothing appears in two splits, and store posix paths so
a manifest built on Windows loads on Linux.

## Train and evaluate

```bash
uv run python -m vaani.train configs/exp/vaani_full_r3_e32.yaml
uv run python -m vaani.eval --system vaani_full_r3_e32 --split test --eval-root data/eval_r2 --workers 8 --asr --asr-device cuda --dnsmos
uv run python -m vaani.report results_r2/*.csv --out results_r2/matrix.md --asr-ref results_r2/asr/clean.csv
```

`scripts/run_round.sh [1|2|3|3b|3c|3d|4]` runs a whole ablation wave and drops a
`ROUND*_DONE` marker; it resumes from `last.pt` and skips evals whose CSV exists.
Round 1 scored on `data/eval` (`results/`); every later round scores on the
frozen `data/eval_r2` test split (`results_r2/`, 2280 items) so rows stay
comparable across waves. The directory suffix names the **eval set**, not the
training round.

Per-system CSVs and `matrix.md` are tracked; eval logs, ASR transcript dumps
and run markers are ignored. Val STOI printed during training is **not**
comparable across seeds or across runs with different noise pools (val is
rendered from the run's own pool); only the test-split matrix is.

### Comparators

- `gtcrn_pretrained` / `gtcrn_finetuned` - the parent architecture.
- `nlms_only`, `raw` - DSP-only and passthrough floors.
- `deepfilternet3` - mono 48 kHz, ~45x the parameter budget. Lives in an
  isolated `.venv-dfn` (py3.11, `deepfilternet==0.5.6`, `torch==2.0.1` cpu,
  `soundfile`) because its pins conflict with the main env;
  `scripts/dfn_worker.py` loads it once per eval process.

### Where things stand

`results_r2/matrix.md` is the source of truth. As of round 3 the full model
clears STOI (0.906) on the nominal envelope but not SNR_out (~13.9 dB) or PESQ
(~2.3); training 32 epochs instead of 8 was the only ablation that moved the
numbers, so round 4 continues that recipe on the wider noise pool (all DNS
shards, Cadre, DEMAND) with a same-data control. Reference-mic ablations zero
the gain; the coherence map and noise estimate taps are inert.

## Layout

- `vaani/dsp/` - NLMS, frame features, controller; `deploy/dsp_reference/vectors/` holds
  float32 golden vectors that `tests/test_golden_vectors.py` replays
- `vaani/data/` - manifests, sources (one `scan_*` per corpus), mixer, impulse synthesis, datasets
- `vaani/models/` - VaaniNet, the GTCRN baseline, comparator wrappers
- `vaani/eval.py`, `vaani/report.py`, `vaani/metrics.py` - bucketed metrics with bootstrap CIs
- `configs/exp/` - one yaml per ablation run; `configs/data/` - corpus URLs, licences, splits
- `scripts/` - fetchers, round runner, diagnostics (`ceiling_analysis.py`, `mask_phase_probe.py`, `diag_*.py`)
- `docs/superpowers/` - design spec and implementation plans
