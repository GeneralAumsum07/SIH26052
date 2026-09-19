# VAANI - dual-mic speech enhancement (SIH26052, DRDO)

Real-time speech enhancement for a two-microphone headset: a time-domain NLMS
front end on the reference mic, a small GTCRN-derived network (VaaniNet)
conditioned on DSP features, and a controller that handles bursts and reference
faults. Targets: SNR gain > 15 dB, STOI > 0.85, PESQ > 2.5.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Torch comes from the cu128
index (Blackwell GPUs need it); CPU-only machines still install and run.

```bash
uv sync --all-extras
uv run pytest -q                      # CUDA tests skip when no GPU; VAANI_REQUIRE_CUDA=1 makes them fail instead
```

`pesq` ships as a vendored Windows wheel in `wheels/`; on Linux/macOS uv builds
it from PyPI, which needs a C compiler.

## Data and eval sets

```bash
uv run python scripts/fetch_data.py          # downloads corpora, writes data/manifests/*.parquet
uv run python scripts/render_eval_sets.py    # frozen val/test buckets under data/eval
```

Manifests split by source recording (speaker / recording group) and drop
byte-identical files so nothing appears in two splits.

## Train and evaluate

```bash
uv run python -m vaani.train configs/exp/vaani_full.yaml
uv run python -m vaani.eval --system vaani_full --split test --workers 8 --asr --asr-device cuda
uv run python -m vaani.report results/*.csv --out results/matrix.md --asr-ref results/asr/clean.csv
```

`scripts/run_eval_r2set.sh` runs the full 11-system matrix. Results for the
current checkpoints are in `results/matrix.md` (round-1 split) and
`results_r2/matrix.md` (round-2 split).

## Layout

- `vaani/dsp/` - NLMS, frame features, controller; `deploy/dsp_reference/vectors/` holds
  float32 golden vectors that `tests/test_golden_vectors.py` replays
- `vaani/data/` - manifests, mixer, impulse synthesis, datasets
- `vaani/models/` - VaaniNet and the GTCRN baseline
- `vaani/eval.py`, `vaani/report.py`, `vaani/metrics.py` - bucketed metrics with bootstrap CIs
- `docs/superpowers/` - design spec and implementation plan
