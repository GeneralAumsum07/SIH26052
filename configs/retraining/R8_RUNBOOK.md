# r8 rental runbook (plan 11.6): 2x RTX 5090, 36-48 h

Exact commands for the r8 retrain. Nothing here has been run on the rental box. The laptop smokes and measurements that
back these commands are listed at the end. **Selection happens on val only. The r8 test set is scored once (G6).
r7 stays the control: none of these configs touches the r7 path** (mixer v1, `val.select` absent, no `ema` key and
no `loss: fe`, which keep train.py's r7 behaviour bit-exact; tests/test_dataset.py and tests/test_train_smoke.py
cover this).

Open decisions that block or shape a step (Rachit):
- **TBD (Q-vCPU):** how many vCPUs the rental has. The loader is the bottleneck (r7: ~130 of each ~144 ms step was CPU).
  Two GPUs need about 2x r7's 32 vCPU (>= 64). Size it from `results_r2/r8/loader_bench.json` (items/s per worker) and
  `results_r2/r8/step_time.json` (GPU items/s): per run, workers >= GPU items/s / (items/s per worker).
- **TBD (Q3, plan 11.9):** dataset downloads not on disk: Lombard GRID, DEMAND 16-channel pairs (no pair extractor
  exists yet, so DEMAND is not in any r8 manifest), the AudioSet label CSV for the DNS speech/music filter. The configs
  run without them.
- **TBD (Q7, plan 11.9):** the final selection metric. The configs use `val.select: composite`. The alternative is
  `val.select: stoi`, which is r7's rule, still with EMA scoring.
- **TBD (Q2, plan 11.9):** NLMS in the default path. `r8_refvalid_v2` runs the fixed NLMS because the C16 model reads
  n_hat. `r8_fe_mini` (inputs `pr`) does not run it. Ablation 2 (`pr_nhat`) measures the difference.
- **TBD:** an M6 training RIR bank. The eval-only `bank_eval_r8.npz` exists; training uses `bank_r3.npz`
  (sha256 e4e67463...). Building a new M6 bank would break comparability with r7, so r8 keeps bank_r3.
- **TBD:** torch_pesq is not installed, so the FE loss's PESQ term (weight 0.001) falls back to 0 and logs it.
  Adding it is an install (pyproject), so Rachit approves it before the rental.

## 0. Environment (first 20 minutes)

```bash
tmux new -s vaani
nproc; free -g; df -h .; nvidia-smi                      # record in the run log
bash scripts/remote_setup.sh                             # uv sync --all-extras (numba included), data, banks, pack
uv run python -c "import numba, torch; print(numba.__version__, torch.__version__, torch.cuda.device_count())"
sha256sum data/rirs/bank_r3.npz                          # must be e4e67463072e1dca14b94bef97a99bb85da34ebee7ec657a1f1dc7554df1b59f
```

`remote_setup.sh` runs `uv sync --quiet --all-extras`, so numba is in the box's venv. Without numba, the NLMS runs in
pure Python and the GPU idles. See docs/box-launch.md for the MIRROR_URL, RIR_BANK_URL and Kaggle/MDC variables.

### Files the box needs beyond remote_setup.sh (sync from the laptop)

| What | Path | Why |
|---|---|---|
| v2 MAD manifest (speech-free, video-grouped) | `data/manifests/mad_v2.parquet` | every r8 config lists it. It is built by `scripts/mad_speech_filter.py` (Silero VAD). Copy the laptop file rather than rebuilding, because the VAD run is not pinned |
| Held-out exclusion | `configs/data/r8_heldout_exclude.json` (in git) | drops the r8 test groups from train/val. A missing file only **warns**, so check that it is present |
| Train RIR bank | `data/rirs/bank_r3.npz` (+ `.noise/.speech/.rt60.npy`) | fetched by sha256 in remote_setup.sh |
| Frozen val set | `data/eval_r2/val` | composite selection, G3 and G4 read it. Never re-render it |
| r7 init | `results_r2/runs/r7_e256_wr64/best.pt` | r8_refvalid_v2 warm start. train.py checks sha256 0bea9818... |
| Composite baseline | `vaani/models/checkpoints/model_trained_on_dns3.tar` | gtcrn_pretrained dSNR for the mono/web-stereo filters |
| r8 test set (G6 only) | `data/eval_r8_test/test` (hash ed024af085a2) | touch it only at step 6 |

Manifests used by every r8 config: `librispeech_100h, ears, cv_hi, esc50, dns ...freesound_000, mad_v2, gunshots`.

## 1. G1 data gate (before any run)

```bash
uv run --with numba python scripts/data_gates.py --versions 1 2 --items 48 --seed 55
```
Pass: the v2 ILD-only AUC is <= 0.75 on both paths (results_r2/r8/data_gates/). If it fails, do not start the full
runs.

Loader and step check on this box (about 2 min; it sizes the workers):
```bash
uv run --with numba python scripts/bench_loader.py --out results_r2/r8/loader_bench_box.json \
    --configs configs/retraining/r8_fe_mini.yaml configs/retraining/r8_refvalid_v2.yaml --workers 8 16 24 30 --batches 20
uv run --with numba python scripts/bench_loader.py --step-time --out results_r2/r8/step_time_box.json \
    --configs configs/retraining/r8_fe_mini.yaml configs/retraining/r8_refvalid_v2.yaml
```
Set `VAANI_WORKERS` per run to the smallest count whose loader items/s meets the step's GPU items/s. Two concurrent
runs share the cores, so each gets about half of `nproc - 2`.

Composite screening at a val point uses the frozen-screen spawn pool (SCREEN_WORKERS = 32). Cap it with
`VAANI_SCREEN_WORKERS` so the two runs' screens do not starve the loaders, for example `VAANI_SCREEN_WORKERS=8`.

## 2. Schedule (plan 11.6; per GPU)

Every command resumes from `runs/<name>/last.pt` when rerun unchanged (`resume: true`). After a crash or preemption,
rerun the same line. Do not edit a config mid-run: train.py refuses to resume when the schedule keys differ.

| Hours | GPU0 | GPU1 |
|---|---|---|
| 0-12 | ab1_fe_mini_s0, ab1_fe_mini_s1, then ab2_*, ab4_* | ab1_refvalid_s0, ab1_refvalid_s1, ab3b_*, ab3_*; then ab6_* if time allows |
| 12-36 | `r8_fe_mini` full run (winning pilot settings copied into it) | `r8_refvalid_v2` full run (warm start from r7) |
| 36-48 | val selection, G3, G4, export + G2 on the selected Mini | shortened second seed of the selected Mini; then G6 once |

```bash
# pilots: one command per arm, queued per GPU (48 epochs = 15 % of the full schedule each)
CUDA_VISIBLE_DEVICES=0 VAANI_WORKERS=<n> uv run --with numba python -m vaani.train configs/retraining/r8_ablations/ab1_fe_mini_s0.yaml
CUDA_VISIBLE_DEVICES=1 VAANI_WORKERS=<n> uv run --with numba python -m vaani.train configs/retraining/r8_ablations/ab1_refvalid_s0.yaml
# full runs (320 epochs = 200,000 steps at B 32; r7-lineage exposure, see the config headers)
CUDA_VISIBLE_DEVICES=0 VAANI_WORKERS=<n> uv run --with numba python -m vaani.train configs/retraining/r8_fe_mini.yaml
CUDA_VISIBLE_DEVICES=1 VAANI_WORKERS=<n> uv run --with numba python -m vaani.train configs/retraining/r8_refvalid_v2.yaml
```
Arm list, baselines and the dropped ablation 5 (VaaniFE has no refiner): `r8_ablations/README.md`.
Pilots run at the full configs' lr/warmup; `optim.warmup: 2000` is 6.7 % of a pilot's 30,000 steps.
Run length is TBD: it comes from step 1's measurement. At r7's 192 items/s the loader alone needs about 9.3 h per full
run (inferred).

## 3. Val selection

Each run writes `runs/<name>/best.pt`, chosen during training on val:
- `val.select: composite`: the per-clip all-three pass rate (SNR 15 dB / STOI 0.85 / PESQ 2.5) on eval_r2 val, with
  hard filters: ILD-sweep speech loss <= 0.06, and mono and web-stereo dSNR >= gtcrn_pretrained - 1 dB. It is scored for
  the raw and the EMA weights every 4 val points and at the last one. The key is (passes, pass_rate, STOI).
- `best.pt` records `weights` (raw/ema) and `selection`. `run.json` has `best_key` and each val point's
  `composite_raw` / `composite_ema`.

Between the two full runs, and to confirm the choice, score both on val with the full G3 sweep (step 4), then pick one.
Record its sha256 in results_r2/r8/testset/PROTOCOL.md **before** step 6.

## 4. G3 quality sweep on val, and G4 field acceptance

```bash
CUDA_VISIBLE_DEVICES=-1 uv run --with numba --with tabulate python scripts/eval_refvalid.py \
    --system ckpt:runs/r8_fe_mini/best.pt --out results_r2/r8/r8_fe_mini_refconditions_val --per-bucket 4 --workers 3
CUDA_VISIBLE_DEVICES=-1 uv run --with numba --with tabulate python scripts/eval_refvalid.py \
    --system ckpt:runs/r8_refvalid_v2/best.pt --out results_r2/r8/r8_refvalid_v2_refconditions_val --per-bucket 4 --workers 3
uv run --with numba python scripts/field_accept.py --system ckpt:runs/r8_fe_mini/best.pt --name r8_fe_mini --workers 3
uv run --with numba python scripts/field_accept.py --system ckpt:runs/r8_refvalid_v2/best.pt --name r8_refvalid_v2 --workers 3
```
The r7 baselines on the same scripts are results_r2/r8/r7_refconditions_val.md and results_r2/field/r7.md.
G4 Part 2's Whisper/VAD hooks are TBD (faster-whisper is not importable; see results_r2/field/README.md).

## 5. G2 export gate (the selected VaaniFE checkpoint)

```bash
uv run python -c "from vaani import export as E; E.export_fe(E.fe_load('runs/r8_fe_mini/best.pt'), 'runs/r8_fe_mini/export/fe_mini.onnx')"
uv run python scripts/graph_gate.py runs/r8_fe_mini/export/fe_mini.onnx --ckpt runs/r8_fe_mini/best.pt \
    --json runs/r8_fe_mini/export/graph_gate.json      # exit 1 = FAIL
```
If the refvalid C16 model is selected, its ONNX path has no `ref_avail` input yet (open issue from the fe/fe-runtime
work). Until that lands, G2 does not apply to it and its runtime ignores validity.

## 6. G6 test once

Follow results_r2/r8/testset/PROTOCOL.md exactly: verify the hash, then run the three scoring commands once
(`<r8 spec>` = `ckpt:runs/<selected>/best.pt`). The eval_r2 test split is not used for anything.

## Laptop evidence behind this runbook (smoke on a loaded laptop, RTX 5060; not reportable)

- 30-step smokes plus resume, composite/EMA selection and the fe export: `tests/test_train_r8_smoke.py`, results in
  `runs/smoke_r8_fe_mini/smoke_result.json` and `runs/smoke_r8_refvalid_v2/smoke_result.json` (git-ignored).
  fe: 30 steps, resumed at 15, loop loss 1.94 -> 0.37 (mean of the first and last 10), single-batch overfit
  5.53 -> 0.53, best.pt = EMA weights by composite, export parity pass, `graph_gate.py` PASS (116 folded nodes,
  layout share 0.259). refvalid_v2: 30 steps, resumed at 15, loop 13.9 -> 8.75, overfit 20.8 -> 7.38.
  The smokes override warmup to 3 and lr to 2e-3; at the real warmup of 2,000, lr is about 1e-6 at step 30.
- Loader items/s: `results_r2/r8/loader_bench.json`. Step time: `results_r2/r8/step_time.json`. The commands are inside
  each file (`command`).

| Config | Loader items/s per worker (1-3 workers) | GPU step, B 32 x 4 s, bf16 (5060) | GPU items/s |
|---|---|---|---|
| r8_fe_mini (v2, front end only) | 16.5-17.3 | 68 ms | 471 |
| r8_refvalid_v2 (v2 + NLMS/DSP) | 8.4-9.2 | 218 ms | 147 |
| r7_e256_wr64 (v1 + DSP, control) | 10.2-10.9 | 197 ms (plan 11.6 recorded 220 ms) | 162 |

The v2 rows are after commit 2158fcd. Before it, v2 ran at about 1.5 items/s per worker because of a per-draw pandas
filter in the scene pool; the old rows are kept in the same JSON. Workers needed to keep a 5060 fed:
about 471 / 17 = 28 for the Mini and 147 / 9 = 16 for refvalid (inferred: linear per-worker scaling beyond 3 workers
is assumed, not measured). The rental's per-vCPU rate differs. r7 managed 192 items/s from 32 vCPU on the box, about
6 items/s per vCPU against 10.5 per worker here, so rental rates may be around 0.6x the laptop's (inferred).
**Measure both on the box (step 1) before fixing the schedule.**
