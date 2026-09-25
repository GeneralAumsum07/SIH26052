# r8 rental runbook (plan 11.6): 2x RTX 5090, 36-48 h

Exact steps for the r8 retrain on a rented 2-GPU box. Nothing here has been run on the rental box; the laptop evidence
behind it is listed at the end. **Selection happens on val only. The r8 test set is scored once (G6), on the laptop.
r7 stays the shipping control: none of these configs touches the r7 path** (mixer v1, `val.select` absent, no `ema`
key and no `loss: fe`, which keep train.py's r7 behaviour bit-exact; tests/test_dataset.py and
tests/test_train_smoke.py cover this). Whether r8 replaces r7 is Rachit's call after the gates.

## Decisions (Rachit, 2026-09-25; final)

- **Checkpoint selection:** `val.select: composite` (the r8 configs' default) is final.
- **Single-channel real recordings:** the headline row is `ref_zero` (reference zeroed, validity 0); `ref_dup`
  (duplicated primary) is reported as a labelled stress row.
- **NLMS stays in the default path** (the "hybrid" pitch). Read ablation 2 (`ab2_pr_nhat_*` against `ab1_fe_mini_*`,
  inputs `pr`) closely: it is the measured cost or gain of feeding the fixed NLMS's n_hat. The queue runs it first in
  ablation 2 for that reason.
- **Licence posture:** research-only trained weights are acceptable.
- **Installs approved:** numba (`fast` extra), torch-pesq (`train` extra), faster-whisper (`asr` extra). The box runs
  `uv sync --all-extras --frozen`, so all three are in its venv; tensorrt-cu12 / polygraphy are not needed on the box.
- **Datasets download directly onto the box**, never onto the laptop. Rachit does every login or registration.
- **The Orin runs JetPack 7.2.1.** Hardware tests (Pi, Orin, two-mic rig) are Rachit's; nothing here needs hardware.
- **DRDO cannot be asked for anything:** every defence condition is simulated.
- **Box specs:** Rachit sends the offer's vCPU / RAM / disk before renting; the sizing rule below says what is enough.

## Before renting (Rachit's checklist)

1. **Dataset accounts.** Log in / register and accept terms for each dataset whose access is not `direct`, then
   create the API tokens the fetcher reads from the environment.
   <!-- TBD(ds-wire): access table -->
   Until ds-wire's table lands (configs/data/r8_datasets.yaml, `credentials` per dataset), the research notes
   (docs/impl/2026-09-24/research/datasets_astra.md) name these: Mozilla Data Collective account + API key
   (Common Voice Hindi, `MDC_API_KEY`), Kaggle API token (MAD), a Hugging Face account approved for gated sets
   (Svarah) whose read token can double as the mirror's `HF_TOKEN`, and a CADRE site registration (manual download,
   no headless path confirmed). The box prints every missing variable in its first seconds
   (`bash scripts/r8_box_setup.sh --env-check`).
2. **Private Hugging Face mirror** of the laptop-only artefacts (licensed audio: the repo must be private):
   ```bash
   bash scripts/r8_mirror_stage.sh            # laptop; stages into data/mirror_stage, uploads nothing
   hf repos create <user>/vaani-r8-mirror --repo-type dataset --private --exist-ok
   hf upload <user>/vaani-r8-mirror data/mirror_stage . --repo-type dataset
   ```
   The script prints these lines with the repo name filled in. What it stages and why is in its header: the frozen
   eval_r2 val split (tar, verified against b5f7a4d43bee before staging and again on the box), the VAD-filtered MAD
   manifests (`mad_v2`, `mad_speech_contamination`: the Silero run is not pinned, so a box rebuild could differ), and
   the laptop scans of every other manifest as a reference the box preflight diffs against. **Not shipped:** the r8
   test set (G6 is scored on the laptop after the selected checkpoint comes back, which keeps it out of reach of
   selection), the RIR banks (GitHub release), bank_eval_r8 (only the test render reads it), r7 best.pt and the gtcrn
   baseline (both tracked in git). Manifests store repo-relative paths; `vaani.data.manifests.read` normalises the
   laptop's backslashes, so they resolve on Linux once the box has the audio.
3. **RIR banks on the GitHub release.** Every asset in configs/data/r8_banks.json must be downloadable, without a
   token, at `$RIR_BANK_URL/<asset>`: `bank.npz`, `bank_r3.npz`, `bank_r8.npz` and bank_r8's three `.npy` sidecars.
   The banks lane recommends adding the bank_r8 files to the existing release `rir-banks-2026-09-21`
   (docs/impl/2026-09-24/reports/banks.md). Upload `bank_r3.npz` from `deploy/rir_banks/` (sha256 e4e67463..., the
   published copy); the laptop's `data/rirs/bank_r3.npz` is a different file (99dcfb26...). A bank is never rebuilt on
   the box: pyroomacoustics differs across machines, so a rebuilt bank is a different file. bank_r8.noise.npy is
   1.92 GB, under GitHub's 2 GiB per-asset limit. If the sidecars are left off the release, RirBank rebuilds them on
   first load and the preflight checks their sha256 (the banks lane found them reproducible with np.save).
4. **Config changes that must be in main before the box clones it** (wire4 owns the configs):
   - FE loss PESQ term: `loss_cfg: {w_pesq: 0.001, pesq_required: true}` in r8_fe_mini and the VaaniFE pilots, so a
     box without torch_pesq fails at start instead of silently training with the term at 0.
   - `data.bank`: bank_r3 (comparable with r7) or bank_r8 (M6 armoured rooms). See reports/banks.md for the choice.
   - `mix.v2.tail_share`: the full configs set 0.25; the mixer default (G1 c5) is 0.40. The box gate gates whatever
     the full configs train (`scripts/r8_preflight.py --g1-cmd`). Laptop prediction on the published bank_r3, seed 202,
     200 items: 0.25 passes by thin margins (param AUC 0.738, CI95 0.692-0.779; M2 share 0.252 against a 0.24 cutoff),
     and 0.40 passes clearly (0.699 / 0.665, M2 0.398). Evidence: docs/impl/2026-09-24/ckpt/box/g1_predict/README.md.
5. **Box spec** (send it before renting):

   | Item | Minimum | Rule |
   |---|---|---|
   | GPUs | 2x RTX 5090 (sm_120) | the queues assume exactly 2 |
   | NVIDIA driver | >= 570 (CUDA 12.8) | torch 2.11.0+cu128; the setup's per-GPU matmul fails fast on a mismatch |
   | vCPU | **64** (96 preferred) | see the sizing rule below |
   | RAM | TBD | no per-worker RSS has been measured; watch `free -g` during the setup's bench stage (question: what RSS does one v2 loader worker reach?) |
   | Disk | TBD(ds-wire): dataset download + extracted size | + about 20 GB fixed (venv 4.7 GB on the laptop, more on Linux with the CUDA libs; banks 4.6 GB; val tar 2.3 GB and its extracted copy) + the pack (about the int16 size of the manifested audio, inferred) + 50 GB headroom for runs/ (the preflight's `--need-gb` default) |

   **vCPU sizing rule.** The loader is the bottleneck (r7: about 130 of each 144 ms step was CPU), so the box is
   rented for cores. `run_r8.sh` gives each queue W = (nproc - 4) / 2 loader workers. A run's rate is
   W x (items/s per worker); a pilot is 960,000 items, a full run 6,400,000. Laptop per-worker rates
   (results_r2/r8/loader_bench.json): VaaniFE-Mini 16.5-17.3, refvalid/NLMS 8.4-9.2. r7's box rate suggests the rental
   runs at about 0.6x the laptop per worker (inferred), i.e. about 10 and 5.3 items/s. With those (inferred):

   | vCPU | W per queue | Mini pilot | NLMS pilot | 11 pilots per queue | Mini full | refvalid full |
   |---|---|---|---|---|---|---|
   | 48 | 22 | 1.2 h | 2.3 h | 15.4 h: the tail arms (ab4, ab6) drop at PILOT_HOURS 12 | 8.1 h | 15.2 h |
   | 64 | 30 | 0.9 h | 1.7 h | 11.4 h | 5.9 h | 11.2 h |
   | 96 | 46 | 0.6 h | 1.1 h | 7.4 h | 3.9 h | 7.3 h |

   Each queue holds 9 Mini-rate pilots and 2 NLMS-rate pilots (GPU0: ab2_pr_nhat; GPU1: ab1_refvalid). The GPU is not
   the limit: a 5060 steps the Mini at 471 and refvalid at 147 items/s (results_r2/r8/step_time.json), and a 5090 is
   faster. The box's own bench (setup stage `bench`) replaces these estimates before the queues start.
6. **Environment to export on the box** (values never go in a file in the repo):
   ```bash
   export MIRROR_HF_REPO=<user>/vaani-r8-mirror HF_TOKEN=<read token>
   export RIR_BANK_URL=https://github.com/<owner>/<repo>/releases/download/<tag>
   # + every dataset credential in configs/data/r8_datasets.yaml (TBD(ds-wire): the list)
   ```

## Bootstrap timeline (estimates, inferred; the log has the real times)

| Minute | What | Blocks the launch? |
|---|---|---|
| 0 | `git clone`, `tmux new -s box`, exports, `bash scripts/r8_box_setup.sh --env-check` (seconds) | yes: fix what is MISSING |
| 1 | specs recorded (nproc, free, df, nvidia-smi -> runs/box_setup/specs.txt); apt aria2/tmux/libsndfile1/build-essential; uv | yes |
| 3-8 | `uv sync --all-extras --frozen` (torch cu128 from download.pytorch.org, numba, torch-pesq, faster-whisper; pesq builds from sdist) | yes |
| 8 | GPU check: a matmul on each device, numba and torch_pesq import | yes |
| 8 | background: banks (about 4.6 GB by sha256: three npz of 0.4-0.8 GB plus bank_r8's sidecars, noise 1.92 GB and speech 0.64 GB), mirror (about 2.3 GB, SHA256SUMS then verify_eval_set), datasets (the first fetch group first) | - |
| 8 + TBD(ds-wire) | first dataset group fetched, scanned, verified: the datasets the two queue heads and the G1 gate read | yes |
| + 2 | tests that guard silent corruption (test_r8_box, test_losses, test_pack, test_mixer_v2, test_data_gates, test_golden_vectors) | yes |
| + TBD | pack the corpus (speed only; verified bit-identical by tests/test_pack.py) | yes |
| + 3 | G1 on the box (laptop: 2 min 14 s for two gates) | yes: the full runs refuse without it |
| + 3 | loader bench (sizes workers) and step time, then the preflight | yes |
| launch | `scripts/run_r8.sh start all` (with `--launch`) | - |

The rest of the datasets keep arriving in the background (`tail -f runs/box_setup/datasets.log`); only the first group
blocks the launch.

## Launch

```bash
git clone https://github.com/GeneralAumsum07/SIH26052.git && cd SIH26052
tmux new -s box
export ...                                         # "Before renting" item 6
bash scripts/r8_box_setup.sh --env-check           # seconds; exit 1 names what is missing
bash scripts/r8_box_setup.sh --launch              # bootstrap, G1, preflight, then both queues in tmux session "r8"
bash scripts/run_r8.sh status                      # DONE / RUNNING / FAILED / DROPPED / PENDING per run, and the G1 line
bash scripts/run_r8.sh next                        # the run each queue starts next
# after the pilots: copy the winning settings into r8_fe_mini.yaml / r8_refvalid_v2.yaml, then
bash scripts/run_r8.sh go-full
```

`r8_box_setup.sh` is idempotent: every finished stage leaves `runs/box_setup/<stage>.ok`, a background stage that is
still alive is joined rather than relaunched, and the log is `runs/box_setup.log`. After a dropped SSH session, rerun
the same line. `--dry-run` prints every stage without installing or downloading anything. Knobs: `GPUS`,
`FETCH_PARALLEL` (8), `G1_SEED` (202), `G1_ITEMS` (200), `SKIP_TESTS/SKIP_PACK/SKIP_BENCH=1`, `PREFLIGHT_SMOKE=N`
(N train steps per config into runs/preflight_*), `ALLOW_MISSING_ENV=1`.

`scripts/r8_preflight.py` (run by the setup; rerun any time) checks, per r8 config: every manifest exists and a sample
of audio paths decodes at 16 kHz; bank and sidecar sha256 against r8_banks.json; the val set verifies
(verify_eval_set.py, b5f7a4d43bee); configs/data/r8_heldout_exclude.json exists and its source_ids are in the manifests
it names; init_from sha256; numba and torch_pesq import (faster_whisper warns only); CUDA device count; free disk; the
box G1 JSON passes, scored >= 200 items on the configs' bank, with the full configs' own mix.v2 block and no scene
overrides. It exits 1 with the list and writes runs/preflight.json.

## 1. G1 data gate (on the box, before any run)

`r8_box_setup.sh` runs it; by hand:
```bash
eval "$(.venv/bin/python scripts/r8_preflight.py --g1-cmd)"     # data_gates v2, seed 202, 200 items, bootstrap 2000
```
Pass: the v2 ILD-only AUC <= 0.75 on both paths and the M2 share >= 0.24 (results_r2/r8/data_gates/box_g1/v2.json).
If it fails, do not start the full runs, and do not override: report it. `run_r8.sh` refuses the pilots too unless
`ALLOW_PILOTS_WITHOUT_G1=1`.

## 2. Schedule (plan 11.6; per GPU, `scripts/run_r8.sh`)

| Hours | GPU0 | GPU1 |
|---|---|---|
| 0-12 | ab1_fe_mini_s0/s1, ab2_pr_nhat_s0/s1, ab2_p_s0/s1, ab2_pr_pld_s0/s1, ab4_* | ab1_refvalid_s0/s1, ab3b_*, ab3_*, ab6_* |
| 12-36 | `r8_fe_mini` full run (winning pilot settings copied into it) | `r8_refvalid_v2` full run (warm start from r7) |
| 36-48 | val selection, G3, G4, export + G2 on the selected Mini | shortened second seed of the selected Mini |

Priority order is plan 11.6's (1, 3b, 2, 3, 4, 6; ablation 5 is dropped because VaaniFE has no refiner). A pilot that
would start later than `PILOT_HOURS` (12) after its queue began is logged as DROPPED in runs/r8_queue/dropped.txt.
Each queue sets `CUDA_VISIBLE_DEVICES` to its GPU and `VAANI_WORKERS` to (nproc - 4) / 2 (override with
`WORKERS_GPU0/1` from the box bench), caps composite screens at `VAANI_SCREEN_WORKERS=8`, and relaunches a failed run
up to 3 times; train.py resumes from `runs/<name>/last.pt` (`resume: true`). A run is DONE when it exits 0 and its
run.json has `end`. Do not edit a config mid-run: train.py refuses to resume when the schedule keys differ. Per-run
logs: runs/r8_queue/logs/<name>.log. Arm list and baselines: `r8_ablations/README.md`. Pilots run at the full
configs' lr/warmup; `optim.warmup: 2000` is 6.7 % of a pilot's 30,000 steps.

**Falsification check** during the first pilots: `nvidia-smi dmon -s u -d 5`. Sustained GPU use above about 60 % means
the box is not loader-bound and the sizing above is pessimistic.

Pull `best.pt`, `last.pt` and `run.json` off the box after each run finishes, not at the end. Stop, do not destroy, the
instance between sessions: the corpus and the pack are expensive to rebuild.

## 3. Val selection

Each run writes `runs/<name>/best.pt`, chosen during training on val:
- `val.select: composite`: the per-clip all-three pass rate (SNR 15 dB / STOI 0.85 / PESQ 2.5) on eval_r2 val, with
  hard filters: ILD-sweep speech loss <= 0.06, and mono and web-stereo dSNR >= gtcrn_pretrained - 1 dB. It is scored for
  the raw and the EMA weights every 4 val points and at the last one. The key is (passes, pass_rate, STOI).
- `best.pt` records `weights` (raw/ema) and `selection`. `run.json` has `best_key` and each val point's
  `composite_raw` / `composite_ema`.
- At 15 % exposure the composite pass rate may be 0 for every pilot; the key then falls back to STOI.

Between the two full runs, and to confirm the choice, score both on val with the full G3 sweep (step 4), then pick one.
Record its sha256 in results_r2/r8/testset/PROTOCOL.md **before** step 6.

## 4. G3 quality sweep on val, and G4 field acceptance

```bash
CUDA_VISIBLE_DEVICES=-1 uv run --with tabulate python scripts/eval_refvalid.py \
    --system ckpt:runs/r8_fe_mini/best.pt --out results_r2/r8/r8_fe_mini_refconditions_val --per-bucket 4 --workers 3
CUDA_VISIBLE_DEVICES=-1 uv run --with tabulate python scripts/eval_refvalid.py \
    --system ckpt:runs/r8_refvalid_v2/best.pt --out results_r2/r8/r8_refvalid_v2_refconditions_val --per-bucket 4 --workers 3
uv run python scripts/field_accept.py --system ckpt:runs/r8_fe_mini/best.pt --name r8_fe_mini --workers 3
uv run python scripts/field_accept.py --system ckpt:runs/r8_refvalid_v2/best.pt --name r8_refvalid_v2 --workers 3
```
The r7 baselines on the same scripts are results_r2/r8/r7_refconditions_val.md and results_r2/field/r7.md. G4 Part 2's
Whisper word-survival and Silero VAD hooks are described in results_r2/field/README.md; faster-whisper is in the box's
venv (`asr` extra) and its model downloads on the box.

## 5. G2 export gate (the selected VaaniFE checkpoint)

```bash
uv run python -c "from vaani import export as E; E.export_fe(E.fe_load('runs/r8_fe_mini/best.pt'), 'runs/r8_fe_mini/export/fe_mini.onnx')"
uv run python scripts/graph_gate.py runs/r8_fe_mini/export/fe_mini.onnx --ckpt runs/r8_fe_mini/best.pt \
    --json runs/r8_fe_mini/export/graph_gate.json      # exit 1 = FAIL
```
If the refvalid C16 model is selected, export it with its `ref_avail` input and check parity across a reference hole
(G2 is a TensorRT-structure gate for VaaniFE; the refvalid graph keeps r7's structure, which G2 fails by design):
```bash
uv run python -c "from vaani import export as E; o = E.export_refvalid('runs/r8_refvalid_v2/best.pt', 'runs/r8_refvalid_v2/export/refvalid.onnx'); print(E.refvalid_parity('runs/r8_refvalid_v2/best.pt', o))"
```

## 6. G6 test once (on the laptop)

Download the selected `best.pt` to the laptop, then follow results_r2/r8/testset/PROTOCOL.md exactly: verify the hash
of data/eval_r8_test, then run the three scoring commands once (`<r8 spec>` = `ckpt:runs/<selected>/best.pt`), on
Rachit's say-so. The eval_r2 test split is not used for anything.

## Laptop evidence behind this runbook (smoke on a loaded laptop, RTX 5060; not reportable)

- Box scripts: tests/test_r8_box.py (preflight checks on a synthetic layout, each broken input fails; G1 check refuses
  a gate on a different mixer, bank, item count or with scene overrides; bank plan; fetch order; env check without
  printing values; run_r8.sh dry run, refusal without G1, `next`/`status`; mirror staging on a fake tree; the setup's
  env check and dry run; requirements files against uv.lock). 28 passed.
- Box G1 prediction on the published bank_r3: docs/impl/2026-09-24/ckpt/box/g1_predict/README.md.
- FE loss PESQ term: docs/impl/2026-09-24/ckpt/box/pesq_smoke/README.md (20 steps, term finite and non-zero).
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
filter in the scene pool; the old rows are kept in the same JSON. Linear per-worker scaling beyond 3 workers is
assumed, not measured. r7 managed 192 items/s from 32 vCPU on the earlier rental, about 6 items/s per vCPU against
10.5 per worker here, which is where the 0.6x rental factor comes from (inferred).
