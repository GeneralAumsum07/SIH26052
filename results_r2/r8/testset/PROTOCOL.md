# r8 test set: pre-registration (plan 11.2 G6, ex-B3)

Registered 2026-09-25, before any r8 training run and before any system was scored on this set.

## The rule

- `data/eval_r8_test/test` is scored **once**, for the final report, on the one r8 checkpoint that was selected on
  **val**. Nothing is tuned, selected, thresholded, early-stopped or debugged on it: no checkpoint choice, no
  controller or limiter setting, no ablation pick, no re-render with other seeds.
- eval_r2 test is used for nothing in r8 (it is burned for selection).
- If a render or scoring bug is found after scoring, the fix and the re-score are reported next to the first score,
  never instead of it.
- The set is frozen: `EVALSET_HASH` = `ed024af085a2`. `guard_frozen` in `scripts/render_eval_sets.py` refuses to
  re-render over it without `--force`.

## What is in it

2,308 items, each 6 s at 16 kHz, 2 channels (primary, reference) plus a clean target. `index.csv` in the split root
has one row per item. Its `category` column is `<subset>/<name>`, and `vaani.eval` copies that column into every
result CSV. Counts per category are in `index_summary.csv` (built by the command in README.md).

| subset | buckets | items | mix |
|---|---|---:|---|
| `v1` (nominal, as eval_r2 test) | 6 classes (stationary, changing, impulsive, impulsive+stationary, recorded_impulsive, recorded_impulsive+stationary) x 6 input SNRs (-10..15 dB) x 20, plus `clean_inf` x 20 | 740 | mixer v1 defaults, the same as the eval_r2 render |
| `defence` (plan A3) | gunshot, blast_small_arms, blast_artillery, helicopter, vehicle, siren x 6 SNRs x 16 | 576 | r7's training transient block (peaks 15-45 dB re speech RMS, room path, soft-clip, recorder gain -32..-18 dBFS). Blasts use physics v2 (M5) |
| `heldout` | `heldout_drone_<snr>`, `heldout_noisex92_<snr>` x 16 | 192 | mixer v1, bed only. The beds are recordings that no r8 training pool may contain (the exclusion file below) |
| `loud` | `loud_ears_<snr>` x 16 | 96 | mixer v1 over the MAD stationary bed. Speech is the 12 `*_loud` clips of the held-out EARS speaker |
| `fault` | 10 faults (as eval_r2 test) x SNR 0 and 5 dB x 16 | 320 | mixer v1 with the random faults off; the fault is the only degradation |
| `v2` (mixer v2 scenes) | `v2_<scene>` x 48 for patrol, apc, firefight, artillery, helicopter, drone, windy_ridge, command_post | 384 | `MixConfig(version=2, p_clean=0, v2={"tail_share": 0})`. The levels are SPL-calibrated, so SNR is an **output** (per-item `snr_in`). `windy_ridge` is the gusty-wind bucket (M4 wind always on, 5-20 m/s) |

Both labelled mix families are present. The v1 nominal subset is comparable with eval_r2. The v2 scene subset is the
mixer-v2 physical distribution. It has the out-of-physics reference tail off, because that tail is a training
augmentation, not a test condition.

## Sources and held-out groups

- Speech: the test-split rows of `librispeech.parquet` and `cv_hi.parquet`. These are the same test utterance pools
  as eval_r2 and eval_defence. They are speaker-disjoint from training, but the crops, pairings, rooms and seeds are
  new. The `loud` subset uses EARS speaker p004, whose speech is entirely held out.
- Noise: the test-split rows of `esc50`, `mad_v2`, DNS `freesound_000`, `gunshots`, `cadre` and `demand` (STRAFFIC),
  plus the held-out rows below.
- **The exclusion file is `configs/data/r8_heldout_exclude.json`** (schema `vaani.heldout_exclude/1`,
  sha256 `5a49f89a...`). The r8 trainer must drop every row whose `source_id` is listed there from every training
  and val pool. It is written by
  `python scripts/render_eval_sets.py --write-heldout configs/data/r8_heldout_exclude.json`. The split rules below
  are deterministic, and each is written into the file.
  - drone: the group is the recording (the source_id minus its `_NNN_` chunk index). A recording is held out when
    `manifests.stable_hash(recording) % 5 == 0`. That gives 52 of 257 recordings, 307 rows and 0.085 h. At test
    time the chunks of a recording are joined in chunk order.
    Inferred caveat: recordings in one family may share a flight session, and the `mixed_*` families may reuse clean
    drone audio. Independence is therefore only at the level of the recording file.
  - noisex92: three whole files are held out. Each has a same-type sibling that stays in training: buccaneer2
    (buccaneer1 stays), m109 (leopard stays) and destroyerengine (destroyerops stays). That is 0.196 h.
  - ears: the whole train-split speaker with the smallest `stable_hash(group_id)` is held out. That speaker is p004:
    149 rows, of which 12 are loud clips. The val speaker, p002, stays in val.
  - mad: the `mad_v2` test split, grouped by YouTube video: 557 rows in 47 groups. The render refuses the
    folder-grouped `mad.parquet`. In that manifest, 141 of 673 test rows (21 %) share a YouTube video with training
    rows (vaani/data/sources.py `scan_mad`, plan 3.6).
- RIR bank: `data/rirs/bank_eval_r8.npz`, sha256 `10e1f3a9...`. It holds 1,000 rooms in room mode. It was built with
  `rirs.build_bank(seed=2610, seed_namespace="eval")`, so its draw stream is disjoint from every training bank
  (mixer2 M6). Room mode uses the image-source method only, so the M6 receiver radius does not apply. The bank has no
  armoured entries, because `RirBank.sample` cannot select them. As a result, apc scenes that take the room path use
  ordinary rooms.
- Seeds: the render seed is 2610. The per-item seed is
  `[2610, stable_hash(subset) % 1000, stable_hash(name) % 1000, snr + 100, i]`.

## Render and integrity (done once, 2026-09-25)

```
M=data/manifests
uv run --with numba python scripts/render_eval_sets.py --r8-test --heldout configs/data/r8_heldout_exclude.json \
  --manifests $M/librispeech.parquet $M/cv_hi.parquet $M/esc50.parquet $M/mad_v2.parquet \
    $M/dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet $M/gunshots.parquet $M/cadre.parquet \
    $M/demand.parquet $M/drone.parquet $M/noisex92.parquet $M/ears.parquet \
  --split test --out data/eval_r8_test --bank data/rirs/bank_eval_r8.npz --build-eval-bank 1000 --bank-seed 2610 \
  --bank-workers 3 --seed 2610
.venv/Scripts/python.exe scripts/verify_eval_set.py data/eval_r8_test/test ed024af085a2   # "verified (2308 items)"
```

As commit 3520f36 notes, the hash is a within-platform integrity check. A render on another OS can differ in its
last float bits.

## Metrics and targets (fixed now)

- Per item, from `vaani.eval`: `snr_out`, `si_sdr`, `stoi`, `pesq_wb`, DNSMOS P.835 `dnsmos_sig/bak/ovrl`, and
  `recovery_s` (impulse buckets with twins). The CSV also carries `category`, `noise_source`, `impulse_source` and
  `snr_in`.
- PS targets (`vaani.report.TARGETS`): SNR_out > 15 dB, STOI > 0.85, PESQ > 2.5, all strict. `pass3` is the per-clip
  share that meets all three.
- Report layout:
  - Per category x input SNR, give the mean with a 95 % clustered-bootstrap CI (`vaani.report.cluster_ci`, 1,000
    resamples). The cluster is `impulse_source` for gunshot and `noise_source` otherwise.
  - Add the paired delta against r7 and against raw.
  - For `v2`, report per scene, with the `snr_in` quantiles alongside (`v2_snr_quantiles.csv`), because SNR is not a
    controlled variable there.
  - Keep `fault` separate from the nominal envelope, as report.py does.
- Systems:
  - the selected r8 checkpoint (exactly one);
  - r7 as the shipping control (`cascade:runs/r7_e256_wr64_refiner/best.pt`, the same spec as scripts/r7_run.sh);
  - `raw` passthrough.

## Scoring command (run ONCE, after r8 selection on val)

```
CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system <r8 spec> --split test \
  --eval-root data/eval_r8_test --workers 3 --dnsmos --out results_r2/r8/testset/r8_selected.csv
CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system cascade:runs/r7_e256_wr64_refiner/best.pt \
  --split test --eval-root data/eval_r8_test --workers 3 --dnsmos --out results_r2/r8/testset/r7_e256_wr64_cascade.csv
CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system raw --split test \
  --eval-root data/eval_r8_test --workers 1 --dnsmos --out results_r2/r8/testset/raw.csv
```

- If a worker dies, rerun the same command with `--resume`.
- Before scoring, verify the hash with the verify command above.
- TBD: `<r8 spec>` is the `--system` string of the checkpoint selected on val. The trainer stage fills it in at
  selection time, before this command runs, and records the checkpoint's sha256 here.
- TBD: there is no table script for the mixed subsets yet. `scripts/defence_table.py` handles the category x SNR
  cells, but not per-scene v2 rows. The table layout above is fixed now, and a script must implement it without
  looking at test scores.

## Known limits (stated before scoring)

- Every mixture is synthetic.
- Lombard GRID is not on disk (TBD: download approval). The `loud` subset is one EARS speaker (CC BY-NC), and the
  Lombard effect in `v2` is mixer v2's alpha tilt only. The +1.9 st F0 shift is not applied (mixer2 TBD).
- Siren has two ESC-50 test recordings. The NOISEX held-out subset has three recordings, so its CIs are wide.
- Two v2 meta flags have physical meanings:
  - `overloaded` means that the primary passed the ICS-43434 soft knee (about 90.7 dB SPL peak, `calib.soft_knee()`),
    not that it reached AOP. It is True on almost every v2 item.
  - `clipped` means that the input passed the 123 dB SPL rails. That happens on 48/48 helicopter items, 44/48
    firefight, 43/48 apc and 22/48 artillery.
- `impulse_peak_db` has different units by subset. In `v1`, `defence` and `fault` it is dB re speech-active RMS. In
  `v2` it is dB SPL peak at the primary.
- The v2 clean target is the boom speech after the linear front end, before saturation. An enhancer cannot undo
  saturation, so the v2 scores include that loss by design (mixer v2 M7).
