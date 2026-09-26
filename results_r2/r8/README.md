# results_r2/r8: r8 gates, data checks, reference validity and the fallback Mini

Everything here is VAL or a deterministic count. No test-set number lives in this folder.

## budget.{json,md}: Mini parameter and MAC budget (spec 6.2)

Command: `python scripts/audit_budget.py` (CPU, deterministic counts with the `vaani.export.layer_macs` hooks).
Budget: at most 60,000 total entries and 90.706 matrix MMAC/s.

| system | total entries | MMAC/s | within budget |
|---|---:|---:|---|
| r7 (shipping) | 52,747 | 82.460 | yes |
| refvalid Mini (C16 + ref_validity + r7 refiner) | 53,147 | 85.621 | yes |
| C32 / C64 / C96 + ref_validity (untrained projections) | 110,683 / 320,219 / 655,707 | 152.064 / 387.622 / 760.076 | no |

The ref_validity extension is 400 entries (Conv2d(5, 16, (1, 5)), no bias) and 26,000 MAC per hop.

## r7_refconditions_val.{csv,md}: r7 under deterministic reference conditions (plan G3)

Command (run from the repo root, 2 workers, CPU):

```
CUDA_VISIBLE_DEVICES=-1 uv run --with numba --with tabulate python scripts/eval_refvalid.py \
  --system ckpt:results_r2/runs/r7_e256_wr64_refiner/best.pt --out results_r2/r8/r7_refconditions_val --per-bucket 4 --workers 2
```

148 val clips x 15 conditions = 2,220 rows, 0 errored. Per clip, fail = speech loss > 0.15 or SNR_out < SNR_in - 1 dB.
Speech loss follows the diag_webaudio frame_stats definition (share of speech-active 20 ms frames whose projected
speech gain is below -15 dB). talker_leak adds the talker to the reference at -2 dB re primary.

Key r7 numbers (from `r7_refconditions_val.md`):

| condition | SNR_out dB | STOI | speech loss | fails / 148 |
|---|---:|---:|---:|---:|
| present | 13.93 | 0.897 | 0.064 | 21 |
| absent (reference zeros) | 7.39 | 0.783 | 0.163 | 54 |
| burst_dropout | 11.32 | 0.853 | 0.096 | 29 |
| gain -12 dB | 5.09 | 0.822 | 0.063 | 16 |
| lowpass (obstruction) | 3.27 | 0.791 | 0.045 | 15 |
| delay | 11.94 | 0.875 | 0.077 | 25 |
| clipped | 7.86 | 0.839 | 0.068 | 23 |
| talker_leak (near-primary level) | 0.13 | 0.489 | 0.953 | 148 |
| ILD -14 / -8 / -6 / -4 / 0 dB | 15.84 / 14.23 / 10.00 / 1.11 / -0.02 | 0.938 / 0.933 / 0.909 / 0.661 / 0.535 | 0.034 / 0.054 / 0.114 / 0.744 / 1.000 | 8 / 14 / 40 / 147 / 148 |

Burst-dropout recovery (from the CSV): 3 of 148 clips never recover within the clip (`recovery_s = inf`);
over the other 145, median 0.14 s and p95 0.96 s. Largest dropout-edge click 23.8 dB over the clip's p99 step.

Reading (inferred from the table, not a separate experiment): r7 collapses once the reference hears the talker
within about 4 dB of the primary (the web-WAV failure); between -8 and -4 dB ILD speech loss rises from 0.05 to 0.74.
The refvalid Mini is scored with the same command and `--system ckpt:runs/r8_mini_refvalid_refiner/best.pt`
(TBD: after the r8 retrain produces it). Any `onnx:` path or `py:module:factory` runner plugs in the same way.

## configs/retraining/r8_mini_refvalid.yaml: 20-step GPU smoke (code-runs proof, NOT a result)

A scratchpad copy of the config with `max_steps: 20`, `num_workers: 2`, `data.epoch_len: 640` and a 32-item dynamic
val screen, run as `uv run --with numba python -m vaani.train <copy>` on the RTX 5060: 20 steps, 0 skipped,
wall 174.4 s by run.json (about 3.7 items/s, loader-bound at 2 workers on a shared, loaded machine: non-reportable).
The zero-init ref_conv weight moved to norm 0.0018, so gradient reaches the new input. No checkpoint was kept.

## Other files in this folder

| file | what | command / source |
|---|---|---|
| `data_gates/` | G1 ILD-shortcut gate: the 48-item seed-55 pass (0.727 / 0.697) is superseded; confirmed with the c5 defaults on fresh seed 202, 200 items: param 0.699, room 0.665 (gate <= 0.75) | see [data_gates/README.md](data_gates/README.md) |
| `g1_bank_r8/` | G1 room path on bank_r8 (the r8 training bank), fresh seed 7331, 200 items: 0.659 [0.609, 0.708], pass; physical-mode items alone 0.817 | see [g1_bank_r8/README.md](g1_bank_r8/README.md) |
| `calib/` | mixer v2 level chain: the 123 dB clipping is physics under the mic model, not a calibration bug; `overloaded` flag meaning fixed | see [calib/README.md](calib/README.md) |
| `banks/` | bank_r8 (M6 receiver radius 0.05 m) build, validation and sidecar checks | see [banks/README.md](banks/README.md) |
| `native_crash/` | eval segfaults root-caused to a pesq 0.0.4 out-of-bounds read; repro inputs and ASan output | see [native_crash/README.md](native_crash/README.md) |
| `mad_speech_filter.json` | MAD speech-contamination pass: 477 of 6,483 clips (0.60 h) flagged; writes `data/manifests/mad_v2.parquet` (git-ignored) | `source <diag env.sh> && .venv/Scripts/python.exe scripts/mad_speech_filter.py`; the VAD is Silero (threshold 0.5) from the faster-whisper bundle in the uv cache, reached through the diag env.sh PYTHONPATH |
| `mixer_bench.json` | mixer v1 vs v2 ms/item (smoke, loaded machine, not reportable) | `python scripts/data_gates.py --bench 200` (in the JSON as `command`) |
| `loader_bench.json` | dataset items/s per worker for r8_fe_mini / r8_refvalid_v2 / r7 (smoke, laptop RTX 5060 box, not reportable) | embedded in the JSON as `command` |
| `step_time.json` | GPU train-step time at B 32 x 4 s, bf16 (smoke, laptop RTX 5060, not reportable) | embedded in the JSON as `command` |
| `testset/` | the pre-registered r8 test set: protocol, hash and index summary | see [testset/README.md](testset/README.md) |
