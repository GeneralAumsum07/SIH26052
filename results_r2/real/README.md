# Real recordings (plan A4): the one non-synthetic row

Every other score in the repo is on `mixer.mix()` output. These are recordings, so there is no clean reference and
**every number here is a reference-free proxy** (DNSMOS P.835 and a level-based attenuation proxy), not SNR/STOI/PESQ.

## Command

    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python scripts/score_real.py --workers 2

Run 2026-09-24 17:59-18:15 at commit 492b600 (scripts/score_real.py, vaani/physical.py), 2 workers on a loaded
shared machine. `--summarise-only` rebuilds the CSVs and table from `work/rows.jsonl` without re-scoring.

## Files

- `table.md`: system x condition means with 95% bootstrap CIs (generated).
- `summary.csv`: the same, every metric with `_lo`/`_hi`, plus paired OVRL minus raw.
- `scores.csv`: one row per clip x system x condition (DNSMOS SIG/BAK/OVRL, atten20_frac, mean_atten_db, r7 gate/burst/limiter).
- `work/subset.json`: the exact clips scored; `work/rows.jsonl`: the per-clip cache.

## Data

- **MAD label 0 ("communication")**: radio speech recorded in military noise, 981 clips in
  `data/download/mad/MAD_dataset/{training,test}.csv`. One clip per YouTube video (seed 0), cap 200, cropped to 10 s:
  **192 clips**. The video id comes from the CSV's `youtube url`; 60 of the 981 clips have an empty URL and are grouped
  by their folder instead (13 of the 192 picks). That those 13 are independent videos is *inferred*, not checked.
- **Web WAV** `C:/Users/Rachit/Downloads/abcd.wav` (stereo 44.1 kHz, 37.2 s; L = primary, R = reference, as run on
  the Pi), resampled to 16 kHz with polyphase 160/441. One file: point values, no CI.

## Systems and conditions

- `raw` / `mono`: the primary, passthrough. `gtcrn_pretrained` / `mono`: DNS3 weights, primary only.
- `r7`: deploy/r7/cascade.onnx with deploy/r7/model_config.json through `vaani.live.StreamEngine`, hop by hop (the board path).
  - `stereo_LR`: the web WAV's own R channel as reference.
  - `ref_zero`: reference all zeros. A **fault condition** for r7 (a dead reference mic).
  - `ref_dup`: reference = primary. A **degenerate copy** r7 never saw in training.

Metrics: DNSMOS P.835 of the output (vaani/dnsmos.py). `atten>20dB` = fraction of active 20 ms input frames (energy
within 30 dB of the clip's p99 frame; not a VAD, so speech and loud noise alike) where the output is more than 20 dB
below the input. CI = 1000-resample bootstrap over clips (`vaani.report.ci`).

## Results (from summary.csv)

| source | system | condition | n | OVRL [95% CI] | dOVRL vs raw | atten>20dB [95% CI] |
|---|---|---|---|---|---|---|
| MAD comm. | raw | mono | 192 | 1.79 [1.72, 1.88] | 0 | 0.00 |
| MAD comm. | gtcrn_pretrained | mono | 192 | 2.25 [2.19, 2.31] | +0.45 [0.40, 0.50] | 0.30 [0.27, 0.33] |
| MAD comm. | r7 | ref_zero | 192 | 2.04 [1.98, 2.09] | +0.25 [0.19, 0.29] | 0.15 [0.13, 0.17] |
| MAD comm. | r7 | ref_dup | 192 | 1.98 [1.96, 2.00] | +0.19 [0.11, 0.27] | 1.00 [0.99, 1.00] |
| web WAV | raw | mono | 1 | 1.12 | 0 | 0.00 |
| web WAV | gtcrn_pretrained | mono | 1 | 1.95 | +0.83 | 0.33 |
| web WAV | r7 | stereo_LR | 1 | 1.61 | +0.49 | 0.79 |
| web WAV | r7 | ref_zero | 1 | 1.81 | +0.70 | 0.14 |
| web WAV | r7 | ref_dup | 1 | 2.16 | +1.04 | 1.00 |

## Reading

- Neither condition is r7's design point (a real two-mic capture); these rows show how r7 behaves off it.
- `ref_dup` removes nearly every active frame (atten>20dB 1.00; mean attenuation 41 dB on MAD, 63 dB on the web WAV)
  yet DNSMOS OVRL still rises over raw. DNSMOS alone therefore cannot catch speech erasure on these inputs, which is
  why the attenuation proxy sits next to it, and why the clean-reference G4 test (`results_r2/field/`) gates real-audio
  claims. That DNSMOS rewards near-silent output here is *inferred* from these two columns, not separately tested.
- On the web WAV as recorded (`stereo_LR`), r7 cuts 79% of active frames by more than 20 dB; with the reference
  zeroed, 14%. This matches the diag_webaudio finding that r7 erases speech at near-equal mic levels (plan 11.1).
- `gtcrn_pretrained` has the highest OVRL on MAD communication and cuts 30% of active frames: the proxy cannot tell
  removed noise from removed speech, so that 30% is not a speech-loss figure.

TBD: speech-level metrics (WER, Whisper word survival) on these clips need faster-whisper, which is not in .venv.
