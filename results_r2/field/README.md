# G4 field acceptance (plan 11.2): r7 baseline

Command (repo root, CPU, 2 workers; 2026-09-24 18:20-18:30 IST, rc=0):

    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python scripts/field_accept.py --system r7 --name r7 --workers 2

Outputs: `r7.json` (every aggregate plus verdicts), `r7.md` (tables), `work/r7_part1.jsonl` (one row per
bed x construction x utterance x SNR, resumable), `work/r7_part2.json` (Part 2 runs).
Definitions and criteria: the docstring of `scripts/field_accept.py`. Speech loss is the diag_webaudio `frame_stats`
definition (20 ms frames where clean speech is within 30 dB of its max; lost if the projection gain < -15 dB).

System: `r7` = deploy/r7/cascade.onnx through `vaani.live.StreamEngine` with deploy/r7/model_config.json (the board path).
Utterances: 20 x 6 s, val split, seed 0, at 0 and +5 dB (pooled, n = 40 per row). Beds: `web` = C:/Users/Rachit/Downloads/abcd.wav,
`mad` = MAD communication clips (one per video, seed 0, as scripts/score_real.py).

Rows (Rachit, 2026-09-25): single-channel audio is headlined by Z / ref_zero (reference zeroed, at validity 0 where
the model takes a validity input; r7 has none, so its Z is the plain zeroed reference); M / mono_dup (duplicated
primary) is a labelled stress row. Criteria are unchanged; `r7.md` / `r7.json` were rebuilt from the cached runs
(`uv run --with numba python scripts/field_accept.py --name r7 --summarise-only`, 2026-09-25): every number is
identical to the 2026-09-24 run (json compared key by key), only the order, labels and provenance keys changed.

## Verdict: Part 1 FAIL, Part 2 FAIL (expected, plan 11.1)

Key numbers (from `r7.json` / `r7.md`):

| row | bed/cons | loss mean | loss p95 | lost run max s | STOI in -> out | dSNR dB | verdict |
|---|---|---|---|---|---|---|---|
| **single-channel headline** | web/Z | 0.022 | 0.073 | 0.18 | 0.778 -> 0.774 | +2.65 | FAIL (dSNR) |
| two-channel | web/W | 0.976 | 1.000 | 1.98 | 0.778 -> 0.316 | -1.18 | FAIL |
| two-channel | web/H8 | 0.043 | 0.148 | 0.30 | 0.778 -> 0.847 | +8.61 | FAIL (PS targets) |
| stress (duplicated primary) | web/M | 1.000 | 1.000 | 2.08 | 0.778 -> 0.535 | -0.59 | FAIL |
| **single-channel headline** | mad/Z | 0.033 | 0.078 | 0.14 | 0.736 -> 0.729 | +1.07 | FAIL (dSNR) |
| two-channel | mad/W | 0.979 | 1.000 | 2.08 | 0.736 -> 0.449 | -0.47 | FAIL |
| two-channel | mad/H8 | 0.034 | 0.137 | 1.16 | 0.736 -> 0.911 | +12.24 | FAIL (lost run, PS targets) |
| stress (duplicated primary) | mad/M | 1.000 | 1.000 | 2.08 | 0.736 -> 0.534 | -0.59 | FAIL |

gtcrn_pretrained on the same primaries: dSNR +7.82 dB (web), +2.31 dB (mad); loss 0.056 / 0.085.
H8 at +5 dB (targets SNR_out > 15, STOI > 0.85, PESQ > 2.5): web 11.0 / 0.880 / 1.96; mad 14.5 / 0.934 / 2.47.

Part 2 (reference-free, web WAV 37.2 s): longest stretch attenuated > 30 dB on the as_is run = 2.38 s (criterion <= 1.0 s: FAIL);
ref_zero 0.10 s, mono_dup 32.64 s (whole-clip suppression), swapped 3.34 s.
Whisper word survival, VAD speech seconds and mono word survival: TBD (faster-whisper is not importable in .venv).
Validity-flag latency: TBD (r7 exposes no reference-informativeness flag).

## Reading

- r7 deletes speech whenever the reference carries the speech at primary level (M: 100 % loss; W: ~98 %), the failure plan 11.1
  predicts from the fixed-array reference assumption. With the reference zeroed (Z) or strongly down-weighted (G) it keeps
  speech (loss 1-3 %) but barely enhances (dSNR +0.1 to +2.7 dB, below both +3 dB and gtcrn_pretrained - 1 dB).
- Inferred: an r8 candidate that passes G4 needs the trained reference-absent mode (spec 6.2) so that M/W behave like Z
  while keeping the H8 separation gain.
- These are single-run numbers from a loaded shared machine; the metrics are timing-independent, so they are reportable
  as the r7 G4 baseline. TBD: rerun Part 2 with faster-whisper installed to fill the ASR/VAD rows.

To score an r8 candidate: `--system stream:<cascade.onnx>@<model_config.json> --name <run>` (or any `vaani.eval.enhance_fn` spec).
