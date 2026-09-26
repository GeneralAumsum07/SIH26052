# G4 field acceptance (plan 11.2): r7 baseline, guards off (r7 default path)

## Runtime guards (Rachit, 2026-09-26)

- Rule: `--guards` is **on for the r8 candidates** and **off for the r7 baseline**; every G4 result is labelled with
  its state. The guards change only Part 2 (they expose the `ref_informative` flag whose latency Part 2 times); Part 1
  never runs them, so every Part 1 table is guards off on every system.
- The r7 baseline below (`r7.json` / `r7.md`) is **guards off (r7 default path)**: it ran without `--guards`, and its
  Part 2 run predates the recorded `guards` key, which `scripts/field_accept.py` reads as off.
- `scripts/field_accept.py` writes a top-level `guards` block in the JSON (`part1`, `part2`, their labels, the rule)
  and a guard-state line above every markdown table. `r7.md` / `r7.json` were not rebuilt and carry no such line; a
  `--summarise-only` rebuild would add it with every number unchanged (checked on synthetic input against the previous
  script: every pre-existing JSON key and value identical; the markdown differs only in the guard lines, the
  `pesq_nan` column and the PESQ footnote).
- Every PESQ figure carries `pesq_nan` (clips whose isolated PESQ child failed) and the footnote below (decision 6).

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
H8 at +5 dB (targets SNR_out > 15, STOI > 0.85, PESQ† > 2.5): web 11.0 / 0.880 / 1.96; mad 14.5 / 0.934 / 2.47.

† PESQ: pesq 0.0.4 reads out of bounds in `utterance_split` on some noise-dominated inputs
(results_r2/r8/native_crash/README.md). A faulting read kills the isolated PESQ child and the clip scores NaN
(`pesq_nan`, left out of the mean); a non-faulting read returns a garbage value that cannot be detected per clip: 2 of
2,280 raw noisy eval_r2_relabel/test inputs (0.09 %) under ASan (results_r2/r8/native_crash/asan/sweep_relabel_test.tsv);
the rate on model outputs was not measured. The r7 run predates the isolation (in-process PESQ, 2026-09-24):
`pesq_nan` = 0 of its 560 Part 1 rows (a fault there would have killed the run), counted from the repo root with

    .venv/Scripts/python.exe -c "import json,math; print(sum(1 for l in open('results_r2/field/work/r7_part1.jsonl') if l.strip() for r in json.loads(l)[1] if r['pesq_out'] is None or not math.isfinite(r['pesq_out'])))"

Part 2 (reference-free, web WAV 37.2 s; guards off, r7 default path): longest stretch attenuated > 30 dB on the as_is run = 2.38 s (criterion <= 1.0 s: FAIL);
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

To score an r8 candidate: `--system stream:<cascade.onnx>@<model_config.json> --name <run> --guards` (or any `vaani.eval.enhance_fn` spec; `--guards` acts on stream systems only).

## Part 2 hooks (G4 Part 2, plan 11.2)

`vaani.asr.WordTranscriber` (faster-whisper word timestamps; `--whisper-model`, default `small`, the multilingual size
eval already uses; `--asr-device auto` = cuda when ctranslate2 sees one) and `vaani.asr.vad_speech_seconds` (the Silero
VAD shipped inside the faster-whisper wheel, run on onnxruntime: no extra download). Both load lazily; `--asr off` or
no faster-whisper makes their criteria TBD. Tested with fakes only (`tests/test_asr_hooks.py`); no Whisper weights are
ever loaded on the laptop.

Which Part 2 criteria each setup can compute:

| criterion | laptop `.venv` (no asr extra) | box, `asr` extra, `--guards` |
|---|---|---|
| longest stretch attenuated > 30 dB <= 1.0 s | yes | yes |
| Whisper confident-word survival >= 0.8 x ref_zero | TBD | yes |
| VAD speech seconds >= 0.8 x ref_zero | TBD | yes |
| mono (mono_dup) word survival >= 0.8 x ref_zero | TBD | yes |
| validity-flag latency <= 0.5 s | only with `--guards` (the guards' `ref_informative`) | yes |

Box command (repo root; the Whisper `small` weights download there on first use; the web WAV must be on the box,
path via `--web-wav` or env `VAANI_WEB_WAV`):

    uv sync --extra asr --extra fast
    uv run python scripts/field_accept.py --system stream:<cascade.onnx>@<model_config.json> --name <run> \
        --guards --asr auto --whisper-model small --asr-device auto --web-wav <path/to/abcd.wav> --workers 3

`--guards` applies to Part 2 only (Part 1 runs the default path). Without it a stream system exposes no reference-informativeness estimate and the latency row is TBD, as for r7.
