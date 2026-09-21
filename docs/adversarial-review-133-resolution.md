# Review at 133 commits: resolution and deferred work

2026-09-21. Review baseline: `2fc2c24`. Scope: correctness, reporting and deployment
fixes; no training, model redesign or new research direction. The external review
is evidence to check, not an execution instruction.

## Implemented

| Review | Resolution |
|---|---|
| §1, uncertainty | Matrix distinguishes lower-bound passes (✓), mean-only passes (~), and mean failures (✗). Nominal cascade SNR/PESQ remain mean-only passes. Frozen evaluation and selection protocol are preserved. |
| §2, §7, negative results | Measured effects and their limitations are recorded below and in the README. No nominal quality benefit is claimed for the controller. |
| §4, deployed system | Exported the trained cascade and measured ONNX parity/timing; recorded hashes, environment, parameters and matrix MAC estimates. Contract now covers the sixth cache and the complete cascade. |
| §5, comparator count | Excluded overlapping partial snapshots; added duplicate evaluation and ASR key checks. Regenerated matrix from final CSVs and included the clean ASR reference with its hash. |
| §6, IVA comparator | Retained the raw CSV for audit, labelled every matrix occurrence as an unreproduced local integration, and withdrew it as evidence of superiority over the published method. No unverified integration fix was made. |
| §8, reporting and DSP | Stated single-seed headline coverage and transient failures, explained fault-table SNR support, and generated separate cascade DSP vectors tied to the trained checkpoint. |

## Matrix root cause and corrected comparison

The local clean ASR reference contains 2,280 rows and zero duplicate `(bucket,id)`
keys. The review's proposed ASR-merge explanation was not the observed cause.
`deepfilternet3.partial1727.csv` (1,726 rows) and
`deepfilternet3.partial582.csv` (581 rows) were included by `results_r2/*.csv`
alongside the final 2,280-row result. Their overlapping observations inflated
DeepFilterNet3 to 1,092 nominal and 720 transient items. The fixed loader excludes
partial snapshots and rejects any remaining duplicate `(system,bucket,id)`.

Corrected nominal DeepFilterNet3: n=617, SNR_out **10.736 [10.455,11.013]**,
STOI **0.877 [0.871,0.884]**, PESQ **2.018 [1.977,2.062]**. Corrected transient
n=240. The cascade remains ahead on these nominal point estimates; this is a
comparison on our frozen mixtures and the released comparator configuration.

`results_r2/asr/clean.csv` is now an included input (Whisper-generated clean
transcripts, not human labels). Its SHA256 is
`d9883667840072bf4271b13cb26e62db884530cf970d67cbf42e5d9bf9834da4`.
The README command regenerates the matrix without audio, models or ASR inference.
Git preserves the reference CSV's exact bytes so its hash is stable across
Windows/Linux checkouts.

## What the ablations establish

All numbers below are nominal means on the same 617 frozen evaluation items.

| Experiment | SNR_out (dB) | PESQ | Seed coverage / interpretation |
|---|---:|---:|---|
| r3 controller on | 13.775 | 2.327 | Mean of seeds 0/1/2 |
| r3 controller off | 13.815 | 2.341 | Mean of seeds 0/1/2; ahead on both metrics in each pair |
| r3 no limiter/blocking | 13.931 | 2.323 | One seed, versus full seed 0: 13.777 / 2.317 |
| r3 df1 | 13.753 | 2.330 | Mean of seeds 0/1/2; df3 advantage is only about 0.022 dB |
| e32 | 14.017 | 2.364 | One seed |
| r4 expanded data | 14.173 / 14.182 | 2.392 / 2.393 | Two seeds |
| r4 same-data control | 14.191 | 2.405 | One seed; wider data did not improve the tested recipe |
| Cascade on frozen r4 control | 15.150 | 2.548 | One refiner seed (4606); +0.959 dB over first stage |

Controller-on minus off averages -0.040 dB nominal. The transient difference
quoted in the review (+0.055 dB) is seed 0; it does not establish a robust benefit.
The no-DSP seed-0 transient score is 9.247 versus 9.174 full. These observations
do not prove zero population effect or statistical significance across training
seeds. Nor does a small df3 gain prove that every causal filter is incapable of
recovering phase headroom. Claims are limited to the configurations tested.

The frozen cascade still consumes the original DSP-conditioned inputs. Removing
limiter/blocking or zeroing reference inputs at deployment would change its input
distribution and invalidate the measured quality claim. The new golden vectors
therefore preserve the actual checkpoint configuration.

## Deferred training and directions

- §3: reliability-conditioned single-channel fallback, reference-fault augmentation,
  retraining and fault-bucket evaluation. The -12 dB reference-gain weakness remains
  visible; no single-channel floor is currently guaranteed.
- §7.3: refiner width/history/scale sweeps, stacked refiners and a quality/compute frontier.
- §8: `w_snr` 0.8/1.2 runs and additional seeds for the refiner, e32 and no-DSP systems.
- §9: conditional refiner/router experiments. Skipping a stateful refiner also needs
  a defined cache-update policy; identity residual initialization alone does not
  guarantee continuity when a trained refiner is toggled.

## Evaluation expansion and remaining deployment work

The honest reporting fallback in §1 is implemented. A larger evaluation is not
required to correct the report and was not used to chase a passing interval.
Increasing n can change the mean as well as the interval; a 1/sqrt(n) projection
assumes comparable independent samples and does not guarantee success. Repeated
mixtures may share speech/noise sources, which also limits an item bootstrap.
Any future expansion should have a separate, declared protocol and preserve the
existing hashed test set, rather than re-rendering it in place after seeing the result.

The cascade's [deployment contract](../deploy/CONTRACT.md) reports model-only
laptop timing. Actual target-board DSP/audio timing and checkpoint distribution
remain outstanding; no public download location is invented. The unreproduced
IVA variant still needs independent integration diagnosis before it can become a
valid comparative result.

## Validation

- 40 focused report/export/cascade/DSP/evaluation/protocol tests passed using Git
  Bash. Windows' WSL bash launcher caused two environment-dependent stub failures
  on the initial run; the same tests passed with Git Bash without code changes.
- After adding the speech-onset vector, all three golden-vector tests passed,
  including explicit checks that limiter and blocking paths affect the output.
- Independent regeneration with the README report command produced a byte-identical
  matrix. The corrected comparator counts and intervals were checked against the
  final CSVs directly.
- Trained ONNX parity error is 1.505e-6 (<1e-4); checkpoint/ONNX hashes and the
  golden-vector configuration identity were verified. `git diff --check` passed.
