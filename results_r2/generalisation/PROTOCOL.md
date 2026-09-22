# Generalisation protocol — unseen-corpus noise

**Registered before any number was produced.** The commit that adds this file is the timestamp. If a
scoring run in this directory predates this commit, the claim it supports is void.

## 1. The question

The frozen `eval_r2` split answers *"does it work on new clips of noise it has seen?"* — splits are
content-hashed and `assign(group_id)` keeps a test clip out of training. It does not answer the
question a DRDO reviewer will actually ask:

> Does it work on noise it has never seen?

Every noise corpus in `eval_r2` — `freesound_000`, ESC-50, MAD, NOISEX-92 — is also a training
corpus. Models in this field reliably overfit to corpus artefacts (recording chain, loudness
normalisation, codec history, silence trimming) rather than to acoustics, and a within-corpus split
cannot detect that by construction.

## 2. Scope of the claim, stated before scoring

This tests **noise** generalisation, not speech generalisation.

- **Noise: wholly unseen.** The Vehicle Interior Sound Dataset never enters any training recipe, and
  `scripts/check_heldout.py` proves the disjointness by content hash rather than asserting it.
- **Speech: same corpora, disjoint clips.** LibriSpeech and Common Voice Hindi, held out by
  `assign(group_id)` exactly as in `eval_r2`.

Claiming more than this would be dishonest: a result here says the enhancement generalises to an
unseen *noise* corpus, with the speech distribution held fixed.

## 3. The corpus and why it was chosen

[Vehicle Interior Sound Dataset](https://zenodo.org/records/5606504) — 5,980 clips across bus,
minibus, pickup, sports car, jeep, truck, crossover and C-class. 48 kHz, 1.2 GB, **CC BY 4.0**, no
human voices.

Chosen because it is **stationary vehicular noise**, which lands the test on the weakest measured
bucket rather than a flattering one: `stationary` scores 5.07 dB SNR_out / 0.690 STOI at −10 dB in,
against `impulsive` at 11.52 dB / 0.851. Picking a corpus the model would find easy would make the
result meaningless.

Secondary reasons: completely different provenance from anything in training; the cleanest licence in
the shortlist; small enough to render in minutes.

**Honest caveat for the write-up:** civilian road vehicles on asphalt, not tracked armour. It is an
unseen-corpus test, not a theatre-representative one.

**Implementation note.** Clips are 3–5 s against a 6 s crop, and `render_bucket_item()` pads speech
but not noise, so a short clip reaches `mix()` short. `scan_vehicle_interior()` concatenates every
clip within a vehicle class into one long file per class, which removes the problem and makes the
vehicle class a sensible `group_id`.

## 4. The measurement

**Set:** `data/eval_gen/test`, content-hashed exactly as `eval_r2`, carrying its own `EVALSET_HASH`.

`EVALSET_HASH` digests the metadata JSON, which carries measured floats, so it is reproducible only
within a platform. A Windows and a Linux render of the same manifests and bank produced
`17a9414959bb` and `aa96a28a9955` while being the same set: no item differed in any selection field
and the audio was bit-identical. Treat a hash mismatch across machines as a question to investigate,
not as proof the sets differ.

**Grid:** `stationary_{−10, −5, 0, 5, 10, 15}` and `changing_{−10, −5, 0, 5, 10, 15}` dB,
40 items per bucket, 480 items total, plus the `clean_inf` bucket that `render_eval_sets` always
writes (40 items, no noise at all, so corpus-independent and reported separately as a sanity check
that the system does not damage clean speech).

Impulse-free buckets only. The property this set exists to protect is that nothing in it comes from a
training corpus, and in `render_eval_sets.CLASSES` both `stationary` and `changing` map to
`impulse=None`: they draw continuous noise from the manifest and nothing else, which here means only
vehicle_interior. It is the impulse-bearing buckets that would contaminate it - `recorded_*` draw
their impulses from the training corpora outright, and the synthetic bursts come from the same
generator training uses.

**Amendment 2026-09-22, before any result was scored.** The grid originally read "only the stationary
class", on the reasoning above - which separates impulse-bearing from impulse-free buckets, and says
nothing about `stationary` versus `changing`. Two things forced the correction.

First, the set could not be rendered at all: `assign()` splits 80/10/10 by group hash, and on this
corpus's eight groups it drew 5/1/2 and put the single stationary class into `train`, leaving the
test split with no stationary noise (`ValueError: no noise rows for bucket 'stationary'`). That is
fixed at source rather than worked around - a corpus that never enters a training recipe has nothing
to protect against, so `scan_vehicle_interior` now marks every row `test` and the corpus is held out
whole.

Second, even once rendered, a stationary-only grid rests on one vehicle class: 240 items of 6 s drawn
from class04's 1731 s is near-exhaustive and heavily correlated, so the bootstrap CI below would read
tighter than the evidence supports. The `changing` buckets carry seven groups. Measured on 5 s windows
inside each concatenated class, six of the eight are genuinely non-stationary (class01 0 % stationary
windows, class02 1 %, class03 15 %, class04 98 %), so `changing` reflects the corpus's real character
rather than a labelling artefact of the concatenation.

Both grids are rendered and both are reported. The stationary grid stands exactly as registered; the
changing grid is the better-powered measurement.

**Nominal envelope**, for comparability with `results_r2/matrix.md`: unclipped, no reference fault,
input SNR 0/5/10 dB.

**Metrics:** SNR_out, SI-SDR, STOI, PESQ-WB, DNSMOS OVRL, each with 1000-sample bootstrap 95 %
confidence intervals, reported per bucket and over the nominal envelope. Targets as everywhere else:
SNR_out > 15 dB, STOI > 0.85, PESQ > 2.5.

**Systems, by sha256, fixed now:**

| system | artefact | sha256 |
|---|---|---|
| deployed cascade (PyTorch) | `results_r2/runs/vaani_tier46_refiner/best.pt` | `932b086a2842eb88f4232b087fd99f8769bd102135e6b887dd6ecf38ab5aa2f6` |
| deployed cascade (ONNX) | `deploy/tier46/cascade.onnx` | `6d1b58e75592753a74b3842c1b6b6f47b5187470de0ccfe5fb45092d14ef14be` |

If an r6 arm (§5) is adopted under its own decision rule, the adopted checkpoint is scored here too
and its sha256 appended **before** that scoring run, by amending this file in a commit of its own.

## 5. Decision rules, registered in advance

**Generalisation (this set).** There is no include/exclude decision — the number is the finding — so
the rule governs reporting, not selection:

- The nominal envelope on `eval_gen` is reported **beside** `eval_r2`, never instead of it.
- The gap is reported as a **paired-by-bucket difference with intervals**, not as two numbers the
  reader must compare by eye.
- If any target that holds on `eval_r2` fails on `eval_gen`, the envelope in the README and the
  traceability doc is amended to state the corpus dependence, in the same pass as the result.

**Corpus arms (DEMAND, WHAM!), for completeness, since they are scored in the same session.** Against
the main sweep's fresh 64-epoch control, same budget, seed, box and day:

- **Include** if the arm beats the control on the nominal envelope with non-overlapping CIs **and**
  does not regress the transient-present envelope.
- **Exclude and report as a measured null** if the CIs overlap.
- **Exclude and report** if it is worse.

DEMAND and WHAM! are run as **separate arms**. Wave 4 failed to be informative precisely because four
changes were bundled and the null could not be attributed; the geometry difference below is exactly
the kind of thing that could make one help and the other hurt.

**WHAM! geometry, predicted before the run.** Binaural spacing ~17 cm puts its diffuse-field coherence
null near 343/(2 × 0.17) = **1009 Hz**, against this rig's 12 cm and 1441 Hz. Not a geometry match.
Two readings, and separate arms are what distinguish them: either the model learns a coherence
structure that does not correspond to the rig and the coherence features lose discrimination, or
training across two real spacings stops it overfitting one geometry. The null frequency is to be
verified on the audio by the same method used for DEMAND, and the measured number recorded here,
before training.

**WHAM! label, predicted before the run.** Bars, restaurants and cafés are babble-heavy, so
`stationarity_class()` will almost certainly call most of it **changing**, not stationary. WHAM! is
therefore **not** a fix for the stationary weakness; it tests inter-channel realism and nothing else.

## 6. Second held-out set, registered but not yet spent

`scan_wham()` refuses to scan the `tt` split, so WHAM! `tt` remains unseen even if the `tr` arm is
adopted. It is reserved as a second generalisation set testing the dual-channel machinery on unseen
real two-channel noise. Using it requires amending this file first.

## 7. Procedure

1. This file is written and committed. ← the registration
2. `data/eval_gen/` rendered, content-hashed. `render_eval_sets.py` refuses to write over a directory
   that already carries an `EVALSET_HASH` unless forced, so `eval_r2` cannot be damaged by this work.
3. Disjointness proven: `scripts/check_heldout.py`, and the assertion in
   `tests/test_heldout_disjoint.py`, which stops skipping once the manifest exists.
4. Score once. Report whatever it says.

## 8. What counts as an honest outcome

Both directions are reportable and both are useful:

- **Envelope holds** — a claim few teams in the room will have: it works on noise that never entered
  training, on a set registered in advance.
- **Envelope degrades** — a real limit characterised honestly, stated as a specific number, published
  before a judge finds it independently.

What is *not* permitted, and the reason this file exists: scoring several candidate corpora and
reporting the flattering one. That is the garden of forking paths; the resulting number means nothing,
and a reviewer who knows the field will ask how many were tried. One corpus, named above, scored once.
