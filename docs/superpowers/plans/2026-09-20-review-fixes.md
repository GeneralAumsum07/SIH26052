# Review fixes: tiered plan after four external reviews

Date: 2026-09-20 (10 days to the 30 Sep deadline). Repo at `13d3f0d`.

Inputs: Astra's original PLAN.md (10 Sep), the Opus-5-extra adversarial review (D1–D15),
Astra's partial findings, `defence-noise-coverage.md`, `reaching-the-targets.md`, and
five diagnostics run tonight on our own test split and checkpoints (`diag_controller`,
`diag_conditioning`, `ceiling_analysis`, `mask_phase_probe`, MAD crest factors).

Everything marked **measured** was run here. Everything marked *estimate* is a reviewer's
or my projection. Tiers are ordered by evidence-per-hour, not by reviewer.

---

## What the measurements say (the constraints every tier is built on)

| # | Finding | Status | Source |
|---|---|---|---|
| M1 | Oracle **magnitude** mask at 0 dB input tops out at 10.0 (stationary) – 14.7 dB (changing). No magnitude-only system passes 15 dB at 0 dB. | measured | `ceiling_analysis`, our test split |
| M2 | Architecture ceiling (complex mask, ERB resolution, tanh bound) is 17.8–22.5 dB at 0 dB. Target is reachable only through the complex part of the mask. | measured | same |
| M3 | `vaani_full` applies 7–14° of mask phase at ≤0 dB, 2–5° at ≥+10 dB. It is a magnitude masker. Pretrained GTCRN is the same; this is inherited. | measured | `mask_phase_probe` |
| M4 | Swapping in oracle magnitude buys +3–4 dB, STOI 0.96–0.98, PESQ 3.5–3.9. Oracle phase buys +1.7–2.5 dB. Both are needed for 15 dB. | measured | same |
| M5 | Zeroing the 18 DSP features costs 0.003 STOI; zeroing `n_hat` costs 0.0003; zeroing the reference channel costs 0.055. The DSP conditioning path is functionally inert. | measured | `diag_conditioning`, n=150 |
| M6 | ERLE goes negative above +5 dB input; `n_hat` correlates more with speech than noise at +10/+15. Controller gate-closed rate rises from 1 % at −10 dB to 34–72 % at +15 dB (inverted). Burst detector fires on 0.36 % of frames. | measured | `diag_controller` |
| M7 | MAD crest factors, event window (±100 ms around the peak, the honest number): shooting **13.3 dB**, shelling **12.3**, footsteps 18.3; speech is 13. Whole-file: 16–21. Synthetic burst 21.0, click_train 22.7, gated_noise 15.1. ESC-50 impulsive classes 13–20 (reviewer, measured). A real transient is >35 dB. **Nothing in the pipeline has ever been impulsive.** | measured | `crest_audit.py`, 40 clips/class |
| M8 | Eval sets never contain a corpus (MAD) impulse; every "impulsive" bucket is synthetic. Round-1 headline is LibriSpeech + ESC-50 only. No drone audio anywhere. | confirmed in code | `render_eval_sets.py:17`, `vaani_full.yaml:7` |
| M9 | `vaani_full` per input SNR: 6.0 / 8.1 / 10.3 / 12.6 / 14.9 / 17.2 dB; STOI passes from 0 dB, PESQ from ≈+6 dB (2.4775 at +5), SNR from ≈+11 dB. | from `results/matrix.md` | |
| M10 | 1.86 / 3.25 ms p99 per frame is a laptop ORT number. Nothing has been timed on a Pi. | `deploy/CONTRACT.md:50` | |

Consequence: the crossover where all three targets pass sits at ≈+11 dB input. Moving it
to 0 dB needs both magnitude and phase work (M1–M4) plus a controller that stops feeding
the network speech as "noise" (M6). Data-physics fixes alone will not move it (M5).

---

## Tier 0 — Done tonight (commits `917df8e` … `13d3f0d`, unpushed)

- Report counts unrecovered bursts (`inf`, not NaN). Golden vectors and eval WAVs float32.
- Manifests dedupe byte-identical audio (56 DNS, 2 MAD cross-split); on-disk manifests rewritten.
- Corpus impulses get waveform-detected onsets and peak normalisation; twin clips share the
  burst clip's `norm_gain`; `snr_achieved_db` in meta; dataset filters hoisted.
- pesq wheel win32-only, CUDA gate skippable, README.
- Suite: 88 passed, 1 skipped.

---

## Tier 1 — Evidence hygiene: no retraining, one render + one eval pass (~half a day, mostly GPU wall-clock)

Closes every "the number you report is not the number you measured" finding. Do first;
everything in Tier 2/3 is evaluated against this.

| ID | Fix | Files | Test | Why |
|---|---|---|---|---|
| 1.1 | Defence-impulse eval buckets: `CLASSES` gains an impulse-source field (`synthetic` / `corpus`); corpus path draws `noise_class == "impulsive"`, peak-normalises, `detect_onsets` — same as `dataset.py`. Adds `defence_impulsive` and `defence_impulsive+stationary` × 6 SNRs. | `scripts/render_eval_sets.py` | render_bucket_item test with a tiny impulsive manifest row | M8. Impulsive noise is named twice in the PS and has never been measured on real recordings. |
| 1.2 | Fault buckets from the adversarial review's `render_fault_sets.py`: clipped, ref-dropout, mic-mismatch at 0/+5 dB. Report them under "severe envelope", never in nominal. | `scripts/render_eval_sets.py` (merge, do not add a second script) | bucket count assertion | D6. PLAN.md §4 asks for a severe envelope; we have none rendered. |
| 1.3 | Re-render `data/eval` and `data/eval_r2` (deduped manifests, `norm_gain` twin fix, 1.1, 1.2). Record eval-set hash in the matrix header. | `scripts/render_eval_sets.py`, `scripts/run_eval_r2set.sh` | — | Current sets predate tonight's fixes. |
| 1.4 | Re-run all 11 systems on the new round-2 test split. Promote `results_r2/matrix.md` to headline; round-1 matrix becomes appendix "development ablation, consumer noise only". | `scripts/run_eval_r2set.sh`, README | — | M8 Gap 3. |
| 1.5 | Report prints 3 decimals and the envelope table: per input-SNR row, pass/fail per metric, both SNR readings (output SNR and SNR improvement) labelled. | `vaani/report.py` | `tests/test_report.py` | M9; 2.4775 currently renders as a bare ✗. |
| 1.6 | Commit the diagnostics as first-class tools: `scripts/diag_controller.py`, `diag_conditioning.py`, `ceiling_analysis.py`, `mask_phase_probe.py`. `mask_phase_probe` becomes the post-training gate ("did phase move?"). | `scripts/` | smoke test on 2 clips each | Reproducibility of every number in this document. |
| 1.7 | ASR WER: keep English-only but say so in the matrix header; Hindi rows show `n/a`, not NaN. | `vaani/report.py`, `vaani/asr.py` normaliser | — | Astra finding; a NaN in a table reads as a bug. |

Exit criterion: a matrix whose every row is rendered from the current code, with a
`defence_impulsive` bucket drawn from real recordings (MAD now, labelled with its measured
13 dB crest; D1/D2 once they land), a severe envelope, and an honest envelope table.

---

## Tier 1.5 — Data sourcing (from `dataset-sourcing-plan.md`; no self-recording — Rachit's decision)

Must land before the Tier-2 retrain starts, because it changes what `impulse_peak_db` can
honestly be set to. Each corpus passes `scripts/crest_audit.py` (or the equivalent
stationarity check) before an adapter is written, and its licence is recorded in `_row`.

| ID | Corpus | Category | Action | Gate / note |
|---|---|---|---|---|
| D1 | **Zenodo 7004819** multi-firearm gunshots (2,148 WAV, 44.1 kHz, CC-BY, no registration) | transients / overload | `scan_gunshot_zenodo()`; `noise_class="impulsive"`; keep clipped takes and mark `clip_frac` in the row | crest audit first. Recorded on Pi/USB mics — expect clipping; that is the overload bucket's source, not the crest reference. |
| D2 | **Cadre Gunshot Audio Forensics** (~10,000 shots, 21 firearms, free, registration) | transients, best provenance | Rachit registers today; adapter once files arrive | Only credible >35 dB source available without recording. If it does not arrive by day 3, r3 trains without it and the limitation is stated. |
| D3 | **DNS-5 `noise_fullband`** shards 001–003 | general noise | `fetch_data.py --dns-shards <urls>` — no new code | Largest diversity gain per hour; round 1 was starved (2.4 h ESC-50). |
| D4 | **DroneAudioDataset** (GitHub, Al-Emadi) | drone — named PS category at zero | `scan_drone()` per the plan's sketch; drone folders only (its "unknown" class is ESC-50); group by session dir | licence from the repo before download; class by `stationarity_class`. |
| D5 | **EARS**, 20 speakers (GitHub Releases, CC-NC 4.0) | speech with vocal effort (whisper → loud, 39 dB span) | `scan_ears()`; 48 kHz → 16 kHz via `to_flac16k` | Gives the "quiet / normal / stressed delivery" axis PLAN.md asks for. CC-NC: fine for the prototype, noted in the licence column. |
| D6 | LibriSpeech `train-clean-100` full 100 h | speech | raise `max_hours` in the data config | free. |
| D7 | **DEMAND** two channels at ≈12 cm spacing | real two-channel noise | adapter that emits a stereo noise row; mixer takes it as a measured noise pair for ~20 % of draws instead of the parametric path | closes the 0.998-coherence sim artifact partially; second priority after D1–D4. |
| — | Dropped: self-recorded balloons/rig session (Rachit), IndicVoices-R, FSD50K, RIR sets | | | recorded here so the omission is deliberate. |

Consequence for Tier 2.2: with D1/D2 in, `impulse_peak_db` can go to (10, 45) with a
measured source behind it; MAD `shooting`/`shelling` are relabelled — they stay in the
pool as **stationary/changing defence noise**, not impulsive, because the audit says so.
The eval bucket from Tier 1.1 (`defence_impulsive`) then draws from D1/D2, not MAD.

---

## Tier 2 — One training round aimed at the crossover (~3 days: 1 day code, 1 night train, ½ day eval)

Single retrain, single config (`vaani_full_r3`), evaluated only on the Tier-1 test split.
Everything below rides in the same round because each alone is under the noise floor of
one training run. Gate: `mask_phase_probe` must show mask phase materially above 10° at
0 dB, and ERLE must be positive at every bucket SNR **before** the GPU is spent (2.1 is
verified with `diag_controller` on the DSP alone, no training).

| ID | Fix | Files | Test | Evidence |
|---|---|---|---|---|
| 2.1 | Controller: `speech_presence` relative to a long-term channel-gain baseline instead of an absolute primary/reference level ratio; freeze threshold re-tuned so gate-closed rate does not rise with input SNR. Burst detector: sub-frame (4 ms) onset with differential jump so `jump_p99` (7–11 dB) actually crosses threshold on real bursts. | `vaani/dsp/features.py`, `vaani/dsp/controller.py`, golden vectors regenerated | `diag_controller` table: `erle_dB > 0` at every SNR, `nhat~noise > nhat~speech` everywhere; burst-frame rate > 1 % in impulsive buckets | M6. This is L1 of *reaching-the-targets*; the review's +2.1 dB / +0.74 PESQ figure is an **oracle-gate** upper bound, not a forecast. |
| 2.2 | Data physics: `impulse_peak_db` (10, 45) with a soft-clip overload model at the primary; impulses through the RIR; drop `gated_noise` as an impulse kind (crest 15 dB — it is a noise step, not an impulse); mic-mismatch ±10 dB; wind level tied to speech RMS already. | `vaani/data/mixer.py`, `vaani/data/impulses.py`, `configs/exp/*.yaml` | distribution test: p50 of achieved impulse crest in a 200-item draw > 25 dB | D2, M7. Note the honest limit: MAD source clips have 16–19 dB crest, so scaling them louder makes loud noise, not gunshots. Only synthetic `burst`/`click_train` and self-recorded transients (Tier 4) carry real crest. |
| 2.3 | Loss: add scale-dependent SNR term `−mean(clamp(snr_db, max=30))` at `w_snr = 0.2`; rebalance `spec_loss` to 50/50 complex/magnitude; magnitude compression `p` becomes a config field, default 0.5. Report SNR **and** PESQ for every setting — this term trades against PESQ. | `vaani/losses.py`, `vaani/train.py`, configs | loss unit test: clean bucket contributes bounded gradient | M3/M4. L2a–c. Config-level, zero latency. |
| 2.4 | Conditioning: replace the 18-scalar FiLM path with (T, F) input channels — `n_hat` spectrogram is already an input; add per-band coherence / residual map as a third channel, zero-init on the first conv so the warm start is preserved. Drop FiLM. | `vaani/models/vaani_net.py`, `vaani/dsp/features.py` | `diag_conditioning`: zeroing the new channels must cost > 0.01 STOI, or the path is still inert and gets removed from the pitch | M5. FiLM buys 0.003 STOI; either the DSP path earns its place here or it goes. |
| 2.5 | Deep-filtering head: per-bin complex FIR of order 3 across **past** frames, replacing the single complex multiply. Zero latency. Few-thousand params. | `vaani/models/vaani_net.py`, `vaani/models/gtcrn_stream.py`, `vaani/export.py` (ONNX parity) | parity test < 1e-4; `mask_phase_probe` phase > 10° at 0 dB after training | M1–M4. The only lever on the list that addresses the phase wall. Export parity is the risk; budget half a day for it. |
| 2.6 | Train `vaani_full_r3` on all five manifests, `num_workers 8`, CUDA. Ablation pair: `_r3` and `_r3_no_controller` (PLAN.md's decisive comparison). | `configs/exp/vaani_full_r3*.yaml`, `scripts/run_round.sh` | — | Two runs, one night on the 5060. |

Expected outcome, *estimate*: crossover moves from ≈+11 dB to somewhere in +3…+7 dB. 15 dB
at 0 dB input is **not** promised by this tier; M1 says it requires the phase work in 2.5
to actually take, which is the thing nobody has demonstrated on this backbone yet.

---

## Tier 3 — Board decision, driven by one measurement (parallel with Tier 2, hardware lead)

PLAN.md §4 already contains the gate: "Zero 2 W: claim successful deployment only after the
physical stability test. If it misses deadlines, simplify first; use a borrowed faster board
for diagnostic comparison and report the limitation honestly." Follow it literally.

| ID | Action | Output |
|---|---|---|
| 3.1 | Run `vaani/export.py::parity_and_timing` (ORT, 1 thread) on the actual Zero 2 W with `deploy/model.onnx`. Add the DSP front end timing. | `ms_per_frame_mean`, `p99`, RTF — the first real target number. |
| 3.2 | Decide with that number against the 16 ms hop: p99 < 12 ms → stay; 12–16 → stay, disable Tier 2.5 on-device; > 16 → Pi 4/5 or Orin Nano, keep the Zero 2 W row in the table as the low-cost operating point. | One line in `deploy/CONTRACT.md` and the BOM table. |

Not a software task. Blocks nothing in Tiers 1–2; blocks whether 2.5 ships in the demo.

---

## Tier 4 — Rachit's calls (sourcing, physical work, scope)

Listed so nothing is silently dropped. Each is a decision, not a task I can start.

| ID | Item | Cost | What it buys |
|---|---|---|---|
| 4.1 | Drone corpus → moved to Tier 1.5 D4 (decided). | | |
| 4.2 | Self-recorded transients — **declined** (Rachit, 20 Sep). Public gunshot sets D1/D2 stand in; the two-channel physical validation remains open and is stated as a limitation. | — | — |
| 4.3 | Critical-word Hindi / Indian-English corpus, 12 speakers, 4 held out; blinded 12-listener pilot. | Days of team time | PLAN.md's "core evidence contribution". Not started. Without it the novelty claim rests on synthetic recovery time only. |
| 4.4 | Comparators: RNNoise is in `baselines.py`; DeepFilterNet3 and H-GTCRN are not. DNSMOS as a non-intrusive column. | ½ day each | PLAN.md §4 baseline matrix names them. Optional for screening; mandatory for the final. |
| 4.5 | Frequency-domain per-bin NLMS with per-band freeze (L1 point 2). | 1–2 days DSP + new golden vectors + C port impact | Cleaner than 2.1 but a rewrite of the front end 10 days out. My recommendation: **not now**; 2.1 first, revisit if 2.1's ERLE gate still fails. |
| 4.6 | Two-stage cascade, post-filter, asymmetric windows (L5, L6, reserve). | Board-dependent | After Tier 3 only. |

---

## Sequence

```
Day 1  (20–21 Sep) Tier 1 code (1.1, 1.2, 1.5, 1.6, 1.7) → re-render → eval matrix overnight
Day 2  (21 Sep)    Tier 1.5 adapters D1, D3, D4, D5 (+ D2 if arrived); 2.1 controller + diag_controller gate; 2.3 loss; 2.2 mixer
Day 3  (22 Sep)    2.4 conditioning + 2.5 deep-filter head + export parity; train r3 overnight
Day 4  (23 Sep)    Eval r3 on Tier-1 split; mask_phase_probe + diag_conditioning gates; decide keep/drop per component
Day 5  (24 Sep)    Second round only if a gate failed and the fix is obvious; else freeze
Days 6–9           Envelope table, figures, CONTRACT.md board line, submission narrative
Day 10 (30 Sep)    Buffer
```

Tier 3 runs whenever the board is in someone's hands. Tiers 4.2–4.3 are calendar-bound by
people, not code.

## What I will not do without being told

- Push, publish, or post anything.
- Start the Tier-2 training run before the 2.1 ERLE gate passes on the DSP alone.
- Download a drone corpus before the licence is read.
- Claim any number in this document that is marked *estimate* as a result.
