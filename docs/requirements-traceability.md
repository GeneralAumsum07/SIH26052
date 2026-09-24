# SIH26052 requirements traceability

Every clause of the problem statement against the artefact that answers it. Where a clause is
answered by a measurement, the number is the evidence; where it is not answered, the row says so
rather than paraphrasing the clause back.

Status key: **met** — implemented and measured · **partial** — implemented, with the limit stated
in the row · **not met** — no artefact, or no evaluation, with the reason.

The shipping system is the **r7 cascade**: `deploy/r7/cascade.onnx` (sha256 `e67a2c42…`), exported
from `results_r2/runs/r7_e256_wr64_refiner/best.pt` (sha256 `121f0c3d…`; full hashes in
[`deploy/CONTRACT.md`](../deploy/CONTRACT.md)).

Numbers quoted here come from:

- `results_r2/r7/*.csv`: r7 on the current render of eval_r2 (2,280 test items; nominal envelope =
  unclipped, no reference fault, input SNR 0/5/10 dB, 617 items) and on eval_gen;
- `results_r2/matrix.md`: the ablations, on the same current render;
- `results_r2/matrix_prerelabel.md`: the earlier frozen render of eval_r2, the only render the
  external baselines and WER were measured on;
- `deploy/r7/` and `deploy/tier46/*.json`: deployment measurements;
- [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md): quantization and
  pruning (earlier render).

**Every score is on synthetic mixtures** made by `vaani/data/mixer.py`. No real noisy recording is
scored yet. <!-- TBD(real): first real-recording row -->

---

## 1. Description, clause by clause

| # | Clause | Status | Evidence |
|---|---|---|---|
| T | Title: "adaptive noise cancellation (ANC)" | partial — **read as transmit-path enhancement** | VAANI cleans the wearer's transmitted voice. That reading follows the statement's own measures: SNR, STOI and PESQ of the enhanced speech, and real-time mask estimation (clause 11). Ear-side ANC is not built. It would need a reference mic outside and an error mic inside the earcup, secondary-path identification, an FxLMS-family controller, a closed-loop stability analysis, and acoustic measurement of the attenuation at the ear. See §3.7 |
| 1 | AI/ML-driven noise suppression with adaptive filtering | partial | VaaniNet + gated NLMS front end: `vaani/dsp/nlms.py` feeds `n_hat` to the network as input channels 4–5, and `vaani/dsp/controller.py` gates adaptation. The adaptive-filter contribution is not demonstrated. On its own the NLMS is worse than passthrough: `nlms_only` 1.115 dB SNR_out against 1.974 dB for raw input (`results_r2/matrix.md`, earlier render). In the r3 conditioning diagnostic (`scripts/diag_conditioning.py`; results not tracked), zeroing `n_hat` changed STOI by −0.0007 and zeroing coherence by −0.0004, against −0.061 for zeroing the reference. The NLMS also has no primary-channel delay, so it cannot cancel noise that reaches the primary first. <!-- TBD(diag): r7 conditioning diagnostic on val (scripts/diag_conditioning.py --split val) --> |
| 2 | Clean speech combined with curated defence noise datasets | met | `data/manifests/*.parquet`, dynamic mixing (`vaani/data/mixer.py`), content-hashed frozen splits. Sources and licences in `configs/data/round1.yaml`; the r7 recipe is `configs/retraining/r7_e256_wr64.yaml` |
| 2a | — gunshots | partial — training only | Zenodo 7004819 (Kabealo et al. 2023, `gunshots.parquet`) is in the r7 recipe. Its 246 test rows have never been rendered into an eval set. Cadre Forensics is not in the r7 recipe. <!-- TBD(defence): recorded-gunshot eval bucket and result --> |
| 2b | — drones | partial — training only | DroneAudioDataset, Al-Emadi et al., IWCMC 2019 (`sources.scan_drone`). `scan_drone` gives every row the same group, so all 1,332 clips land in train and none is held out |
| 2c | — artillery | partial — training only, by physics (§3.1) | Synthesised Friedlander blast waves, `vaani/data/blast.py`; the r7 recipe draws `impulse_kinds: [blast, blast, burst, click_train]` at 15–45 dB peaks. No eval set contains a blast. MAD `shelling` is in training but measured at speech-level crest, so it is labelled `changing`. <!-- TBD(defence): blast eval bucket and result --> |
| 2d | — vehicle engines | partial | NOISEX-92 `leopard` / `m109` / `volvo` / `destroyerengine` and MAD `vehicle` in training. NOISEX has 0 test rows. MAD vehicles are scored only inside the `stationary` class, which misses SNR and PESQ even in the nominal envelope (13.741 / 0.899 / 2.361, 102 clips) |
| 2e | — wind | partial | ESC-50 `wind` (40 clips) plus synthetic wind augmentation at `p_wind = 0.15`: AR(1) low-passed noise at −20 to −5 dB relative, added to one microphone at random (`mixer.py`). No dedicated wind corpus and no wind evaluation bucket |
| 2f | — helicopter rotor | **not met** as an evaluated noise | MAD `helicopter` is in training, mapped to the `stationary` class (`MAD_CLASS_MAP`, `vaani/data/sources.py`). There is no helicopter-specific score. <!-- TBD(defence): MAD helicopter test rows, per source --> |
| 2g | — sirens | **not met** | ESC-50 `siren` clips (36) are in training. No siren score exists |
| 2h | — armoured vehicles | **not met** as an evaluated condition | Tracked-vehicle noise (NOISEX `leopard`, `m109`) and armoured-compartment reverb (20 % of `bank_r3`, `scripts/make_rir_bank.py --armoured-frac 0.2`) are used in **training only**. The eval RIR bank has no armoured rooms and no armoured bucket exists |
| 3 | At varying SNR levels | partial | Training `snr_range = (−10, 15)` dB; evaluation bucketed at −10/−5/0/5/10/15 dB. No per-class × per-SNR r7 table is committed yet. At 0 dB input and below every class misses the SNR and PESQ targets. <!-- TBD(diag): committed per-class x per-SNR r7 table (results_r2/r7/) --> |
| 4 | Both stationary and impulsive noise | partial | `stationary` / `changing` / `impulsive` classes assigned by measurement (`sources.stationarity_class`, crest audit), bucketed and reported separately. The nominal impulsive classes use transients at −6 to +12 dB peak re speech RMS (median +3.3 dB), with no room path or soft-clip; the recorded ones are ESC-50 household transients. Loud transients (synthetic bursts at +24 / +36 dB peak, and a clipped overload bucket) are scored only at input 0/5 dB, where they fail all three targets: 10.866 / 0.848 / 1.797 (240 clips), against 12.771 / 0.882 / 2.103 for the matched no-burst control |
| 5 | STFT spectrograms or raw waveform | met | STFT, `n_fft` 512 / hop 256 / sqrt-Hann, `vaani/dsp/stft.py`; one model call per 16 ms hop |
| 6 | **Full-band and sub-band features, global and local dependencies** | met | Four-way correspondence, see §3.2: ERB band-split (full-band) · SFE (sub-band) · DPGRNN intra-frame path (local, across frequency) · DPGRNN inter-frame path (global, across time) |
| 7 | **Operates in the complex domain to preserve phase** | partial | Implemented: a complex ratio mask plus a deep-filter head (a per-bin complex FIR over the current and two past frames, `out[t] = Σ_k M_k[t]·X[t−k]`, `model_cfg.df_order = 3`) and a `w_complex` loss term on real and imaginary parts. Behaviour not shown: an earlier phase probe (`scripts/mask_phase_probe.py`, pre-r7 checkpoint) found the network acting as a magnitude masker, with mask phase of about 7–18°, while oracle phase would be worth about 2 dB. The deep-filter head is within seed noise (at most +0.08 dB). <!-- TBD(diag): r7 phase probe on val (scripts/mask_phase_probe.py --split val) --> |
| 8a | Loss: SI-SNR | met | `HybridLoss`, plus an absolute-SNR term (`w_snr = 0.2`) because SI-SNR is scale-blind and the target is absolute SNR |
| 8b | Loss: L1 / L2 | partial | L2 is in r7's loss as `wmse` on complex parts and magnitude. L1 is not: `clean_l1` exists only in `SpeechPreservationLoss` (`vaani/losses.py`), used by the `vaani_full_sp*` variants, and r7 trains with `loss: hybrid` |
| 8c | **Loss: perceptual** | met | The `mag ** p` compressed-magnitude term, `p = 0.5` in r7's config. Power-law magnitude compression is a perceptual weighting of the loudness response, not a numerical convenience — see `vaani/losses.py` and §3.3 for what it is and is not |
| 9 | Metrics: SNR, STOI, PESQ | met | Plus SI-SDR, DNSMOS P.835, bootstrap confidence intervals and post-transient recovery time. WER against a clean-reference transcript appears in `results_r2/matrix_prerelabel.md` for earlier-render systems only; the r7 CSVs have an empty `asr_text` column, so r7 has no WER |
| 10a | Augmentation: random noise mixing | met | `vaani/data/mixer.py`, a fresh mixture per training item |
| 10b | Augmentation: reverberation | met | RIR banks (`vaani/data/rirs.py`, `scripts/make_rir_bank.py`). The training bank `bank_r3` includes 20 % armoured-compartment rooms; that mode is used in training only and never evaluated |
| 10c | Augmentation: clipping | met | `p_clip = 0.10`, `overload_softclip`, and dedicated `fault_clip_mild` / `fault_clip_hard` evaluation buckets |
| 11 | Real-time mask estimation / speech reconstruction | met — model only | Streaming ONNX export with explicit caches, `deploy/r7/cascade.onnx`. Model time 1.135 ms mean / 1.960 ms p99 per 16 ms hop on one cloud x86 core (`deploy/r7/cascade_parity_timing.json`). Excludes DSP, STFT/iSTFT, resampling and audio I/O. <!-- TBD(runtime): complete-hop benchmark (scripts/hop_benchmark.py) and its committed result --> |
| 12 | Optionally a lightweight adaptive filter (e.g. LMS) for residual suppression | met | NLMS upstream of the network (whose benefit is not shown, row 1) *and* a trained residual refiner stage, which adds +0.72 dB / +0.10 PESQ over its backbone on eval_r2 nominal. A third, pre-registered residual post-filter was built, tested and rejected on its own evidence (`d7ee3b4`) |
| 13 | Deployed on embedded/edge hardware (Jetson AGX Orin or similar) | **not met** | Target confirmed 24 Sep 2026: NVIDIA Jetson AGX Orin 64GB, with no hardware access yet and nothing measured on it. The Raspberry Pi 5 is the development board; the r7 graph has run there, but no valid per-hop board measurement is committed: `scripts/board_timing.py`'s total p99 adds a whole-clip DSP average to the model p99 and times no STFT, resampling or I/O. See §3.5. Larger Orin tiers are projections only (§3.8) |
| 14a | Optimization: ONNX conversion | met | `deploy/r7/cascade.onnx`; agreement with the batch PyTorch model 1.11e-6, tolerance 1e-4 (`deploy/r7/cascade_parity_timing.json`) |
| 14b | Optimization: quantization | met — **negative result** | INT8 dynamic quantization implemented (`vaani/quantize.py`) and measured on the tier46 cascade (`deploy/tier46/int8_report.json`). It makes this model *larger and slower*; see §3.4. Reported as a measured rejection, not omitted |
| 14c | Optimization: pruning | met — **negative result** | Global magnitude pruning implemented (`vaani/prune.py`) and swept at 10–50 %. Quality falls off well before any useful saving; see §3.4 |
| 14d | Optimization: TensorRT conversion | **not met** | Requires the Orin. The r7 graph has 14 GRU and 18 ScatterND nodes, whose TensorRT parsing must be checked against the JetPack actually used. The VaaniFE family (§3.8) is built to export without GRU, Loop or ScatterND nodes (Gemm-cell GRUs, Slice+Concat caches); that is a graph property checked on the laptop, not a TensorRT or Orin result <!-- TBD(fe): G2 graph-gate result per tier (scripts/graph_gate.py) --> |
| 15 | Integrated with microphones (primary + reference) | partial | Dual-microphone primary/reference is the architecture's central assumption: the NLMS, the blocking matrix, the coherence features and the far-field burst test all consume the reference channel. Never validated against real microphones. On a two-channel web recording whose reference carried the talker at primary level, r7 lost 79 % of active frames (14 % with the reference zeroed); *inferred* reliance on the inter-mic level difference, under investigation. At −12 dB reference gain r7 scores 6.862 dB against 8.858 dB for a mono model. <!-- TBD(refvalid): diagnosis and reference-validity Mini result --> |
| 16 | Headphones / communication units in practical environments | **not met** | No hardware. No radio or PTT interface and no narrowband or codec evaluation, so wideband gains may not survive a military radio (*inferred*). The demo loop (`scripts/capture_loop.py`) plays the enhanced primary to both of the wearer's ears, which is the wearer's own voice; it belongs on a far-end or radio path. <!-- TBD(runtime): output routing fix --> |
| 17 | Optimised for latency **and power** constraints | **not met** for power | No power measurement exists. The only basis for an estimate is compute: r7 needs 82.460 MMAC/s of counted matrix work (§3.8). Converting that into watts needs a measured energy per operation on the target, which does not exist, so no wattage is claimed; the MMAC/s figure is the estimate's only input. NVIDIA's 15–60 W range for AGX Orin 64GB is the module's power-mode envelope, not VAANI's draw. TBD: measured board power at a stated power mode, on the Pi 5 and the Orin |

## 2. Expected solution, deliverable by deliverable

| Deliverable | Status | Note |
|---|---|---|
| A scalable dataset pipeline for realistic noisy–clean pairs | met | Manifests split by source recording, dynamic mixing, RIR banks, content-hashed frozen eval splits, and a crest-factor gate that every impulse corpus must pass before an adapter is written for it |
| A state-of-the-art model for robust noise suppression | partial | On the earlier render, the tier46 cascade beats DeepFilterNet3, H-GTCRN and RNNoise on SNR_out, STOI and PESQ, with intervals. DeepFilterNet3 beats it on DNSMOS: OVRL 2.851 against 2.702 (also SIG and BAK). The baselines have not been re-scored on the current render, and the in-house `gtcrn_finetuned` baseline had 12+6 epochs against 256+64 for VAANI, so the comparison is not equal-budget |
| A training framework with optimised hyper-parameters and perceptual loss | partial | The perceptual term is the compressed-magnitude loss at `p = 0.5` (§3.3). Hyper-parameter search is thin: width and refiner-grid configs (`configs/retraining/r5_*`) have no tracked results; the tier46 refiner selection on val is tracked (`results_r2/tier46_v2/refiner_screen/`); `w_snr` has three single-seed points, scored on the test split (`results_r2/matrix.md`) |
| A real-time inference engine deployable on edge hardware | partial | `vaani/live.py::StreamEngine` runs the r7 graph (`deploy/r7/`) one hop at a time with numpy + onnxruntime + numba, and matches the offline path within 1e-5 (`tests/test_live.py`). Export and parity are done (1.11e-6). Model time is about 1–2 ms per 16 ms hop on x86 CPU; the complete per-hop cost and any board measurement are pending. <!-- TBD(runtime): complete-hop benchmark and backend API --> |
| A prototype demonstrating live cancellation with microphones / headset | **not met** | The known gap (§3.5) |
| SNR > 15 dB, STOI > 0.85, PESQ > 2.5 | partial | r7 cascade, eval_r2 nominal (617 clips): 14.864 dB [14.565, 15.183] · 0.917 STOI [0.911, 0.922] · 2.462 PESQ [2.412, 2.511]. STOI clears on the interval; **SNR and PESQ miss at the point estimate**. Full test split (2,280): 12.78 · 0.870 · 2.138. eval_gen, the registered stationary grid (102 clips): 14.295 · 0.932 · 2.490, **missing SNR and PESQ**; the changing grid added later (99 clips): 18.37 · 0.970 · 3.153. Loud transients, input 0/5 dB: 10.866 · 0.848 · 1.797, all three fail. Per clip, all three targets are met together on 35.5 % of nominal clips, 24.1 % of the full split and 3.8 % of transient clips. The previous candidate, tier46, scores 14.753 · 0.915 · 2.473 on eval_r2; paired, r7 is +0.110 dB [+0.061, +0.163] SNR_out and −0.012 [−0.021, −0.001] PESQ. One refiner seed. <!-- TBD(diag): per-grid eval_gen table and per-clip pass-rate table, committed with their command --> |
| Low latency suitable for real-time communication | partial | 32 ms algorithmic (one 16 ms hop plus the STFT window's 16 ms lookahead — arithmetic from the framing, not a measurement), plus 4 ms resampler group delay on the 48 kHz path, and about 1 ms/hop model compute. End-to-end mic-to-ear latency, including ALSA buffering, has not been measured |

**Which eval render.** The eval_r2 scores in the targets row are from the current render of eval_r2, re-made from the crest-audit relabelled manifests (EVALSET_HASH `17a9414959bb` on Windows, `aa96a28a9955` on Linux: the same audio, the hash digests float text). [`results_r2/matrix.md`](../results_r2/matrix.md) is on the current render too. [`results_r2/matrix_prerelabel.md`](../results_r2/matrix_prerelabel.md) (which alone has the external baselines and WER) and [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md) were measured on the earlier frozen render, which differs at 606 of the 617 nominal items: the same tier46 checkpoint scores 15.150 dB there and 14.753 dB here. Comparisons within one render are valid; comparisons across the two are not.

**Held-out status.** The eval_r2 test split informed development decisions (r7 was launched after r6 scored 14.83 dB on it, `configs/retraining/r7_e256_wr64.yaml`), so it is not untouched. eval_gen's noise is hash-disjoint from the r6/r7 recipes, but r7 was scored on it without the checkpoint registration `results_r2/generalisation/PROTOCOL.md` asks for.

---

## 3. Notes on the rows that need one

### 3.1 Artillery is synthesised from physics for training; it is not yet evaluated

The public artillery recordings obtainable for this project are YouTube-sourced. `scripts/crest_audit.py`
measured MAD's `shelling` class at **12.3 dB event crest** (±100 ms around the peak) against
**13 dB for ordinary speech** — loudness-normalised and lossy-coded, so the transient the class is
named for is simply not in the audio. `MAD_CLASS_MAP` therefore labels it `changing`, not
`impulsive`, and the audit's blunt conclusion was that nothing in the pipeline had ever been
impulsive.

The fix was to generate the transient from its physics instead. `vaani/data/blast.py` implements the
Friedlander blast wave, p(t) = P₀(1 − t/T)·e^(−t/T): effectively instantaneous rise to peak
overpressure, exponential decay, zero crossing at the positive-phase duration T, then a rarefaction
phase — with T around 0.15–0.6 ms for small arms at close range and several milliseconds for
artillery. It is synthesised at 192 kHz and decimated (a near-instantaneous rise built directly at
16 kHz aliases), a ground reflection arrives 1–9 ms later inverted and attenuated, and distance
enters as a first-order low-pass. Measured event crest: **~26.5 dB**, against 15.2 dB for the
previous synthetic burst.

Every impulsive source is then gated on measured crest before it may enter training. The r7 recipe
draws impulses from `impulse_kinds: [blast, blast, burst, click_train]` at 15–45 dB peaks with the
room path and soft-clip on.

What this does not yet show: no eval set uses the blast generator (`impulses.generate(rng)` in the
eval render is called with no kind, so it draws burst, click_train and gated_noise), and the
nominal eval impulses peak at −6 to +12 dB re speech RMS. How r7 handles artillery-like transients
is therefore not measured. <!-- TBD(defence): blast bucket at 15-45 dB peaks and its score -->

### 3.2 Full-band, sub-band, complex domain: the four-way mapping

Clauses 6 and 7 describe a specific feature structure. The architectural correspondence, in the
statement's own vocabulary:

| Problem statement | This architecture |
|---|---|
| full-band features (global) | ERB band-split across the whole spectrum, `gtcrn.ERB` |
| sub-band features (local) | SFE, the sub-band feature extraction module, `gtcrn.SFE` |
| global dependencies | DPGRNN inter-frame path — recurrence across time |
| local dependencies | DPGRNN intra-frame path — recurrence across frequency within one frame |
| complex domain, phase preserved | complex ratio mask + deep-filter head (a per-bin complex FIR over three frames), with a complex-part loss term |

The last row describes what is built, not what the trained network does with it: see row 7 for the
phase-probe result.

### 3.3 The perceptual loss term

`HybridLoss` compresses both the target and the estimate by `mag ** p` before the spectral MSE.
Power-law magnitude compression approximates the compressive loudness response of human hearing —
it is why the term appears in the DNS Challenge baselines and in GTCRN, and choosing `p` chooses how
strongly quiet spectral detail is weighted against loud. r7's config sets `p = 0.5` against the
upstream default of 0.3, weighting quiet detail more heavily. It is a perceptual objective and is
documented as one in `vaani/losses.py`.

The limit, stated so the claim is not overread: it is a psychoacoustic *magnitude weighting*, not a
PESQ or PMSQE surrogate. It does not optimise a perceptual metric directly. Since PESQ is one of the
two marginal targets, an explicitly metric-oriented perceptual term (PMSQE, or a PESQ proxy) is a
defensible next experiment — it is listed as such and has not been run.

### 3.4 Quantization and pruning: implemented, measured, and rejected

Both are named by the statement and neither needs hardware, so both were implemented and measured
rather than deferred. Both came back negative, and the numbers are more useful than the words would
have been. Full tables in [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md).
All of §3.4 was measured on the **tier46** cascade (checkpoint sha256 `932b086a…`, the same
architecture and parameter count as r7) and on the earlier frozen render of eval_r2. Every
comparison in it is paired within that render, so the deltas and ratios stand; the absolute levels
are not comparable with the current-render scores in §2.

**INT8 dynamic quantization makes this model larger, slower and worse.** On the trained cascade the
graph goes 474,599 -> 567,969 bytes (**+19.7 %**) and latency 0.906 -> 1.319 ms per frame (**x1.45**)
(`deploy/tier46/int8_report.json`). Per-channel weight scales do not rescue it: 569,805 bytes
(+20.1 %), 1.310 ms (x1.45) (`deploy/tier46/int8_perchannel_report.json`). These latencies are
best-of-5 interleaved repeats on a synthetic stream, which is why the FP32 figure reads lower than
the 0.999 ms single-run tier46 measurement in `deploy/tier46/trained_cascade_timing.json`; the
ratio is the claim here, not the absolute. It also costs quality: paired per-clip against the same
checkpoint, SNR_out **-1.18 dB** [-1.27, -1.10] and PESQ **-0.166** [-0.176, -0.156], which on that
render moves PESQ from 2.548 to 2.382 and takes the system below the statement's 2.5 target. So
there is no trade to weigh -- the compressed model is worse on every axis the clause cares about.

The reason is structural and worth knowing before anyone tries again on a model this size: the FP32
graph's weights account for only **210 KB of its 474 KB** -- the rest is node protobuf. Quantization
cut weight bytes to 157 KB but added 150 nodes, and the node overhead exceeded the weight saving.
The latency result has the same shape: ORT's dynamic path has no integer kernel for `GRU`, which is
most of this model's recurrence, so the 29 inserted `DynamicQuantizeLinear` nodes are pure added
work on top of an unchanged float recurrence. Quantization pays on models whose weights dominate
their graph; at 52,747 parameters this one is the opposite case.

The FP32 ONNX row is the control for all of this. It reproduces the PyTorch checkpoint to +-0.000 on
every metric, so the INT8 deltas are attributable to quantization and not to export.

**Pruning has almost nothing to remove, and removing it costs more than it saves.** Of 52,747
parameters, only **21,952** are learned weights in prunable layer types -- the ERB analysis and
synthesis matrices are 24,576 more, but they are a fixed signal transform, not capacity, and are
excluded from every sparsity level. Measured falloff, paired per clip against the unpruned
checkpoint:

| level | weights zeroed | d SNR_out (dB) | d PESQ | still meets targets |
|---|---:|---:|---:|---|
| p10 | 2,195 | -0.13 | -0.019 | yes (STOI, and SNR/PESQ at the mean) |
| p20 | 4,390 | -0.74 | -0.070 | no (PESQ 2.478) |
| p30 | 6,586 | -2.17 | -0.401 | no (PESQ 2.147) |
| p40 | 8,781 | -8.38 | -0.732 | no (STOI 0.832 also fails) |
| p50 | 10,976 | -14.71 | -1.359 | no (SNR_out 0.44 dB; the model is destroyed) |

Every delta's 95 % interval excludes zero, so even p10's -0.13 dB is a real loss rather than noise.
The collapse between p30 and p50 is not graceful degradation; it is the model failing.

The decisive point is that none of this buys anything. Zeroed weights in a dense graph still occupy
their bytes and still get multiplied -- unstructured sparsity needs sparse kernels ORT's CPU provider
does not apply here, so p10 costs 0.13 dB for a 0 % saving in size or latency. Structured pruning,
which would actually shrink the dense matrices, has little to work with at 16 channels and GRU hidden
16, and the ONNX cache shapes in `deploy/CONTRACT.md` are keyed to those widths, so it is a re-export
and a contract change rather than a tuning knob.

Neither result is a failure to implement the clause. Both are the clause answered with a number.

### 3.5 The hardware gap

Clauses 13, 15, 16 and the prototype deliverable need a board and two microphones that this project
does not have in a headset. The position is not hidden: a live software path (`scripts/capture_loop.py`),
the 32 ms algorithmic latency derived from the framing, and a counted compute budget — with every
board figure labelled as not yet measured.

- **Target and development board.** The deployment target is the NVIDIA Jetson AGX Orin 64GB
  (confirmed 24 Sep 2026; no hardware access yet). The Raspberry Pi 5 is the development board.
- **The per-hop measurement is not done yet.** `scripts/board_timing.py` exists (numpy-only, torch
  optional), but its total p99 is not a per-hop cost (row 13). <!-- TBD(runtime): complete-hop benchmark and a board run -->
- **Real microphones.** The dual-microphone primary + reference requirement in clause 15 is the
  design's central assumption. What is missing is validation against real microphones, and the one
  real two-channel recording tried so far exposed a failure (row 15), so this is an engineering gap
  as well as a procurement one.

### 3.6 Corpus licensing

[`docs/licences.md`](licences.md) is generated from the manifests and a training recipe, so it
states what training actually reads rather than what a hand-maintained table remembers.
<!-- TBD(hygiene): confirm docs/licences.md is regenerated from the r7 recipe, and cite NOTICE and the DNSMOS attribution -->

Three corpora in the deployed recipe are not commercially usable: EARS and ESC-50 are CC BY-NC, and
MAD is YouTube-sourced, which is an absence of licence rather than a restrictive one. Three more have
unresolved terms: DNS noise is licensed per clip, DroneAudioDataset is citation-on-use, and NOISEX-92's
SPIB redistribution terms are unstated.

Non-commercial licensing is fine for a competition and for research. It is not fine for a claim that
the system is ready to transfer to a deployable defence product. A fielded version would retrain on
licensed or government-collected data; because a recipe is a list of manifest paths and nothing in
the model or the DSP front end is tied to a corpus, that substitution is mechanical.

### 3.7 What "ANC" means here

The title's "adaptive noise cancellation (ANC)" is read as **transmit-path speech enhancement**:
VAANI cleans the voice that the wearer sends. The statement's own measures support that reading:
SNR, STOI and PESQ are measures of the enhanced speech, and clause 11 asks for mask estimation and
speech reconstruction. The adaptive part is the NLMS front end (row 1).

Ear-side ANC, which cancels noise at the wearer's ear, is a different subsystem and none of it is
built:

- a reference mic outside the earcup and an error mic inside it;
- secondary-path identification (speaker to error mic), tracked as the fit changes;
- an FxLMS-family controller with far lower latency than a 16 ms hop;
- closed-loop stability analysis and margins;
- acoustic measurement of the attenuation delivered at the ear.

Timing of the communications model says nothing about hearing protection.

### 3.8 Compute scaling and the Orin target

The chosen scaling family is **VaaniFE** (dual-mic, adapted from FastEnhancer). Its Pi-Mini is
the only tier to be trained, in the r8 retrain, which has not run yet. Orin-Mid, Orin-Large and
Orin-Large+ are **projections**: counts, graph checks on untrained exports and laptop timing only.
No Orin latency, power or quality is measured or claimed, and no larger model is trained.
<!-- TBD(fe): VaaniFE tier table (params, MMAC/s, state bytes, folded nodes, laptop ORT CPU timing) from scripts/fe_tiers.py and results_r2/fe_tiers/ -->
<!-- TBD(fe): G2 graph-gate results per tier (scripts/graph_gate.py) -->

The r7 width profiles remain **measurements, not the scaling path**. Counted with the convention in
`deploy/CONTRACT.md`, C16 (r7, trained) is 52,747 entries and 82.460 MMAC/s; the **untrained**
variants are C32 109,883 / 147.278, C64 318,619 / 379.586 and C96 653,307 / 748.790. None has a
quality result. Width scaling of r7 is dispatch-bound on a CPU (the folded node count does not
change with width), and its GRU and ScatterND nodes are what the VaaniFE graph rules remove for
TensorRT.
<!-- TBD(refvalid): committed source for the r7 width-profile counts (scripts/audit_budget.py output); the original profile JSON is local-only --> <!-- TBD(refvalid): budget audit of the reference-validity Mini (scripts/audit_budget.py) -->
