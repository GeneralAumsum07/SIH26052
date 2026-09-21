# SIH26052 requirements traceability

Every clause of the problem statement against the artefact that answers it. Where a clause is
answered by a measurement, the number is the evidence; where it is not answered, the row says so
rather than paraphrasing the clause back.

Status key: **met** — implemented and measured · **partial** — implemented, with the limit stated
in the row · **not met** — no artefact, with the reason.

Numbers quoted here come from `results_r2/matrix.md` (frozen round-2 test split, 2280 items,
nominal envelope = unclipped, no reference fault, input SNR 0/5/10 dB), `deploy/CONTRACT.md`
(deployment measurements) and [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md)
(quantization and pruning).

---

## 1. Description, clause by clause

| # | Clause | Status | Evidence |
|---|---|---|---|
| 1 | AI/ML-driven noise suppression with adaptive filtering | met | VaaniNet + gated NLMS front end. The hybrid *is* the architecture: `vaani/dsp/nlms.py` feeds `n_hat` to the network as input channels 4–5, and `vaani/dsp/controller.py` gates adaptation. Not a bolt-on post-filter |
| 2 | Clean speech combined with curated defence noise datasets | met | `data/manifests/*.parquet`, dynamic mixing (`vaani/data/mixer.py`), content-hashed frozen splits. Sources and licences in `configs/data/round1.yaml` |
| 2a | — gunshots | met | Cadre Forensics (NIJ 2016-DN-BX-0183) and Zenodo 7004819, both crest-gated by `scripts/crest_audit.py` before admission |
| 2b | — drones | met | DroneAudioDataset, Al-Emadi et al., IWCMC 2019 (`sources.scan_drone`) |
| 2c | — artillery | met, by physics — see §3.1 | MAD `shelling` (measured at speech-level crest, so labelled `changing`, not `impulsive`) plus synthesised Friedlander blast waves, `vaani/data/blast.py`, `impulse_kinds: [blast, blast]` in the deployed model's training config |
| 2d | — vehicle engines | met | NOISEX-92 `leopard` / `m109` / `volvo` / `destroyerengine`, MAD `vehicle` |
| 2e | — wind | partial | ESC-50 `wind` (40 clips) plus synthetic wind augmentation at `p_wind = 0.15`: AR(1) low-passed noise at −20 to −5 dB relative, added to one microphone at random (`mixer.py`). No dedicated wind corpus and no wind-noise-specific evaluation bucket |
| 3 | At varying SNR levels | met | Training `snr_range = (−10, 15)` dB; evaluation bucketed at −10/−5/0/5/10/15 dB and reported per bucket, never only in aggregate |
| 4 | Both stationary and impulsive noise | met | `stationary` / `changing` / `impulsive` classes assigned by measurement (`sources.stationarity_class`, crest audit), bucketed and reported separately. Transient-present clips are reported as their own envelope, including where they fail |
| 5 | STFT spectrograms or raw waveform | met | STFT, `n_fft` 512 / hop 256 / sqrt-Hann, `vaani/dsp/stft.py`; one model call per 16 ms hop |
| 6 | **Full-band and sub-band features, global and local dependencies** | met — exact | Four-way correspondence, see §3.2: ERB band-split (full-band) · SFE (sub-band) · DPGRNN intra-frame path (local, across frequency) · DPGRNN inter-frame path (global, across time) |
| 7 | **Operates in the complex domain to preserve phase** | met — exact | Complex ratio mask plus a deep-filter head — a per-bin complex FIR over the current and two past frames, `out[t] = Σ_k M_k[t]·X[t−k]` (`model_cfg.df_order = 3`) — and a `w_complex` loss term on real and imaginary parts |
| 8a | Loss: SI-SNR | met | `HybridLoss`, plus an absolute-SNR term (`w_snr = 0.2`) because SI-SNR is scale-blind and the target is absolute SNR |
| 8b | Loss: L1 / L2 | met | L2 as `wmse` on complex parts and magnitude; L1 as `clean_l1` in `SpeechPreservationLoss` |
| 8c | **Loss: perceptual** | met | The `mag ** p` compressed-magnitude term, `p = 0.5` in the deployed model's config. Power-law magnitude compression is a perceptual weighting of the loudness response, not a numerical convenience — see `vaani/losses.py` and §3.3 for what it is and is not |
| 9 | Metrics: SNR, STOI, PESQ | met — exceeds | Plus SI-SDR, DNSMOS P.835, WER against a clean-reference transcript, bootstrap confidence intervals, and post-transient recovery time |
| 10a | Augmentation: random noise mixing | met | `vaani/data/mixer.py`, a fresh mixture per training item |
| 10b | Augmentation: reverberation | met | RIR banks including an armoured-compartment mode (`vaani/data/rirs.py`, `scripts/make_rir_bank.py`) |
| 10c | Augmentation: clipping | met | `p_clip = 0.10`, `overload_softclip`, and dedicated `fault_clip_mild` / `fault_clip_hard` evaluation buckets |
| 11 | Real-time mask estimation / speech reconstruction | met | Streaming ONNX export with explicit caches; 0.999 ms/frame mean, 1.696 ms p99 against a 16 ms hop budget, single-core CPU (`deploy/CONTRACT.md`). Model only — excludes DSP, STFT/iSTFT and audio I/O |
| 12 | Optionally a lightweight adaptive filter (e.g. LMS) for residual suppression | met — exceeds | NLMS upstream of the network *and* a trained residual refiner stage. A third, pre-registered residual post-filter was built, tested and rejected on its own evidence (`d7ee3b4`) |
| 13 | Deployed on embedded/edge hardware (Jetson AGX Orin or similar) | **not met** | No board. `scripts/board_timing.py` is written (numpy-only, ORT plus the DSP front end per frame) and has never been run on target. See §3.5 |
| 14a | Optimization: ONNX conversion | met | `deploy/tier46/cascade.onnx`; agreement with the batch PyTorch model 1.5e-6, tolerance 1e-4 |
| 14b | Optimization: quantization | met — **negative result** | INT8 dynamic quantization implemented (`vaani/quantize.py`) and measured. It makes this model *larger and slower*; see §3.4. Reported as a measured rejection, not omitted |
| 14c | Optimization: pruning | met — **negative result** | Global magnitude pruning implemented (`vaani/prune.py`) and swept at 10–50 %. Quality falls off well before any useful saving; see §3.4 |
| 14d | Optimization: TensorRT conversion | **not met** | Requires the Jetson. Scoped as roadmap in the same breath as clause 13 |
| 15 | Integrated with microphones (primary + reference) | partial | Dual-microphone primary/reference is the architecture's central assumption, not an option: the NLMS, the blocking matrix, the coherence features and the far-field burst test all consume the reference channel. Never validated against real microphones |
| 16 | Headphones / communication units in practical environments | **not met** | No hardware |

## 2. Expected solution, deliverable by deliverable

| Deliverable | Status | Note |
|---|---|---|
| A scalable dataset pipeline for realistic noisy–clean pairs | met | Manifests split by source recording, dynamic mixing, RIR banks, content-hashed frozen eval splits, and a crest-factor gate that every impulse corpus must pass before an adapter is written for it |
| A state-of-the-art model for robust noise suppression | met | Outperforms DeepFilterNet3, H-GTCRN and RNNoise on the frozen test split, with intervals |
| A training framework with optimised hyper-parameters and perceptual loss | met | Hyper-parameter sweeps over `w_snr`, width and the refiner grid; the perceptual term is the compressed-magnitude loss at `p = 0.5` (§3.3) |
| A real-time inference engine deployable on edge hardware | partial | The engine exists and is measured on desktop CPU at 0.999 ms per 16 ms hop. "Deployable on edge hardware" remains an argument from a compute budget, not a board measurement |
| A prototype demonstrating live cancellation with microphones / headset | **not met** | The known gap (§3.5) |
| SNR > 15 dB, STOI > 0.85, PESQ > 2.5 | partial | 15.150 dB [14.865, 15.461] · 0.922 STOI [0.917, 0.927] · 2.548 PESQ [2.498, 2.603] on the nominal envelope. STOI clears on the interval; **SNR and PESQ clear only at the point estimate** — their lower bounds sit below target. One refiner seed |
| Low latency suitable for real-time communication | met | 32 ms algorithmic (one 16 ms hop plus the STFT window's 16 ms lookahead — arithmetic from the framing, not a measurement) and ~1 ms/frame mean compute |

---

## 3. Notes on the rows that need one

### 3.1 Artillery is covered by physics, and that is the stronger answer

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

Every impulsive source is then gated on measured crest before it may enter training, and the
deployed model's config draws impulses from `blast` (`impulse_kinds: [blast, blast]`).

This answers the clause, and it reports a measurement rather than an assumption — the audit is the
reason the design changed.

### 3.2 Full-band, sub-band, complex domain: the four-way mapping

Clauses 6 and 7 describe a specific feature structure. The correspondence is exact, and is worth
stating in the statement's own vocabulary:

| Problem statement | This architecture |
|---|---|
| full-band features (global) | ERB band-split across the whole spectrum, `gtcrn.ERB` |
| sub-band features (local) | SFE, the sub-band feature extraction module, `gtcrn.SFE` |
| global dependencies | DPGRNN inter-frame path — recurrence across time |
| local dependencies | DPGRNN intra-frame path — recurrence across frequency within one frame |
| complex domain, phase preserved | complex ratio mask + deep-filter head (a per-bin complex FIR over three frames), with a complex-part loss term |

### 3.3 The perceptual loss term

`HybridLoss` compresses both the target and the estimate by `mag ** p` before the spectral MSE.
Power-law magnitude compression approximates the compressive loudness response of human hearing —
it is why the term appears in the DNS Challenge baselines and in GTCRN, and choosing `p` chooses how
strongly quiet spectral detail is weighted against loud. The deployed model's config sets `p = 0.5`
against the upstream default of 0.3, weighting quiet detail more heavily. It is a perceptual
objective and is now documented as one in `vaani/losses.py`.

The limit, stated so the claim is not overread: it is a psychoacoustic *magnitude weighting*, not a
PESQ or PMSQE surrogate. It does not optimise a perceptual metric directly. Since PESQ is one of the
two marginal targets, an explicitly metric-oriented perceptual term (PMSQE, or a PESQ proxy) is a
defensible next experiment — it is listed as such and has not been run.

### 3.4 Quantization and pruning: implemented, measured, and rejected

Both are named by the statement and neither needs hardware, so both were implemented and measured
rather than deferred. Both came back negative, and the numbers are more useful than the words would
have been. Full tables in [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md).

**INT8 dynamic quantization makes this model larger and slower.** On the trained cascade the graph
goes 474,599 → 567,969 bytes (**+19.7 %**) and latency 0.908 → 1.318 ms per frame (**×1.45**).
Per-channel weight scales do not rescue it (+20.1 %, ×1.43).

The reason is structural and worth knowing before anyone tries again on a model this size: the FP32
graph's weights account for only **210 KB of its 474 KB** — the rest is node protobuf. Quantization
cut weight bytes to 157 KB but added 150 nodes, and the node overhead exceeded the weight saving.
The latency result has the same shape: ORT's dynamic path has no integer kernel for `GRU`, which is
most of this model's recurrence, so the 29 inserted `DynamicQuantizeLinear` nodes are pure added
work on top of an unchanged float recurrence. Quantization pays on models whose weights dominate
their graph; at 52,747 parameters this one is the opposite case.

**Pruning has almost nothing to remove.** Of 52,747 parameters, only **21,952** are learned weights
in prunable layer types — the ERB analysis and synthesis matrices are 24,576 more, but they are a
fixed signal transform, not capacity, and are excluded from every sparsity level. Quality falls off
sharply and well before any useful saving; the sweep table records where. Structured pruning, which
would actually shrink the dense matrices, has little to work with at 16 channels and GRU hidden 16,
and the ONNX cache shapes in `deploy/CONTRACT.md` are keyed to those widths, so it is a re-export
and a contract change rather than a tuning knob.

Neither result is a failure to implement the clause. Both are the clause answered with a number.

### 3.5 The hardware gap

Clauses 13, 15, 16 and the prototype deliverable need a board and two microphones that this project
does not have. The position is not hidden and not padded: a live software demonstration, an honest
three-component latency decomposition, a costed BOM, and a measured compute budget — with the board
measurement explicitly labelled as not yet done, exactly as `deploy/CONTRACT.md` already labels its
timing figures.

Two things that are true and worth saying precisely:

- `scripts/board_timing.py` **already exists** — numpy-only, torch optional, ORT plus the DSP front
  end per frame. The measurement harness is written and waiting for hardware. That is a materially
  different claim from "the measurement is unplanned".
- The dual-microphone primary + reference requirement in clause 15 is the design's central
  assumption rather than a feature of it. What is missing is validation against real microphones,
  not a path to supporting them.

This is a procurement gap, not an engineering one.
