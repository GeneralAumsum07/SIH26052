# SIH26052 requirements traceability

Every clause of the problem statement against the artefact that answers it. Where a clause is
answered by a measurement, the number is the evidence; where it is not answered, the row says so
rather than paraphrasing the clause back.

Status key: **met** — implemented and measured · **partial** — implemented, with the limit stated
in the row · **not met** — no artefact, or no evaluation, with the reason.

Two systems are traced, and every row says which one its evidence is from:

- **VAANI-LD** (r8, the proposed system): the low-delay front end with the decoupled-cadence NLMS, a
  streaming VaaniFE network and the asymmetric STFT of the Arm B contract
  `vaanife_ld_asym512_h128_s160_v1` (32 ms analysis, 8 ms hop, **10 ms algorithmic delay**). The
  flagship configuration is `inputs: pr_nhat` (`configs/retraining/r8_ld_ablations/ld_b_nhat_s{0,1}.yaml`)
  and is **not trained yet**. The same contract without `n_hat` (`inputs: pr`) is trained as the
  Arm B Mini (200,000 steps, `r8_runs_final/r8_ld_fe_mini_armb/`, local only).
- **r7** (the 32 ms baseline): `deploy/r7/cascade.onnx` (sha256 `e67a2c42…`), exported from
  `results_r2/runs/r7_e256_wr64_refiner/best.pt` (sha256 `121f0c3d…`; full hashes in
  [`deploy/CONTRACT.md`](../deploy/CONTRACT.md)). It is the only system with test-split, defence-noise
  and real-recording scores, and the only exported graph committed under `deploy/`.

Numbers quoted here come from:

- `results_r2/r7/*.csv`: r7 on the current render of eval_r2 (2,280 test items; nominal envelope =
  unclipped, no reference fault, input SNR 0/5/10 dB, 617 items) and on eval_gen;
- `results_r2/matrix.md`: the ablations, on the same current render;
- `results_r2/matrix_prerelabel.md`: the earlier frozen render of eval_r2, the only render the
  external baselines and WER were measured on;
- `deploy/r7/` and `deploy/tier46/*.json`: r7-era deployment measurements;
- [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md): quantization and
  pruning (earlier render);
- [`results_r2/r7/breakdown.md`](../results_r2/r7/breakdown.md) and
  [`results_r2/generalisation/per_grid.md`](../results_r2/generalisation/per_grid.md): r7 per class,
  per input SNR, per clip and per eval_gen grid (`uv run python scripts/r7_breakdown.py`);
- [`results_r2/r7/diag/README.md`](../results_r2/r7/diag/README.md): r7 conditioning and mask-phase
  diagnostics on eval_r2 **val**;
- [`results_r2/defence/README.md`](../results_r2/defence/README.md): the defence-noise set;
- [`results_r2/real/table.md`](../results_r2/real/table.md) and
  [`results_r2/field/README.md`](../results_r2/field/README.md): real recordings (reference-free
  proxies) and the G4 field-acceptance baseline;
- [`results_r2/r8/README.md`](../results_r2/r8/README.md),
  [`results_r2/fe_tiers/README.md`](../results_r2/fe_tiers/README.md): r8 readiness (budgets, data
  gates, the pre-registered test set) and the VaaniFE tiers;
- [`results_r2/r8/r7_refconditions_val.md`](../results_r2/r8/r7_refconditions_val.md) and
  [`results_r2/r8/r8_fe_mini_refconditions_val.md`](../results_r2/r8/r8_fe_mini_refconditions_val.md):
  r7 against the trained r8 C0 Mini under 15 reference conditions on val;
- [`results_r2/r8_ld/gate0/README.md`](../results_r2/r8_ld/gate0/README.md) and
  `results_r2/r8_ld/native/arm64_build.json`: the low-delay latency gate and the native runtime;
- the r8 training runs in `r8_runs_final/` (local only, gitignored): the in-training val monitor
  (`scorer_state.json`) and the Pi bundle `r8_runs_final/pi_bundle/tiers_armb/`.

**Every SNR/STOI/PESQ score is on synthetic mixtures** made by `vaani/data/mixer.py`. The only real
recordings scored are 192 MAD "communication" clips and one two-channel web WAV, with
reference-free proxies (DNSMOS P.835 and an attenuation proxy), all on r7: on MAD, DNSMOS OVRL 1.79
raw, 2.25 `gtcrn_pretrained`, 2.04 r7 with the reference zeroed, 1.98 r7 with the primary duplicated
into the reference, which also cuts 100 % of active frames by more than 20 dB
(`results_r2/real/table.md`). No physical headset recording exists yet (`python -m vaani.physical`,
schema [`docs/physical_test_schema.md`](physical_test_schema.md)).

**Selection is on val only for r8.** The eval_r2 test split is burned; the pre-registered r8 test set
(`data/eval_r8_test`, hash `ed024af085a2`, 2,308 items, `results_r2/r8/testset/PROTOCOL.md`) is scored
once, after selection, and **is not scored yet**.

---

## 1. Description, clause by clause

| # | Clause | Status | Evidence |
|---|---|---|---|
| T | Title: "adaptive noise cancellation (ANC)" | partial — **read as transmit-path enhancement** | VAANI cleans the wearer's transmitted voice. That reading follows the statement's own measures: SNR, STOI and PESQ of the enhanced speech, and real-time mask estimation (clause 11). Ear-side ANC is not built. It would need a reference mic outside and an error mic inside the earcup, secondary-path identification, an FxLMS-family controller, a closed-loop stability analysis, and acoustic measurement of the attenuation at the ear. See §3.7 |
| 1 | AI/ML-driven noise suppression with adaptive filtering | partial — **built; the adaptive filter's gain is unmeasured** | **VAANI-LD:** the decoupled-cadence NLMS (`vaani/dsp/decoupled_nlms.py`) runs on 32-sample chunks behind the blocking matrix, gated by the unchanged legacy controller on its own 256-sample cadence. It adds no delay, is hop-invariant and is bit-exact with `pipeline.run` (`tests/test_decoupled_nlms.py`). Its noise estimate `n_hat` enters VaaniFE as two extra planes, multiplied by reference validity. Decisions (R8 runbook): the NLMS stays in the default path (Rachit, 2026-09-25), and on the low-delay path it is measured as an input ablation, `ld_b_nhat_s{0,1}` against `ld_b_s{0,1}` (D5 option 1, Rachit, 2026-09-27), because the Arm A Mini with `n_hat` is over the 60,000-entry budget. **`ld_b_nhat` is not trained yet** (queued in `configs/retraining/laptop_queue.txt`), so no low-delay result shows what `n_hat` adds. **r7:** on its own the NLMS is worse than passthrough (`nlms_only` 1.115 dB SNR_out against 1.974 dB raw, `results_r2/matrix_prerelabel.md`), and zeroing `n_hat` inside r7 changes SNR_out by −0.030 dB [−0.051, −0.008] on val (`results_r2/r7/diag/README.md`), against −7.094 dB for zeroing the reference: r7 leaned on the inter-mic level cue instead (row 15) |
| 2 | Clean speech combined with curated defence noise datasets | met | `data/manifests/*.parquet`, dynamic mixing (`vaani/data/mixer.py`), content-hashed frozen splits. Sources and licences in `configs/data/round1.yaml` and `configs/data/r8_datasets.yaml`. r8 trains on mixer v2: SPL-calibrated battlefield scenes in which SNR is an output of the scene, plus DEMAND two-mic pairs, AVQ drone, C3GD, FSD50K and Lombard GRID on top of r7's corpora |
| 2a | — gunshots | partial — trained on; targets not met on recorded shots (r7) | Zenodo 7004819 (Kabealo et al. 2023, `gunshots.parquet`) is in the r7 and r8 recipes. The defence set's `gunshot` category (local render `data/eval_defence/test`, hash `d033568bdf98`) places recorded test-split shots from Zenodo 7004819 and Cadre Forensics (178 unique) at 15–45 dB peak re speech RMS on MAD beds. r7 never meets all three targets there; at 15 dB input SNR_out is 14.90 [13.26, 16.59] and PESQ 2.49 [2.29, 2.72] (`results_r2/defence/table.md`). No r8 model is scored on the defence set |
| 2b | — drones | partial — trained (r8); held-out score pending | DroneAudioDataset (Al-Emadi et al., IWCMC 2019) trained r7 with no held-out clip. For r8, AVQ drone is in training, and 52 DroneAudioDataset recordings (307 clips) are held out of every training pool (`configs/data/r8_heldout_exclude.json`) for the pre-registered r8 test set (`results_r2/r8/testset/PROTOCOL.md`), **not scored yet** |
| 2c | — artillery | partial — trained by physics (§3.1); evaluated on synthetic blasts (r7) | Synthesised Friedlander blast waves, `vaani/data/blast.py`; the r7 and r8 recipes draw `impulse_kinds: [blast, blast, burst, click_train]` at 15–45 dB peaks. MAD `shelling` is in training but measured at speech-level crest, so it is labelled `changing`. The defence set's `blast_artillery` category uses physics-v2 blasts: r7 meets all three targets there only at 15 dB input (16.49 / 0.944 / 2.78) (`results_r2/defence/table.md`) |
| 2d | — vehicle engines | partial | NOISEX-92 `leopard` / `m109` / `volvo` / `destroyerengine` and MAD `vehicle` in r7 training. In eval_r2, MAD vehicles are scored only inside the `stationary` class, which misses SNR and PESQ even in the nominal envelope for r7 (13.741 / 0.899 / 2.361, 102 clips). The defence set's `vehicle` category meets all three on the interval at 10 and 15 dB input only (r7). NOISEX `m109` and `destroyerengine` (with `buccaneer2`) are held out for the r8 test set (not scored) |
| 2e | — wind | partial — modelled; no score | ESC-50 `wind` (40 clips) plus synthetic wind augmentation. Mixer v2 adds a wind model (AR(5) with gust states, independent per mic under a shared envelope) and a `windy_ridge` scene, which the pre-registered r8 test set contains as its gusty-wind bucket (not scored). No wind score exists for any system |
| 2f | — helicopter rotor | partial | MAD `helicopter` is in training. The defence set's `helicopter` category (68 MAD test beds) meets all three targets on the interval only at 15 dB input for r7 (19.05 / 0.952 / 2.91); at 10 dB PESQ misses (2.49) |
| 2g | — sirens | partial — anecdotal | ESC-50 `siren` clips (36) are in training. The defence set's `siren` category has two test recordings: r7 meets all three targets at 10 and 15 dB input. Two recordings support no general claim |
| 2h | — armoured vehicles | partial — trained; **not evaluated** | Tracked-vehicle noise (NOISEX `leopard`, `m109`) and armoured-compartment rooms are used in training: 20 % of `bank_r3` (r7) and the M6 armoured rooms of `bank_r8` (every r8 mixer v2 config). No armoured-compartment evaluation condition exists; the defence set's `vehicle` category uses MAD `vehicle` beds in ordinary rooms |
| 3 | At varying SNR levels | partial | Training `snr_range = (−10, 15)` dB (r7); under mixer v2 SNR is a scene output. Evaluation bucketed at −10/−5/0/5/10/15 dB. For r7 at 0 dB input and below every class misses the SNR and PESQ targets; the all-three pass rate is 9.2 % at 0 dB, 7.1 % at −5 dB and 2.5 % at −10 dB (`results_r2/r7/breakdown.md`). No r8 per-SNR table exists yet |
| 4 | Both stationary and impulsive noise | partial | `stationary` / `changing` / `impulsive` classes assigned by measurement (`sources.stationarity_class`, crest audit), bucketed and reported separately. Loud transients (synthetic bursts at +24 / +36 dB peak, and a clipped overload bucket) at input 0/5 dB fail all three targets for r7: 10.866 / 0.848 / 1.797 (240 clips), against 12.771 / 0.882 / 2.103 for the matched no-burst control. The VAANI-LD front end adds a delay-free 32-sample limiter ahead of the network |
| 5 | STFT spectrograms or raw waveform | met | **VAANI-LD:** an asymmetric STFT, analysis window K = 512 (32 ms), hop H = 128 (8 ms), synthesis support L = 160 (10 ms) (`vaani/audio_contract.py`, `vaani/dsp/low_delay_stft.py`); one network step per 8 ms hop. **r7 / C0:** `n_fft` 512 / hop 256 / sqrt-Hann, `vaani/dsp/stft.py` |
| 6 | **Full-band and sub-band features, global and local dependencies** | met | See §3.2. **VaaniFE:** frequency convolutions over sub-band windows (local, sub-band) · self-attention across all frequency tokens of a frame (full-band, global across frequency) · a time GRU per frequency token (global across time). **r7:** ERB band-split · SFE · DPGRNN intra- and inter-frame paths |
| 7 | **Operates in the complex domain to preserve phase** | partial | **VaaniFE:** an unbounded complex mask on the compressed primary spectrum, which corrects phase as well as magnitude, plus a 3-tap complex deep filter over the lowest 144 bins (0–4.5 kHz, lags 0, 2, 4), with a complex-part loss term. No phase diagnostic has been run on an r8 model. **r7:** complex ratio mask + 3-frame deep filter; its phase probe on val measures a mean mask phase of 21.73°, and oracle magnitude would add +4.176 dB against +2.164 dB for oracle phase, so magnitude error dominated (`results_r2/r7/diag/README.md`) |
| 8a | Loss: SI-SNR | partial | **r7:** `HybridLoss` uses SI-SNR plus an absolute-SNR term (`w_snr = 0.2`). **r8:** the FE loss has no SI-SNR term; it uses a clamped **absolute**-SNR term (`w_snr = 0.002`) because the target is absolute SNR and SI-SNR is scale-blind (`vaani/losses.py::FELoss`) |
| 8b | Loss: L1 / L2 | met | **r8:** waveform L1 (`w_wave = 0.2`) and squared error on compressed magnitude (`w_mag`) and complex parts (`w_complex`), computed after re-synthesis through the real low-delay synthesis window (`vaani/enhance_low_delay.py::ResynthesisFELoss`). **r7:** L2 only (`wmse`) |
| 8c | **Loss: perceptual** | met | **r8:** a differentiable PESQ term (`torch_pesq`, `w_pesq = 0.001`, `pesq_required: true` so no run trains without it) plus power-law compressed magnitude (p = 0.3) and an asymmetric over-suppression weight (κ = 3, applied before squaring, so removing speech costs κ² = 9 times a same-size error of leaving noise). **r7:** compressed magnitude at p = 0.5 only. See §3.3 |
| 9 | Metrics: SNR, STOI, PESQ | met | Plus SI-SDR, DNSMOS P.835, bootstrap confidence intervals, post-transient recovery time and, for r8, a composite val pass rate used for checkpoint selection. WER exists only for earlier-render systems (`results_r2/matrix_prerelabel.md`); no r7 or r8 WER |
| 10a | Augmentation: random noise mixing | met | `vaani/data/mixer.py`, a fresh mixture per training item; mixer v2 builds a battlefield scene per item |
| 10b | Augmentation: reverberation | met | RIR banks (`vaani/data/rirs.py`, `scripts/make_rir_bank.py`): `bank_r3` (r7) and `bank_r8` (r8, with M6 armoured rooms) |
| 10c | Augmentation: clipping | met | `p_clip = 0.10`, `overload_softclip`, and dedicated `fault_clip_mild` / `fault_clip_hard` evaluation buckets; mixer v2 adds a microphone front end with clipping |
| 11 | Real-time mask estimation / speech reconstruction | met — model and runtime; **not end to end** | **VAANI-LD:** a loop-free streaming step graph per 8 ms hop, run by the Python engine (`vaani/low_delay_live.py`, ORT + numba, accepts `pr_nhat`) and by the native C++ runtime (`native/vaani_ld`, golden-vector parity, arm64 build: `results_r2/r8_ld/native/arm64_build.json`). The native runtime has **no NLMS stage yet**. On a Raspberry Pi 5 (2026-10-05, one core, native runtime, whole hop including the 48 kHz resampler pair, front end, analysis, ORT step and synthesis) the untrained Arm B `pr` tier graphs take 0.50 / 1.05 / 2.32 / 3.04 ms mean per 8 ms hop (Mini / Mid / Large / Large+); Large+ under SCHED_FIFO peaks at 3.63 ms. These numbers are from the board's terminal output; **TBD: commit `pi_results/tiers_armb/*.json`**. **r7:** model time 1.135 ms mean / 1.960 ms p99 per 16 ms hop on one x86 core (`deploy/r7/cascade_parity_timing.json`) |
| 12 | Optionally a lightweight adaptive filter (e.g. LMS) for residual suppression | partial | The NLMS sits upstream of the network on both pipelines (row 1), and its benefit is not shown yet on the low-delay path. r7 also has a trained residual refiner (+0.72 dB / +0.10 PESQ over its backbone on eval_r2 nominal). VaaniFE drops the refiner. A pre-registered residual post-filter was built, tested and rejected on its own evidence (`d7ee3b4`) |
| 13 | Deployed on embedded/edge hardware (Jetson AGX Orin or similar) | partial — **Pi 5 timed; Orin not** | Target: NVIDIA Jetson AGX Orin 64GB (confirmed 24 Sep 2026), with no hardware access and nothing measured on it. The Raspberry Pi 5 is the development board: the Arm B tier graphs run there in real time with headroom (row 11), and the Python ORT route at 1 and 4 threads had 0 late hops. Not yet on a board: a trained graph, the `n_hat` graph and the NLMS stage. Gate 0a (latency eligibility) is `pending_board` (§3.5). See §3.8 |
| 14a | Optimization: ONNX conversion | met | VaaniFE step graphs exported loop-free (Gemm-cell GRUs, static shapes, Slice+Concat caches) with ORT-vs-torch and streaming-vs-offline parity within 1e-5 (G2); the trained Arm B Mini export is in its run's `export/`. r7: `deploy/r7/cascade.onnx`, agreement 1.11e-6 |
| 14b | Optimization: quantization | met — **negative result** (r7-era) | INT8 dynamic quantization implemented (`vaani/quantize.py`) and measured on the tier46 cascade: it makes that model *larger and slower* (§3.4). Not re-run on VaaniFE |
| 14c | Optimization: pruning | met — **negative result** (r7-era) | Global magnitude pruning implemented (`vaani/prune.py`) and swept at 10–50 % on tier46. Quality falls off well before any useful saving (§3.4). Not re-run on VaaniFE |
| 14d | Optimization: TensorRT conversion | **not met** | Requires the Orin. The VaaniFE family is built to export without GRU, Loop or ScatterND nodes; G2 (`scripts/graph_gate.py`) passes on all four tier exports (116–200 folded nodes, 0 loops, 0 ScatterND) and fails on r7 (539 folded nodes, 14 GRU, 18 ScatterND) (`results_r2/fe_tiers/README.md`). That is a graph property, not a TensorRT result. No TensorRT build was attempted |
| 15 | Integrated with microphones (primary + reference) | partial | Dual-microphone primary/reference is the architecture's central assumption, and mixer v2 plus reference corruption with a per-frame validity label make one VaaniFE model its own mono fallback. On 148 val clips × 15 reference conditions (`scripts/eval_refvalid.py`) the r8 C0 Mini keeps working where r7 collapses: talker leak into the reference (−2 dB) 7.99 dB SNR_out and 52 failing clips against r7's 0.13 dB and 148; equal level on both mics 6.54 dB / 65 against −0.02 dB / 148; at the price of 3.7 dB on the clean-reference case (10.28 against 13.93) (`results_r2/r8/*_refconditions_val.md`). The low-delay arms have not been run through this table. **Never validated against real microphones**; the one real two-channel web recording tried with r7 exposed the level-cue failure (`results_r2/real/table.md`), and the G4 field-acceptance baseline fails for r7 (`results_r2/field/README.md`) |
| 16 | Headphones / communication units in practical environments | **not met** | No hardware. No radio or PTT interface and no narrowband or codec evaluation, so wideband gains may not survive a military radio (*inferred*). The enhanced output is the wearer's own voice, so the live loop (`scripts/capture_loop.py`) plays nothing by default (`--output-route off`) |
| 17 | Optimised for latency **and power** constraints | partial for latency · **not met** for power | **Latency:** 10 ms algorithmic (Arm B) + 3.63 ms worst-case Large+ compute on the Pi = 13.6 ms, under the 15 ms target; this is a sum, not an acoustic measurement (§3.5). **Power:** no measurement exists. The only input to an estimate is compute: the VAANI-LD Mini with `n_hat` is 89.220 MMAC/s, inside the 90.706 MMAC/s Pi budget (`results_r2/r8/budget.md`). NVIDIA's 15–60 W range for AGX Orin 64GB is the module's power-mode envelope, not VAANI's draw. TBD: measured board power at a stated power mode |

## 2. Expected solution, deliverable by deliverable

| Deliverable | Status | Note |
|---|---|---|
| A scalable dataset pipeline for realistic noisy–clean pairs | met | Manifests split by source recording, dynamic mixing, mixer v2 battlefield scenes, RIR banks, content-hashed frozen eval splits, a crest-factor gate for impulse corpora, a Freesound-sibling held-out check (`scripts/heldout_freesound.py`), and a registry-driven fetcher (`scripts/r8_datasets.py`) |
| A state-of-the-art model for robust noise suppression | partial | On the earlier render, the r7-era tier46 cascade beats DeepFilterNet3, H-GTCRN and RNNoise on SNR_out, STOI and PESQ; DeepFilterNet3 beats it on DNSMOS (OVRL 2.851 against 2.702). The baselines have not been re-scored since, and no r8 model has been compared with any external baseline |
| A training framework with optimised hyper-parameters and perceptual loss | partial | The r8 framework: the FE loss with a differentiable PESQ term (row 8c), EMA weights (0.999), composite val selection, `torch.compile` + CUDA-graph steps, an async scorer, and a resumable multi-GPU queue (`scripts/run_r8.sh`) or a single-GPU one (`scripts/laptop_queue.py`). Hyper-parameter evidence: the r8 low-delay pilots (Stage 1 arms, Stage 2 items, P4 ablations; `configs/retraining/r8_ld_ablations/README.md`). Several pilots are unfinished and **the registered comparison (`scripts/compare_r8_ld.py`) has not been run** |
| A real-time inference engine deployable on edge hardware | partial | VAANI-LD runs one hop at a time in the Python engine (with NLMS) and in the native C++ runtime (without NLMS), and is timed on a Pi 5 (row 11). The stream contract (`vaani/backend.py`: versioned state, ONNX and Torch backends, telemetry) is tested in `tests/test_stream_contract.py` and `tests/test_backend.py`. No Orin run |
| A prototype demonstrating live cancellation with microphones / headset | **not met** | The known gap (§3.5) |
| SNR > 15 dB, STOI > 0.85, PESQ > 2.5 | partial — **not met by any trained model on its full evaluation** | **r8 (val monitor, EMA, 200 dynamic val items, last scored snapshot; not the registered comparison):** C0 Mini 10.43 / 0.849 / 1.878, Arm A Mini 9.81 / 0.835 / 1.734, Arm B Mini 10.05 / 0.839 / 1.785. Scoring of the three full runs is unfinished (307, 135 and 164 of 320 snapshots); the final selection may move these. **r7, eval_r2 nominal (617 clips):** 14.864 dB [14.565, 15.183] · 0.917 [0.911, 0.922] · 2.462 [2.412, 2.511]: STOI clears on the interval, SNR and PESQ miss at the point estimate. Full test split (2,280): 12.78 · 0.870 · 2.138. eval_gen registered stationary grid (102): 14.295 · 0.932 · 2.490; changing grid (99): 18.37 · 0.970 · 3.153. All three targets met together on 35.5 % [31.8, 39.1] of nominal clips (`results_r2/r7/breakdown.md`) |
| Low latency suitable for real-time communication | partial | **VAANI-LD:** 10 ms algorithmic (Arm B synthesis support), plus 0.407 ms for the 48 kHz resampler pair (`results_r2/r8_ld/gate0/resampler.json`) and 0.50–3.63 ms of measured Pi compute per hop by tier. **r7 / C0:** 32 ms algorithmic. End-to-end mic-to-speaker latency has not been measured for either; [`docs/acoustic_latency.md`](acoustic_latency.md) is the procedure |

**Which eval render (r7).** The eval_r2 scores are from the current render of eval_r2, re-made from the crest-audit relabelled manifests (EVALSET_HASH `17a9414959bb` on Windows, `aa96a28a9955` on Linux: the same audio, the hash digests float text). [`results_r2/matrix_prerelabel.md`](../results_r2/matrix_prerelabel.md) (which alone has the external baselines and WER) and [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md) were measured on the earlier frozen render, which differs at 606 of the 617 nominal items. Comparisons within one render are valid; comparisons across the two are not.

**Held-out status.** The eval_r2 test split informed r7's launch (r6 scored 14.83 dB on it), so it is not untouched; r8 never reads it. eval_gen's noise is hash-disjoint from the r6/r7 recipes, but r7 was scored on it without the checkpoint registration `results_r2/generalisation/PROTOCOL.md` asks for.

---

## 3. Notes on the rows that need one

### 3.1 Artillery is synthesised from physics for training; it is not yet evaluated on r8

The public artillery recordings obtainable for this project are YouTube-sourced. `scripts/crest_audit.py`
measured MAD's `shelling` class at **12.3 dB event crest** (±100 ms around the peak) against
**13 dB for ordinary speech** — loudness-normalised and lossy-coded, so the transient the class is
named for is simply not in the audio. `MAD_CLASS_MAP` therefore labels it `changing`, not
`impulsive`.

The fix was to generate the transient from its physics instead. `vaani/data/blast.py` implements the
Friedlander blast wave, p(t) = P₀(1 − t/T)·e^(−t/T): effectively instantaneous rise to peak
overpressure, exponential decay, zero crossing at the positive-phase duration T, then a rarefaction
phase — with T around 0.15–0.6 ms for small arms at close range and several milliseconds for
artillery. It is synthesised at 192 kHz and decimated (a near-instantaneous rise built directly at
16 kHz aliases), a ground reflection arrives 1–9 ms later, and distance enters as a first-order
low-pass. Measured event crest: **~26.5 dB**, against 15.2 dB for the previous synthetic burst.

Every impulsive source is gated on measured crest before it may enter training. The r7 and r8
recipes draw impulses from `impulse_kinds: [blast, blast, burst, click_train]` at 15–45 dB peaks
with the room path and soft-clip on.

The defence set (local render `data/eval_defence/test`, hash `d033568bdf98`) adds
`blast_small_arms` and `blast_artillery` categories made with the physics-v2 generator (same-sign
ground reflection, ISO 9613-1 air absorption, SPL-referenced levels). r7 meets all three targets on
the interval only at 15 dB input: small arms 18.03 / 0.955 / 2.87, artillery 16.49 / 0.944 / 2.78;
at −10 to 0 dB input it reaches 4.97–10.09 dB SNR_out (`results_r2/defence/table.md`). r7 trained
on v1 blasts, so these rows measure robustness to more realistic blasts. No r8 model is scored on
the defence set yet.

### 3.2 Full-band, sub-band, complex domain: the mapping

Clauses 6 and 7 describe a specific feature structure. In the statement's own vocabulary:

| Problem statement | VaaniFE (VAANI-LD) | r7 (GTCRN-derived) |
|---|---|---|
| sub-band features (local) | frequency convolutions with time kernel 1 over sub-band windows (`p32` tiling at Arm B) | SFE, the sub-band feature extraction module |
| full-band features (global) | multi-head self-attention across every frequency token of the frame | ERB band-split across the whole spectrum |
| global dependencies (time) | a one-step time GRU per frequency token, K blocks | DPGRNN inter-frame path |
| local dependencies (frequency) | the encoder's sub-band convolutions; attention also spans local neighbours | DPGRNN intra-frame path |
| complex domain, phase preserved | unbounded complex mask + 3-tap complex deep filter over 0–4.5 kHz, complex-part loss | complex ratio mask + 3-frame deep filter, complex-part loss |

The table describes what is built, not what a trained network does with it: row 7 has the only
phase measurement, and it is r7's.

### 3.3 The perceptual loss terms

r8's FE loss (`vaani/losses.py::FELoss`, used through `ResynthesisFELoss` on the low-delay
contracts) is a FastEnhancer-style mix on power-law compressed spectra (p = 0.3): magnitude,
complex parts, waveform L1, a clamped absolute-SNR term, and a **differentiable PESQ term**
(`torch_pesq`, weight 0.001). Every term is computed after re-synthesis through the contract's real
synthesis window, so the loss sees the samples the listener hears. κ = 3 scales an over-suppression
error (removing speech) by κ before squaring, so it costs κ² = 9 times a same-size residual-noise error. `pesq_required: true` makes a run refuse to
start without `torch_pesq`, so the term cannot silently drop to zero.

r7's `HybridLoss` had only the compressed-magnitude weighting (p = 0.5). Power-law compression
approximates the compressive loudness response of hearing; it is a psychoacoustic magnitude
weighting, not a perceptual metric. The PESQ term is the metric-oriented objective r7 lacked. Its
effect has not been isolated by an ablation.

### 3.4 Quantization and pruning: implemented, measured, and rejected (r7 era)

Both are named by the statement and neither needs hardware, so both were implemented and measured.
Both came back negative. Full tables in [`results_r2/optim/optimization.md`](../results_r2/optim/optimization.md).
All of §3.4 was measured on the **tier46** cascade (checkpoint sha256 `932b086a…`, the same
architecture and parameter count as r7) and on the earlier frozen render of eval_r2; every
comparison is paired within that render. Neither has been re-run on VaaniFE, whose loop-free
Gemm-cell graph removes the reason dynamic INT8 failed here (no integer GRU kernel); whether INT8
pays on it is untested.

**INT8 dynamic quantization makes this model larger, slower and worse.** On the trained cascade the
graph goes 474,599 -> 567,969 bytes (**+19.7 %**) and latency 0.906 -> 1.319 ms per frame (**x1.45**)
(`deploy/tier46/int8_report.json`). Per-channel weight scales do not rescue it: 569,805 bytes
(+20.1 %), 1.310 ms (x1.45) (`deploy/tier46/int8_perchannel_report.json`). Paired per clip against
the same checkpoint, SNR_out **-1.18 dB** [-1.27, -1.10] and PESQ **-0.166** [-0.176, -0.156].

The reason is structural: the FP32 graph's weights account for only **210 KB of its 474 KB** -- the
rest is node protobuf. Quantization cut weight bytes to 157 KB but added 150 nodes. ORT's dynamic
path has no integer kernel for `GRU`, so the 29 inserted `DynamicQuantizeLinear` nodes are pure
added work on top of an unchanged float recurrence. The FP32 ONNX row reproduces the PyTorch
checkpoint to +-0.000 on every metric, so the INT8 deltas are attributable to quantization.

**Pruning has almost nothing to remove, and removing it costs more than it saves.** Of 52,747
parameters, only **21,952** are learned weights in prunable layer types; the ERB matrices are a
fixed transform and are excluded. Measured falloff, paired per clip against the unpruned checkpoint:

| level | weights zeroed | d SNR_out (dB) | d PESQ | still meets targets |
|---|---:|---:|---:|---|
| p10 | 2,195 | -0.13 | -0.019 | yes (STOI, and SNR/PESQ at the mean) |
| p20 | 4,390 | -0.74 | -0.070 | no (PESQ 2.478) |
| p30 | 6,586 | -2.17 | -0.401 | no (PESQ 2.147) |
| p40 | 8,781 | -8.38 | -0.732 | no (STOI 0.832 also fails) |
| p50 | 10,976 | -14.71 | -1.359 | no (SNR_out 0.44 dB; the model is destroyed) |

Every delta's 95 % interval excludes zero. Zeroed weights in a dense graph still occupy their bytes
and still get multiplied, so p10 costs 0.13 dB for a 0 % saving in size or latency.

### 3.5 The hardware gap and the latency evidence

Clauses 13, 15, 16 and the prototype deliverable need a board and two microphones in a headset.

- **Target and development board.** The deployment target is the NVIDIA Jetson AGX Orin 64GB
  (confirmed 24 Sep 2026; no hardware access yet). The Raspberry Pi 5 is the development board.
- **What the Pi has measured (2026-10-05).** `r8_runs_final/pi_bundle/tiers_armb/time_tiers.sh`
  ran the native runtime's `vld_step_bench` on the four untrained Arm B tier graphs (`inputs: pr`):
  7,500 hops each, one thread pinned to core 3, flush-to-zero, random input on the 48 kHz path
  with the deploy resampler, not throttled. Whole-hop means 0.50 / 1.05 / 2.32 / 3.04 ms per 8 ms
  hop; Large+ under SCHED_FIFO with `mlockall` peaks at 3.63 ms. Timing does not depend on the
  weights. **TBD: commit the board's `pi_results/tiers_armb/*.json`**; until then the numbers rest
  on the terminal output quoted in the README.
- **What the Pi has not measured.** A trained graph, the `pr_nhat` graph, the NLMS stage (absent
  from the native runtime; the Python engine runs it) and any acoustic path.
- **Gate 0a** (`results_r2/r8_ld/gate0/README.md`) budgets 13.0 ms for the processing path. With a
  0.5 ms converter estimate, L = 10 ms comes to 13.007 ms, so the gate stays `pending_board` until a
  measured converter path replaces the estimate (Gate 0b).
- **End to end.** Mic-to-speaker delay has not been measured. [`docs/acoustic_latency.md`](acoustic_latency.md)
  is the procedure: an independent two-mic recorder on one clock, scored by `scripts/acoustic_latency.py`.
- **Real microphones.** The dual-microphone requirement in clause 15 is the design's central
  assumption. What is missing is validation against real microphones; the physical-test schema is
  [`docs/physical_test_schema.md`](physical_test_schema.md).

### 3.6 Corpus licensing

[`docs/licences.md`](licences.md) is generated from the manifests and the training recipes by
`scripts/licence_table.py`, so it states what training actually reads. [`NOTICE`](../NOTICE) states
that the code is MIT while trained weights are research-only and bound by the training-data
licences (decision, Rachit, 2026-09-25: research-only weights are acceptable), and
[`deploy/dnsmos/NOTICE.md`](../deploy/dnsmos/NOTICE.md) attributes the DNSMOS model. TBD: which
DNS-Challenge licence file covers the DNSMOS `.onnx` files, and at which commit the copy was taken.

Several corpora in the r7 and r8 recipes are not commercially usable: EARS and ESC-50 are CC BY-NC,
and MAD is YouTube-sourced (an absence of licence). Others have unresolved terms: DNS noise is
licensed per clip; DroneAudioDataset is citation-on-use and NOISEX-92's SPIB redistribution terms
are unstated (both train r7, and r8 only evaluates on them); the r8 additions carry their own terms
(listed per corpus in `docs/licences.md`). A fielded version would retrain on licensed or government-collected data;
because a recipe is a list of manifest paths, that substitution is mechanical.

### 3.7 What "ANC" means here

The title's "adaptive noise cancellation (ANC)" is read as **transmit-path speech enhancement**:
VAANI cleans the voice that the wearer sends. The statement's own measures support that reading:
SNR, STOI and PESQ are measures of the enhanced speech, and clause 11 asks for mask estimation and
speech reconstruction. The adaptive part is the NLMS front end (row 1).

Ear-side ANC, which cancels noise at the wearer's ear, is a different subsystem and none of it is
built: a reference mic outside the earcup and an error mic inside it; secondary-path identification;
an FxLMS-family controller with far lower latency than an 8 ms hop; closed-loop stability analysis;
and acoustic measurement of the attenuation at the ear.

### 3.8 Compute scaling and the Orin target

The scaling family is **VaaniFE** (dual-mic, adapted from FastEnhancer, arXiv 2509.21867). All tiers
share the contract, the front end and the step-graph rules; only width and depth change.

| Tier | C1/C2/F/K/L | Parameters (Arm B graph) | Pi 5 whole hop, mean / p99 (ms per 8 ms hop) | Trained |
|---|---|---:|---|---|
| Mini | 32/24/16/2/1 | 29,597 native tiling · 40,302 p32 (trained) · 44,782 entries with `n_hat` | 0.50 / 0.53 | `pr`: yes (Arm A, Arm B, C0; 200,000 steps each) · `pr_nhat`: no |
| Mid | 48/40/32/3/2 | 107,934 · 123,910 p32 | 1.05 / 1.11 | no — final run configured (`r8_final_mid.yaml`) |
| Large | 80/64/48/4/2 | 322,519 | 2.32 / 2.48 | no |
| Large+ | 96/72/48/4/3 | 500,367 · 533,463 p32 | 3.04 / 3.29 | no — final run configured (`r8_final_large_plus.yaml`) |
| r7 (comparison) | GTCRN cascade | 52,747 | — (16 ms hop; no board timing) | yes |

The final Mid and Large+ runs were set up for a 2× RTX 5090 rental on 2026-10-05
(`scripts/final_launch.sh`), but the box was stopped before its setup finished, so **neither was
trained**. Their laptop step budget is TBD (Rachit). Only the Mini is under the spec 6.2 Pi budget
(60,000 entries, 90.706 MMAC/s, asserted in `tests/test_vaani_fe.py`); the larger tiers are aimed at
the Orin. G2 (`scripts/graph_gate.py`): folded nodes < 250, no Loop/Scan/If/GRU/LSTM/RNN, no
ScatterND, no symbolic dims or Shape/Range, layout ops < 30 %, ORT-vs-torch parity ≤ 1e-5, streaming
step equal to the offline forward; every tier passes, r7 fails. No Orin latency, power or quality is
measured or claimed.

r8 gates and where they live:

- **G1** ILD shortcut (`results_r2/r8/data_gates/README.md`): mixer v2 AUC 0.727 (parametric path)
  and 0.697 (room path), limit 0.75: **pass**.
- **G2** graph gate: **pass** on every tier (above).
- **Gate 0a** latency eligibility: **pending_board** (§3.5).
- **G3** quality on val: the registered low-delay comparison (`scripts/compare_r8_ld.py`, margins
  approved by Rachit 2026-09-27) is **not run**; it needs the unfinished pilots and the finished
  scoring of the full runs.
- **G4** field acceptance (`scripts/field_accept.py`): r7 baseline **fails**; no r8 run yet.
- **G5** Orin: deferred, needs the board.
- **G6** the pre-registered test set (`data/eval_r8_test`, hash `ed024af085a2`, 2,308 items): scored
  once after selection, **not scored**.

Procedure: [`configs/retraining/R8_RUNBOOK.md`](../configs/retraining/R8_RUNBOOK.md); the laptop
queue for the remaining runs is `configs/retraining/laptop_queue.txt`.
