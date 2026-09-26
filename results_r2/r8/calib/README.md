# Mixer v2 level chain: is the 123 dB clipping a calibration bug or physics?

Short answer: **physics, given the ICS-43434 mic model.** The calibration chain does what calib.py and plan 11.5 M1 say.
Four things push the input past the converter's full scale (123.01 dB SPL peak), each measured below:
- beds of 101-113 dBA with 11-18 dB crest factors (helicopter, APC);
- the added near-field source;
- raised-to-shouted speech at a 2-3 cm boom;
- gunshots and blasts at the mic.

The only defect was the meaning of the `overloaded` flag, fixed below. It means "past the 90.7 dB soft knee", so it is
true on every item in every scene. The questionable parts are model choices (knee curve, HPF order, near-source level
rule, mic full scale). They are implemented as v2 options and left for Rachit's decision (report: Decisions).

## Decisions (Rachit, 2026-09-26)

- **Mic full scale: 120 dB SPL for r8** (`mix.v2.mic_fs_spl_db` 120.0, the ICS-43434 default in vaani/data/mixer.py).
  It matches the rig: the demo hardware is two ICS-43434 breakouts (H5 below). No r8 config overrides it.
- **For the pitch:** a fielded boom wants a mic rated **>= 130 dB AOP**. The H5 column below shows why: with a 130 dB
  full scale the share of items past the rails falls from 0.958 / 0.975 / 0.883 to 0.383 / 0.508 / 0.492 in helicopter
  / APC / firefight (trace_r8test, the pre-4dcfa90 v2 defaults). It still rails a third to a half of those items, so
  the 130 dB figure is a floor, not a fix (inferred).
- **Unchanged:** knee curve (`fe_curve: knee105`), HPF order (`fe_hpf_order: pre`) and the near-source rule
  (`near_mode: add`) stay as they are. The alternatives below remain options, off in every r8 config.

## Chain as implemented (vaani/data/mixer.py mix_v2, vaani/data/calib.py)

| stage | rule | level definition |
|---|---|---|
| speech | talker level from SCENES `talker` (dB SPL at the boom) | speech-active rms (P.56-style active level), unweighted |
| beds, points, near | SCENES `beds`/`points` spl; near = bed + U(0, 8) dB ("add") | **A-weighted Leq over the whole crop**, on the primary mic |
| wind | M4 level at the drawn speed | unweighted rms in the gust "high" state |
| impulse / event | SCENES `event.peak` | peak dB SPL on the primary, before the HPF |
| float | float rms dB = SPL - 123.01 (-26 dBFS at 94 dB SPL); float 1.0 = 123.01 dB SPL peak | round trip error < 1e-6 dB (data_gates `spl_round_trip`) |
| front end | mic gain +-1 dB, then 2nd-order 60 Hz HPF, **then** the saturator (knee105), rails at float 1.0, then self-noise -93 dB | the HPF sits BEFORE the nonlinearity |

## Measurement

Command (repo root, 2026-09-25, job calib-trace3, 2.5 min, train pools, `render_eval_sets.render_scene_item` path with
impulses, 6 s crops, mix_v2 traced in memory, nothing written under data/):

    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python scripts/data_gates.py --calib 120 \
      --calib-scenes helicopter apc firefight artillery patrol command_post windy_ridge drone --seed 4242 \
      --out results_r2/r8/calib/trace_r8test
    .venv/Scripts/python.exe results_r2/r8/calib/scene_table.py results_r2/r8/calib/trace_r8test

These are the v2 defaults **before** commit 4dcfa90 (p_near 0.7, near_pos_share 0.8, near ILD 6-12 dB), the ones the
frozen `data/eval_r8_test` was rendered with. Files: `trace_r8test/level_trace_items.csv` (one row per item),
`level_trace.json` (p10/p50/p90 per column), `scene_table.csv` (the table below). `trace_baseline/` is the same seed
run before the flag code existed (no counterfactual columns). All levels are dB SPL at the primary mic. The table gives
medians over 120 items per scene. "Alone past rails" is the share of items where that part by itself peaks at 123.01 dB
or more.

| scene | bed dBA | bed dBZ | Z-A | bed crest | bed peak | speech set / peak | total peak | past knee | past AOP (20 ms rms > 120) | past rails | railed samples, by largest part | alone past rails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| helicopter | 106.6 | 110.6 | 4.0 | 15.0 | 125.8 | 105.1 / 123.3 | 131.9 | 1.00 | 0.55 | 0.958 | near .78, bed .21 | speech .53, bed .67, near .60, wind .12 |
| apc | 106.7 | 111.2 | 4.5 | 13.3 | 126.4 | 104.9 / 123.4 | 133.2 | 1.00 | 0.64 | 0.975 | near .63, bed .36 | speech .52, bed .65, near .56, impulse .07 |
| firefight | 80.2 | 83.7 | 3.5 | 16.6 | 101.9 | 104.8 / 123.0 | 133.0 | 1.00 | 0.57 | 0.883 | impulse .98 | speech .49, impulse .76 |
| artillery | 69.6 | 74.3 | 4.7 | 17.6 | 93.8 | 102.0 / 120.7 | 123.3 | 1.00 | 0.24 | 0.508 | impulse .99 | speech .31, impulse .27 |
| patrol | 53.1 | 58.3 | 5.2 | 18.1 | 77.8 | 86.0 / 102.4 | 106.1 | 1.00 | 0.01 | 0.075 | impulse 1.00 | impulse .06 |
| command_post | 64.7 | 68.8 | 4.1 | 16.4 | 85.6 | 88.3 / 105.7 | 107.2 | 1.00 | 0 | 0 | - | - |
| windy_ridge | 49.7 | 54.8 | 5.1 | 18.7 | 74.8 | 95.5 / 112.5 | 115.4 | 1.00 | 0 | 0.092 | wind .99 | wind .03 |
| drone | 55.0 | 59.2 | 4.2 | 17.5 | 78.3 | 90.0 / 107.0 | 108.3 | 1.00 | 0 | 0 | - | - |

The meta flags agree: `clipped` equals past rails in every scene, and `overloaded` is 1.00 everywhere. The share of
samples past the knee is 0.93 for helicopter and APC and 0.06-0.56 elsewhere. The share of samples past the rails is
0.095 / 0.144 / 0.025 / 0.010 for helicopter / APC / firefight / artillery. The frozen test render agrees
(reports/testset.md): clipped on 48/48 helicopter, 44/48 firefight, 43/48 APC and 22/48 artillery items.

Counterfactuals computed on the same parts of the same items (share of items past the rails):

| scene | as built | beds on dBZ instead of dBA (H1) | HPF after the saturator (H2) | near "split" | mic full scale 130 dB (H5) |
|---|---|---|---|---|---|
| helicopter | 0.958 | 0.883 | 0.975 | 0.925 | 0.383 |
| apc | 0.975 | 0.900 | 0.975 | 0.958 | 0.508 |
| firefight | 0.883 | 0.883 | 0.883 | 0.883 | 0.492 |
| artillery | 0.508 | 0.500 | 0.517 | 0.500 | 0.208 |
| patrol | 0.075 | 0.075 | 0.058 | 0.075 | 0 |
| command_post | 0 | 0 | 0.083 | 0 | 0 |
| windy_ridge | 0.092 | 0.092 | 0.117 | 0.092 | 0 |
| drone | 0 | 0 | 0 | 0 | 0 |

The saturator's distortion re the linear signal (primary, median dB) is -14.6 / -13.1 / -12.5 for helicopter / APC /
firefight with knee105, and -11.7 / -10.3 / -10.6 with tanh120. It is -50 dB or lower in the quiet scenes under either
curve.

## Hypotheses and verdicts

| # | hypothesis | measurement | verdict |
|---|---|---|---|
| H1 | Scaling an LF-heavy bed to its dBA target inflates its unweighted peak | median(bed Z) - median(bed A) is 3.5-5.2 dB; the median of the per-item Z-A is 2.6-4.5 dB (calib-verify). If the beds were scaled on unweighted rms instead, helicopter rails fall from 0.958 to 0.883 and APC from 0.975 to 0.900; no other scene moves. tests/test_calib.py checks A < Z on an LF bed. | **PHYSICS / as intended.** The tables are dBA Leq, and the cited anchors are dBA: CH-47D cockpit 107 dBA, ramp 115 dBA, UH-60 cockpit 102-103 dBA ([USAARL 96-02](http://www.chinook-helicopter.com/Technical_Reports/Communication_and_Noise_Hazard_Survey_of_CH-47D_Crewmembers.pdf)). A dBA table implies a higher unweighted level, and that is real pressure at the mic. The APC range (100-115 dBA) is inferred (scenes.py, battlefield_physics s7). TBD: a primary source for tracked-vehicle interior dBA. |
| H2 | The 60 Hz HPF applied after the nonlinearity inflates LF peaks | The code applies the HPF **before** the saturator, so the premise is reversed. The datasheet's only filter is a digital HPF (-3 dB at 24 Hz) after the sigma-delta ADC (DS-000069 "Digital filter characteristics"). "60 Hz to 20 kHz" appears only in the features list. Moving the HPF after the saturator (`fe_hpf_order: post`) raises rails slightly (helicopter 0.958 -> 0.975, command post 0 -> 0.083, windy 0.092 -> 0.117). | **MODEL CHOICE.** Neither order is the datasheet's. It gives a 24 Hz digital HPF after the ADC and a 60 Hz lower response edge, which is inferred to be the acoustic vent roll-off acting before the converter. `pre` treats the whole 60 Hz roll-off as acoustic; `post` treats it as digital. `pre` (as built) is the optimistic one: it removes LF before it can clip. |
| H3 | Impulse peak SPL at distance, or gunshot scaling | Impulses cause 0.98-1.00 of railed samples in firefight, artillery and patrol. The drawn event peaks (p10/p50/p90) are firefight 119/131/149, artillery 129/144/155 and patrol 103/113/122 dB peak. They match the scene tables, which are a mixture: own weapon 158-162, squad 140-150, others 115-135 dB. | **PHYSICS.** Any gunshot above 123 dB peak at the head rails a 120 dB AOP part. Plan M5 puts a gunshot at 150-160 dB peak at 1 m. Inferred: the table's 115-135 band corresponds to tens to a few hundred metres under 1/r. The mixture shares are inferred (scenes.py comment). |
| H4 | Soft-knee flag semantics ('overloaded') | `overloaded` = a sample above soft_knee() = 0.0243 float = 90.7 dB SPL peak (the 0.2 % THD at 105 dB fit). It is true on 100 % of items in all 8 scenes, including drone and command post. The old docstring called this knee "the AOP". | **BUG (semantics and docs).** Fixed: `overloaded` is kept unchanged (the frozen metas use it). New flags are `past_knee` (the same test), `past_aop` (a 20 ms frame above 120 dB SPL rms on either mic; the AOP is a sine level) and `past_rails` (float >= 1.0, i.e. >= 123.01 dB peak, = `clipped`), plus `peak_db_spl`. Docstrings are corrected (calib.py module doc, `front_end_nonlinear`, `level_flags`). tests/test_calib.py covers the thresholds. |
| H5 | ICS-43434 AOP 120 dB SPL is exceeded by real helicopter/APC noise at a boom mic | AOP = 10 % THD at 120 dB SPL = digital full scale for a sine (-26 dBFS at 94 dB); the rails are the ADC's full scale (DS-000069 spec table). A 107 dBA cabin with the measured 4 dB Z-A and a 15 dB crest peaks near 126 dB. Shouted speech at 2-3 cm alone peaks at 123 dB (median) in helicopter, APC and firefight. A mic with a 10 dB higher full scale still rails 38 % / 51 % / 49 % of those items. | **PHYSICS**, for the mic modelled. The demo rig is two ICS-43434 breakouts ("Claude outputs/HARDWARE.md", parts list). The model therefore matches the hardware that will be tested, and the clipping is what that rig will record. Whether a fielded headset should use a higher-AOP boom mic is a product question (Decisions). |
| - | Knee curve vs plan M7 ("soft saturation from 120 dB SPL") | THD of a sine through curve + rails (`calib.thd_of`): knee105 gives 0.20 % at 105 dB and 6.7 % at 120 dB. tanh120 gives 0.44 % at 105 dB and 10.0 % at 120 dB. Datasheet: 0.2 % typ / 1 % max at 105 dB; 10 % at 120 dB (AOP). | **MODEL CHOICE.** knee105 matches the typical THD but under-distorts at the AOP. tanh120 meets both datasheet points (within the 1 % max at 105 dB) and is the battlefield_physics s1 recipe. Option `fe_curve: tanh120`. |
| - | Near source "add" (bed + near exceeds the scene's dBA range by up to 8.6 dB) | Near is the largest part in 0.78 / 0.63 of railed helicopter / APC samples. With "split" (bed + near hold the drawn level), rails are 0.925 / 0.958. | **MODEL CHOICE.** It was a G1 tuning addition, not physics. `scenes.sample_scene(near_mode="split")` is available. G1 was tuned and confirmed with "add"; switching needs a G1 re-run. |

## Options added (all v2-only; defaults reproduce the pre-change audio, v1 bit-exact)

- `mix.v2.fe_curve`: `knee105` (default) or `tanh120` (x_s = 0.771 float, fitted to 10 % THD at 120 dB).
- `mix.v2.fe_hpf_order`: `pre` (default) or `post` (saturate, then HPF, then self-noise).
- `mix.v2.fe_hpf_hz`: 60.0 (default); 24.0 is the datasheet corner.
- `mix.v2.mic_fs_spl_db`: 120.0 (default, ICS-43434). 130 models a 130 dB AOP part: acoustic SPLs stay the same and
  every float level drops by 10 dB. Inferred: the Infineon IM69D130 is such a part (AOP 130 dB SPL). TBD: datasheet URL.
- `scenes.sample_scene(near_mode=)`: `add` (default) or `split`.

Proof that the defaults are unchanged: `results_r2/r8/calib/mixer_hash.py` gives v1 `a1f77d8e...` and v2 `ae6d0d50...`
at HEAD and on the tree with the options. The new meta keys are excluded from the hash. That run was before the G1
c5 defaults. After commit 4dcfa90 the v1 hash is still `a1f77d8ee2f9b4ccaa1286262dd4545d4012a531`. The v2 hash is now
`273516fb...`, as expected, because the G1 defaults moved; see ../data_gates/README.md. `tests/test_golden_vectors.py`
and `tests/test_mixer.py` pass.

## M11 F0 (Lombard): decided off

The only pitch shifter in this environment is torchaudio's phase vocoder plus resample. It moves the formants with F0
(real Lombard F1 moves by only about +54 Hz) and smears transients. Synthetic F0 therefore stays off: `LOMBARD_F0_ST`
is recorded, not applied. Real Lombard speech (Lombard GRID, EARS loud and shout) carries the F0 change. The alpha-ratio
tilt (+4.5 dB) stays on. See the calib.py comment at `LOMBARD_F0_ST`.
