# r8 low-delay pilots (plan Task 8)

Every file here, and `../r8_ld_fe_*.yaml`, is written by `scripts/gen_r8_configs.py --low-delay` from
`../r8_fe_mini.yaml` (C0) and the Gate 0a record (`results_r2/r8_ld/gate0/eligibility.json`). Do not edit them by hand:
`--low-delay --check` fails on any difference, missing file or extra file (only `.md` files are exempt), and the
low-delay preflight runs that check. Nothing here has been run.

```bash
python scripts/gen_r8_configs.py --low-delay                                 # regenerate after Gate 0a or a C0 edit
python scripts/gen_r8_configs.py --low-delay --check                         # exit 1 on any drift
python scripts/gen_r8_configs.py --low-delay --promote ld_s2_overparam[,..]  # wave 2, after the Stage-2 decision
```

## Derivation

`../r8_ld_fe_mini.yaml` (Arm A) is `r8_fe_mini.yaml` plus one documented overlay (`LD_OVERLAY` in the generator):

- `model_cfg.audio_contract`: Arm A at the support Gate 0a selected. The committed record is `pending_board`, so the
  generator uses its provisional selection, `vaanife_ld_asym512_h96_s128_v1` (L = 8 ms). Every Arm A file names it.
  Regenerate once the board record is complete. The queue and the preflight both refuse files generated for a
  different selection.
- `freq_windows: p18`, `valid_bias: true`, `df_bins: 96`, `df_lags: [0, 3, 5]`, `df_taps: 3`, `gru_init: tc_matched`;
- `dsp.ref_policy.ramp_samples: 3072`, which replaces `ramp_frames`;
- `loss_cfg.loss_domain: resynthesis`, `w_consistency: 0.0`.

Every arm inherits C0's framing-independent corrections unchanged. These are `model_cfg.fp32_islands: true`,
`dsp.limiter_kernel: numba` (the compiled limiter) and the `perf` block. `perf.numerics` is a resume key and is the
same in every file. The preflight fails if it differs. `perf.ops` is bit-exact and set per run by the launcher
(`VAANI_PERF_OPS`), never in a file. `dsp.limiter: true` is inherited.

| Files | Arm / stage | Changes from Arm A | Priority, wave |
|---|---|---|---|
| `ld_a_s{0,1}` | Arm A, Stage 1 | seed | P2 (pilot class), wave 1 |
| `ld_b_s{0,1}` | Arm B, Stage 1: L = 10 ms, `p32`, 144 DF bins, lags [0, 2, 4] | contract, tiling, DF band | P2, wave 1, **only when Gate 0a pilots Arm B** |
| `ld_r_s{0,1}` | Arm R: native tiling, no validity term (P18's deep filter kept) | drops `freq_windows`, `valid_bias` | P4, wave 1 |
| `ld_s2_overparam`, `ld_s2_gru_default`, `ld_s2_mrstft05`, `ld_s2_warmup480`, `ld_s2_native` | Stage 2 items 1-5, seed 0 | one field each (item 6 is conditional and not generated) | P2, wave 1 |
| `ld_a_s{2,3,4}` | Arm A confirmation (the no-addition control) | seed | P3, wave 1 |
| `ab1_fe_mini_s{2,3,4}` | C0 seeds 2-4, from `r8_fe_mini.yaml` like `../r8_ablations/ab1_fe_mini_s{0,1}` | none (C0) | P3, wave 1 |
| `ld_p4_tail{00,10}`, `ld_p4_refdrop{00,30}`, `ld_p4_bounded`, `ld_p4_kappa{1,2,4}` | P4: ab3b, ab3, ab4, ab6 on Arm A | one field each | P4, wave 1 |
| `ld_conf_s{0..4}` | confirmation of the promoted recipe | the promoted Stage-2 fields | P3, wave 2 (`--promote`) |
| `../r8_ld_fe_mini.yaml` | Arm A full run (D8, speculative) | none | full, wave 1 |
| `../r8_ld_fe_mini_overparam.yaml` | Arm A + over-parameterization full run (D8, speculative) | `overparam` | full, wave 1 |
| `../r8_ld_fe_mini_armb.yaml` | Arm B full run (D8, speculative) | as Arm B | full, wave 1, only when Arm B is piloted |
| `../r8_ld_fe_mini_conf.yaml` | the promoted recipe's full run | the promoted fields | full, wave 2, unless the early overparam run is that recipe |
| `../r8_ld_fe_{mid,large,large_plus}.yaml` | tier projections (native tiling) | tier | **not queued** in r8 |

C0 seeds 0/1 are the existing `../r8_ablations/ab1_fe_mini_s{0,1}.yaml`, and the full C0 is `../r8_fe_mini.yaml`. They
share their run directories with the legacy queue, so neither queue trains them twice. There are no distillation
fields or jobs (D9).

## `arms.json`

This file holds one record per generated file and per external C0 run:

- seed, epochs, batch and `optimizer_steps`;
- scored and prefix exposure in seconds;
- precision: AMP, FP32 islands and `perf.numerics`;
- tier, tiling, contract and its hash, support and hop;
- the deployment audio path. This stays `pending` until the Gate 0a board record is complete, and then lists the eligible,
  period-verified and processing-measured rows;
- `deployable`: Arms A and B and C0 at the Mini tier. Spec 6.2 applies to these;
- the queue fields: kind, stage, priority, wave, speculative and `stop_if`.

It also records the Gate 0a status the files were generated from, and the promoted Stage-2 set.

## Queue

The launcher is `bash scripts/run_r8.sh ld-plan | ld-start pilots|full|all | ld-status | ld-decide | ld-go-full`, with
`scripts/r8_ld_queue.py` behind it. Each name resolves to exactly one file, and a missing or ambiguous file fails the
queue. Nothing starts until the Gate 0a record is complete. Full runs also need `ld-go-full` and a passing low-delay
preflight younger than 24 h (`runs/r8_queue/preflight_ld.json`). `ld-decide stage1 arm_a|arm_b` and
`ld-decide stage2 <ld_s2_...>|none` record a decision, write `STOPPED` for the runs whose `stop_if` it matches, and
stop them if they are running. After a Stage-2 promotion, run the `--promote` line above so that wave 2 can resolve.
