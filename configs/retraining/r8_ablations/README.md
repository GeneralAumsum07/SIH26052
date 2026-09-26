# r8 ablation pilots (plan 11.6)

Pilots, not full runs: every config is **48 epochs = 15 % of the 320-epoch full schedule** (48 x 20,000 = 960,000
items = 30,000 steps at B 32), mixer v2, held-out exclusion, EMA and composite val selection, exactly as the full
configs `../r8_fe_mini.yaml` / `../r8_refvalid_v2.yaml` except for the one field each arm changes. Nothing here has been
run. `scripts/gen_r8_configs.py` writes every pilot from the two full configs (`--check` verifies the tree), so the
untouched fields cannot drift.

Arms whose value equals the r8_fe_mini default are **not duplicated**: the baseline for ablations 2, 3, 3b, 4 and 6 is
`ab1_fe_mini_s0` (and `_s1` where two seeds are listed): inputs `pr`, reference absent 15 %, tail share 40 % (G1 c5),
unbounded mask with no DF taps, kappa 3.

| # | Question | Files | Varies | Baseline arm |
|---|---|---|---|---|
| 1 | Family gate: VaaniFE-Mini vs refvalid C16 on the same data and selection | `ab1_fe_mini_s{0,1}`, `ab1_refvalid_s{0,1}` | model family | each other |
| 3b | Low-ILD tail share of the M8 scenes | `ab3b_tail00`, `ab3b_tail10` | `data.mix.v2.tail_share` 0 / 0.10 | `ab1_fe_mini_s0` (0.40) |
| 2 | Inputs: raw primary only, + PLD, + classical n_hat | `ab2_{p,pr_pld,pr_nhat}_s{0,1}` | `model_cfg.inputs` | `ab1_fe_mini_s{0,1}` (`pr`) |
| 3 | Reference-absent dropout rate (M9 validity 0) | `ab3_refdrop{00,30}_s{0,1}` | `data.ref_corrupt.p_absent` 0 / 0.30 | `ab1_fe_mini_s{0,1}` (0.15) |
| 4 | Mask range and low-band deep-filter taps | `ab4_bounded_df0`, `ab4_unbounded_df3`, `ab4_bounded_df3` | `model_cfg.mask`, `model_cfg.df_taps` | `ab1_fe_mini_s0` (unbounded, 0) |
| 5 | Refiner stage on/off | **dropped** | - | VaaniFE has no refiner stage (vaani/models/vaani_fe.py); the refiner belongs to the r7 cascade, which is not an r8 candidate |
| 6 | Asymmetric over-suppression weight | `ab6_kappa{1,2,4}` | `loss_cfg.kappa` (1 = term off) | `ab1_fe_mini_s0` (3) |
| 7 | RIR bank alone: r7's bank under the r8 recipe | `ab7_bank_r3` (val-biased; runs last) | `data.bank` bank_r3 | `ab1_fe_mini_s0` (bank_r8) |

`ab2_pr_nhat_*` also carries the worker DSP block (fixed NLMS with the robust kernel, blocking, ref_policy) because
the `pr_nhat` input needs n_hat; the other VaaniFE arms use only the classical front end (limiter + ref_gain).

Data and loss are shared by every arm and by the full configs (2026-09-26): RIR bank `bank_r8`, tail share 0.40
(the G1-confirmed c5 default), the plan 11.5 corpora (DEMAND pairs, AVQ drone, C3GD, FSD50K, Lombard GRID), and for
every VaaniFE arm `loss_cfg.w_pesq: 0.001` with `pesq_required: true`.

Ablation 3b under the c5 mixer: plan 11.6 lists 0 / 10 / 25 %; the baseline is now 40 %, so the arms read 0 / 10 / 40 %
(final, Rachit 2026-09-26; `mix.v2.tail_share` stays 0.40 in every other config).
`tail_mix` (mono .25, stereo .125, low-ILD .625) scales with the share, so the 10 % arm is 2.5 % mono, 1.25 % stereo
and 6.25 % low-ILD. Only the share varies between arms; the composition is fixed on purpose (one factor per arm).

Priority order when rental hours run short (plan 11.6):
1 (gates the family), 3b, 2, 3, 4, 6, 7. Two seeds where listed; single-seed arms are read as directional only.

Ablation 7 (decision 2026-09-26: separate the bank change from the recipe change) is written by
`python scripts/gen_r8_configs.py --bank-arm`. Once `ab7_bank_r3.yaml` exists, `scripts/run_r8.sh` queues it last on
GPU 0 (beside its baseline), `--check` and the box tests stage verify it like any pilot, and the box setup fetches
bank_r3 as a trained-on bank. bank_r3 shares 1,394 rooms (27.9 %) with `bank.npz`, the bank of the eval_r2 val render
that selects checkpoints (about 246 of 1,480 val items, inferred), while bank_r8 shares none
(`results_r2/r8/banks/README.md`, `overlap_bank_r3.json`); on bank_r3 the M6 armoured scene/room pairing is also off
(no receiver_radius/armoured arrays). Its val scores are therefore biased in its favour against its baseline.

Decision (Rachit, 2026-09-26): run it anyway, last in priority (option b), so `ab7_bank_r3.yaml` is committed here. Read
its val gap to `ab1_fe_mini_s0` one way only: bank_r8 beating it is evidence for the bank change; ab7 beating bank_r8
is not evidence against it. It answers an attribution question, not a candidate-selection one, and it is the pilot
PILOT_HOURS drops first when the memory cap leaves 28 or fewer loader workers per queue.

Launch one arm per GPU, same as the full runs (see `../R8_RUNBOOK.md`):

    uv run --with numba python -m vaani.train configs/retraining/r8_ablations/ab1_fe_mini_s0.yaml

Every arm resumes from `runs/<name>/last.pt` when relaunched with the same command (`resume: true`).
Selection and all comparisons use **val only**; the r8 test set is scored once, for the final candidate (G6).

Selection is `val.select: composite` (final, Rachit 2026-09-25). Comparing arms by the
composite pass rate at 15 % exposure may be dominated by zeros early on, in which case read `val_stoi_ema` too.
