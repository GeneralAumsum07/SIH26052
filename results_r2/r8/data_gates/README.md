# G1 data gate: is ILD alone a speech-vs-noise shortcut? (plan 11.2 / 11.8)

**Status 2026-09-25:** passes with the c5 defaults on the fresh seed 202, 200 items: param 0.699 [0.651, 0.745],
room 0.665 [0.620, 0.709]. See "G1 confirmation" below, including the caveat. The first section is the original
48-item run; its defaults are superseded.

Per STFT bin of the mixer output, ILD = 10 log10(|P|^2/|R|^2); bins with local SNR > +10 dB are speech bins,
< -10 dB noise bins; AUC = P(ILD of a speech bin > ILD of a noise bin). Gate: v2 AUC <= 0.75 on both paths,
and the M2 reference-gain draw puts >= 25 % of items in -6..+3 dB. Split: train, 48 items of 6 s, seed 55.

Command (from the repo root, 2026-09-25):

    uv run --with numba python scripts/data_gates.py --versions 1 2 --items 48 --seed 55
    uv run --with numba python scripts/data_gates.py --bench 200

| mixer | path  | AUC(ILD) | speech-bin ILD p10/p50/p90 dB | noise-bin ILD p10/p50/p90 dB | source |
|-------|-------|---------:|-------------------------------|------------------------------|--------|
| v1    | param | 0.989    | 7.4 / 13.0 / 19.2             | -4.9 / 0.8 / 4.6             | v1.json |
| v1    | room  | 0.874    | 5.9 / 10.9 / 15.7             | -6.1 / 1.5 / 10.5            | v1.json |
| v2    | param | 0.727    | -0.6 / 12.2 / 17.9            | -6.9 / 3.6 / 11.1            | v2.json |
| v2    | room  | 0.697    | -1.0 / 11.3 / 18.2            | -7.3 / 2.7 / 13.7            | v2.json |

- v1 reproduces the shortcut (plan 11.1 quoted about 0.975 / 0.897 on a different sample; same direction and size).
- v2 passes (`gate_pass: true` in v2.json). M2 draw (20,000 draws, v2.json `m2_draw`): physical 74.9 %,
  low-ILD tail 9.9 %, mono 10.0 %, produced-stereo 5.2 %; share in -6..+3 dB = 25.1 %.
- SPL round trip (both files): max error 7.6e-7 dB over 40-123 dB SPL; a 94 dB SPL sine peaks at -26.00 dBFS.

## v2 parameter changes made to pass the gate

With the first v2 defaults the gate failed (16 items, seed 55: 0.899 param / 0.846 room). Sweeps (scratch, not kept):
the tail share barely moves AUC (tail 0.5: 0.883 / 0.821); positive-ILD near-field noise is the lever.
Sweep 2 (32 items, seed 77): near_pos_share 0.8 + p_near 0.7 -> 0.735 / 0.697; 0.85 + 0.8 -> 0.748 / 0.701;
0.85 + 0.8 + near_spl_rel [0, 8] -> 0.736 / 0.687. Defaults now (all inside plan 11.5 M3, ILD +-12 dB of either sign):

- `vaani/data/mixer.py` V2_DEFAULTS `near_pos_share` 0.5 -> 0.8 (near-field noise louder at the boom mic 80 % of the time)
- `vaani/data/scenes.py` `P_NEAR` 0.35 -> 0.7, `NEAR_SPL_REL` (-6, 6) -> (0, 8) dB re the bed

Inferred at the time: the 48-item margin (0.727 vs 0.75) was small, and the 16-item seed-55 run of a near-identical
setting gave 0.756 room. The runs below show that it was also selection bias, because seed 55 was a tuning seed.
**The defaults in this section are superseded by the c5 defaults below.**

## G1 confirmation, retuning and fresh-seed confirmation (2026-09-25)

Every run below uses v2 only, the train split and bank_r3, with 200 items unless stated. Bootstrap CIs resample items
with replacement; the bins stay with their item (`--bootstrap B`). The per-item ILD histograms are
`items_v2_<path>.{csv,npz}` in each directory. `breakdown` in each v2.json splits the AUC by scene, ref mode, near
source, wind, M2 bucket and dominant noise component.

### Uncertainty of the first pass, and the failure

These use the defaults above and the legacy item seeds `seed + i`. They ran before the seed fix; reproduce them with
`--legacy-seeds`.

| run | dir | param AUC [CI95] | room AUC [CI95] | gate |
|---|---|---|---|---|
| seed 55, 48 items (a tuning seed) | baseline_s55_n48 | 0.727 [0.606, 0.839] | 0.697 [0.590, 0.797] | pass |
| seed 101, 200 items | baseline_s101_n200 (= confirm200_s101) | 0.786 [0.742, 0.828] | 0.751 [0.708, 0.792] | fail |

Commands: `docs/impl/2026-09-24/jobs/calib-g1base-s55/cmd` and `calib-g1base-s101/cmd`:
`CUDA_VISIBLE_DEVICES=-1 uv run --with numba python scripts/data_gates.py --versions 2 --items 48 --seed 55 --bootstrap 2000 --out results_r2/r8/data_gates/baseline_s55_n48`.
Both reproduce the earlier numbers exactly. At 48 items the CI is about +-0.12 wide, so the first pass was never
evidence of a pass.

### Seed-scheme bug

With `seed + i`, a seed 202 x 200 run would reuse items 101-199 of seed 101. `item_rng` now uses `[seed, i]`, which
gives independent streams. `--legacy-seeds` keeps the old scheme. Every run below uses `[seed, i]`.

### What makes ILD separable

From baseline_s101_n200, param path:
- Physical-mode items (147 of 200) have a within-group AUC of 0.897. The M2 tail items (mono, stereo, low ILD; 53)
  sit at 0.50-0.51.
- Negative-ILD near sources are the easiest noise to separate (within-group 0.947). Positive near sources are the
  hardest (0.783).
- By component, speech against bed-dominated noise bins gives 0.812, and against near-dominated bins 0.754.
- By scene, drone (0.881) and patrol (0.840) separate most easily, and artillery (0.673) least.
- Clipped and unclipped items do not differ (0.793 / 0.790).

### Tuning on the seed set {55, 77, 101}

The knobs and ranges come from plan 11.5:
- M2 tail: at least 25 % of items in -6..+3 dB, with about 10 % mono and 5 % produced-stereo.
- M3 near-field noise: ILD within +-12 dB, of either sign.

Every setting tried is listed. "Pooled" is the 600 items of the three seeds
(`--from-items ... --pooled-out tune/cN_pooled`). The rule was fixed before pooling: take the smallest tail with
pooled param and pooled room both <= 0.725.

| id | change vs the old defaults | s55 param/room | s77 | s101 | pooled param [CI95] | pooled room [CI95] | M2 -6..+3 share |
|---|---|---|---|---|---|---|---|
| c0 | none | .827/.799 | .832/.810 | .793/.764 | .818 [.797, .839] | .792 [.772, .812] | .251 |
| c1 | near_pos_share .9, near ILD 8-12 dB | .810/.781 | .816/.798 | .772/.755 | .800 [.779, .821] | .779 [.759, .800] | .251 |
| c2 | c1 + p_near .9 | .791/.760 | .804/.785 | .784/.759 | .794 [.774, .816] | .769 [.747, .791] | .251 |
| c3 | c2 + tail .35, old mix (mono 14 %, outside plan) | .744/.719 | .756/.743 | .739/.719 | .747 [.722, .772] | .727 [.703, .753] | .348 |
| c4 | c2 + tail .35, mix mono .2857 / stereo .1429 / low .5714 | .737/.710 | .761/.750 | .730/.716 | .742 [.718, .767] | .725 [.700, .749] | .353 |
| **c5** | c2 + tail .40, mix mono .25 / stereo .125 / low .625 | .718/.691 | .747/.737 | .707/.693 | **.7235 [.697, .750]** | **.707 [.681, .733]** | .398 |
| c6 | c2 + tail .45, mix mono .2222 / stereo .1111 / low .6667 | .700/.675 | .737/.722 | .666/.658 | .702 [.675, .730] | .686 [.661, .712] | .456 |

The rule selects c5. Its tail makes 10 % of items mono, 5 % produced-stereo and 25 % low-ILD, so 40 % of items lie in
-6..+3 dB. The commands are in the `cmd` files under `docs/impl/2026-09-24/jobs/calib-t0-s*`, `calib-t123-s*` and
`calib-t456-s*`.

### One-shot confirmation on the fresh seed 202

Seed 202 was never tuned on. It ran once, before c5 was made the default:

    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python scripts/data_gates.py --versions 2 --items 200 --seed 202 --bootstrap 2000 --scene '{"p_near": 0.9}' --v2 '{"near_pos_share": 0.9, "near_ild_db": [8.0, 12.0], "tail_share": 0.40, "tail_mix": {"mono": 0.25, "stereo": 0.125, "low_ild": 0.625}}' --out results_r2/r8/data_gates/confirm200_s202

| run | param AUC [CI95] | room AUC [CI95] | M2 -6..+3 share (20,000 draws) | gate |
|---|---|---|---|---|
| confirm200_s202 | 0.699 [0.651, 0.745] | 0.665 [0.620, 0.709] | 0.398 (mono .100, stereo .048, low ILD .250) | **pass** (`gate_pass: true`) |

After this run c5 became the default (commit 4dcfa90):
- mixer `V2_DEFAULTS`: `tail_share` 0.40; `tail_mix` {mono .25, stereo .125, low_ild .625}; `near_ild_db` (8, 12);
  `near_pos_share` 0.9.
- scenes: `P_NEAR` 0.9.

### Second fresh seed, 5150 (committed defaults, no overrides)

Run once, after c5 became the default, as an adversarial re-check; seed 5150 was never tuned on:

    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python scripts/data_gates.py --versions 2 --items 200 --seed 5150 --bootstrap 2000 --out results_r2/r8/data_gates/confirm200_s5150

| run | param AUC [CI95] | room AUC [CI95] | M2 -6..+3 share (20,000 draws) | gate |
|---|---|---|---|---|
| confirm200_s5150 | 0.732 [0.689, 0.775] | 0.701 [0.657, 0.742] | 0.398 | **pass** (`gate_pass: true`) |
| pooled fresh 400 items (s202 + s5150) | 0.7156 [0.683, 0.748] | 0.6831 [0.653, 0.714] | - | - |

Physical-mode items alone (115 of 200): 0.897 (param) / 0.849 (room). The pooled-fresh row was computed with
`--from-items` over the two item files (docs/impl/2026-09-24/reports/calib-verify.md, G1 detail). The rule that
selected c5 ("smallest tail with pooled param and room <= 0.725") has no timestamped record from before pooling; two
fresh seeds passing is the mitigation.

### Caveat: the pass rests on the M2 tail

The gate is a pooled AUC, and the out-of-physics M2 tail carries the pass. In confirm200_s202 the physical-mode items
alone (118 of 200) still give a within-group AUC of 0.858 (param) and 0.805 (room); the tail items sit at 0.50-0.52.
So ILD is still a strong cue on the physical distribution; the 40 % tail only stops it being a sufficient cue. Whether
that meets the intent of G1 is Rachit's call (calib report, Decisions).

### Training configs must carry these values

Done in 015f044: `configs/retraining/r8_fe_mini.yaml`, `r8_refvalid_v2.yaml` and the `r8_ablations/*.yaml` pin
`data.mix.v2.tail_share: 0.4` (the ab3b arms keep 0.0 and 0.1), and they train on bank_r8. The room path on bank_r8
passes on fresh seed 7331: results_r2/r8/g1_bank_r8/README.md.

The M2 criterion in `data_gates.py` is `share >= 0.25` (plan M2). Before 2026-09-26 it was `>= 0.24`; every run
recorded here has a share of 0.251 or more, so no gate result changes.

## Mixer throughput (smoke, not reportable)

`results_r2/r8/mixer_bench.json`: 200 items of 4 s, in-memory audio, one process, numba on, RIR bank loaded, on a
shared loaded machine: v1 6.5 ms/item (153 items/s), v2 23.3 ms/item (43 items/s).
