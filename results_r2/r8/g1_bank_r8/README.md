# G1 room path on bank_r8 (the bank every r8 mixer v2 config trains on)

The G1 confirmations in [../data_gates/README.md](../data_gates/README.md) ran on bank_r3. The r8 configs switched to
bank_r8 (M6 receiver radius 0.05 m), so the room path was rerun on it once, on a fresh seed, with the committed c5
mixer defaults (no overrides; `mix.v2.tail_share` 0.40 is the default the configs set).

Command (from the repo root, CPU, 2026-09-26):

```
CUDA_VISIBLE_DEVICES=-1 uv run --with numba python scripts/data_gates.py --versions 2 --paths room --items 200 \
    --seed 7331 --bank data/rirs/bank_r8.npz --bootstrap 2000 --out results_r2/r8/g1_bank_r8
```

(The run wrote to a scratch folder and was copied here unchanged; bank sha256 325ef372... per configs/data/r8_banks.json.)

| quantity | value | source |
|---|---|---|
| v2 room ILD-only AUC | 0.659, CI95 [0.609, 0.708] | `v2.json` room.auc_ild, log |
| gate (AUC <= 0.75) | pass | `v2.json` gate_pass |
| M2 share of the 20,000-draw (-6..+3 dB) | 0.398 | `v2.json` m2_draw.share_-6_to_+3 |
| items by ref mode | physical 115, low_ild 59, mono 18, stereo 8 | `v2.json` room.ref_modes |
| physical-mode items only | 0.817 (115 items) | `hist_auc` over the physical rows of `items_v2_room.npz` (below) |

Physical-only number:

```
.venv/Scripts/python.exe -c "import numpy as np,csv,sys; sys.path.insert(0,'scripts'); from data_gates import hist_auc; \
z=np.load('results_r2/r8/g1_bank_r8/items_v2_room.npz'); r=list(csv.DictReader(open('results_r2/r8/g1_bank_r8/items_v2_room.csv'))); \
m=np.array([x['ref_mode']=='physical' for x in r]); print(hist_auc(z['S'][m].sum(0), z['N'].sum(1)[m].sum(0)))"
```

### Gate AUC and physical-only AUC side by side (2026-09-26)

**Decision (Rachit, 2026-09-26): accept G1 as written; report the physical-only AUC beside it.**

| path | items (physical / tail) | gate AUC (rank, `auc_ild`) [CI95] | physical-only AUC (reported, not gated) [CI95] | gate_pass |
|---|---|---|---|---|
| room, bank_r8, seed 7331 | 115 / 85 | 0.659 [0.609, 0.708] | 0.817 [0.790, 0.841] | true |

Source: `v2.json` `room.auc_ild`, `room.bootstrap.ci95`, `room.physical_only` (auc_ild, bootstrap.ci95, items,
items_tail) and `gate_pass`. The `physical_only` block was added without re-rendering:

```
.venv/Scripts/python.exe scripts/data_gates.py --from-items results_r2/r8/g1_bank_r8 --bootstrap 2000 --add-physical-only --pooled-out <scratch>
```

The command recomputed the item statistics from `items_v2_room.{csv,npz}`. The item count, `breakdown` and
`bootstrap` matched the recorded ones exactly, so it added `room.physical_only`: 0.8172, the same value as the
one-liner above and as `breakdown.by_ref_mode.physical.auc_within`. `gate_pass` was re-derived and is unchanged
(true). `git diff` of v2.json shows only the added block. The physical-only CI is an item bootstrap over the 115
physical items (B 2000, seed 0).

Reading: the pooled pass comes from the out-of-physics reference tail, as on bank_r3 (physical-only 0.85-0.90 there).
The param path does not read a bank, so it is not rerun here. The noise pool is `data_gates.V2_NOISE`, not the r8
training corpus list.
