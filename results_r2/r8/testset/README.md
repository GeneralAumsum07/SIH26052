# results_r2/r8/testset: the pre-registered r8 test set

This folder holds no scores yet. It holds the pre-registration and the set's identity. The first CSV to land here is
the single post-selection scoring run described in `PROTOCOL.md` (Amendment 1).

- `PROTOCOL.md` is the pre-registration: contents, sources, held-out rules, seeds, metrics, the scoring command, and
  the score-once rule. **Amendment 1 (2026-09-26, Rachit's decision)** registers test root B. Where the original text
  and the amendment disagree, the amendment wins.
- The test root is `data/eval_r8_test_b/test`.
- `EVALSET_HASH` = `5bfda53eacbf`. It is a copy of `data/eval_r8_test_b/test/EVALSET_HASH`. Check it with
  `.venv/Scripts/python.exe scripts/verify_eval_set.py data/eval_r8_test_b/test 5bfda53eacbf`, which prints
  "verified (2308 items)".
- **Superseded root:** `data/eval_r8_test/test`, `EVALSET_HASH` `ed024af085a2`.
  - It was the original render of 2026-09-25 and the root registered at first. It stays on disk, frozen and unscored.
  - It matches B item for item outside the v2 scene subset (PROTOCOL.md Amendment 1). It still verifies with
    `scripts/verify_eval_set.py data/eval_r8_test/test ed024af085a2`.
  - `guard_frozen` refuses to re-render either root, even with `--force`.
- `index_summary.csv` has the items, buckets, and distinct noise and speech sources per `subset/category`.
- `v2_snr_quantiles.csv` has the per-scene quantiles of the output SNR (`snr_db`) of the v2 subset.
- The G6 table is written by `python scripts/r8_test_table.py` from the three per-item CSVs of the scoring commands in
  PROTOCOL.md Amendment 1 (`table.md` + `table_cells.csv` here). It refuses partial, duplicate or wrong-hash inputs.
- `configs/data/r8_test_sources.json` lists this set's speech and noise/impulse source_ids, so the box can check the
  held-out groups without the set.
  - It is written by `scripts/heldout_freesound.py --write-sources` from B's index and pins `5bfda53eacbf`.
  - `heldout_freesound.py --check` fails on a list written from any other set.

Both CSVs were built from `data/eval_r8_test_b/test/index.csv` (sha256 `c8fd562ac6eed91e...`) with:

```
.venv/Scripts/python.exe -c "import pandas as pd; d=pd.read_csv('data/eval_r8_test_b/test/index.csv'); \
d.groupby(['subset','category']).agg(items=('id','size'), buckets=('bucket','nunique'), noise_sources=('noise_source','nunique'), \
speech_sources=('speech_source','nunique')).reset_index().to_csv('results_r2/r8/testset/index_summary.csv', index=False); \
d[d.subset=='v2'].groupby('category').snr_db.describe()[['min','25%','50%','75%','max']].round(1).reset_index()\
.to_csv('results_r2/r8/testset/v2_snr_quantiles.csv', index=False)"
```

The earlier versions came from the superseded root's index (sha256 `99a62834...`) and are in git history.

What the rebuild from B changed:

- `index_summary.csv`: only v2 rows changed, and only `noise_sources`. Old -> new: apc 38 -> 45, artillery 30 -> 35,
  helicopter 46 -> 48, patrol 42 -> 45, windy_ridge 39 -> 40. Item, bucket and speech-source counts did not change.
- `v2_snr_quantiles.csv`: every 25/50/75 % quantile moved, except the windy_ridge 75 % (9.1 dB). The minimum is
  unchanged except for windy_ridge. The maximum is unchanged for artillery, command_post and drone. The medians are in
  the table below. The widest changes:
  - apc max 3.4 -> -1.9 dB;
  - firefight max 14.5 -> 19.2 dB;
  - patrol max 39.6 -> 33.0 dB;
  - windy_ridge range -10.8..31.6 -> -13.7..22.1 dB.

## Counts per subset

| subset | items |
|---|---:|
| v1 | 740 |
| defence | 576 |
| v2 | 384 |
| fault | 320 |
| heldout | 192 |
| loud | 96 |
| **total** | **2,308** |

eval_r2 test holds 2,280 items. The size was kept close to it and nothing was cut. Both roots have the same counts.

Median v2 output SNR per scene, in dB (from `v2_snr_quantiles.csv`):

| scene | B (5bfda53eacbf) | superseded (ed024af085a2) |
|---|---:|---:|
| apc | -12.9 | -11.8 |
| helicopter | -13.1 | -12.5 |
| firefight | -3.2 | -1.3 |
| windy_ridge | 2.9 | 2.7 |
| command_post | 7.9 | 8.3 |
| patrol | 10.2 | 13.0 |
| drone | 13.6 | 14.4 |
| artillery | 15.1 | 14.5 |

## Render and bank timing

The render timings come from a loaded, shared machine and are not reportable.

- B ran as the detached job calib-testset-b from 2026-09-25 23:34:25 to 23:37:27, with rc 0. It reused the eval bank.
- The superseded root ran as the detached job testset-render from 05:23:35 to 05:26:30 on 2026-09-25, with rc 0.
  That time includes building the 1,000-room eval bank with 3 workers.

## A smoke check of the scoring path, not a test score

`vaani.eval --system raw` was run on a separate 119-item scratch render (`--r8-size 1 --no-bank`, in a temporary
directory, not this set). It wrote all 119 rows, and the rows included the `category`, `noise_source` and
`impulse_source` columns. No system has been scored on `data/eval_r8_test_b` or on `data/eval_r8_test`.
