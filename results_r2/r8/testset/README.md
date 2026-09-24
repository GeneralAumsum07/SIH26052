# results_r2/r8/testset: the pre-registered r8 test set

This folder holds no scores yet. It holds the pre-registration and the set's identity. The first CSV to land here is
the single post-selection scoring run described in `PROTOCOL.md`.

- `PROTOCOL.md` is the pre-registration: contents, sources, held-out rules, seeds, metrics, the scoring command, and
  the score-once rule.
- `EVALSET_HASH` = `ed024af085a2`. It is a copy of `data/eval_r8_test/test/EVALSET_HASH`.
  Check it with `.venv/Scripts/python.exe scripts/verify_eval_set.py data/eval_r8_test/test ed024af085a2`, which
  prints "verified (2308 items)".
- `index_summary.csv` has items, buckets, and distinct noise and speech sources per `subset/category`.
- `v2_snr_quantiles.csv` has the per-scene quantiles of the output SNR (`snr_db`) of the v2 subset.

Both CSVs were built from `data/eval_r8_test/test/index.csv` (sha256 `99a62834...`) with:

```
.venv/Scripts/python.exe -c "import pandas as pd; d=pd.read_csv('data/eval_r8_test/test/index.csv'); \
d.groupby(['subset','category']).agg(items=('id','size'), buckets=('bucket','nunique'), noise_sources=('noise_source','nunique'), \
speech_sources=('speech_source','nunique')).reset_index().to_csv('results_r2/r8/testset/index_summary.csv', index=False); \
d[d.subset=='v2'].groupby('category').snr_db.describe()[['min','25%','50%','75%','max']].round(1).reset_index()\
.to_csv('results_r2/r8/testset/v2_snr_quantiles.csv', index=False)"
```

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

eval_r2 test holds 2,280 items. The size is kept close to it and nothing was cut.

Median v2 output SNR per scene, in dB (from `v2_snr_quantiles.csv`):

| scene | median SNR (dB) |
|---|---:|
| apc | -11.8 |
| helicopter | -12.5 |
| firefight | -1.3 |
| windy_ridge | 2.7 |
| command_post | 8.3 |
| patrol | 13.0 |
| drone | 14.4 |
| artillery | 14.5 |

## Render and bank timing

The render ran as the detached job testset-render from 05:23:35 to 05:26:30, with rc 0. That time includes building
the 1,000-room eval bank with 3 workers. The timing comes from a loaded, shared machine and is not reportable.

## A smoke check of the scoring path, not a test score

`vaani.eval --system raw` was run on a separate 119-item scratch render (`--r8-size 1 --no-bank`, in a temporary
directory, not this set). It wrote all 119 rows, and the rows included the `category`, `noise_source` and
`impulse_source` columns. No system has been scored on `data/eval_r8_test`.
