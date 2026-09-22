# Corpus licences

Generated from `data/manifests/*.parquet` by `scripts/licence_table.py`. Training recipe: `configs/retraining/r5_continue128.yaml`. Regenerate rather than edit.

## In the deployed recipe

| corpus | kind | clips | hours | licence | commercial use |
|---|---|---:|---:|---|---|
| dns_noise | noise | 7,944 | 22.0 | per-clip, see DNS README | unclear |
| drone | noise | 1,332 | 0.4 | citation-on-use | unclear |
| esc50 | noise | 1,760 | 2.4 | CC BY-NC | no |
| gunshots | noise | 2,117 | 2.6 | CC BY 4.0 | yes |
| mad | noise | 6,483 | 9.5 | YouTube-sourced | no |
| noisex92 | noise | 15 | 1.0 | SPIB redistribution terms unstated | unclear |
| cv_hi | speech | 10,971 | 14.2 | CC0 | yes |
| ears | speech | 894 | 5.2 | CC BY-NC | no |
| librispeech | speech | 28,539 | 100.6 | CC BY 4.0 | yes |

## Downloaded but not in the deployed recipe

| corpus | kind | clips | hours | licence | commercial use |
|---|---|---:|---:|---|---|
| cadre | noise | 2,121 | 1.2 | US DOJ/NIJ award output | unclear |
| demand | noise | 18 | 1.5 | CC BY-SA 4.0 | yes, share-alike |
| dns_noise | noise | 7,997 | 22.2 | per-clip, see DNS README | unclear |
| dns_noise | noise | 7,998 | 22.2 | per-clip, see DNS README | unclear |
| dns_noise | noise | 2,287 | 6.3 | per-clip, see DNS README | unclear |
| librispeech | speech | 5,820 | 20.0 | CC BY 4.0 | yes |

## What this means for transfer

Non-commercial material in the deployed recipe: **ears, esc50, mad**. Unresolved redistribution terms: **dns_noise, drone, noisex92**.

The research prototype trains on corpora including non-commercial-licensed and unclear-licence material. A fielded version would retrain on licensed or government-collected data; the pipeline is corpus-agnostic and the manifest layer makes the substitution mechanical -- a recipe is a list of manifest paths, and nothing in the model, the DSP front end or the training loop is tied to a particular corpus.

This is a licence statement, not a legal opinion, and `unclear` rows are unresolved rather than cleared.
