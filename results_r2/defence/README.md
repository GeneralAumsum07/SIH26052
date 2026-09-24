# Defence-noise evaluation (plan A3)

`data/eval_defence/test`, eval-set hash `d033568bdf98`, 1296 items: 6 categories x 6 input SNRs (-10, -5, 0, 5, 10,
15 dB) x 36 items, 6 s clips, test-split rows only. Results: [`table.md`](table.md) (per category x SNR means with
95% CIs, per-clip all-three pass rate, PASS/FAIL marks against the PS targets, r7-minus-raw and gtcrn-minus-raw
paired deltas, gaps). Per-item scores: `<system>.csv` (vaani/eval.py columns plus `category`, `noise_source`,
`impulse_source` from the set's `index.csv`).

## Headline (read from `table.md`)

- r7 cascade carries PASS on all three PS targets (CI lower bound above target) only at the easy end: blast_small_arms,
  blast_artillery and helicopter at 15 dB input SNR; vehicle and siren at 10 and 15 dB. `gunshot` never passes all
  three: at 15 dB input PESQ is 2.49 [2.29, 2.72] and SNR_out 14.90 [13.26, 16.59].
- At -10..0 dB input no category passes; SNR_out is 4-11 dB (siren excepted, 2 recordings only).
- r7 minus raw is positive in SNR_out and STOI in every category x SNR cell (paired per item, see the delta table).
- raw passthrough has pass3 = 0.00 in every cell (its STOI alone passes at 15 dB input, and for siren from 0 dB);
  gtcrn_pretrained is in `table.md` for comparison.

## What is in the set

| category | bed (speech-to-bed SNR) | transient on top |
|---|---|---|
| gunshot | MAD test stationary clips | recorded shot: gunshots.parquet test rows (Zenodo 7004819) + cadre.parquet test rows, cropped 2 s around the loudest sample |
| blast_small_arms | MAD test stationary clips | synthetic Friedlander, `vaani/data/blast.py` physics v2, small arms (150-160 dB SPL at 1 m, 1-300 m) |
| blast_artillery | MAD test stationary clips | synthetic, physics v2, artillery (Kinney-Graham, 0.1-20 kg TNT, 30-3000 m) |
| helicopter | MAD test helicopter clips | none |
| vehicle | MAD test vehicle/armoured clips | none |
| siren | ESC-50 test siren clips (only 2 exist) | none |

Speech: the eval_r2 recipe's test speech (librispeech.parquet + cv_hi.parquet test rows). Mix block = r7's training
mix (`configs/retraining/r7_e256_wr64.yaml` data.mix): transients 15-45 dB peak re speech RMS, `impulse_room` and
`overload_softclip` on, speech RMS -32..-18 dBFS; RIRs from `data/rirs/bank.npz`; seed 2609. The transient is
peak-normalised before it is placed, so the physics-v2 SPL sets the waveform shape (absorption, ground bounce,
N-wave), not its level in the mix.

Unique sources per category (from `index.csv`): beds 158 / 148 / 158 / 68 / 83 / 2 (gunshot, blast_small_arms,
blast_artillery, helicopter, vehicle, siren); recorded shots 178 (117 gunshots + 99 cadre items). Command:
`.venv/Scripts/python.exe -c "import pandas as pd; d=pd.read_csv('data/eval_defence/test/index.csv'); print(d.groupby('category').agg(beds=('noise_source','nunique'), imps=('impulse_source','nunique')))"`.

## Commands (repo root)

Render (about 2 min; refuses to overwrite a frozen set):

    .venv/Scripts/python.exe scripts/render_eval_sets.py --defence --manifests data/manifests/librispeech.parquet data/manifests/cv_hi.parquet data/manifests/mad.parquet data/manifests/esc50.parquet data/manifests/gunshots.parquet data/manifests/cadre.parquet --split test --out data/eval_defence --bank data/rirs/bank.npz --per-bucket 36 --seed 2609
    .venv/Scripts/python.exe scripts/verify_eval_set.py data/eval_defence/test d033568bdf98    # -> verified (1296 items)

Score (one per system; `r7_e256_wr64_cascade` is the same system spec as `scripts/r7_run.sh`):

    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system cascade:runs/r7_e256_wr64_refiner/best.pt --split test --eval-root data/eval_defence --workers 3 --dnsmos --out results_r2/defence/r7_e256_wr64_cascade.csv
    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system raw --split test --eval-root data/eval_defence --workers 1 --dnsmos --out results_r2/defence/raw.csv
    CUDA_VISIBLE_DEVICES=-1 uv run --with numba python -m vaani.eval --system gtcrn_pretrained --split test --eval-root data/eval_defence --workers 1 --dnsmos --out results_r2/defence/gtcrn_pretrained.csv

raw and gtcrn were scored with `--workers 1`: a first 3-worker raw run hung at item 881 after a pool worker died on
the loaded 16 GB machine (inferred: out of memory), so it was discarded and re-run. Worker count does not change
the scores (each item is scored independently).

Table:

    .venv/Scripts/python.exe scripts/defence_table.py --results results_r2/defence/r7_e256_wr64_cascade.csv results_r2/defence/raw.csv results_r2/defence/gtcrn_pretrained.csv

## Reading the CIs

Clustered bootstrap (`vaani/report.py` `cluster_ci`, 1000 resamples), cluster = the recording items share: the gunshot
recording for `gunshot`, the bed clip everywhere else. `siren` has only 2 bed clips, so its CIs rest on 2 clusters
and are not meaningful; treat siren as anecdotal.

## Not measured here

drone (all 1332 drone rows are train), NOISEX-92 (0 test rows), wind (8 ESC-50 test rows, not rendered), Lombard
speech (all speech is read in quiet), real recordings in noise (every item is a synthetic mixture).
