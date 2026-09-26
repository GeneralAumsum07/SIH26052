# r8 training RIR bank (bank_r8, M6 receiver radius)

`data/rirs/bank_r8.npz` is the r8 training bank: 5000 rooms, 3 noise sources, 20 % armoured-vehicle interiors,
1.0 s RIRs, like bank_r3, but drawn from a new `train` seed namespace and ray-traced with the M6 receiver radius
0.05 m instead of the legacy 0.3 m. Fetch table: `configs/data/r8_banks.json`. The file is git-ignored; it travels
by GitHub release asset and sha256.

## Build

```
.venv/Scripts/python.exe scripts/make_rir_bank.py --out data/rirs/bank_r8.npz --n 5000 --seed 8 --seed-namespace train \
    --armoured-frac 0.2 --max-len-s 1.0 --receiver-radius 0.05 --workers 5 --parts-dir data/rirs/bank_r8.parts
```

Started 17:22 IST 2026-09-25, finished 20:16 IST, rc 0: 10386 s on 5 workers (log: `docs/impl/2026-09-24/jobs/banks-build/log`).
SMOKE timing, non-reportable: shared 16-core laptop with other jobs running. Resumable: a relaunch of the same command
reuses the finished 50-room part files in `--parts-dir` (checked against a sha256 of the drawn parameters).

Only armoured rooms use the pyroomacoustics ray tracer, so only they see the receiver radius (`rirs.simulate_from_params`:
`armoured -> ray_tracing=True`); plain rooms are image-source only. Rays scale as `(0.3/r)^2` to keep the rays per
receiver area constant (`rirs.ARMOURED_RAYS`).

bank_r3 provenance (for comparison): `make_rir_bank.py --out data/rirs/bank_r3.npz --armoured-frac 0.2 --max-len-s 1.0`,
n 5000, seed 0, legacy `default_rng(0)` stream, radius 0.3 m, one process (evidence: `scripts/remote_setup.sh` fallback
line; `bench_radius.py` re-derives bank_r3's draws and reproduces the stored speech/noise RIRs with maxdiff 0.0 and all
5000 rt60 and the armoured mask equal, `bench_radius.json` "rederive_check").

## Radius choice (smoke)

`bench_radius.py --rooms 5 --workers 5` -> `bench_radius.json` (5 of bank_r3's armoured rooms at each radius, medians):

| radius | s/room (5 loaded workers) | noise T30 mic0 / mic1 (s) | noise late-envelope corr | speech DRR mic0 (dB) |
|---|---|---|---|---|
| 0.30 m (legacy) | 4.5 | 0.733 / 0.743 | 0.997 | 23.3 |
| 0.06 m | 69.7 | 0.742 / 0.740 | 0.997 | 23.3 |
| 0.05 m | 66.5 | 0.743 / 0.739 | 0.998 | 23.0 |

Plain (image-source) rooms: 0.03 s/room. Isolated single-process ray-trace time is linear in the ray count (room 9:
10k rays 0.80 s, 250k 20.96 s, 360k 32.26 s). Decay and envelope stats do not move with the radius beyond run-to-run
noise. The speech late energy at the reference mic does swing up to about 5 dB between runs at the SAME radius
(`probe_tail_variance.py --radius 0.3 --reps 3` -> `probe_tail_variance.json`: bank_r3 room 7, speech at the
reference, late-from-50 ms 10.27 / 5.72 / 10.02 dB while late-from-100 ms is 5.88 / 5.72 / 5.87 dB); inferred cause: pyroomacoustics' sparse random onset of the synthesised tail.

Picked 0.05 m (`rirs.M6_RECEIVER_RADIUS`). The mic pair is 0.1202 m apart, so 0.06 m spheres would leave 0.2 mm
between them; 0.05 m leaves 2 cm, keeping the two receivers separate as M6 (plan 11.5) asks. Cost at 0.05 m is the
same as at 0.06 m.

## Composition: fresh bank (A), not a merge (B)

(B) would keep bank_r3's 4000 plain rooms and re-trace only the 1000 armoured ones. It saves about 2 CPU-minutes
(4000 x 0.03 s), so cost gives no reason for it, and it would carry bank_r3's rooms that also sit in the legacy
`bank.npz` (rendered into eval_r2 / val) into r8 training. (A) draws from the `train` namespace
(`rirs.bank_rng(seed, "train")`), disjoint by construction from `eval` (bank_eval_r8) and the legacy stream.
`scripts/merge_rir_banks.py` was therefore not written.

## Validation

```
.venv/Scripts/python.exe results_r2/r8/banks/validate_bank.py --bank data/rirs/bank_r8.npz --ref data/rirs/bank_r3.npz \
    --overlap data/rirs/bank.npz data/rirs/bank_r3.npz data/rirs/bank_eval_r8.npz
```
-> `validate.json`, `validate.csv` (log: `docs/impl/2026-09-24/jobs/banks-validate/log`). The bank_r3 here is the
laptop copy (sha256 99dcfb26...), not the published box copy (e4e67463...) r7 trained on.

- `RirBank` loads it; all three sidecars are `np.memmap`; shapes speech (5000, 2, 16000), noise (5000, 3, 2, 16000);
  all finite, no all-zero RIR, max |h| 39.05 (bank_r3 39.09).
- Metadata in the npz: `receiver_radius` 0.05, `seed_namespace` "train"; armoured 1000/5000 (0.200).
- Mixer v2 (40 items per scene, `MixConfig(version=2)`): apc 26 room-path items, 26 armoured rooms served, 0 plain;
  command_post 25 room-path items, 0 armoured, 25 plain; patrol 0 room-path items. On bank_r3 the same draws gave apc
  3 armoured / 23 plain and command_post 9 armoured / 16 plain (legacy banks keep their uniform draw, commit 5918277).
- Room overlap (same rt60 and identical speech RIR): 0 with bank.npz (1 rt60 match), 0 with bank_r3 (3), 0 with
  bank_eval_r8 (0).

Stats per room class, median [p10, p90], all 5000 rooms:

| metric | class | bank_r3 | bank_r8 |
|---|---|---|---|
| rt60 drawn (s) | plain | 0.302 [0.146, 0.460] | 0.299 [0.141, 0.457] |
| rt60 drawn (s) | armoured | 0.652 [0.449, 0.845] | 0.653 [0.445, 0.848] |
| noise T30 primary (s) | plain | 0.197 [0.118, 0.255] | 0.197 [0.114, 0.255] |
| noise T30 primary (s) | armoured | 0.636 [0.432, 0.813] | 0.632 [0.428, 0.818] |
| noise late (>100 ms) share (dB) | armoured | -10.05 [-14.38, -7.93] | -9.98 [-14.62, -7.94] |
| speech DRR primary (dB) | plain | 28.63 [25.70, 32.65] | 28.75 [25.77, 32.71] |
| speech DRR primary (dB) | armoured | 24.09 [21.69, 25.37] | 24.02 [21.16, 25.35] |
| speech ILD (dB) | armoured | 11.23 [10.58, 11.71] | 11.23 [10.57, 11.69] |
| noise late-envelope corr | armoured | 0.998 [0.996, 0.999] | 0.998 [0.996, 0.999] |
| noise late-envelope rms diff (dB) | armoured | 0.90 [0.79, 1.03] | 0.95 [0.82, 1.10] |
| noise late waveform corr | armoured | 0.000 [-0.038, 0.038] | -0.002 [-0.037, 0.036] |

The rest (T30 at the reference, plain-room envelope rows) is in `validate.csv`.

## bank_r3 as an ablation arm: room overlap with the val render (2026-09-26)

Question: can a pilot on bank_r3 (to separate the bank change from the recipe change) be compared with
`ab1_fe_mini_s0` on val? The val render that selects checkpoints (`data/eval_r2/val`, hash b5f7a4d43bee) used
`data/rirs/bank.npz`: `scripts/render_all_eval_sets.sh` and `scripts/remote_setup.sh` render it with no `--bank`, and
`render_eval_sets.py --bank` defaults to `data/rirs/bank.npz`.

```
.venv/Scripts/python.exe results_r2/r8/banks/overlap_bank_r3.py results_r2/r8/banks/overlap_bank_r3.json
```
The script re-derives every bank's drawn room parameters from the builder's seed code (`rirs.bank_rng`,
`draw_room_params`), with no simulation, and then checks the shared pairs on the laptop files.

- The derivations are the banks. They reproduce every stored rt60: bank 5000/5000, bank_r3 5000/5000, bank_r8
  5000/5000, bank_eval_r8 1000/1000. bank.npz is the legacy `default_rng(0)` stream with no armoured shuffle. It
  predates c053f7b, and the current `build_bank(armoured_frac=0)` shuffles the all-plain mask, which consumes draws, so
  it no longer rebuilds bank.npz. The stored rt60 match 0/5000 under the shuffled stream.
- **bank_r3 and bank.npz share 1394 rooms**, 27.9 % of bank.npz (plan 3.5's "28 %"). All 1394 are plain rooms, and the
  streams realign after bank_r3's armoured draws (first pairs: bank_r3 room 8 is bank.npz room 209, 10 is 211, and so
  on). On the laptop files, all 1394 speech RIR pairs are bit-equal over bank.npz's 0.6 s, and bank_r3's extra 0.4 s
  is all zeros. bank_r3 is the laptop copy (sha256 99dcfb26...). The plain rooms are image-source only, so the
  published copy is inferred to share the same rooms.
- 0 shared rooms for bank_r3 ~ bank_eval_r8 (the r8 test bank), bank_r8 ~ bank.npz, bank_r8 ~ bank_eval_r8 and
  bank_r3 ~ bank_r8.
- Val exposure: 884 of the 1480 val items take the room path (596 take the parametric path; count of `path` in the
  item JSONs). With the uniform room draw, about 0.279 x 884 = 246 val items (about 1 in 6) use a room that is also in
  bank_r3 (inferred; the item JSONs do not record the room index).

So an arm trained on bank_r3 trains on the exact RIRs of about a sixth of the val items, and ab1 on bank_r8 does not.
Its val scores, and the checkpoint selection built on them, would be biased toward bank_r3. The arm was **not added**
to `configs/retraining/r8_ablations/` (docs/impl/2026-09-24/reports/boxplan.md, Decisions). r7 trained on bank_r3 and
was selected on this val set, so r7's own selection carried the same overlap (plan 3.5).

## sha256

| file | sha256 |
|---|---|
| bank_r8.npz (812018800 bytes) | 325ef372776de592ed1fb865f0e82a3dec1428ab65ea69d75afc01a680e1e3a7 |
| bank_r8.speech.npy | 89d3f4b793a1cc0c21489ee9467d593c4e11d691ca1ffbbb3e669f88d740a687 |
| bank_r8.noise.npy | 821719936d172e157a23f09ff0ac5418fdb1a4e6313a9cd656057ca3d158daf5 |
| bank_r8.rt60.npy | 4c33401d4f476e49da544253a4c95b2ddfcba6e2297783723d7679bce6311a10 |

The sidecars are not uploaded: `RirBank` writes them from the npz on first load with `np.save`. Re-saving `rt60` from
the npz under numpy 2.5.3 reproduced the hash above byte for byte; that the box's numpy writes the same header is
inferred (same `.npy` format 1.0), and `scripts/r8_preflight.py` checks it.

## Upload (Rachit)

`scripts/remote_setup.sh` fetches every bank from ONE base URL (`$RIR_BANK_URL/<file>`), and bank.npz / bank_r3.npz
are already on `rir-banks-2026-09-21`. Recommended: add bank_r8 to that release, so the RIR_BANK_URL in use keeps working.

```
cd C:/Users/Rachit/Desktop/Projects/SIH_2026
gh release download rir-banks-2026-09-21 -p BANKS.sha256 -D data/rirs --clobber
echo "325ef372776de592ed1fb865f0e82a3dec1428ab65ea69d75afc01a680e1e3a7  bank_r8.npz" >> data/rirs/BANKS.sha256
gh release upload rir-banks-2026-09-21 data/rirs/bank_r8.npz data/rirs/BANKS.sha256 --clobber
```
(`--clobber` replaces only BANKS.sha256; bank_r8.npz is new. Then add a bank_r8 paragraph to the release notes.)

Alternative, a new tag (then RIR_BANK_URL must point at it, and the old banks have to be copied across, since
the fetch uses a single base; about 1.2 GB down and up again):
```
gh release download rir-banks-2026-09-21 -p bank.npz -p bank_r3.npz -p BANKS.sha256 -D rel_tmp
echo "325ef372776de592ed1fb865f0e82a3dec1428ab65ea69d75afc01a680e1e3a7  bank_r8.npz" >> rel_tmp/BANKS.sha256
gh release create rir-banks-2026-09-25 rel_tmp/bank.npz rel_tmp/bank_r3.npz data/rirs/bank_r8.npz rel_tmp/BANKS.sha256 \
    --title "RIR banks 2026-09-25" --notes "Adds bank_r8.npz (r8 training bank, M6 receiver radius 0.05 m); see results_r2/r8/banks/README.md"
```

## Box fetch

```
export RIR_BANK_URL=https://github.com/GeneralAumsum07/SIH26052/releases/download/rir-banks-2026-09-21
aria2c -c -x8 -s8 -k 10M --max-tries=5 -d data/rirs -o bank_r8.npz "$RIR_BANK_URL/bank_r8.npz"
echo "325ef372776de592ed1fb865f0e82a3dec1428ab65ea69d75afc01a680e1e3a7  data/rirs/bank_r8.npz" | sha256sum -c -
```
The same `fetch_bank bank_r8.npz <sha256>` line belongs in `scripts/remote_setup.sh` / `scripts/r8_box_setup.sh`
(box-owned; not added here). A regenerated bank would NOT reproduce these bytes (ray tracing draws from the global
`np.random`), so the box must fetch, never rebuild, if runs are to be comparable.

## Measured RIRs (BUT ReverbDB): feasibility only

From the publisher's `read_me.txt` (http://merlin.fit.vutbr.cz/ReverbDB/read_me.txt, text only, fetched 2026-09-25):
layout `PLACE/MIC_SETUP/SPK_SETUP/MIC_ID/RIR/*.wav`, 16 kHz mono 16-bit; every RIR is "compensated for speaker ->
microphone delay"; per-mic absolute and speaker-relative positions are in `env_full_meta.txt` / `mic_meta.txt`
(one room is flagged as having untrustworthy measured positions). Licence CC BY 4.0 on the current page
(`docs/impl/2026-09-24/research/datasets_astra.md` item 19). The RIR-only archive is 9308593693 bytes (header probe).

- A 12 cm headset pair cannot be read straight out of it: the mics are distant room mics placed around a
  loudspeaker, and the per-mic delay compensation removes the inter-mic TDOA. A pair would need two mics about 0.12 m
  apart in the metadata (TBD: does any MIC_SETUP have one?), with the delay put back from the positions.
- Neither channel is a near-field mouth path, so at best measured RIRs could stand in for the NOISE-source paths
  (both mics far from the source), with the speech path staying simulated. That is the likely shape of a measured
  bank: `noise` from ReverbDB pairs, `speech` from the ISM, `rt60` from the metadata or a Schroeder fit.
- `vaani.data.sources.scan_but_reverbdb` already makes single-mic mono rows (kind "rir"); a pair bank would need a
  new builder that reads the metadata. TBD on the box after the download: mic spacing distribution per MIC_SETUP.

## Files

- `bench_radius.py` / `bench_radius.json`: radius smoke and bank_r3 re-derivation check.
- `probe_tail_variance.py` / `probe_tail_variance.json`: run-to-run tail variance at a fixed radius.
- `validate_bank.py` / `validate.json` / `validate.csv`: loader, mixer v2 draws, stats, overlap, sha256.
- `overlap_bank_r3.py` / `overlap_bank_r3.json`: re-derived room draws of bank, bank_r3, bank_r8 and bank_eval_r8,
  their shared rooms, and the bank_r3 ~ bank.npz pairs checked on the files.
