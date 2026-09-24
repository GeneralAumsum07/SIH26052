# G1 data gate: is ILD alone a speech-vs-noise shortcut? (plan 11.2 / 11.8)

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

Inferred: the 48-item margin (0.727 vs 0.75) is small and the 16-item seed-55 run of a near-identical setting gave
0.756 room, so AUC varies by about +-0.03 between samples. TBD: is a 200-item confirmation run wanted before r8?

## Mixer throughput (smoke, not reportable)

`results_r2/r8/mixer_bench.json`: 200 items of 4 s, in-memory audio, one process, numba on, RIR bank loaded, on a
shared loaded machine: v1 6.5 ms/item (153 items/s), v2 23.3 ms/item (43 items/s).
