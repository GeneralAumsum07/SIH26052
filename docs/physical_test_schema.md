# Physical test recordings — drop-in schema

Place files in a folder (default `data/manifests/physical_test/`) and score them with:

    python -m vaani.physical --dir data/manifests/physical_test --out results_r2/real/physical_scores.csv

`vaani/physical.py` runs each clip hop by hop through the same per-hop engine the board runs, not the offline torch
path, and writes one row per clip. The engine follows the graph's `model_config.json`:

- **r7 / C0 (legacy 512/256 contract):** `vaani.live.StreamEngine`. The default graph is `deploy/r7/cascade.onnx`
  with `deploy/r7/model_config.json`.
- **VAANI-LD (any low-delay contract, e.g. Arm B `vaanife_ld_asym512_h128_s160_v1`):** `vaani.low_delay_live.
  LowDelayStreamEngine`, with the low-delay front end. A `pr_nhat` graph runs the decoupled-cadence NLMS here; the
  native C++ runtime has no NLMS stage yet. The output is aligned to the input (the engine handles the release lead
  and the flush hop). Pass the export with `--onnx` / `--config`, e.g. the trained Arm B Mini (`inputs: pr`):

      python -m vaani.physical --dir data/manifests/physical_test --out results_r2/real/physical_scores_armb.csv \
          --onnx r8_runs_final/r8_ld_fe_mini_armb/export/ld_fe_mini_armb.onnx \
          --config r8_runs_final/r8_ld_fe_mini_armb/export/model_config.json

  `r8_runs_final/` is local only (gitignored). TBD: no `pr_nhat` export exists until `ld_b_nhat` is trained.

Every row gets DNSMOS P.835 SIG/BAK/OVRL of input and output and a reference-free attenuation proxy (fraction of
active 20 ms input frames the output cut by more than 20 dB). Rows with `has_clean=1` also get SNR, STOI and PESQ
(wideband) against the clean primary, input and output. `--save-wav <dir>` keeps the enhanced audio; `--threads`
sets ORT's thread count (default 1, as on the board). Clips at 44.1 or 48 kHz are resampled to 16 kHz; 16 kHz is
still the expected format.

- `clips/<id>.wav` — 2-channel, 16 kHz, PCM16. Channel 0 = primary (near-mouth), channel 1 = reference.
- `clips/<id>.clean.wav` — optional; if a clean primary exists (lab replay), same length.
- `physical.csv` with columns: `id, speaker, language, condition, transcript, has_clean` (`has_clean` is 0/1 or true/false; ids unique)
  - `condition` ∈ {engine, engine+burst, env_change, ref_fault, quiet}

No physical recording has been scored yet. End-to-end acoustic delay is a separate measurement:
[`docs/acoustic_latency.md`](acoustic_latency.md).
