# Physical test recordings — drop-in schema

Place files in a folder (default `data/manifests/physical_test/`) and score them with:

    python -m vaani.physical --dir data/manifests/physical_test --out results_r2/real/physical_scores.csv

`vaani/physical.py` runs each clip hop by hop through `vaani.live.StreamEngine` with `deploy/r7/cascade.onnx` and
`deploy/r7/model_config.json` (the board path; override with `--onnx` / `--config`), and writes one row per clip.
Every row gets DNSMOS P.835 SIG/BAK/OVRL of input and output and a reference-free attenuation proxy (fraction of
active 20 ms input frames the output cut by more than 20 dB). Rows with `has_clean=1` also get SNR, STOI and PESQ
(wideband) against the clean primary, input and output. `--save-wav <dir>` keeps the enhanced audio.
Clips at 44.1 or 48 kHz are resampled to 16 kHz; 16 kHz is still the expected format.

- `clips/<id>.wav` — 2-channel, 16 kHz, PCM16. Channel 0 = primary (near-mouth), channel 1 = reference.
- `clips/<id>.clean.wav` — optional; if a clean primary exists (lab replay), same length.
- `physical.csv` with columns: `id, speaker, language, condition, transcript, has_clean` (`has_clean` is 0/1 or true/false; ids unique)
  - `condition` ∈ {engine, engine+burst, env_change, ref_fault, quiet}
