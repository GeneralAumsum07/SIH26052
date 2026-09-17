# Physical test recordings — drop-in schema

Place files here; `vaani/eval.py --split physical` consumes them.

- `clips/<id>.wav` — 2-channel, 16 kHz, PCM16. Channel 0 = primary (near-mouth), channel 1 = reference.
- `clips/<id>.clean.wav` — optional; if a clean primary exists (lab replay), same length.
- `physical.csv` with columns: `id, speaker, language, condition, transcript, has_clean`
  - `condition` ∈ {engine, engine+burst, env_change, ref_fault, quiet}
