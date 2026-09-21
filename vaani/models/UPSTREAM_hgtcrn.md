H-GTCRN vendored from https://github.com/Max1Wz/H-GTCRN @ 36dde1b on 2026-09-20 (MIT, LICENSE.hgtcrn).
Paper: Wang, Rong, Sun, Sun, Lin, Lu, "A Lightweight Hybrid Dual Channel Speech Enhancement System under
Low-SNR Conditions", Interspeech 2025 (arXiv 2505.19597). Two-mic 16 kHz input, mono output; FD-WPE + AuxIVA (20 it)
front end feeding a dual-channel GTCRN complex mask. 24,389 trainable + 24,576 fixed ERB weights.

Files pulled verbatim (only the module path changed):
- hgtcrn/masking_on_noisy.py <- masking_on_noisy/gtcrn_iva.py
- hgtcrn/masking_on_iva.py   <- masking_on_iva/gtcrn_iva.py
- checkpoints/hgtcrn_masking_on_noisy.tar <- masking_on_noisy/best_model_0121.tar  sha256 1c982bf805d99176c58bc9824287fdd8922f4c16eec358336d3edfaad04b4219
- checkpoints/hgtcrn_masking_on_iva.tar   <- masking_on_iva/best_model_0110.tar    sha256 a4627c2be5bdb42663f05614b3be23853d532857f98de8a3ed1c2d8f82c58b95

Upstream README (2026-09-18) warns the paper's Masking 1/2 labels are swapped vs its figure; the folder names are
authoritative. The released code uses a plain Hann window where the paper says sqrt-Hann; kept as released.
Registered as `h_gtcrn` (masking on noisy) and `h_gtcrn_iva`. On our 12 cm rig mixtures the IVA variant collapses to
near-silence on the first clip tried (rms 0.004 vs 0.048 for the noisy variant); the authors document that variant as
sensitive to IVA permutation/leakage. `h_gtcrn` is the reference row.

The `h_gtcrn_iva` CSV is retained as a diagnostic of **our unreproduced integration**.
We have not reproduced the authors' configuration/results and do not present this
row as their result or use it to claim superiority over their method. Nominal
SI-SDR -19.349 dB and STOI 0.696 warrant an integration investigation; they do not
by themselves identify a scale, channel-order or permutation bug. The generated
matrix labels the variant explicitly in every table.
