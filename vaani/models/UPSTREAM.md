Vendored from https://github.com/Xiaobin-Rong/gtcrn @ main on 2026-09-18.

Files pulled verbatim:
- gtcrn.py <- gtcrn.py
- gtcrn_stream.py <- stream/gtcrn_stream.py
- modules/convolution.py <- stream/modules/convolution.py
- modules/convert.py <- stream/modules/convert.py
- LICENSE.gtcrn <- LICENSE (MIT)
- checkpoints/model_trained_on_dns3.tar <- checkpoints/model_trained_on_dns3.tar

Checkpoint provenance: upstream .tar, trained on DNS3.
sha256: a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79

Edits (only these, everything else byte-identical to upstream):
- gtcrn_stream.py: `from modules.convolution import ...` -> `from vaani.models.modules.convolution import ...`
- gtcrn_stream.py: `from modules.convert import ...` (inside the `if __name__ == "__main__"` demo block) -> `from vaani.models.modules.convert import ...`
- gtcrn_stream.py: appended `init_caches(device="cpu")` at end of file (not upstream - needed by our streaming-parity test; shapes taken from the zero-cache init in upstream's own `__main__` demo block).

Checkpoint loading: `torch.load(..., weights_only=True)` works directly on this .tar (its dict is `{epoch, optimizer, model}` and all values are safe tensors/ints under the current torch weights_only allowlist) - no conversion script was needed.

Streaming parity: full-sequence GTCRN vs frame-by-frame StreamGTCRN agrees only to atol=1e-3 on a random 1s clip (not 1e-4) - float accumulation order differs frame-by-frame vs batched. Matches upstream's own loose-tolerance stream demo.

GTCRN param count (measured): 48,245 (matches upstream's ~48.2K figure).

Streaming-parity test loads the batch model's checkpoint directly, then builds
the stream model's weights via upstream's `convert_to_stream` (a direct
`load_state_dict` on `StreamGTCRN` fails: its Conv2d/ConvTranspose2d wrapper
modules nest weights one level deeper than plain GTCRN's flat keys).

rnnoise baseline: no working Python rnnoise binding installs cleanly on Windows in a short window, so `baselines.get("rnnoise")` is registered but `.enhance()` raises `NotImplementedError("rnnoise binding unavailable on this platform")` unless a `rnnoise_demo` binary happens to be on PATH.
