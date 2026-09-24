# Raspberry Pi setup

Inference-only runtime for the shipping r7 cascade (`deploy/r7/cascade.onnx`) on a Raspberry Pi
(aarch64). The board runs numpy + ONNX Runtime + numba; no torch, no CUDA. The streaming contract is
`deploy/CONTRACT.md`.

**Status: unverified on board.** Every step below was written against `uv.lock` and smoke-checked on the
x86_64 dev laptop only. No install, timing run or live capture on a Pi has been recorded. TBD: run
`bash scripts/pi_setup.sh --timing` on the target Pi and commit `deploy/board_timing.json`.

## One command

From the repo checkout on the Pi (it must include `deploy/r7/`):

```bash
bash scripts/pi_setup.sh            # venv + pinned wheels + torch-free import check + graph hash check
bash scripts/pi_setup.sh --timing   # the same, then scripts/board_timing.py -> deploy/board_timing.json
```

What it does, in order:

1. Installs `alsa-utils` with `sudo apt-get` if `arecord`/`aplay` are missing (`scripts/capture_loop.py`
   does its audio I/O through them).
2. Creates `.venv-board` (override with `VENV=...`). Raspberry Pi OS Bookworm and later mark the system
   Python externally managed (PEP 668), so nothing is ever pip-installed into it.
   - System `python3` is 3.12: plain `python3 -m venv`.
   - Otherwise, with `uv` on PATH: `uv venv --python 3.12` (uv fetches a managed 3.12).
   - Otherwise it stops; `ALLOW_OTHER_PY=1` tries the system Python, where the pinned wheels may not exist.
3. `pip install --only-binary=:all: -r requirements-deploy.txt`. Binary-only because a source build of
   numpy or llvmlite on a Pi takes hours; a missing wheel fails immediately instead.
4. Imports `vaani.live` and `vaani.dsp.{nlms,pipeline,stft}` with torch blocked, and prints the versions and
   ONNX Runtime providers.
5. Checks the sha256 prefix of `deploy/r7/cascade.onnx` (`e67a2c42`; the full hash is pinned in
   `tests/test_r7_artifact.py`).

## Pinned versions

From `uv.lock`; `uv.lock` lists cp312 manylinux aarch64 wheels for each of the four native packages.

| package | version | aarch64 cp312 wheel in uv.lock | on board |
|---|---|---|---|
| numpy | 2.5.3 | yes (manylinux_2_27/2_28) | unverified |
| onnxruntime | 1.30.0 | yes (manylinux_2_28) | unverified |
| numba | 0.67.0 | yes (manylinux_2_27/2_28) | unverified |
| llvmlite | 0.49.0 | yes | unverified |
| flatbuffers, packaging, protobuf | 25.12.19, 26.3, 7.36.2 | pure Python / per lock | unverified |

manylinux_2_28 needs glibc 2.28 or newer; Bookworm ships 2.36 (inferred from the Debian release, not
checked on a board).

## Running

```bash
.venv-board/bin/python scripts/board_timing.py deploy/r7/cascade.onnx --seconds 30 --out deploy/board_timing.json
.venv-board/bin/python scripts/capture_loop.py --device hw:0,0 --out-device plughw:1,0
.venv-board/bin/python scripts/capture_loop.py --in-wav in.wav --out-wav out.wav   # file mode, no audio hardware
```

`capture_loop.py` refuses live mode without numba: the pure-Python NLMS fallback does not fit the 16 ms
hop. The first call JIT-compiles the NLMS kernel and caches it on disk, so the first start is slow.

## Open questions

- TBD: which Pi model and OS image is the target (Pi 4 or Pi 5; Bookworm or Trixie), and does its
  `python3` match 3.12?
- TBD: measured per-hop latency and real-time factor on that board (`deploy/board_timing.json`).
- TBD: the I2S microphone overlay and ALSA card names on the assembled headset; the `hw:0,0` /
  `plughw:1,0` defaults come from `scripts/capture_loop.py`, not from a board.
