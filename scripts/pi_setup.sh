#!/usr/bin/env bash
# Set up the inference runtime on a Raspberry Pi (aarch64, Raspberry Pi OS / Debian). Run from anywhere:
#   bash scripts/pi_setup.sh            # venv + requirements-deploy.txt + import and hash checks
#   bash scripts/pi_setup.sh --timing   # also runs scripts/board_timing.py on the shipping graph
# Idempotent: an existing .venv-board is reused. Unverified on board: no Pi run has been recorded yet.
# See deploy/PI_SETUP.md.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv-board}
PY_WANT=3.12   # uv.lock's aarch64 wheels (numpy, onnxruntime, numba, llvmlite) are cp312
ONNX=deploy/r7/cascade.onnx
ONNX_SHA=e67a2c42   # prefix of the shipping r7 graph's sha256 (tests/test_r7_artifact.py pins the full hash)

[ "$(uname -m)" = aarch64 ] || echo "warning: $(uname -m), not aarch64; the pins target the Pi but should install here too"

# --- system packages: arecord/aplay are capture_loop.py's audio I/O, not pip packages --------------------
if ! command -v arecord >/dev/null || ! command -v aplay >/dev/null; then
  echo "alsa-utils missing; installing (needs sudo)"
  sudo apt-get update -qq && sudo apt-get install -y -qq alsa-utils
fi

# --- venv: Bookworm and later mark the system Python externally managed (PEP 668), so never pip into it ----
if [ ! -x "$VENV/bin/python" ]; then
  sys_py=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo none)
  if [ "$sys_py" = "$PY_WANT" ]; then
    python3 -m venv "$VENV"
  elif command -v uv >/dev/null; then
    # uv fetches a managed 3.12 so the pinned cp312 wheels apply regardless of the OS Python
    uv venv --python "$PY_WANT" --seed "$VENV"
  else
    echo "system Python is $sys_py, the pins are tested on $PY_WANT."
    echo "install uv (https://docs.astral.sh/uv/) and rerun, or rerun with ALLOW_OTHER_PY=1 to try the system Python (the pinned wheels may not exist for it)."
    [ "${ALLOW_OTHER_PY:-0}" = 1 ] || exit 1
    python3 -m venv "$VENV"
  fi
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
# --only-binary: a source build of numpy/llvmlite on a Pi takes hours; fail fast instead
"$VENV/bin/python" -m pip install --quiet --only-binary=:all: -r requirements-deploy.txt

# --- checks: the board must import the runtime without torch and see the shipping graph --------------------
"$VENV/bin/python" - <<'PY'
import sys
sys.modules["torch"] = None   # the board has no torch; prove the runtime path does not need it
sys.path.insert(0, ".")
import numpy, onnxruntime, numba
from vaani import live           # noqa: F401
from vaani.dsp import nlms, pipeline, stft   # noqa: F401
print(f"python {sys.version.split()[0]}  numpy {numpy.__version__}  onnxruntime {onnxruntime.__version__}  "
      f"numba {numba.__version__}  providers {onnxruntime.get_available_providers()}")
PY
if [ -f "$ONNX" ]; then
  sha=$(sha256sum "$ONNX" | cut -c1-8)
  [ "$sha" = "$ONNX_SHA" ] && echo "ok: $ONNX sha256 $sha..." || { echo "MISMATCH: $ONNX sha256 $sha..., want $ONNX_SHA..."; exit 1; }
else
  echo "missing $ONNX (copy the repo checkout including deploy/r7/)"; exit 1
fi

if [ "${1:-}" = --timing ]; then
  "$VENV/bin/python" scripts/board_timing.py "$ONNX" --seconds 30 --out deploy/board_timing.json
fi
echo "ready: $VENV/bin/python scripts/capture_loop.py --device hw:0,0 --out-device plughw:1,0"
