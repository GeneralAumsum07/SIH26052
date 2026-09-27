#!/usr/bin/env bash
# Linux/ARM build of the native runtime (low-delay plan Task 7), cross-compiled on an x86 Ubuntu/Debian host, then the
# golden and unit tests run under qemu-user. Writes the build report JSON to stdout (or to $1).
# Host packages: g++-aarch64-linux-gnu qemu-user libc6:arm64 libstdc++6:arm64 (dpkg --add-architecture arm64 with an
# arm64 package source). libasound for arm64 is unpacked into a private sysroot, so the host's ALSA is untouched.
# qemu checks numerics and the build, not timing: the board steps (step_bench, period_test, live) stay with the owner.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=${VLD_CROSS_WORK:-$HERE/build-arm64-work}
BUILD=$HERE/build-arm64
OUT=${1:-/dev/stdout}
command -v aarch64-linux-gnu-g++ >/dev/null || { echo "aarch64-linux-gnu-g++ missing (g++-aarch64-linux-gnu)" >&2; exit 2; }
ORT=$("$HERE/deps/fetch_ort.sh" aarch64)
SYSROOT=$WORK/sysroot
if [ ! -f "$SYSROOT/usr/include/alsa/asoundlib.h" ]; then
  mkdir -p "$WORK/deb" "$SYSROOT"
  (cd "$WORK/deb" && apt-get download libasound2-dev:arm64 libasound2t64:arm64)
  for d in "$WORK"/deb/*.deb; do dpkg -x "$d" "$SYSROOT"; done
fi
cmake -S "$HERE" -B "$BUILD" -DCMAKE_TOOLCHAIN_FILE="$HERE/cmake/aarch64-linux-gnu.cmake" -DCMAKE_BUILD_TYPE=Release \
      -DVLD_ALSA_ROOT="$SYSROOT" >&2
cmake --build "$BUILD" -j"$(nproc)" >&2
VEC=$ROOT/deploy/dsp_reference/vectors_ld
RJSON=$ROOT/deploy/resampler
NPY=$WORK/npy
python3 - "$VEC" "$NPY" <<'EOF'
import sys, numpy as np
from pathlib import Path
vec, dst = Path(sys.argv[1]), Path(sys.argv[2])
for f in vec.rglob("*.npz"):
    d = dst / f.relative_to(vec).with_suffix(""); d.mkdir(parents=True, exist_ok=True)
    z = np.load(f)
    for k in z.files: np.save(d / f"{k}.npy", z[k])
EOF
TOL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tolerance'])" "$VEC/manifest.json")
QEMU=""
if [ "$(uname -m)" != aarch64 ]; then
  command -v qemu-aarch64 >/dev/null || { echo "qemu-aarch64 missing (qemu-user)" >&2; exit 2; }
  QEMU=qemu-aarch64
  export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
fi
export LD_LIBRARY_PATH=$ORT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
set +e
GOLDEN=$($QEMU "$BUILD/vld_golden_test" "$NPY" "$VEC" "$RJSON" "$TOL" | tail -1); G=$?
UNIT=$($QEMU "$BUILD/vld_unit_test" "$VEC" "$RJSON" | tail -1); U=$?
set -e
python3 - "$BUILD" "$GOLDEN" "$G" "$UNIT" "$U" "$QEMU" "$ORT" > "$OUT" <<'EOF'
import json, subprocess, sys, hashlib
from pathlib import Path
build, golden, g, unit, u, qemu, ort = sys.argv[1:]
exes = ["vaani_ld_run", "vld_step_bench", "vld_period_test", "vld_golden_test", "vld_unit_test"]
def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout.strip()
rep = {
    "target": "aarch64-linux-gnu",
    "compiler": sh("aarch64-linux-gnu-g++", "--version").splitlines()[0],
    "onnxruntime": Path(ort, "VERSION_NUMBER").read_text().strip(),
    "alsa": True,
    "executables": {e: {"file": sh("file", "-b", str(Path(build, e))).split(",")[1].strip(),
                        "sha256": hashlib.sha256(Path(build, e).read_bytes()).hexdigest()} for e in exes},
    "runner": qemu or "native",
    "golden": json.loads(golden) if golden.startswith("{") else golden, "golden_exit": int(g),
    "unit": json.loads(unit) if unit.startswith("{") else unit, "unit_exit": int(u),
    "note": "qemu-user checks the ARM build and its numerics only; board timing is Gate 0a/Gate D (owner).",
}
rep["pass"] = int(g) == 0 and int(u) == 0
print(json.dumps(rep, indent=2))
EOF
[ "$G" = 0 ] && [ "$U" = 0 ]
