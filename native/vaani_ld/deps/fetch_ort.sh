#!/usr/bin/env bash
# Fetch the pinned ONNX Runtime C/C++ release into native/vaani_ld/_deps/ and verify its SHA-256.
#   native/vaani_ld/deps/fetch_ort.sh [x64|aarch64]      (default: this machine's architecture)
# 1.30.0 matches requirements-deploy.txt (the board's Python onnxruntime), so the native and Python paths run the
# same inference library. ONNX Runtime is MIT-licensed (LICENSE inside the archive).
set -euo pipefail
VERSION=1.30.0
declare -A SHA=(
  [x64]=a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd
  [aarch64]=e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da
)
ARCH=${1:-$(uname -m)}
case "$ARCH" in x86_64|x64) ARCH=x64 ;; aarch64|arm64) ARCH=aarch64 ;; *) echo "unsupported arch $ARCH" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST="$HERE/_deps"
NAME=onnxruntime-linux-$ARCH-$VERSION
mkdir -p "$DEST"
if [ ! -d "$DEST/$NAME" ]; then
  TGZ="$DEST/$NAME.tgz"
  curl -fsSL -o "$TGZ" "https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$NAME.tgz"
  echo "${SHA[$ARCH]}  $TGZ" | sha256sum -c - >&2
  tar xzf "$TGZ" -C "$DEST"
  rm "$TGZ"
fi
echo "$DEST/$NAME"
