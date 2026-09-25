#!/usr/bin/env bash
# Regenerate requirements.txt from uv.lock. Do not hand-edit requirements.txt - edit pyproject.toml,
# run `uv lock`, then run this.
#
# The repo's source of truth is pyproject.toml + uv.lock, which pins every transitive dependency with
# hashes for every platform. requirements.txt exists so that anyone evaluating this without uv can
# still reproduce the environment with pip alone.
#
# `uv export` records torch==2.11.0+cu128 but does not emit the index that serves that build, so the
# extra-index-url is prepended here. Without it pip searches PyPI, fails to find a +cu128 wheel, and
# either errors or silently installs the CPU build - which runs, slowly, and makes every GPU result
# look wrong.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=requirements.txt
{
  echo "# Generated from uv.lock by scripts/export_requirements.sh - do not edit by hand."
  echo "#"
  echo "# SCOPE: the x86_64 + CUDA + Python 3.12 TRAINING environment. Not for a target board."
  echo "# The hashes below were resolved for Windows and Linux x86_64 only, so this file cannot install"
  echo "# on aarch64 - pip checks hashes all-or-nothing and an ARM wheel will never match. For a"
  echo "# Raspberry Pi or any inference target use requirements-deploy.txt (numpy + onnxruntime)."
  echo "# Source of truth: pyproject.toml + uv.lock. Regenerate after any dependency change."
  echo "#"
  echo "# torch/torchaudio are CUDA 12.8 builds and are not on PyPI; this index serves them."
  echo "# For a CPU-only install, replace cu128 with cpu below."
  echo "--extra-index-url https://download.pytorch.org/whl/cu128"
  echo ""
  uv export --format requirements-txt --no-emit-project --all-extras
} > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines, $(grep -c -- --hash "$OUT") hashes)"
