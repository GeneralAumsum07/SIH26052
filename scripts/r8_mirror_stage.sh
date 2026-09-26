#!/usr/bin/env bash
# LAPTOP ONLY. Stage the artefacts the r8 box needs but cannot rebuild bit-identically, with a sha256 manifest, and
# print the commands Rachit runs to put them in a PRIVATE Hugging Face dataset repo. Uploads nothing itself.
#   bash scripts/r8_mirror_stage.sh [stage_dir]          (default data/mirror_stage; data/ is git-ignored)
# Staged (role -> where r8_box_setup.sh puts it):
#   eval_r2_val.tar                      install -> data/eval_r2/val   frozen val set: composite selection, G3; verified
#                                        by scripts/verify_eval_set.py against b5f7a4d43bee before staging and on the box
#   manifests/mad_v2.parquet             install -> data/manifests/    VAD-filtered MAD (Silero run not pinned: rebuild
#   manifests/mad_speech_contamination.parquet                         on the box could differ, so the laptop list travels)
#   manifests_laptop/<m>.parquet         reference -> data/mirror/manifests_laptop/  the laptop scans of every other
#                                        manifest the r8 configs and G1 read; the box scans its own, preflight diffs
#                                        source_ids against these (paths are repo-relative, backslashes normalised by
#                                        vaani.data.manifests.read, so they resolve on Linux once the audio is scanned)
#   field/abcd.wav                       install -> data/field/     the stereo web WAV (WEB_WAV, default
#                                        ~/Downloads/abcd.wav): G4 Part 1's web bed and all of Part 2 (--web-wav)
# Not staged: data/eval_r8_test (G6 is scored once on the laptop after the chosen checkpoint is pulled back; keeping it
# off the box removes any chance of it touching selection), RIR banks (GitHub release, RIR_BANK_URL, by sha256),
# bank_eval_r8 (only the r8 test render reads it), r7 best.pt and the gtcrn baseline (tracked in git), raw audio
# (downloaded on the box). Idempotent: an existing staged file with the recorded sha256 is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
S="${1:-data/mirror_stage}"; REPO_ID="${MIRROR_HF_REPO:-<hf-user>/vaani-r8-mirror}"
VAL="${VAL:-data/eval_r2/val}"; VAL_HASH="${VAL_HASH:-b5f7a4d43bee}"   # overridable for the staging test only
PY="${PY:-}"; [ -n "$PY" ] || for c in .venv/Scripts/python.exe .venv/bin/python python3 python; do
  if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi; done
WEB_WAV="${WEB_WAV:-$HOME/Downloads/abcd.wav}"
mkdir -p "$S/manifests" "$S/manifests_laptop"

# every manifest an r8 config or the G1 gate reads (same lists the box scans)
MANS_PY=$(cat <<'EOF'
import glob, sys, yaml
sys.path.insert(0, "scripts")
ms = set()
for c in ["configs/retraining/r8_fe_mini.yaml", "configs/retraining/r8_refvalid_v2.yaml", *glob.glob("configs/retraining/r8_ablations/*.yaml")]:
    try: ms.update(p.replace("\\", "/").split("/")[-1] for p in yaml.safe_load(open(c))["data"]["manifests"])
    except FileNotFoundError: pass
try:
    import data_gates as g; ms.update(g.SPEECH_MANIFESTS + g.V2_NOISE)
except Exception as e: print(f"# data_gates lists unavailable: {e}", file=sys.stderr)
print("\n".join(sorted(ms)))
EOF
)
# -c, not a heredoc on stdin: Git Bash hands a heredoc inside <(...) to a Windows python as an empty script
mapfile -t MANS < <("$PY" -c "$MANS_PY" | tr -d '\r')

"$PY" scripts/verify_eval_set.py "$VAL" "$VAL_HASH" || { echo "val set does not verify; not staging it"; exit 1; }
[ -f "$S/eval_r2_val.tar" ] && [ -f "$S/.eval_r2_val.ok" ] || {
  tar -C data -cf "$S/eval_r2_val.tar" eval_r2/val && touch "$S/.eval_r2_val.ok"; }
for f in mad_v2.parquet mad_speech_contamination.parquet; do cp -p "data/manifests/$f" "$S/manifests/$f"; done
for m in "${MANS[@]}"; do
  case "$m" in mad_v2.parquet) continue;; esac
  if [ -f "data/manifests/$m" ]; then cp -p "data/manifests/$m" "$S/manifests_laptop/$m"; else echo "WARN: no laptop data/manifests/$m"; fi
done
if [ -f "$WEB_WAV" ]; then mkdir -p "$S/field"; cp -p "$WEB_WAV" "$S/field/abcd.wav"
else echo "WARN: no web WAV at $WEB_WAV (G4 --web-wav needs it on the box; set WEB_WAV)"; fi

# sha256 manifest (sha256sum -c format, paths relative to the stage root) + sizes
( cd "$S" && find . -type f ! -name 'SHA256SUMS' ! -name 'MANIFEST.tsv' ! -name '.*' | sed 's|^\./||' | sort \
    | while read -r f; do h=$(sha256sum "$f"); printf '%s  %s\n' "${h%% *}" "$f"; done > SHA256SUMS )   # Git Bash writes "*f"
( cd "$S" && while read -r h f; do printf '%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$h"; done < SHA256SUMS > MANIFEST.tsv )
tot=$(awk -F'\t' '{s += $2} END {printf "%.2f", s / 1e9}' "$S/MANIFEST.tsv")
echo "staged $(wc -l < "$S/SHA256SUMS") files, $tot GB in $S:"; column -t -s $'\t' "$S/MANIFEST.tsv" 2>/dev/null || cat "$S/MANIFEST.tsv"
cat <<EOF

Upload (Rachit; licensed audio, so the repo MUST be private - create it private first, since 'hf upload' creates a
missing repo with default visibility):
  hf auth login                                   # or: export HF_TOKEN=<a write token>
  hf repos create $REPO_ID --repo-type dataset --private --exist-ok
  hf upload $REPO_ID $S . --repo-type dataset
Then on the box: export MIRROR_HF_REPO=$REPO_ID HF_TOKEN=<a read token>; scripts/r8_box_setup.sh fetches and checks
SHA256SUMS before installing anything.
EOF
