#!/usr/bin/env bash
# Bootstrap a rented Linux GPU box (Vast.ai, 2026-09-20: 1x RTX 5090, 32 vCPU, 15 Gbps down) for round 3.
# Run as root inside tmux from the repo checkout: `bash scripts/remote_setup.sh`. Idempotent: every stage is
# skipped when its marker exists, so a dropped SSH session costs nothing. Public archives are pulled here in
# parallel; the gated ones (cv_hi.tar.gz, mad.zip, noisex92/) and the frozen eval set are scp'd from the laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/download; mkdir -p $D/{dns,ears,gunshots,drone} data/manifests data/rirs runs results_r2/asr

# --- tooling ------------------------------------------------------------------------------------------------
command -v aria2c >/dev/null || { apt-get update -qq && apt-get install -y -qq aria2 tmux rsync libsndfile1 build-essential > /dev/null; }  # build-essential: pesq builds from sdist on Linux
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d .venv ] || uv sync --quiet --all-extras --python 3.12   # torch cu128 wheels from pyproject; --all-extras: without numba the NLMS runs in pure Python and the GPU idles (2026-09-21)
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# --- public archives, all at once (the box has the pipe; the laptop did not) --------------------------------
B=https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/noise_fullband
EARS=https://github.com/facebookresearch/ears_dataset/releases/download/dataset
{
  echo "https://www.openslr.org/resources/12/train-clean-100.tar.gz"; echo "  dir=$D"; echo "  out=train-clean-100.tar.gz"
  echo "https://github.com/karolpiczak/ESC-50/archive/master.zip"; echo "  dir=$D/esc50"; echo "  out=master.zip"
  echo "https://zenodo.org/api/records/7004819/files/edge-collected-gunshot-audio.zip/content"; echo "  dir=$D/gunshots"; echo "  out=edge-collected-gunshot-audio.zip"
  echo "https://github.com/saraalemadi/DroneAudioDataset/archive/master.zip"; echo "  dir=$D/drone"; echo "  out=master.zip"
  for i in $(seq -f %03g 1 20); do echo "$EARS/p$i.zip"; echo "  dir=$D/ears"; echo "  out=p$i.zip"; done
  # freesound_002/003 do not exist on the blob (404, 2026-09-20)
  for s in freesound_000 freesound_001 audioset_000 audioset_001 audioset_002 audioset_003 audioset_004 audioset_005 audioset_006; do
    echo "$B/datasets_fullband.noise_fullband.$s.tar.bz2"; echo "  dir=$D/dns"; echo "  out=datasets_fullband.noise_fullband.$s.tar.bz2"; done
} > $D/public.aria2
[ -f $D/PUBLIC_OK ] || { aria2c -c -j8 -x8 -s8 -k 10M --max-tries=0 --retry-wait=10 --file-allocation=none \
  --console-log-level=warn --summary-interval=30 -i $D/public.aria2 && touch $D/PUBLIC_OK; }
# gunshot zip: aria2c writes it cleanly here, but keep the md5 gate the laptop needed (Zenodo metadata)
[ -f $D/CLEANED ] || echo "6724e9085801fa4c7865f5ee312a0886  $D/gunshots/edge-collected-gunshot-audio.zip" | md5sum -c -
# fetch_data.download() re-fetches anything without a .ok marker, so mark every archive aria2c finished
[ -f $D/CLEANED ] || for f in $D/train-clean-100.tar.gz $D/esc50/master.zip $D/gunshots/edge-collected-gunshot-audio.zip $D/drone/master.zip $D/ears/p*.zip $D/dns/*.tar.bz2; do touch "$f.ok"; done

# --- gated inputs come from the laptop (rsync, slow uplink); wait for the marker rsync drops last -----------
until [ -f $D/GATED_OK ]; do echo "$(date +%H:%M) waiting for the laptop upload (data/download/GATED_OK)"; sleep 60; done

# --- corpus -> 16 kHz flac + manifests (scan is CPU-bound; 32 cores make this minutes, not the laptop's hour)
[ -f data/manifests/MANIFESTS_OK ] || { uv run python scripts/fetch_data.py --dns-shards \
  $(for s in freesound_000 freesound_001 audioset_000 audioset_001 audioset_002 audioset_003 audioset_004 audioset_005 audioset_006; do
      echo $B/datasets_fullband.noise_fullband.$s.tar.bz2; done) && touch data/manifests/MANIFESTS_OK; }
# relabel pass (MAD shooting/shelling/footsteps -> changing, ESC-50 impulsive shortlist) is inside the scanners now;
# run it anyway so a manifest restored from the laptop matches
uv run python scripts/relabel_noise_class.py data/manifests/*.parquet   # positional manifests are required; bare call aborted the bootstrap
# r3+ recipes train on the full train-clean-100; fetch_data only writes the 20 h librispeech.parquet (missed on two boxes)
[ -f data/manifests/librispeech_100h.parquet ] || uv run python scripts/make_librispeech_100h.py

# --- RIR banks: r1/r2 bank (eval render + r1/r2 configs) and the r3 armoured bank -----------------------------
# A regenerated bank is never the same file: pyroomacoustics differs across machines even for the non-armoured bank
# (laptop dafb2e84 vs box 491b85f2, 2026-09-21) and ray tracing is nondeterministic. Fetch the published copies by hash
# (RIR_BANK_URL: a GitHub release assets base, e.g. https://github.com/<owner>/<repo>/releases/download/rir-banks-2026-09-21)
# and fall back to generation only when no release is configured or the download fails.
fetch_bank() {  # $1 file, $2 sha256
  [ -n "${RIR_BANK_URL:-}" ] || return 1
  aria2c -c -x8 -s8 -k 10M --max-tries=5 --console-log-level=warn -d data/rirs -o "$1" "$RIR_BANK_URL/$1" || return 1
  echo "$2  data/rirs/$1" | sha256sum -c - || { rm -f "data/rirs/$1"; return 1; }
}
# workers capped: a 64-process pool hit the host's pid cgroup (pids.max 7680, threads count) even with the thread caps build_bank sets
W=$(( $(nproc) < 32 ? $(nproc) : 32 ))
[ -f data/rirs/bank.npz ]    || fetch_bank bank.npz    dafb2e84e34099ff4a92db4e0941fbceb6aa2ef85c5408c8f3f336c652e1f988 \
  || uv run python scripts/make_rir_bank.py --out data/rirs/bank.npz --workers $W
[ -f data/rirs/bank_r3.npz ] || fetch_bank bank_r3.npz e4e67463072e1dca14b94bef97a99bb85da34ebee7ec657a1f1dc7554df1b59f \
  || uv run python scripts/make_rir_bank.py --out data/rirs/bank_r3.npz --armoured-frac 0.2 --max-len-s 1.0 --workers $W
sha256sum data/rirs/bank.npz data/rirs/bank_r3.npz   # in the log for the run record; a regenerated bank shows up here as a new hash

# --- frozen eval set: use the laptop copy if it arrived, else re-render and compare the hash -----------------
if [ ! -f data/eval_r2/test/EVALSET_HASH ]; then
  M="data/manifests/librispeech.parquet data/manifests/esc50.parquet data/manifests/cv_hi.parquet data/manifests/mad.parquet data/manifests/dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet"
  uv run python scripts/render_eval_sets.py --manifests $M --split val  --out data/eval_r2
  uv run python scripts/render_eval_sets.py --manifests $M --split test --out data/eval_r2 --faults
fi
echo "eval_r2 test hash: $(cat data/eval_r2/test/EVALSET_HASH)  (laptop: eda217ab2a38)"

# --- extracted archives: the manifests point at data/raw, so drop everything except the stage markers ----------------
if [ ! -f $D/CLEANED ]; then
  uv run python scripts/check_manifests_off_download.py
  find $D -mindepth 1 -maxdepth 1 ! -name '*_OK' ! -name public.aria2 -exec rm -rf {} +
  touch $D/CLEANED; df -h . | tail -1
fi

# --- training: `remote_setup.sh 3` (default) runs the round-3 matrix; `remote_setup.sh tier46 [config]` runs the refiner
if [ "${1:-3}" = tier46 ]; then bash scripts/run_tier46.sh "${2:-configs/exp/vaani_tier46_refiner.yaml}"; else bash scripts/run_round.sh "${1:-3}"; fi
