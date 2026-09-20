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
[ -d .venv ] || uv sync --quiet --python 3.12      # torch cu128 wheels from pyproject; driver 595 / CUDA 13.2 runs them
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
echo "6724e9085801fa4c7865f5ee312a0886  $D/gunshots/edge-collected-gunshot-audio.zip" | md5sum -c -
# fetch_data.download() re-fetches anything without a .ok marker, so mark every archive aria2c finished
for f in $D/train-clean-100.tar.gz $D/esc50/master.zip $D/gunshots/edge-collected-gunshot-audio.zip $D/drone/master.zip $D/ears/p*.zip $D/dns/*.tar.bz2; do touch "$f.ok"; done

# --- gated inputs come from the laptop (rsync, slow uplink); wait for the marker rsync drops last -----------
until [ -f $D/GATED_OK ]; do echo "$(date +%H:%M) waiting for the laptop upload (data/download/GATED_OK)"; sleep 60; done

# --- corpus -> 16 kHz flac + manifests (scan is CPU-bound; 32 cores make this minutes, not the laptop's hour)
[ -f data/manifests/MANIFESTS_OK ] || { uv run python scripts/fetch_data.py --dns-shards \
  $(for s in freesound_001 audioset_000 audioset_001 audioset_002 audioset_003 audioset_004 audioset_005 audioset_006; do
      echo $B/datasets_fullband.noise_fullband.$s.tar.bz2; done) && touch data/manifests/MANIFESTS_OK; }
# relabel pass (MAD shooting/shelling/footsteps -> changing, ESC-50 impulsive shortlist) is inside the scanners now;
# run it anyway so a manifest restored from the laptop matches
uv run python scripts/relabel_noise_class.py

# --- RIR banks: r1/r2 bank (eval render + r1/r2 configs) and the r3 armoured bank -----------------------------
[ -f data/rirs/bank.npz ]    || uv run python scripts/make_rir_bank.py --out data/rirs/bank.npz
[ -f data/rirs/bank_r3.npz ] || uv run python scripts/make_rir_bank.py --out data/rirs/bank_r3.npz --armoured-frac 0.2 --max-len-s 1.0

# --- frozen eval set: use the laptop copy if it arrived, else re-render and compare the hash -----------------
if [ ! -f data/eval_r2/test/EVALSET_HASH ]; then
  M="data/manifests/librispeech.parquet data/manifests/esc50.parquet data/manifests/cv_hi.parquet data/manifests/mad.parquet data/manifests/dns_datasets_fullband.noise_fullband.freesound_000.tar.parquet"
  uv run python scripts/render_eval_sets.py --manifests $M --split val  --out data/eval_r2
  uv run python scripts/render_eval_sets.py --manifests $M --split test --out data/eval_r2 --faults
fi
echo "eval_r2 test hash: $(cat data/eval_r2/test/EVALSET_HASH)  (laptop: eda217ab2a38)"

# --- round 3 --------------------------------------------------------------------------------------------------
bash scripts/run_round.sh 3
