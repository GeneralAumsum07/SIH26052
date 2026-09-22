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

# --- RIR banks, started now and joined just before the eval render (3.2: they depend on nothing) ------------
# A regenerated bank is never the same file: pyroomacoustics differs across machines even for the non-armoured
# bank (laptop dafb2e84 vs box 491b85f2, 2026-09-21) and ray tracing is nondeterministic. Fetch the published
# copies by hash (RIR_BANK_URL: a GitHub release assets base) and generate only as a last resort.
fetch_bank() {  # $1 file, $2 sha256
  [ -n "${RIR_BANK_URL:-}" ] || return 1
  aria2c -c -x8 -s8 -k 10M --max-tries=5 --console-log-level=warn -d data/rirs -o "$1" "$RIR_BANK_URL/$1" || return 1
  echo "$2  data/rirs/$1" | sha256sum -c - || { rm -f "data/rirs/$1"; return 1; }
}
# The cap is the host's pid cgroup, not the core count: a 64-process pool hit pids.max (threads count) even with
# the thread caps build_bank sets. Read the real limit instead of assuming the 32 that happened to work once.
pid_budget() {
  local m; m=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo max)
  [ "$m" = max ] && { echo "$(nproc)"; return; }
  local room=$(( (m - $(ls /proc | grep -c '^[0-9]') ) / 96 ))   # ~96 threads per simulation worker, measured
  [ "$room" -lt 4 ] && room=4
  [ "$room" -gt "$(nproc)" ] && room=$(nproc)
  echo "$room"
}
W=$(pid_budget)
echo "rir bank workers: $W (nproc $(nproc), pids.max $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo max))"
{
  [ -f data/rirs/bank.npz ]    || fetch_bank bank.npz    dafb2e84e34099ff4a92db4e0941fbceb6aa2ef85c5408c8f3f336c652e1f988 \
    || uv run python scripts/make_rir_bank.py --out data/rirs/bank.npz --workers $W
  [ -f data/rirs/bank_r3.npz ] || fetch_bank bank_r3.npz e4e67463072e1dca14b94bef97a99bb85da34ebee7ec657a1f1dc7554df1b59f \
    || uv run python scripts/make_rir_bank.py --out data/rirs/bank_r3.npz --armoured-frac 0.2 --max-len-s 1.0 --workers $W
} > data/rirs/bank_fetch.log 2>&1 &
BANK_PID=$!
wait_banks() { wait "$BANK_PID" || { echo "bank stage failed; see data/rirs/bank_fetch.log"; tail -20 data/rirs/bank_fetch.log; exit 1; }; }

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

# --- gated inputs: pull them here at line rate instead of waiting on the laptop's uplink (3.1) --------------
# This stage used to sit idle for ~2.5 h while rsync crawled up a campus connection. MAD and Common Voice Hindi
# have APIs the box can call directly; anything else (and the frozen eval set) comes from MIRROR_URL, published
# once with a sha256 the same way the RIR banks are. The old wait remains as the last resort, so an unconfigured
# box still works - it is just slow.
mirror_get() {  # $1 file, $2 sha256 ("-" to skip the check), $3 dest dir
  [ -n "${MIRROR_URL:-}" ] || return 1
  aria2c -c -x16 -s16 -k 10M --max-tries=3 --console-log-level=warn -d "$3" -o "$1" "$MIRROR_URL/$1" || return 1
  [ "$2" = - ] || echo "$2  $3/$1" | sha256sum -c - || { rm -f "$3/$1"; return 1; }
}
if [ ! -f $D/GATED_OK ]; then
  # MAD: Kaggle API (KAGGLE_USERNAME/KAGGLE_KEY in the environment)
  if [ ! -f $D/mad.zip ] && [ -n "${KAGGLE_KEY:-}" ]; then
    uv run --with kaggle kaggle datasets download -d junewookim/mad-dataset-military-audio-dataset \
      -p $D -o && mv -f $D/mad-dataset-military-audio-dataset.zip $D/mad.zip || true
  fi
  # Common Voice Hindi: Mozilla Data Collective returns a presigned URL for an API key
  if [ ! -f $D/cv_hi.tar.gz ] && [ -n "${MDC_API_KEY:-}" ]; then
    CV_URL=$(curl -sX POST "https://mozilladatacollective.com/api/datasets/${MDC_DATASET:-mcv-hi-v23.0}/download" \
      -H "Authorization: Bearer $MDC_API_KEY" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("url") or d.get("downloadUrl") or "")' 2>/dev/null || true)
    [ -n "$CV_URL" ] && aria2c -c -x16 -s16 -d $D -o cv_hi.tar.gz "$CV_URL" || true
  fi
  mirror_get noisex92.tar.gz - $D || true
  for f in $D/mad.zip $D/cv_hi.tar.gz $D/noisex92.tar.gz; do [ -f "$f" ] && touch "$f.ok"; done
  [ -f $D/mad.zip ] && [ -f $D/cv_hi.tar.gz ] && touch $D/GATED_OK || true
fi
# the frozen eval set must be byte-identical, so it is fetched whole rather than re-rendered when possible
if [ ! -f data/eval_r2/test/EVALSET_HASH ] && [ -n "${MIRROR_URL:-}" ]; then
  mirror_get eval_r2.tar - . && tar xf eval_r2.tar && rm -f eval_r2.tar || true
fi
until [ -f $D/GATED_OK ]; do echo "$(date +%H:%M) waiting for the laptop upload (set MIRROR_URL/KAGGLE_KEY/MDC_API_KEY to skip this)"; sleep 60; done

# --- corpus -> 16 kHz flac + manifests (scan is CPU-bound; 32 cores make this minutes, not the laptop's hour)
[ -f data/manifests/MANIFESTS_OK ] || { uv run python scripts/fetch_data.py --dns-shards \
  $(for s in freesound_000 freesound_001 audioset_000 audioset_001 audioset_002 audioset_003 audioset_004 audioset_005 audioset_006; do
      echo $B/datasets_fullband.noise_fullband.$s.tar.bz2; done) && touch data/manifests/MANIFESTS_OK; }
# relabel pass (MAD shooting/shelling/footsteps -> changing, ESC-50 impulsive shortlist) is inside the scanners now;
# run it anyway so a manifest restored from the laptop matches
uv run python scripts/relabel_noise_class.py data/manifests/*.parquet   # positional manifests are required; bare call aborted the bootstrap
# r3+ recipes train on the full train-clean-100; fetch_data only writes the 20 h librispeech.parquet (missed on two boxes)
[ -f data/manifests/librispeech_100h.parquet ] || uv run python scripts/make_librispeech_100h.py

# --- packed corpus: one int16 memmap the loader reads instead of decoding FLAC per sample (4.3) --------------
# The training step is ~90 % CPU and a large slice of that is 3-5 FLAC decodes per item. Packing is verified
# bit-identical, so this changes speed and nothing else.
[ -f data/pack/index.json ] || uv run python scripts/pack_corpus.py --manifests 'data/manifests/*.parquet' --out data/pack

# --- page cache: pay the cold reads once here rather than through the first epoch (3.4) ----------------------
# Speech is read on every single item; noise is sampled from a much larger pool, so warm speech first and let
# the rest land only if there is RAM for it.
warm() { [ -e "$1" ] || return 0; find "$1" -type f -print0 2>/dev/null | xargs -0 -P "$(nproc)" -n 64 cat > /dev/null 2>&1 || true; }
warm data/pack
warm data/raw/librispeech; warm data/raw/cv_hi; warm data/raw/ears
free -g | sed -n '1,2p'   # the buffer/cache column should have grown by roughly the corpus size

# --- RIR banks: joined below, started at the top of the script --------------------------------------------
wait_banks
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
