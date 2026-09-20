#!/usr/bin/env bash
# Sequential data-sourcing queue (plan Tier 1.5). Big archives go through aria2c (real Range requests, per-segment
# control file, infinite retries with back-off) because the requests-based downloader died on a Zenodo read-timeout
# on a flaky mobile link; fetch_data.py then extracts + scans (it skips any archive that already has a .ok marker).
# Order: gunshot zip -> DNS freesound_001 + audioset_000-001 -> EARS p001-p020 -> drone -> audioset_002-006.
set -u
cd "$(dirname "$0")/.."
export PATH="$PATH:/c/Users/Rachit/AppData/Local/Microsoft/WinGet/Links:/c/Users/Rachit/AppData/Local/Microsoft/WinGet/Packages/aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe/aria2-1.37.0-win-64bit-build1"
LOG=data/download/queue.log
B=https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/noise_fullband

fetch() {  # fetch <url> <dst>: resume whatever is on disk, mark .ok on completion, never give up
  local url=$1 dst=$2
  [ -f "$dst.ok" ] && return 0
  mkdir -p "$(dirname "$dst")"
  # 8 streams, not 16: parallel streams reach a carrier's throttle threshold proportionally faster
  aria2c -c -x8 -s8 -k 10M --max-tries=0 --retry-wait=10 --timeout=60 --file-allocation=none \
    --console-log-level=warn --summary-interval=60 -d "$(dirname "$dst")" -o "$(basename "$dst")" "$url" \
    && touch "$dst.ok" && echo "$(date +%H:%M) done $dst" >> "$LOG"
}

fetch https://zenodo.org/api/records/7004819/files/edge-collected-gunshot-audio.zip/content data/download/gunshots/edge-collected-gunshot-audio.zip
uv run python scripts/fetch_data.py --only gunshots > data/download/fetch_gunshots.log 2>&1

# freesound_000 is already local; 002/003 do not exist on the blob (404 checked 2026-09-20). List from Rachit.
DNS1="freesound_001 audioset_000 audioset_001"
for s in $DNS1; do fetch $B/datasets_fullband.noise_fullband.$s.tar.bz2 data/download/dns/datasets_fullband.noise_fullband.$s.tar.bz2; done
uv run python scripts/fetch_data.py --dns-shards $(for s in $DNS1; do echo $B/datasets_fullband.noise_fullband.$s.tar.bz2; done) > data/download/fetch_dns.log 2>&1

for i in $(seq -w 1 20); do
  fetch https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p0$i.zip data/download/ears/p0$i.zip
done
# drone corpus (attribution accepted 2026-09-20) + EARS scan; the round1 sources are idempotent
uv run python scripts/fetch_data.py --only drone,ears > data/download/fetch_drone.log 2>&1

# remaining audioset shards after EARS so speech diversity lands before more noise; each ~1-5 GB
DNS2="audioset_002 audioset_003 audioset_004 audioset_005 audioset_006"
for s in $DNS2; do fetch $B/datasets_fullband.noise_fullband.$s.tar.bz2 data/download/dns/datasets_fullband.noise_fullband.$s.tar.bz2; done
uv run python scripts/fetch_data.py --dns-shards $(for s in $DNS2; do echo $B/datasets_fullband.noise_fullband.$s.tar.bz2; done) > data/download/fetch_dns2.log 2>&1
echo "fetch queue done $(date)" > data/download/QUEUE_DONE
