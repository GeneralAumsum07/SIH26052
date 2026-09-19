#!/usr/bin/env bash
# Sequential data-sourcing queue (plan Tier 1.5): waits for the gunshot zip already in flight, then pulls
# DNS-5 noise shards freesound_001 + audioset_000-001 (fetch_data extracts + scans them and refreshes every manifest, incl. gunshots),
# then EARS speakers p001-p020 (~590 MB each; scan_ears runs once the layout is known).
set -u
cd "$(dirname "$0")/.."
until [ -f data/download/gunshots/edge-collected-gunshot-audio.zip.ok ]; do sleep 60; done
B=https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/noise_fullband
# freesound_000 is already local; 002/003 do not exist on the blob (404 checked 2026-09-20). List from Rachit.
uv run python scripts/fetch_data.py --dns-shards   $B/datasets_fullband.noise_fullband.freesound_001.tar.bz2   $B/datasets_fullband.noise_fullband.audioset_000.tar.bz2   $B/datasets_fullband.noise_fullband.audioset_001.tar.bz2 > data/download/fetch_dns.log 2>&1
for i in $(seq -w 1 20); do
  uv run python -c "
from pathlib import Path; from scripts.fetch_data import download
download('https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p0$i.zip', Path('data/download/ears/p0$i.zip'))"
done > data/download/ears/queue.log 2>&1
echo "fetch queue done $(date)" > data/download/QUEUE_DONE
# drone corpus (attribution accepted 2026-09-20): fetch_data handles download + scan; the round1 sources are idempotent
uv run python scripts/fetch_data.py > data/download/fetch_drone.log 2>&1
echo "fetch queue done $(date)" > data/download/QUEUE_DONE
# remaining audioset shards after EARS so speech diversity lands before more noise; each ~1-3.5 GB
uv run python scripts/fetch_data.py --dns-shards   $B/datasets_fullband.noise_fullband.audioset_002.tar.bz2   $B/datasets_fullband.noise_fullband.audioset_003.tar.bz2   $B/datasets_fullband.noise_fullband.audioset_004.tar.bz2   $B/datasets_fullband.noise_fullband.audioset_005.tar.bz2   $B/datasets_fullband.noise_fullband.audioset_006.tar.bz2 > data/download/fetch_dns2.log 2>&1
echo "fetch queue done $(date)" > data/download/QUEUE_DONE
