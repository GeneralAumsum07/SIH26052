#!/usr/bin/env bash
# DEMAND (Thiemann et al. 2013, Zenodo 1227121, CC BY-SA 4.0): 18 environments, 16 kHz zips, 16-channel grid
# recordings. Adjacent grid mics sit 5 cm apart; the adapter picks a pair ~12 cm apart for the two-mic noise rows.
# Same aria2c resume-forever pattern as run_fetch_queue.sh; safe to re-run.
set -u
cd "$(dirname "$0")/.."
export PATH="$PATH:/c/Users/Rachit/AppData/Local/Microsoft/WinGet/Links:/c/Users/Rachit/AppData/Local/Microsoft/WinGet/Packages/aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe/aria2-1.37.0-win-64bit-build1"
LOG=data/download/queue.log
D=data/download/demand; mkdir -p "$D"
for env in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE PCAFETER PRESTO PSTATION SCAFE SPSQUARE STRAFFIC TBUS TCAR TMETRO; do
  dst=$D/${env}_16k.zip
  [ -f "$dst.ok" ] && continue
  aria2c -c -x4 -s4 -k 10M --max-tries=0 --retry-wait=10 --timeout=60 --file-allocation=none \
    --console-log-level=warn --summary-interval=60 -d "$D" -o "$(basename "$dst")" \
    "https://zenodo.org/records/1227121/files/${env}_16k.zip?download=1" \
    && touch "$dst.ok" && echo "$(date +%H:%M) done $dst" >> "$LOG"
done
touch "$D/QUEUE_DONE"
