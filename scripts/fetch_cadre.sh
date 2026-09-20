#!/usr/bin/env bash
# Cadre Gunshot Audio Forensics dataset (registration-gated page; the archives themselves are public Box
# shared links). Zoom H4N recorder only: the iPhone/Samsung files went through phone AGC, so their crest
# factor says nothing about the muzzle blast. 18 firearms, mono 44.1 kHz. Same aria2c resume pattern as
# fetch_demand.sh; safe to re-run. Whether the files count as "impulsive" is decided by crest_audit.py later.
set -u
cd "$(dirname "$0")/.."
export PATH="$PATH:/c/Users/Rachit/AppData/Local/Microsoft/WinGet/Links:/c/Users/Rachit/AppData/Local/Microsoft/WinGet/Packages/aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe/aria2-1.37.0-win-64bit-build1"
LOG=data/download/queue.log
D=data/download/cadre; mkdir -p "$D"
# Box ids copied from https://cadreforensics.com/audio/download/ ("Datafile (Zoom)" links) on 2026-09-20.
while read -r gun id; do
  dst=$D/${gun}_zoom.zip
  [ -f "$dst.ok" ] && continue
  aria2c -c -x4 -s4 -k 10M --max-tries=0 --retry-wait=10 --timeout=60 --file-allocation=none \
    --console-log-level=warn --summary-interval=60 -d "$D" -o "$(basename "$dst")" \
    "https://tulane.box.com/shared/static/$id" \
    && touch "$dst.ok" && echo "$(date +%H:%M) done $dst" >> "$LOG"
done <<'EOF'
High_Standard_Sport_King wr56pjy6amdxseu1q4iam1sp2e3dm7zo.zip
S_W_34_1 78pwg2z297wl6crlbyss9xezpzah87k3.zip
Ruger_10_22 npm8gg0ho91kqvjueg6i3i42anxr4bow.zip
Remington_33_Bolt_Action_Rifle 1wqle7vj6k00xt7dcodtsn1sg3htp7nm.zip
Lorcin_L380 jp7dgd5wswfahwovf8j03oe01h609xeq.zip
S_W_10_8 m6o6z7oyt2mkrcvizgrk2eq4un2poqc0.zip
Ruger_Blackhawk gyeeij3ppmhfwf1wziln2tevw3nxeubb.zip
Glock_19 uzst0osy4ida973barrpbddt5s4bxqhy.zip
Sig_P225 ywptb1ykkgwr7bhzfw79eufgygymr394.zip
M_P_40 qqikn09dgs6kwtyjx4xazbty8a73ziy4.zip
HK_USP_Compact wl2i1btzl94zc6i3s4m0zmsbd74eg8gb.zip
Glock_21 9e04uffrunbf9v0akqvdrb3v6d6m7i20.zip
Colt_1911 apn7f8xa3t6hnwfljnqo0hahrrg6mo6r.zip
Kimber_Tactical_Custom 8mxj4phb3pr090pmytvspkj9n8dqex6y.zip
M16A1_AR15 5tft9m5nxpn40vdytcxip2n2iun4tx81.zip
WASR_10_63_AK47 27pgqxuiu7dax7zipsvzd90wxuygzm8h.zip
Winchester_M14 m30u8xshzu5uzdp5bc7wzwyvpktd1a5b.zip
Remington_700 nj7g2bonbmtydc43ag0aeny82x5lnkb3.zip
EOF
touch "$D/QUEUE_DONE"
