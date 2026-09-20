# Tier 4.6 anchor re-selection (plan §2, 2026-09-21 00:30 IST)

Rule (declared in the plan, not a statistical threshold): keep candidates with nominal val STOI >= e32 - 0.003 and
PESQ >= e32 - 0.01; pick the highest nominal SNR_out, ties by PESQ then path. All four candidates scored on the same
frozen `data/eval_r2/val` (1480 items, content-hashed in `anchor.json`), `vaani.eval --workers 8`, no ASR/DNSMOS.

| checkpoint | val nominal SNR_out | STOI | PESQ | eligible |
|---|---|---|---|---|
| runs/vaani_full_r3_e32/best.pt | 14.022 | 0.9142 | 2.330 | yes |
| runs/vaani_full_r4/best.pt | 14.138 | 0.9168 | 2.355 | yes |
| runs/vaani_full_r4_s1/best.pt | 14.136 | 0.9168 | 2.353 | yes |
| **runs/vaani_full_r4_ctl/best.pt** | **14.186** | **0.9170** | **2.370** | **selected** |

Evidence: `results_r2/tier46/valsel/*.csv`. Frozen copy: `runs/tier46_anchor_v2/best.pt`, sha256 `8c67be30eb01...`.

Why a second protocol directory: the post-filter experiment (`results_r2/tier46/`) was pre-registered and killed under
the e32 anchor; the plan forbids overwriting an anchor mid-experiment, so the refiner experiment starts here with its
own `anchor.json`. e32 remains the anchor of record for the post-filter result.

Observation (evidence, test and val agree): r4_ctl trains on the r3 manifests for 32 more epochs from e32; r4 / r4_s1
add AudioSet, FreeSound, Cadre and DEMAND and land slightly below it. The gain over e32 is epochs, not data.
