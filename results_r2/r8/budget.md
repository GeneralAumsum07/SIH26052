# Mini budget audit

Source: `budget.json`. Command: `python scripts/audit_budget.py` (CPU, deterministic; counts, not timings).

Budget (spec 6.2): total entries <= 60,000, matrix MMAC/s <= 90.706.

| system | total entries | learnable | ref extension entries | MMAC/s | ref extension MAC/hop | within budget |
|---|---:|---:|---:|---:|---:|---|
| r7 (shipping) | 52,747 | 28,171 | 0 | 82.460 | 0 | yes |
| refvalid Mini (C16 + ref_validity + r7 refiner) | 53,147 | 28,571 | 400 | 85.621 | 26,000 | yes |
| C32 + ref_validity (untrained projection) | 110,683 | 86,107 | 800 | 152.064 | 52,000 | NO |
| C64 + ref_validity (untrained projection) | 320,219 | 295,643 | 1,600 | 387.622 | 104,000 | NO |
| C96 + ref_validity (untrained projection) | 655,707 | 631,131 | 2,400 | 760.076 | 156,000 | NO |

Total entries count every parameter, the frozen ERB banks included; BN running buffers and the streaming
state are listed separately in the JSON. C32/C64/C96 rows are untrained projections of cost only.

## VaaniFE per audio contract (vaani_fe.count_macs at the contract's hop rate)

| network @ contract | hops/s | MAC/hop | MMAC/s | entries (training form) | within budget |
|---|---:|---:|---:|---:|---|
| C0 Mini (legacy 512/256) | 62.50 | 1,116,160 | 69.760 | 29,914 | yes |
| Arm A Mini-P18 @ vaanife_ld_asym512_h96_s160_v1 | 166.67 | 521,504 | 86.917 | 55,862 | yes |
| Arm A Mini-P18 @ vaanife_ld_asym512_h96_s144_v1 | 166.67 | 521,504 | 86.917 | 55,862 | yes |
| Arm A Mini-P18 @ vaanife_ld_asym512_h96_s128_v1 | 166.67 | 521,504 | 86.917 | 55,862 | yes |
| Arm B Mini-P32 @ vaanife_ld_asym512_h128_s160_v1 | 125.00 | 697,376 | 87.172 | 40,942 | yes |
| Arm A Mini-P18 pr_nhat @ vaanife_ld_asym512_h96_s160_v1 (over the entry budget; not eligible) | 166.67 | 537,888 | 89.648 | 63,798 | NO |
| Arm A Mini-P18 pr_nhat @ vaanife_ld_asym512_h96_s144_v1 (over the entry budget; not eligible) | 166.67 | 537,888 | 89.648 | 63,798 | NO |
| Arm A Mini-P18 pr_nhat @ vaanife_ld_asym512_h96_s128_v1 (over the entry budget; not eligible) | 166.67 | 537,888 | 89.648 | 63,798 | NO |
| Arm B Mini-P32 pr_nhat @ vaanife_ld_asym512_h128_s160_v1 | 125.00 | 713,760 | 89.220 | 44,782 | yes |
| Arm R Mini + P18 deep filter @ vaanife_ld_asym512_h96_s160_v1 (reference, never eligible) | 166.67 | 1,153,024 | 192.171 | 31,456 | NO |
