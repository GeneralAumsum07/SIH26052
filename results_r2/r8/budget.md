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
