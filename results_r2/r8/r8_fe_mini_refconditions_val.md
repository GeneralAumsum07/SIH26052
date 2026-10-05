# Reference conditions on VAL: ckpt:r8_runs_final/r8_fe_mini/best.pt

Source: `r8_fe_mini_refconditions_val.csv` (2220 rows, 148 val clips, 0 errored rows).
Command: `python scripts/eval_refvalid.py --system ckpt:r8_runs_final/r8_fe_mini/best.pt --out results_r2/r8/r8_fe_mini_refconditions_val --per-bucket 4 --workers 4`

Fail = speech loss > 0.15 or SNR_out < SNR_in - 1 dB (per clip). recovery_s: burst_dropout only, worst of the
two reconnects, against the same clip's present-condition output. click_db: largest sample step at a dropout
edge over the clip's p99 step. VAL only; no test-set number here.

| condition     |       n |   snr_out |   d_snr_vs_present |   stoi |   pesq_wb |   speech_loss |   speech_loss_p95 |   longest_lost_s |   recovery_s_max |   click_db_max |   fails |
|:--------------|--------:|----------:|-------------------:|-------:|----------:|--------------:|------------------:|-----------------:|-----------------:|---------------:|--------:|
| present       | 148.000 |    10.276 |              0.000 |  0.850 |     1.871 |         0.107 |             0.451 |            0.116 |              nan |        nan     |  32.000 |
| absent        | 148.000 |     8.713 |             -1.563 |  0.816 |     1.704 |         0.128 |             0.467 |            0.128 |              nan |        nan     |  44.000 |
| burst_dropout | 148.000 |     9.701 |             -0.576 |  0.834 |     1.812 |         0.113 |             0.440 |            0.124 |              inf |         11.020 |  35.000 |
| gain_m12      | 148.000 |     9.626 |             -0.650 |  0.843 |     1.814 |         0.109 |             0.447 |            0.120 |              nan |        nan     |  33.000 |
| lowpass       | 148.000 |     9.084 |             -1.192 |  0.833 |     1.730 |         0.109 |             0.429 |            0.116 |              nan |        nan     |  36.000 |
| delay         | 148.000 |     9.044 |             -1.232 |  0.821 |     1.722 |         0.130 |             0.490 |            0.126 |              nan |        nan     |  42.000 |
| clipped       | 148.000 |     9.629 |             -0.647 |  0.839 |     1.818 |         0.111 |             0.438 |            0.116 |              nan |        nan     |  34.000 |
| talker_leak   | 148.000 |     7.994 |             -2.283 |  0.811 |     1.650 |         0.141 |             0.505 |            0.152 |              nan |        nan     |  52.000 |
| ild_-14       | 148.000 |     9.416 |             -0.860 |  0.847 |     1.804 |         0.096 |             0.365 |            0.103 |              nan |        nan     |  30.000 |
| ild_-10       | 148.000 |     9.195 |             -1.082 |  0.840 |     1.768 |         0.100 |             0.368 |            0.108 |              nan |        nan     |  32.000 |
| ild_-8        | 148.000 |     9.052 |             -1.224 |  0.835 |     1.744 |         0.105 |             0.361 |            0.113 |              nan |        nan     |  34.000 |
| ild_-6        | 148.000 |     8.270 |             -2.006 |  0.825 |     1.669 |         0.118 |             0.422 |            0.125 |              nan |        nan     |  37.000 |
| ild_-4        | 148.000 |     7.431 |             -2.845 |  0.812 |     1.625 |         0.143 |             0.484 |            0.155 |              nan |        nan     |  47.000 |
| ild_-2        | 148.000 |     6.572 |             -3.704 |  0.800 |     1.596 |         0.195 |             0.650 |            0.269 |              nan |        nan     |  62.000 |
| ild_0         | 148.000 |     6.542 |             -3.735 |  0.797 |     1.585 |         0.200 |             0.665 |            0.279 |              nan |        nan     |  65.000 |
