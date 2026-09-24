# Reference conditions on VAL: ckpt:results_r2/runs/r7_e256_wr64_refiner/best.pt

Source: `r7_refconditions_val.csv` (2220 rows, 148 val clips, 0 errored rows).
Command: `python scripts/eval_refvalid.py --system ckpt:results_r2/runs/r7_e256_wr64_refiner/best.pt --out results_r2/r8/r7_refconditions_val --per-bucket 4 --workers 2`

Fail = speech loss > 0.15 or SNR_out < SNR_in - 1 dB (per clip). recovery_s: burst_dropout only, worst of the
two reconnects, against the same clip's present-condition output. click_db: largest sample step at a dropout
edge over the clip's p99 step. VAL only; no test-set number here.

| condition     |       n |   snr_out |   d_snr_vs_present |   stoi |   pesq_wb |   speech_loss |   speech_loss_p95 |   longest_lost_s |   recovery_s_max |   click_db_max |   fails |
|:--------------|--------:|----------:|-------------------:|-------:|----------:|--------------:|------------------:|-----------------:|-----------------:|---------------:|--------:|
| present       | 148.000 |    13.930 |              0.000 |  0.897 |     2.315 |         0.064 |             0.285 |            0.060 |              nan |        nan     |  21.000 |
| absent        | 148.000 |     7.392 |             -6.538 |  0.783 |     1.512 |         0.163 |             0.609 |            0.155 |              nan |        nan     |  54.000 |
| burst_dropout | 148.000 |    11.318 |             -2.612 |  0.853 |     1.936 |         0.096 |             0.383 |            0.093 |              inf |         23.844 |  29.000 |
| gain_m12      | 148.000 |     5.094 |             -8.836 |  0.822 |     1.570 |         0.063 |             0.279 |            0.058 |              nan |        nan     |  16.000 |
| lowpass       | 148.000 |     3.270 |            -10.661 |  0.791 |     1.480 |         0.045 |             0.247 |            0.042 |              nan |        nan     |  15.000 |
| delay         | 148.000 |    11.936 |             -1.994 |  0.875 |     2.090 |         0.077 |             0.328 |            0.064 |              nan |        nan     |  25.000 |
| clipped       | 148.000 |     7.857 |             -6.073 |  0.839 |     1.782 |         0.068 |             0.284 |            0.058 |              nan |        nan     |  23.000 |
| talker_leak   | 148.000 |     0.129 |            -13.801 |  0.489 |     1.107 |         0.953 |             1.000 |            1.193 |              nan |        nan     | 148.000 |
| ild_-14       | 148.000 |    15.835 |              1.905 |  0.938 |     2.753 |         0.034 |             0.149 |            0.038 |              nan |        nan     |   8.000 |
| ild_-10       | 148.000 |    15.358 |              1.428 |  0.937 |     2.721 |         0.042 |             0.181 |            0.043 |              nan |        nan     |  11.000 |
| ild_-8        | 148.000 |    14.233 |              0.303 |  0.933 |     2.661 |         0.054 |             0.245 |            0.057 |              nan |        nan     |  14.000 |
| ild_-6        | 148.000 |     9.998 |             -3.932 |  0.909 |     2.305 |         0.114 |             0.447 |            0.123 |              nan |        nan     |  40.000 |
| ild_-4        | 148.000 |     1.112 |            -12.819 |  0.661 |     1.166 |         0.744 |             1.000 |            0.742 |              nan |        nan     | 147.000 |
| ild_-2        | 148.000 |     0.093 |            -13.837 |  0.569 |     1.074 |         0.998 |             1.000 |            1.445 |              nan |        nan     | 148.000 |
| ild_0         | 148.000 |    -0.017 |            -13.947 |  0.535 |     1.089 |         1.000 |             1.000 |            1.452 |              nan |        nan     | 148.000 |
