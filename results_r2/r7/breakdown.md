# r7 cascade: per class, per input SNR, per clip

Source: `results_r2/r7/r7_e256_wr64_cascade_eval_r2.csv` (system `cascade:runs/r7_e256_wr64_refiner/best.pt`, eval_r2 test, current render). Raw passthrough: `results_r2/r7/raw_eval_r2_relabel.csv`. Regenerate: `uv run python scripts/r7_breakdown.py`.

Targets: SNR_out > 15 dB, STOI > 0.85, PESQ-WB > 2.5. All-three pass = the fraction of clips on which all three hold at once (strict >).
Row CI: 1000-sample bootstrap over clips (`vaani.report.ci`, as in matrix.md). Cluster CI: 1000-sample bootstrap over scene clusters (snr_in, id) (`vaani.report.cluster_ci`); clips from one seed at one SNR share speech and room across classes.

## Headline rows

| rows | n | SNR_out (row CI) | SNR_out (cluster CI) | STOI (row) | STOI (cluster) | PESQ-WB (row) | PESQ-WB (cluster) | all-three pass (row) | all-three pass (cluster) |
|---|---|---|---|---|---|---|---|---|---|
| Nominal (0/5/10 dB, unclipped, no fault) | 617 | 14.864 [14.565, 15.183] | 14.864 [14.421, 15.339] | 0.917 [0.911, 0.922] | 0.917 [0.910, 0.923] | 2.462 [2.412, 2.511] | 2.462 [2.391, 2.531] | 35.5% [31.8, 39.1] | 35.5% [30.2, 40.6] |
| Full test split | 2280 | 12.778 [12.520, 13.023] | 12.778 [12.292, 13.264] | 0.870 [0.866, 0.874] | 0.870 [0.862, 0.879] | 2.138 [2.106, 2.169] | 2.138 [2.076, 2.203] | 24.1% [22.5, 25.9] | 24.1% [20.7, 28.0] |
| Non-fault, input 0 dB | 240 | 12.050 [11.668, 12.477] | 12.050 [11.581, 12.520] | 0.878 [0.868, 0.887] | 0.878 [0.869, 0.888] | 2.036 [1.966, 2.111] | 2.036 [1.959, 2.115] | 9.2% [5.8, 12.9] | 9.2% [5.4, 13.8] |
| Non-fault, input -5 dB | 240 | 9.782 [9.322, 10.258] | 9.782 [9.318, 10.288] | 0.832 [0.818, 0.845] | 0.832 [0.819, 0.844] | 1.807 [1.737, 1.882] | 1.807 [1.739, 1.881] | 7.1% [3.8, 10.4] | 7.1% [4.2, 10.4] |
| Non-fault, input -10 dB | 240 | 7.210 [6.775, 7.678] | 7.210 [6.889, 7.582] | 0.749 [0.733, 0.767] | 0.749 [0.733, 0.766] | 1.487 [1.432, 1.548] | 1.487 [1.446, 1.531] | 2.5% [0.8, 5.0] | 2.5% [0.8, 4.6] |
| Transients (fault_burst_*, input 0/5 dB) | 240 | 10.866 [10.494, 11.206] | 10.866 [10.288, 11.462] | 0.848 [0.837, 0.858] | 0.848 [0.831, 0.863] | 1.797 [1.753, 1.843] | 1.797 [1.719, 1.875] | 3.8% [1.7, 6.2] | 3.8% [0.8, 7.5] |

## Per noise class x input SNR (non-fault buckets; clipped items included, so these are not nominal cells)

| noise class | input SNR (dB) | n | SNR_out | STOI | PESQ-WB | all-three pass |
|---|---|---|---|---|---|---|
| changing | -10 | 40 | 8.901 | 0.801 | 1.656 | 5.0% |
| changing | -5 | 40 | 11.107 | 0.870 | 2.024 | 15.0% |
| changing | 0 | 40 | 13.138 | 0.898 | 2.216 | 15.0% |
| changing | 5 | 40 | 16.059 | 0.942 | 2.654 | 42.5% |
| changing | 10 | 40 | 18.327 | 0.963 | 3.092 | 82.5% |
| changing | 15 | 40 | 21.720 | 0.966 | 3.334 | 92.5% |
| clean | inf | 40 | 41.113 | 1.000 | 4.584 | 100.0% |
| impulsive | -10 | 40 | 9.211 | 0.814 | 1.683 | 10.0% |
| impulsive | -5 | 40 | 11.656 | 0.881 | 2.108 | 10.0% |
| impulsive | 0 | 40 | 12.562 | 0.901 | 2.113 | 15.0% |
| impulsive | 5 | 40 | 16.169 | 0.931 | 2.640 | 50.0% |
| impulsive | 10 | 40 | 17.589 | 0.961 | 2.884 | 67.5% |
| impulsive | 15 | 40 | 20.082 | 0.958 | 3.143 | 87.5% |
| impulsive+stationary | -10 | 40 | 6.111 | 0.706 | 1.373 | 0.0% |
| impulsive+stationary | -5 | 40 | 8.354 | 0.789 | 1.571 | 0.0% |
| impulsive+stationary | 0 | 40 | 11.683 | 0.874 | 1.980 | 5.0% |
| impulsive+stationary | 5 | 40 | 13.248 | 0.905 | 2.201 | 2.5% |
| impulsive+stationary | 10 | 40 | 16.975 | 0.937 | 2.618 | 50.0% |
| impulsive+stationary | 15 | 40 | 19.410 | 0.960 | 3.069 | 80.0% |
| recorded_impulsive | -10 | 40 | 8.101 | 0.801 | 1.639 | 0.0% |
| recorded_impulsive | -5 | 40 | 11.178 | 0.868 | 1.892 | 10.0% |
| recorded_impulsive | 0 | 40 | 13.459 | 0.888 | 2.170 | 15.0% |
| recorded_impulsive | 5 | 40 | 16.238 | 0.937 | 2.674 | 45.0% |
| recorded_impulsive | 10 | 40 | 18.197 | 0.953 | 2.880 | 67.5% |
| recorded_impulsive | 15 | 40 | 19.423 | 0.966 | 3.157 | 85.0% |
| recorded_impulsive+stationary | -10 | 40 | 5.382 | 0.691 | 1.271 | 0.0% |
| recorded_impulsive+stationary | -5 | 40 | 8.777 | 0.791 | 1.697 | 5.0% |
| recorded_impulsive+stationary | 0 | 40 | 10.694 | 0.846 | 1.823 | 2.5% |
| recorded_impulsive+stationary | 5 | 40 | 13.868 | 0.896 | 2.273 | 15.0% |
| recorded_impulsive+stationary | 10 | 40 | 16.316 | 0.943 | 2.688 | 62.5% |
| recorded_impulsive+stationary | 15 | 40 | 18.023 | 0.964 | 2.907 | 75.0% |
| stationary | -10 | 40 | 5.554 | 0.684 | 1.299 | 0.0% |
| stationary | -5 | 40 | 7.621 | 0.793 | 1.551 | 2.5% |
| stationary | 0 | 40 | 10.762 | 0.860 | 1.916 | 2.5% |
| stationary | 5 | 40 | 13.827 | 0.901 | 2.400 | 20.0% |
| stationary | 10 | 40 | 16.621 | 0.929 | 2.735 | 55.0% |
| stationary | 15 | 40 | 19.179 | 0.963 | 3.114 | 92.5% |

## Fault buckets (input 0/5 dB; compare with fault_none, not with nominal)

| fault | input SNR (dB) | n | SNR_out | STOI | PESQ-WB | all-three pass |
|---|---|---|---|---|---|---|
| fault_burst_overload | 0 | 40 | 9.960 | 0.818 | 1.654 | 0.0% |
| fault_burst_overload | 5 | 40 | 10.658 | 0.858 | 1.783 | 2.5% |
| fault_burst_p24dB | 0 | 40 | 11.021 | 0.843 | 1.833 | 2.5% |
| fault_burst_p24dB | 5 | 40 | 12.845 | 0.891 | 2.068 | 12.5% |
| fault_burst_p36dB | 0 | 40 | 10.022 | 0.819 | 1.665 | 0.0% |
| fault_burst_p36dB | 5 | 40 | 10.691 | 0.859 | 1.777 | 5.0% |
| fault_clip_hard | 0 | 40 | 7.361 | 0.800 | 1.453 | 0.0% |
| fault_clip_hard | 5 | 40 | 7.846 | 0.852 | 1.633 | 2.5% |
| fault_clip_mild | 0 | 40 | 11.259 | 0.853 | 1.890 | 2.5% |
| fault_clip_mild | 5 | 40 | 13.567 | 0.904 | 2.181 | 17.5% |
| fault_none | 0 | 40 | 11.443 | 0.856 | 1.934 | 5.0% |
| fault_none | 5 | 40 | 14.099 | 0.907 | 2.271 | 17.5% |
| fault_refdesync | 0 | 40 | 9.565 | 0.816 | 1.598 | 0.0% |
| fault_refdesync | 5 | 40 | 12.486 | 0.892 | 1.991 | 7.5% |
| fault_refdrop_long | 0 | 40 | 9.308 | 0.799 | 1.554 | 0.0% |
| fault_refdrop_long | 5 | 40 | 12.326 | 0.879 | 1.849 | 2.5% |
| fault_refgain_-12dB | 0 | 40 | 4.830 | 0.782 | 1.324 | 0.0% |
| fault_refgain_-12dB | 5 | 40 | 8.893 | 0.852 | 1.594 | 7.5% |
| fault_refobstruct | 0 | 40 | 4.535 | 0.764 | 1.269 | 0.0% |
| fault_refobstruct | 5 | 40 | 8.996 | 0.840 | 1.523 | 5.0% |

## Paired vs raw passthrough, nominal clips, by noise class (delta = r7 - raw on the same clip)

| noise_class | n | snr_out r7 / raw / delta [cluster CI] | stoi r7 / raw / delta [cluster CI] | pesq_wb r7 / raw / delta [cluster CI] |
|---|---|---|---|---|
| changing | 99 | 16.098 / 1.874 / 14.224 [13.320, 15.293] | 0.940 / 0.836 / 0.104 [0.090, 0.119] | 2.715 / 1.313 / 1.401 [1.291, 1.503] |
| impulsive | 107 | 15.522 / 1.960 / 13.562 [12.856, 14.248] | 0.931 / 0.812 / 0.119 [0.106, 0.132] | 2.562 / 1.255 / 1.307 [1.193, 1.416] |
| impulsive+stationary | 100 | 14.153 / 2.377 / 11.777 [11.049, 12.498] | 0.905 / 0.772 / 0.133 [0.121, 0.146] | 2.290 / 1.163 / 1.127 [1.032, 1.221] |
| recorded_impulsive | 104 | 15.966 / 1.916 / 14.050 [13.144, 14.943] | 0.928 / 0.811 / 0.117 [0.103, 0.129] | 2.580 / 1.259 / 1.321 [1.229, 1.422] |
| recorded_impulsive+stationary | 105 | 13.704 / 2.243 / 11.461 [10.810, 12.152] | 0.899 / 0.770 / 0.128 [0.117, 0.141] | 2.265 / 1.160 / 1.105 [1.011, 1.206] |
| stationary | 102 | 13.741 / 2.252 / 11.489 [10.852, 12.126] | 0.899 / 0.773 / 0.126 [0.114, 0.138] | 2.361 / 1.185 / 1.176 [1.075, 1.278] |
