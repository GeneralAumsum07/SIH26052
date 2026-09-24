# Optimization: ONNX, INT8 quantization and magnitude pruning

Frozen round-2 test split (2280 items); tables below use the nominal envelope (unclipped, no reference fault, no fault bucket, input SNR 0/5/10 dB) so they are directly comparable with [the earlier-render ablation matrix](../matrix_prerelabel.md), the render they were measured on.

> **Eval render.** Like the matrix, this was measured on the earlier frozen render of eval_r2, not the current render used in [`../r6/`](../r6/) (they differ at 606 of the 617 nominal items; the reference cascade scores 15.150 dB here and 14.753 dB there). Every comparison below is paired within this render, so the export parity, the INT8 deltas and the pruning falloff stand; the absolute levels are not comparable with `r6/`.

Targets (problem statement): SNR_out > 15 dB, STOI > 0.85, PESQ > 2.5. ✓ = lower 95 % bootstrap bound exceeds target, ~ = mean does but the bound does not, ✗ = mean does not.

## Graph size and single-core latency

ORT CPU provider, one intra-op thread, 10 s synthetic stream (624 timed frames), best of 5 interleaved repeats. Model only: no DSP, STFT/iSTFT or audio I/O. The budget is one 16 ms hop.

| graph | bytes | nodes | initializer bytes | ms/frame mean | ms/frame p99 |
|---|---:|---:|---:|---:|---:|
| `cascade.onnx` | 474,599 | 1,756 | 210,108 | 0.906 | 1.152 |
| `cascade.int8.onnx` | 567,969 | 1,906 | 157,133 | 1.319 | 1.626 |
| `cascade.int8_pc.onnx` | 569,805 | 1,906 | 158,913 | 1.310 | 1.582 |

## Quality cost of export and quantization

The FP32 ONNX row is the control: it separates any export difference from the quantization difference, so a delta on the INT8 row cannot be blamed on ONNX.

| system | n | snr_out | si_sdr | stoi | pesq_wb | dnsmos_ovrl |
|---|---|---|---|---|---|---|
| PyTorch checkpoint (reference) | 617 | 15.150 [14.865,15.461] ~ | 15.345 [15.031,15.684] | 0.922 [0.917,0.927] ✓ | 2.548 [2.498,2.603] ~ | 2.702 [2.674,2.734] |
| ONNX FP32 graph | 617 | 15.150 [14.865,15.461] ~ | 15.345 [15.031,15.684] | 0.922 [0.917,0.927] ✓ | 2.548 [2.498,2.603] ~ | 2.702 [2.674,2.734] |
| ONNX INT8 dynamic | 617 | 13.968 [13.749,14.218] ✗ | 14.019 [13.776,14.297] | 0.915 [0.910,0.920] ✓ | 2.382 [2.335,2.433] ✗ | 2.652 [2.623,2.685] |

Paired per-clip change against the reference row (`*` = 95 % interval excludes zero):

| system | n paired | Δ snr_out | Δ si_sdr | Δ stoi | Δ pesq_wb | Δ dnsmos_ovrl |
|---|---|---|---|---|---|---|
| ONNX FP32 graph | 617 | +0.000 [-0.000,+0.000] | +0.000 [-0.000,+0.000] | -0.000 [-0.000,+0.000] | -0.000 [-0.000,+0.000] | +0.000 [+0.000,+0.000]* |
| ONNX INT8 dynamic | 617 | -1.183 [-1.272,-1.102]* | -1.326 [-1.433,-1.231]* | -0.008 [-0.008,-0.007]* | -0.166 [-0.176,-0.156]* | -0.051 [-0.057,-0.043]* |

## Pruning budget

Global magnitude pruning over 21,952 learned weight elements in 72 tensors, out of 52,747 total parameters. Excluded: `erb_fc`, `ierb_fc` -- the fixed ERB analysis/synthesis matrices, which are a signal transform rather than learned capacity.

| level | requested | achieved (prunable) | achieved (all parameters) | weights zeroed |
|---|---:|---:|---:|---:|
| p00 | 0% | 0.0000 | 0.0000 | 0 |
| p10 | 10% | 0.1000 | 0.0416 | 2,195 |
| p20 | 20% | 0.2000 | 0.0832 | 4,390 |
| p30 | 30% | 0.3000 | 0.1249 | 6,586 |
| p40 | 40% | 0.4000 | 0.1665 | 8,781 |
| p50 | 50% | 0.5000 | 0.2081 | 10,976 |

## Quality cost of magnitude pruning

Pruned in PyTorch and evaluated through the same path as the reference row, so the only variable is which weights are zero. No fine-tuning after pruning: this measures what the trained weights tolerate, which is the question a deployment budget asks.

| system | n | snr_out | si_sdr | stoi | pesq_wb | dnsmos_ovrl |
|---|---|---|---|---|---|---|
| p00 (unpruned reference) | 617 | 15.150 [14.865,15.461] ~ | 15.345 [15.031,15.684] | 0.922 [0.917,0.927] ✓ | 2.548 [2.498,2.603] ~ | 2.702 [2.674,2.734] |
| p10 (10 % of learned weights zeroed) | 617 | 15.019 [14.735,15.333] ~ | 15.208 [14.897,15.549] | 0.922 [0.917,0.926] ✓ | 2.529 [2.479,2.584] ~ | 2.697 [2.669,2.728] |
| p20 (20 % of learned weights zeroed) | 617 | 14.413 [14.139,14.710] ✗ | 14.616 [14.319,14.944] | 0.916 [0.910,0.921] ✓ | 2.478 [2.430,2.531] ✗ | 2.700 [2.671,2.732] |
| p30 (30 % of learned weights zeroed) | 617 | 12.980 [12.726,13.257] ✗ | 12.955 [12.669,13.256] | 0.905 [0.899,0.910] ✓ | 2.147 [2.102,2.195] ✗ | 2.563 [2.534,2.594] |
| p40 (40 % of learned weights zeroed) | 617 | 6.775 [6.546,7.023] ✗ | 7.288 [6.983,7.615] | 0.832 [0.824,0.840] ✗ | 1.816 [1.779,1.858] ✗ | 2.414 [2.384,2.448] |
| p50 (50 % of learned weights zeroed) | 617 | 0.437 [0.412,0.464] ✗ | -0.684 [-0.954,-0.432] | 0.752 [0.744,0.761] ✗ | 1.189 [1.175,1.204] ✗ | 1.798 [1.769,1.831] |

Paired per-clip change against the reference row (`*` = 95 % interval excludes zero):

| system | n paired | Δ snr_out | Δ si_sdr | Δ stoi | Δ pesq_wb | Δ dnsmos_ovrl |
|---|---|---|---|---|---|---|
| p10 (10 % of learned weights zeroed) | 617 | -0.131 [-0.158,-0.104]* | -0.136 [-0.160,-0.116]* | -0.001 [-0.001,-0.001]* | -0.019 [-0.022,-0.016]* | -0.005 [-0.009,-0.002]* |
| p20 (20 % of learned weights zeroed) | 617 | -0.738 [-0.803,-0.677]* | -0.728 [-0.794,-0.670]* | -0.007 [-0.007,-0.006]* | -0.070 [-0.080,-0.059]* | -0.003 [-0.010,+0.005] |
| p30 (30 % of learned weights zeroed) | 617 | -2.170 [-2.285,-2.064]* | -2.389 [-2.516,-2.269]* | -0.018 [-0.019,-0.016]* | -0.401 [-0.424,-0.379]* | -0.139 [-0.153,-0.125]* |
| p40 (40 % of learned weights zeroed) | 617 | -8.375 [-8.628,-8.111]* | -8.056 [-8.342,-7.784]* | -0.090 [-0.094,-0.086]* | -0.732 [-0.760,-0.701]* | -0.288 [-0.307,-0.269]* |
| p50 (50 % of learned weights zeroed) | 617 | -14.713 [-15.019,-14.418]* | -16.029 [-16.364,-15.697]* | -0.170 [-0.175,-0.165]* | -1.359 [-1.405,-1.312]* | -0.904 [-0.931,-0.876]* |

