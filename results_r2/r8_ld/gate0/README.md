# Gate 0a evidence (low-delay plan Task 0)

| File | What | Made by |
|---|---|---|
| `resampler.json` | R0/R1/R2 magnitude, stopband, group delay, identity SNR, streaming vs offline | `scripts/make_resampler_fir.py` |
| `limiter_bench.json` | 32-sample limiter: loop vs compiled (numba) vs vectorized (NumPy), seconds per 4 s crop (laptop smoke) | `scripts/bench_loader.py --limiter-only` |
| `eligibility.json` | D_proc per arm, eligibility of every support under R0/R1/R2, the registered support selection and the release-schedule simulation | `scripts/ld_gate0.py` |

`eligibility.json` has status **pending_board**. The budget arithmetic, resampler records and schedule simulation are complete. The board measurements listed under `missing` are the owner's. Its selection is marked `provisional` and is **not** a Gate 0a selection.

**Budget finding.** The Section 4 budget row counts the resampler pair at its D3 maximum group delay over 300–4,000 Hz. For R1 that is 0.407 ms, not the table's rounded 0.4 ms. With the converter term at its 0.5 ms upper estimate, the upper estimate is therefore 13.007 ms, which is over 13.0 ms, in three cases:
- L = 10 ms with D_proc 1 ms;
- L = 9 ms with D_proc 2 ms;
- L = 8 ms with 2 ms periods.

The threshold is not relaxed. A Gate 0b measurement of the converter path replaces the 0.5 ms estimate (`--converters-ms X --converters-source FILE`). For example, the PCM512x low-latency filter plus the ICS-43434 is about 0.12 ms by datasheet, and that keeps all three cases eligible.

## Owner steps on the Pi 5 (PREEMPT_RT or the documented low-latency setup)

1. **Export the untrained graphs** on a torch machine: `python scripts/fe_tiers.py --low-delay`. The files go to `runs/fe_tiers_ld/`.
2. **Build the runtime:** `native/vaani_ld/deps/fetch_ort.sh`, then run cmake as described in `native/vaani_ld/README.md`.
3. **C++ step timing.** The C++ loop is the primary timing. Run this for each graph (Mini-P18 at each Arm A contract, Mini-P32 at Arm B, Arm R), each input and FZ on/off, at 10 minutes or more per run:
   `vld_step_bench --model G.onnx --contract C --hops 100000 --input random|silent|lowlevel --fz on|off --resampler deploy/resampler/r1_minphase_kaiser193_v1.json --cpu 3 >> step_<arm>.jsonl`
   (100,000 hops is 10 min at H = 96; use 75,000 at H = 128.)
4. **Python step timing:** `python scripts/ld_step_timing.py G.onnx --seconds 600 --out py_step_<arm>.json`
5. **Wake-up latency:** `cyclictest -m -S -p 80 -i 1000 -D 10m`. Record the maximum and the kernel's RT configuration, and whether you have Orin access.
6. **Period test (D6 hardware):** `vld_period_test --capture hw:X,0 --playback hw:X,0 --period 48 --seconds 1800 --cpu 3 > period_test_48.json`
7. **Build the report:**
   `python scripts/ld_gate0.py --timing arm_a=step_arm_a.jsonl,py_step_arm_a.json --timing arm_b=... --timing arm_r=... --cyclictest-max-us N --period-test period_test_48.json --out results_r2/r8_ld/gate0/eligibility.json`
