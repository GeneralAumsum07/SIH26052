# vaani_ld — native low-delay runtime

This is the C++ runtime for the low-delay VaaniFE contracts (`vaanife_ld_asym512_h{96,128}_s{160,144,128}_v1`), built for low-delay plan Task 7. It runs the same frontend, analysis, ONNX step, synthesis and R0/R1/R2 resamplers as the Python reference (`vaani/low_delay_live.py`), and it is checked against the Task 6 golden vectors in `deploy/dsp_reference/vectors_ld/`.

## Dependencies (pinned)

| Dependency | Version | Where |
|---|---|---|
| ONNX Runtime C/C++ | 1.30.0 (sha256-checked release) | `deps/fetch_ort.sh [x64\|aarch64]` → `_deps/` |
| PocketFFT (BSD-3) | `cpp` @ c90e55b, with a patch that adds scratch-buffer overloads | `third_party/pocketfft/` (`VERSION`, `vaani_ld_scratch.patch`) |
| ALSA (`libasound2-dev`) | system | required for `live` and `vld_period_test` unless `-DVLD_WITH_ALSA=OFF` |

CMake stops with a clear error if ONNX Runtime is missing or is not version 1.30.0. It also stops if ALSA is missing and `VLD_WITH_ALSA` is ON.

## Build

```sh
native/vaani_ld/deps/fetch_ort.sh
cmake -S native/vaani_ld -B native/vaani_ld/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/vaani_ld/build -j
pytest tests/test_low_delay_native.py        # builds into build-test/ and runs every check below
```

**Arm (cross build from x86):** `native/vaani_ld/deps/cross_build_arm64.sh results_r2/r8_ld/native/arm64_build.json` builds with `cmake/aarch64-linux-gnu.cmake` and a private arm64 ALSA sysroot, then runs the golden and unit tests under qemu-user.

**On the Pi:** build natively with the same commands; `fetch_ort.sh` picks `aarch64`.

## Executables

| Binary | Purpose |
|---|---|
| `vaani_ld_run info` | Checks the graph's contract metadata, window hash and state size. |
| `vaani_ld_run wav` | Offline 2-channel (primary, reference) processing at 16 kHz, or at 48 kHz with `--resampler`. `--split-at N` saves and restores the state mid-stream; the output must stay bit-identical. |
| `vaani_ld_run simulate` | Deterministic device-timeline model for early, nominal, late, edge and jitter step times. It refuses settings that cannot fit Section 4 (exit code 3). |
| `vaani_ld_run live` | Direct ALSA duplex on linked hw PCMs, with SCHED_FIFO, `mlockall`, a pinned CPU and denormal flushing. A period-writer thread feeds from a bounded ring; missed deadlines trigger recovery through the delay-matched bypass. It writes a JSON report with `qualifies`. |
| `vld_step_bench` | Gate 0a step timing (paced; random, silent and low-level inputs; FZ on and off), including allocation counts. |
| `vld_period_test` | Gate 0a/0b identity passthrough at 1 ms periods: negotiated settings, xruns and wake latency. |
| `vld_golden_test`, `vld_unit_test` | Parity against the golden vectors; tests for contract refusal, state, no allocation in the DSP, bypass and the simulator. |

## Left for the owner (hardware)

These steps need the Pi 5 and the D6 audio hardware:

- `vld_step_bench` for 10 minutes or more per graph, with FZ set and cleared.
- cyclictest.
- `vld_period_test --period 48 --seconds 1800`.
- The Gate 0b measurements.
- `vaani_ld_run live` qualification runs. Any recovery, xrun or drift disqualifies a run.

qemu results show build and numerics only; they say nothing about timing.

ONNX Runtime's `Run` allocates on every hop; the count is reported by the unit test and `vld_step_bench`, as the plan requires ("measure separately"). The runtime's own DSP makes no allocations after warmup.
