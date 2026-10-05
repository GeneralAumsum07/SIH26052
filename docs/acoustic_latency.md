# Measure full acoustic input-to-speaker delay

## What may be reported

| Quantity | What it measures | Existing evidence |
|---|---|---|
| Hop interval/deadline | Audio represented by one processing block | r7 and r8 `fe_mini`: 16 ms; LD A: 6 ms; LD B: 8 ms |
| Complete-hop compute time | CPU time to process that block, optionally including resampler computation | `board_timing.py`, trained-FE benchmark helpers, `vld_step_bench` |
| Algorithmic delay | Framing/lookahead/filter support that cannot be removed by a faster CPU | r7/standard r8: 32 ms; LD A/B transform support: 8/10 ms |
| Modeled native path budget | Arithmetic support, resampler, scheduled I/O, converter estimate and FIFO budget | Native `path_delay_ms_upper`; this is not an acoustic observation |
| Acoustic arrival delay | Original signal at the input boundary to reproduced sound near the speaker | New external-recording procedure below; hardware measurement still required |

The supplied r7 log showed about 4.27 ms mean engine processing, no lost hops and no missed 16 ms
deadlines. That supports a processing-time statement for that run. It does **not** demonstrate
sub-15 ms full delay. The r7 and standard r8 48 kHz paths already have 32 ms algorithmic plus 4 ms
resampler group delay, before compute, devices and playback queues. See `deploy/CONTRACT.md`.

Doubling compute time is not a latency measurement. Doubling the 16 ms hop happens to equal r7's
32 ms algorithmic delay; that arithmetic does not include the rest of the system.

`capture_loop.py` writes its output WAV **before** interpolation, output gain and `aplay`.
Do not analyze its saved capture/output pair as an end-to-end test. File mode also aligns its
output to the input. Neither route captures physical speaker playback.

## Equipment and recording contract

Use an **independent recorder with two genuinely independent, synchronized microphone inputs**:

- Measurement channel 0: reference microphone next to the prototype's primary microphone.
- Measurement channel 1: response microphone close to the prototype's output speaker.
- Play a speech probe from a separate source speaker near the prototype's input microphone.
- The prototype continues using its own two microphones for inference. The measurement mics
  belong to the independent recorder; they do not replace the model's primary/reference pair.

```text
source speech -> prototype input mics -> Pi/model/output transport -> output speaker
       |                                                               |
measurement reference mic                                  measurement response mic
       +---------------- one stereo recorder / one clock ---------------+
```

Keep direct source sound out of the response microphone, and reproduced output out of the reference
microphone. Distance, barriers and near-field mic placement help. If channels contain the same
direct sound, a correlation peak can otherwise manufacture a false low-latency claim. Disable the
measurement recorder's AGC, echo cancellation and noise suppression. Record uncompressed WAV.

A USB device advertising two channels may expose duplicated mono rather than two inputs. Confirm
independence by addressing each measurement mic separately. A typical mono phone room recording
mixes direct and delayed sound and cannot substitute for this test. Two independently recorded
files have unknown start offset/clock drift and are also unsuitable. Merging them into stereo
does not make their clocks synchronized.

TBD before a physical run: **which available recorder/interface has two independent synchronized
microphone inputs?** This is the current equipment question; no measured prototype result is
implied by software validation.

## Files and dependency scope

- `scripts/acoustic_latency.py`: `prepare`, `calibrate`, `isolation`, `analyze` commands.
- `vaani/acoustic_latency.py`: NumPy estimator and stimulus preparation.
- `tests/test_acoustic_latency.py`: known-delay and failure-mode checks.

Use the existing `.venv-board` on the Pi or `.venv` on the laptop. No training packages or new
Python dependencies are needed. The CLI reuses `vaani.live`'s WAV reader.

If the Pi does not have these new files yet, transfer them from Windows PowerShell; no git push is
needed:

```powershell
scp "C:\Users\Rachit\Desktop\Projects\SIH_2026\scripts\acoustic_latency.py" "thirdimpact@raspberrypi.local:~/Desktop/SIH26052/scripts/"
scp "C:\Users\Rachit\Desktop\Projects\SIH_2026\vaani\acoustic_latency.py" "thirdimpact@raspberrypi.local:~/Desktop/SIH26052/vaani/"
```

Replace `raspberrypi.local` with the IP used for SSH if necessary. Transfer the core module and CLI
together; transferring only the CLI will fail its import.

## 1. Prepare a speech probe

On the Pi, in the SSH terminal:

```bash
cd ~/Desktop/SIH26052
PY="$PWD/.venv-board/bin/python"
mkdir -p pi_results/acoustic

"$PY" scripts/acoustic_latency.py prepare \
  --speech-wav pi_bundle/clips/stationary_5.clean.wav \
  --out pi_results/acoustic/speech_probe.wav \
  --repeats 10 --gap-s 2
```

It writes a mono speech WAV and a JSON manifest with hashes, level and trial starts. The six-second
clip repeated ten times, with lead/tail padding and gaps, lasts about 84 seconds. Use varied real
speech: a denoiser can intentionally remove clicks, white noise or pure tones. A periodic tone is
also an ambiguous delay probe. The analyzer measures reference windows from the actual external
recording, not the manifest's playback timestamps.

Copy the probe to the Windows laptop if that laptop drives the source speaker:

```powershell
scp "thirdimpact@raspberrypi.local:~/Desktop/SIH26052/pi_results/acoustic/speech_probe.wav" "C:\Users\Rachit\Downloads\speech_probe.wav"
Invoke-Item "C:\Users\Rachit\Downloads\speech_probe.wav"
```

Route this playback to the **source** speaker, not the prototype's output speaker. Adjust physical
source volume to put speech near -25 dBFS at the prototype primary mic. Do not infer microphone
sound pressure from the digital probe level.

## 2. Calibrate recorder channel skew

Co-locate the two measurement microphones and record them hearing the same varied speech source.
Keep the sample rate, channel order, recorder processing and settings identical for calibration and
the experiment. Calibration estimates the differential offset of the measurement paths; the
recorder's common buffering cancels because both channels share a clock.

If the independent recorder is an ALSA interface attached to the Pi, identify its capture PCM with
`arecord -l`. **Use its PCM, not `hw:2,0` just because that was the prototype's input card.**
The following `hw:4,0` is an example and must be replaced with the actual independent recorder:

```bash
REC=hw:4,0  # example ONLY: replace using arecord -l
arecord -D "$REC" -c 2 -r 48000 -f S16_LE -d 90 pi_results/acoustic/calibration.wav
```

Start recording before playing the speech probe from the source speaker. A standalone stereo
recorder works too: transfer its stereo WAV to `pi_results/acoustic/calibration.wav`. Some devices
require a different PCM format; use supported settings consistently. Do not open the prototype's
capture device twice.

Analyze the calibration on the Pi:

```bash
"$PY" scripts/acoustic_latency.py calibrate pi_results/acoustic/calibration.wav \
  --recorder-description "external stereo recorder, 48 kHz, processing off" \
  --synchronized-stereo \
  --out pi_results/acoustic/calibration.json
```

The description must truthfully identify your recorder/settings and be reused exactly in analysis.
Only a calibration whose `status` is `ok` may be used. Keep the hardware recording duration long
enough for several active windows. If the WAV includes long startup time, increase `--warmup-s`
consistently or trim both channels together. Never shift one channel separately.

## 3. Record the running prototype acoustically

Move the reference measurement mic next to the prototype input mic and the response measurement mic
close to the output speaker. Measure these distances in metres:

1. Source speaker to prototype input microphone.
2. Source speaker to measurement reference microphone.
3. Prototype output speaker to measurement response microphone.

Use the same placement for all models where hardware permits. Start the model, let it warm up,
start the independent stereo recorder, then replay the probe. Leave enhancement enabled throughout;
do not press Enter or mix BYPASS and ENHANCED in one measurement recording. Record extra tail after
the probe to preserve full response windows.

### Required source-only isolation control

Before the measured run, keep **the experiment's exact microphone/source placement, source volume,
recorder gain and settings**, but physically mute the prototype's output speaker. Replay and record
the same probe; save that independent stereo WAV as `pi_results/acoustic/source_only.wav`.
Only the separate source speaker should make sound. A PipeWire output may be muted with
`pactl set-sink-mute @DEFAULT_SINK@ 1` and restored with `pactl set-sink-mute @DEFAULT_SINK@ 0`.
For raw native output use its physical amplifier/speaker mute; PipeWire controls do not mute a raw
ALSA route. Verify the output is actually silent.

```bash
"$PY" scripts/acoustic_latency.py isolation pi_results/acoustic/source_only.wav \
  --recorder-description "external stereo recorder, 48 kHz, processing off" \
  --synchronized-stereo --output-muted \
  --out pi_results/acoustic/isolation.json
```

Any strongly correlated source sound at the response mic is conservatively flagged, even when it
arrives several milliseconds late. A 2 ms near-zero guard alone cannot detect that leakage.
If `status` is `leak_detected` or `inconclusive`, improve isolation or recording quality and repeat
the control before claiming speaker-path latency. Restore prototype playback for the measured run.
Repeat this control whenever speaker/microphone placement or recording/source settings change.

For r7, the Pi live command can remain:

```bash
"$PY" scripts/capture_loop.py \
  --device hw:2,0 --output-route far-end --out-device pipewire \
  --seconds 120 \
  2>&1 | tee pi_results/acoustic/r7_runtime.txt
```

For standard r8:

```bash
"$PY" scripts/capture_loop.py \
  --onnx pi_bundle/fe_mini/fe_mini.folded.onnx \
  --config pi_bundle/fe_mini/model_config.json \
  --device hw:2,0 --output-route far-end --out-device pipewire \
  --seconds 120 \
  2>&1 | tee pi_results/acoustic/r8_runtime.txt
```

Confirm card numbers again if hardware changed. `pipewire` routes to its selected default output;
verify that output before recording and label its transport truthfully. Capture the external WAV
in a separate SSH terminal or on the standalone recorder. Save separate files such as
`r7_external.wav`, `r8_external.wav`, `r8_ld_arm_a_external.wav` and `r8_ld_arm_b_external.wav`.

### Low-delay runtime limitation

The existing native LD live runtime requires raw compatible ALSA capture/playback devices. It
refuses the current Bluetooth/PipeWire playback route. Measure LD with compatible wired output
using the native commands in `docs/pi5_test_guide.md`, and the same external recording procedure.
Arm B may need `--allow-ineligible` for a deliberately non-qualifying control; preserve its native
report and label this. An observed acoustic delay does not override native eligibility failures.

Do not describe an r7 Bluetooth versus LD wired comparison as an isolated model improvement:
their output transports differ. For model comparison, use a common output path where supported.

## 4. Analyze and export JSON

For each external recording, use its matching calibration, model label and output transport.
The distances below are **illustrative**, not measurements of this rig. Replace them with measured
values. Here 0.20/0.20/0.03 metres means source paths cancel and the response mic is 3 cm from the
output speaker:

```bash
"$PY" scripts/acoustic_latency.py analyze pi_results/acoustic/r7_external.wav \
  --model-label r7 --transport bluetooth \
  --recorder-description "external stereo recorder, 48 kHz, processing off" \
  --synchronized-stereo \
  --calibration-json pi_results/acoustic/calibration.json \
  --isolation-json pi_results/acoustic/isolation.json \
  --source-to-input-m 0.20 --source-to-reference-m 0.20 --speaker-to-response-m 0.03 \
  --onnx deploy/r7/cascade.onnx --config deploy/r7/model_config.json \
  --out pi_results/acoustic/r7_e2e.json
```

For r8, change the recording, label and graph/config:

```bash
"$PY" scripts/acoustic_latency.py analyze pi_results/acoustic/r8_external.wav \
  --model-label r8_fe_mini --transport bluetooth \
  --recorder-description "external stereo recorder, 48 kHz, processing off" \
  --synchronized-stereo \
  --calibration-json pi_results/acoustic/calibration.json \
  --isolation-json pi_results/acoustic/isolation.json \
  --source-to-input-m 0.20 --source-to-reference-m 0.20 --speaker-to-response-m 0.03 \
  --onnx pi_bundle/fe_mini/fe_mini.folded.onnx --config pi_bundle/fe_mini/model_config.json \
  --out pi_results/acoustic/r8_e2e.json
```

For LD recordings use the same analysis command with the actual wired transport and respective
labels `r8_ld_arm_a`/`r8_ld_arm_b`. Omit `--onnx`/`--config` if that graph's sidecar is unavailable;
retain its native runtime report alongside the acoustic report. Do not fabricate graph provenance.

Correction is:

```text
system delay = acoustic channel lag - recorder channel skew
             - (source→prototype input - source→measurement reference
                + prototype speaker→measurement response) / sound speed
```

The default sound-speed estimate is 343 m/s; the value is stored. You may instead supply an
independently measured `--geometry-correction-ms`, or measured `--channel-offset-ms` rather than a
calibration JSON. Explicit zero is an assertion based on your setup, not a convenient default.
Without both corrections, `corrected_system_delay_ms` stays null: only acoustic arrival lag is
reported. Calibration must match the recorder description, sample rate and channel order.
The isolation control must also match these and cover the same delay search and quality policy.

## Acceptance and reporting

The estimator uses centered normalized waveform correlation on complete 1-second reference
windows, with a symmetric ±1000 ms search. It accommodates gain differences and inverted polarity,
and reports every active window's accepted/rejected result. It skips inactive reference windows.

Rejections include silent response, clipping, low correlation, ambiguous distant peaks, a search
boundary, negative lag/channel reversal and a strongest near-zero peak suggesting direct leakage.
The default near-zero guard is 2 ms, not the 15 ms target. It searches the whole range first;
raising the guard cannot force it to ignore a stronger direct peak and select a desired echo.
Aligned 20 ms activity checks also reject shorter disappearing output sections that a whole-second
correlation might conceal. The level check is relative to the typical response/input gain and flags
a drop greater than 25 dB within active source frames. This is an activity proxy on the speech probe,
not a trained speech-quality scorer; inspect rejected windows.

Exit code 0: enough accepted windows and no failed active windows. Exit code 2: JSON was written but
the measurement is inconclusive. Exit code 1: invalid input/settings/IO. Missing output windows
must remain visible; a fast surviving subset does not prove low latency while speech disappears.

Important fields:

- `status`, `active_windows`, `accepted_windows`, `rejected_windows`, `accepted_fraction`.
- `acoustic_delay_ms`: descriptive min/median/mean/p95/max of accepted windows.
- `corrected_system_delay_ms`: corrected statistics, only when corrections were supplied.
- `windows`: individual delays, correlation quality and rejection reasons.
- `threshold_assessment`: threshold check for this recording only. Support requires successful
  analysis, supplied corrections and the explicit synchronized-recorder setup assertion. It is
  additionally conditional on a passed output-muted isolation control. It is not a hardware
  certification or a worst-case bound.
- Recording/calibration/graph hashes and model/transport/recorder provenance.

Waveform correlation estimates dominant signal arrival. Nonlinear speech modification can change
its peak or remove the signal entirely. The quality checks reduce this risk but do not establish
instrument accuracy: inspect channels, repeat experiments and review inconclusive windows. Keep
settings/levels/geometry fixed. Do not weaken quality thresholds until a desired latency passes.
Stats summarize windows within a recording; they are not independent repeated experiments or a
proven p95 bound. Acquire repeated runs, including quiet/noisy conditions relevant to your claim.

An apparent full r7/r8 delay below their documented 32 ms algorithmic delay is a reason to check
direct leakage, boundary placement, calibration, model route or bypass. It does not invalidate the
contract. Distinguish observed mean/tail, uncertainty, configuration and compute timing in reports.
Recognized model labels set the declared lower bound (r7/r8: 32 ms; LD A/B: 8/10 ms). If corrected
estimates contradict it by more than one sample, the CLI marks the result inconclusive. A custom
`--expected-min-system-ms` is a declared property of the actual tested pipeline, not a search filter.

The CLI refuses output paths that alias input recordings, source speech, calibration/isolation
reports or graph/config files. Stimulus output must end in `.wav`, and measurement reports in
`.json`. Original evidence is preserved, including through symlink/hardlink aliases.

Copy all acoustic recordings, calibration and JSONs to the laptop in Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "C:\Users\Rachit\Downloads\PiLatency" | Out-Null
scp -r "thirdimpact@raspberrypi.local:~/Desktop/SIH26052/pi_results/acoustic" "C:\Users\Rachit\Downloads\PiLatency"
Invoke-Item "C:\Users\Rachit\Downloads\PiLatency\acoustic"
```

## Software validation versus hardware evidence

Run `.venv\Scripts\python.exe -m pytest tests/test_acoustic_latency.py -q` on the laptop.
Tests use known synthetic delay, polarity/gain/DC, coloring/noise, varying delay, clock/channel
mistakes, clipping, silence/dropouts, ambiguous echoes, calibration, geometry and CLI JSON output.
They validate the measurement software. They do **not** measure the prototype, validate the
external recording instrument, or turn synthetic results into a sub-15 ms hardware claim.
