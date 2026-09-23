"""Live noise suppression: two microphones in, enhanced speech out, one 16 ms hop at a time.

    # live, ICS-43434 pair on the Pi's I2S bus, USB sound card for the headphones
    python scripts/capture_loop.py --device hw:0,0 --out-device plughw:1,0

    # milestone 1, no microphones: feed a recorded pair through the ALSA loopback card
    #   (terminal 1)  aplay -D hw:Loopback,0,0 pair_48k.wav
    python scripts/capture_loop.py --device hw:Loopback,1,0 --format S16_LE

    # offline, any OS, no audio hardware: the same block-by-block code path on a file
    python scripts/capture_loop.py --in-wav mix.wav --out-wav enhanced.wav

Signal path, per hop: capture 768 frames at 48 kHz (or 256 at 16 kHz) -> channel pick (primary / reference) -> input
gain -> streaming 3:1 decimation -> `vaani.live.StreamEngine` (limiter, blocking matrix, NLMS, 18 features, controller,
ONNX cascade, overlap-add) -> streaming 1:3 interpolation -> output gain -> playback, mono duplicated to both ears.

Latency: 32 ms algorithmic (deploy/CONTRACT.md) + 2 x 2 ms resampler group delay + whatever ALSA buffers; the
period is one hop and the buffers are kept to a few periods. Stats on stderr report processing time against the
16 ms budget.

Audio I/O goes through `arecord` / `aplay` (alsa-utils) pipes: no Python audio package to build on the board, and
both mics come in as one interleaved stereo stream on ONE clock - the property the coherence features need. Never
point --device at two separate USB microphones.

Press Enter during a live run to toggle enhanced <-> bypass (the raw primary, delayed by the same hop so the switch
is a fair A/B). --record-dir keeps the 16 kHz stereo capture and the enhanced output, so a live session can be
scored offline afterwards (Hardware_build.md, acceptance test 7).
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani import live           # noqa: E402  (numpy + onnxruntime only)
from vaani.dsp import nlms       # noqa: E402

HOP = live.HOP
BUDGET_MS = 1000 * HOP / live.SR
FORMATS = {"S16_LE": ("<i2", 2 ** 15), "S32_LE": ("<i4", 2 ** 31)}


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


class Stats:
    """Rolling per-window numbers for the stderr line; totals for the summary."""

    def __init__(self):
        self.reset(); self.all_ms = []; self.late_total = 0; self.frames_total = 0

    def reset(self):
        self.ms, self.gate, self.lim, self.burst, self.pin, self.rin = [], [], 0, 0, [], []

    def add(self, eng_last, prim16, ref16):
        self.ms.append(eng_last["ms"]); self.gate.append(eng_last["gate"])
        self.lim += eng_last["limiter"]; self.burst += eng_last["burst"]
        self.pin.append(prim16); self.rin.append(ref16)
        self.all_ms.append(eng_last["ms"]); self.frames_total += 1
        self.late_total += eng_last["ms"] > BUDGET_MS

    def line(self, t, mode):
        m = np.array(self.ms)
        p, r = np.concatenate(self.pin), np.concatenate(self.rin)
        s = (f"[{t:7.1f}s] {mode:8s} proc {m.mean():5.2f} ms mean / {np.percentile(m, 99):5.2f} p99 / {m.max():5.2f} max"
             f" (budget {BUDGET_MS:.0f})  RTF {m.mean() / BUDGET_MS:4.2f}  in: prim {db(p):6.1f} ref {db(r):6.1f} dBFS"
             f"  gate {np.mean(self.gate):4.2f}  limiter {self.lim:3d}  bursts {self.burst:3d}")
        self.reset()
        return s


def build(args):
    cfg = live.load_model_config(args.config)
    eng = live.StreamEngine(args.onnx, cfg["controller_on"], cfg["dsp"], threads=args.threads)
    return eng, cfg


def run_file(args):
    x, sr = live.read_wav(args.in_wav)
    if x.shape[0] < 2:
        sys.exit(f"{args.in_wav}: need a 2-channel (primary, reference) file, got {x.shape[0]} channel(s)")
    if sr not in (16000, 48000):
        sys.exit(f"{args.in_wav}: sample rate {sr}; this path takes 16000 or 48000")
    eng, _ = build(args)
    up = sr // live.SR
    blk = HOP * up
    x = x[[args.primary, args.reference]] * 10 ** (args.input_gain_db / 20)
    dec = live.Decimate3(2) if up == 3 else None
    itp = live.Interpolate3(1) if up == 3 else None
    n_blk = x.shape[1] // blk
    out, cap16, stats = [], [], Stats()
    for j in range(n_blk):
        b = x[:, j * blk:(j + 1) * blk]
        b16 = dec(b) if dec else b.astype(np.float32)
        y = eng.process(b16[0], b16[1])
        cap16.append(b16); stats.add(eng.last, b16[0], b16[1])
        y = y * 10 ** (args.output_gain_db / 20)
        out.append(itp(y[None])[0] if itp else y)
    y = np.concatenate(out)
    # drop the one-hop lag (plus the resamplers' group delay) so the output lines up with the input file
    lag = blk + (2 * (len(live.lowpass_fir()) - 1) // 2 if up == 3 else 0)
    y = np.concatenate([y[lag:], np.zeros(lag, np.float32)])
    live.write_wav(args.out_wav, y, sr)
    m = np.array(stats.all_ms[1:])
    print(f"{args.in_wav} -> {args.out_wav}: {n_blk} hops at {sr} Hz, proc {m.mean():.2f} ms mean / "
          f"{np.percentile(m, 99):.2f} p99 per 16 ms hop (RTF {m.mean() / BUDGET_MS:.3f}), "
          f"NLMS kernel {'numba' if nlms._HAVE_NUMBA else 'pure-python'}", file=sys.stderr)


def read_exact(stream, n):
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = stream.readinto(view[got:])
        if not k:
            return None
        got += k
    return bytes(buf)


def run_live(args):
    for tool in ("arecord", "aplay"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found (sudo apt install alsa-utils). Without ALSA, use --in-wav/--out-wav.")
    if not nlms._HAVE_NUMBA and not args.allow_slow_dsp:
        sys.exit("numba is not installed, so the NLMS runs as the pure-Python reference (~10 ms per 16 ms hop on a desktop "
                 "core, several times that on a Pi): the live loop cannot keep up. `pip install numba`, or pass "
                 "--allow-slow-dsp to try anyway.")
    eng, cfg = build(args)
    # warm up on a throwaway engine: numba's first call compiles the NLMS kernel (up to a second) and ORT's first
    # run allocates. Doing that on the live engine would overrun the capture buffer at the very start.
    warm = live.StreamEngine(args.onnx, cfg["controller_on"], cfg["dsp"], threads=args.threads)
    for _ in range(4):
        warm.process(*(np.random.default_rng(0).standard_normal((2, HOP)).astype(np.float32) * 1e-3))
    del warm
    up = args.rate // live.SR
    if args.rate % live.SR or up not in (1, 3):
        sys.exit("--rate must be 16000 or 48000")
    dtype, scale = FORMATS[args.format]
    bps = np.dtype(dtype).itemsize
    blk_in, blk_out = HOP * up, HOP * (args.out_rate // live.SR)
    cap = subprocess.Popen(["arecord", "-q", "-D", args.device, "-t", "raw", "-f", args.format, "-r", str(args.rate),
                            "-c", str(args.channels), f"--period-size={blk_in}", f"--buffer-size={blk_in * args.buffer_periods}"],
                           stdout=subprocess.PIPE, bufsize=0)
    play = subprocess.Popen(["aplay", "-q", "-D", args.out_device, "-t", "raw", "-f", "S16_LE", "-r", str(args.out_rate),
                             "-c", str(args.out_channels), f"--period-size={blk_out}",
                             f"--buffer-size={blk_out * args.buffer_periods}"], stdin=subprocess.PIPE, bufsize=0)
    dec = live.Decimate3(2) if up == 3 else None
    itp = live.Interpolate3(1) if args.out_rate == 48000 else None
    g_in, g_out = 10 ** (args.input_gain_db / 20), 10 ** (args.output_gain_db / 20)

    state = {"bypass": args.bypass, "stop": False}
    if sys.stdin.isatty():
        def toggler():
            for _ in sys.stdin:
                state["bypass"] = not state["bypass"]
                print(f"-> {'BYPASS (raw primary)' if state['bypass'] else 'ENHANCED'}", file=sys.stderr)
        threading.Thread(target=toggler, daemon=True).start()
    signal.signal(signal.SIGINT, lambda *_: state.__setitem__("stop", True))
    signal.signal(signal.SIGTERM, lambda *_: state.__setitem__("stop", True))

    print(f"model {args.onnx} (controller {'on' if cfg['controller_on'] else 'off'}); capture {args.device} "
          f"{args.channels}ch {args.format} @ {args.rate}; primary=ch{args.primary} reference=ch{args.reference}; "
          f"playback {args.out_device} @ {args.out_rate}. Enter toggles enhanced/bypass, Ctrl-C stops.", file=sys.stderr)
    play.stdin.write(np.zeros(blk_out * args.out_channels * 2, np.uint8).tobytes() * 2)   # prefill: no underrun at start

    stats, rec_in, rec_out = Stats(), [], []
    prev_prim = np.zeros(HOP, np.float32)            # bypass output is the raw primary delayed by the engine's hop
    t_start = t_line = time.time()
    try:
        while not state["stop"]:
            raw = read_exact(cap.stdout, blk_in * args.channels * bps)
            if raw is None:
                print("capture stream ended (arecord exited - check the device name and format)", file=sys.stderr); break
            x = np.frombuffer(raw, dtype).reshape(blk_in, args.channels).T.astype(np.float32) / scale
            x = x[[args.primary, args.reference]] * g_in
            x16 = dec(x) if dec else x
            y = eng.process(x16[0], x16[1])            # always run, so a toggle back to enhanced is already warm
            stats.add(eng.last, x16[0], x16[1])
            out16 = prev_prim.copy() if state["bypass"] else y
            prev_prim = x16[0].copy()
            if args.record_dir:
                rec_in.append(x16.copy()); rec_out.append(out16.copy())
            o = itp(out16[None] * g_out)[0] if itp else out16 * g_out
            pcm = (np.clip(o, -1, 1) * 32767).astype("<i2")
            play.stdin.write(np.repeat(pcm[:, None], args.out_channels, axis=1).tobytes())
            now = time.time()
            if now - t_line >= args.stats_every:
                print(stats.line(now - t_start, "BYPASS" if state["bypass"] else "ENHANCED"), file=sys.stderr); t_line = now
            if args.seconds and now - t_start >= args.seconds:
                break
    except BrokenPipeError:
        print("playback stream closed (aplay exited - check --out-device)", file=sys.stderr)
    finally:
        for p in (cap, play):
            try:
                if p is play: p.stdin.close()
                p.terminate(); p.wait(timeout=2)
            except Exception:
                p.kill()
    m = np.array(stats.all_ms[1:]) if len(stats.all_ms) > 1 else np.array([np.nan])
    print(f"stopped after {stats.frames_total} hops ({stats.frames_total * BUDGET_MS / 1000:.1f} s): proc {np.nanmean(m):.2f} ms "
          f"mean / {np.nanpercentile(m, 99):.2f} p99; {stats.late_total} hops over the {BUDGET_MS:.0f} ms budget", file=sys.stderr)
    if args.record_dir and rec_in:
        d = Path(args.record_dir); d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        live.write_wav(d / f"{stamp}_capture16k.wav", np.concatenate(rec_in, axis=1), live.SR)
        live.write_wav(d / f"{stamp}_output16k.wav", np.concatenate(rec_out), live.SR)
        print(f"recorded to {d}/{stamp}_capture16k.wav (primary, reference) and _output16k.wav (one hop behind)", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="deploy/r7/cascade.onnx", help="exported cascade graph")
    ap.add_argument("--config", default="deploy/r7/model_config.json",
                    help="DSP configuration the weights were trained behind (vaani.live.write_model_config)")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads (contract budget: 1)")
    ap.add_argument("--primary", type=int, default=0, help="capture channel of the near-mouth mic (SEL->GND = left = 0)")
    ap.add_argument("--reference", type=int, default=1, help="capture channel of the rear mic (SEL->3V3 = right = 1)")
    ap.add_argument("--input-gain-db", type=float, default=0.0,
                    help="applied to both mics before everything else; aim for primary speech around -25 dBFS "
                         "(the training range is -32..-18) using the stats line")
    ap.add_argument("--output-gain-db", type=float, default=0.0)
    f = ap.add_argument_group("file mode (no audio hardware)")
    f.add_argument("--in-wav", help="2-channel WAV at 16 or 48 kHz (primary, reference)")
    f.add_argument("--out-wav", help="enhanced mono WAV, same rate, aligned to the input")
    l = ap.add_argument_group("live mode (ALSA)")
    l.add_argument("--device", default="hw:0,0", help="capture device: the I2S mic card, or hw:Loopback,1,0")
    l.add_argument("--format", default="S32_LE", choices=sorted(FORMATS),
                   help="capture sample format; ICS-43434 over I2S delivers 24-bit data in S32_LE")
    l.add_argument("--rate", type=int, default=48000, help="capture rate: 48000 (decimated here) or 16000")
    l.add_argument("--channels", type=int, default=2)
    l.add_argument("--out-device", default="default", help="playback device, e.g. plughw:1,0 for the USB sound card")
    l.add_argument("--out-rate", type=int, default=48000, choices=(16000, 48000))
    l.add_argument("--out-channels", type=int, default=2, help="the mono output is duplicated to this many channels")
    l.add_argument("--buffer-periods", type=int, default=4, help="ALSA buffer size in 16 ms periods (latency vs xruns)")
    l.add_argument("--bypass", action="store_true", help="start in bypass (raw primary); Enter toggles")
    l.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    l.add_argument("--stats-every", type=float, default=5.0)
    l.add_argument("--record-dir", help="save the 16 kHz capture and output WAVs here on exit")
    l.add_argument("--allow-slow-dsp", action="store_true", help="run live without numba (will likely fall behind)")
    args = ap.parse_args(argv)
    for p in (args.onnx, args.config):
        if not Path(p).exists():
            sys.exit(f"{p} not found. Export the cascade first (see deploy/CONTRACT.md).")
    if args.in_wav or args.out_wav:
        if not (args.in_wav and args.out_wav):
            sys.exit("file mode needs both --in-wav and --out-wav")
        run_file(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
