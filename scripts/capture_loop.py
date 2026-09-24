"""Live noise suppression: two microphones in, enhanced speech out, one 16 ms hop at a time.

    # live, ICS-43434 pair on the Pi's I2S bus, enhanced speech to a LISTENER's device (not the wearer's ears)
    python scripts/capture_loop.py --device hw:0,0 --output-route far-end --out-device plughw:1,0

    # live, no playback at all (the default route): stats line + a bounded recording to score offline
    python scripts/capture_loop.py --device hw:0,0 --record-dir rec/ --seconds 60

    # milestone 1, no microphones: feed a recorded pair through the ALSA loopback card
    #   (terminal 1)  aplay -D hw:Loopback,0,0 pair_48k.wav
    python scripts/capture_loop.py --device hw:Loopback,1,0 --format S16_LE --output-route far-end

    # offline, any OS, no audio hardware: the same block-by-block code path on a file
    python scripts/capture_loop.py --in-wav mix.wav --out-wav enhanced.wav

Signal path, per hop: capture 768 frames at 48 kHz (or 256 at 16 kHz) -> channel pick (primary / reference) -> input
gain -> streaming 3:1 decimation -> `vaani.live.StreamEngine` (limiter, blocking matrix, NLMS, 18 features, controller,
ONNX cascade, overlap-add) -> streaming 1:3 interpolation -> output gain -> playback, mono duplicated to both ears.

Output routing (--output-route). The enhanced signal is the WEARER'S OWN VOICE with the noise removed; it is meant for
the far end (radio uplink, a listener's headset). Playing it into the wearer's own ears adds a 32+ ms delayed copy of
their own voice (disorienting, and an acoustic feedback path through a leaky earcup), so that is never the default:
  off       (default) no playback; stats and --record-dir only
  far-end   play to --out-device, which must be the listener's / uplink device
  wearer    play to --out-device even though it is the wearer's headset (explicit opt-in; a warning is printed)

Latency: 32 ms algorithmic (deploy/CONTRACT.md) + 2 x 2 ms resampler group delay + whatever ALSA buffers; the
period is one hop and the buffers are kept to a few periods.

Transport. A reader thread pulls periods from `arecord` into a `vaani.live.BoundedHopQueue` (--queue-hops). Overload
policy: when the queue is full the OLDEST hop is dropped and counted (the engine is told: `StreamEngine.skip`); when
the backlog reaches --bypass-depth the loop passes the raw primary through for that hop (counted as bypass) instead
of running the model, so latency stays bounded and old audio is never replayed. The health line reports whole-
iteration time (decode, resample, engine, interpolate, playback write) against the 16 ms hop, how far processing
trails capture, dropped / bypassed hops, reference-invalid hops, guard events, and the ALSA overrun / underrun
counts parsed from arecord / aplay stderr (alsa-utils "overrun!!!" / "underrun!!!" lines; parser not yet exercised
on the board).

Audio I/O goes through `arecord` / `aplay` (alsa-utils) pipes: no Python audio package to build on the board, and
both mics come in as one interleaved stereo stream on ONE clock - the property the coherence features need. Never
point --device at two separate USB microphones. The board path is numpy + onnxruntime only (no torch).

Press Enter during a live run to toggle enhanced <-> bypass (the raw primary, delayed by the same hop so the switch
is a fair A/B). --record-dir streams the 16 kHz stereo capture and the output to disk as the run goes (bounded by
--record-max-s), so a live session can be scored offline afterwards (Hardware_build.md, acceptance test 7).
"""
from __future__ import annotations

import argparse
import collections
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
ROUTES = ("off", "far-end", "wearer")


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def resampler_delay(up: int) -> int:
    """Group delay of the 3:1 + 1:3 FIR pair at the capture rate (0 at 16 kHz)."""
    return 2 * (len(live.lowpass_fir()) - 1) // 2 if up == 3 else 0


class Stats:
    """Per-window numbers for the stderr line; bounded ring (all_ms) plus exact running totals for the summary."""

    def __init__(self, ring: int = 4096):
        self.reset()
        self.all_ms = collections.deque(maxlen=ring)          # engine ms, last `ring` hops (plan 2.10: bounded)
        self.iter_ms = collections.deque(maxlen=ring)         # whole-iteration ms, last `ring` hops
        self.late_total = self.iter_late_total = self.frames_total = 0
        self.iter_max = 0.0

    def reset(self):
        self.ms, self.it, self.gate, self.lim, self.burst, self.pin, self.rin = [], [], [], 0, 0, [], []

    def add(self, eng_last, prim16, ref16):
        self.ms.append(eng_last["ms"]); self.gate.append(eng_last["gate"])
        self.lim += eng_last["limiter"]; self.burst += eng_last["burst"]
        self.pin.append(prim16); self.rin.append(ref16)
        self.all_ms.append(eng_last["ms"]); self.frames_total += 1
        self.late_total += eng_last["ms"] > BUDGET_MS

    def add_iter(self, ms):
        self.it.append(ms); self.iter_ms.append(ms)
        self.iter_late_total += ms > BUDGET_MS; self.iter_max = max(self.iter_max, ms)

    def line(self, t, mode, extra=""):
        m = np.array(self.ms) if self.ms else np.array([np.nan])
        it = np.array(self.it) if self.it else np.array([np.nan])
        p = np.concatenate(self.pin) if self.pin else np.zeros(1)
        r = np.concatenate(self.rin) if self.rin else np.zeros(1)
        s = (f"[{t:7.1f}s] {mode:8s} proc {np.nanmean(m):5.2f} ms mean / {np.nanpercentile(m, 99):5.2f} p99 / "
             f"{np.nanmax(m):5.2f} max, iter {np.nanpercentile(it, 99):5.2f} p99 (budget {BUDGET_MS:.0f})"
             f"  RTF {np.nanmean(m) / BUDGET_MS:4.2f}  in: prim {db(p):6.1f} ref {db(r):6.1f} dBFS"
             f"  gate {np.mean(self.gate) if self.gate else float('nan'):4.2f}  limiter {self.lim:3d}"
             f"  bursts {self.burst:3d}{extra}")
        self.reset()
        return s


class Recorder:
    """--record-dir: capture (16 kHz stereo) and output (16 kHz mono) streamed to disk hop by hop, stopped after
    `max_s` seconds so a forgotten session cannot fill the card."""

    def __init__(self, d, max_s: float):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.cap_path, self.out_path = d / f"{stamp}_capture16k.wav", d / f"{stamp}_output16k.wav"
        self.cap, self.out = live.WavWriter(self.cap_path, live.SR, 2), live.WavWriter(self.out_path, live.SR, 1)
        self.max_n, self.full = int(max_s * live.SR), False

    def write(self, x16, y16):
        if self.full:
            return
        if self.cap.n + HOP > self.max_n:
            self.full = True
            print(f"--record-max-s reached: recording stopped at {self.cap.n / live.SR:.1f} s", file=sys.stderr)
            return
        self.cap.write(x16); self.out.write(y16)

    def close(self):
        self.cap.close(); self.out.close()
        print(f"recorded {self.cap.n / live.SR:.1f} s to {self.cap_path} (primary, reference) and {self.out_path} "
              "(one hop behind)", file=sys.stderr)


def xrun_kind(line: str):
    """'overrun' / 'underrun' for an alsa-utils xrun report line (aplay.c prints e.g. 'overrun!!! (at least
    1.234 ms long)'), else None."""
    s = line.lower()
    if "overrun" in s:
        return "overrun"
    if "underrun" in s:
        return "underrun"
    return None


def watch_stderr(proc, counts: collections.Counter, tag: str) -> threading.Thread:
    """Count xruns on a child's stderr; pass every other line through, prefixed."""
    def run():
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            k = xrun_kind(line)
            if k:
                counts[k] += 1
            elif line:
                print(f"[{tag}] {line}", file=sys.stderr)
    t = threading.Thread(target=run, daemon=True); t.start()
    return t


def build(args, cfg=None):
    cfg = cfg or live.load_model_config(args.config)
    # a VaaniFE config (kind vaani_fe) gets its step-graph backend and tier profile; r7 builds as before
    fe = {"kind": cfg["kind"], "profile": cfg.get("profile")} if cfg.get("kind", "cascade") != "cascade" else {}
    eng = live.StreamEngine(args.onnx, cfg["controller_on"], cfg["dsp"], threads=args.threads,
                            onnx_sha256=cfg.get("onnx_sha256"), allow_hash_mismatch=args.allow_hash_mismatch,
                            guards=True if args.guards else None, **fe)
    return eng, cfg


def ref_validity(args):
    """-> callable(ref16 hop) -> bool for the chosen --ref-validity policy."""
    if args.ref_validity == "always":
        return lambda r: True
    if args.ref_validity == "never":
        return lambda r: False
    mon = live.ChannelMonitor()
    return mon.update


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
    n = x.shape[1]
    # tail flush: the output lags by one hop plus the resamplers' delay, and the last hop's overlap-add half only
    # comes out on the next call, so feed zeros past the end until every input sample has been emitted
    lag = blk + resampler_delay(up)
    n_blk = -(-(n + lag) // blk)
    x = np.concatenate([x, np.zeros((2, n_blk * blk - n), np.float32)], axis=1)
    dec = live.Decimate3(2) if up == 3 else None
    itp = live.Interpolate3(1) if up == 3 else None
    valid = ref_validity(args) if args.ref_validity != "auto" else (lambda r: True)   # file default: always
    rec = Recorder(args.record_dir, args.record_max_s) if args.record_dir else None
    out, stats = [], Stats()
    for j in range(n_blk):
        b = x[:, j * blk:(j + 1) * blk]
        b16 = dec(b) if dec else b.astype(np.float32)
        y = eng.process(b16[0], b16[1], valid(b16[1]))
        stats.add(eng.last, b16[0], b16[1])
        if rec:
            rec.write(b16, y)
        y = y * 10 ** (args.output_gain_db / 20)
        out.append(itp(y[None])[0] if itp else y)
    if rec:
        rec.close()
    y = np.concatenate(out)[lag:lag + n]              # aligned to the input, same length
    live.write_wav(args.out_wav, y, sr)
    s = eng.telemetry.summary()
    print(f"{args.in_wav} -> {args.out_wav}: {n_blk} hops at {sr} Hz ({n} samples in, {len(y)} out), proc "
          f"{s['mean_ms']:.2f} ms mean / {s['p99_ms']:.2f} p99 per 16 ms hop (RTF {s['mean_ms'] / BUDGET_MS:.3f}), "
          f"NLMS kernel {'numba' if nlms._HAVE_NUMBA else 'pure-python'}; fallback events {s['fallback_events']}",
          file=sys.stderr)


def read_exact(stream, n):
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = stream.readinto(view[got:])
        if not k:
            return None
        got += k
    return bytes(buf)


def run_live(args):
    need = ("arecord", "aplay") if args.output_route != "off" else ("arecord",)
    for tool in need:
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found (sudo apt install alsa-utils). Without ALSA, use --in-wav/--out-wav.")
    if not nlms._HAVE_NUMBA and not args.allow_slow_dsp:
        sys.exit("numba is not installed, so the NLMS runs as the pure-Python reference (~10 ms per 16 ms hop on a desktop "
                 "core, several times that on a Pi): the live loop cannot keep up. `pip install numba`, or pass "
                 "--allow-slow-dsp to try anyway.")
    if args.output_route == "wearer":
        print("WARNING: --output-route wearer plays the wearer's own enhanced voice into their own headset, 32+ ms "
              "late. Use it only for a bench check with the headset off the head.", file=sys.stderr)
    eng, cfg = build(args)
    # warm up on a throwaway engine: numba's first call compiles the NLMS kernel (up to a second) and ORT's first
    # run allocates. Doing that on the live engine would overrun the capture buffer at the very start.
    warm, _ = build(args, cfg)
    for _ in range(4):
        warm.process(*(np.random.default_rng(0).standard_normal((2, HOP)).astype(np.float32) * 1e-3))
    del warm
    up = args.rate // live.SR
    if args.rate % live.SR or up not in (1, 3):
        sys.exit("--rate must be 16000 or 48000")
    dtype, scale = FORMATS[args.format]
    bps = np.dtype(dtype).itemsize
    blk_in, blk_out = HOP * up, HOP * (args.out_rate // live.SR)
    xruns = collections.Counter()
    # no -q: arecord/aplay report xruns on stderr, which watch_stderr counts
    cap = subprocess.Popen(["arecord", "-D", args.device, "-t", "raw", "-f", args.format, "-r", str(args.rate),
                            "-c", str(args.channels), f"--period-size={blk_in}", f"--buffer-size={blk_in * args.buffer_periods}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    watch_stderr(cap, xruns, "arecord")
    play = None
    if args.output_route != "off":
        play = subprocess.Popen(["aplay", "-D", args.out_device, "-t", "raw", "-f", "S16_LE", "-r", str(args.out_rate),
                                 "-c", str(args.out_channels), f"--period-size={blk_out}",
                                 f"--buffer-size={blk_out * args.buffer_periods}"],
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        watch_stderr(play, xruns, "aplay")
    dec = live.Decimate3(2) if up == 3 else None
    itp = live.Interpolate3(1) if args.out_rate == 48000 else None
    g_in, g_out = 10 ** (args.input_gain_db / 20), 10 ** (args.output_gain_db / 20)
    valid = ref_validity(args)

    state = {"bypass": args.bypass, "stop": False}
    if sys.stdin.isatty():
        def toggler():
            for _ in sys.stdin:
                state["bypass"] = not state["bypass"]
                print(f"-> {'BYPASS (raw primary)' if state['bypass'] else 'ENHANCED'}", file=sys.stderr)
        threading.Thread(target=toggler, daemon=True).start()
    signal.signal(signal.SIGINT, lambda *_: state.__setitem__("stop", True))
    signal.signal(signal.SIGTERM, lambda *_: state.__setitem__("stop", True))

    q = live.BoundedHopQueue(args.queue_hops)
    nbytes = blk_in * args.channels * bps

    def reader():
        while not state["stop"]:
            raw = read_exact(cap.stdout, nbytes)
            if raw is None:
                break
            q.push(raw)
        q.close()
    threading.Thread(target=reader, daemon=True).start()

    print(f"model {args.onnx} (controller {'on' if cfg['controller_on'] else 'off'}, guards "
          f"{'on' if args.guards else 'off'}); capture {args.device} {args.channels}ch {args.format} @ {args.rate}; "
          f"primary=ch{args.primary} reference=ch{args.reference} (validity {args.ref_validity}); output route "
          f"{args.output_route}" + (f" -> {args.out_device} @ {args.out_rate}" if play else "") +
          ". Enter toggles enhanced/bypass, Ctrl-C stops.", file=sys.stderr)
    if play:
        play.stdin.write(np.zeros(blk_out * args.out_channels * 2, np.uint8).tobytes() * 2)   # prefill: no underrun at start

    stats = Stats()
    rec = Recorder(args.record_dir, args.record_max_s) if args.record_dir else None
    prev_prim = np.zeros(HOP, np.float32)            # bypass output is the raw primary delayed by the engine's hop
    overload = dropped = 0
    t_start = t_line = time.time()
    try:
        while not state["stop"]:
            item = q.pop(timeout=1.0)
            if item is None:
                if q.closed:
                    print("capture stream ended (arecord exited - check the device name and format)", file=sys.stderr)
                    break
                continue
            raw, gap = item
            t_it = time.perf_counter()
            for _ in range(gap):                     # hops the queue dropped: no audio, tell the engine
                eng.skip(); dropped += 1
            x = np.frombuffer(raw, dtype).reshape(blk_in, args.channels).T.astype(np.float32) / scale
            x = x[[args.primary, args.reference]] * g_in
            x16 = dec(x) if dec else x
            if q.depth() >= args.bypass_depth:       # overload: pass the raw primary, keep the stream moving
                eng.skip(x16[0], x16[1]); overload += 1
                y = prev_prim.copy()
            else:
                y = eng.process(x16[0], x16[1], valid(x16[1]))   # always run, so a toggle back is already warm
                stats.add(eng.last, x16[0], x16[1])
            out16 = prev_prim.copy() if state["bypass"] else y
            prev_prim = x16[0].copy()
            if rec:
                rec.write(x16, out16)
            if play:
                o = itp(out16[None] * g_out)[0] if itp else out16 * g_out
                pcm = (np.clip(o, -1, 1) * 32767).astype("<i2")
                play.stdin.write(np.repeat(pcm[:, None], args.out_channels, axis=1).tobytes())
            stats.add_iter((time.perf_counter() - t_it) * 1000)
            now = time.time()
            if now - t_line >= args.stats_every:
                trail = (now - t_start) - q.pushed * BUDGET_MS / 1000      # >0 and growing: capture is stalling
                extra = (f"  queue {q.depth()}/{q.capacity} dropped {q.dropped_hops} bypassed {overload}"
                         f"  ref-invalid {eng.ref_invalid_hops}  xrun o/u {xruns['overrun']}/{xruns['underrun']}"
                         f"  capture-lag {trail * 1000:+.0f} ms  events {dict(eng.telemetry.event_counts)}")
                print(stats.line(now - t_start, "BYPASS" if state["bypass"] else "ENHANCED", extra), file=sys.stderr)
                t_line = now
            if args.seconds and now - t_start >= args.seconds:
                break
    except BrokenPipeError:
        print("playback stream closed (aplay exited - check --out-device)", file=sys.stderr)
    finally:
        state["stop"] = True
        for p in (cap, play):
            if p is None:
                continue
            try:
                if p is play: p.stdin.close()
                p.terminate(); p.wait(timeout=2)
            except Exception:
                p.kill()
        if rec:
            rec.close()
    s = eng.telemetry.summary()
    it = np.array(stats.iter_ms) if stats.iter_ms else np.array([np.nan])
    print(f"stopped after {stats.frames_total} processed hops ({q.pushed * BUDGET_MS / 1000:.1f} s captured): engine "
          f"{s['mean_ms']:.2f} ms mean / {s['p99_ms']:.2f} p99 / {s['max_ms']:.2f} max, {s['deadline_misses']} over "
          f"{BUDGET_MS:.0f} ms; whole iteration p99 {np.nanpercentile(it, 99):.2f} (last {len(it)} hops) / max "
          f"{stats.iter_max:.2f}, {stats.iter_late_total} over; queue dropped {q.dropped_hops} hops "
          f"({q.dropped_samples} samples), bypassed {overload}; xruns overrun {xruns['overrun']} underrun "
          f"{xruns['underrun']}; fallback events {s['fallback_events']}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="deploy/r7/cascade.onnx", help="exported graph: r7 cascade or a VaaniFE step graph")
    ap.add_argument("--config", default="deploy/r7/model_config.json",
                    help="DSP configuration the weights were trained behind (vaani.live.write_model_config); its "
                         "onnx_sha256 is checked against --onnx")
    ap.add_argument("--allow-hash-mismatch", action="store_true",
                    help="run even if --onnx does not match the config's onnx_sha256 (prints a warning)")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads (contract budget: 1)")
    ap.add_argument("--primary", type=int, default=0, help="capture channel of the near-mouth mic (SEL->GND = left = 0)")
    ap.add_argument("--reference", type=int, default=1, help="capture channel of the rear mic (SEL->3V3 = right = 1)")
    ap.add_argument("--ref-validity", default="auto", choices=("auto", "always", "never"),
                    help="reference availability: auto = live ChannelMonitor (silent / railed mic -> dropout policy, "
                         "vaani.live.StreamEngine) in live mode, always-valid in file mode; always; never")
    ap.add_argument("--guards", action="store_true",
                    help="plan 11.3 runtime guards: zero a duplicated-mono reference, fall back when the output "
                         "vanishes under speech (vaani/guards.py). Off by default")
    ap.add_argument("--input-gain-db", type=float, default=0.0,
                    help="applied to both mics before everything else; aim for primary speech around -25 dBFS "
                         "(the training range is -32..-18) using the stats line")
    ap.add_argument("--output-gain-db", type=float, default=0.0)
    ap.add_argument("--record-dir", help="stream the 16 kHz capture and the output WAVs here as the run goes")
    ap.add_argument("--record-max-s", type=float, default=3600.0, help="stop recording after this many seconds")
    f = ap.add_argument_group("file mode (no audio hardware)")
    f.add_argument("--in-wav", help="2-channel WAV at 16 or 48 kHz (primary, reference)")
    f.add_argument("--out-wav", help="enhanced mono WAV, same rate and length, aligned to the input")
    l = ap.add_argument_group("live mode (ALSA)")
    l.add_argument("--device", default="hw:0,0", help="capture device: the I2S mic card, or hw:Loopback,1,0")
    l.add_argument("--format", default="S32_LE", choices=sorted(FORMATS),
                   help="capture sample format; ICS-43434 over I2S delivers 24-bit data in S32_LE")
    l.add_argument("--rate", type=int, default=48000, help="capture rate: 48000 (decimated here) or 16000")
    l.add_argument("--channels", type=int, default=2)
    l.add_argument("--output-route", default="off", choices=ROUTES,
                   help="where the enhanced speech goes: off (default: no playback), far-end (--out-device is the "
                        "listener's / uplink device), wearer (the wearer's own headset: explicit opt-in, bench only). "
                        "The output is the wearer's own voice, so it never plays into their ears by default")
    l.add_argument("--out-device", default="default", help="playback device for --output-route far-end / wearer")
    l.add_argument("--out-rate", type=int, default=48000, choices=(16000, 48000))
    l.add_argument("--out-channels", type=int, default=2, help="the mono output is duplicated to this many channels")
    l.add_argument("--buffer-periods", type=int, default=4, help="ALSA buffer size in 16 ms periods (latency vs xruns)")
    l.add_argument("--queue-hops", type=int, default=8,
                   help="capture->engine queue capacity in hops; when full the oldest hop is dropped and counted")
    l.add_argument("--bypass-depth", type=int, default=4,
                   help="backlog (hops) at which a hop is passed through raw instead of enhanced, to catch up")
    l.add_argument("--bypass", action="store_true", help="start in bypass (raw primary); Enter toggles")
    l.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    l.add_argument("--stats-every", type=float, default=5.0)
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
