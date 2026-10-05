"""Prepare, calibrate and analyze external synchronized acoustic recordings.

See docs/acoustic_latency.md for hardware placement and reporting boundaries.
An input/output pair saved by capture_loop.py is NOT an external measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vaani.acoustic_latency import AnalysisConfig, analyze_capture, inspect_isolation, make_stimulus  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strict JSON is intentional: NaN must never masquerade as a measured delay.
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def protect_inputs(args):
    """Keep generated artifacts from replacing evidence, including hardlinks.

    Resolving paths catches relative/symlink aliases; samefile catches distinct
    names linked to the same existing file. Perform this before ANY write.
    """
    if args.command == "prepare":
        if args.out.suffix.lower() != ".wav":
            raise ValueError("stimulus output must end in .wav; its manifest uses .json")
        inputs = [args.speech_wav]
        outputs = [args.out, args.out.with_suffix(".json")]
    else:
        if args.out.suffix.lower() != ".json":
            raise ValueError("measurement reports must end in .json")
        inputs = [args.recording] + [getattr(args, name, None) for name in
                                    ("calibration_json", "isolation_json", "onnx", "config")]
        outputs = [args.out]
    inputs = [p for p in inputs if p is not None]
    for i, output in enumerate(outputs):
        for original in inputs + outputs[:i]:
            if (output.resolve() == original.resolve()
                    or (output.exists() and original.exists() and output.samefile(original))):
                raise ValueError(f"output would overwrite an input or another artifact: {output}")


def capture_options(parser):
    parser.add_argument("recording", type=Path, help="one WAV from an independent synchronized recorder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recorder-description", required=True,
                        help="recorder/interface and unchanged recording settings; must match calibration")
    parser.add_argument("--synchronized-stereo", action="store_true", required=True,
                        help="confirm independent measurement microphones share ONE recorder clock")
    parser.add_argument("--reference-channel", type=int, default=0)
    parser.add_argument("--response-channel", type=int, default=1)
    parser.add_argument("--window-s", type=float, default=1)
    parser.add_argument("--step-s", type=float, default=1)
    parser.add_argument("--warmup-s", type=float, default=2)
    parser.add_argument("--max-delay-ms", type=float, default=None)
    parser.add_argument("--min-correlation", type=float, default=0.35)
    parser.add_argument("--peak-margin", type=float, default=0.05)
    parser.add_argument("--notes", default="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="build a repeated real-speech probe and timing manifest")
    prepare.add_argument("--speech-wav", type=Path, required=True)
    prepare.add_argument("--speech-channel", type=int, default=0)
    prepare.add_argument("--speech-start-s", type=float, default=0)
    prepare.add_argument("--speech-seconds", type=float, default=None)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--repeats", type=int, default=10)
    prepare.add_argument("--gap-s", type=float, default=2)
    prepare.add_argument("--lead-s", type=float, default=3)
    prepare.add_argument("--tail-s", type=float, default=3)
    prepare.add_argument("--target-dbfs", type=float, default=-25)
    calibrate = sub.add_parser("calibrate", help="measure channel skew with the measurement mics co-located")
    capture_options(calibrate)
    isolation = sub.add_parser("isolation", help="check source leakage with prototype output physically muted")
    capture_options(isolation)
    isolation.add_argument("--output-muted", action="store_true", required=True,
                           help="confirm only the source plays and placement/settings match the experiment")
    analyze = sub.add_parser("analyze", help="measure acoustic input-to-output arrival delay")
    capture_options(analyze)
    analyze.add_argument("--model-label", required=True, help="e.g. r7, r8_fe_mini, r8_ld_arm_a")
    analyze.add_argument("--transport", required=True, choices=("wired", "bluetooth", "other"))
    analyze.add_argument("--onnx", type=Path, help="optional exact graph used by the prototype")
    analyze.add_argument("--config", type=Path, help="optional matching graph sidecar")
    analyze.add_argument("--min-delay-ms", type=float, default=2,
                         help="near-zero leakage rejection; does not restrict the correlation search")
    analyze.add_argument("--threshold-ms", type=float, default=15)
    analyze.add_argument("--isolation-json", type=Path,
                         help="output-muted control from the same recorder, channels and placement")
    analyze.add_argument("--expected-min-system-ms", type=float,
                         help="declared algorithmic lower bound; known model labels use contract defaults")
    offset = analyze.add_mutually_exclusive_group()
    offset.add_argument("--calibration-json", type=Path)
    offset.add_argument("--channel-offset-ms", type=float, help="independently measured signed channel skew")
    analyze.add_argument("--geometry-correction-ms", type=float,
                         help="measured external-path correction; explicit 0 requires co-located boundaries")
    analyze.add_argument("--source-to-input-m", type=float)
    analyze.add_argument("--source-to-reference-m", type=float)
    analyze.add_argument("--speaker-to-response-m", type=float)
    analyze.add_argument("--sound-speed-m-s", type=float, default=343)
    args = parser.parse_args(argv)
    try:
        protect_inputs(args)
        # Reuse the repository's PCM/float WAV reader. Imports stay lazy so CLI
        # help and the pure numerical estimator need no model runtime startup.
        from vaani.live import read_wav, write_wav, verify_onnx
        if args.command == "prepare":
            source_hash = digest(args.speech_wav)
            x, sr = read_wav(args.speech_wav)
            if not 0 <= args.speech_channel < len(x):
                raise ValueError("selected speech channel is absent")
            if args.speech_start_s < 0 or (args.speech_seconds is not None and args.speech_seconds <= 0):
                raise ValueError("speech start must be nonnegative and duration positive")
            start = int(round(args.speech_start_s * sr))
            stop = None if args.speech_seconds is None else start + int(round(args.speech_seconds * sr))
            y, report = make_stimulus(x[args.speech_channel, start:stop], sr,
                                     repeats=args.repeats, gap_s=args.gap_s, lead_s=args.lead_s,
                                     tail_s=args.tail_s, target_dbfs=args.target_dbfs)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_wav(args.out, y, sr)
            report.update(source=str(args.speech_wav), source_sha256=source_hash,
                          speech_channel=args.speech_channel, speech_start_s=args.speech_start_s,
                          stimulus=str(args.out), stimulus_sha256=digest(args.out), seconds=len(y) / sr)
            write_report(args.out.with_suffix(".json"), report)
            print(f"Saved speech probe: {args.out} ({len(y) / sr:.1f} seconds)")
            return 0
        if not args.recorder_description.strip():
            raise ValueError("describe the actual common-clock recorder and its settings")
        x, sr = read_wav(args.recording)
        is_calibration = args.command == "calibrate"
        is_analysis = args.command == "analyze"
        offset_ms, geometry_ms = None, None
        calibration = None
        if is_analysis:
            offset_ms = args.channel_offset_ms
            if args.calibration_json:
                calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
                if (calibration.get("schema") != "vaani.acoustic_latency/1"
                        or calibration.get("quantity") != "recorder_channel_offset"
                        or calibration.get("status") != "ok"):
                    raise ValueError("calibration must be a successful channel-offset measurement")
                if (calibration["sample_rate_hz"] != sr
                        or calibration["recorder_description"] != args.recorder_description
                        or calibration["reference_channel"] != args.reference_channel
                        or calibration["response_channel"] != args.response_channel):
                    raise ValueError("calibration recorder, sample rate and channel order must match")
                offset_ms = calibration["acoustic_delay_ms"]["median"]
            distances = (args.source_to_input_m, args.source_to_reference_m, args.speaker_to_response_m)
            geometry_ms = args.geometry_correction_ms
            if any(d is not None for d in distances):
                if geometry_ms is not None or any(d is None for d in distances):
                    raise ValueError("supply all three distances OR a measured geometry correction")
                if not all(0 <= d < float("inf") for d in distances) or not 0 < args.sound_speed_m_s < float("inf"):
                    raise ValueError("distances must be finite/nonnegative and sound speed finite/positive")
                # Reference hears the source after its own propagation delay;
                # subtracting that path is essential, rather than subtracting
                # only the response-mic distance or all acoustic distances.
                geometry_ms = 1000 * (distances[0] - distances[1] + distances[2]) / args.sound_speed_m_s
        cfg = AnalysisConfig(window_s=args.window_s, step_s=args.step_s, warmup_s=args.warmup_s,
                             max_delay_ms=args.max_delay_ms if args.max_delay_ms is not None else (20 if is_calibration else 1000),
                             min_delay_ms=args.min_delay_ms if is_analysis else 0,
                             min_correlation=args.min_correlation, peak_margin=args.peak_margin,
                             threshold_ms=args.threshold_ms if is_analysis else 15,
                             calibration=is_calibration)
        isolation_control, isolation_ok = None, False
        if is_analysis and args.isolation_json:
            isolation_control = json.loads(args.isolation_json.read_text(encoding="utf-8"))
            if (isolation_control.get("schema") != "vaani.acoustic_isolation/1"
                    or not isolation_control.get("output_muted")):
                raise ValueError("isolation JSON must come from an output-muted source-only control")
            if (isolation_control["sample_rate_hz"] != sr
                    or isolation_control["recorder_description"] != args.recorder_description
                    or isolation_control["reference_channel"] != args.reference_channel
                    or isolation_control["response_channel"] != args.response_channel):
                raise ValueError("isolation recorder, sample rate and channel order must match")
            if (isolation_control["settings"]["max_delay_ms"] < cfg.max_delay_ms
                    or isolation_control["settings"]["min_correlation"] > cfg.min_correlation):
                raise ValueError("isolation search/quality policy must cover the analysis policy")
            isolation_ok = isolation_control["status"] == "ok"
        if args.command == "isolation":
            report = inspect_isolation(x, sr, config=cfg, reference_channel=args.reference_channel,
                                       response_channel=args.response_channel)
            report["output_muted"] = True
        else:
            report = analyze_capture(x, sr, config=cfg, reference_channel=args.reference_channel,
                                     response_channel=args.response_channel, channel_offset_ms=offset_ms,
                                     geometry_correction_ms=geometry_ms, setup_verified=args.synchronized_stereo,
                                     isolation_verified=isolation_ok)
        report.update(recording=str(args.recording), recording_sha256=digest(args.recording),
                      recorder_description=args.recorder_description,
                      recording_kind="independent_external_synchronized_channels",
                      created_utc=datetime.now(timezone.utc).isoformat(), notes=args.notes,
                      command=list(sys.argv[1:] if argv is None else argv))
        if is_analysis:
            report.update(model_label=args.model_label, transport=args.transport,
                          calibration_json=str(args.calibration_json) if args.calibration_json else None,
                          calibration_sha256=digest(args.calibration_json) if args.calibration_json else None,
                          isolation_json=str(args.isolation_json) if args.isolation_json else None,
                          isolation_sha256=digest(args.isolation_json) if args.isolation_json else None,
                          geometry_distances_m=dict(zip(("source_to_input", "source_to_reference", "speaker_to_response"), distances)),
                          sound_speed_m_s=args.sound_speed_m_s)
            if isolation_control is not None and not isolation_ok:
                report["status"] = "inconclusive"
                report["threshold_assessment"]["supported_for_this_recording"] = False
                report["warnings"].append("The supplied output-muted isolation control failed.")
            expected_min = args.expected_min_system_ms
            if expected_min is None:
                expected_min = {"r7": 32, "r8": 32, "r8_fe_mini": 32,
                                "r8_ld_arm_a": 8, "r8_ld_arm_b": 10}.get(args.model_label)
            if expected_min is not None and not 0 <= expected_min < float("inf"):
                raise ValueError("expected algorithmic minimum must be finite and nonnegative")
            report["expected_min_system_ms"] = expected_min
            corrected = report["corrected_system_delay_ms"]
            if (expected_min is not None and corrected is not None
                    and corrected["min"] + report["sample_period_ms"] < expected_min):
                report["status"] = "inconclusive"
                report["threshold_assessment"]["supported_for_this_recording"] = False
                report["warnings"].append("Measured peak contradicts the declared algorithmic minimum; check leakage, bypass and channel placement.")
            if (args.onnx is None) != (args.config is None):
                raise ValueError("graph provenance needs both --onnx and its --config sidecar")
            if args.onnx:
                sidecar = json.loads(args.config.read_text(encoding="utf-8"))
                if not sidecar.get("onnx_sha256"):
                    raise ValueError("model sidecar lacks its bound graph hash")
                verify_onnx(args.onnx, sidecar["onnx_sha256"])
                report.update(onnx=str(args.onnx), onnx_sha256=digest(args.onnx),
                              config=str(args.config), config_sha256=digest(args.config))
        write_report(args.out, report)
        if args.command == "isolation":
            print(f"{report['status']}: coherent source leakage in {report['coherent_leak_windows']}/{report['active_windows']} active windows")
        else:
            print(f"{report['status']}: {report['accepted_windows']}/{report['active_windows']} active windows accepted")
            print("Acoustic delay (ms):", report["acoustic_delay_ms"])
        if is_analysis:
            print("Corrected system delay (ms):", report["corrected_system_delay_ms"])
        print("Saved:", args.out)
        return 0 if report["status"] == "ok" else 2
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        print(f"acoustic_latency: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
