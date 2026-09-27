"""Gate 0a tooling (low-delay plan Task 0): the torch-free Python step timer and the eligibility report's arithmetic,
support selection and refusals, on synthetic board records (the real ones come from the owner's Pi 5)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import ld_gate0 as G  # noqa: E402
from vaani import audio_contract as ac  # noqa: E402

A10, A9, A8 = ac.ARM_A_IDS
NO_RUNNER = Path("/nonexistent/vaani_ld_run")


def _bench(path: Path, contract: str, max_ms: float, fz=True, inputs=("random", "silent", "lowlevel"), paced=True):
    rows = [{"contract": contract, "state_floats": 1, "input": i, "fz": fz, "paced": paced,
             "resampler": "r1_minphase_kaiser193_v1", "hops": 100000, "warmup": 500, "rt": "{}", "machine": "aarch64",
             "whole_hop": {"n": 100000, "max_ms": max_ms, "p999_ms": max_ms * 0.8, "p99_ms": max_ms * 0.6, "mean_ms": 0.4},
             "step": {"n": 100000, "max_ms": max_ms * 0.7, "p999_ms": 0.5, "p99_ms": 0.4, "mean_ms": 0.3},
             "allocations_after_warmup": 0, "arena_growth_bytes": 0, "hop_budget_ms": 6.0} for i in inputs]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _period(path: Path, ok=True, period=48, seconds=1800.0):
    path.write_text(json.dumps({"alsa": {"period": period}, "queue_periods": 2, "seconds": seconds, "xruns": 0 if ok else 3,
                                "pass": ok, "kernel": "6.12.1-rt aarch64"}))
    return path


def _rep(tmp_path, a_ms=0.55, b_ms=0.6, wake_us=150.0, period_ok=True, conv=0.5, src="section4_upper_estimate"):
    t = {"arm_a": [str(_bench(tmp_path / "a.jsonl", A10, a_ms))],
         "arm_b": [str(_bench(tmp_path / "b.jsonl", ac.ARM_B_ID, b_ms))],
         "arm_r": [str(_bench(tmp_path / "r.jsonl", A10, 1.4))]}
    return G.report(t, wake_us, _period(tmp_path / "p.json", period_ok), conv, src, G.RESERVE_MS, NO_RUNNER, 1000)


def test_resampler_budget_uses_the_d3_group_delay_maximum():
    rs = G.load_resamplers()
    assert rs["R1"]["budget_ms"] == rs["R1"]["pair_group_delay_300_4000_max_ms"] > rs["R1"]["pair_peak_ms"]
    assert rs["R0"]["budget_ms"] == pytest.approx(4.0)
    # arithmetic of Section 4's rows, recomputed from the parts
    b = G.budget(A8, rs["R1"]["budget_ms"], 1.0, 1.0, 0.5)
    assert b["total_ms"] == pytest.approx(8 + rs["R1"]["budget_ms"] + 2 + 0.5 + 0.1)


def test_dproc_rounds_up_to_whole_periods():
    assert G.dproc_periods(0.55, 0.15, 0.2, 1.0) == 1          # 0.9 ms -> 1 period
    assert G.dproc_periods(0.65, 0.15, 0.2, 1.0) == 1          # exactly 1.0 ms stays 1
    assert G.dproc_periods(0.66, 0.15, 0.2, 1.0) == 2
    assert G.dproc_periods(0.66, 0.15, 0.2, 2.0) == 1


def test_full_board_record_selects_by_the_registered_rule(tmp_path):
    rep = _rep(tmp_path)
    assert rep["status"] == "complete" and rep["pending_gate0b"]
    assert rep["arms"]["arm_a"]["dproc_periods"][1.0] == 1 and rep["arms"]["arm_r"]["dproc_periods"][1.0] == 2
    sel = rep["selection"]
    # with the 0.5 ms converter upper estimate, R1's 0.407 ms puts L = 10 ms at 13.007 ms: not eligible; L = 9 ms is
    assert sel["support_contract"] == A9 and "not eligible" in sel["trail"][0]
    assert not sel["arm_b"]["piloted"]
    # a Gate 0b converter measurement (low-latency DAC filter) makes L = 10 ms eligible, and Arm B with it
    rep = _rep(tmp_path, conv=0.12, src="gate0b.json")
    assert rep["selection"]["support_contract"] == A10 and rep["selection"]["arm_b"]["piloted"]


def test_unverified_periods_force_8ms_and_r0_is_never_eligible(tmp_path):
    rep = _rep(tmp_path, period_ok=False)
    assert rep["selection"]["support_contract"] == A8 and not rep["period_test"]["verified_1ms"]
    assert not any(r["eligible"] for r in rep["eligibility"] if r["resampler"] == "R0")
    assert not any(r["eligible"] for r in rep["eligibility"]
                   if r["period_ms"] == 2.0 and ac.get_audio_contract(r["contract"]).support > 128)


def test_slow_mini_p18_is_returned_to_the_owner(tmp_path):
    rep = _rep(tmp_path, a_ms=5.5, wake_us=400.0)
    sel = rep["selection"]
    assert sel["support_contract"] == A8 and sel["mini_p18_fails_8ms"] and "owner" in sel["failure_action"]


def test_missing_measurements_leave_the_report_pending(tmp_path):
    t = {"arm_a": [str(_bench(tmp_path / "a.jsonl", A10, 0.5, fz=False))]}
    rep = G.report(t, None, None, 0.5, "section4_upper_estimate", G.RESERVE_MS, NO_RUNNER, 1000)
    assert rep["status"] == "pending_board" and rep["selection"]["provisional"]
    joined = " ".join(rep["missing"])
    for k in ("cyclictest", "period_test", "arm_b", "arm_r", "FZ set"):
        assert k in joined
    assert rep["schedule"]["status"] == "not_run"


def test_refusals(tmp_path):
    with pytest.raises(ValueError, match="not arm_b"):
        G.measured_max("arm_b", G._timing_rows(_bench(tmp_path / "x.jsonl", A10, 0.5)))
    (tmp_path / "e.json").write_text(json.dumps({"error": "boom"}))
    with pytest.raises(ValueError, match="failed"):
        G._timing_rows(tmp_path / "e.json")
    with pytest.raises(SystemExit):
        G.main(["--reserve-ms", "0.1"])
    with pytest.raises(SystemExit):
        G.main(["--converters-ms", "0.1"])                     # a measured value needs its Gate 0b source


def test_schedule_simulation_through_the_native_runner(tmp_path):
    runner = ROOT / "native/vaani_ld/build-test/vaani_ld_run"
    if not runner.exists():
        pytest.skip("native runner not built (tests/test_low_delay_native.py builds it)")
    rs = G.load_resamplers()
    arms = {"arm_a": {"total_ms": 0.9, "dproc_periods": {1.0: 1, 2.0: 1}}}
    sch = G.schedule(runner, rs, 0.5, arms, 3000)
    assert sch["pass"] and any(r.get("arm") == "arm_a" for r in sch["rows"])
    late = [r for r in sch["rows"] if r["proc"] == "late"]
    assert late and all(r["result"]["late_hops"] > 0 for r in late)


def test_python_step_timer_runs_torch_free(tmp_path):
    import subprocess
    onnx = ROOT / "deploy/dsp_reference/vectors_ld" / A10 / "model.onnx"
    code = ("import sys, json; sys.path.insert(0, 'scripts'); import ld_step_timing as T; "
            f"r = T.main([{str(onnx)!r}, '--seconds', '0.3', '--warmup', '5', '--inputs', 'silent,lowlevel', "
            f"'--out', {str(tmp_path / 't.json')!r}]); "
            "assert 'torch' not in sys.modules, 'torch imported'")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((tmp_path / "t.json").read_text())
    g = rep["graphs"][0]
    assert g["audio_contract"] == A10 and [x["input"] for x in g["runs"]] == ["silent", "lowlevel"]
    assert not rep["reportable"] and g["runs"][0]["step"]["n"] == g["runs"][0]["hops"] == 50
    m = G.measured_max("arm_a", G._timing_rows(tmp_path / "t.json"))
    assert m["basis"].startswith("python only")


def test_loader_bench_kernel_axis_and_limiter_only():
    import bench_loader as B
    cfg = {"dsp": {"limiter": True}, "model": "vaani_fe"}
    assert B.with_kernel(cfg, "numba")["dsp"]["limiter_kernel"] == "numba" and "limiter_kernel" not in cfg["dsp"]
    assert B.with_kernel({"dsp": {}}, "numba") == {"dsp": {}} and B.with_kernel(cfg, None) is cfg
    rows = B.limiter_only(["loop", "numpy"], crops=2, crop_s=0.5)
    assert [r["kernel"] for r in rows] == ["loop", "numpy"] and rows[1]["max_abs_vs_first_kernel"] < 1e-6
