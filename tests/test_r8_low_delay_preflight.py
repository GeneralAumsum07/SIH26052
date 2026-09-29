"""Low-delay r8 configurations, queue and preflight (plan Task 8): the generator's records, the queue's resolution,
gating, decisions and release, the launcher's dry runs, the VAANI_PERF_OPS override and every preflight refusal.
Nothing here trains, downloads or needs a GPU."""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import r8_ld_queue as LQ     # noqa: E402
import r8_preflight as P     # noqa: E402
from vaani import audio_contract as ac   # noqa: E402

BASH = shutil.which("bash")
ARMS = json.loads((REPO / LQ.ARMS_JSON).read_text(encoding="utf-8"))
REC = {Path(a["config"]).stem: a for a in ARMS["arms"]}
SEL = ARMS["gate0"]["support_contract"]


@pytest.fixture(autouse=True)
def strict_validation_for_existing_cases(monkeypatch):
    # Keep coverage of the opt-in validation policy independently of the ungated default.
    monkeypatch.setenv("REQUIRE_TRAINING_VALIDATION", "1")


def test_default_training_has_no_measurement_or_release_gates(tmp_path, monkeypatch):
    monkeypatch.delenv("REQUIRE_TRAINING_VALIDATION")
    q = _queue(tmp_path, status="pending_board", provisional=True)
    rows = q.runnable("all")
    assert {r["name"] for r in rows} == {n for n in NAMES if n in REC}
    assert all(r["status"] == "PENDING" for r in rows)
    assert q.readiness()[0] and not q.gate0_ready()[0]
    q.gate0_path.unlink()
    q = LQ.Queue(REPO, tmp_path / "runs", NAMES, gate0=q.gate0_path)
    assert len(q.runnable("all")) == len(rows)


def test_default_launcher_needs_no_g1_or_full_release(tmp_path, monkeypatch):
    monkeypatch.delenv("REQUIRE_TRAINING_VALIDATION")
    for cmd in ("start", "ld-start"):
        env = dict(DRY_RUN="1", RUNS_DIR=str(tmp_path / cmd), G1_JSON=str(tmp_path / "missing.json"),
                   GATE0_JSON=str(tmp_path / "missing_gate0.json"), LD_GPUS="1", PILOT_HOURS="-1")
        r = _bash(["scripts/run_r8.sh", cmd, "all"], env)
        assert r.returncode == 0, r.stdout + r.stderr
        dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY gpu")]
        assert len(dry) == (3 if cmd == "start" else len([n for n in NAMES if n in REC]))
        assert any(ln.endswith("r8_fe_mini.yaml") for ln in dry)


def _registry():
    """The launcher's registered names, read from scripts/run_r8.sh (LD_P1..LD_P4, LD_FULL)."""
    txt = (REPO / "scripts/run_r8.sh").read_text(encoding="utf-8")
    out = []
    for k in ("LD_P1", "LD_P2", "LD_P3", "LD_P4", "LD_FULL"):
        m = re.search(rf"^{k}=\(([^)]*)\)", txt, re.M)
        out += m.group(1).split()
    return out


NAMES = _registry()


def _gate0(tmp, status="complete", provisional=False, sel=SEL, arm_b=False):
    j = json.loads((REPO / LQ.GATE0).read_text(encoding="utf-8"))
    j["status"] = status
    j["selection"] = dict(j["selection"], provisional=provisional, support_contract=sel,
                          arm_b=dict(piloted=arm_b, reason="test"))
    p = tmp / "g0.json"; p.write_text(json.dumps(j)); return p


def _queue(tmp, **g0):
    return LQ.Queue(REPO, tmp / "runs", NAMES, gate0=_gate0(tmp, **g0))


# --- generator -------------------------------------------------------------------------------------------------------
def test_generator_check_passes_and_every_arm_records_its_budget():
    r = subprocess.run([sys.executable, "scripts/gen_r8_configs.py", "--low-delay", "--check"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for a in ARMS["arms"]:
        for k in ("seed", "scored_exposure_s", "prefix_exposure_s", "optimizer_steps", "batch", "precision",
                  "support_ms", "tiling", "deploy_path", "audio_contract", "contract_hash", "queued"):
            assert k in a, (a["name"], k)
        if a["queued"]:
            assert {"priority", "wave", "kind", "stage"} <= set(a), a["name"]
        assert not any("distil" in k for k in a), a   # D9
    # every Arm A configuration names Gate 0a's selected support; Arm B its own contract
    assert all(a["audio_contract"] == SEL for a in ARMS["arms"] if a["arm"] == "arm_a")
    assert all(a["audio_contract"] == ac.ARM_B_ID for a in ARMS["arms"] if a["arm"] == "arm_b")


def test_every_registered_name_is_a_generated_arm_or_a_wave2_promotion():
    assert len(set(NAMES)) == len(NAMES)
    assert set(NAMES) - set(REC) <= set(LQ.WAVE2), set(NAMES) - set(REC)


# --- queue resolution --------------------------------------------------------------------------------------------------
def test_every_queued_name_resolves_to_one_config_network_contract_and_run_directory(tmp_path):
    q = _queue(tmp_path)
    jobs = [j for j in q.jobs("all") if "config" in j]
    assert len({j["train_dir"] for j in jobs}) == len(jobs)
    by = {j["name"]: j for j in jobs}
    # C0 seeds 0/1 from the legacy directory (shared run directories with the legacy queue), 2-4 from the low-delay one
    assert by["ab1_fe_mini_s0"]["config"] == "configs/retraining/r8_ablations/ab1_fe_mini_s0.yaml"
    assert by["ab1_fe_mini_s2"]["config"] == "configs/retraining/r8_ld_ablations/ab1_fe_mini_s2.yaml"
    assert by["r8_fe_mini"]["config"] == "configs/retraining/r8_fe_mini.yaml"
    assert by["ab1_fe_mini_s0"]["train_dir"].endswith("r8ab1_fe_mini_s0")
    for n in ("ab1_fe_mini_s0", "ab1_fe_mini_s3", "r8_fe_mini"):
        assert by[n]["network"] == "mini" and by[n]["contract"] == ac.LEGACY_ID
    for n in ("ld_a_s0", "ld_s2_native", "ld_p4_kappa2", "r8_ld_fe_mini", "r8_ld_fe_mini_overparam"):
        assert by[n]["network"] == "mini_p18" and by[n]["contract"] == SEL, by[n]
    assert by["ld_r_s0"]["network"] == "mini_df96" and by["ld_b_s0"]["contract"] == ac.ARM_B_ID
    assert by["ld_b_s0"]["network"] == "mini_p32"
    # every resolved config really names the recorded contract
    for j in jobs:
        cfg = yaml.safe_load(open(REPO / j["config"], encoding="utf-8"))
        assert ac.contract_of(cfg["model_cfg"]).audio_contract_id == j["contract"], j["name"]


def test_a_missing_or_ambiguous_configuration_fails_the_queue(tmp_path):
    root = tmp_path / "repo"
    for rel in {r["config"] for r in ARMS["arms"]} | {LQ.ARMS_JSON}:
        (root / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(REPO / rel, root / rel)
    g0 = _gate0(tmp_path)
    q = LQ.Queue(root, tmp_path / "runs", NAMES, gate0=g0)
    assert q.resolve("ld_a_s0")[0] == REC["ld_a_s0"]["config"]
    (root / REC["ld_a_s0"]["config"]).unlink()
    with pytest.raises(LQ.QueueError, match="missing"):
        LQ.Queue(root, tmp_path / "runs", NAMES, gate0=g0).jobs("pilots")
    # a same-named file in a second candidate directory is ambiguous, never a silent substitute
    shutil.copy(REPO / REC["ld_a_s0"]["config"], root / REC["ld_a_s0"]["config"])
    (root / "configs/retraining/ld_a_s0.yaml").write_text("x: 1\n")
    with pytest.raises(LQ.QueueError, match="ambiguous"):
        q.resolve("ld_a_s0")
    with pytest.raises(LQ.QueueError, match="not recorded"):
        q.resolve("ab2_p_s0")
    with pytest.raises(LQ.QueueError, match="twice"):
        LQ.Queue(REPO, tmp_path / "runs", NAMES + ["ld_a_s0"], gate0=g0)


def test_arms_generated_for_another_support_refuse_to_queue(tmp_path):
    other = next(c for c in ac.ARM_A_IDS if c != SEL)
    with pytest.raises(LQ.QueueError, match="regenerate"):
        _queue(tmp_path, sel=other)


# --- gating, decisions, release -------------------------------------------------------------------------------------
def test_nothing_runs_before_gate0_is_complete(tmp_path):
    q = _queue(tmp_path, status="pending_board", provisional=True)
    rows = q.runnable("all")
    assert rows and all(r["status"] == "BLOCKED" for r in rows)
    assert q.claim_next("pilots", 0, 1) is None


def test_arm_b_waits_for_gate0_and_wave2_for_stage2(tmp_path):
    js = {j["name"]: j for j in _queue(tmp_path).jobs("all")}
    for n in ("ld_b_s0", "ld_b_s1", "r8_ld_fe_mini_armb"):
        assert js[n]["status"] == "NOT_ELIGIBLE"
    for n in LQ.WAVE2:
        assert js[n]["status"] == "WAITING"
    js = {j["name"]: j for j in _queue(tmp_path, arm_b=True).jobs("all")}
    assert js["ld_b_s0"]["status"] == "PENDING"


def test_unvalidated_arm_override_is_persistent_and_scoped(tmp_path):
    q = _queue(tmp_path, status="pending_board", provisional=True)
    original = q.gate0_path.read_bytes()
    q.allow_unvalidated_arm("arm_b")
    q = LQ.Queue(REPO, tmp_path / "runs", NAMES, gate0=q.gate0_path)
    rows = {j["name"]: j for j in q.runnable("all")}
    for n in ("ld_b_s0", "ld_b_s1", "ld_b_nhat_s0", "ld_b_nhat_s1"):
        assert rows[n]["status"] == "PENDING"
        assert rows[n]["unvalidated"] and "unvalidated" in rows[n]["why"]
    assert rows["ld_a_s0"]["status"] == "BLOCKED"
    assert rows["r8_ld_fe_mini_armb"]["status"] == "BLOCKED"
    assert not q.gate0_ready()[0]
    assert q.gate0_path.read_bytes() == original
    assert all(j["arm"] in ("arm_b", "arm_b_nhat") for j in q.streams("pilots").values())
    q.decide("stage1", "arm_b")
    q.decide("stage1", "arm_a")
    assert "r8_ld_fe_mini_armb" not in {j["name"] for j in q.runnable("all")}


def test_unvalidated_override_rejects_other_arms(tmp_path):
    with pytest.raises(LQ.QueueError, match="arm_b"):
        _queue(tmp_path).allow_unvalidated_arm("arm_a")


def test_preflight_records_training_only_override(tmp_path):
    q = _queue(tmp_path, status="pending_board", provisional=True)
    q.allow_unvalidated_arm("arm_b")
    rep = P.Report()
    P.check_low_delay(REPO, rep, str(q.gate0_path), set(QUICK), runs_dir=q.runs)
    assert not _fails(rep, "gate0")
    assert any(r["status"] == "WARN" and "arm_b" in r["detail"] for r in rep.rows)


def test_claims_follow_priority_and_are_exclusive(tmp_path):
    q = _queue(tmp_path)
    got = []
    while (j := q.claim_next("pilots", 0, 1)) is not None:
        got.append(j)
    pr = [j["priority"] for j in got]
    assert pr == sorted(pr) and got[0]["name"] == "ab1_fe_mini_s0"
    assert not any(j["full"] for j in got) and len({j["name"] for j in got}) == len(got)
    assert {j["name"] for j in got} == {j["name"] for j in q.jobs("pilots") if j.get("status") == "PENDING"}


def test_decisions_stop_the_ruled_out_recipes(tmp_path):
    q = _queue(tmp_path)
    with pytest.raises(LQ.QueueError, match="not piloted"):
        q.decide("stage1", "arm_b")
    with pytest.raises(LQ.QueueError, match="Arm R"):
        q.decide("stage1", "arm_r")
    with pytest.raises(LQ.QueueError, match="not Stage-2"):
        q.decide("stage2", "ld_p4_kappa1")
    stop = q.decide("stage2", "ld_s2_overparam")
    assert "r8_ld_fe_mini" in stop and "r8_ld_fe_mini_overparam" not in stop
    assert json.loads((tmp_path / "runs/r8_queue/decisions.json").read_text())["stage2"] == ["ld_s2_overparam"]
    js = {j["name"]: j for j in LQ.Queue(REPO, tmp_path / "runs", NAMES, gate0=tmp_path / "g0.json").jobs("all")}
    # wave 2 waits for its generation; the recipe's full run is the early overparam one
    assert js["r8_ld_fe_mini_conf"]["status"] == "NOT_NEEDED" and js["ld_conf_s0"]["status"] == "WAITING"
    assert "--promote ld_s2_overparam" in js["ld_conf_s0"]["why"]
    q2 = LQ.Queue(REPO, tmp_path / "runs2", NAMES, gate0=tmp_path / "g0.json")
    q2.decide("stage2", "none")
    js = {j["name"]: j for j in q2.jobs("all")}
    assert all(js[n]["status"] == "NOT_NEEDED" for n in LQ.WAVE2) and js["r8_ld_fe_mini"]["status"] == "PENDING"


def test_wave2_generated_for_another_recipe_is_refused(tmp_path):
    root = tmp_path / "repo"
    for rel in {r["config"] for r in ARMS["arms"]} | {LQ.ARMS_JSON}:
        (root / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(REPO / rel, root / rel)
    man = json.loads((root / LQ.ARMS_JSON).read_text()); man["promoted"] = ["ld_s2_mrstft05"]
    (root / LQ.ARMS_JSON).write_text(json.dumps(man))
    q = LQ.Queue(root, tmp_path / "runs", NAMES, gate0=_gate0(tmp_path))
    with pytest.raises(LQ.QueueError, match="generated for"):
        q.decide("stage2", "ld_s2_native")


def _ready(tmp, status="pass", age_h=0.0, go=True):
    q = tmp / "runs/r8_queue"; q.mkdir(parents=True, exist_ok=True)
    if go:
        (q / LQ.FULL_GO).write_text("")
    p = q / "preflight_ld.json"; p.write_text(json.dumps(dict(status=status)))
    t = time.time() - age_h * 3600; os.utime(p, (t, t))
    return LQ.Queue(REPO, tmp / "runs", NAMES, gate0=_gate0(tmp))


@pytest.mark.parametrize("kw, ok, why", [
    (dict(), True, "released"),
    (dict(go=False), False, "not authorized"),
    (dict(status="fail"), False, "does not pass"),
    (dict(age_h=LQ.READY_MAX_H + 1), False, "stale"),
])
def test_full_runs_need_authorization_and_fresh_readiness_evidence(tmp_path, kw, ok, why):
    q = _ready(tmp_path, **kw)
    got, msg = q.readiness()
    assert got is ok and why in msg, msg
    fulls = [r for r in q.runnable("full")]
    assert fulls and all((r["status"] == "BLOCKED") is (not ok) for r in fulls)


# --- launcher dry runs ------------------------------------------------------------------------------------------------
def _bash(args, env, cwd=REPO):
    if not BASH:
        pytest.skip("bash unavailable")
    return subprocess.run([BASH, *args], cwd=cwd, env=dict(os.environ, PY=sys.executable, **env),
                          capture_output=True, text=True, timeout=800)


def _g1(tmp):
    p = tmp / "g1.json"
    p.write_text(json.dumps(dict(gate_pass=True, seed=202, bank="bank_r8.npz", param=dict(items=200), room=dict(items=200))))
    return p


@pytest.mark.timeout(900)
def test_launcher_dry_run_selects_only_intended_jobs(tmp_path):
    env = dict(DRY_RUN="1", RUNS_DIR=str(tmp_path / "runs"), G1_JSON=str(_g1(tmp_path)), GATE0_JSON=str(_gate0(tmp_path)),
               LD_GPUS="2", NO_TMUX="1")
    r = _bash(["scripts/run_r8.sh", "ld-start", "pilots"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY gpu")]
    ran = [ln.rsplit("/", 1)[1][:-5] for ln in dry]
    pilots = [n for n in NAMES if n in REC and REC[n]["kind"] != "full" and not REC[n].get("needs_gate0_arm_b")]
    assert sorted(ran) == sorted(pilots) and len(set(ran)) == len(ran), ran
    prio = [int(ln.split(" P", 1)[1].split(":")[0]) for ln in dry]
    assert prio == sorted(prio) and ran[0] == "ab1_fe_mini_s0"
    assert all("VAANI_PERF_OPS='{\"scorer\":\"async\",\"stream\":\"shared\"" in ln for ln in dry)
    assert all(f"nice -n {LQ.CLASS_NICE[p]} " in ln for ln, p in zip(dry, prio))
    assert "DRY scorer" in r.stdout and "DRY stream" in r.stdout
    # the full phase is held until the explicit release and fresh readiness evidence
    f = _bash(["scripts/run_r8.sh", "ld-start", "full"], env)
    assert f.returncode == 0 and not [ln for ln in f.stdout.splitlines() if ln.startswith("DRY gpu")], f.stdout
    assert "not authorized" in f.stdout


@pytest.mark.timeout(900)
def test_launcher_refuses_before_gate0_and_plans_blocked(tmp_path):
    env = dict(RUNS_DIR=str(tmp_path / "runs"), G1_JSON=str(_g1(tmp_path)),
               GATE0_JSON=str(_gate0(tmp_path, status="pending_board", provisional=True)))
    r = _bash(["scripts/run_r8.sh", "ld-start", "pilots"], env)
    assert r.returncode == 1 and "REFUSED" in r.stdout and "Gate 0a" in r.stdout, r.stdout + r.stderr
    p = _bash(["scripts/run_r8.sh", "ld-plan", "all"], env)
    rows = [ln.split("\t") for ln in p.stdout.splitlines() if ln.startswith("gpu")]
    assert rows and all(x[3] == "BLOCKED" for x in rows), p.stdout


def test_launcher_override_dry_run_starts_only_arm_b(tmp_path):
    env = dict(RUNS_DIR=str(tmp_path / "runs"), G1_JSON=str(_g1(tmp_path)),
               GATE0_JSON=str(_gate0(tmp_path, status="pending_board", provisional=True)), LD_GPUS="1")
    r = _bash(["scripts/run_r8.sh", "ld-allow-unvalidated-arm", "arm_b"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = _bash(["scripts/run_r8.sh", "ld-start", "pilots"], dict(env, DRY_RUN="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY gpu")]
    assert {ln.rsplit("/", 1)[1][:-5] for ln in dry} == {"ld_b_s0", "ld_b_s1", "ld_b_nhat_s0", "ld_b_nhat_s1"}
    scorer = "\n".join(ln for ln in r.stdout.splitlines() if ln.startswith("DRY scorer"))   # one per run
    assert "r8_ld_a_s0" not in scorer and "r8_ld_b_s0" in scorer


@pytest.mark.timeout(900)
def test_launcher_decide_writes_stopped_markers(tmp_path):
    env = dict(RUNS_DIR=str(tmp_path / "runs"), G1_JSON=str(_g1(tmp_path)), GATE0_JSON=str(_gate0(tmp_path)))
    r = _bash(["scripts/run_r8.sh", "ld-decide", "stage2", "ld_s2_native"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "stopped r8_ld_fe_mini " in r.stdout and (tmp_path / "runs/r8_ld_fe_mini/STOPPED").exists()
    s = _bash(["scripts/run_r8.sh", "ld-status"], env)
    assert re.search(r"STOPPED\s+r8_ld_fe_mini\s", s.stdout), s.stdout


# --- perf ops override -----------------------------------------------------------------------------------------------
def test_perf_ops_come_from_the_launcher_and_never_touch_numerics(monkeypatch):
    from vaani import train as T
    cfg = yaml.safe_load(open(REPO / "configs/retraining/r8_ld_fe_mini.yaml", encoding="utf-8"))
    num0, ops0 = T.perf_settings(cfg)
    monkeypatch.setenv("VAANI_PERF_OPS", json.dumps(dict(scorer="async", stream="shared", priority=4)))
    num, ops = T.perf_settings(cfg)
    assert num == num0 and ops == dict(ops0, scorer="async", stream="shared", priority=4)
    monkeypatch.setenv("VAANI_PERF_OPS", "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        T.perf_settings(cfg)
    monkeypatch.setenv("VAANI_PERF_OPS", json.dumps(dict(scorer="sometimes")))
    with pytest.raises(ValueError, match="scorer"):
        T.perf_settings(cfg)
    monkeypatch.setenv("VAANI_PERF_OPS", json.dumps(dict(render="gpu")))   # numerics cannot enter through ops
    with pytest.raises(ValueError, match="unknown perf.ops"):
        T.perf_settings(cfg)


# --- preflight -----------------------------------------------------------------------------------------------------
QUICK = ("ld_gen", "evidence")   # the generator check runs in its own test; the bench evidence is box-only


def _pf(tmp, skip=QUICK, **g0):
    rep = P.Report()
    P.check_low_delay(REPO, rep, str(_gate0(tmp, **g0)), set(skip))
    return rep


def _fails(rep, check=None):
    return [r for r in rep.failed() if check is None or r["check"] == check]


def _gru_parity(tmp, monkeypatch):
    # the committed queue trains the fused GRU: the box's ld_bench stage writes this record before preflight
    (tmp / "perf").mkdir(exist_ok=True)
    (tmp / "perf" / "gru_parity.json").write_text(json.dumps({"pass": True}))
    monkeypatch.setattr(P, "LD_PERF", str(tmp / "perf"))


def test_preflight_passes_the_committed_low_delay_queue(tmp_path, monkeypatch):
    _gru_parity(tmp_path, monkeypatch)
    rep = _pf(tmp_path)
    assert not rep.failed(), rep.failed()
    checks = {r["check"] for r in rep.rows}
    assert {"queued", "gate0", "contract", "spec62", "numerics", "perf_parity", "versions"} <= checks


def test_preflight_fails_without_a_complete_gate0_record(tmp_path):
    rep = P.Report(); P.check_low_delay(REPO, rep, str(tmp_path / "none.json"), set(QUICK))
    assert _fails(rep, "gate0") and "missing" in _fails(rep, "gate0")[0]["detail"]
    assert _fails(_pf(tmp_path, status="pending_board", provisional=True), "gate0")


def test_preflight_fails_when_arm_a_is_not_the_selected_support(tmp_path):
    other = next(c for c in ac.ARM_A_IDS if c != SEL)
    rep = _pf(tmp_path, sel=other)
    assert "regenerate" in _fails(rep, "gate0")[0]["detail"]
    bad = _fails(rep, "contract")
    assert bad and all(REC_BY_RUN[r["what"]]["arm"] == "arm_a" for r in bad) and "Gate 0a selects" in bad[0]["detail"]


REC_BY_RUN = {a["name"]: a for a in ARMS["arms"]}


def _patched(monkeypatch, edit):
    meta = json.loads(json.dumps(ARMS)); arms = {a["name"]: a for a in meta["arms"]}
    edit(arms)
    monkeypatch.setattr(P, "ld_records", lambda root: (meta, arms))


@pytest.mark.parametrize("edit, check, needle", [
    (lambda a: a["r8_ld_a_s0"].update(contract_hash="0" * 16), "contract", "contract hash"),
    (lambda a: a["r8_ld_a_s0"].update(audio_contract=ac.ARM_B_ID), "contract", "recorded"),
    (lambda a: a["r8_ld_a_s0"].update(support_ms=10.0), "contract", "support"),
    (lambda a: a["r8_ld_r_s0"].update(deployable=True), "spec62", "mmac_per_s"),
    (lambda a: a["r8_ld_a_s0"].update(config="configs/retraining/r8_ld_ablations/nope.yaml"), "queued", "missing"),
])
def test_preflight_refusals_on_altered_records(tmp_path, monkeypatch, edit, check, needle):
    _patched(monkeypatch, edit)
    bad = _fails(_pf(tmp_path), check)
    assert bad and needle in bad[0]["detail"], _pf(tmp_path).rows


def test_preflight_fails_an_altered_contract_in_the_registry(tmp_path, monkeypatch):
    import dataclasses
    real = ac.get_audio_contract

    def fake(cid):
        c = real(cid)
        return dataclasses.replace(c, window_hash="0" * len(c.window_hash)) if cid == SEL else c
    monkeypatch.setattr(ac, "get_audio_contract", fake)
    bad = _fails(_pf(tmp_path), "contract")
    assert bad and all(REC_BY_RUN[r["what"]]["audio_contract"] == SEL for r in bad)
    assert "hash" in bad[0]["detail"]


def test_preflight_fails_mismatched_numerics_and_unrecorded_gpu_paths(tmp_path, monkeypatch):
    real = P.train_perf

    def mixed(cfg):
        p = real(cfg)
        if cfg["name"] == "r8_ld_s2_native":
            p["numerics"] = dict(p["numerics"], cuda_graph=not p["numerics"]["cuda_graph"])
        return p
    monkeypatch.setattr(P, "train_perf", mixed)
    assert "differs" in _fails(_pf(tmp_path), "numerics")[0]["what"]

    def gpu(cfg):
        p = real(cfg); p["numerics"] = dict(p["numerics"], render="gpu", gru_kernel="fused"); return p
    monkeypatch.setattr(P, "train_perf", gpu)
    monkeypatch.setattr(P, "LD_PERF", str(tmp_path / "perf"))
    bad = {Path(r["what"]).name for r in _fails(_pf(tmp_path), "perf_parity")}
    assert bad == {"render_parity.json", "render_g1.json", "gru_parity.json"}
    (tmp_path / "perf").mkdir()
    for f, k in (("render_parity.json", "pass"), ("render_g1.json", "gate_pass"), ("gru_parity.json", "pass")):
        (tmp_path / "perf" / f).write_text(json.dumps({k: True}))
    assert not _fails(_pf(tmp_path), "perf_parity")
    (tmp_path / "perf/render_g1.json").write_text(json.dumps(dict(gate_pass=False)))
    assert [Path(r["what"]).name for r in _fails(_pf(tmp_path), "perf_parity")] == ["render_g1.json"]


def test_preflight_fails_unpinned_loss_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "_version", lambda d: {"torch_pesq": "0.1.1", "torchaudio": "2.12.0"}[d])
    assert len(_fails(_pf(tmp_path), "versions")) == 2
    monkeypatch.setattr(P, "_version", lambda d: None)
    assert len(_fails(_pf(tmp_path), "versions")) == 2
    import importlib
    real = importlib.import_module

    def imp(name, *a):
        if name == "torch_pesq":
            raise ImportError("no torch_pesq")
        return real(name, *a)
    monkeypatch.setattr(importlib, "import_module", imp)
    rep = P.Report(); P.check_imports(rep)
    assert [r["what"] for r in rep.failed()] == ["torch_pesq"]


def test_preflight_fails_a_mismatched_checkpoint(tmp_path):
    ck = tmp_path / "init.pt"; ck.write_bytes(b"weights")
    rep = P.Report()
    P.check_init(tmp_path, {"c": dict(init_from="init.pt", init_sha256="0" * 64)}, rep, {})
    assert rep.failed() and "want" in rep.failed()[0]["detail"]


def test_preflight_fails_missing_bench_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "LD_EVIDENCE", (str(tmp_path / "loader_bench_box.json"),))
    assert _fails(_pf(tmp_path, skip=("ld_gen",)), "evidence")
    (tmp_path / "loader_bench_box.json").write_text("{}")
    assert not _fails(_pf(tmp_path, skip=("ld_gen",)), "evidence")


def test_preflight_cli_writes_the_readiness_evidence_the_queue_reads(tmp_path, monkeypatch):
    out = tmp_path / "ready.json"
    _gru_parity(tmp_path, monkeypatch)
    monkeypatch.setenv("LD_READY_JSON", str(out))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    skip = "manifests,banks,val,heldout,init,imports,cuda,disk,g1,ld_gen,evidence"
    rc = P.main(["--low-delay", "--skip", skip, "--gate0", str(_gate0(tmp_path))])
    j = json.loads(out.read_text())
    assert rc == 0 and j["status"] == "pass" and j["low_delay"] and "evidence" in j["skipped"]
    assert set(j["configs"]) == {a["config"] for a in ARMS["arms"] if a.get("queued")}
    rc = P.main(["--low-delay", "--skip", skip, "--gate0", str(_gate0(tmp_path, status="pending_board"))])
    assert rc == 1 and json.loads(out.read_text())["status"] == "fail"


# --- box setup and bench ---------------------------------------------------------------------------------------------
@pytest.mark.timeout(900)
def test_box_setup_benches_and_preflights_the_low_delay_queue():
    env = {k: "" for k in ("MIRROR_HF_REPO", "HF_TOKEN", "RIR_BANK_URL")}
    d = _bash(["scripts/r8_box_setup.sh", "--dry-run"], env)
    assert d.returncode == 0, d.stdout + d.stderr
    assert "DRY ld_bench" in d.stdout and "DRY preflight-ld" in d.stdout and "--low-delay" in d.stdout, d.stdout


@pytest.mark.parametrize("cfg", ["configs/retraining/r8_ld_fe_mini.yaml", "configs/retraining/r8_ld_ablations/ld_r_s0.yaml",
                                 "configs/retraining/r8_ld_ablations/ld_b_nhat_s0.yaml"])
def test_step_time_bench_trains_the_low_delay_framing(cfg, monkeypatch):
    import bench_loader as B
    c = yaml.safe_load(open(REPO / cfg, encoding="utf-8"))
    n = 8000; clean = torch.randn(n) * 0.05
    item = {"mix": torch.stack([clean + torch.randn(n) * 0.02, clean * 0.7]), "clean": clean, "meta": {},
            "avail": torch.ones(n, dtype=torch.uint8)}
    if c["model_cfg"]["inputs"] == "pr_nhat":   # the loader workers' decoupled-cadence NLMS output
        item["n_hat"] = clean * 0.1
    monkeypatch.setattr(B, "build_dataset", lambda c: [item] * 2)
    monkeypatch.setattr(B, "collate", lambda its: {k: torch.stack([i[k] for i in its]) for k in its[0] if k != "meta"}
                        | {"meta": [{} for _ in its]})
    r = B.step_time(c, 2, 1, 0, device="cpu")
    assert r["step_s_median"] > 0 and r["gpu"] == "cpu" and r["params"] > 0
