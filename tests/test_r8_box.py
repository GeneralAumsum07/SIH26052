"""r8 rental-box scripts: preflight checks on a tiny synthetic layout, run_r8.sh dry run, mirror staging on a fake
tree, and r8_box_setup.sh's up-front credential check. Nothing here downloads, installs or trains."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import r8_preflight as P   # noqa: E402

BASH = shutil.which("bash")
V2 = {"tail_share": 0.4}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _layout(tmp):
    """Smallest tree the preflight accepts: one full config, one manifest of two decodable 16 kHz files, a bank listed
    in r8_banks.json, a val split with its recomputed hash, the held-out list, a passing G1 file."""
    r = tmp / "box"
    for d in ("configs/retraining/r8_ablations", "configs/data", "data/manifests", "data/raw/c", "data/rirs",
              "data/eval_r2/val/b", "results_r2/r8/data_gates/box_g1"):
        (r / d).mkdir(parents=True, exist_ok=True)
    for i in range(2):
        sf.write(str(r / f"data/raw/c/{i}.flac"), np.zeros(1600, "float32"), 16000)
    pd.DataFrame(dict(source_id=["c:0", "c:1"], corpus="c", kind="noise", group_id=["g0", "g1"], speaker_id="",
                      path=["data\\raw\\c\\0.flac", "data/raw/c/1.flac"], duration_s=0.1, licence="x",
                      split=["train", "val"], sha1="", noise_class="")).to_parquet(r / "data/manifests/c.parquet")
    (r / "data/rirs/bank_x.npz").write_bytes(b"bank")
    (r / "data/rirs/bank_x.rt60.npy").write_bytes(b"side")
    (r / P.BANKS_JSON).write_text(json.dumps({"bank_x": {
        "file": "data/rirs/bank_x.npz", "sha256": _sha(r / "data/rirs/bank_x.npz"),
        "sidecars": {"bank_x.rt60.npy": _sha(r / "data/rirs/bank_x.rt60.npy")}, "asset": "bank_x.npz", "built_with": "t"}}))
    (r / "configs/data/r8_heldout_exclude.json").write_text(json.dumps(
        {"sources": {"c": {"manifest": "c.parquet", "source_ids": ["c:1"]}}}))
    (r / "data/eval_r2/val/b/000.json").write_text("{}")
    for s in (".mix.wav", ".clean.wav"):
        (r / f"data/eval_r2/val/b/000{s}").write_bytes(b"")
    val_hash = hashlib.sha1(b"{}").hexdigest()[:12]
    cfg = dict(name="r8_fe_mini", batch_size=2, data=dict(
        bank="data/rirs/bank_x.npz", manifests=["data/manifests/c.parquet"],
        exclude_groups_file="configs/data/r8_heldout_exclude.json", mix=dict(version=2, v2=dict(V2))),
        val=dict(eval_root="data/eval_r2"))
    (r / P.FULL[0]).write_text(yaml.safe_dump(cfg))
    g1 = dict(gate_pass=True, seed=202, bank="bank_x.npz", v2_overrides=dict(V2), scene_overrides={},
              param=dict(items=200), room=dict(items=200))
    (r / P.G1_JSON).write_text(json.dumps(g1))
    return r, val_hash


def _run(r, val_hash, *extra):
    out = r / "runs/pf.json"
    rc = P.main(["--root", str(r), "--skip", "imports,cuda", "--need-gb", "0", "--json", str(out),
                 "--val-hash", f"data/eval_r2/val={val_hash}", *extra])
    return rc, {(x["check"], x["status"]) for x in json.loads(out.read_text())["rows"]}, json.loads(out.read_text())["rows"]


def test_preflight_passes_on_a_complete_layout(tmp_path):
    r, h = _layout(tmp_path)
    rc, st, rows = _run(r, h)
    assert rc == 0, rows
    assert {c for c, s in st if s == "ok"} >= {"manifests", "banks", "val", "heldout", "disk", "g1"}


@pytest.mark.parametrize("breakit, check", [
    (lambda r: (r / "data/raw/c/1.flac").unlink(), "manifests"),
    (lambda r: (r / "data/raw/c/0.flac").write_bytes(b"not audio"), "manifests"),
    (lambda r: (r / "data/rirs/bank_x.npz").write_bytes(b"other"), "banks"),
    (lambda r: (r / "data/rirs/bank_x.rt60.npy").unlink(), "banks"),
    (lambda r: (r / "data/eval_r2/val/b/000.clean.wav").unlink(), "val"),
    (lambda r: (r / "configs/data/r8_heldout_exclude.json").unlink(), "heldout"),
    (lambda r: (r / P.G1_JSON).unlink(), "g1"),
])
def test_preflight_fails_each_broken_input(tmp_path, breakit, check):
    r, h = _layout(tmp_path)
    breakit(r)
    rc, st, rows = _run(r, h)
    assert rc == 1 and (check, "FAIL") in st, rows


@pytest.mark.parametrize("edit", [
    dict(gate_pass=False),
    dict(param=dict(items=48)),
    dict(bank="bank_r3.npz"),
    dict(v2_overrides={"tail_share": 0.25}),          # gated a different mixer than the full config trains
    dict(scene_overrides={"p_near": 0.9}),           # train.py cannot apply scene overrides
])
def test_g1_check_refuses_a_gate_that_does_not_cover_the_training_mixer(tmp_path, edit):
    r, h = _layout(tmp_path)
    j = json.loads((r / P.G1_JSON).read_text()); j.update(edit); (r / P.G1_JSON).write_text(json.dumps(j))
    rc, st, rows = _run(r, h)
    assert rc == 1 and ("g1", "FAIL") in st, rows


def test_bank_plan_marks_trained_banks_and_sidecars_needed(tmp_path):
    r, _ = _layout(tmp_path)
    t = json.loads((r / P.BANKS_JSON).read_text())
    t["other"] = dict(file="data/rirs/other.npz", sha256="0" * 64, sidecars={}, asset="other.npz")
    (r / P.BANKS_JSON).write_text(json.dumps(t))
    rows, warn = P.bank_plan(r, P.load_cfgs(r, [P.FULL[0]]))
    need = {f: n for f, _, _, n in rows}
    assert need == {"data/rirs/bank_x.npz": "need", "data/rirs/bank_x.rt60.npy": "need", "data/rirs/other.npz": "opt"}
    assert not warn


def test_bank_plan_falls_back_to_the_published_bank_r3_hash(tmp_path):
    r, _ = _layout(tmp_path)
    (r / P.BANKS_JSON).unlink()
    c = yaml.safe_load((r / P.FULL[0]).read_text()); c["data"]["bank"] = "data/rirs/bank_r3.npz"
    (r / P.FULL[0]).write_text(yaml.safe_dump(c))
    rows, warn = P.bank_plan(r, P.load_cfgs(r, [P.FULL[0]]))
    assert rows == [("data/rirs/bank_r3.npz", P.KNOWN_BANKS["data/rirs/bank_r3.npz"], "bank_r3.npz", "need")]
    assert any("absent" in w for w in warn)


def test_g1_command_gates_the_full_configs_own_mixer(tmp_path):
    r, _ = _layout(tmp_path)
    cmd = P.g1_command(r, P.load_cfgs(r, [P.FULL[0]]), 202, 200, "out", py="py")
    assert cmd[cmd.index("--v2") + 1] == json.dumps(V2, sort_keys=True)
    assert cmd[cmd.index("--bank") + 1] == "data/rirs/bank_x.npz" and "--scene" not in cmd


def _datasets_yaml(r):
    (r / P.DATASETS_YAML).write_text(yaml.safe_dump({"datasets": {
        "c": dict(access="login", credentials=["C_KEY"], manifest="data/manifests/c.parquet", needed_by=[]),
        "late": dict(access="direct", credentials=["LATE_KEY"], manifest="data/manifests/late.parquet",
                     needed_by=["r8_fe_mini"]),
        "spare": dict(access="direct", credentials=[], manifest="data/manifests/spare.parquet", needed_by=[])}}))
    (r / "configs/retraining/r8_ablations/ab1_fe_mini_s0.yaml").write_text(yaml.safe_dump(
        {"data": {"manifests": ["data/manifests/c.parquet"]}}))


def test_fetch_order_puts_the_queue_heads_datasets_first(tmp_path):
    r, _ = _layout(tmp_path)
    _datasets_yaml(r)
    first, rest = P.fetch_order(r)
    assert first == ["c"] and rest == ["late", "spare"]


def test_env_check_names_missing_credentials_without_values(tmp_path):
    r, _ = _layout(tmp_path)
    _datasets_yaml(r)
    lines, block = P.env_check(r, env={"MIRROR_HF_REPO": "u/r", "HF_TOKEN": "secret", "RIR_BANK_URL": "x", "LATE_KEY": ""})
    text = "\n".join(lines)
    assert block and "MISSING C_KEY" in text and "WARN LATE_KEY" in text and "secret" not in text
    lines, block = P.env_check(r, env={"MIRROR_HF_REPO": "u/r", "HF_TOKEN": "t", "RIR_BANK_URL": "x", "C_KEY": "k"})
    assert not block


def test_mem_summary_reads_a_bench_memwatch_log(tmp_path):
    # root 100 with two DataLoader workers (ppid 100) and one grandchild that is not a worker
    log = tmp_path / "bench_mem_loader.log"
    log.write_text("\n".join([
        "# root 100; lines: ...",
        "10 mem 200000000", "10 proc 100 1 4000000 3000000 2500000 python",
        "10 proc 101 100 3000000 1000000 500000 python", "10 proc 102 100 3100000 1100000 600000 python",
        "10 proc 103 101 50000 50000 50000 sh",
        "15 mem 150000000", "15 proc 101 100 3500000 1500000 900000 python", ""]), encoding="utf-8")
    s = P.mem_summary(log)
    kb = lambda v: round(v * 1024 / 1e9, 3)   # noqa: E731
    assert s["samples"] == 2 and s["root_pid"] == "100"
    assert s["mem_available_gb"] == dict(first=kb(200000000), min=kb(150000000))
    assert s["main"]["rss_gb_max"] == kb(4000000)
    w = s["worker"]
    assert w["n_max"] == 2 and w["rss_gb_max"] == kb(3500000) and w["private_gb_max"] == kb(900000)
    assert w["pss_gb_sum_max"] == kb(2100000) and w["private_gb_sum_max"] == kb(1100000)
    assert s["mem_available_drop_gb"] == round(kb(200000000) - kb(150000000), 3)
    out = tmp_path / "mem.json"
    assert P.main(["--root", str(tmp_path), "--mem-summary", str(log), "--mem-out", "mem.json"]) == 0
    assert json.loads(out.read_text())["logs"]["bench_mem_loader"]["worker"] == w


# --- shell scripts ---------------------------------------------------------------------------------------------------
def _bash(args, env=None, cwd=REPO):
    if not BASH:
        pytest.skip("bash unavailable")
    e = dict(os.environ, PY=sys.executable, **(env or {}))
    return subprocess.run([BASH, *args], cwd=cwd, env=e, capture_output=True, text=True, timeout=800)


@pytest.mark.parametrize("script", ["scripts/r8_box_setup.sh", "scripts/run_r8.sh", "scripts/r8_mirror_stage.sh"])
def test_scripts_parse(script):
    r = _bash(["-n", script])
    assert r.returncode == 0, r.stderr


def _g1(path, ok=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(gate_pass=ok, seed=202, bank="bank_r3.npz", param=dict(items=200), room=dict(items=200))))


@pytest.mark.timeout(900)   # Git Bash forks slowly on a loaded Windows host
def test_run_r8_dry_run_queues_both_gpus_in_priority_order(tmp_path):
    g1 = tmp_path / "g1.json"; _g1(g1)
    env = dict(DRY_RUN="1", RUNS_DIR=str(tmp_path / "runs"), G1_JSON=str(g1), WORKERS_GPU0="7", WORKERS_GPU1="5")
    r = _bash(["scripts/run_r8.sh", "start", "pilots"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY gpu")]
    g0 = [ln for ln in dry if ln.startswith("DRY gpu0")]; g1l = [ln for ln in dry if ln.startswith("DRY gpu1")]
    assert "CUDA_VISIBLE_DEVICES=0 VAANI_WORKERS=7" in g0[0] and g0[0].endswith("ab1_fe_mini_s0.yaml")
    assert "CUDA_VISIBLE_DEVICES=1 VAANI_WORKERS=5" in g1l[0] and g1l[0].endswith("ab1_refvalid_s0.yaml")
    assert g0[2].endswith("ab2_pr_nhat_s0.yaml")   # the NLMS arm leads ablation 2
    assert not any("r8_fe_mini.yaml" in ln or "r8_refvalid_v2.yaml" in ln for ln in dry)
    # every generated pilot is queued exactly once, across both GPUs
    heads = [ln.split(": ", 1)[1].split() for ln in r.stdout.splitlines() if ln.startswith(("gpu0 workers", "gpu1 workers"))]
    queued = heads[0] + heads[1]
    stems = sorted(p.stem for p in (REPO / "configs/retraining/r8_ablations").glob("*.yaml"))
    assert len(heads) == 2 and sorted(queued) == stems and len(set(queued)) == len(queued), heads
    # a finished run is skipped, and `next` names the head of each queue
    (tmp_path / "runs/ab1_fe_mini_s0").mkdir(parents=True); (tmp_path / "runs/ab1_fe_mini_s0/DONE").write_text("x")
    n = _bash(["scripts/run_r8.sh", "next"], env)
    assert "gpu0 PENDING ab1_fe_mini_s1" in n.stdout and "gpu1 PENDING ab1_refvalid_s0" in n.stdout, n.stdout
    s = _bash(["scripts/run_r8.sh", "status"], env)
    assert "DONE     ab1_fe_mini_s0" in s.stdout and "G1: gate_pass True" in s.stdout, s.stdout


def _workers(tmp_path, meminfo=None, **env):
    """run_r8.sh workers on a fake 96-vCPU box; meminfo None = no meminfo file at all."""
    fake = tmp_path / "bin"; fake.mkdir(exist_ok=True)
    (fake / "nproc").write_text("#!/bin/sh\necho 96\n", newline="\n"); (fake / "nproc").chmod(0o755)
    mi = tmp_path / "meminfo"
    if meminfo is not None:
        mi.write_text(meminfo, newline="\n")
    # Git Bash needs POSIX paths on PATH; cygpath is absent on Linux
    sh = ('u() { cygpath -u "$1" 2>/dev/null || echo "$1"; }; PATH="$(u "$FAKEBIN"):$PATH" MEMINFO="$(u "$MI")" '
          'exec bash scripts/run_r8.sh workers')
    r = _bash(["-c", sh], dict(FAKEBIN=str(fake), MI=str(mi), RUNS_DIR=str(tmp_path / "runs"), **env))
    assert r.returncode == 0, r.stdout + r.stderr
    got = dict(ln.split(" workers ") for ln in r.stdout.splitlines() if " workers " in ln)
    return {k: int(v) for k, v in got.items()}, r.stderr


@pytest.mark.timeout(900)
@pytest.mark.parametrize("meminfo, env, want, capped", [
    (None, {}, 46, False),                                                     # no meminfo: the core rule, (96 - 4) / 2
    ("MemTotal: 1 kB\n", {}, 46, False),                                        # no MemAvailable line: no cap
    ("MemAvailable: 9000000000 kB\n", {}, 46, False),                           # plenty: never above the core rule
    ("MemAvailable: 250000000 kB\n", {}, 17, True),                             # 256 GB / 2 queues / 3 GB - 8 screens, / 2 loaders
    ("MemAvailable: 250000000 kB\n", {"VAANI_WORKER_RSS_GB": "1"}, 46, False),  # a measured 1 GB lifts the cap
    ("MemAvailable: 10000000 kB\n", {}, 1, True),                               # tiny box: floor of 1
])
def test_run_r8_workers_capped_by_memavailable(tmp_path, meminfo, env, want, capped):
    got, err = _workers(tmp_path, meminfo, **env)
    assert got == {"gpu0": want, "gpu1": want}, (got, err)
    assert ("capped 46 -> %d by memory" % want in err) == capped, err


@pytest.mark.timeout(900)
@pytest.mark.parametrize("rss", ["0", "abc", "-2"])
def test_run_r8_bad_worker_rss_keeps_the_default_cap(tmp_path, rss):
    got, err = _workers(tmp_path, "MemAvailable: 250000000 kB\n", VAANI_WORKER_RSS_GB=rss)
    assert got == {"gpu0": 17, "gpu1": 17} and "is not a positive number; using 3" in err, (got, err)
    assert "capped 46 -> 17 by memory" in err and "awk" not in err and "integer expression" not in err, err


@pytest.mark.timeout(900)
def test_run_r8_queues_the_bank_arm_last_on_gpu0_once_generated(tmp_path):
    (tmp_path / "scripts").mkdir(); (tmp_path / "scripts/run_r8.sh").write_bytes((REPO / "scripts/run_r8.sh").read_bytes())
    a = tmp_path / "configs/retraining/r8_ablations"; a.mkdir(parents=True)
    stems = [p.stem for p in (REPO / "configs/retraining/r8_ablations").glob("*.yaml") if p.stem != "ab7_bank_r3"]
    g1 = tmp_path / "g1.json"; _g1(g1)
    env = dict(DRY_RUN="1", RUNS_DIR=str(tmp_path / "runs"), G1_JSON=str(g1), WORKERS_GPU0="7", WORKERS_GPU1="5")
    for with_arm in (False, True):
        for s in stems + (["ab7_bank_r3"] if with_arm else []):
            (a / f"{s}.yaml").write_text("x\n")
        r = _bash(["scripts/run_r8.sh", "start", "pilots"], env, cwd=tmp_path)
        assert r.returncode == 0, r.stdout + r.stderr
        heads = {ln.split(" workers ")[0]: ln.split(": ", 1)[1].split() for ln in r.stdout.splitlines()
                 if ln.startswith(("gpu0 workers", "gpu1 workers"))}
        dry0 = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY gpu0")]
        assert ("ab7_bank_r3" in heads["gpu1"]) is False and len(heads["gpu1"]) == 11, heads
        if with_arm:
            assert heads["gpu0"][-1] == "ab7_bank_r3" and len(heads["gpu0"]) == 12 and dry0[-1].endswith("ab7_bank_r3.yaml"), heads
        else:
            assert "ab7_bank_r3" not in heads["gpu0"] and len(heads["gpu0"]) == 11, heads


@pytest.mark.timeout(900)
def test_run_r8_explicit_workers_skip_the_memory_cap(tmp_path):
    got, err = _workers(tmp_path, "MemAvailable: 250000000 kB\n", WORKERS_GPU0="60")
    assert got == {"gpu0": 60, "gpu1": 17} and "workers gpu0" not in err, (got, err)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("g1", [None, False])
def test_run_r8_refuses_without_a_passing_g1(tmp_path, g1):
    p = tmp_path / "g1.json"
    if g1 is not None:
        _g1(p, ok=g1)
    env = dict(DRY_RUN="1", RUNS_DIR=str(tmp_path / "runs"), G1_JSON=str(p))
    for phase in ("full", "all"):
        r = _bash(["scripts/run_r8.sh", "start", phase], env)
        assert r.returncode == 1 and "REFUSED" in r.stdout and "DRY gpu" not in r.stdout, r.stdout


@pytest.mark.timeout(900)   # Git Bash forks slowly on a loaded Windows host
def test_mirror_stage_on_a_fake_tree(tmp_path):
    t = tmp_path / "laptop"
    for d in ("scripts", "configs/retraining/r8_ablations", "data/manifests", "data/eval_r2/val/b"):
        (t / d).mkdir(parents=True)
    for s in ("r8_mirror_stage.sh", "verify_eval_set.py"):
        shutil.copy(REPO / "scripts" / s, t / "scripts" / s)
    (t / "configs/retraining/r8_fe_mini.yaml").write_text(yaml.safe_dump(
        {"data": {"manifests": ["data/manifests/a.parquet", "data/manifests/mad_v2.parquet", "data/manifests/gone.parquet"]}}))
    for m in ("a", "mad_v2", "mad_speech_contamination"):
        (t / f"data/manifests/{m}.parquet").write_bytes(m.encode())
    (t / "data/eval_r2/val/b/000.json").write_text("{}")
    for s in (".mix.wav", ".clean.wav"):
        (t / f"data/eval_r2/val/b/000{s}").write_bytes(b"")
    (t / "web.wav").write_bytes(b"RIFF")
    env = dict(VAL_HASH=hashlib.sha1(b"{}").hexdigest()[:12], MIRROR_HF_REPO="me/private-mirror", WEB_WAV="web.wav")
    r = _bash(["scripts/r8_mirror_stage.sh", "data/stage"], env, cwd=t)
    assert r.returncode == 0, r.stdout + r.stderr
    st = t / "data/stage"
    sums = dict(ln.split("  ", 1)[::-1] for ln in (st / "SHA256SUMS").read_text().split("\n") if ln)
    assert set(sums) == {"eval_r2_val.tar", "manifests/mad_v2.parquet", "manifests/mad_speech_contamination.parquet",
                         "manifests_laptop/a.parquet", "field/abcd.wav"}
    assert all(_sha(st / f) == h for f, h in sums.items())
    assert "WARN: no laptop data/manifests/gone.parquet" in r.stdout
    assert "--private" in r.stdout and "hf upload me/private-mirror data/stage . --repo-type dataset" in r.stdout
    # rerun is idempotent: same manifest
    before = (st / "SHA256SUMS").read_text()
    assert _bash(["scripts/r8_mirror_stage.sh", "data/stage"], env, cwd=t).returncode == 0
    assert (st / "SHA256SUMS").read_text() == before


@pytest.mark.timeout(900)   # Git Bash forks slowly on a loaded Windows host
def test_box_setup_names_missing_credentials_up_front():
    env = {k: "" for k in ("MIRROR_HF_REPO", "HF_TOKEN", "RIR_BANK_URL")}
    r = _bash(["scripts/r8_box_setup.sh", "--env-check"], env)
    assert r.returncode == 1
    for k in env:
        assert f"MISSING {k}" in r.stdout, r.stdout
    d = _bash(["scripts/r8_box_setup.sh", "--dry-run"], env)
    assert d.returncode == 0, d.stdout + d.stderr
    for s in ("DRY sync", "DRY banks", "DRY mirror", "DRY sidecars", "DRY g1", "DRY bench", "DRY preflight"):
        assert s in d.stdout, d.stdout


# --- requirements files ----------------------------------------------------------------------------------------------
def _lock():
    import tomllib
    return {p["name"]: p["version"] for p in tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))["package"]}


def test_deploy_requirements_pin_the_lock_and_stay_inference_only():
    lock = _lock()
    pins = dict(ln.split("==") for ln in (REPO / "requirements-deploy.txt").read_text().splitlines()
                if ln and not ln.startswith("#"))
    assert {"numpy", "onnxruntime", "numba", "llvmlite"} <= set(pins)
    assert {k: lock.get(k) for k in pins} == pins
    assert not {"torch", "torchaudio", "torch-pesq", "faster-whisper", "tensorrt", "tensorrt-cu12"} & set(pins)


def test_requirements_txt_is_the_current_lock_export():
    txt = (REPO / "requirements.txt").read_text()
    assert txt.startswith("# Generated from uv.lock by scripts/export_requirements.sh") and "# SCOPE:" in txt
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" in txt
    lock = _lock()
    for name in ("torch", "torchaudio", "onnxruntime", "numba", "torch-pesq", "faster-whisper"):
        assert f"\n{name}=={lock[name]} " in txt, name
