"""Task 9 comparison tooling on synthetic per-clip tables: the registered uncertainty model and decision rule, the
refusals (populations, configurations, metric definitions, exposure, seed counts, margins and floors), the screen,
the modulation index, blinded listening export and the gate-independent recipe freeze. No real scores are involved."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_r8_ld as C  # noqa: E402

POP = {"split": "val", "eval_root": "data/eval_r2", "route": "unified", "sample_rate": 16000, "resampler": None}
EXPO = {"epochs": 30, "batch": 32, "optimizer_steps": 6000, "scored_s": 768000.0, "prefix_s": 0.0}
NUM = {"gru_kernel": "fused", "render": "gpu", "cuda_graph": True}
CLASSES = ["stationary", "impulsive", "changing", "clean", "impulsive+stationary"]


def table(path, seed, shift=None, seed_sd=0.0, n_scene=30, drop=0, rng_base=0, faults=True):
    """Deterministic clip-level base (same for every arm and seed) plus per-arm shifts and per-seed noise."""
    shift = shift or {}
    base = np.random.default_rng(1234)
    g = np.random.default_rng(rng_base * 1000 + seed + 17)
    rows = []
    for s in range(n_scene):
        for j, nc in enumerate(CLASSES):
            fault = "fault_refdrop_long" if faults and j == 2 and s % 3 == 0 else None
            rows.append({"bucket": f"{nc}_{s % 3}", "id": f"{s:04d}", "noise_class": nc, "snr_in": s % 3 * 5,
                         "fault": fault, "impulse_peak_db": 20.0 if "impulsive" in nc else None,
                         "snr_out": 16 + base.normal(0, 2), "stoi": 0.88 + base.normal(0, 0.02),
                         "pesq_wb": 2.7 + base.normal(0, 0.2), "speech_loss": abs(0.03 + base.normal(0, 0.01)),
                         "dnsmos_ovrl": 3.0 + base.normal(0, 0.1),
                         "mod_idx_hop96_db": -40 + base.normal(0, 1), "mod_idx_clean_hop96_db": -42 + base.normal(0, 1)})
    df = pd.DataFrame(rows)
    noise = {m: g.normal(0, seed_sd * sc) for m, sc in
             {"snr_out": 1, "stoi": 0.02, "pesq_wb": 0.3, "speech_loss": 0.02, "dnsmos_ovrl": 0.3}.items()}
    for m in ("snr_out", "stoi", "pesq_wb", "speech_loss", "dnsmos_ovrl", "mod_idx_hop96_db"):
        df[m] = df[m] + shift.get(m, 0.0) + noise.get(m, 0.0)
    if "transient" in shift:   # a shift on the transient subset only
        for m, v in shift["transient"].items():
            df.loc[df.noise_class.str.contains("impulsive"), m] += v
    if drop:
        df = df.iloc[drop:]
    df.to_csv(path, index=False)
    return str(path)


def run(tmp, arm, seed, shift=None, seed_sd=0.0, **kw):
    over = {k: kw.pop(k) for k in list(kw) if k in ("recipe", "contract_id", "metric_defs", "population", "exposure",
                                                     "numerics", "composite", "runtime")}
    r = {"seed": seed, "csv": table(tmp / f"{arm}_s{seed}.csv", seed, shift, seed_sd, rng_base=sum(map(ord, arm)), **kw),
         "recipe": f"{arm}_recipe", "contract_id": f"{arm}_contract", "metric_defs": C.METRIC_DEFS,
         "population": POP, "exposure": dict(EXPO), "numerics": NUM, "runtime": {"step_ms": 10.0, "peak_mem_mb": 900}}
    r.update(over)
    return r


def manifest(tmp, seeds=range(5), cand_shift=None, comparison="ld_vs_c0", kind="confirmation", cand="arm_a",
             ctrl="c0", seed_sd=0.0, fixed=False, **kw):
    ctrl_runs = [run(tmp, ctrl, 0)] if fixed else [run(tmp, ctrl, s, seed_sd=seed_sd) for s in seeds]
    return {"comparison": comparison, "kind": kind, "population": POP, "disclosed_exposure": {},
            "candidate": {"arm": cand, "hop": 96, "runs": [run(tmp, cand, s, cand_shift, seed_sd) for s in seeds]},
            "control": {"arm": ctrl, "hop": 256, "fixed_reference": fixed, "runs": ctrl_runs}, **kw}


@pytest.fixture
def reg(tmp_path):
    a = table(tmp_path / "c0_s0_reg.csv", 0)
    b = table(tmp_path / "c0_s1_reg.csv", 1, {"speech_loss": 0.001})
    return C.register(a, b)


# ---- registry, margins and floors ------------------------------------------------------------------------------------

def test_register_sets_the_speech_loss_floor_from_c0_seeds(tmp_path):
    a = table(tmp_path / "a.csv", 0)
    small = C.register(a, table(tmp_path / "b.csv", 1, {"speech_loss": 0.001}))
    assert small["metrics"]["speech_loss"]["floor"] == pytest.approx(0.002)   # max(0.002, 0.001)
    big = C.register(a, table(tmp_path / "c.csv", 1, {"speech_loss": 0.004}))
    assert big["metrics"]["speech_loss"]["floor"] == pytest.approx(0.004)
    assert small["status"] == "proposed" and C.check_registry(small) == []
    assert small["metrics"]["snr_out"] == {"margin": 0.25, "floor": 0.053, "higher_better": True, "unit": "dB"}


def test_register_is_never_redone_after_results(tmp_path):
    a, b = table(tmp_path / "a.csv", 0), table(tmp_path / "b.csv", 1)
    out = tmp_path / "reg.json"
    assert C.main(["register", "--c0-s0", a, "--c0-s1", b, "--out", str(out)]) == 0
    first = out.read_text()
    assert C.main(["register", "--c0-s0", a, "--c0-s1", b, "--approved-by", "x", "--out", str(out)]) == 2
    assert out.read_text() == first


@pytest.mark.parametrize("field", ["margin", "floor"])
def test_missing_margin_or_floor_refuses(tmp_path, reg, field):
    reg["metrics"]["pesq_wb"][field] = None
    with pytest.raises(C.ComparisonError, match=f"pesq_wb has no registered {field}"):
        C.compare(manifest(tmp_path), reg)
    reg["metrics"]["pesq_wb"][field] = 0.05
    del reg["metrics"]["speech_loss"]
    with pytest.raises(C.ComparisonError, match="speech_loss has no registered margin"):
        C.compare(manifest(tmp_path), reg)


def test_interval_and_decide_refuse_without_floor_or_margin():
    with pytest.raises(C.ComparisonError, match="floor"):
        C.interval([0.1, 0.2], np.zeros(10), np.arange(10), None)
    with pytest.raises(C.ComparisonError, match="margin"):
        C.decide(-0.1, 0.1, None)


# ---- the uncertainty model and the decision rule ---------------------------------------------------------------------

@pytest.mark.parametrize("lo,hi,hb,want", [
    (0.01, 0.3, True, "superior"), (-0.2, 0.3, True, "non_inferior"), (-0.25, 0.0, True, "non_inferior"),
    (-0.6, -0.3, True, "inferior"), (-0.4, 0.1, True, "inconclusive"),
    (-0.3, -0.01, False, "superior"), (-0.3, 0.2, False, "non_inferior"), (0.3, 0.6, False, "inferior"),
    (-0.1, 0.4, False, "inconclusive")])
def test_decide_bounds_and_mirroring(lo, hi, hb, want):
    assert C.decide(lo, hi, 0.25, hb) == want


def test_interval_is_t_times_the_floored_seed_plus_clip_variance():
    d = [0.10, 0.10, 0.10, 0.10, 0.10]                  # zero observed spread: the floor binds
    e = np.full(40, 0.1); sc = np.arange(40)             # zero clip variance
    r = C.interval(d, e, sc, 0.053)
    assert r["floor_binding"] and r["s_d_used"] == 0.053 and r["df"] == 4
    h = 2.7764451 * 0.053 / math.sqrt(5)                 # Section 2.9: 0.066 dB at five seeds
    assert r["hi"] - r["d_bar"] == pytest.approx(h, rel=1e-5) and h == pytest.approx(0.066, abs=5e-4)
    wide = C.interval([0.0, 0.2, -0.1, 0.3, 0.1], e, sc, 0.053)
    assert not wide["floor_binding"] and wide["s_d_used"] == pytest.approx(np.std([0, .2, -.1, .3, .1], ddof=1))
    noisy = C.interval(d, np.random.default_rng(0).normal(0.1, 1, 40), sc, 0.053)
    assert noisy["v_clip"] > 0 and noisy["hi"] - noisy["lo"] > 2 * h


def test_clip_bootstrap_resamples_whole_scenes():
    e = np.repeat(np.random.default_rng(1).normal(0, 1, 10), 20)     # 20 identical clips per scene
    clustered = C.cluster_boot_var(e, np.repeat(np.arange(10), 20))
    naive = C.cluster_boot_var(e, np.arange(200))
    assert clustered > 5 * naive


def test_single_seed_needs_the_confirmation_variance():
    with pytest.raises(C.ComparisonError, match="at least 2 seeds"):
        C.interval([0.1], np.zeros(5), np.arange(5), 0.05)
    r = C.interval([0.1], np.zeros(5), np.arange(5), 0.05, seed_var=(0.08, 4))
    assert r["s_d_used"] == 0.08 and r["df"] == 4


def test_identical_arms_are_non_inferior_not_superior(tmp_path, reg):
    rep = C.compare(manifest(tmp_path), reg)
    for m in C.D2:
        assert rep["registered"][m]["overall"]["status"] == "non_inferior", m
        assert rep["registered"][m]["overall"]["d_bar"] == pytest.approx(0, abs=1e-9)
    assert rep["gate_b_quality"]["pass"] and rep["summary"]["improvements"] == []


def test_direction_of_every_bound(tmp_path, reg):
    better = C.compare(manifest(tmp_path, cand_shift={"snr_out": 0.5, "speech_loss": -0.02}), reg)
    assert better["registered"]["snr_out"]["overall"]["status"] == "superior"
    assert better["registered"]["speech_loss"]["overall"]["status"] == "superior"   # lower is better
    worse = C.compare(manifest(tmp_path, cand_shift={"stoi": -0.02, "speech_loss": 0.02}), reg)
    assert worse["registered"]["stoi"]["overall"]["status"] == "inferior"
    assert worse["registered"]["speech_loss"]["overall"]["status"] == "inferior"
    assert "stoi" in worse["summary"]["regressions"] and not worse["gate_b_quality"]["pass"]
    r = worse["registered"]["stoi"]["overall"]
    assert r["lo"] < r["d_bar"] < r["hi"] < -r["margin"]


def test_seed_spread_makes_a_mean_gain_inconclusive(tmp_path, reg):
    rep = C.compare(manifest(tmp_path, cand_shift={"pesq_wb": 0.02}, seed_sd=0.5), reg)
    r = rep["registered"]["pesq_wb"]["overall"]
    assert r["s_d_observed"] > r["floor"] and r["status"] == "inconclusive"
    assert "pesq_wb" in rep["summary"]["uncertain"] and not rep["gate_b_quality"]["pass"]


def test_subsets_gate_only_on_inferior(tmp_path, reg):
    rep = C.compare(manifest(tmp_path, cand_shift={"transient": {"snr_out": -2.0}}), reg)
    assert rep["registered"]["snr_out"]["transient"]["status"] == "inferior"
    assert rep["registered"]["snr_out"]["transient"]["gating"] == "inferior_only"
    assert rep["registered"]["snr_out"]["ref_fault"]["n_clips"] > 0
    assert "snr_out/transient" in rep["summary"]["regressions"] and not rep["gate_b_quality"]["pass"]


def test_pass_rate_is_all_three_in_pp_and_excludes_clean(tmp_path):
    df = C.load_table(table(tmp_path / "t.csv", 0))
    assert df.loc[df.subset_clean, "pass3"].isna().all()
    ok = (df.snr_out > 15) & (df.stoi > 0.85) & (df.pesq_wb > 2.5)
    assert (df.loc[~df.subset_clean, "pass3"] == 100 * ok[~df.subset_clean]).all()


# ---- refusals: populations, configurations, definitions, exposure, seed counts ---------------------------------------

def test_unmatched_clip_ids_refused(tmp_path, reg):
    m = manifest(tmp_path)
    m["candidate"]["runs"][2] = run(tmp_path, "arm_a", 2, drop=3)
    with pytest.raises(C.ComparisonError, match="clip IDs do not match"):
        C.compare(m, reg)


def test_other_population_refused_even_for_r7(tmp_path, reg):
    m = manifest(tmp_path, comparison="c0_vs_r7", cand="c0", ctrl="r7", fixed=True,
                 disclosed_exposure={k: "r7 is a warm-start fine-tune" for k in C.EXPOSURE_KEYS})
    m["control"]["runs"][0]["population"] = dict(POP, split="test")
    with pytest.raises(C.ComparisonError, match="population"):
        C.compare(m, reg)
    m = manifest(tmp_path)
    m["population"] = dict(POP, clip_keys_sha256="0" * 64)
    for r in m["candidate"]["runs"] + m["control"]["runs"]:
        r["population"] = m["population"]
    with pytest.raises(C.ComparisonError, match="registered clip set"):
        C.compare(m, reg)


def test_incompatible_metric_definitions_refused(tmp_path, reg):
    m = manifest(tmp_path)
    m["candidate"]["runs"][0]["metric_defs"] = dict(C.METRIC_DEFS, pesq_wb="pesq nb")
    with pytest.raises(C.ComparisonError, match="metric definitions"):
        C.compare(m, reg)
    reg["metric_defs"] = dict(C.METRIC_DEFS, stoi="estoi")
    with pytest.raises(C.ComparisonError, match="registry: metric definitions"):
        C.compare(manifest(tmp_path), reg)


def test_undisclosed_exposure_refused_disclosed_allowed(tmp_path, reg):
    m = manifest(tmp_path)
    for r in m["candidate"]["runs"]:
        r["exposure"] = dict(EXPO, optimizer_steps=7000)
    with pytest.raises(C.ComparisonError, match="undisclosed exposure difference in optimizer_steps"):
        C.compare(m, reg)
    m["disclosed_exposure"] = {"optimizer_steps": "warm-up arm adds 480 steps"}
    assert C.compare(m, reg)["disclosed_exposure"]
    m["candidate"]["runs"][1]["exposure"] = {"epochs": 30}
    with pytest.raises(C.ComparisonError, match="not disclosed"):
        C.compare(m, reg)


def test_mismatched_configurations_refused(tmp_path, reg):
    m = manifest(tmp_path)
    m["candidate"]["runs"][3]["recipe"] = "other"
    with pytest.raises(C.ComparisonError, match="runs differ in recipe"):
        C.compare(m, reg)
    m = manifest(tmp_path)
    m["candidate"]["runs"][0]["contract_id"] = "x"
    with pytest.raises(C.ComparisonError, match="contract_id"):
        C.compare(m, reg)
    m = manifest(tmp_path)
    for r in m["candidate"]["runs"]:
        r["numerics"] = dict(NUM, cuda_graph=False)
    with pytest.raises(C.ComparisonError, match="perf.numerics"):
        C.compare(m, reg)


def test_seeds_must_pair(tmp_path, reg):
    m = manifest(tmp_path)
    m["control"]["runs"][4] = run(tmp_path, "c0", 7)
    with pytest.raises(C.ComparisonError, match="not paired"):
        C.compare(m, reg)
    m = manifest(tmp_path)
    m["candidate"]["runs"][1] = run(tmp_path, "arm_a", 0)
    with pytest.raises(C.ComparisonError, match="listed twice"):
        C.compare(m, reg)


def test_confirmation_needs_five_seeds_per_arm(tmp_path, reg):
    with pytest.raises(C.ComparisonError, match="at least 5 seeds"):
        C.compare(manifest(tmp_path, seeds=range(4)), reg)
    m = manifest(tmp_path)
    m["control"]["runs"] = m["control"]["runs"][:4]
    m["candidate"]["runs"] = m["candidate"]["runs"][:4]
    with pytest.raises(C.ComparisonError, match="at least 5 seeds"):
        C.compare(m, reg)
    assert C.compare(manifest(tmp_path, seeds=range(2), kind="stage1"), reg)["kind"] == "stage1"


def test_fixed_reference_only_for_r7(tmp_path, reg):
    with pytest.raises(C.ComparisonError, match="seed-paired"):
        C.compare(manifest(tmp_path, fixed=True), reg)
    with pytest.raises(C.ComparisonError, match="control arm"):
        C.compare(manifest(tmp_path, ctrl="r7"), reg)
    m = manifest(tmp_path, comparison="final_vs_r7", ctrl="r7", fixed=True,
                 disclosed_exposure={k: "r7 recipe" for k in C.EXPOSURE_KEYS})
    rep = C.compare(m, reg)
    assert rep["fixed_reference"] and "gate_b_quality" not in rep


def test_three_comparison_types_and_arm_r(tmp_path, reg):
    assert set(C.COMPARISONS) == {"c0_vs_r7", "ld_vs_c0", "arm_a_vs_arm_r", "final_vs_r7"}
    rep = C.compare(manifest(tmp_path, comparison="arm_a_vs_arm_r", ctrl="arm_r"), reg)
    assert rep["control"] == "arm_r"
    with pytest.raises(C.ComparisonError, match="candidate arm"):
        C.compare(manifest(tmp_path, comparison="arm_a_vs_arm_r", cand="arm_b", ctrl="arm_r"), reg)


def test_full_run_uses_the_confirmation_seed_variance(tmp_path, reg):
    conf = C.compare(manifest(tmp_path), reg)
    p = tmp_path / "conf.json"; p.write_text(json.dumps(conf, default=float))
    with pytest.raises(C.ComparisonError, match="seed_var_from"):
        C.compare(manifest(tmp_path, seeds=[0], kind="full"), reg)
    rep = C.compare(manifest(tmp_path, seeds=[0], kind="full", seed_var_from=str(p)), reg)
    r = rep["registered"]["snr_out"]["overall"]
    assert r["S"] == 1 and r["df"] == 4 and r["s_d_used"] == conf["registered"]["snr_out"]["overall"]["s_d_used"]
    assert rep["gate_c_full"]["not_inferior"]
    stage = C.compare(manifest(tmp_path, seeds=range(2), kind="stage1"), reg)
    p.write_text(json.dumps(stage, default=float))
    with pytest.raises(C.ComparisonError, match="not a confirmation"):
        C.compare(manifest(tmp_path, seeds=[0], kind="full", seed_var_from=str(p)), reg)


def test_stage2_screen(tmp_path, reg):
    m = manifest(tmp_path, seeds=[0], kind="screen")
    m["candidate"]["runs"][0]["composite"], m["control"]["runs"][0]["composite"] = 0.61, 0.60
    assert C.compare(m, reg)["screen"]["promote"]
    m = manifest(tmp_path, seeds=[0], kind="screen", cand_shift={"pesq_wb": -0.08})
    m["candidate"]["runs"][0]["composite"], m["control"]["runs"][0]["composite"] = 0.61, 0.60
    s = C.compare(m, reg)["screen"]
    assert not s["promote"] and "pesq_wb" in s["worse_than_margin"] and "snr_out" not in s["worse_than_margin"]
    m["candidate"]["runs"][0]["composite"] = 0.59
    assert "composite decreased" in C.compare(m, reg)["screen"]["reason"]
    with pytest.raises(C.ComparisonError, match="exactly one seed"):
        C.compare(manifest(tmp_path, seeds=range(2), kind="screen"), reg)


# ---- reported metrics, report layout --------------------------------------------------------------------------------

def test_reported_metrics_are_never_gated(tmp_path, reg):
    rep = C.compare(manifest(tmp_path, cand_shift={"dnsmos_ovrl": -1.0, "mod_idx_hop96_db": 6.0}), reg)
    assert rep["reported"]["dnsmos_ovrl"]["d_bar"] == pytest.approx(-1.0)
    mi = rep["reported"]["modulation_index"]
    assert mi["hop_hz"] == pytest.approx(166.667, abs=1e-3) and mi["measured"]
    assert mi["stationary"]["db_vs_control"] == pytest.approx(6.0) and mi["clean"]["n_clips"] > 0
    assert rep["gate_b_quality"]["pass"]
    assert rep["runtime"]["candidate"] == {"peak_mem_mb": 900.0, "step_ms": 10.0}


def test_cli_report_separates_improvements_regressions_uncertainty(tmp_path, reg):
    m = manifest(tmp_path, cand_shift={"snr_out": 0.5, "stoi": -0.02})
    mp, rp = tmp_path / "m.json", tmp_path / "reg.json"
    mp.write_text(json.dumps(m)); rp.write_text(json.dumps(reg))
    out = tmp_path / "rep"
    assert C.main(["compare", str(mp), "--registry", str(rp), "--out", str(out)]) == 0
    md = (out / "report.md").read_text()
    for h in ("## Measured improvements", "## Regressions", "## Uncertain", "## Reported, not gated"):
        assert h in md
    rep = json.loads((out / "report.json").read_text())
    assert rep["summary"]["improvements"] == ["snr_out"] and "stoi" in rep["summary"]["regressions"]
    assert rep["provenance"]["revision"] is None or len(rep["provenance"]["revision"]) == 40
    m["candidate"]["runs"] = m["candidate"]["runs"][:3]
    mp.write_text(json.dumps(m))
    assert C.main(["compare", str(mp), "--registry", str(rp), "--out", str(tmp_path / "x")]) == 2


# ---- modulation index ------------------------------------------------------------------------------------------------

def test_modulation_index_sees_hop_rate_roughness_and_ignores_level():
    t = np.arange(2 * C.SR) / C.SR
    noise = np.random.default_rng(0).normal(0, 0.1, len(t))
    rough = noise * (1 + 0.3 * np.sin(2 * np.pi * (C.SR / 96) * t))
    assert C.modulation_index(rough, 96) > C.modulation_index(noise, 96) + 10
    assert C.modulation_index(rough, 96) > C.modulation_index(rough, 128) + 10   # Arm B's 125 Hz lines are clean
    assert C.modulation_index(3 * rough, 96) == pytest.approx(C.modulation_index(rough, 96), abs=1e-6)
    assert math.isnan(C.modulation_index(np.zeros(100), 96))


def test_modindex_table_from_wav_trees(tmp_path):
    sf = pytest.importorskip("soundfile")
    for d in ("out", "clean"):
        (tmp_path / d / "stationary_0").mkdir(parents=True)
        sf.write(tmp_path / d / "stationary_0" / "0001.wav", np.random.default_rng(1).normal(0, .1, 8000), C.SR)
    rows = C.modindex_table(tmp_path / "out", tmp_path / "clean", [96, 128])
    assert rows[0]["bucket"] == "stationary_0" and rows[0]["id"] == "0001"
    assert rows[0]["mod_idx_hop96_db"] == pytest.approx(rows[0]["mod_idx_clean_hop96_db"])


# ---- listening examples ---------------------------------------------------------------------------------------------

def test_listening_export_blinds_labels_and_keeps_unaligned_apart(tmp_path):
    sf = pytest.importorskip("soundfile")
    systems = {}
    for arm in ("c0", "arm_a", "r7"):
        d = tmp_path / arm; d.mkdir()
        for k in range(len(C.LISTEN_CATEGORIES)):
            sf.write(d / f"clip{k}.wav", np.full(160, {"c0": .1, "arm_a": .2, "r7": .3}[arm]), C.SR)
        systems[arm] = str(d)
    (tmp_path / "raw").mkdir(); (tmp_path / "raw" / "rec.wav").write_bytes(b"x")
    (tmp_path / "timing.json").write_text("{}")
    spec = {"categories": {c: [f"clip{k}"] for k, c in enumerate(C.LISTEN_CATEGORIES)}, "systems": systems,
            "unaligned": {"arm_a": str(tmp_path / "raw")}, "timing": str(tmp_path / "timing.json")}
    out = tmp_path / "listen"
    key = C.export_listening(spec, out, seed=3)
    files = sorted((out / "aligned").rglob("*.wav"))
    assert len(files) == 3 * len(C.LISTEN_CATEGORIES)
    assert {f.stem for f in files} == {"S1", "S2", "S3"}
    assert not any(a in str(f) for f in files for a in systems)                    # no arm names in the blinded tree
    assert (out / "key" / "key.json").exists() and not list((out / "aligned").rglob("*.json"))
    orders = {tuple(key[f"{c}/clip{k}/S{i}"] for i in (1, 2, 3)) for k, c in enumerate(C.LISTEN_CATEGORIES)}
    assert len(orders) > 1                                                        # labels are shuffled per clip
    for k, c in enumerate(C.LISTEN_CATEGORIES):
        for i in (1, 2, 3):
            v = sf.read(out / "aligned" / c / f"clip{k}" / f"S{i}.wav")[0][0]
            assert v == pytest.approx({"c0": .1, "arm_a": .2, "r7": .3}[key[f"{c}/clip{k}/S{i}"]], abs=1e-3)
    assert (out / "unaligned" / "arm_a" / "rec.wav").exists() and (out / "unaligned" / "timing.json").exists()
    spec["categories"].pop("talker_leakage")
    with pytest.raises(C.ComparisonError, match="talker_leakage"):
        C.export_listening(spec, tmp_path / "l2", seed=3)


# ---- freeze: absolute gates independent of the comparison ------------------------------------------------------------

def _gates(**over):
    g = {k: {"status": "pass", "evidence": f"results_r2/r8_ld/{k}.json"} for k in C.FREEZE_GATES}
    g.update(over)
    return g


def test_freeze_requires_complete_gates_and_quality(tmp_path, reg):
    good = tmp_path / "good.json"; good.write_text(json.dumps(C.compare(manifest(tmp_path), reg), default=float))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(C.compare(manifest(tmp_path, cand_shift={"stoi": -0.02}), reg), default=float))
    three = tmp_path / "three.json"
    three.write_text(json.dumps(C.compare(manifest(tmp_path, seeds=range(3), kind="stage1"), reg), default=float))
    sel = {"tier": "mini", "variants": [
        {"name": "superior_but_gate_failed", "confirmation": str(good), "gates": _gates(gate0a={"status": "fail", "evidence": "e"})},
        {"name": "missing_evidence", "confirmation": str(good), "gates": _gates(export={"status": "pass"})},
        {"name": "pending_gate", "confirmation": str(good), "gates": _gates(gate_a={"status": "pending"})},
        {"name": "inferior", "confirmation": str(bad), "gates": _gates()},
        {"name": "three_seeds", "confirmation": str(three), "gates": _gates()},
        {"name": "chosen", "recipe": "r8_ld_fe_mini", "confirmation": str(good), "gates": _gates()},
        {"name": "late", "confirmation": str(good), "gates": _gates()}]}
    res = C.freeze(sel)
    assert res["status"] == "frozen" and res["frozen"]["name"] == "chosen"
    why = {r["name"]: " ".join(r["reasons"]) for r in res["rejected"]}
    assert "gate0a: fail" in why["superior_but_gate_failed"]
    assert "export: incomplete evidence" in why["missing_evidence"]
    assert "gate_a: incomplete evidence" in why["pending_gate"]
    assert "stoi" in why["inferior"] and "owner decision" in why["inferior"]
    assert "five-seed confirmation" in why["three_seeds"]
    assert "chosen was frozen first" in why["late"]


def test_owner_decision_covers_quality_never_gates(tmp_path, reg):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(C.compare(manifest(tmp_path, cand_shift={"stoi": -0.02}), reg), default=float))
    od = {"by": "owner", "gap": "STOI -0.02 [..], Arm R beside it"}
    assert C.freeze({"variants": [{"name": "v", "confirmation": str(bad), "gates": _gates(), "owner_decision": od}]})["frozen"]
    res = C.freeze({"variants": [{"name": "v", "confirmation": str(bad), "owner_decision": od,
                                  "gates": _gates(spec62={"status": "fail", "evidence": "e"})}]})
    assert res["frozen"] is None and "spec62: fail" in res["rejected"][0]["reasons"][0]
    assert C.freeze({"variants": [{"name": "v", "gates": _gates()}]})["rejected"][0]["reasons"] == [
        "no five-seed confirmation report"]
