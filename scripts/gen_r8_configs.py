"""Writes the r8 ablation pilots (configs/retraining/r8_ablations/*.yaml) from the two full r8 configs.

usage (repo root):
    python scripts/gen_r8_configs.py              # rewrite every pilot from r8_fe_mini.yaml / r8_refvalid_v2.yaml
    python scripts/gen_r8_configs.py --check      # exit 1 if any pilot on disk differs from what this would write
    python scripts/gen_r8_configs.py --bank-arm   # also write ab7_bank_r3 (NOT in the default set: see below)
Once ab7_bank_r3.yaml exists, every mode treats it as a pilot: --check verifies it and a rewrite refreshes it.

The full configs are the source: each pilot is a full config with 48 epochs, its own name and seed, and the one field
(or DSP block) its arm varies, so the untouched fields cannot drift. The full configs are only read, never written.
Body: yaml.safe_dump(sort_keys=True) after the '#' header, LF line ends (the format of the committed files).

ab7_bank_r3 (ab1_fe_mini_s0 on data/rirs/bank_r3.npz) is opt-in because bank_r3 shares 1394 rooms with bank.npz, the
eval_r2 val render that selects checkpoints (results_r2/r8/banks/README.md), so its val scores would be biased.
Rachit opted in (2026-09-26, run it last): the file is committed, so plain --check and rewrites keep it.

Low-delay mode (low-delay plan Task 8): every low-delay configuration derives from r8_ld_fe_mini.yaml, which is itself
r8_fe_mini.yaml plus the overlay LD_OVERLAY below (Arm A at the Gate 0a-selected support, read from GATE0).
    python scripts/gen_r8_configs.py --low-delay            # rewrite r8_ld_*.yaml, r8_ld_ablations/*.yaml, arms.json
    python scripts/gen_r8_configs.py --low-delay --check    # exit 1 if any of them differs, is missing or is extra
    python scripts/gen_r8_configs.py --low-delay --promote ld_s2_overparam[,ld_s2_...]   # wave 2 (Stage-2 decision)
The support is never an argument: a changed Gate 0a record makes --check fail until the configs are regenerated.
Arm B's configurations are always written; the queue (scripts/r8_ld_queue.py) runs them only when Gate 0a pilots it.
"""
import argparse, copy, json, sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = "configs/retraining/r8_ablations"
FULL = {"fe": "configs/retraining/r8_fe_mini.yaml", "refvalid": "configs/retraining/r8_refvalid_v2.yaml"}
PILOT_EPOCHS = 48   # 15 % of the 320-epoch full schedule (960,000 items, 30,000 steps)
# the pr_nhat input needs n_hat, so that arm runs the worker DSP (fixed NLMS, robust kernel, blocking, ref_policy)
# (limiter_kernel: every r8 arm, C0 included, runs the compiled limiter; low-delay plan Section 3.3)
NHAT_DSP = dict(limiter=True, limiter_kernel="numba", blocking=True,
                controller=dict(block_margin_db=10.0, diff_jump_max_db=3.0),
                ref_policy=dict(nlms=True, absent="freeze", ramp_frames=12))
BANK_R3 = "data/rirs/bank_r3.npz"


def load(root, p):
    return yaml.safe_load(open(root / p, encoding="utf-8"))


def pilot(base, stem, seed, **over):
    c = copy.deepcopy(base); c["name"] = "r8" + stem; c["seed"] = seed; c["epochs"] = PILOT_EPOCHS
    for k, v in over.items():
        cur = c
        for kk in k.split(".")[:-1]:
            cur = cur[kk]
        cur[k.split(".")[-1]] = copy.deepcopy(v)
    return c


def arms(fe, rv, bank_arm=False):
    """[(stem, cfg, what)] in the committed file order; `what` goes into the header."""
    out = []
    for s in (0, 1):
        out.append((f"ab1_fe_mini_s{s}", pilot(fe, f"ab1_fe_mini_s{s}", s), f"1 family gate: VaaniFE-Mini, seed {s}"))
        out.append((f"ab1_refvalid_s{s}", pilot(rv, f"ab1_refvalid_s{s}", s), f"1 family gate: refvalid C16, seed {s}"))
    for tail in (0.0, 0.10):
        t = f"{int(round(tail * 100)):02d}"
        out.append((f"ab3b_tail{t}", pilot(fe, f"ab3b_tail{t}", 0, **{"data.mix.v2": dict(tail_share=tail)}),
                    f"3b low-ILD tail share {int(t)} %"))
    for inp in ("p", "pr_nhat", "pr_pld"):
        for s in (0, 1):
            over = {"model_cfg.inputs": inp}
            if inp == "pr_nhat":
                over["dsp"] = NHAT_DSP
            out.append((f"ab2_{inp}_s{s}", pilot(fe, f"ab2_{inp}_s{s}", s, **over), f"2 inputs {inp}, seed {s}"))
    for pa in (0.0, 0.3):
        for s in (0, 1):
            t = f"{int(round(pa * 100)):02d}"
            out.append((f"ab3_refdrop{t}_s{s}", pilot(fe, f"ab3_refdrop{t}_s{s}", s, **{"data.ref_corrupt.p_absent": pa}),
                        f"3 reference dropout (absent) {pa}, seed {s}"))
    for mask, df in (("bounded", 0), ("unbounded", 3), ("bounded", 3)):
        out.append((f"ab4_{mask}_df{df}", pilot(fe, f"ab4_{mask}_df{df}", 0, **{"model_cfg.mask": mask, "model_cfg.df_taps": df}),
                    f"4 mask {mask}, low-band DF taps {df}"))
    for k in (1.0, 2.0, 4.0):
        out.append((f"ab6_kappa{int(k)}", pilot(fe, f"ab6_kappa{int(k)}", 0, **{"loss_cfg.kappa": k}),
                    "6 loss: asymmetric term off (kappa 1)" if k == 1 else f"6 loss: kappa {k:g}"))
    if bank_arm:   # r7's bank under the r8 recipe: separates the bank change from the recipe change (baseline ab1_fe_mini_s0)
        out.append(("ab7_bank_r3", pilot(fe, "ab7_bank_r3", 0, **{"data.bank": BANK_R3}),
                    "7 RIR bank: bank_r3 (r7's bank), seed 0"))
    return out


def render(cfg, what):
    parent = "r8_refvalid_v2" if cfg["model"] == "vaani" else "r8_fe_mini"
    head = (f"# r8 ablation pilot ({what}). {PILOT_EPOCHS} epochs = 15 % of the full schedule, mixer v2 like the\n"
            f"# full runs; everything else as {parent}.yaml. See r8_ablations/README.md.\n")
    return head + yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--check", action="store_true", help="compare with the files on disk; write nothing")
    ap.add_argument("--bank-arm", action="store_true", help="include ab7_bank_r3 (val-biased: see the module doc)")
    ap.add_argument("--low-delay", action="store_true", help="the low-delay configurations (module doc)")
    ap.add_argument("--promote", help="--low-delay: comma-separated Stage-2 stems the Stage-2 decision promoted")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.low_delay:
        return ld_main(root, a.check, a.promote)
    if a.promote:
        ap.error("--promote needs --low-delay")
    d = root / OUT
    # once generated, the opt-in arm is a pilot like the rest: --check verifies it (not EXTRA), a rewrite keeps it current
    bank_arm = a.bank_arm or (d / "ab7_bank_r3.yaml").exists()
    want = {f"{stem}.yaml": render(cfg, what)
            for stem, cfg, what in arms(load(root, FULL["fe"]), load(root, FULL["refvalid"]), bank_arm)}
    if a.check:
        have = {p.name for p in d.glob("*.yaml")}
        # CRLF-normalised: a Windows checkout with core.autocrlf rewrites line ends, not content
        bad = sorted(f for f, t in want.items()
                     if not (d / f).exists() or (d / f).read_bytes().replace(b"\r\n", b"\n") != t.encode())
        extra = sorted(have - set(want))
        for f in bad:
            print(f"DIFFERS {OUT}/{f}")
        for f in extra:
            print(f"EXTRA   {OUT}/{f} (not generated)")
        print(f"{len(want) - len(bad)}/{len(want)} generated pilots match; {len(extra)} extra")
        return 1 if bad or extra else 0
    d.mkdir(parents=True, exist_ok=True)
    for f, t in want.items():
        (d / f).write_bytes(t.encode())
    print(f"wrote {len(want)} pilots to {OUT}")
    return 0



# ---- low-delay mode ------------------------------------------------------------------------------------------------
LD_OUT = "configs/retraining/r8_ld_ablations"
LD_FULL_DIR = "configs/retraining"
GATE0 = "results_r2/r8_ld/gate0/eligibility.json"
PROMOTED = "promoted.json"   # in LD_OUT, written by --promote; its presence keeps wave 2 in every later mode
# Arm A (Section 3.4, "Mini-P18"): the tiling, its deep-filter band and lag-matched taps, the validity term, the
# time-constant-matched GRU init; the reference ramp in samples (3072 = 12 legacy frames, framing-independent); the
# re-synthesis loss, with consistency off (it is zero on re-synthesized spectra up to rounding; Section 3.6)
LD_OVERLAY = {"model_cfg.freq_windows": "p18", "model_cfg.valid_bias": True, "model_cfg.df_bins": 96,
              "model_cfg.df_lags": [0, 3, 5], "model_cfg.df_taps": 3, "model_cfg.gru_init": "tc_matched",
              "dsp.ref_policy.ramp_samples": 3072, "loss_cfg.loss_domain": "resynthesis", "loss_cfg.w_consistency": 0.0}
LD_DROP = ("dsp.ref_policy.ramp_frames",)
ARM_B = {"model_cfg.freq_windows": "p32", "model_cfg.df_bins": 144, "model_cfg.df_lags": [0, 2, 4]}
ARM_R_DROP = ("model_cfg.freq_windows", "model_cfg.valid_bias")   # native tiling, P18's deep filter kept
# Stage 2 (Section 3.8), one addition each on Arm A seed 0; item 6 (continuity) is conditional and not generated
STAGE2 = {
    "ld_s2_overparam": ({"model_cfg.overparam": True}, "1 training-time over-parameterization"),
    "ld_s2_gru_default": ({"model_cfg.gru_init": "default"}, "2 GRU init: PyTorch default (control: tc_matched)"),
    "ld_s2_mrstft05": ({"loss_cfg.w_mrstft": 0.05}, "3 multi-resolution supervision, w_mrstft 0.05"),
    "ld_s2_warmup480": ({"data.warmup_samples": 7680, "data.crop_s": 4.48},
                        "4 past-context warm-up: 7,680-sample (480 ms) unscored prefix on a 4.48 s crop"),
    "ld_s2_native": ({"loss_cfg.loss_domain": "native", "loss_cfg.w_consistency": 0.3},
                     "5 native low-delay-domain loss (consistency 0.3 on the low-delay spectra)"),
}
# the ab2 input ablation on the low-delay path (owner decision D5; Rachit 2026-09-27: option 1): Arm B with n_hat from
# the decoupled-cadence NLMS against Arm B's own 'pr' seeds. Arm B only: Mini-P18 with n_hat exceeds the 60,000-entry
# budget (results_r2/r8/budget.md), Mini-P32 fits. Same NLMS/blocking/controller as ab2_pr_nhat, the ramp in samples.
LD_NHAT = {"model_cfg.inputs": "pr_nhat", "dsp.blocking": True, "dsp.controller": NHAT_DSP["controller"],
           "dsp.ref_policy.nlms": True}
# P4 (Section 3.10): r8 recipe ablations ported to Arm A, seed 0 screens; ab2 runs on Arm B (LD_NHAT above)
P4 = {"ld_p4_tail00": ({"data.mix.v2.tail_share": 0.0}, "3b low-ILD tail share 0 %"),
      "ld_p4_tail10": ({"data.mix.v2.tail_share": 0.10}, "3b low-ILD tail share 10 %"),
      "ld_p4_refdrop00": ({"data.ref_corrupt.p_absent": 0.0}, "3 reference dropout (absent) 0.0"),
      "ld_p4_refdrop30": ({"data.ref_corrupt.p_absent": 0.3}, "3 reference dropout (absent) 0.3"),
      "ld_p4_bounded": ({"model_cfg.mask": "bounded"}, "4 bounded mask"),
      "ld_p4_kappa1": ({"loss_cfg.kappa": 1.0}, "6 loss: asymmetric term off (kappa 1)"),
      "ld_p4_kappa2": ({"loss_cfg.kappa": 2.0}, "6 loss: kappa 2"),
      "ld_p4_kappa4": ({"loss_cfg.kappa": 4.0}, "6 loss: kappa 4")}
TIERS_LD = ("mid", "large", "large_plus")   # projections: native tiling (Jetson), not queued in r8


def derive(base, name, seed=None, epochs=None, over=None, drop=()):
    """A copy of `base` with dotted-key overrides set and dotted keys removed; everything else untouched."""
    c = copy.deepcopy(base); c["name"] = name
    if seed is not None:
        c["seed"] = seed
    if epochs is not None:
        c["epochs"] = epochs
    for k in drop:
        cur = c
        for kk in k.split(".")[:-1]:
            cur = cur[kk]
        cur.pop(k.split(".")[-1], None)
    for k, v in (over or {}).items():
        cur = c
        for kk in k.split(".")[:-1]:
            cur = cur.setdefault(kk, {})
        cur[k.split(".")[-1]] = copy.deepcopy(v)
    return c


def gate0_selection(root):
    """(Arm A contract id, record) from the Gate 0a eligibility report. A missing record, a failed Mini-P18 or a
    contract that is not one of Arm A's stops generation: nothing low-delay is derived without Gate 0a."""
    from vaani import audio_contract as ac
    p = root / GATE0
    if not p.exists():
        raise SystemExit(f"{GATE0} missing: run scripts/ld_gate0.py first (Gate 0a selects the support)")
    j = json.loads(p.read_text(encoding="utf-8"))
    sel = j.get("selection") or {}
    if sel.get("mini_p18_fails_8ms"):
        raise SystemExit(f"Gate 0a: Mini-P18 fails L = 8 ms; {sel.get('failure_action')}")
    cid = sel.get("support_contract")
    if cid not in ac.ARM_A_IDS:
        raise SystemExit(f"Gate 0a record selects {cid!r}, not one of Arm A's contracts {ac.ARM_A_IDS}")
    return cid, j


def deploy_path(g0, arm, cid):
    """The deployment paths (48 kHz I2S, resampler pair, ALSA period, D_proc) Gate 0a finds eligible for (arm,
    contract) on measured D_proc and verified periods; a pending note while the board record is incomplete."""
    if g0.get("status") != "complete":
        return f"pending: Gate 0a board record {g0.get('status')}"
    rows = [r for r in g0.get("eligibility", []) if r.get("arm") == arm and r.get("contract") == cid
            and r.get("eligible") and r.get("period_verified") and r.get("dproc_measured")]
    return [dict(resampler=r["resampler"], period_ms=r["period_ms"], dproc_ms=r["dproc_ms"], total_ms=r["total_ms"])
            for r in rows] or "none eligible at Gate 0a"


def ld_arms(fe, cid, g0, promoted=()):
    """[(relpath, cfg, what, record)] for every low-delay configuration plus C0 seeds 2-4 (from r8_fe_mini.yaml)."""
    from vaani import audio_contract as ac
    base = derive(fe, "r8_ld_fe_mini", over=dict(LD_OVERLAY, **{"model_cfg.audio_contract": cid}), drop=LD_DROP)
    arm_b = dict(ARM_B, **{"model_cfg.audio_contract": ac.ARM_B_ID})
    piloted_b = bool(((g0.get("selection") or {}).get("arm_b") or {}).get("piloted"))
    out = []

    def add(rel, cfg, what, **rec):
        out.append((rel, cfg, what, rec))

    F, A = LD_FULL_DIR, LD_OUT
    add(f"{F}/r8_ld_fe_mini.yaml", base, "the r8 low-delay product: Arm A at the Gate 0a support",
        kind="full", arm="arm_a", stage="full", priority=1, wave=1, speculative=True, stop_if=["stage1=arm_b", "stage2!=none"])
    add(f"{F}/r8_ld_fe_mini_overparam.yaml", derive(base, "r8_ld_fe_mini_overparam", over=STAGE2["ld_s2_overparam"][0]),
        "early full run (D8): Arm A with training-time over-parameterization", kind="full", arm="arm_a", stage="full",
        priority=1, wave=1, speculative=True, stop_if=["stage1=arm_b", "stage2!=ld_s2_overparam"])
    add(f"{F}/r8_ld_fe_mini_armb.yaml", derive(base, "r8_ld_fe_mini_armb", over=arm_b),
        "early full run (D8): Arm B (L = 10 ms only)", kind="full", arm="arm_b", stage="full", priority=1, wave=1,
        speculative=True, stop_if=["stage1=arm_a"], needs_gate0_arm_b=True)
    for t in TIERS_LD:
        add(f"{F}/r8_ld_fe_{t}.yaml", derive(base, f"r8_ld_fe_{t}", over={"model_cfg.tier": t}, drop=ARM_R_DROP),
            f"{t} tier projection (native tiling, Arm A's contract and deep filter); not trained in r8",
            kind="tier", arm="tier", stage="projection", queued=False)
    for s in range(5):
        add(f"{A}/ld_a_s{s}.yaml", derive(base, f"r8_ld_a_s{s}", s, PILOT_EPOCHS), f"Arm A, seed {s}",
            kind="pilot", arm="arm_a", stage="P1" if s < 2 else "P3", priority=2 if s < 2 else 3, wave=1,
            **({} if s < 2 else {"stop_if": ["stage1=arm_b"]}))
    for s in (0, 1):
        add(f"{A}/ld_b_s{s}.yaml", derive(base, f"r8_ld_b_s{s}", s, PILOT_EPOCHS, arm_b), f"Arm B (Mini-P32), seed {s}",
            kind="pilot", arm="arm_b", stage="P1", priority=2, wave=1, needs_gate0_arm_b=True)
        add(f"{A}/ld_b_nhat_s{s}.yaml", derive(base, f"r8_ld_b_nhat_s{s}", s, PILOT_EPOCHS, dict(arm_b, **LD_NHAT)),
            f"Arm B with inputs pr_nhat (the ab2 input ablation on the low-delay path, D5), seed {s}",
            kind="pilot", arm="arm_b_nhat", stage="P4", priority=4, wave=1, needs_gate0_arm_b=True,
            stop_if=["stage1=arm_a"])
        add(f"{A}/ld_r_s{s}.yaml", derive(base, f"r8_ld_r_s{s}", s, PILOT_EPOCHS, drop=ARM_R_DROP),
            f"Arm R (native tiling, Arm A's contract and deep filter; never selected), seed {s}",
            kind="pilot", arm="arm_r", stage="P1", priority=4, wave=1)
    for stem, (over, what) in STAGE2.items():
        add(f"{A}/{stem}.yaml", derive(base, "r8_" + stem, 0, PILOT_EPOCHS, over), f"Stage 2 item {what}, seed 0",
            kind="pilot", arm="arm_a", stage="P2", priority=2, wave=1, speculative=True)
    for s in (2, 3, 4):   # C0's confirmation seeds, exactly like the ab1_fe_mini pilots (seeds 0/1 in r8_ablations/)
        add(f"{A}/ab1_fe_mini_s{s}.yaml", pilot(fe, f"ab1_fe_mini_s{s}", s), f"C0 (VaaniFE-Mini, 32 ms), seed {s}",
            kind="pilot", arm="c0", stage="P3", priority=3, wave=1)
    for stem, (over, what) in P4.items():
        add(f"{A}/{stem}.yaml", derive(base, "r8_" + stem, 0, PILOT_EPOCHS, over), f"P4 on Arm A: {what}, seed 0",
            kind="pilot", arm="arm_a", stage="P4", priority=4, wave=1, stop_if=["stage1=arm_b"])
    if promoted:
        over = {}
        for p in promoted:
            if p not in STAGE2:
                raise SystemExit(f"--promote: {p} is not a Stage-2 arm ({sorted(STAGE2)})")
            over.update(STAGE2[p][0])
        if "ld_s2_native" in promoted and "ld_s2_warmup480" in promoted:
            raise SystemExit("--promote: the native-domain loss is not combined with the warm-up prefix")
        tag = "+".join(sorted(promoted))
        for s in range(5):
            add(f"{A}/ld_conf_s{s}.yaml", derive(base, f"r8_ld_conf_s{s}", s, PILOT_EPOCHS, over),
                f"confirmation of the promoted recipe ({tag}), seed {s}", kind="pilot", arm="arm_a", stage="P3",
                priority=3, wave=2)
        if set(promoted) != {"ld_s2_overparam"}:   # otherwise the early overparam full run is the recipe's full run
            add(f"{F}/r8_ld_fe_mini_conf.yaml", derive(base, "r8_ld_fe_mini_conf", over=over),
                f"full run of the promoted recipe ({tag})", kind="full", arm="arm_a", stage="full", priority=1, wave=2)
    return out


def _contract_rec(mc):
    from vaani import audio_contract as ac
    c = ac.get_audio_contract((mc or {}).get("audio_contract"))
    return dict(audio_contract=c.audio_contract_id, contract_hash=c.contract_hash, support_ms=c.algorithmic_delay_ms,
                hop=c.hop, legacy=c.is_legacy)


def arm_record(rel, cfg, rec, g0):
    """What Task 8 asks each arm to record: seed, scored and prefix exposure, optimizer budget, batch, precision,
    support, tiling and deployment audio path, plus the queue fields."""
    d, mc = cfg["data"], cfg.get("model_cfg") or {}
    items = cfg["epochs"] * d["epoch_len"]
    warm = int(d.get("warmup_samples", 0) or 0)
    crop_n = int(round(d["crop_s"] * 16000))
    con = _contract_rec(mc)
    arm = rec.get("arm")
    r = dict(config=rel, name=cfg["name"], seed=cfg["seed"], epochs=cfg["epochs"], batch=cfg["batch_size"],
             optimizer_steps=cfg["epochs"] * (d["epoch_len"] // cfg["batch_size"]),
             scored_exposure_s=items * (crop_n - warm) / 16000, prefix_exposure_s=items * warm / 16000,
             precision=dict(amp=cfg.get("amp", False), fp32_islands=bool(mc.get("fp32_islands", False)),
                            perf_numerics=(cfg.get("perf") or {}).get("numerics")),
             tier=mc.get("tier", "mini"), tiling=mc.get("freq_windows") or "native", **con,
             deploy_path=("C0: 32 ms legacy framing, not deployable at the low-delay budget" if con["legacy"]
                          else deploy_path(g0, "arm_b" if arm == "arm_b_nhat" else arm, con["audio_contract"])
                          if arm in ("arm_a", "arm_b", "arm_b_nhat", "arm_r") else "Jetson projection (not a Pi 5 deployment)"))
    r.update({k: v for k, v in rec.items()})
    r.setdefault("queued", True)
    # spec 6.2 binds the Pi tier's deployable arms; Arm R is a reference and the other tiers are Jetson projections
    r["deployable"] = arm in ("arm_a", "arm_b", "arm_b_nhat", "c0") and r["tier"] == "mini"
    return r


def ld_render(cfg, what, rel, g0status):
    kind = "pilot" if LD_OUT in rel else "configuration"
    ep = f"{cfg['epochs']} epochs" + (" = 15 % of the full schedule" if cfg["epochs"] == PILOT_EPOCHS else "")
    src = ("r8_fe_mini.yaml like the r8_ablations/ab1_fe_mini pilots" if cfg["name"].startswith("r8ab1")
           else "r8_ld_fe_mini.yaml (r8_fe_mini.yaml plus the gen_r8_configs.LD_OVERLAY)")
    head = (f"# r8 low-delay {kind}: {what}.\n# {ep}; everything else as {src}.\n"
            f"# Gate 0a support: {g0status}.\n"
            f"# Generated by scripts/gen_r8_configs.py --low-delay; see r8_ld_ablations/README.md.\n")
    return head + yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)


def ld_want(root, promoted=()):
    fe = load(root, FULL["fe"])
    cid, g0 = gate0_selection(root)
    sel = g0.get("selection") or {}
    status = f"{cid} ({'provisional: ' + str(g0.get('status')) if sel.get('provisional') else 'selected'})"
    files, records = {}, []
    for rel, cfg, what, rec in ld_arms(fe, cid, g0, promoted):
        files[rel] = ld_render(cfg, what, rel, status)
        records.append(arm_record(rel, cfg, rec, g0))
    # C0 seeds 0/1 and the full C0 are the existing configurations: recorded here, never rewritten by this mode
    for s in (0, 1):
        c = pilot(fe, f"ab1_fe_mini_s{s}", s)
        records.append(arm_record(f"{OUT}/ab1_fe_mini_s{s}.yaml", c, dict(kind="pilot", arm="c0", stage="P1", priority=2,
                                                                          wave=1, external=True), g0))
    records.append(arm_record(FULL["fe"], fe, dict(kind="full", arm="c0", stage="full", priority=1, wave=1, external=True), g0))
    man = dict(generated_by="scripts/gen_r8_configs.py --low-delay", gate0=dict(
        record=GATE0, status=g0.get("status"), support_contract=cid, provisional=bool(sel.get("provisional")),
        arm_b_piloted=bool((sel.get("arm_b") or {}).get("piloted"))), promoted=sorted(promoted), arms=records)
    files[f"{LD_OUT}/arms.json"] = json.dumps(man, indent=2, sort_keys=True) + "\n"
    if promoted:
        files[f"{LD_OUT}/{PROMOTED}"] = json.dumps({"stage2": sorted(promoted)}, indent=2) + "\n"
    return files


def ld_main(root, check, promote=None):
    pf = root / LD_OUT / PROMOTED
    if promote:
        promoted = tuple(sorted(x for x in promote.split(",") if x))
    elif pf.exists():
        promoted = tuple(json.loads(pf.read_text(encoding="utf-8"))["stage2"])
    else:
        promoted = ()
    want = ld_want(root, promoted)
    have = {p.relative_to(root).as_posix() for p in (root / LD_OUT).glob("*.*")} if (root / LD_OUT).exists() else set()
    have |= {p.relative_to(root).as_posix() for p in (root / LD_FULL_DIR).glob("r8_ld_*.yaml")}
    if check:
        bad = sorted(f for f, t in want.items()
                     if not (root / f).exists() or (root / f).read_bytes().replace(b"\r\n", b"\n") != t.encode())
        extra = sorted(f for f in have - set(want) if not f.endswith(".md"))
        for f in bad:
            print(f"DIFFERS {f}")
        for f in extra:
            print(f"EXTRA   {f} (not generated)")
        print(f"{len(want) - len(bad)}/{len(want)} generated low-delay files match; {len(extra)} extra")
        return 1 if bad or extra else 0
    (root / LD_OUT).mkdir(parents=True, exist_ok=True)
    for f, t in want.items():
        (root / f).write_bytes(t.encode())
    print(f"wrote {len(want)} low-delay files ({LD_OUT}, {LD_FULL_DIR}/r8_ld_*.yaml)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
