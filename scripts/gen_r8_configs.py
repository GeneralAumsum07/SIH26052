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
"""
import argparse, copy, sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = "configs/retraining/r8_ablations"
FULL = {"fe": "configs/retraining/r8_fe_mini.yaml", "refvalid": "configs/retraining/r8_refvalid_v2.yaml"}
PILOT_EPOCHS = 48   # 15 % of the 320-epoch full schedule (960,000 items, 30,000 steps)
# the pr_nhat input needs n_hat, so that arm runs the worker DSP (fixed NLMS, robust kernel, blocking, ref_policy)
NHAT_DSP = dict(limiter=True, blocking=True, controller=dict(block_margin_db=10.0, diff_jump_max_db=3.0),
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
    a = ap.parse_args(argv)
    root = Path(a.root)
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


if __name__ == "__main__":
    sys.exit(main())
