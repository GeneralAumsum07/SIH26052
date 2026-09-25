"""r8 launch smoke (not training): <= 30 loop steps of a real r8 config with a tiny val, then a resume from last.pt,
plus a 30-step single-batch overfit that shows the loss falls. Needs the real data, so pytest skips it unless
VAANI_R8_SMOKE=1; normally run as a script from the repo root:
    uv run --with numba python tests/test_train_r8_smoke.py configs/retraining/r8_fe_mini.yaml --export
Writes runs/smoke_<name>/smoke_result.json. Smoke-only overrides: warmup 3 and lr 2e-3, so 30 steps move the weights
(the real 2,000-step warmup leaves lr ~1e-6 at step 30)."""
import json, os, shutil, sys, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def overfit(cfg, steps=30):
    """One fixed train batch, `steps` AdamW steps: the loss/gradient path must fall (loop batches are too noisy)."""
    import torch
    from vaani import losses, train
    from vaani.data.dataset import DynamicMixDataset, collate
    from vaani.data.mixer import MixConfig
    d = cfg["data"]
    dsk = dict(with_dsp=train.needs_dsp(cfg), controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"),
               pack_root=d.get("pack", "data/pack"), ref_corrupt=d.get("ref_corrupt"))
    if cfg["model"] == "vaani_fe":
        dsk["fe_inputs"] = True
    if d.get("exclude_groups_file"):
        dsk["exclude_groups_file"] = str(train._abs(d["exclude_groups_file"]))
    ds = DynamicMixDataset(d["manifests"], "train", d.get("bank"), MixConfig(**d.get("mix", {})), d.get("crop_s", 4.0),
                           8, cfg["seed"], **dsk)
    batch = collate([ds[i] for i in range(8)])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = train.build_model(cfg["model"], cfg.get("init_from"), cfg.get("model_cfg")).to(dev).train()
    lc = cfg.get("loss_cfg", {})
    loss_fn = (losses.build_loss("fe", lc) if cfg["loss"] == "fe" else losses.HybridLoss(**lc)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs, target, fw, is_clean = train.prepare_batch(batch, cfg["model"], dev)
    out = []
    for _ in range(steps):
        loss = loss_fn(model(*inputs).float(), target, fw, is_clean)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); out.append(loss.item())
    return out


def run(src, export=False, runs_dir="runs"):
    from vaani import train
    src = Path(src); cfg = yaml.safe_load(open(src))
    name = "smoke_" + cfg["name"]
    cfg.update(name=name, runs_dir=runs_dir, epochs=1, max_steps=30, num_workers=3, log_every=1)
    cfg["optim"].update(warmup=3, lr=2e-3)
    cfg["data"]["epoch_len"] = 15 * cfg["batch_size"]   # epoch 0 = steps 1..15; the resume run does 16..30
    cfg["val"] = dict(dynamic_items=16, select="composite", composite=dict(every=1, per_bucket=1, limit=6, ilds=[-8, -4, 0]))
    rd = Path(runs_dir) / name
    if rd.exists():
        shutil.rmtree(rd)
    cp = Path(runs_dir) / f"{name}.yaml"; cp.parent.mkdir(exist_ok=True)
    yaml.safe_dump(cfg, open(cp, "w"))
    t0 = time.time(); train.main(str(cp)); t1 = time.time()
    cfg["epochs"] = 2; yaml.safe_dump(cfg, open(cp, "w"))   # same max_steps schedule: resumes from last.pt
    train.main(str(cp)); t2 = time.time()

    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(rd)); ea.Reload()
    loss = [v.value for v in ea.Scalars("train/loss")]
    last = torch.load(rd / "last.pt", weights_only=True); best = torch.load(rd / "best.pt", weights_only=True)
    info = json.loads((rd / "run.json").read_text())
    ov = overfit(cfg)
    m = lambda xs: sum(xs) / len(xs)
    out = dict(label="smoke (loaded laptop, RTX 5060); not a training result", config=str(src), smoke_config=str(cp),
               steps=last["step"], epochs_done=last["epoch"] + 1, resumed_from_step=15,
               loop_loss=[round(v, 3) for v in loss], loop_loss_first10_mean=m(loss[:10]), loop_loss_last10_mean=m(loss[-10:]),
               loss_finite=all(v == v and abs(v) < 1e9 for v in loss) and all(v == v for v in ov),
               overfit_loss=[round(v, 3) for v in ov], overfit_fell=m(ov[-5:]) < m(ov[:5]),
               history=[{k: h[k] for k in h if k in ("epoch", "step", "val_stoi", "val_stoi_ema")} |
                        {f"{c}_{k}": h[c][k] for c in ("composite_raw", "composite_ema") if c in h
                         for k in ("pass_rate", "stoi", "ild_loss_max", "d_snr_mono", "d_snr_web", "passes")}
                        for h in info["history"]],
               best_weights=best.get("weights"), best_selection=best.get("selection"), best_key=info.get("best_key"),
               last_has_ema="ema" in last, wall_first_s=round(t1 - t0, 1), wall_resume_s=round(t2 - t1, 1))
    if export:
        from vaani import export as E
        rep = E.export_fe(E.fe_load(rd / "best.pt"), rd / "export" / "fe_mini.onnx")
        out["export"] = {k: rep[k] for k in ("onnx", "folded", "onnx_sha256", "folded_sha256", "inputs", "outputs")}
        out["export"]["parity"] = rep.get("parity")
    print("SMOKE_RESULT " + json.dumps(out, default=str))
    json.dump(out, open(rd / "smoke_result.json", "w"), indent=2, default=str)
    return out


def test_r8_fe_mini_smoke():
    import pytest   # imported here so the script runs in the project venv, which has no pytest
    if os.environ.get("VAANI_R8_SMOKE") != "1":
        pytest.skip("real-data smoke; set VAANI_R8_SMOKE=1")
    out = run(ROOT / "configs/retraining/r8_fe_mini.yaml", export=True)
    assert out["steps"] == 30 and out["loss_finite"] and out["overfit_fell"] and out["best_selection"] == "composite"


if __name__ == "__main__":
    run(sys.argv[1], export="--export" in sys.argv)
