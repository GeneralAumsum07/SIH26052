import json, math, time
import pytest
import numpy as np, soundfile as sf, torch, yaml
from pathlib import Path
from vaani.data import manifests
from vaani.data.dataset import DynamicMixDataset, EpochSampler, collate
from vaani.data.mixer import MixConfig
from vaani.dsp.features import N_FEATURES
from vaani.models.vaani_net import VaaniNet
from vaani import losses, train


def _tiny(tmp_path):
    rows = []
    for i in range(3):
        p = tmp_path / f"s{i}.flac"; sf.write(p, np.random.randn(32000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"s{i}", corpus="t", kind="speech", group_id=f"g{i}", speaker_id=f"g{i}", path=str(p), duration_s=2.0, licence="", split="train", sha1="", noise_class=""))
        rows.append({**rows[-1], "source_id": f"v{i}", "split": "val"})
    for i in range(2):
        p = tmp_path / f"n{i}.flac"; sf.write(p, np.random.randn(48000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"n{i}", corpus="t", kind="noise", group_id=f"ng{i}", speaker_id="", path=str(p), duration_s=3.0, licence="", split="train", sha1="", noise_class="stationary"))
        rows.append({**rows[-1], "source_id": f"vn{i}", "split": "val"})
    m = tmp_path / "m.parquet"; manifests.write(rows, m); return m


def test_two_steps_each_model(tmp_path):
    m = _tiny(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    for model in ("gtcrn", "vaani"):
        cfg = dict(name=f"smoke_{model}", model=model, controller_on=True, loss="hybrid",
                   init_from="vaani/models/checkpoints/model_trained_on_dns3.tar",
                   data=dict(manifests=[str(m)], bank=None, crop_s=1.0, epoch_len=4, mix={"p_room": 0.0}),
                   val=dict(dynamic_items=2),
                   optim=dict(lr=1e-4, lr_new=1e-3, warmup=1, clip=5.0), batch_size=2, epochs=1, max_steps=2,
                   amp=False, device=device, runs_dir=str(tmp_path / "runs"), num_workers=0, seed=0)
        cp = tmp_path / f"{model}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
        train.main(str(cp))
        rd = tmp_path / "runs" / f"smoke_{model}"
        assert (rd / "last.pt").exists() and (rd / "best.pt").exists()
        ck = torch.load(rd / "last.pt", weights_only=True)
        assert ck["step"] == 2 and {"model", "config", "step", "optim", "sched", "epoch"} <= set(ck)
        assert set(torch.load(rd / "best.pt", weights_only=True)) == {"model", "config", "step"}
        info = json.loads((rd / "run.json").read_text())
        for k in ("name", "config", "git_sha", "manifest_hash", "params", "best_val_stoi", "steps", "wall_s", "torch", "cuda"):
            assert k in info, k
        assert info["steps"] == 2 and info["wall_s"] > 0 and np.isfinite(info["best_val_stoi"])
        assert info["best_metric"] == "stoi_dynamic_val" and Path(info["init_from"]).is_absolute()
        assert not (rd / "last.tmp").exists()
    # A longer cosine budget is a new experiment, not a resume of optimizer time.
    cfg.update(max_steps=3, epochs=2); yaml.safe_dump(cfg, open(cp, "w"))
    with pytest.raises(RuntimeError, match="changed cosine schedule"):
        train.main(str(cp))
    ck = torch.load(rd / "last.pt", weights_only=True)
    assert ck["step"] == 2 and "optim" in ck and "sched" in ck
    print(f"smoke wall {time.time() - t0:.1f}s on {device}")


def test_build_model_vaani_resumes_from_vaani_checkpoint(tmp_path):
    m = VaaniNet()
    ck = tmp_path / "last.pt"
    torch.save({"model": m.state_dict(), "config": {"model": "vaani"}, "step": 5}, ck)
    m2 = train.build_model("vaani", str(ck))
    sd, sd2 = m.state_dict(), m2.state_dict()
    assert sd.keys() == sd2.keys()
    assert all(torch.equal(sd[k], sd2[k]) for k in sd)


def test_dataset_with_dsp_runs_pipeline_in_getitem(tmp_path):
    m = _tiny(tmp_path)
    ds = DynamicMixDataset([m], "train", None, MixConfig(p_room=0.0), crop_s=1.0, epoch_len=2, seed=0, with_dsp=True)
    b = collate([ds[0], ds[1]])
    assert b["n_hat"].shape == (2, 16000) and b["feats"].shape == (2, 63, N_FEATURES)
    plain = DynamicMixDataset([m], "train", None, MixConfig(p_room=0.0), crop_s=1.0, epoch_len=2, seed=0)
    assert "n_hat" not in plain[0]


def _epoch_first_mix(m, epoch):
    # num_workers=2 forces spawn on Windows: persistent workers must still see the new epoch
    ds = DynamicMixDataset([m], "train", None, MixConfig(p_room=0.0), crop_s=0.5, epoch_len=2, seed=0)
    sampler = EpochSampler(len(ds)); sampler.set_epoch(epoch)
    dl = torch.utils.data.DataLoader(ds, 2, sampler=sampler, collate_fn=collate, num_workers=2, persistent_workers=True)
    return next(iter(dl))["mix"][0]


def test_epoch_sampler_changes_mixtures_across_persistent_workers(tmp_path):
    m = _tiny(tmp_path)
    e0, e1, e0b = _epoch_first_mix(m, 0), _epoch_first_mix(m, 1), _epoch_first_mix(m, 0)
    assert not torch.equal(e0, e1) and torch.equal(e0, e0b)


def test_frame_weights_from_meta():
    metas = [{"impulse_onsets_s": [0.5]}, {"impulse_onsets_s": []}]
    w = train.frame_weights_from_meta(metas, 63, burst_weight=3.0)
    assert w.shape == (2, 63) and (w[1] == 1).all()
    k = int(0.5 * 16000 / 256)
    assert w[0, k] == 3.0 and w[0, 0] == 1.0 and w[0, -1] == 1.0


def test_losses_shapes_and_clean_term():
    pred, true = torch.randn(2, 257, 10, 2), torch.randn(2, 257, 10, 2)
    h = losses.HybridLoss()(pred, true)
    assert h.ndim == 0 and torch.isfinite(h)
    assert torch.allclose(losses.HybridLoss()(pred, true, torch.ones(2, 10)), h)
    sp = losses.SpeechPreservationLoss(clean_l1=1.0)
    assert torch.allclose(sp(pred, true, None, torch.tensor([False, False])), h)
    assert sp(pred, true, None, torch.tensor([True, False])) > h


def test_df_param_group_gets_own_lr_and_clip():
    import torch
    from vaani.models.vaani_net import VaaniNet
    from vaani.train import build_param_groups, clip_groups
    m = VaaniNet(df_order=3, film=False, coh=True)
    groups = build_param_groups(m, dict(lr=1e-4, lr_df=1e-3, clip_df=1.0))
    assert [g["lr"] for g in groups] == [1e-4, 1e-3] and groups[1]["params"] == list(m.df.parameters())
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))
    # without lr_df the head stays in the shared group, as in r3
    assert len(build_param_groups(m, dict(lr=1e-4))) == 1
    for p in m.parameters(): p.grad = torch.full_like(p, 10.0)
    clip_groups(groups, 5.0)
    df_norm = torch.stack([p.grad.norm() for p in m.df.parameters()]).norm()
    rest = torch.stack([p.grad.norm() for p in groups[0]["params"]]).norm()
    assert abs(df_norm - 1.0) < 1e-4 and abs(rest - 5.0) < 1e-3


# --- r8: VaaniFE, FE loss, EMA and composite selection ---
from vaani.training_controls import composite_key, composite_summary, ema_decay


def _rendered_val(tmp_path, n_items=2):
    """A tiny eval_root/val: two buckets, 1 s clips (enough for the composite screen's constructions)."""
    root = tmp_path / "eval"
    for b in ("stationary_0", "clean_inf"):
        (root / "val" / b).mkdir(parents=True)
        for k in range(n_items):
            g = np.random.default_rng(k)
            c = (g.standard_normal(16000) * 0.1).astype(np.float32)
            x = np.stack([c + 0.05 * g.standard_normal(16000), 0.3 * c + 0.05 * g.standard_normal(16000)]).astype(np.float32)
            sf.write(root / "val" / b / f"{k:03d}.mix.wav", x.T, 16000); sf.write(root / "val" / b / f"{k:03d}.clean.wav", c, 16000)
            json.dump({"snr_db": 0, "clean_bucket": b == "clean_inf"}, open(root / "val" / b / f"{k:03d}.json", "w"))
    return root


def _fe_cfg(tmp_path, m, **kw):
    cfg = dict(name="smoke_fe", model="vaani_fe", model_cfg={"tier": "mini", "inputs": "pr"}, controller_on=True,
               loss="fe", loss_cfg={"kappa": 3.0, "w_mrstft": 0.05, "w_phase": 0.05},
               data=dict(manifests=[str(m)], bank=None, crop_s=1.0, epoch_len=4, mix={"p_room": 0.0},
                         ref_corrupt={"p": 0.15, "p_absent": 0.5}),
               dsp={"limiter": True, "ref_policy": {"absent": "freeze", "ramp_frames": 12}},
               val=dict(dynamic_items=2), ema={"decay": 0.9},
               optim=dict(lr=1e-3, warmup=1, clip=5.0), batch_size=2, epochs=1, max_steps=2,
               amp=False, device="cpu", runs_dir=str(tmp_path / "runs"), num_workers=0, seed=0)
    cfg.update(kw); return cfg


def test_vaani_fe_ema_stoi_selection_and_resume(tmp_path):
    m = _tiny(tmp_path)
    cfg = _fe_cfg(tmp_path, m, epochs=1, max_steps=4)
    cp = tmp_path / "fe.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
    train.main(str(cp))
    rd = tmp_path / "runs" / "smoke_fe"
    ck = torch.load(rd / "last.pt", weights_only=True)
    assert ck["step"] == 2 and "ema" in ck and ck["ema"].keys() == ck["model"].keys()
    assert not all(torch.equal(ck["ema"][k], ck["model"][k]) for k in ck["model"] if ck["model"][k].is_floating_point())
    best = torch.load(rd / "best.pt", weights_only=True)
    assert best["weights"] in ("raw", "ema") and best["selection"] == "stoi"
    h = json.loads((rd / "run.json").read_text())["history"]
    assert "val_stoi_ema" in h[0] and np.isfinite(h[0]["val_stoi_ema"])
    # resume from last.pt: same schedule (max_steps), more epochs -> continues at step 2 to 4
    cfg["epochs"] = 3; yaml.safe_dump(cfg, open(cp, "w"))
    train.main(str(cp))
    ck2 = torch.load(rd / "last.pt", weights_only=True)
    assert ck2["step"] == 4 and ck2["epoch"] == 1 and len(ck2["history"]) == 2
    # the checkpoint rebuilds through export.fe_load's contract
    from vaani.models import vaani_fe
    fm = vaani_fe.from_arch(best["config"]["model_cfg"]); fm.load_state_dict(best["model"])


def test_vaani_fe_composite_selection(tmp_path):
    m = _tiny(tmp_path); root = _rendered_val(tmp_path)
    cfg = _fe_cfg(tmp_path, m, name="smoke_fe_comp",
                  val=dict(dynamic_items=2, select="composite",
                           composite={"every": 2, "per_bucket": 2, "ilds": [-8, 0]}, eval_root=str(root), split="val"))
    cp = tmp_path / "fec.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
    train.main(str(cp))
    rd = tmp_path / "runs" / "smoke_fe_comp"
    best = torch.load(rd / "best.pt", weights_only=True)
    assert best["selection"] == "composite" and best["weights"] in ("raw", "ema")
    s = best["composite"]
    assert set(s["ild_loss"]) == {"ild_-8", "ild_0"} and s["n_clips"] == 2 and 0 <= s["pass_rate"] <= 1
    h = json.loads((rd / "run.json").read_text())
    assert "composite_raw" in h["history"][0] and "composite_ema" in h["history"][0]   # final point always screens
    assert h["best_metric"] == "composite_val" and len(h["best_key"]) == 3
    assert (rd / "composite_baseline.json").exists()


def test_composite_summary_known_answer():
    def row(cond, snr_out, stoi=None, pesq=None, loss=0.0, snr_in=0.0, ci=False):
        return dict(cond=cond, clean_item=ci, snr_in=snr_in, snr_out=snr_out, stoi=stoi, pesq=pesq, speech_loss=loss)
    rows = [row("present", 16, .9, 2.6), row("present", 16, .9, 2.4), row("present", 30, .99, 4.0, ci=True),
            row("ild_-8", 5, loss=0.02), row("ild_-8", 5, loss=0.04), row("ild_0", 0, loss=0.5),
            row("mono", 8), row("web_stereo", 9)]
    s = composite_summary(rows, base_d_snr=8.5)
    assert s["pass_rate"] == 0.5 and s["n_clips"] == 2 and math.isclose(s["ild_loss"]["ild_-8"], 0.03)
    assert s["ild_loss_max"] == 0.5 and not s["filters"]["ild"] and s["filters"]["mono"] and s["filters"]["web_stereo"]
    assert not s["passes"] and composite_key(s) == (0, 0.5, 0.9)
    s2 = composite_summary([r for r in rows if r["cond"] != "ild_0"], base_d_snr=10.0)
    assert s2["filters"]["ild"] and not s2["filters"]["mono"] and s2["filters"]["web_stereo"]
    s3 = composite_summary([r for r in rows if r["cond"] != "ild_0"], base_d_snr=None)
    assert s3["passes"] and composite_key(s3) > composite_key(s)
    assert ema_decay(0.999, 0) == 0.1 and ema_decay(0.999, 10 ** 6) == 0.999 and ema_decay(0.9, 0, warmup=False) == 0.9


def test_prepare_batch_vaani_fe_inputs():
    b = {"mix": torch.randn(2, 2, 16000) * 0.1, "clean": torch.randn(2, 16000) * 0.1, "meta": [{}, {}],
         "ref_avail": torch.ones(2, 63)}
    inputs, target, fw, _ = train.prepare_batch(b, "vaani_fe", "cpu")
    assert inputs[0].shape == (2, 257, 63, 4) and inputs[1] is None and inputs[2].shape == (2, 63)
    b["n_hat"] = torch.zeros(2, 16000)
    assert train.prepare_batch(b, "vaani_fe", "cpu")[0][0].shape == (2, 257, 63, 6)
    fe = train.build_model("vaani_fe", model_cfg={"tier": "mini", "inputs": "pr_nhat"})
    assert fe(*train.prepare_batch(b, "vaani_fe", "cpu")[0]).shape == (2, 257, 63, 2)
    assert train.needs_dsp({"model": "vaani_fe", "model_cfg": {"inputs": "pr_nhat"}})
    assert not train.needs_dsp({"model": "vaani_fe", "model_cfg": {"inputs": "pr"}}) and train.needs_dsp({"model": "vaani"})
