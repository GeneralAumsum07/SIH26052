import json, time
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
    # resume: a second main() on the same run_dir continues from step 2 into epoch 1
    cfg.update(max_steps=3, epochs=2); yaml.safe_dump(cfg, open(cp, "w")); train.main(str(cp))
    ck = torch.load(rd / "last.pt", weights_only=True)
    assert ck["step"] == 3 and "optim" in ck and "sched" in ck
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
