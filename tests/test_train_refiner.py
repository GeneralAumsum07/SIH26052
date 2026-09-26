"""Tier 4.6 refiner trainer: the loop is exercised end to end on CPU with a synthetic loader and screen, so the tests
prove the freezing, selection, resume and self-containment contracts without the 20k-item mixer."""
import json
from pathlib import Path

import numpy as np, pytest, torch, yaml

from vaani import export
from vaani import train_refiner as tr
from vaani.dsp import stft
from vaani.models.vaani_net import VaaniNet

MC = dict(df_order=3, film=False, coh=True)
N = 8000
ROOT = Path(__file__).resolve().parents[1]


def _first_stage(tmp_path):
    torch.manual_seed(0); m = VaaniNet(**MC)
    with torch.no_grad(): m.df.conv.weight.normal_(0, 0.05)
    cfg = {"model": "vaani", "controller_on": True, "dsp": {"limiter": True}, "model_cfg": MC, "seed": 0,
           "data": {"manifests": ["data/manifests/none.parquet"], "mix": {}}}
    p = tmp_path / "first.pt"; torch.save({"model": m.state_dict(), "config": cfg, "step": 0}, p); return p


def _batch(seed):
    g = torch.Generator().manual_seed(seed)
    clean = torch.randn(2, N, generator=g) * 0.05; mix = torch.stack([clean + 0.02 * torch.randn(2, N, generator=g), 0.03 * torch.randn(2, N, generator=g)], 1)
    T = stft.stft(clean).shape[2]
    return {"mix": mix, "clean": clean, "n_hat": mix[:, 1] * 0.5, "feats": torch.zeros(2, T, 18),
            "meta": [{"clean_bucket": False}, {"clean_bucket": True}]}


class _Sampler:
    def set_epoch(self, e): self.e = e


def _patch(monkeypatch, n_batches=2):
    monkeypatch.setattr(tr, "build_train_data", lambda *a, **k: (None, _Sampler(), [_batch(i) for i in range(n_batches)]))
    monkeypatch.setattr(tr, "screen_items", lambda root, split: (None, [0, 1, 2]))
    calls = {"n": 0}

    def fake_score(model, ds, idx, first_cfg, device):   # first call = anchor; each later call drifts upward so a winner exists
        calls["n"] += 1; base = np.array([[10.0, 0.90, 2.0]] * len(idx))
        return base + (0 if calls["n"] == 1 else np.array([0.2 * (calls["n"] - 1), 0.001, 0.02]))
    monkeypatch.setattr(tr, "score_items", fake_score)
    return calls


def _cfg(tmp_path, first, epochs=2):
    cfg = {"name": "ref", "model": "vaani_cascade", "base_checkpoint": str(first), "seed": 1, "epochs": epochs, "batch_size": 2,
           "num_workers": 0, "amp": False, "optim": {"lr": 1e-3, "warmup": 2, "clip": 1.0, "weight_decay": 1e-4},
           "val": {"eval_root": "x", "split": "val"}, "runs_dir": str(tmp_path / "runs"), "device": "cpu"}
    p = tmp_path / "ref.yaml"; p.write_text(yaml.safe_dump(cfg)); return p, cfg


def test_training_updates_only_the_refiner_and_keeps_first_stage_bytes(tmp_path, monkeypatch):
    first = _first_stage(tmp_path); _patch(monkeypatch); cfgp, _ = _cfg(tmp_path, first)
    info = tr.main(cfgp)
    ck = torch.load(tmp_path / "runs/ref/best.pt", weights_only=True); anchor = torch.load(first, weights_only=True)["model"]
    assert ck["config"]["model"] == "vaani_cascade" and ck["config"]["first_stage"]["sha256"] == tr._sha(first)
    firsts = {k[len("first."):]: v for k, v in ck["model"].items() if k.startswith("first.")}
    assert set(firsts) >= set(anchor) and all(torch.equal(firsts[k], anchor[k]) for k in anchor)   # params and BN buffers
    assert ck["model"]["refiner.c2.weight"].abs().sum() > 0 and info["steps"] == 4 and info["selected_epoch"] in (0, 1)


def test_data_builder_refuses_non_train_split():
    with pytest.raises(AssertionError):
        tr.build_train_data({"data": {"manifests": []}, "controller_on": True}, 0, 2, 0, split="val")


def test_selection_is_deterministic_and_early_stop_rule():
    h = [(0, 0.30, -0.001, 0.02), (1, 0.30, -0.001, 0.02), (2, 0.50, -0.004, 0.10), (3, 0.10, 0.0, 0.05)]
    assert tr.select_epoch(h) == 0                         # epoch 2 ineligible (STOI), 0/1 tie -> earliest; 3 scores lower (0.35 < 0.40)
    assert tr.select_epoch(list(reversed(h))) == 0
    assert tr.select_epoch([(0, 1.0, -0.01, 0.1)]) is None
    assert tr.should_stop_early([(0, 0.05, 0.0, 0.005)]) is False          # not yet two epochs
    assert tr.should_stop_early([(0, 0.05, 0.0, 0.005), (1, 0.08, 0.0, 0.009)]) is True
    assert tr.should_stop_early([(0, 0.05, 0.0, 0.005), (1, 0.12, 0.0, 0.0)]) is False
    assert tr.should_stop_early([(0, 0.5, -0.01, 0.1), (1, 0.5, -0.01, 0.1)]) is True   # gains that violate STOI do not count


def test_resume_checks_anchor_and_config_and_restores_state(tmp_path, monkeypatch):
    first = _first_stage(tmp_path); _patch(monkeypatch); cfgp, cfg = _cfg(tmp_path, first, epochs=1)
    tr.main(cfgp); last = torch.load(tmp_path / "runs/ref/last.pt", weights_only=True)
    assert last["epoch"] == 0 and "optim" in last and "sched" in last
    cfg["epochs"] = 2; cfgp.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="another anchor or config"):   # a changed training config is not the same run
        tr.main(cfgp)
    cfg["epochs"] = 1; cfgp.write_text(yaml.safe_dump(cfg))
    info = tr.main(cfgp)                                                    # same config: resumes past the finished epoch, no new steps
    assert info["steps"] == last["step"] and info["history"] == [tuple(h) for h in last["history"]]


def test_best_checkpoint_is_self_contained_and_exportable(tmp_path, monkeypatch):
    first = _first_stage(tmp_path); _patch(monkeypatch); cfgp, _ = _cfg(tmp_path, first)
    tr.main(cfgp); first.unlink()   # the first-stage file is gone; the cascade must still load and export
    onnx = export.export(tmp_path / "runs/ref/best.pt", tmp_path / "cascade.onnx")
    r = export.parity_and_timing(tmp_path / "runs/ref/best.pt", onnx, seconds=0.5)
    assert r["max_abs_err"] < 1e-4
    assert json.load(open(tmp_path / "runs/ref/run.json"))["params"] == 2498


def test_parallel_screen_scorer_matches_serial_loop(tmp_path, monkeypatch):
    """score_items fans DSP and metrics out to a pool; the numbers must equal the plain per-item loop."""
    from vaani.data.dataset import RenderedDataset
    from vaani.dsp import pipeline, stft
    from vaani import metrics
    first = _first_stage(tmp_path); cfg = torch.load(first, weights_only=True)["config"]
    model = VaaniNet(**MC); model.load_state_dict(torch.load(first, weights_only=True)["model"]); model.eval()
    ds = RenderedDataset("data/eval_r2/val"); idx = [0, 1]
    monkeypatch.setattr(tr, "SCREEN_WORKERS", 2)
    par = tr.score_items(model, ds, idx, cfg, torch.device("cpu"))
    ser = []
    with torch.no_grad():
        for i in idx:
            it = ds[i]; mix, clean = it["mix"].numpy(), it["clean"].numpy()
            r = pipeline.run(mix, controller_on=True, dsp_cfg=cfg["dsp"]); x = torch.from_numpy(r["mix"])[None]
            spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1)
            y = stft.istft(model(spec6, torch.from_numpy(r["features"])[None]).float(), length=mix.shape[1])[0].numpy()
            ser.append((metrics.snr_db(clean, y), metrics.stoi(clean, y), metrics.pesq_wb(clean, y)))
    assert np.allclose(par, np.asarray(ser), atol=1e-6)


def test_base_checkpoint_falls_back_to_the_tracked_copy(tmp_path, monkeypatch):
    import yaml
    from vaani.train_refiner import resolve_base_checkpoint
    from vaani.training_controls import verify_checkpoint_hash
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results_r2/runs/b").mkdir(parents=True); (tmp_path / "results_r2/runs/b/best.pt").write_bytes(b"x")
    assert Path(resolve_base_checkpoint("runs/b/best.pt")) == Path("results_r2/runs/b/best.pt")
    (tmp_path / "runs/b").mkdir(parents=True); (tmp_path / "runs/b/best.pt").write_bytes(b"y")
    assert resolve_base_checkpoint("runs/b/best.pt") == str(Path("runs/b/best.pt"))   # the local run wins
    # r7's refiner recipe now pins its backbone, so a clone's fallback is checked, not trusted
    monkeypatch.chdir(ROOT)
    cfg = yaml.safe_load(open("configs/retraining/r7_e256_wr64_refiner.yaml"))
    verify_checkpoint_hash("results_r2/runs/r7_e256_wr64/best.pt", cfg["base_checkpoint_sha256"])


def test_screen_pool_map_fails_loudly_instead_of_hanging(monkeypatch):
    from multiprocessing import TimeoutError as PoolTimeout

    class Stuck:
        terminated = False
        def map_async(self, f, xs, chunksize): return self
        def get(self, timeout): raise PoolTimeout()
        def terminate(self): Stuck.terminated = True

    monkeypatch.setattr(tr, "_pool", Stuck())
    with pytest.raises(RuntimeError, match="timed out"):
        tr._pool_map(tr._metric_item, [1, 2])
    assert Stuck.terminated and tr._pool is None   # the next screen gets a fresh pool
