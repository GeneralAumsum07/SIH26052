import math
import pandas as pd
import pytest
from types import SimpleNamespace

from scripts import refiner_frontier as frontier


def test_frontier_keeps_metric_validity_and_cost_tradeoffs():
    rows = [dict(system=name, id="1", bucket="stationary_0", noise_class="stationary",
                 snr_in=0, clipped=False, ref_dropout=False, fault=None,
                 snr_out=snr, stoi=.8, pesq_wb=float("nan"), matrix_mmacs_per_second=cost)
            for name, snr, cost in [("small", 10, 20), ("large", 11, 40), ("dominated", 9, 40)]]
    result = frontier.summarize(rows)
    severe = {r["system"]: r for r in result if r["envelope"] == "severe_stationary"}
    assert severe["small"]["pareto_snr"] and severe["large"]["pareto_snr"]
    assert not severe["dominated"]["pareto_snr"]
    assert severe["small"]["n_pesq_wb"] == 0 and math.isnan(severe["small"]["pesq_wb"])


def test_frontier_nan_quality_is_not_a_winner():
    assert frontier.pareto_flags([dict(snr_out=float("nan"), matrix_mmacs_per_second=1)]) == [False]


def test_existing_rejects_duplicate_items_before_loading_checkpoint(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame([dict(system="ckpt:missing.pt", bucket="b", id="1")] * 2).to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate observations"):
        frontier.existing(SimpleNamespace(csvs=[str(p)]))


def test_sweep_protects_test_split_before_loading_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        frontier.main(["sweep", "--checkpoint", "missing.pt", "--split", "test", "--out", str(tmp_path)])


def test_sweep_writes_quality_and_activation_frontier_without_training(tmp_path, monkeypatch):
    import json
    import numpy as np
    import soundfile as sf
    import torch
    from vaani.models.cascade import FrozenCascade
    cfg = dict(model="vaani_cascade", model_cfg=dict(channels=8, noise_floor=True),
               refiner_cfg=dict(hidden=5, past=4), controller_on=True)
    model = FrozenCascade.from_config(cfg)
    checkpoint = tmp_path / "cascade.pt"
    torch.save(dict(config=cfg, model=model.state_dict()), checkpoint)
    bucket = tmp_path / "val" / "stationary_0"
    bucket.mkdir(parents=True)
    wave = np.random.default_rng(0).normal(0, .03, 1600).astype(np.float32)
    sf.write(bucket / "0000.mix.wav", np.stack([wave, wave], -1), 16000)
    sf.write(bucket / "0000.clean.wav", wave, 16000)
    (bucket / "0000.json").write_text(json.dumps(dict(noise_class="stationary", snr_db=0, clipped=False, ref_dropout=False)))
    monkeypatch.setattr(frontier.metrics, "stoi", lambda *args: .8)
    monkeypatch.setattr(frontier.metrics, "pesq_wb", lambda *args: 2.)
    out = tmp_path / "out"
    frontier.main(["sweep", "--checkpoint", str(checkpoint), "--eval-root", str(tmp_path),
                   "--thresholds", "12", "--out", str(out)])
    stats = json.loads((out / "provenance.json").read_text())["stats"]
    assert stats["always"]["fire_rate"] == 1 and stats["bypass"]["fire_rate"] == 0
    assert stats["always"]["total"] == stats["bypass"]["total"] > 0
    assert len(pd.read_csv(out / "items.csv")) == 3
    assert set(pd.read_csv(out / "frontier.csv").envelope) == {"nominal", "severe_stationary"}
