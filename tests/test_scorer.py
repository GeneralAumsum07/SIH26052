"""Validation beside training (plan Task 4b): async scoring equals synchronous validation."""
import copy
import json

import pytest
import torch
import yaml

from vaani import scorer, train
from tests.test_low_delay_training import _cfg
from tests.test_train_smoke import _tiny


def _run(tmp_path, cfg):
    cp = tmp_path / f"{cfg['name']}.yaml"
    yaml.safe_dump(cfg, open(cp, "w"))
    train.main(str(cp))
    return tmp_path / "runs" / cfg["name"]


def _strip(h):
    return [{k: v for k, v in r.items()} for r in h]


@pytest.mark.parametrize("workers", [False, True])
@pytest.mark.parametrize("ema", [None, {"decay": 0.9}])
def test_async_history_and_best_equal_sync(tmp_path, ema, workers):
    m = _tiny(tmp_path)
    base = _cfg(tmp_path, m, name="sync", epochs=3, max_steps=6, ema=ema)
    base["data"]["epoch_len"] = 4
    rs = _run(tmp_path, base)
    a = copy.deepcopy(base); a["name"] = "async"
    a["perf"] = dict(a["perf"], ops={"scorer": "async", "priority": 1})
    ra = _run(tmp_path, a)
    assert (ra / "train_done.json").exists() and not (ra / "scored_done.json").exists()
    if workers:   # a measure worker takes every snapshot first; the scorer then only applies the selection rule
        w = scorer.Scorer([ra], "cpu")
        while w.measure_step():
            pass
        assert len(list((ra / "snapshots").glob("meas_*.json"))) == 3
    sc = scorer.Scorer([ra], "cpu")
    sc.run(0.0, until_done=True, max_idle=2)
    assert (ra / "scored_done.json").exists() and not list((ra / "snapshots").glob("*"))
    hs, ha = json.loads((rs / "run.json").read_text()), json.loads((ra / "run.json").read_text())
    assert _strip(hs["history"]) == _strip(ha["history"])
    assert hs["best_val_stoi"] == ha["best_val_stoi"]
    bs, ba = torch.load(rs / "best.pt", weights_only=True), torch.load(ra / "best.pt", weights_only=True)
    assert bs["step"] == ba["step"] and bs.get("weights") == ba.get("weights")
    for k in bs["model"]:
        assert torch.equal(bs["model"][k], ba["model"][k]), k


def test_scorer_restart_resumes_without_rescoring(tmp_path, monkeypatch):
    m = _tiny(tmp_path)
    a = _cfg(tmp_path, m, name="restart", epochs=3, max_steps=6, ema=None)
    a["data"]["epoch_len"] = 4
    a["perf"] = dict(a["perf"], ops={"scorer": "async"})
    ra = _run(tmp_path, a)
    sc = scorer.Scorer([ra], "cpu")
    assert sc.step()                       # scores snapshot 1 only, then "crashes"
    calls = []
    orig = scorer.Selector.measure

    def counting(self, row, cands, vdl, n_points, final):
        calls.append(n_points)
        return orig(self, row, cands, vdl, n_points, final)
    monkeypatch.setattr(scorer.Selector, "measure", counting)
    sc2 = scorer.Scorer([ra], "cpu")        # a fresh scorer process
    sc2.run(0.0, until_done=True, max_idle=2)
    assert calls == [2, 3]                 # neither re-scored nor skipped
    h = json.loads((ra / "run.json").read_text())["history"]
    assert all("val_stoi" in r for r in h) and len(h) == 3


def test_async_refused_with_patience(tmp_path):
    m = _tiny(tmp_path)
    a = _cfg(tmp_path, m, name="pat")
    a["perf"] = dict(a["perf"], ops={"scorer": "async"})
    a["early_stopping"] = {"patience": 2}
    with pytest.raises(ValueError, match="patience"):
        _run(tmp_path, a)


def test_priority_order():
    class R:
        def __init__(self, rd, items):
            self.run_dir, self.items = rd, items

        def pending(self):
            return self.items
    sc = scorer.Scorer([], "cpu", {"a": 3, "b": 1})
    import pathlib, os, tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    pa, pb = d / "a.pt", d / "b.pt"
    pa.write_bytes(b"x"); pb.write_bytes(b"x")
    os.utime(pb, (pa.stat().st_mtime + 5,) * 2)     # b arrived later but has the higher priority
    sc.runs = {"a": R(d, [(1, pa)]), "b": R(d, [(1, pb)])}
    assert [it[3] for it in sc.queue()] == ["b", "a"]
