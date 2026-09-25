import json
import re
from pathlib import Path

import numpy as np
import pytest

from vaani.data import rirs

ROOT = Path(__file__).resolve().parents[1]


def _dims(rng, k=200):
    return {tuple(np.round(rirs.draw_room_params(rng)["dims"], 9)) for _ in range(k)}


def test_train_namespace_disjoint_from_legacy_and_eval():
    # r8 training rooms must not repeat a room of the legacy stream (bank.npz / bank_r3) or of an eval bank
    tr = _dims(rirs.bank_rng(8, "train"))
    assert not tr & _dims(rirs.bank_rng(8)) and not tr & _dims(rirs.bank_rng(8, "eval"))
    assert not tr & _dims(rirs.bank_rng(0)) and not tr & _dims(rirs.bank_rng(2610, "eval"))
    assert rirs.bank_rng(8, "train").random() == rirs.bank_rng(8, "train").random()


def test_m6_bank_records_radius_and_namespace_and_loads_mmapped(tmp_path, monkeypatch):
    monkeypatch.setattr(rirs, "ARMOURED_RAYS", 40)   # 1440 rays at 0.05 m: cheap, the plumbing is what is checked
    p = tmp_path / "bank_r8.npz"
    rirs.build_bank(p, n=3, seed=8, n_noise=1, armoured_frac=1 / 3, max_len=2000, workers=1,
                    receiver_radius=rirs.M6_RECEIVER_RADIUS, seed_namespace="train")
    z = np.load(p)
    assert float(z["receiver_radius"]) == pytest.approx(0.05) and str(z["seed_namespace"]) == "train"
    assert z["armoured"].sum() == 1 and z["speech"].shape == (3, 2, 2000) and z["noise"].shape == (3, 1, 2, 2000)
    b = rirs.RirBank(p)
    assert isinstance(b.noise, np.memmap) and len(b) == 3
    for k in rirs.RirBank.KEYS:
        assert (tmp_path / f"bank_r8.{k}.npy").exists()
    s = b.sample(np.random.default_rng(0))
    assert s["speech"].shape == (2, 2000) and np.isfinite(s["speech"]).all() and (np.abs(b.speech).sum(axis=(1, 2)) > 0).all()


def test_parts_build_resumes_and_matches_one_shot(tmp_path, monkeypatch):
    kw = dict(n=7, seed=3, n_noise=1, max_len=1500, workers=1, seed_namespace="train")
    rirs.build_bank(tmp_path / "ref.npz", **kw)
    calls = {"n": 0}
    orig = rirs.simulate_from_params

    def dies_after_five(*a, **k):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("killed")
        return orig(*a, **k)
    monkeypatch.setattr(rirs, "simulate_from_params", dies_after_five)
    parts = tmp_path / "parts"
    with pytest.raises(RuntimeError):
        rirs.build_bank(tmp_path / "b.npz", parts_dir=parts, part_size=2, **kw)
    # rooms 0-3 were saved as two blocks; room 4 finished but its block was still open
    assert sorted(f.name for f in parts.glob("part_*.npz")) == ["part_00000.npz", "part_00002.npz"]
    assert not list(parts.glob("*.tmp.npz")) and not (tmp_path / "b.npz").exists()
    calls["n"] = -100
    rirs.build_bank(tmp_path / "b.npz", parts_dir=parts, part_size=2, **kw)
    assert calls["n"] == -100 + 3   # only rooms 4-6 simulated again
    assert (tmp_path / "b.npz").read_bytes() == (tmp_path / "ref.npz").read_bytes() and not parts.exists()


def test_parts_dir_of_another_build_is_refused(tmp_path):
    parts = tmp_path / "parts"
    rirs.build_bank(tmp_path / "a.npz", n=2, seed=1, n_noise=1, max_len=1000, workers=1, parts_dir=parts)
    parts.mkdir(); (parts / "spec.json").write_text(json.dumps({"n": 2, "seed": 1}))
    with pytest.raises(ValueError, match="different build"):
        rirs.build_bank(tmp_path / "b.npz", n=2, seed=2, n_noise=1, max_len=1000, workers=1, parts_dir=parts)


def test_legacy_bank_bytes_unchanged_by_parts_path(tmp_path):
    rirs.build_bank(tmp_path / "a.npz", n=4, seed=0, n_noise=2, max_len=1200, workers=1)
    rirs.build_bank(tmp_path / "b.npz", n=4, seed=0, n_noise=2, max_len=1200, workers=1, parts_dir=tmp_path / "p")
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes()
    assert set(np.load(tmp_path / "a.npz").files) == {"speech", "noise", "rt60", "armoured"}
