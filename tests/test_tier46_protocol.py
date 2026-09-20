"""Freeze step of the Tier 4.6 protocol: content hashes of the frozen split and an immutable anchor copy."""
import hashlib, json, sys
from pathlib import Path

import numpy as np, pytest, soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tier46_protocol as tp  # noqa: E402


def _split(root, buckets=("stationary_0", "impulsive_0"), n=2, twins=True):
    for b in buckets:
        d = root / b; d.mkdir(parents=True)
        for i in range(n):
            x = (np.random.default_rng(i).standard_normal((16000, 2)) * 0.1).astype(np.float32)
            sf.write(d / f"{i:04d}.mix.wav", x, 16000); sf.write(d / f"{i:04d}.clean.wav", x[:, 0], 16000)
            imp = b.startswith("impulsive")
            (d / f"{i:04d}.json").write_text(json.dumps({"impulse_onsets_s": [0.5] if imp else [], "snr_db": 0}))
            if imp and twins: sf.write(d / f"{i:04d}.twin.mix.wav", x, 16000)
    (root / "EVALSET_HASH").write_text("abc")


def _anchor(p):
    import torch
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.zeros(1)}, "config": {"model": "vaani", "name": "e32"}}, p)


def test_freeze_writes_manifest_and_anchor(tmp_path):
    root = tmp_path / "eval"; _split(root / "test"); _split(root / "val")
    ck = tmp_path / "runs" / "e32" / "best.pt"; _anchor(ck)
    out = tmp_path / "out"
    r = tp.freeze(root, [ck], out, anchor_dir=tmp_path / "runs" / "tier46_anchor")
    proto = json.loads((out / "anchor.json").read_text())
    assert proto["anchor"]["sha256"] == hashlib.sha256(ck.read_bytes()).hexdigest()
    assert (tmp_path / "runs" / "tier46_anchor" / "best.pt").read_bytes() == ck.read_bytes()
    assert proto["splits"]["test"]["n_items"] == 4 and len(proto["splits"]["test"]["files"]) == 4 * 3 + 2 + 1
    assert proto["splits"]["test"]["keys"][0] == ["impulsive_0", "0000"]


def test_freeze_rejects_changed_wav_bytes(tmp_path):
    root = tmp_path / "eval"; _split(root / "test"); _split(root / "val")
    ck = tmp_path / "runs" / "e32" / "best.pt"; _anchor(ck); out = tmp_path / "out"
    tp.freeze(root, [ck], out, anchor_dir=tmp_path / "runs" / "tier46_anchor")
    w = root / "test" / "stationary_0" / "0000.mix.wav"; x, sr = sf.read(w, dtype="float32"); sf.write(w, x * 0.5, sr)
    assert tp.check(root, json.loads((out / "anchor.json").read_text()))   # returns the list of problems


def test_freeze_rejects_missing_twins_bad_wavs_and_unequal_anchor(tmp_path):
    root = tmp_path / "eval"; _split(root / "test", twins=False); _split(root / "val")
    ck = tmp_path / "runs" / "e32" / "best.pt"; _anchor(ck)
    with pytest.raises(SystemExit): tp.freeze(root, [ck], tmp_path / "o1", anchor_dir=tmp_path / "a1")
    root2 = tmp_path / "eval2"; _split(root2 / "test"); _split(root2 / "val")
    bad = root2 / "test" / "stationary_0" / "0001.clean.wav"; sf.write(bad, np.zeros(8000, np.float32), 8000)   # wrong rate + length
    with pytest.raises(SystemExit): tp.freeze(root2, [ck], tmp_path / "o2", anchor_dir=tmp_path / "a2")
    root3 = tmp_path / "eval3"; _split(root3 / "test"); _split(root3 / "val")
    adir = tmp_path / "a3"; adir.mkdir(); (adir / "best.pt").write_bytes(b"other")   # an existing, different anchor is never overwritten
    with pytest.raises(SystemExit): tp.freeze(root3, [ck], tmp_path / "o3", anchor_dir=adir)
    assert (adir / "best.pt").read_bytes() == b"other"
