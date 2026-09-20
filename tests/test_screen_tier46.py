"""Tier 4.6 val screen: cache provenance and deterministic selection (the audio path is covered by test_eval_tier46)."""
import json, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import screen_tier46 as sc  # noqa: E402


def _proto(sha="a" * 64, files=None):
    return {"anchor": {"sha256": sha, "config": {"model": "vaani", "controller_on": True, "dsp": {"k": 1}}},
            "splits": {"val": {"files": files or {"x.mix.wav": "1"}}}}


def test_cache_rejects_other_checkpoint_or_split(tmp_path):
    key = sc.open_cache(tmp_path / "c", _proto(), "val")
    assert sc.open_cache(tmp_path / "c", _proto(), "val") == key          # same provenance reopens
    with pytest.raises(RuntimeError, match="another anchor"):
        sc.open_cache(tmp_path / "c", _proto(sha="b" * 64), "val")          # other checkpoint
    with pytest.raises(RuntimeError):
        sc.open_cache(tmp_path / "c", _proto(files={"y.mix.wav": "2"}), "val")   # other split content
    meta = json.loads((tmp_path / "c" / "cache_meta.json").read_text())
    assert meta["anchor_sha256"] == "a" * 64 and meta["stft"]["hop"] == 256


def _res(d_snr, d_pesq, gf, ok=True, **override):
    gates = {g: ok for g in sc.SCREEN_GATES}; gates["paired_uncertainty"] = False; gates.update(override)
    return ({"gain_floor": gf, "noise_bias": 1.0}, {"gates": gates, "nominal": {"d_snr_out": {"mean": d_snr}, "d_pesq_wb": {"mean": d_pesq}}})


def test_selection_is_deterministic_and_gated():
    r = {"a": _res(0.5, 0.05, 0.70), "b": _res(0.5, 0.05, 0.85), "c": _res(0.9, 0.01, 0.70, per_bucket=False), "d": _res(0.4, 0.10, 0.85)}
    assert sc.select(r, "postfilter")[0] == "b"                        # c fails a protection gate; a/b tie -> higher floor
    assert sc.select(dict(reversed(list(r.items()))), "postfilter")[0] == "b"   # insertion order does not matter
    assert sc.select({"z": _res(0.5, 0.05, 0.85, False)}, "postfilter") == (None, None)
    r["e"] = _res(0.5, 0.06, 0.70); assert sc.select(r, "postfilter")[0] == "e"   # PESQ breaks the SNR tie before the floor
