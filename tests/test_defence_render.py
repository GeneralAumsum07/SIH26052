"""Defence-noise render (plan A3) and the --bank guard in scripts/render_eval_sets.py."""
import csv, json, sys
from pathlib import Path

import numpy as np, pandas as pd, pytest, soundfile as sf, yaml

from scripts import render_eval_sets as R
from scripts import verify_eval_set

ROOT = Path(__file__).resolve().parents[1]


def _args(out, **kw):
    a = {"manifests": [], "split": "test", "out": str(out), "bank": str(out / "missing_bank.npz"), "per_bucket": 1,
         "clip_s": 1.0, "seed": 2609, "faults": False, "force": False, "classes": None, "no_bank": False, "defence": False}
    a.update(kw)
    return type("A", (), a)()


def test_missing_bank_is_a_hard_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        R.guard_bank(_args(tmp_path))
    assert "--no-bank" in str(e.value) and "missing_bank.npz" in str(e.value)


def test_no_bank_is_an_explicit_opt_out(tmp_path):
    assert R.guard_bank(_args(tmp_path, no_bank=True)) is None


def test_old_args_without_the_flag_still_error(tmp_path):
    a = _args(tmp_path); del type(a).no_bank
    with pytest.raises(SystemExit):
        R.guard_bank(a)


def test_bank_guard_fires_before_the_manifests_are_read(tmp_path, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("manifests read before the bank guard")
    monkeypatch.setattr(R.manifests, "read", explode)
    with pytest.raises(SystemExit):
        R.main(_args(tmp_path))


def test_defence_mix_is_r7s_training_mix_block():
    mix = yaml.safe_load(open(ROOT / "configs/retraining/r7_e256_wr64.yaml", encoding="utf-8"))["data"]["mix"]
    mix.pop("impulse_kinds")
    assert {k: tuple(v) if isinstance(v, list) else v for k, v in mix.items()} == R.DEFENCE_MIX


def test_peak_window_puts_the_shot_near_the_start(tmp_path):
    x = np.random.default_rng(0).standard_normal(60000).astype(np.float32) * 0.01
    x[50000:50200] += 0.9
    p = tmp_path / "shot.wav"; sf.write(p, x, 16000, subtype="FLOAT")
    imp, on, src = R._peak_window(pd.Series({"path": str(p), "source_id": "gun:x/1"}))
    a = int(np.argmax(np.abs(x))) - 4000
    assert len(imp) == 60000 - a and src == "gun:x/1"      # the file ends inside the 2 s window
    assert abs(np.abs(imp).max() - 1.0) < 1e-5 and abs(on[0] - 0.25) < 0.02


def test_tag_noise_off_keeps_the_old_meta_keys(tmp_path):
    rng = np.random.default_rng(0)
    sp = tmp_path / "s.flac"; sf.write(sp, rng.standard_normal(32000).astype(np.float32) * 0.1, 16000)
    nz = tmp_path / "n.flac"; sf.write(nz, rng.standard_normal(32000).astype(np.float32) * 0.1, 16000)
    speech, pool = pd.DataFrame({"path": [str(sp)]}), pd.DataFrame({"path": [str(nz)], "source_id": ["mad:vehicle/1"]})
    _, _, m0, _ = R.render_bucket_item([1, 2, 3], speech, pool, 16000, 0.0, None, None)
    m1, _, meta1, _ = R.render_bucket_item([1, 2, 3], speech, pool, 16000, 0.0, None, None, tag_noise=True)
    assert "noise_source" not in m0 and meta1["noise_source"] == "mad:vehicle/1"
    assert {k: v for k, v in meta1.items() if k != "noise_source"} == m0


def _manifest(tmp_path):
    rng = np.random.default_rng(1); rows = []

    def wav(name, x):
        p = tmp_path / name; sf.write(p, x.astype(np.float32), 16000, subtype="FLOAT"); return str(p)

    for i in range(2):
        rows.append(dict(source_id=f"ls:{i}", corpus="librispeech", kind="speech", noise_class="", split="test",
                         path=wav(f"sp{i}.wav", rng.standard_normal(24000) * 0.1)))
    for cls, nc in (("vehicle", "stationary"), ("helicopter", "stationary"), ("fighter", "stationary"), ("shooting", "changing")):
        rows.append(dict(source_id=f"mad:{cls}/0", corpus="mad", kind="noise", noise_class=nc, split="test",
                         path=wav(f"mad_{cls}.wav", rng.standard_normal(24000) * 0.1)))
    rows.append(dict(source_id="esc50:siren/0", corpus="esc50", kind="noise", noise_class="changing", split="test",
                     path=wav("siren.wav", np.sin(np.arange(24000) * 0.1) * 0.1)))
    shot = np.zeros(24000); shot[12000:12100] = 0.9
    for c in R.GUNSHOT_CORPORA:
        rows.append(dict(source_id=f"{c}:g/0", corpus=c, kind="noise", noise_class="impulsive", split="test",
                         path=wav(f"{c}.wav", shot)))
    rows.append(dict(source_id="gunshots:g/tr", corpus="gunshots", kind="noise", noise_class="impulsive", split="train",
                     path=wav("train_shot.wav", shot)))
    p = tmp_path / "m.parquet"; pd.DataFrame(rows).to_parquet(p)
    return str(p)


def test_defence_render_writes_tags_index_and_a_verifiable_hash(tmp_path):
    out = tmp_path / "eval_defence"
    R.main(_args(out, manifests=[_manifest(tmp_path)], no_bank=True, defence=True))
    root = out / "test"
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == sorted(
        f"{c}_{s}" for c in R.DEFENCE for s in R.BUCKET_SNRS)
    idx = list(csv.DictReader(open(root / "index.csv", encoding="utf-8")))
    assert len(idx) == len(R.DEFENCE) * len(R.BUCKET_SNRS)
    by = {(r["category"], r["snr_db"]): r for r in idx}
    g = by[("gunshot", "0.0")]
    assert g["impulse_source"].split(":")[0] in R.GUNSHOT_CORPORA and g["impulse_source"] != "gunshots:g/tr"
    assert 15.0 <= float(g["impulse_peak_db"]) <= 45.0 and g["noise_source"].startswith("mad:")
    assert by[("blast_artillery", "5.0")]["impulse_source"] == "synthetic:blast:artillery"
    assert by[("blast_small_arms", "5.0")]["impulse_source"] == "synthetic:blast:small_arms"
    assert by[("helicopter", "-10.0")]["noise_source"] == "mad:helicopter/0" and by[("helicopter", "-10.0")]["impulse_source"] == ""
    assert by[("vehicle", "15.0")]["noise_source"] == "mad:vehicle/0"
    assert by[("siren", "10.0")]["noise_source"] == "esc50:siren/0"
    # transient buckets carry twins, bed-only buckets do not
    assert (root / "gunshot_0" / "0000.twin.mix.wav").exists() and not (root / "vehicle_0" / "0000.twin.mix.wav").exists()
    meta = json.load(open(root / "blast_small_arms_5" / "0000.json"))
    assert meta["category"] == "blast_small_arms" and meta["noise_class"] == "impulsive+stationary"
    assert verify_eval_set.verify(root, (root / "EVALSET_HASH").read_text().strip()) == []
    # a second render is refused by the frozen guard
    with pytest.raises(SystemExit):
        R.main(_args(out, manifests=[_manifest(tmp_path)], no_bank=True, defence=True))


def test_defence_render_fails_loudly_on_an_empty_pool(tmp_path):
    m = _manifest(tmp_path); df = pd.read_parquet(m)
    df[~df.source_id.str.startswith("esc50:siren")].to_parquet(m)
    with pytest.raises(ValueError, match="siren"):
        R.main(_args(tmp_path / "e", manifests=[m], no_bank=True, defence=True))
