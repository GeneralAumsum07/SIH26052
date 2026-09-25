import json, numpy as np, pandas as pd, soundfile as sf, torch
from pathlib import Path
from vaani.data import dataset, manifests, mixer
from scripts.render_eval_sets import render_bucket_item


def _tiny_manifest(tmp_path):
    rows = []
    for i in range(4):
        p = tmp_path / f"s{i}.flac"; sf.write(p, np.random.randn(32000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"s{i}", corpus="t", kind="speech", group_id=f"g{i}", speaker_id=f"g{i}",
                         path=str(p), duration_s=2.0, licence="", split="train", sha1="", noise_class=""))
    for i in range(3):
        p = tmp_path / f"n{i}.flac"; sf.write(p, np.random.randn(48000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"n{i}", corpus="t", kind="noise", group_id=f"ng{i}", speaker_id="",
                         path=str(p), duration_s=3.0, licence="", split="train", sha1="", noise_class=["stationary", "changing", "impulsive"][i]))
    m = tmp_path / "m.parquet"; manifests.write(rows, m); return m


def test_dynamic_dataset_yields_batches(tmp_path):
    m = _tiny_manifest(tmp_path)
    ds = dataset.DynamicMixDataset([m], "train", None, mixer.MixConfig(p_room=0.0), crop_s=1.0, epoch_len=6, seed=0)
    dl = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=dataset.collate, num_workers=0)
    b = next(iter(dl))
    assert b["mix"].shape == (3, 2, 16000) and b["clean"].shape == (3, 16000) and len(b["meta"]) == 3


def test_dynamic_dataset_split_isolation(tmp_path):
    rows = []
    for i in range(4):
        p = tmp_path / f"s{i}.flac"; sf.write(p, np.random.randn(32000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"s{i}", corpus="t", kind="speech", group_id=f"g{i}", speaker_id=f"g{i}",
                         path=str(p), duration_s=2.0, licence="", split="train", sha1="", noise_class=""))
    # one val-only speech row with a distinctive id, to prove split="val" never reaches it
    pv = tmp_path / "sval.flac"; sf.write(pv, np.random.randn(32000).astype(np.float32) * 0.1, 16000)
    rows.append(dict(source_id="sval", corpus="t", kind="speech", group_id="gval", speaker_id="gval",
                     path=str(pv), duration_s=2.0, licence="", split="val", sha1="", noise_class=""))
    for i in range(3):
        p = tmp_path / f"n{i}.flac"; sf.write(p, np.random.randn(48000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"n{i}", corpus="t", kind="noise", group_id=f"ng{i}", speaker_id="",
                         path=str(p), duration_s=3.0, licence="", split="train", sha1="", noise_class=["stationary", "changing", "impulsive"][i]))
    pnv = tmp_path / "nval.flac"; sf.write(pnv, np.random.randn(48000).astype(np.float32) * 0.1, 16000)
    rows.append(dict(source_id="nval", corpus="t", kind="noise", group_id="ngval", speaker_id="",
                     path=str(pnv), duration_s=3.0, licence="", split="val", sha1="", noise_class="stationary"))
    m = tmp_path / "m.parquet"; manifests.write(rows, m)

    ds = dataset.DynamicMixDataset([m], "train", None, mixer.MixConfig(p_room=0.0), crop_s=1.0, epoch_len=10, seed=0)
    assert "sval" not in set(ds.speech.source_id) and "nval" not in set(ds.noise.source_id)
    for i in range(len(ds)):
        item = ds[i]  # would only ever touch train-split files; nothing to assert beyond it not erroring
        assert item["mix"].shape == (2, 16000)


def test_dynamic_dataset_deterministic_across_instances(tmp_path):
    m = _tiny_manifest(tmp_path)
    cfg = mixer.MixConfig(p_room=0.0)
    ds1 = dataset.DynamicMixDataset([m], "train", None, cfg, crop_s=1.0, epoch_len=6, seed=0)
    ds2 = dataset.DynamicMixDataset([m], "train", None, cfg, crop_s=1.0, epoch_len=6, seed=0)
    a = ds1[3]; b = ds2[3]
    assert torch.equal(a["mix"], b["mix"]) and torch.equal(a["clean"], b["clean"])
    assert a["meta"] == b["meta"]


def test_render_bucket_item_twin_matches_outside_impulse_window(tmp_path):
    speech_rows, noise_rows = [], []
    for i in range(2):
        p = tmp_path / f"s{i}.flac"; sf.write(p, np.random.randn(32000).astype(np.float32) * 0.1, 16000)
        speech_rows.append(dict(path=str(p)))
    for i in range(2):
        p = tmp_path / f"n{i}.flac"; sf.write(p, np.random.randn(96000).astype(np.float32) * 0.1, 16000)
        noise_rows.append(dict(path=str(p)))
    speech_df, pool_df = pd.DataFrame(speech_rows), pd.DataFrame(noise_rows)

    m, c, meta, twin = render_bucket_item([1234, 7, 105, 0], speech_df, pool_df, n=32000, snr=5, impulse="synthetic", bank=None)
    assert twin is not None
    start = int(round(meta["impulse_onsets_s"][0] * dataset.SR))
    end = start + 1  # exact impulse length isn't exposed here; check the region strictly before onset
    assert np.array_equal(m[:, :start], twin[:, :start])
    assert not np.array_equal(m[:, start:], twin[:, start:])  # impulse actually changed something


def test_rendered_roundtrip(tmp_path):
    root = tmp_path / "eval"; (root / "stationary_0").mkdir(parents=True)
    x = np.random.randn(2, 16000).astype(np.float32) * 0.1
    sf.write(root / "stationary_0" / "a.mix.wav", x.T, 16000); sf.write(root / "stationary_0" / "a.clean.wav", x[0], 16000)
    json.dump({"snr_db": 0}, open(root / "stationary_0" / "a.json", "w"))
    sf.write(root / "stationary_0" / "a.twin.mix.wav", x.T, 16000)  # twin is an attachment, not an item
    ds = dataset.RenderedDataset(root)
    assert len(ds) == 1
    it = ds[0]
    assert it["mix"].shape == (2, 16000) and it["meta"]["bucket"] == "stationary_0" and it["meta"]["id"] == "a"
    assert it["twin"].shape == (2, 16000)


def test_corpus_impulse_onset_comes_from_waveform(tmp_path, monkeypatch):
    # a recorded impulse whose bang sits at 0.5 s must be reported 0.5 s after insertion, not at it
    m = _tiny_manifest(tmp_path)
    imp = np.zeros(32000, np.float32); imp[8000:8400] = np.random.randn(400).astype(np.float32)
    sf.write(tmp_path / "n2.flac", imp, 16000)  # n2 is the "impulsive" row of the tiny manifest
    ds = dataset.DynamicMixDataset([m], "train", None, mixer.MixConfig(p_room=0.0, p_clean=0.0), crop_s=2.0, epoch_len=40)
    # crop from the file start so the bang is always at 0.5 s inside the impulse clip
    # _load takes the packed-corpus reader as a fourth argument; accept and ignore it here
    monkeypatch.setattr(dataset, "_load",
                        lambda path, n, rng, pack=None: sf.read(path, dtype="float32")[0][:n])
    real_mix = dataset.mix
    starts = {}
    def spy(rng, s, noises, imp, onsets, bank, cfg):
        out = real_mix(rng, s, noises, imp, onsets, bank, cfg)
        if imp is not None: starts[id(out[2])] = (onsets, out[2])
        return out
    monkeypatch.setattr(dataset, "mix", spy)
    for i in range(40): ds[i]
    corpus = [on for on, meta in starts.values() if len(on) == 1 and abs(on[0] - 0.5) < 0.02]
    assert corpus, "corpus impulse branch never hit or onset not detected at 0.5 s"
    for on, meta in starts.values():
        assert meta["impulse_onsets_s"] and meta["impulse_onsets_s"][0] >= on[0] - 1e-6


# --- r8 wiring: mixer v2 scenes, VaaniFE front end, validity labels, held-out groups ---
import importlib.util, subprocess, sys, types
import pytest
from vaani.dsp import pipeline

BASE = "5c006b7"   # dataset.py before the r8 wiring: v1 items must stay bit-exact to it


def _old_dataset(tmp_path):
    src = subprocess.check_output(["git", "show", f"{BASE}:vaani/data/dataset.py"], text=True,
                                  cwd=Path(__file__).resolve().parents[1])
    p = tmp_path / "old_dataset.py"; p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("old_dataset", p); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


@pytest.mark.parametrize("dsp", [False, True])
def test_v1_items_bit_exact_to_pre_r8(tmp_path, dsp):
    m = _tiny_manifest(tmp_path); old = _old_dataset(tmp_path)
    cfg = mixer.MixConfig(p_room=0.0)
    kw = dict(crop_s=1.0, epoch_len=8, seed=3, with_dsp=dsp, ref_corrupt={"p": 0.5, "p_absent": 0.2} if dsp else None)
    a = dataset.DynamicMixDataset([m], "train", None, cfg, **kw)
    b = old.DynamicMixDataset([m], "train", None, cfg, **kw)
    for i in range(8):
        x, y = a[i], b[i]
        assert x.keys() == y.keys() and x["meta"] == y["meta"]
        for k in x:
            if k != "meta":
                assert torch.equal(x[k], y[k]), (i, k)


def test_v2_scene_items(tmp_path):
    m = _tiny_manifest(tmp_path)
    cfg = mixer.MixConfig(version=2, p_room=0.0)
    ds = dataset.DynamicMixDataset([m], "train", None, cfg, crop_s=1.0, epoch_len=12, seed=0)
    assert ds.scene_pool is not None
    scenes = set()
    for i in range(12):
        it = ds[i]
        assert it["mix"].shape == (2, 16000) and torch.isfinite(it["mix"]).all()
        assert it["meta"]["mix_version"] == 2; scenes.add(it["meta"]["scene"])
    assert len(scenes) >= 2
    again = dataset.DynamicMixDataset([m], "train", None, cfg, crop_s=1.0, epoch_len=12, seed=0)[5]
    assert torch.equal(again["mix"], ds[5]["mix"])


def test_front_end_matches_pipeline_mix():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((2, 16000)) * 0.3).astype(np.float32); x[:, 4000:4300] *= 8
    av = np.ones(16000, bool); av[6000:9000] = False
    cfg = {"limiter": True, "ref_policy": {"nlms": True, "absent": "freeze", "ramp_frames": 12}}
    m2, fa = dataset.front_end(x, cfg, av)
    r = pipeline.run(x, controller_on=True, dsp_cfg=cfg, ref_avail=av)
    assert np.array_equal(m2, r["mix"]) and np.array_equal(fa, r["ref_avail"])
    m0, f0 = dataset.front_end(x)
    assert np.array_equal(m0, x) and (f0 == 1).all() and f0.shape == (63,)


def test_fe_inputs_emit_validity_labels(tmp_path):
    m = _tiny_manifest(tmp_path)
    cfg = mixer.MixConfig(p_room=0.0)
    ds = dataset.DynamicMixDataset([m], "train", None, cfg, crop_s=1.0, epoch_len=4, seed=0, fe_inputs=True,
                                   dsp_cfg={"limiter": True})
    it = ds[0]
    assert "n_hat" not in it and it["ref_avail"].shape == (63,) and (it["ref_avail"] == 1).all()
    absent = dataset.DynamicMixDataset([m], "train", None, cfg, crop_s=1.0, epoch_len=4, seed=0, fe_inputs=True,
                                       ref_corrupt={"p": 0.0, "p_absent": 1.0},
                                       dsp_cfg={"limiter": True, "ref_policy": {"absent": "freeze"}})
    for i in range(4):
        it = absent[i]
        assert (it["ref_avail"] == 0).all() and (it["mix"][1] == 0).all() and it["meta"]["ref_fault"]["kind"] == "dropout"
    b = dataset.collate([absent[0], absent[1]])
    assert b["ref_avail"].shape == (2, 63)


def test_m9_rates(tmp_path):
    c = dataset.ref_corrupt_config({"p": 0.15, "p_absent": 0.15})
    kinds = []
    for i in range(2000):
        g = np.random.default_rng([0, i])
        x = g.standard_normal((2, 800)).astype(np.float32) * 0.1
        _, av, tr = dataset.corrupt_reference(g, x, x[0], c)
        kinds.append(None if tr is None else ("absent" if not av.any() else "fault"))
    absent = kinds.count("absent") / 2000; fault = kinds.count("fault") / 2000
    assert 0.12 < absent < 0.20 and 0.10 < fault < 0.20   # dropout faults also count as absent here


def test_exclude_groups_file(tmp_path, capsys):
    m = _tiny_manifest(tmp_path)
    ex = tmp_path / "ex.json"; json.dump({"speakers": ["g0", "g1"], "noise": {"groups": ["ng0"]}}, open(ex, "w"))
    ds = dataset.DynamicMixDataset([m], "train", None, mixer.MixConfig(p_room=0.0), crop_s=1.0, epoch_len=2,
                                   exclude_groups_file=str(ex))
    assert set(ds.speech.group_id) == {"g2", "g3"} and "ng0" not in set(ds.noise.group_id)
    with pytest.warns(UserWarning, match="ABSENT"):
        ds2 = dataset.DynamicMixDataset([m], "train", None, mixer.MixConfig(p_room=0.0), crop_s=1.0, epoch_len=2,
                                        exclude_groups_file=str(tmp_path / "missing.json"))
    assert len(ds2.speech) == 4 and "WARNING" in capsys.readouterr().out


def test_exclude_groups_schema_file(tmp_path):
    m = _tiny_manifest(tmp_path)
    ex = tmp_path / "ex.json"
    json.dump({"schema": "vaani.heldout_exclude/1", "apply": "g1 is only in this text",
               "sources": {"x": {"group_ids": ["g2"], "source_ids": ["s0", "n1"]}}}, open(ex, "w"))
    ds = dataset.DynamicMixDataset([m], "train", None, mixer.MixConfig(p_room=0.0), crop_s=1.0, epoch_len=2,
                                   exclude_groups_file=str(ex))
    assert set(ds.speech.source_id) == {"s1", "s2", "s3"} and "n1" not in set(ds.noise.source_id)
    real = Path(__file__).resolve().parents[1] / "configs/data/r8_heldout_exclude.json"
    if real.exists():
        ids = dataset.load_exclude_groups(real)
        assert len(ids) > 100 and all(":" in s for s in ids)
