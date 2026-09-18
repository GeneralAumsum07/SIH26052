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

    m, c, meta, twin = render_bucket_item([1234, 7, 105, 0], speech_df, pool_df, n=32000, snr=5, burst=True, bank=None)
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
