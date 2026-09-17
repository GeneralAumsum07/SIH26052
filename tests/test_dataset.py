import json, numpy as np, soundfile as sf, torch
from pathlib import Path
from vaani.data import dataset, manifests, mixer


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


def test_rendered_roundtrip(tmp_path):
    root = tmp_path / "eval"; (root / "stationary_0").mkdir(parents=True)
    x = np.random.randn(2, 16000).astype(np.float32) * 0.1
    sf.write(root / "stationary_0" / "a.mix.wav", x.T, 16000); sf.write(root / "stationary_0" / "a.clean.wav", x[0], 16000)
    json.dump({"snr_db": 0}, open(root / "stationary_0" / "a.json", "w"))
    ds = dataset.RenderedDataset(root)
    it = ds[0]
    assert it["mix"].shape == (2, 16000) and it["meta"]["bucket"] == "stationary_0" and it["meta"]["id"] == "a"
