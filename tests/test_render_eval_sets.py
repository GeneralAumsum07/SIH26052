"""Eval-set renderer: recorded-impulse buckets and reliability-fault buckets."""
import json
import numpy as np, pandas as pd, soundfile as sf
from scripts import render_eval_sets as R


def _dfs(tmp_path):
    rng = np.random.default_rng(0)
    sp = tmp_path / "s.flac"; sf.write(sp, rng.standard_normal(64000).astype(np.float32) * 0.1, 16000)
    nz = tmp_path / "n.flac"; sf.write(nz, rng.standard_normal(96000).astype(np.float32) * 0.1, 16000)
    imp = np.zeros(32000, np.float32); imp[8000:8400] = rng.standard_normal(400).astype(np.float32)
    ip = tmp_path / "i.wav"; sf.write(ip, imp, 16000, subtype="FLOAT")
    return (pd.DataFrame({"path": [str(sp)]}), pd.DataFrame({"path": [str(nz)]}),
            pd.DataFrame({"path": [str(ip)], "source_id": ["mad:shot1"]}))


def test_corpus_impulse_bucket_uses_the_recording_and_its_detected_onset(tmp_path):
    speech, pool, impd = _dfs(tmp_path)
    m, c, meta, twin = R.render_bucket_item([1, 2, 100, 0], speech, pool, 32000, 5.0, "corpus", None, imp_df=impd)
    assert twin is not None and meta["impulse_source"] == "mad:shot1"
    # the bang is 0.5 s into the file; onset must be insertion + ~0.5, never insertion + 0
    on = meta["impulse_onsets_s"]; assert len(on) == 1
    start = int(round((on[0] - 0.5) * 16000)); assert start >= 0
    assert np.array_equal(m[:, :start], twin[:, :start])


def test_synthetic_bucket_records_its_source_kind(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    _, _, meta, twin = R.render_bucket_item([1, 2, 100, 0], speech, pool, 32000, 5.0, "synthetic", None)
    assert twin is not None and meta["impulse_source"].startswith("synthetic:")


def test_no_impulse_bucket_has_no_twin_and_no_source(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    _, _, meta, twin = R.render_bucket_item([1, 2, 100, 0], speech, pool, 32000, 5.0, None, None)
    assert twin is None and meta["impulse_source"] is None


def test_fault_item_degrades_only_the_observed_mixture(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    base = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_none", None)
    clip = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_clip_hard", None)
    gain = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_refgain_-12dB", None)
    # same seed -> same clean target; only the observed mixture differs, and only by the fault
    assert np.array_equal(base[1], clip[1]) and np.array_equal(base[1], gain[1])
    assert np.abs(clip[0][0]).max() < 0.5 * np.abs(base[0][0]).max()
    assert np.array_equal(base[0][1], clip[0][1])                      # clip fault leaves the reference alone
    assert np.allclose(gain[0][1], base[0][1] * 10 ** (-12 / 20), atol=1e-6) and np.array_equal(gain[0][0], base[0][0])
    assert clip[2]["fault"] == "fault_clip_hard" and clip[2]["clipped"] is True
    assert gain[2]["fault"] == "fault_refgain_-12dB" and gain[2]["clipped"] is False
    assert base[2]["fault"] == "fault_none"


def test_fault_burst_bucket_twin_gets_the_same_degradation(tmp_path):
    speech, pool, _ = _dfs(tmp_path)
    m, c, meta, twin = R.render_fault_item([9, 100, 0], speech, pool, 32000, 0.0, "fault_burst_overload", None)
    assert twin is not None and meta["impulse_peak_db"] == 36.0 and meta["clipped"] is True
    start = int(round(meta["impulse_onsets_s"][0] * 16000))
    # twin is clipped at the burst clip's level, so the pre-burst region still matches exactly
    assert np.array_equal(m[:, :start], twin[:, :start])


# --- r8 test set (plan 11.2 G6): held-out spec, scene items, the full render on tiny manifests ---
import csv
import pytest
from vaani.data import manifests as MF
from scripts import verify_eval_set


def _wav(p, n, rng, ch=1):
    x = (rng.standard_normal((n, ch) if ch > 1 else n) * 0.1).astype(np.float32)
    sf.write(p, x, 16000); return str(p)


def _row(sid, corpus, kind, group, path, split, cls="", dur=1.0, spk=""):
    return dict(source_id=sid, corpus=corpus, kind=kind, group_id=group, speaker_id=spk, path=path, duration_s=dur,
                licence="x", split=split, sha1="0", noise_class=cls)


def _manifests(tmp_path):
    """Tiny manifests with every pool the r8 render draws from, written as parquet under tmp_path/m."""
    rng = np.random.default_rng(1); a = tmp_path / "a"; a.mkdir(); md = tmp_path / "m"; md.mkdir()
    w = lambda name, n=16000, ch=1: _wav(a / f"{name}.wav", n, rng, ch)
    shot = np.zeros(16000, np.float32); shot[4000:4200] = 0.9; sf.write(a / "shot.wav", shot, 16000)
    rows = {"librispeech": [_row("ls:1", "librispeech", "speech", "ls-spk-1", w("s1", 32000), "test")],
            "esc50": [_row("esc50:engine/1", "esc50", "noise", "esc50-1", w("e1"), "test", "stationary"),
                      _row("esc50:rain/2", "esc50", "noise", "esc50-2", w("e2"), "test", "changing"),
                      _row("esc50:can_opening/3", "esc50", "noise", "esc50-3", str(a / "shot.wav"), "test", "impulsive"),
                      _row("esc50:siren/4", "esc50", "noise", "esc50-4", w("e4"), "test", "changing")],
            "mad_v2": [_row("mad:helicopter/1_0", "mad", "noise", "mad-yt-aaa", w("m1"), "test", "stationary"),
                       _row("mad:vehicle/2_0", "mad", "noise", "mad-yt-bbb", w("m2"), "test", "stationary"),
                       _row("mad:shooting/3_0", "mad", "noise", "mad-yt-ccc", w("m3"), "train", "changing")],
            "gunshots": [_row("gun:g/1", "gunshots", "noise", "gun-1", str(a / "shot.wav"), "test", "impulsive")],
            "demand": [_row("demand:STRAFFIC", "demand", "noise", "demand-STRAFFIC", w("d1", 16000, 2), "test", "stationary")],
            "drone": [_row(f"drone:yes_drone/R{r}-bebop_{c:03d}_", "drone", "noise", "drone-yes_drone", w(f"dr{r}_{c}"),
                           "train", "stationary") for r in range(12) for c in range(2)],
            "noisex92": [_row(f"noisex92:{k}", "noisex92", "noise", f"noisex92-{k}", w(f"nx{k}"), "train", "stationary")
                         for k in ("buccaneer1", "buccaneer2", "m109", "leopard", "destroyerengine", "destroyerops")],
            "ears": [_row(f"ears:{s}/rainbow_01_{st}", "ears", "speech", f"ears-spk-{s}", w(f"{s}{st}", 24000), sp, spk=s)
                     for s, sp in (("p001", "train"), ("p002", "val"), ("p003", "train")) for st in ("loud", "regular")]}
    for k, v in rows.items():
        pd.DataFrame(v).to_parquet(md / f"{k}.parquet")
    return md


def _pools(md):
    spec = R.heldout_spec(md)
    held = {s for v in spec["sources"].values() for s in v["source_ids"]}
    return spec, held, pd.concat([MF.read(p) for p in sorted(md.glob("*.parquet"))])


def test_drone_recording_strips_only_the_chunk_index():
    assert R.drone_recording("drone:yes_drone/B_S2_D1_067-bebop_003_") == "drone:yes_drone/B_S2_D1_067-bebop"
    assert R.drone_recording("drone:yes_drone/extra_membo_D2_055") == "drone:yes_drone/extra_membo_D2"


def test_heldout_spec_follows_its_written_rules(tmp_path):
    md = _manifests(tmp_path)
    spec = R.heldout_spec(md)
    assert spec == R.heldout_spec(md) and spec["schema"] == R.HELDOUT_SCHEMA   # deterministic
    S = spec["sources"]
    recs = {R.drone_recording(s) for s in pd.read_parquet(md / "drone.parquet").source_id}
    assert S["drone"]["group_ids"] == sorted(r for r in recs if MF.stable_hash(r) % R.DRONE_HOLDOUT_MOD == 0)
    # both chunks of a held-out recording go together: no recording straddles train and test
    assert all(R.drone_recording(s) in S["drone"]["group_ids"] for s in S["drone"]["source_ids"])
    assert S["drone"]["n_rows"] == 2 * len(S["drone"]["group_ids"])
    assert S["noisex92"]["source_ids"] == sorted(R.NOISEX_HELDOUT)
    spk = min(["ears-spk-p001", "ears-spk-p003"], key=MF.stable_hash)   # train speakers only; p002 is val
    assert S["ears"]["group_ids"] == [spk] and all(c.endswith("_loud") for c in S["ears"]["test_clips"])
    assert S["mad"]["source_ids"] == ["mad:helicopter/1_0", "mad:vehicle/2_0"]


def test_read_heldout_rejects_another_schema(tmp_path):
    p = tmp_path / "h.json"; p.write_text(json.dumps({"schema": "other", "sources": {}}))
    with pytest.raises(ValueError):
        R.read_heldout(p)


def test_r8_pools_refuse_the_folder_grouped_mad(tmp_path):
    spec, held, df = _pools(_manifests(tmp_path))
    R.r8_pools(df, held, spec)   # the video-grouped cut passes
    old = df.copy(); m = old.corpus == "mad"
    old.loc[m, "group_id"] = "mad-" + old.loc[m, "source_id"].str.split("/").str[1]
    with pytest.raises(ValueError, match="mad_v2"):
        R.r8_pools(old, held, spec)
    extra = pd.concat([df, pd.DataFrame([_row("mad:vehicle/9_0", "mad", "noise", "mad-yt-zzz", df.path.iloc[0], "test")])])
    with pytest.raises(ValueError, match="mad_v2"):
        R.r8_pools(extra, held, spec)


def test_scene_item_is_a_seeded_v2_mix_that_plays_whole_drone_recordings(tmp_path):
    spec, held, df = _pools(_manifests(tmp_path))
    P = R.r8_pools(df, held, spec)
    load, recs = R._drone_loader(P["drone"])
    pool = R._scene_pool(P["noise"], P["drone"])
    assert len(pool.index["drone"]) == len(spec["sources"]["drone"]["group_ids"])
    a = R.render_scene_item([5, 1, 2, 100, 0], P["speech"], pool, 16000, "drone", None, drone_recs=recs)
    b = R.render_scene_item([5, 1, 2, 100, 0], P["speech"], pool, 16000, "drone", None, drone_recs=recs)
    assert np.array_equal(a[0], b[0]) and a[3] is None
    meta = a[2]
    assert meta["mix_version"] == 2 and meta["scene"] == "drone" and meta["ref_mode"] == "physical"
    assert "drone:" in meta["noise_source"] and np.isfinite(meta["snr_db"])
    x, name = load(np.random.default_rng(0), 48000)
    assert len(x) == 48000 and name in recs and len(recs[name]) == 2


def test_heldout_item_takes_its_bed_from_the_loader(tmp_path):
    speech, _, _ = _dfs(tmp_path)
    load = lambda rng, n: (np.full(n // 2, 0.1, np.float32), "noisex92:m109")   # short beds are zero-padded
    m, c, meta, twin = R.render_heldout_item([3, 1, 100, 0], speech, load, 32000, 0.0, None)
    assert twin is None and meta["noise_source"] == "noisex92:m109" and meta["impulse_source"] is None
    assert m.shape == (2, 32000)


def test_r8_render_writes_every_subset_an_index_and_a_verifiable_hash(tmp_path):
    md = _manifests(tmp_path)
    hp = tmp_path / "heldout.json"; hp.write_text(json.dumps(R.heldout_spec(md)))
    a = type("A", (), dict(manifests=sorted(str(p) for p in md.glob("*.parquet")), split="test", out=str(tmp_path / "o"),
                           bank=str(tmp_path / "none.npz"), no_bank=True, r8_test=True, heldout=str(hp), r8_size=1,
                           clip_s=1.0, seed=2610, force=False, defence=False, build_eval_bank=None))()
    R.main(a)
    root = tmp_path / "o" / "test"
    with open(root / "index.csv", encoding="utf-8") as fh:
        idx = list(csv.DictReader(fh))
    by = {}
    for r in idx:
        by[r["subset"]] = by.get(r["subset"], 0) + 1
    n_snr = len(R.BUCKET_SNRS)
    assert by == {"v1": len(R.CLASSES) * n_snr + 1, "defence": len(R.DEFENCE) * n_snr, "heldout": 2 * n_snr,
                  "loud": n_snr, "fault": len(R.FAULTS) * len(R.FAULT_SNRS), "v2": len(R.scenes.SCENE_WEIGHTS)}
    assert all(r["category"].startswith(r["subset"] + "/") for r in idx)
    assert {r["scene"] for r in idx if r["subset"] == "v2"} == set(R.scenes.SCENE_WEIGHTS)
    assert all(r["speech_source"].startswith("ears:") for r in idx if r["subset"] == "loud")
    assert all(r["noise_source"].startswith("drone:") for r in idx if r["bucket"].startswith("heldout_drone"))
    assert not verify_eval_set.verify(root, (root / "EVALSET_HASH").read_text().strip())
    with pytest.raises(SystemExit):   # the finished set is frozen
        R.main(a)
