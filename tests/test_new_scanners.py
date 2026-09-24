"""scan_wham and scan_vehicle_interior, exercised on synthetic corpora so they run without the
downloads. What matters here is not that they produce rows but that they preserve the two
properties the r6 experiments depend on: WHAM! keeps its inter-channel relation, and the vehicle
set is long enough to fill an evaluation crop."""
import numpy as np
import pytest
import soundfile as sf

from vaani.data import manifests, sources

SR = 16000


def _wham_corpus(tmp_path, split="tr", sr=SR, sessions=(("s1", 2), ("s2", 1))):
    d = tmp_path / "wham" / split
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name, k in sessions:
        for i in range(k):
            # decorrelated channels: a scanner that collapses to mono would destroy exactly this
            x = rng.standard_normal((sr * 2, 2)).astype(np.float32) * 0.1
            sf.write(d / f"{name}_{i}.wav", x, sr)
    return tmp_path / "wham"


def test_wham_rows_keep_two_channels(tmp_path):
    rows = sources.scan_wham(_wham_corpus(tmp_path), tmp_path / "raw")
    assert len(rows) == 3
    for r in rows:
        x, _ = sf.read(r["path"], always_2d=True)
        assert x.shape[1] == 2, "the two-mic relation is the entire reason this corpus is here"


def test_wham_groups_by_session_not_clip(tmp_path):
    # two clips from one session must share a group so assign() cannot straddle them across splits
    rows = sources.scan_wham(_wham_corpus(tmp_path), tmp_path / "raw")
    groups = {r["source_id"]: r["group_id"] for r in rows}
    assert groups["wham:tr:s1_0"] == groups["wham:tr:s1_1"] != groups["wham:tr:s2_0"]
    assert len({r["split"] for r in rows if r["group_id"] == "wham-s1"}) == 1


def test_wham_refuses_to_scan_the_held_out_split(tmp_path):
    # tt is the reserve generalisation set; putting it in a manifest spends it silently
    with pytest.raises(ValueError, match="held-out"):
        sources.scan_wham(_wham_corpus(tmp_path, split="tt"), tmp_path / "raw", split="tt")


def test_wham_rejects_mono(tmp_path):
    d = tmp_path / "wham" / "tr"
    d.mkdir(parents=True)
    sf.write(d / "s1_0.wav", np.zeros(SR, np.float32), SR)
    with pytest.raises(AssertionError, match="channels"):
        sources.scan_wham(tmp_path / "wham", tmp_path / "raw")


def test_wham_resamples_to_16k(tmp_path):
    rows = sources.scan_wham(_wham_corpus(tmp_path, sr=48000), tmp_path / "raw")
    for r in rows:
        assert sf.info(r["path"]).samplerate == SR


def _vehicle_corpus(tmp_path, clip_s=3.0, classes=("bus", "jeep"), per_class=4, sr=48000):
    rng = np.random.default_rng(0)
    for c in classes:
        d = tmp_path / "vehicle" / c
        d.mkdir(parents=True)
        for i in range(per_class):
            sf.write(d / f"{i}.wav", (rng.standard_normal(int(sr * clip_s)) * 0.1).astype(np.float32), sr)
    return tmp_path / "vehicle"


def test_vehicle_concatenates_one_row_per_class(tmp_path):
    rows = sources.scan_vehicle_interior(_vehicle_corpus(tmp_path), tmp_path / "raw")
    assert len(rows) == 2
    assert {r["group_id"] for r in rows} == {"vehicle-bus", "vehicle-jeep"}


def test_vehicle_rows_outlast_the_evaluation_crop(tmp_path):
    # the defect this scanner exists to avoid: 3 s clips against a 6 s crop reach mix() short,
    # because render_bucket_item pads speech but not noise
    rows = sources.scan_vehicle_interior(_vehicle_corpus(tmp_path, clip_s=3.0), tmp_path / "raw")
    for r in rows:
        assert r["duration_s"] >= 6.0


def test_vehicle_resamples_to_16k(tmp_path):
    rows = sources.scan_vehicle_interior(_vehicle_corpus(tmp_path), tmp_path / "raw")
    for r in rows:
        assert sf.info(r["path"]).samplerate == SR


@pytest.mark.parametrize("scan,fixture", [(sources.scan_wham, _wham_corpus),
                                          (sources.scan_vehicle_interior, _vehicle_corpus)])
def test_rows_match_the_manifest_schema(tmp_path, scan, fixture):
    rows = scan(fixture(tmp_path), tmp_path / "raw")
    assert rows and all(set(r) == set(manifests.COLUMNS) for r in rows)


def test_brickwall_ratio_separates_upsampled_from_genuine():
    """The guard protecting the defence-stationary corpus: a mirror copy laundered through an
    upsample keeps its 4 kHz cliff, and a samplerate check alone cannot see it."""
    from scipy.signal import resample_poly
    rng = np.random.default_rng(0)
    genuine = rng.standard_normal(SR * 5).astype(np.float32)
    assert sources.brickwall_ratio(genuine, SR) < sources.BRICKWALL_MAX

    # 8 kHz source upsampled back to 16 kHz: the container now says 16 kHz, the content does not
    narrow = rng.standard_normal(8000 * 5).astype(np.float32)
    laundered = resample_poly(narrow, 2, 1).astype(np.float32)
    assert sources.brickwall_ratio(laundered, SR) > sources.BRICKWALL_MAX


def test_brickwall_ratio_passes_genuinely_lowpass_content():
    # a tank or car interior really does have almost nothing above 4 kHz; that must not read as a cliff
    rng = np.random.default_rng(0)
    from scipy.signal import butter, lfilter
    b, a = butter(4, 1200 / (SR / 2))
    lowpass = lfilter(b, a, rng.standard_normal(SR * 5)).astype(np.float32)
    assert sources.brickwall_ratio(lowpass, SR) < sources.BRICKWALL_MAX


# --- r8 / mixer v2 scanners (plan 11.5 Datasets); synthetic layouts, nothing downloaded ---

def _noise(sec=1.0, seed=0, ch=None):
    rng = np.random.default_rng(seed)
    shape = (int(SR * sec),) if ch is None else (int(SR * sec), ch)
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


@pytest.mark.parametrize("scan", [sources.scan_librittsr, sources.scan_fsd50k, sources.scan_c3gd, sources.scan_avq_drone,
                                  sources.scan_lombard_grid, sources.scan_musan_noise, sources.scan_demand_pairs,
                                  sources.scan_but_reverbdb])
def test_v2_scanners_tolerate_an_absent_corpus(tmp_path, scan):
    assert scan(tmp_path / "not_downloaded", tmp_path / "out") == []


def test_fsd50k_licence_filter_label_and_uploader_groups(tmp_path):
    import json
    root = tmp_path / "fsd"; (root / "FSD50K.ground_truth").mkdir(parents=True)
    (root / "FSD50K.metadata").mkdir(); (root / "FSD50K.dev_audio").mkdir()
    clips = {"1": ("Gunshot_and_gunfire,Explosion", "http://creativecommons.org/licenses/by/3.0/", "alice"),
             "2": ("Wind", "http://creativecommons.org/publicdomain/zero/1.0/", "alice"),
             "3": ("Siren", "http://creativecommons.org/licenses/by-nc/3.0/", "bob"),
             "4": ("Dog", "http://creativecommons.org/licenses/by/3.0/", "bob")}
    with open(root / "FSD50K.ground_truth" / "dev.csv", "w") as fh:
        fh.write("fname,labels,mids,split\n")
        for k, (lab, _, _) in clips.items():
            fh.write(f'{k},"{lab}",x,train\n')
            sf.write(root / "FSD50K.dev_audio" / f"{k}.wav", _noise(seed=int(k)), SR)
    json.dump({k: {"license": lic, "uploader": up} for k, (_, lic, up) in clips.items()},
              open(root / "FSD50K.metadata" / "dev_clips_info_FSD50K.json", "w"))
    rows = {r["source_id"]: r for r in sources.scan_fsd50k(root, tmp_path / "out")}
    assert set(rows) == {"fsd50k:Gunshot_and_gunfire/1", "fsd50k:Wind/2"}          # NC and unwanted labels refused
    assert rows["fsd50k:Gunshot_and_gunfire/1"]["noise_class"] == "impulsive"
    assert rows["fsd50k:Gunshot_and_gunfire/1"]["group_id"] == rows["fsd50k:Wind/2"]["group_id"] == "fsd50k-up-alice"


def test_lombard_grid_keeps_style_and_groups_by_talker(tmp_path):
    root = tmp_path / "lg" / "audio"; root.mkdir(parents=True)
    for n in ("s1_l_bbaf2n", "s1_p_bbaf2n", "s2_l_lgaz5a", "readme_x"):
        sf.write(root / f"{n}.wav", _noise(0.5), SR)
    rows = {r["source_id"]: r for r in sources.scan_lombard_grid(tmp_path / "lg", tmp_path / "out")}
    assert set(rows) == {"lgrid:s1/l/s1_l_bbaf2n", "lgrid:s1/p/s1_p_bbaf2n", "lgrid:s2/l/s2_l_lgaz5a"}
    assert rows["lgrid:s1/l/s1_l_bbaf2n"]["group_id"] == rows["lgrid:s1/p/s1_p_bbaf2n"]["group_id"] == "lgrid-spk-s1"


def test_mad_video_grouping_joins_folders_of_one_video_and_default_is_unchanged(tmp_path):
    root = tmp_path / "MAD_dataset"
    for d in ("training/7", "test/9"):
        (root / d).mkdir(parents=True); sf.write(root / d / "0.wav", _noise(), SR)
    url = "https://www.youtube.com/watch?v=Oh4q6ck6ufc&t=1s"
    open(root / "training.csv", "w").write(f",path,label,youtube title,youtube url\n0,training/7/0.wav,1,t,{url}\n")
    open(root / "test.csv", "w").write(f",path,label,youtube title,youtube url\n0,test/9/0.wav,1,t,{url}\n")
    by_folder = sources.scan_mad(root, tmp_path / "out")
    assert sorted(r["group_id"] for r in by_folder) == ["mad-7", "mad-9"]
    by_video = sources.scan_mad(root, tmp_path / "out", group_by="video")
    assert {r["group_id"] for r in by_video} == {"mad-yt-Oh4q6ck6ufc"} and len({r["split"] for r in by_video}) == 1
    assert sources.youtube_id("https://youtu.be/Oh4q6ck6ufc") == "Oh4q6ck6ufc" and sources.youtube_id("u") == ""


def test_demand_pairs_found_by_measured_coherence_null(tmp_path):
    from vaani.data import mixer
    env = tmp_path / "demand" / "NPARK"; env.mkdir(parents=True)
    rng = np.random.default_rng(11)   # not _noise's seed: the same stream would make both channels one noise
    pair = mixer.diffuse_pair(rng, _noise(20.0), d=0.12)
    sf.write(env / "ch01.wav", pair[0], SR); sf.write(env / "ch02.wav", pair[1], SR)
    sf.write(env / "ch03.wav", _noise(20.0, seed=5), SR)                 # independent: no 12 cm null
    found = sources.demand_pairs_by_null(env)
    assert [(a, b) for a, b, _ in found] == [(1, 2)] and 1270 <= found[0][2] <= 1630
    rows = sources.scan_demand_pairs(tmp_path / "demand", tmp_path / "out")
    assert [r["source_id"] for r in rows] == ["demand:NPARK/ch01-02"] and sf.info(rows[0]["path"]).channels == 2


def test_dns_audioset_filter_drops_speech_and_music_ids(tmp_path, capsys):
    rows = [dict(path=f"x/{y}_30.000_40.000.flac", source_id=f"dns:{y}") for y in ("AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC")]
    assert sources.dns_audioset_filter(rows, tmp_path / "missing.csv") == rows and "TBD" in capsys.readouterr().out
    csvf = tmp_path / "segments.csv"
    csvf.write_text('# YTID, start_seconds, end_seconds, positive_labels\n'
                    'AAAAAAAAAAA, 30.000, 40.000, "/m/09x0r,/m/0jbk"\n'
                    'BBBBBBBBBBB, 30.000, 40.000, "/m/04rlf"\n'
                    'CCCCCCCCCCC, 30.000, 40.000, "/m/0jbk"\n')
    assert [r["source_id"] for r in sources.dns_audioset_filter(rows, csvf)] == ["dns:CCCCCCCCCCC"]


def test_but_reverbdb_rir_rows_grouped_by_room_and_misc_scanners(tmp_path):
    d = tmp_path / "but" / "Hotel_SkalskyDvur_ConferenceRoom2" / "MicID01" / "SpkID01_20170906_S" / "01" / "RIR"
    d.mkdir(parents=True); sf.write(d / "IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav", _noise(0.3), SR)
    rows = sources.scan_but_reverbdb(tmp_path / "but", tmp_path / "out")
    assert len(rows) == 1 and rows[0]["kind"] == "rir" and rows[0]["group_id"] == "butrdb-Hotel_SkalskyDvur_ConferenceRoom2"
    (tmp_path / "c3gd" / "AK47" / "s1").mkdir(parents=True); sf.write(tmp_path / "c3gd" / "AK47" / "s1" / "a.wav", _noise(0.3), SR)
    r = sources.scan_c3gd(tmp_path / "c3gd", tmp_path / "out")
    assert r[0]["group_id"] == "c3gd-AK47" and r[0]["noise_class"] == "impulsive"
    m = tmp_path / "musan" / "noise" / "free-sound"; m.mkdir(parents=True)
    sf.write(m / "noise-free-sound-0000.wav", _noise(0.5), SR)
    (m / "LICENSE").write_text("noise-free-sound-0000.wav CC BY 3.0\n")
    r = sources.scan_musan_noise(tmp_path / "musan", tmp_path / "out")
    assert r[0]["licence"] == "CC BY 3.0" and r[0]["corpus"] == "musan"
    a = tmp_path / "avq" / "dji"; a.mkdir(parents=True); sf.write(a / "x.wav", _noise(0.5), SR)
    assert sources.scan_avq_drone(tmp_path / "avq", tmp_path / "out")[0]["group_id"] == "avq-dji"
    t = tmp_path / "lttsr" / "train-clean-100" / "19" / "198"; t.mkdir(parents=True)
    sf.write(t / "19_198_000000_000000.wav", _noise(0.5), 24000)
    r = sources.scan_librittsr(tmp_path / "lttsr", tmp_path / "out")
    assert r[0]["group_id"] == "lttsr-spk-19" and sf.info(r[0]["path"]).samplerate == SR
    assert "drone" in sources.V2_DROPPED_CORPORA
