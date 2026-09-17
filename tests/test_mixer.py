import numpy as np
from vaani.data import mixer, rirs


def _speech(rng, n=32000):
    t = np.arange(n) / 16000
    env = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)
    return (np.sin(2 * np.pi * 180 * t) * env * 0.3).astype(np.float32)


def test_param_path_hits_target_snr_and_channels_differ():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=0.0, p_clip=0.0, p_ref_dropout=0.0, p_wind=0.0, p_clean=0.0, snr_range=(5.0, 5.0))
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], None, cfg)
    assert mix.shape == (2, len(s)) and clean.shape == (len(s),)
    assert not np.allclose(mix[0], mix[1])
    achieved = 10 * np.log10(mixer.speech_active_power(clean) / mixer.speech_active_power(mix[0] - clean))
    assert abs(achieved - 5.0) < 0.5
    # reference carries much less speech than primary
    assert meta["path"] == "param" and meta["ref_speech_gain_db"] <= -8


def test_room_path_runs(tmp_path):
    rirs.build_bank(tmp_path / "b.npz", n=2, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=1.0, p_clean=0.0)
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], bank, cfg)
    assert meta["path"] == "room" and np.isfinite(mix).all()


def test_clean_bucket_is_identity():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_clean=1.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0, p_room=0.0)
    s = _speech(rng)
    mix, clean, meta = mixer.mix(rng, s, [rng.standard_normal(len(s)).astype(np.float32)], None, [], None, cfg)
    assert meta["clean_bucket"] and np.allclose(mix[0], clean, atol=1e-6)
