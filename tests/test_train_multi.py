"""The shared-loader trainer is only correct when every config would have produced the same
batches. The r6 corpus arms differ by one manifest line, so a guard that fails open would make
them identical and void the comparison while still printing plausible numbers."""
import copy

import pytest
import yaml

from vaani.train_multi import assert_shared, stream_signature

BASE = dict(
    name="a", model="vaani", loss="hybrid", controller_on=True, batch_size=32, seed=0,
    epochs=64, device="cpu", optim=dict(lr=2.5e-4, warmup=200),
    dsp=dict(limiter=True, blocking=True),
    data=dict(manifests=["data/manifests/librispeech_100h.parquet"], bank="data/rirs/bank_r3.npz",
              crop_s=4.0, epoch_len=20000, mix=dict(impulse_peak_db=[15, 45])),
)


def _cfg(**over):
    c = copy.deepcopy(BASE); c.update(over); return c


def test_identical_data_configs_may_share():
    a, b = _cfg(name="a", epochs=32), _cfg(name="b", epochs=256)
    assert_shared([a, b])                       # epochs/optim/model_cfg may differ freely


def test_differing_manifests_are_refused():
    """The exact r6 mistake: ctl vs a corpus arm."""
    a = _cfg(name="r6_ctl64")
    b = _cfg(name="r6_wham64")
    b["data"] = copy.deepcopy(a["data"])
    b["data"]["manifests"] = a["data"]["manifests"] + ["data/manifests/wham.parquet"]
    with pytest.raises(SystemExit) as e:
        assert_shared([a, b])
    assert "r6_wham64" in str(e.value) and "data.manifests" in str(e.value)


@pytest.mark.parametrize("key,value", [
    ("batch_size", 16), ("seed", 1), ("controller_on", False), ("model", "gtcrn"), ("loss", "speech_preservation"),
])
def test_stream_defining_scalars_are_refused(key, value):
    with pytest.raises(SystemExit):
        assert_shared([_cfg(name="a"), _cfg(name="b", **{key: value})])


@pytest.mark.parametrize("key,value", [
    ("crop_s", 6.0), ("epoch_len", 10000), ("bank", "data/rirs/bank.npz"),
    ("mix", dict(impulse_peak_db=[-6, 12])), ("pack", "data/other_pack"),
])
def test_stream_defining_data_fields_are_refused(key, value):
    b = _cfg(name="b"); b["data"] = dict(b["data"], **{key: value})
    with pytest.raises(SystemExit):
        assert_shared([_cfg(name="a"), b])


def test_dsp_difference_is_refused():
    """dsp feeds the DataLoader workers, so it changes n_hat/feats in the batch itself."""
    b = _cfg(name="b", dsp=dict(limiter=False, blocking=False))
    with pytest.raises(SystemExit):
        assert_shared([_cfg(name="a"), b])


def test_signature_ignores_what_may_differ():
    a, b = _cfg(name="a", epochs=32), _cfg(name="b", epochs=256)
    b["optim"] = dict(lr=1e-3, warmup=500)
    b["model_cfg"] = dict(df_order=1)
    b["loss_cfg"] = dict(w_snr=0.8)
    assert stream_signature(a) == stream_signature(b)


def test_shipped_r6_arms_do_not_share_a_stream():
    """Guard against someone batching the real arms into one shared run."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "configs" / "retraining"
    arms = [root / f"r6_{n}64.yaml" for n in ("ctl", "demand", "wham")]
    if not all(p.exists() for p in arms):
        pytest.skip("r6 arms not present")
    cfgs = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in arms]
    with pytest.raises(SystemExit):
        assert_shared(cfgs)
