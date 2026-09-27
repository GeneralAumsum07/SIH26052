"""Audio contract registry (low-delay plan, Sections 3.1-3.2)."""
import dataclasses

import numpy as np
import pytest

from vaani import audio_contract as ac


def test_registry_holds_c0_and_every_low_delay_contract():
    ids = set(ac.registered())
    assert ac.LEGACY_ID in ids
    assert set(ac.ARM_A_IDS) <= ids and ac.ARM_B_ID in ids
    assert ac.get_audio_contract(None).audio_contract_id == ac.LEGACY_ID
    assert ac.contract_of({}).is_legacy and ac.contract_of({"tier": "mini"}).is_legacy


def test_unknown_contract_rejected():
    with pytest.raises(ValueError, match="unknown audio contract"):
        ac.get_audio_contract("vaanife_ld_asym512_h96_s999_v1")


@pytest.mark.parametrize("cid,h,l", [(ac.ARM_A_IDS[0], 96, 160), (ac.ARM_A_IDS[1], 96, 144),
                                     (ac.ARM_A_IDS[2], 96, 128), (ac.ARM_B_ID, 128, 160)])
def test_low_delay_fields(cid, h, l):
    c = ac.get_audio_contract(cid)
    assert (c.k, c.hop, c.support, c.crossfade, c.offset, c.history) == (512, h, l, l - h, h, 512 - h)
    assert c.hops_per_s == pytest.approx(16000 / h)
    assert c.lookahead_ms == pytest.approx((l - 1) / 2 / 16)
    assert c.ramp_samples == 3072 and c.limiter_sub == 32 and c.hop % 32 == 0
    assert c.resampler_id == ac.RESAMPLER_R1
    assert c.n_frames(64000) == -(-64000 // h) + 1


def test_c0_fields():
    c = ac.get_audio_contract(ac.LEGACY_ID)
    assert c.is_legacy and (c.hop, c.hops_per_s, c.n_frames(64000)) == (256, 62.5, 251)
    assert c.resampler_id == ac.RESAMPLER_R0


@pytest.mark.parametrize("cid", sorted(ac.registered()))
def test_windows_energy_and_product_overlap(cid):
    c = ac.get_audio_contract(cid)
    a, p, s = c.windows()
    assert (a * a).sum() == pytest.approx(256.0, abs=1e-9)
    ola = np.zeros(c.k + 20 * c.hop)
    for j in range(21):
        ola[j * c.hop:j * c.hop + c.k] += p
    assert np.allclose(ola[c.k:20 * c.hop], 1.0, atol=1e-12)
    if not c.is_legacy:
        assert np.allclose(a * s, p, atol=1e-15) and (p[:c.k - c.support] == 0).all()


def test_rejects_inconsistent_fields_and_altered_hash():
    c = ac.get_audio_contract(ac.ARM_A_IDS[0])
    for bad in (dict(support=200), dict(support=96), dict(crossfade=10), dict(hops_per_s=100.0),
                dict(window_hash="0" * 64), dict(offset=0), dict(window_energy=255.0), dict(k=256)):
        with pytest.raises(ValueError):
            dataclasses.replace(c, **bad).validate()


def test_roundtrip_and_record_verification():
    for c in ac.registered().values():
        d = c.to_dict()
        assert ac.verify_record(d) == c
        assert ac.AudioContract.from_dict(d).contract_hash == c.contract_hash
    d = ac.get_audio_contract(ac.ARM_A_IDS[0]).to_dict()
    d["ramp_samples"] = 2048
    with pytest.raises(ValueError):
        ac.verify_record(d)


def test_contract_hashes_distinguish_supports():
    hashes = {c.contract_hash for c in ac.registered().values()}
    assert len(hashes) == len(ac.registered())


def test_ld_windows_reject_support_outside_range():
    for h, l in ((96, 96), (96, 193), (128, 257)):
        with pytest.raises(ValueError):
            ac.ld_windows(512, h, l)
