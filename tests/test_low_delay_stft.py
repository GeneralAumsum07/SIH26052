"""Low-delay asymmetric STFT: identity, lengths, streaming parity, validity (plan Task 1)."""
import numpy as np
import pytest
import torch

from vaani import audio_contract as ac
from vaani.dsp import low_delay_stft as ld, pipeline, stft

A, B = ac.ARM_A_IDS, ac.ARM_B_ID
LD = [*A, B]


@pytest.mark.parametrize("cid", LD)
def test_identity_fp64_representative_lengths(cid):
    c = ac.get_audio_contract(cid)
    g = torch.Generator().manual_seed(0)
    for n in (1, 95, 96, 97, 127, 128, 129, 143, 144, 145, 159, 160, 161, 511, 512, 513, 64000, 96000):
        x = torch.randn(1, n, dtype=torch.float64, generator=g)
        z = ld.analyze(x, c)
        y, mask = ld.synthesize(z, torch.tensor([n]), c)
        assert z.shape == (1, 257, (n + c.hop - 1) // c.hop + 1, 2)
        assert y.shape == x.shape and mask.all()
        torch.testing.assert_close(y, x, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("cid", LD)
def test_identity_fp32_normalized(cid):
    x = torch.rand(2, 16000, generator=torch.Generator().manual_seed(1)) * 2 - 1
    y, _ = ld.synthesize(ld.analyze(x, cid), torch.tensor([16000, 16000]), cid)
    assert (y - x).abs().max() <= 1e-5


@pytest.mark.parametrize("cid", LD)
def test_every_hop_offset(cid):
    c = ac.get_audio_contract(cid)
    base = 3 * c.hop
    x = torch.randn(1, base + c.hop, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    for off in range(c.hop):
        n = base + off + 1
        y, m = ld.synthesize(ld.analyze(x[:, :n], c), [n], c)
        torch.testing.assert_close(y, x[:, :n], atol=1e-12, rtol=0)


def test_batch_lengths_mask_and_zeroing():
    c = ac.get_audio_contract(A[0])
    x = torch.randn(2, 1000, dtype=torch.float64)
    x[1, 700:] = 0
    y, m = ld.synthesize(ld.analyze(x, c), torch.tensor([1000, 700]), c)
    assert m[0].all() and m[1, :700].all() and not m[1, 700:].any()
    assert (y[1, 700:] == 0).all()
    torch.testing.assert_close(y[:, :700], x[:, :700], atol=1e-12, rtol=0)


@pytest.mark.parametrize("cid", LD)
def test_dc_nyquist_silence(cid):
    n = 4000
    t = torch.arange(n, dtype=torch.float64)
    for x in (torch.ones(1, n, dtype=torch.float64), torch.cos(np.pi * t)[None], torch.zeros(1, n, dtype=torch.float64)):
        y, _ = ld.synthesize(ld.analyze(x, cid), [n], cid)
        torch.testing.assert_close(y, x, atol=1e-12, rtol=0)


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="empty"):
        ld.analyze(torch.zeros(1, 0), A[0])


@pytest.mark.parametrize("cid", LD)
def test_gradients_finite_and_projection_idempotent(cid):
    x = torch.randn(1, 1234, dtype=torch.float64, requires_grad=True)
    z = ld.analyze(x, cid)
    y, _ = ld.synthesize(z * 0.5, [1234], cid)
    y.pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()
    # analysis of a synthesized arbitrary spectrum, synthesized again: a projection
    zr = torch.randn_like(z.detach())
    y1, _ = ld.synthesize(zr, [1234], cid)
    y2, _ = ld.synthesize(ld.analyze(y1, cid), [1234], cid)
    torch.testing.assert_close(y2, y1, atol=1e-10, rtol=0)


@pytest.mark.parametrize("cid", LD)
def test_streaming_matches_offline(cid):
    c = ac.get_audio_contract(cid)
    for n in (c.hop - 1, c.hop, c.hop + 1, c.support + 1, 64000):
        x = np.random.default_rng(n).standard_normal(n)
        y = ld.stream_identity(x, c)
        assert np.allclose(y, x, atol=1e-12)
        zs = ld.np_analyze(x, c)
        zt = ld.analyze(torch.from_numpy(x)[None], c)[0].numpy()
        assert np.allclose(zs, zt[..., 0] + 1j * zt[..., 1], atol=1e-9)
        assert np.allclose(ld.np_synthesize(zs, n, c), x, atol=1e-12)


def test_stream_release_schedule():
    """Frame j releases stream samples [(j+1)H - L, (j+2)H - L): an impulse at sample m appears in the release of
    the first frame whose window ends at or after m, L - H samples into the release sequence."""
    c = ac.get_audio_contract(A[0])
    an, sy = ld.StreamAnalyzer(c, np.float64), ld.StreamSynthesizer(c, np.float64)
    x = np.zeros(20 * c.hop); m = 5 * c.hop + 7; x[m] = 1.0
    out = np.concatenate([sy.push(an.push(x[j * c.hop:(j + 1) * c.hop])) for j in range(20)])
    assert np.argmax(np.abs(out)) == m + c.release_lead
    j = m // c.hop   # frame computable once sample (j+1)H - 1 has arrived
    released_through = (j + 2) * c.hop - c.support
    assert released_through > m   # the impulse left in frame j's release: it waited (j+1)H - 1 - m + 1 <= L samples


def test_stream_state_export_import_and_mismatch():
    c = ac.get_audio_contract(A[0])
    x = np.random.default_rng(0).standard_normal(30 * c.hop)
    an, sy = ld.StreamAnalyzer(c, np.float64), ld.StreamSynthesizer(c, np.float64)
    ref = [sy.push(an.push(x[j * c.hop:(j + 1) * c.hop])) for j in range(30)]
    an2, sy2 = ld.StreamAnalyzer(c, np.float64), ld.StreamSynthesizer(c, np.float64)
    out = [sy2.push(an2.push(x[j * c.hop:(j + 1) * c.hop])) for j in range(10)]
    sa, ss = an2.export_state(), sy2.export_state()
    an3, sy3 = ld.StreamAnalyzer(c, np.float64), ld.StreamSynthesizer(c, np.float64)
    an3.import_state(sa); sy3.import_state(ss)
    out += [sy3.push(an3.push(x[j * c.hop:(j + 1) * c.hop])) for j in range(10, 30)]
    assert np.array_equal(np.concatenate(ref), np.concatenate(out))
    other = ld.StreamAnalyzer(ac.ARM_A_IDS[1], np.float64)
    with pytest.raises(ValueError):
        other.import_state(sa)   # same shapes (both H = 96), different support
    with pytest.raises(ValueError):
        ld.StreamSynthesizer(B).import_state(ss)


def test_c0_dispatch_uses_legacy_transform():
    x = torch.randn(2, 16000)
    z = ld.analyze(x, None)
    torch.testing.assert_close(z, stft.stft(x))
    y, m = ld.synthesize(z, [16000, 16000], ac.LEGACY_ID)
    torch.testing.assert_close(y, stft.istft(z, length=16000))
    with pytest.raises(ValueError):
        ld.StreamAnalyzer(ac.LEGACY_ID)


def test_c0_frame_validity_equals_frame_avail_bit_for_bit():
    rng = np.random.default_rng(3)
    for n in (4000, 16000, 64000, 64123):
        av = np.ones(n, bool)
        for _ in range(3):
            a = int(rng.integers(0, n)); av[a:a + int(rng.integers(1, 3000))] = False
        ref = pipeline.frame_avail(av, n // 256 + 1)
        got = ld.frame_validity(torch.from_numpy(av)[None], ac.LEGACY_ID)[0].numpy()
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("cid", LD)
def test_low_delay_frame_validity(cid):
    c = ac.get_audio_contract(cid)
    n = 5000
    av = np.ones(n, bool); av[1000] = False
    v = ld.frame_validity(torch.from_numpy(av)[None], c)[0].numpy()
    assert v.shape == (c.n_frames(n),)
    for j in range(len(v)):
        lo, hi = j * c.hop - (c.k - c.hop), (j + 1) * c.hop
        assert v[j] == float(not (lo <= 1000 < hi))
    # padding counts as known: all-available -> all valid, including the flush frame
    assert (ld.frame_validity(torch.ones(1, n), c) == 1).all()


def test_boundary_weights():
    c = ac.get_audio_contract(A[0])
    w = ld.boundary_weights(960, c)
    assert w.shape == (c.n_frames(960),)
    assert w[0] == pytest.approx(96 / 160) and w[3] == 1.0 and 0 < w[-1] < 1
