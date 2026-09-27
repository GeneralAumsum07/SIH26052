"""Re-synthesis loss (plan Section 3.6 / Task 3)."""
import numpy as np
import pytest
import torch

from vaani import audio_contract as ac, losses
from vaani.dsp import low_delay_stft as ld, stft
from vaani.enhance_low_delay import (NativeFELoss, ResynthesisFELoss, WARMUP_SAMPLES, build_fe_loss,
                                     term_grad_norms)
from vaani.models import vaani_fe as V

A, B = ac.ARM_A_IDS[0], ac.ARM_B_ID
R8_LOSS = dict(w_mag=0.3, w_complex=0.2, w_consistency=0.3, w_wave=0.2, w_pesq=0.001, w_snr=0.002, kappa=1.0)


def _speech_like(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / 16000
    env = (torch.sin(2 * np.pi * 3 * t) > 0).float()
    x = sum(torch.sin(2 * np.pi * f * t) / k for k, f in enumerate((150, 300, 450, 900, 1800), 1)) * env
    return (x[None] * 0.1 + torch.randn(b, n, generator=g) * 0.003).float()


@pytest.mark.parametrize("cid", [A, B])
def test_alignment_of_target_reconstruction(cid):
    clean = _speech_like(2, 64000)
    target = ld.analyze(clean, cid)
    y, valid = ld.synthesize(target, torch.tensor([64000, 64000]), cid)
    torch.testing.assert_close(y[valid], clean[valid], atol=1e-5, rtol=1e-5)


def test_loss_is_resynthesis_through_unchanged_feloss():
    clean = _speech_like(2, 64000)
    noisy = clean + torch.randn_like(clean) * 0.05
    pred = ld.analyze(noisy, A)
    fe = losses.build_loss("fe", R8_LOSS)
    rl = ResynthesisFELoss(losses.build_loss("fe", R8_LOSS), A)
    got = rl(pred, clean)
    y, _ = ld.synthesize(pred, [64000, 64000], A)
    ref = fe.__class__(**R8_LOSS)
    ref.w["consistency"] = 0.0
    want = ref(stft.stft(y), stft.stft(clean))   # spectra-only call on the same waveform
    assert torch.isfinite(got) and abs(float(got) - float(want)) <= 1e-5 * max(1.0, abs(float(want)))
    assert rl.w["consistency"] == 0.0 and "consistency" in rl.last_terms
    assert float(rl.last_terms["consistency"]) < 1e-9          # zero up to rounding, still logged


def test_waveform_inputs_equal_spectra_only_call():
    clean = _speech_like(2, 32000, 1)
    noisy = clean + torch.randn_like(clean) * 0.05
    P, T = stft.stft(noisy), stft.stft(clean)
    fe = losses.build_loss("fe", R8_LOSS)
    a = fe(P, T)
    n = P.shape[2] * 256 - 256
    b = fe(P, T, y_pred=stft.istft(P, length=n), y_true=stft.istft(T, length=n))
    assert abs(float(a) - float(b)) <= 1e-6 * abs(float(a))


def test_lengths_exact_for_256_multiples_and_padded_otherwise():
    rl = ResynthesisFELoss(losses.build_loss("fe", R8_LOSS), A)
    clean = _speech_like(1, 64000)
    y, yc = rl.synthesize(ld.analyze(clean, A), clean)
    assert y.shape[-1] == 64000 and stft.stft(y).shape[2] == 251
    clean = _speech_like(1, 5000)
    y, yc = rl.synthesize(ld.analyze(clean, A), clean)
    assert y.shape[-1] == 5120 and (y[:, 5000:] == 0).all() and (yc[:, 5000:] == 0).all()


def test_warmup_and_padding_never_enter_the_loss():
    rl = ResynthesisFELoss(losses.build_loss("fe", dict(R8_LOSS, w_pesq=0.0)), A)
    n = WARMUP_SAMPLES + 64000
    clean = _speech_like(1, n, 2)
    noisy = clean + torch.randn_like(clean) * 0.05
    scored = (WARMUP_SAMPLES, n)
    a = rl(ld.analyze(noisy, A), clean, scored=scored)
    noisy2 = noisy.clone(); noisy2[:, :WARMUP_SAMPLES - 200] = torch.randn(1, WARMUP_SAMPLES - 200)  # prefix junk
    clean2 = clean.clone(); clean2[:, :WARMUP_SAMPLES - 200] = 0
    b = rl(ld.analyze(noisy2, A), clean2, scored=scored)
    # an identity spectrum resynthesizes exactly, so only the scored samples [7680, n) can move the loss:
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    y, yc = rl.synthesize(ld.analyze(noisy, A), clean, scored=scored)
    assert y.shape[-1] == 64000
    # padding past an item's length never enters: its samples are zeroed on both sides
    y, yc = rl.synthesize(ld.analyze(noisy, A), clean, lengths=[n - 1000])
    assert (y[:, n - 1000:] == 0).all() and (yc[:, n - 1000:] == 0).all()


@pytest.mark.parametrize("kind", ["silence", "partial", "speech", "transient"])
@pytest.mark.parametrize("cid,cfg", [(A, V.MINI_P["p18"]), (B, V.MINI_P["p32"])])
def test_finite_loss_and_gradients_through_mini_p(kind, cid, cfg):
    torch.manual_seed(0)
    m = V.build("mini", audio_contract=cid, gru_init="tc_matched", fp32_islands=True, **cfg)
    n = {"partial": 16000 + 37}.get(kind, 16000)
    clean = _speech_like(2, n) if kind != "silence" else torch.zeros(2, n)
    mix = clean + (torch.randn(2, n) * 0.02 if kind != "silence" else 0)
    if kind == "transient":
        mix[:, 8000:8050] += 0.9
    spec = torch.cat([ld.analyze(mix, cid), ld.analyze(mix * 0.7, cid)], -1)
    rl = ResynthesisFELoss(losses.build_loss("fe", R8_LOSS), cid)
    loss = rl(m(spec, None, torch.ones(2, spec.shape[2])).float(), clean, lengths=[n, n])
    loss.backward()
    assert torch.isfinite(loss).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert set(rl.last_terms) == set(losses.FELoss.TERMS)   # no term goes missing silently


def test_native_domain_loss():
    clean = _speech_like(2, 16000)
    nl = NativeFELoss(losses.build_loss("fe", R8_LOSS), A)
    target = ld.analyze(clean, A)
    loss = nl(target, clean)
    assert torch.isfinite(loss)
    assert float(nl.last_terms["mag"]) < 1e-12 and float(nl.last_terms["consistency"]) < 1e-9
    noisy = clean + torch.randn_like(clean) * 0.05
    assert float(nl(ld.analyze(noisy, A), clean)) > float(loss)
    assert nl.w["consistency"] == 0.3


def test_mrstft_on_synthesized_waveform_and_builder():
    lf = build_fe_loss(dict(R8_LOSS, w_mrstft=0.05), {"audio_contract": A})
    assert isinstance(lf, ResynthesisFELoss)
    clean = _speech_like(1, 16000)
    loss = lf(ld.analyze(clean + 0.01 * torch.randn_like(clean), A), clean)
    assert torch.isfinite(loss) and float(lf.last_terms["mrstft"]) > 0
    assert type(build_fe_loss(R8_LOSS, {})) is losses.FELoss                 # C0 keeps its native FELoss
    assert isinstance(build_fe_loss(R8_LOSS, {"audio_contract": A}, "native"), NativeFELoss)
    with pytest.raises(ValueError):
        ResynthesisFELoss(losses.build_loss("fe", R8_LOSS), ac.LEGACY_ID)


def test_term_gradient_norms_logged():
    torch.manual_seed(0)
    m = V.build("mini", audio_contract=A, **V.MINI_P["p18"])
    clean = _speech_like(1, 8000)
    spec = torch.cat([ld.analyze(clean, A), ld.analyze(clean, A)], -1)
    lf = build_fe_loss(dict(R8_LOSS, w_pesq=0.0), {"audio_contract": A})
    lf.fe.keep_live_terms = True
    loss = lf(m(spec, None, None), clean)
    g = term_grad_norms(lf, m.parameters())
    assert g["mag"] > 0 and g["wave"] > 0 and g["consistency"] == 0.0
    loss.backward()
