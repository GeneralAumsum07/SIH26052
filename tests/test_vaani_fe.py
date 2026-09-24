import itertools

import pytest
import torch

from vaani.models import vaani_fe as V

torch.set_num_threads(1)


def _rand(b, t, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 257, t, n, generator=g) * 0.1


def _stream(m, spec, avail):
    st, outs = m.init_state(spec.shape[0]), []
    for t in range(spec.shape[2]):
        o, st = m.step(spec[:, :, t:t + 1], None if avail is None else avail[:, t:t + 1], st)
        outs.append(o)
    return torch.cat(outs, 2), st


def _randomise_bn(m, seed=1):
    # eval-mode BN with non-trivial stats, so parity tests exercise the folded path
    g = torch.Generator().manual_seed(seed)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
            mod.running_var.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
    return m


@pytest.mark.parametrize("tier", list(V.TIERS))
def test_shapes_every_tier(tier):
    m = V.build(tier).eval()
    with torch.no_grad():
        y = m(_rand(2, 5))
        o, st = m.step(_rand(2, 1), torch.ones(2, 1), m.init_state(2))
    assert y.shape == (2, 257, 5, 2) and o.shape == (2, 257, 1, 2)
    assert st.shape == (2, m.state_size) and m.state_size == m.k * m.f * m.c2


def test_gemm_cell_equals_nn_gru():
    torch.manual_seed(0)
    rnn = torch.nn.GRU(24, 24, batch_first=True)
    x, h = torch.randn(16, 7, 24), torch.randn(1, 16, 24)
    ref, _ = rnn(x, h)
    hh, outs = h[0], []
    for t in range(7):
        hh = V.gru_cell(x[:, t], hh, rnn)
        outs.append(hh)
    assert (torch.stack(outs, 1) - ref).abs().max().item() < 1e-6


@pytest.mark.parametrize("tier", ["mini", "mid"])
def test_streaming_equals_offline(tier):
    m = _randomise_bn(V.build(tier)).eval()
    spec = _rand(2, 30)
    avail = (torch.rand(2, 30, generator=torch.Generator().manual_seed(3)) > 0.3).float()
    with torch.no_grad():
        ref = m(spec, None, avail)
        got, _ = _stream(m, spec, avail)
    assert (got - ref).abs().max().item() <= 1e-5


def test_causality_future_perturbation():
    m = _randomise_bn(V.build("mini", df_taps=3)).eval()
    a, b = _rand(1, 40), _rand(1, 40, seed=5)
    b[:, :, :25] = a[:, :, :25]
    with torch.no_grad():
        ya, yb = m(a), m(b)
    assert torch.equal(ya[:, :, :25], yb[:, :, :25])
    assert not torch.equal(ya[:, :, 25:], yb[:, :, 25:])


def test_validity_zero_with_zeroed_reference_is_finite_and_ignores_reference():
    m = V.build("mini").eval()
    spec = _rand(1, 12)
    spec[..., 2:] = 0
    with torch.no_grad():
        y0 = m(spec, None, torch.zeros(1, 12))
        junk = spec.clone(); junk[..., 2:4] = _rand(1, 12, 2, seed=9)
        y1 = m(junk, None, torch.zeros(1, 12))
    assert torch.isfinite(y0).all()
    assert torch.equal(y0, y1)  # validity 0 gates every reference-derived plane


def test_mini_fits_spec_budget_and_matches_prototype():
    s = V.summary(V.build("mini"))
    assert s["params"] <= 60_000 and s["params_training_form"] <= 60_000
    assert s["mmac_per_s"] <= 90.706
    # verified prototype: 29,274 params, 69.8 MMAC/s, 3,072 B state
    assert s["params"] == 29_274 and abs(s["mmac_per_s"] - 69.76) < 0.01 and s["state_bytes"] == 3072
    assert V.param_count(V.build("mini", norm="none")) == 29_274


@pytest.mark.parametrize("inputs,mask,df_taps,norm",
                         list(itertools.product(list(V.INPUTS), ["unbounded", "bounded"], [0, 2, 3], ["bn", "none"])))
def test_option_combinations_build_and_stream(inputs, mask, df_taps, norm):
    m = V.build("mini", inputs=inputs, mask=mask, df_taps=df_taps, norm=norm).eval()
    spec = _rand(1, 6)
    with torch.no_grad():
        ref = m(spec)
        got, st = _stream(m, spec, None)
    assert torch.isfinite(ref).all() and st.shape == (1, m.state_size)
    assert (got - ref).abs().max().item() <= 1e-5
    s = V.summary(m)
    assert s["params"] <= 60_000 and s["mmac_per_s"] <= 90.706  # every ablation arm stays in budget


def test_training_signature_matches_prepare_batch():
    m = V.build("mini")
    spec6, feats, avail = _rand(2, 8), torch.randn(2, 8, 18), torch.ones(2, 8)
    y = m(spec6, feats, avail)
    y.abs().mean().backward()
    assert y.shape == (2, 257, 8, 2)
    assert all(p.grad is not None for n, p in m.named_parameters() if not n.startswith("df."))


def test_nyquist_bin_reuses_bin_255_mask():
    m = V.build("mini", norm="none").eval()
    spec = _rand(1, 3)
    spec[:, 256, :, :2] = spec[:, 255, :, :2]
    with torch.no_grad():
        y = m(spec)
    assert torch.allclose(y[:, 256], y[:, 255])


def test_from_arch_and_invalid_options():
    m = V.from_arch({"model_cfg": {"tier": "mid", "df_taps": 2}})
    assert (m.c1, m.c2, m.f, m.k, m.l, m.df_taps) == (48, 40, 32, 3, 2, 2)
    with pytest.raises(ValueError):
        V.build("mini", inputs="bogus")
    with pytest.raises(ValueError):
        V.build("mini", df_taps=4)
