import torch

from vaani.models.residual_refiner import ResidualRefiner, count_params, init_refine_cache


def _specs(T=40, seed=0, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 257, T, 2, generator=g) * scale for _ in range(3)]   # P, R, Y


def _nonzero(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters(): p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return m


def test_param_count_and_zero_init_identity():
    m = ResidualRefiner()
    assert count_params(m) == (2498, 2498)
    P, R, Y = _specs()
    torch.testing.assert_close(m(P, R, Y), Y)


def test_finite_nonzero_gradients():
    m = ResidualRefiner(); P, R, Y = _specs()
    m(P, R, Y).pow(2).sum().backward()
    assert m.c2.weight.grad is not None and torch.isfinite(m.c2.weight.grad).all() and m.c2.weight.grad.abs().sum() > 0


def test_step_matches_forward_and_prefix_is_causal():
    m = _nonzero(ResidualRefiner()); P, R, Y = _specs(T=30)
    ref = m(P, R, Y)
    cache = init_refine_cache(); outs = []
    for t in range(30):
        z, cache = m.step(P[:, :, t:t + 1], R[:, :, t:t + 1], Y[:, :, t:t + 1], cache); outs.append(z)
    assert (torch.cat(outs, 2) - ref).abs().max() < 1e-4
    P2, R2, Y2 = P.clone(), R.clone(), Y.clone(); P2[:, :, 20:] = 5; R2[:, :, 20:] = -3; Y2[:, :, 20:] = 2
    torch.testing.assert_close(m(P2, R2, Y2)[:, :, :20], ref[:, :, :20])   # random future frames leave the prefix alone


def test_streams_do_not_contaminate_each_other():
    m = _nonzero(ResidualRefiner()); a = _specs(T=12, seed=3); b = _specs(T=12, seed=4)
    ca, cb = init_refine_cache(), init_refine_cache(); oa, ob = [], []
    for t in range(12):   # interleave two independent streams through one module
        za, ca = m.step(*(x[:, :, t:t + 1] for x in a), ca); zb, cb = m.step(*(x[:, :, t:t + 1] for x in b), cb)
        oa.append(za); ob.append(zb)
    assert (torch.cat(oa, 2) - m(*a)).abs().max() < 1e-4 and (torch.cat(ob, 2) - m(*b)).abs().max() < 1e-4
    c = init_refine_cache(); z0, _ = m.step(*(x[:, :, :1] for x in a), c)   # a reset cache starts a fresh stream
    torch.testing.assert_close(z0, m(*a)[:, :, :1])


def test_real_signal_constraint_and_finite_extremes():
    m = _nonzero(ResidualRefiner())
    P, R, Y = _specs(); z = m(P, R, Y) - Y
    assert (z[:, 0, :, 1] == 0).all() and (z[:, -1, :, 1] == 0).all()   # no imaginary correction at DC / Nyquist
    zeros = torch.zeros(1, 257, 20, 2)
    for P, R, Y in [(zeros, zeros, zeros), (torch.full_like(zeros, 1e6), torch.full_like(zeros, -1e6), torch.full_like(zeros, 1e6)),
                    (_specs(T=20)[0], zeros, _specs(T=20)[2])]:   # silence, huge but finite, reference dropout
        assert torch.isfinite(m(P, R, Y)).all()
