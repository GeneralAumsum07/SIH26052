"""Mini-P (low-delay plan, Section 3.4 / Task 4): tiling, budget, validity term, deep filter, causality,
over-parameterization fold, FP32 islands and the time-constant-matched GRU initialization."""
import io

import pytest
import torch

from vaani import audio_contract as ac
from vaani.models import vaani_fe as V

torch.set_num_threads(1)
A, B = ac.ARM_A_IDS[0], ac.ARM_B_ID
SPEC_MMAC, SPEC_ENTRIES = 90.706, 60_000


def p18(**kw):
    return V.build("mini", audio_contract=A, **{**V.MINI_P["p18"], **kw})


def p32(**kw):
    return V.build("mini", audio_contract=B, **{**V.MINI_P["p32"], **kw})


def arm_r(**kw):
    return V.build("mini", audio_contract=A, df_bins=96, df_lags=(0, 3, 5), **kw)


def _rand(b, t, n=4, seed=0):
    return torch.randn(b, 257, t, n, generator=torch.Generator().manual_seed(seed)) * 0.1


def _randomise_bn(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
            mod.running_var.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
            mod.weight.data.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
            mod.bias.data.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
    return m


def _stream(m, spec, avail):
    st, outs = m.init_state(spec.shape[0]), []
    for t in range(spec.shape[2]):
        o, st = m.step(V.frame_to_step(spec[:, :, t:t + 1]), None if avail is None else avail[:, t:t + 1], st)
        outs.append(V.step_to_frame(o))
    return torch.cat(outs, 2), st


# ---- budget (Section 2.3) --------------------------------------------------------------------
@pytest.mark.parametrize("build,macs,params,train,mmac,df_bytes", [
    (p18, 521_504, 55_222, 55_862, 86.917, 3840),
    (p32, 697_376, 40_302, 40_942, 87.172, 4608),
    (arm_r, 1_153_024, 30_816, 31_456, 192.171, 3840),
])
def test_counts_reproduce_design_basis(build, macs, params, train, mmac, df_bytes):
    s = V.summary(build())
    assert (s["mac_per_hop"], s["params"], s["params_training_form"]) == (macs, params, train)
    assert s["mmac_per_s"] == pytest.approx(mmac, abs=1e-3)
    assert s["gru_state_bytes"] == 3072 and s["df_cache_bytes"] == df_bytes


@pytest.mark.parametrize("build", [p18, p32])
def test_spec_6_2_limits_at_contract_hop_rate(build):
    s = V.summary(build())
    assert s["mmac_per_s"] <= SPEC_MMAC and s["params_training_form"] <= SPEC_ENTRIES
    assert s["hops_per_s"] == pytest.approx(ac.get_audio_contract(build().cfg["audio_contract"]).hops_per_s)


def test_arm_r_exceeds_the_per_second_limit_by_design():
    assert V.summary(arm_r())["mmac_per_s"] > SPEC_MMAC


def test_core_is_the_minis():
    m, mini = p18(), V.build("mini")
    assert (m.c1, m.c2, m.f, m.k, m.l) == (mini.c1, mini.c2, mini.f, mini.k, mini.l)
    assert m.hidden_size == mini.hidden_size and m.n_pos == 18 and p32().n_pos == 32


# ---- legacy defaults -------------------------------------------------------------------------
def test_default_constructor_is_the_legacy_mini_and_checkpoints_load_bit_for_bit():
    torch.manual_seed(0)
    m = V.build("mini")
    assert set(m.cfg) == {"c1", "c2", "f", "k", "l", "heads", "inputs", "mask", "df_taps", "norm"}
    assert m.contract.audio_contract_id == ac.LEGACY_ID and V.summary(m)["hops_per_s"] == 62.5
    buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
    m2 = V.from_arch({"model_cfg": {"tier": "mini"}})
    m2.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    x = _rand(1, 5, 6)
    with torch.no_grad():
        assert torch.equal(m.eval()(x), m2.eval()(x))


def test_from_arch_accepts_new_fields_and_rejects_unknown():
    cfg = {"tier": "mini", "audio_contract": A, "freq_windows": "p18", "valid_bias": True, "df_bins": 96,
           "df_lags": [0, 3, 5], "gru_init": "tc_matched", "fp32_islands": True, "overparam": False}
    m = V.from_arch({"model_cfg": cfg})
    assert m.cfg["freq_windows"] == "p18" and m.cfg["df_lags"] == [0, 3, 5]
    with pytest.raises(TypeError):
        V.from_arch({"model_cfg": {"tier": "mini", "bogus": 1}})
    for bad in (dict(freq_windows="p99"), dict(freq_windows="p18", inputs="pr_nhat"), dict(valid_bias=True),
                dict(freq_windows="p18", df_bins=100, df_lags=(0, 1)), dict(df_lags=(1, 2)),
                dict(df_lags=(0, 2, 2)), dict(gru_init="orthogonal"), dict(audio_contract="nope")):
        with pytest.raises(ValueError):
            V.build("mini", **bad)


# ---- validity term ---------------------------------------------------------------------------
def test_validity_term_equals_a_convolved_constant_plane():
    """v x valid_vec == convolving a constant validity plane (kernel = stride = window width, no padding): each
    window's kernel sums to the same C1-vector, which is valid_vec."""
    torch.manual_seed(0)
    m = p18().eval()
    u = torch.randn(m.c1, 1)
    m.valid_vec.weight.data.copy_(u[..., None])
    planes = torch.randn(3, 4, 257)
    v = torch.tensor([[1.0], [0.0], [0.37]])
    got = torch.cat([conv(planes[..., b0:b1]) for conv, (_, b0, b1, _, _) in zip(m.inp, m.res)], -1) + m.valid_vec(v[..., None])
    ref = []
    for conv, (w, b0, b1, _, _) in zip(m.inp, m.res):
        wv = torch.rand(m.c1, 1, w)
        wv = wv / wv.sum(-1, keepdim=True) * u[..., None]     # any split whose taps sum to u
        plane = torch.cat([planes[..., b0:b1], v[..., None].expand(-1, 1, b1 - b0)], 1)
        ref.append(torch.nn.functional.conv1d(plane, torch.cat([conv.weight, wv], 1), stride=w))
    torch.testing.assert_close(got, torch.cat(ref, -1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("build", [p18, p32])
def test_validity_zero_ignores_reference_and_silence_is_finite(build):
    m = _randomise_bn(build()).eval()
    spec = _rand(1, 12); spec[..., 2:] = 0
    junk = spec.clone(); junk[..., 2:4] = _rand(1, 12, 2, seed=9)
    with torch.no_grad():
        y0, y1 = m(spec, None, torch.zeros(1, 12)), m(junk, None, torch.zeros(1, 12))
        ys = m(torch.zeros(1, 257, 12, 4))
    assert torch.equal(y0, y1) and torch.isfinite(ys).all() and torch.isfinite(y0).all()


# ---- streaming, cache, causality ---------------------------------------------------------------
@pytest.mark.parametrize("build", [p18, p32, arm_r])
def test_sequence_step_parity_with_lagged_cache(build):
    m = _randomise_bn(build()).eval()
    spec = _rand(2, 30)
    avail = (torch.rand(2, 30, generator=torch.Generator().manual_seed(3)) > 0.3).float()
    with torch.no_grad():
        ref = m(spec, None, avail)
        got, st = _stream(m, spec, avail)
    assert (got - ref).abs().max().item() <= 1e-5
    assert st.shape == (2, m.state_size)


def test_cache_holds_lagged_frames_oldest_first():
    m = p18().eval()
    st = m.init_state(1)
    frames = []
    with torch.no_grad():
        for t in range(7):
            x = torch.randn(1, 4, 257) * 0.1
            _, pc = m._planes(x, torch.ones(1, 1))
            frames.append(pc[..., :96].reshape(1, -1))
            _, st = m.step(x, torch.ones(1, 1), st)
    cache = st[:, m.hidden_size:].reshape(1, 5, 192)
    for j in range(5):   # after frame 6 the cache holds frames 2..6, oldest first: lags 5..1 of the next frame
        torch.testing.assert_close(cache[:, j], frames[2 + j])


def test_reset_restores_the_initial_output():
    m = _randomise_bn(p18()).eval()
    x = torch.randn(1, 4, 257) * 0.1
    with torch.no_grad():
        o0, _ = m.step(x, torch.ones(1, 1), m.init_state(1))
        st = m.init_state(1)
        for _ in range(5):
            _, st = m.step(torch.randn(1, 4, 257), torch.ones(1, 1), st)
        o1, _ = m.step(x, torch.ones(1, 1), m.init_state(1))
    assert torch.equal(o0, o1)


@pytest.mark.parametrize("build", [p18, p32])
def test_future_prefix_causality(build):
    m = _randomise_bn(build()).eval()
    a, b = _rand(1, 40), _rand(1, 40, seed=5)
    b[:, :, :25] = a[:, :, :25]
    with torch.no_grad():
        ya, yb = m(a), m(b)
    assert torch.equal(ya[:, :, :25], yb[:, :, :25]) and not torch.equal(ya[:, :, 25:], yb[:, :, 25:])


def test_reference_faults_are_finite():
    m = _randomise_bn(p18()).eval()
    spec = _rand(1, 20)
    spec[:, :, 5:9, 2:] = 0          # dropout
    spec[:, :, 12:, 2:] *= 1e3       # railed/loud reference
    avail = torch.ones(1, 20); avail[:, 5:9] = 0
    with torch.no_grad():
        assert torch.isfinite(m(spec, None, avail)).all()


def test_nyquist_is_a_learned_constant_mask():
    m = p18(norm="none").eval()
    with torch.no_grad():
        m.nyq.copy_(torch.tensor([0.5, 0.25]))
        x = _rand(1, 3)
        y = m(x)
    pc = x[:, 256, :, :2]
    mag = (pc.pow(2).sum(-1, keepdim=True) + V.EPS) ** ((V.ALPHA - 1) / 2)
    c = pc * mag
    yc = torch.stack([0.5 * c[..., 0] - 0.25 * c[..., 1], 0.5 * c[..., 1] + 0.25 * c[..., 0]], -1)
    ref = yc * (yc.pow(2).sum(-1, keepdim=True) + V.EPS) ** ((1 / V.ALPHA - 1) / 2)
    torch.testing.assert_close(y[:, 256], ref, atol=1e-6, rtol=1e-5)


def test_training_backward_reaches_every_parameter():
    m = p18()
    y = m(_rand(2, 8), None, torch.ones(2, 8))
    y.abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


# ---- over-parameterization ------------------------------------------------------------------
@pytest.mark.parametrize("build", [p18, p32, lambda **kw: V.build("mini", **kw)])
def test_overparam_folds_exactly_and_loads_into_the_plain_network(build):
    torch.manual_seed(0)
    m = build(overparam=True)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    for s in range(3):   # train a little so BN stats and factors are non-trivial
        y = m(_rand(4, 6, seed=s), None, torch.ones(4, 6))
        y.pow(2).mean().backward(); opt.step(); opt.zero_grad()
    m.eval()
    f = m.fold()
    x = _rand(2, 9, seed=7)
    with torch.no_grad():
        a, b = m(x, None, torch.ones(2, 9)), f(x, None, torch.ones(2, 9))
    assert ((a - b).abs().max() / a.abs().max()).item() <= 1e-6
    plain = build()
    plain.load_state_dict(f.state_dict(), strict=True)
    assert "overparam" not in f.cfg and V.summary(f)["mac_per_hop"] == V.summary(plain)["mac_per_hop"]
    assert V.param_count(f) == V.param_count(plain)
    # the step graph of the folded network equals the offline pass
    with torch.no_grad():
        got, _ = _stream(f.eval(), x, torch.ones(2, 9))
    assert (got - b).abs().max().item() <= 1e-5


def test_overparam_is_larger_only_while_training():
    m = p18(overparam=True)
    assert V.param_count(m) > V.param_count(p18())
    assert V.summary(m)["params"] == V.summary(p18())["params"]


# ---- FP32 islands ---------------------------------------------------------------------------
def test_fp32_island_switches_tf32_off_and_restores():
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        with V.fp32_island(True, "cpu"):
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def test_islands_match_float64_under_bf16_autocast():
    torch.manual_seed(0)
    m = p18(fp32_islands=True).eval()
    m64 = p18(fp32_islands=True).double().eval()
    m64.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in m.state_dict().items()})
    x = _rand(1, 4)
    x[..., :2] *= 1e-4   # very low level
    with torch.no_grad():
        planes, pc = m._planes(x.permute(0, 2, 3, 1).reshape(4, 4, 257), torch.ones(4, 1))
        p64, pc64 = m64._planes(x.double().permute(0, 2, 3, 1).reshape(4, 4, 257), torch.ones(4, 1, dtype=torch.float64))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            pb, _ = m._planes(x.permute(0, 2, 3, 1).reshape(4, 4, 257), torch.ones(4, 1))
            yb = m._decompress(pc)
    assert planes.dtype == torch.float32 and pb.dtype == torch.float32
    torch.testing.assert_close(planes.double(), p64, rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(pb, planes)
    assert yb.dtype == torch.float32
    torch.testing.assert_close(yb.double(), m64._decompress(pc64), rtol=1e-4, atol=1e-10)


def test_gru_island_runs_fp32_under_autocast_and_silence_is_finite():
    m = p18(fp32_islands=True)
    seq = torch.randn(3, 5, 24)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = m._gru_seq(m.blocks[0], seq)
        out = m(torch.zeros(1, 257, 6, 4), None, torch.ones(1, 6))
    assert y.dtype == torch.float32 and torch.isfinite(out.float()).all()
    torch.testing.assert_close(y, m.blocks[0].rnn(seq)[0])


# ---- GRU initialization ----------------------------------------------------------------------
def test_tc_matched_shift_matches_formula():
    torch.manual_seed(0)
    ref = p18()
    torch.manual_seed(0)
    m = p18(gru_init="tc_matched")
    for b0, b1 in zip(ref.blocks, m.blocks):
        c = 24
        before = b0.rnn.bias_ih_l0[c:2 * c] + b0.rnn.bias_hh_l0[c:2 * c]
        after = b1.rnn.bias_ih_l0[c:2 * c] + b1.rnn.bias_hh_l0[c:2 * c]
        torch.testing.assert_close(torch.sigmoid(after.double()), torch.sigmoid(before.double()) ** (96 / 256),
                                   atol=1e-6, rtol=1e-6)
        assert torch.equal(b0.rnn.bias_hh_l0, b1.rnn.bias_hh_l0)
    # b ~ 0: the bias rises by about +1.215 at H = 96 and +0.881 at H = 128
    assert V.tc_matched_bias(torch.zeros(1), 96).item() == pytest.approx(1.215, abs=2e-3)
    assert V.tc_matched_bias(torch.zeros(1), 128).item() == pytest.approx(0.881, abs=2e-3)


def test_tc_matched_consumes_no_random_numbers_and_skips_at_h256():
    torch.manual_seed(5)
    V.build("mini", audio_contract=A, gru_init="tc_matched", **V.MINI_P["p18"])
    a = torch.rand(3)
    torch.manual_seed(5)
    V.build("mini", audio_contract=A, **V.MINI_P["p18"])
    assert torch.equal(a, torch.rand(3))
    torch.manual_seed(1); m0 = V.build("mini")
    torch.manual_seed(1); m1 = V.build("mini", gru_init="tc_matched")   # C0 contract: H = 256
    for (k, x), (_, y) in zip(m0.state_dict().items(), m1.state_dict().items()):
        assert torch.equal(x, y), k


def test_audit_budget_has_a_vaanife_row_per_contract():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("audit_budget", Path(__file__).resolve().parents[1] / "scripts/audit_budget.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows = mod.fe_rows()
    contracts = {r["audio_contract"] for r in rows}
    assert set(ac.ARM_A_IDS) | {ac.ARM_B_ID, ac.LEGACY_ID} <= contracts
    for r in rows:
        assert r["within_budget"] == r["deployable"], r["name"]   # every deployable arm fits; Arm R never does
