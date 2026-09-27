"""Low-delay plan Task 5: contract metadata in exported step graphs, contract-aware profiles, config hashes and
state import, sidecar/checkpoint agreement, and the extended parity (state, reset, interleaved streams)."""
import json

import numpy as np
import onnx
import pytest
import torch

from vaani import audio_contract as ac
from vaani import backend as bk
from vaani import export as E
from vaani import live
from vaani.models import vaani_fe as V

torch.set_num_threads(1)
A, A144, B = ac.ARM_A_IDS[0], ac.ARM_A_IDS[1], ac.ARM_B_ID
TOL = E.FE_PARITY_TOL


def p18(cid=A, seed=0, **kw):
    return E.fe_untrained({"tier": "mini", "audio_contract": cid, **V.MINI_P["p18"], **kw}, seed)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("ld_export")
    out = {}
    for name, m in (("a160", p18(A)), ("a144", p18(A144)), ("c0", E.fe_untrained("mini", 0))):
        out[name] = (m, E.export_fe(m, d / f"{name}.onnx", streams=2, hops=60))
    return out


def test_metadata_stamped_before_hash_and_kept_by_folding(exported):
    m, rep = exported["a160"]
    c = ac.get_audio_contract(A)
    for path, sha in ((rep["onnx"], rep["onnx_sha256"]), (rep["folded"], rep["folded_sha256"])):
        meta = E.onnx_metadata(path)
        assert meta[ac.META_ID] == A and meta[ac.META_HASH] == c.contract_hash
        assert ac.verify_record(json.loads(meta[ac.META_RECORD])) == c
        assert meta[ac.META_PROFILE] == "mini_p18"
        assert json.loads(meta[ac.META_MODEL_CFG])["audio_contract"] == A
        assert bk.file_sha256(path) == sha                  # the hash covers the stamped graph
    assert rep["audio_contract"] == A and rep["audio_contract_hash"] == c.contract_hash
    assert ac.contract_from_metadata(E.onnx_metadata(rep["folded"]), A) == c


def test_parity_includes_state_reset_and_interleaved_streams(exported):
    for name in ("a160", "a144", "c0"):
        p = exported[name][1]["parity"]
        for k in ("ort_vs_torch_max_abs", "state_max_abs", "stream_vs_offline_max_abs", "reset_max_abs",
                  "interleaved_max_abs"):
            assert p[k] <= TOL, (name, k, p[k])
        assert "state_max_rel" in p and "ort_vs_torch_max_rel" in p
        assert p["pass"] and not p["failed"]


def test_state_error_fails_parity(exported, monkeypatch):
    m, rep = exported["a160"]
    real = E.fe_stream_ort

    def skewed(sess, model, spec, valid):
        y, st = real(sess, model, spec, valid)
        return y, st + 1e-3
    monkeypatch.setattr(E, "fe_stream_ort", skewed)
    p = E.fe_parity(m, rep["folded"], streams=2, hops=20)
    assert not p["pass"] and "state" in p["failed"]


def test_backend_profile_deadline_and_contract(exported):
    b = bk.open_onnx(exported["a160"][1]["folded"])
    assert isinstance(b, bk.FeOrtBackend)
    assert b.profile_id == f"vaani_fe-mini_p18@{A}"
    assert b.audio_contract == ac.get_audio_contract(A)
    assert b.telemetry.deadline_ms == pytest.approx(6.0)
    legacy = bk.open_onnx(exported["c0"][1]["folded"])
    assert legacy.profile_id == "vaani_fe-mini" and legacy.telemetry.deadline_ms == bk.DEADLINE_MS
    assert legacy.audio_contract.is_legacy
    t = bk.FeTorchBackend(exported["a160"][0])
    assert t.profile_id == b.profile_id and t.audio_contract == b.audio_contract


def test_config_hash_carries_the_contract_and_leaves_c0_unchanged(exported):
    dsp = {"x": 1}
    c0 = bk.open_onnx(exported["c0"][1]["folded"])
    a160 = bk.open_onnx(exported["a160"][1]["folded"])
    a144 = bk.open_onnx(exported["a144"][1]["folded"])
    assert c0.config_hash(True, dsp) == bk.config_hash(True, dsp)
    assert a160.config_hash(True, dsp) != c0.config_hash(True, dsp)
    assert a160.config_hash(True, dsp) != a144.config_hash(True, dsp)
    assert a160.config_hash(True, dsp) == bk.config_hash(True, dsp, ac.config_extra(A))


def test_c0_mini_state_is_refused_by_an_arm_a_backend(exported):
    c0 = bk.open_onnx(exported["c0"][1]["folded"])
    a = bk.open_onnx(exported["a160"][1]["folded"])
    st = c0.to_host(c0.new_state(c0.config_hash(True, {})))
    with pytest.raises(ValueError):
        a.from_host(st, config_hash=a.config_hash(True, {}))
    with pytest.raises(ValueError, match="profile"):
        a.from_host(st)


def test_arm_a_state_is_refused_under_another_support_despite_identical_shapes(exported, tmp_path):
    a160 = bk.open_onnx(exported["a160"][1]["folded"])
    a144 = bk.open_onnx(exported["a144"][1]["folded"])
    assert a160.cache_specs() == a144.cache_specs()              # the shapes cannot tell them apart
    st = a160.new_state(a160.config_hash(True, {}))
    st.save(tmp_path / "s.npz")
    loaded = bk.StreamState.load(tmp_path / "s.npz")
    with pytest.raises(ValueError, match="profile"):
        a144.from_host(loaded)
    # even under a forced common profile id the config_hash refuses it
    forced = bk.FeOrtBackend(exported["a144"][1]["folded"], profile_id=a160.profile_id)
    with pytest.raises(ValueError, match="config_hash"):
        forced.from_host(loaded, config_hash=forced.config_hash(True, {}))
    assert a160.from_host(loaded, config_hash=a160.config_hash(True, {})).profile_id == a160.profile_id


def _strip(src, dst, drop=(), change=None):
    mo = onnx.load(str(src))
    props = {p.key: p.value for p in mo.metadata_props if p.key not in drop}
    props.update(change or {})
    del mo.metadata_props[:]
    onnx.helper.set_model_props(mo, props)
    onnx.save(mo, str(dst))
    return dst


def test_missing_metadata_is_never_a_low_delay_artifact(exported, tmp_path):
    bare = _strip(exported["a160"][1]["folded"], tmp_path / "bare.onnx", drop=(ac.META_ID, ac.META_HASH, ac.META_RECORD))
    with pytest.raises(ValueError, match="never accepted"):
        bk.open_onnx(bare, audio_contract=A)
    b = bk.open_onnx(bare)                                          # explicit legacy dispatch: C0
    assert b.audio_contract.is_legacy and "@" not in b.profile_id


def test_tampered_or_mismatched_metadata_is_refused(exported, tmp_path):
    src = exported["a160"][1]["folded"]
    with pytest.raises(ValueError):
        bk.open_onnx(_strip(src, tmp_path / "h.onnx", change={ac.META_HASH: "0" * 16}))
    with pytest.raises(ValueError):
        bk.open_onnx(_strip(src, tmp_path / "i.onnx", change={ac.META_ID: A144}))
    with pytest.raises(ValueError, match="expected"):
        bk.open_onnx(src, audio_contract=A144)
    with pytest.raises(ValueError):
        bk.open_onnx(exported["c0"][1]["folded"], audio_contract=A)


def test_sidecar_keeps_model_cfg_and_must_agree(exported, tmp_path):
    m, rep = exported["a160"]
    p = tmp_path / "model_config.json"
    live.write_fe_model_config(p, rep["folded"], "mini_p18", model_cfg=dict(m.cfg))
    cfg = live.load_model_config(p)
    assert cfg["model_cfg"]["audio_contract"] == A and cfg["audio_contract"] == A
    assert cfg["audio_contract_hash"] == ac.get_audio_contract(A).contract_hash
    raw = json.loads(p.read_text())
    for bad in ({ac.META_HASH: "0" * 16}, {ac.META_ID: A144}):
        q = tmp_path / "bad.json"
        q.write_text(json.dumps({**raw, **bad}))
        with pytest.raises(ValueError):
            live.load_model_config(q)
    q = tmp_path / "norecord.json"
    q.write_text(json.dumps({k: v for k, v in raw.items() if k not in (ac.META_ID, ac.META_HASH, ac.META_RECORD)}))
    with pytest.raises(ValueError, match="missing"):
        live.load_model_config(q)
    # a C0 sidecar is unchanged: no contract fields written
    c0 = tmp_path / "c0.json"
    live.write_fe_model_config(c0, exported["c0"][1]["folded"], "mini", model_cfg=dict(exported["c0"][0].cfg))
    assert ac.META_ID not in json.loads(c0.read_text())
    assert live.load_model_config(c0)["audio_contract"] == ac.LEGACY_ID


def test_legacy_stream_engine_refuses_a_low_delay_graph(exported):
    with pytest.raises(ValueError, match="LowDelayStreamEngine"):
        live.StreamEngine(exported["a160"][1]["folded"])
    with pytest.raises(ValueError):
        live.StreamEngine(exported["c0"][1]["folded"], audio_contract=A)


def test_overparam_checkpoint_is_folded_and_rechecked(tmp_path):
    torch.manual_seed(3)
    m = V.build("mini", audio_contract=A, overparam=True, **V.MINI_P["p18"])
    m.train()
    with torch.no_grad():                       # move the BatchNorm statistics off their defaults
        m(torch.randn(2, 257, 12, m.n_raw) * 0.1, None, torch.ones(2, 12))
    m.eval()
    rep = E.export_fe(m, tmp_path / "op.onnx", streams=2, hops=40)
    assert rep["overparam_fold"]["pass"] and rep["overparam_fold"]["fold_max_abs"] <= TOL
    assert "overparam" not in json.loads(rep["metadata"][ac.META_MODEL_CFG])
    assert rep["parity"]["pass"]
    ck = tmp_path / "best.pt"
    torch.save({"model": m.state_dict(), "config": {"model": "vaani_fe", "model_cfg": dict(m.cfg)},
                "audio_contract": ac.get_audio_contract(A).to_dict()}, ck)
    f = E.fe_load(ck)
    assert not f.overparam and f.contract.audio_contract_id == A
    assert E.fold_check(m, f) <= TOL
    torch.save({"model": m.state_dict(), "config": {"model": "vaani_fe", "model_cfg": dict(m.cfg)},
                "audio_contract": ac.get_audio_contract(A144).to_dict()}, ck)
    with pytest.raises(ValueError, match="recorded contract"):
        E.fe_load(ck)


def test_graph_gate_reads_the_contract(exported):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import graph_gate
    m, rep = exported["a160"]
    g = graph_gate.gate(rep["onnx"], m, streams=2, hops=30)
    assert g["checks"]["contract"] and g["audio_contract"]["audio_contract_id"] == A
    assert g["pass"], g["failed"]
    other = graph_gate.gate(rep["onnx"], exported["a144"][0], streams=2, hops=30)
    assert "contract" in other["failed"]


def test_every_low_delay_network_is_listed_once():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fe_tiers
    names = [n for n, _ in fe_tiers.low_delay_networks()]
    assert len(names) == len(set(names)) == 4 * 4 + 3 + 1 + 3
    for n, mc in fe_tiers.low_delay_networks():
        assert n.split("@")[1] == mc["audio_contract"]
