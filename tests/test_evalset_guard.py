"""The frozen eval sets are the project's only fixed measuring stick: every number in results_r2/
and every claim in the traceability doc is relative to data/eval_r2. Re-rendering over one silently
invalidates all of it, and nothing in the CLI used to stop that. These tests pin the guard."""
import sys

import pytest

from scripts import render_eval_sets


def _args(out, force=False):
    """The parsed-args shape main() reads, minus everything the guard runs before."""
    return type("A", (), {"manifests": [], "split": "test", "out": str(out), "bank": "b.npz",
                          "per_bucket": 1, "clip_s": 6.0, "seed": 1234, "faults": False, "force": force})()


def test_refuses_to_render_over_a_frozen_set(tmp_path):
    root = tmp_path / "eval_r2" / "test"
    root.mkdir(parents=True)
    (root / "EVALSET_HASH").write_text("deadbeef1234")
    with pytest.raises(SystemExit) as e:
        render_eval_sets.guard_frozen(_args(tmp_path / "eval_r2"))
    # the message must name the directory and the escape hatch, or the operator cannot act on it
    assert "EVALSET_HASH" in str(e.value) and "--force" in str(e.value)


def test_force_overrides_the_guard(tmp_path):
    root = tmp_path / "eval_r2" / "test"
    root.mkdir(parents=True)
    (root / "EVALSET_HASH").write_text("deadbeef1234")
    render_eval_sets.guard_frozen(_args(tmp_path / "eval_r2", force=True))   # must not raise


def test_a_fresh_directory_is_allowed(tmp_path):
    render_eval_sets.guard_frozen(_args(tmp_path / "eval_gen"))


def test_a_partially_written_directory_is_allowed(tmp_path):
    # an interrupted render leaves audio but no EVALSET_HASH; that set was never frozen, so resuming is fine
    root = tmp_path / "eval_gen" / "test" / "stationary_0"
    root.mkdir(parents=True)
    (root / "0000.mix.wav").write_bytes(b"")
    render_eval_sets.guard_frozen(_args(tmp_path / "eval_gen"))


def test_the_guard_runs_before_any_rendering(tmp_path, monkeypatch):
    # a guard that fires after the manifests are read and the RIR bank is loaded still wastes minutes
    # and, worse, may have already written files. It must be the first thing main() does.
    root = tmp_path / "eval_r2" / "test"
    root.mkdir(parents=True)
    (root / "EVALSET_HASH").write_text("deadbeef1234")

    def explode(*_a, **_k):
        raise AssertionError("main() touched the manifests before the guard fired")

    monkeypatch.setattr(render_eval_sets.manifests, "read", explode)
    monkeypatch.setattr(sys, "argv", ["render_eval_sets.py"])
    with pytest.raises(SystemExit):
        render_eval_sets.main(_args(tmp_path / "eval_r2"))


def test_both_r8_test_roots_are_refused_even_with_force(tmp_path, monkeypatch):
    # root B is the pre-registered r8 test set and the original render is superseded but frozen: neither is re-rendered
    assert set(render_eval_sets.PREREGISTERED) == {"data/eval_r8_test_b", "data/eval_r8_test"}
    monkeypatch.chdir(render_eval_sets.REPO)
    for rel in render_eval_sets.PREREGISTERED:
        for out in (rel, rel + "/", str(render_eval_sets.REPO / rel)):
            with pytest.raises(SystemExit) as e:
                render_eval_sets.guard_frozen(_args(out, force=True))
            assert "pre-registered" in str(e.value)
    # a new root beside them is not caught by the path rule
    render_eval_sets.guard_frozen(_args(tmp_path / "eval_r8_test_c", force=True))


def test_the_guard_names_the_registered_hash():
    reg = (render_eval_sets.REPO / "results_r2/r8/testset/EVALSET_HASH").read_text(encoding="utf-8").strip()
    assert render_eval_sets.PREREGISTERED["data/eval_r8_test_b"].startswith(reg)
    assert render_eval_sets.PREREGISTERED["data/eval_r8_test"].startswith("ed024af085a2")
