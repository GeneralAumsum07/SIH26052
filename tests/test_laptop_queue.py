"""Laptop queue circuit breaker: consecutive FAILED runs halt the queue; a success resets the count."""
import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("laptop_queue", ROOT / "scripts/laptop_queue.py")
LQ = importlib.util.module_from_spec(spec); spec.loader.exec_module(LQ)


def _run(monkeypatch, tmp_path, outcomes, max_fails="2"):
    states, trained = {}, []
    monkeypatch.setattr(LQ, "Q", tmp_path)
    monkeypatch.setattr(LQ, "parse_queue", lambda: [(n, {}) for n in outcomes])
    monkeypatch.setattr(LQ, "run_name", lambda c: c)
    monkeypatch.setattr(LQ, "state", lambda n: states.get(n, "PENDING"))
    monkeypatch.setattr(LQ, "ensure_scorer", lambda h: None)
    monkeypatch.setenv("MAX_FAILS", max_fails)

    def train_one(c, o, dry):
        trained.append(c); states[c] = outcomes[c]
        return True
    monkeypatch.setattr(LQ, "train_one", train_one)
    rc = LQ.cmd_run(types.SimpleNamespace(dry_run=False))
    return rc, trained


def test_two_consecutive_failures_halt(monkeypatch, tmp_path):
    rc, trained = _run(monkeypatch, tmp_path, {"a": "DONE", "b": "FAILED", "c": "FAILED", "d": "DONE"})
    assert rc == 1 and trained == ["a", "b", "c"]


def test_success_resets_the_count(monkeypatch, tmp_path):
    rc, trained = _run(monkeypatch, tmp_path, {"a": "FAILED", "b": "DONE", "c": "FAILED", "d": "DONE"})
    assert rc == 0 and trained == ["a", "b", "c", "d"]
