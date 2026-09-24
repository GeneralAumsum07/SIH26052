"""Two RirBank constructions racing on a cold cache must not corrupt it.

RirBank unpacks bank*.npz into .npy memmaps on first use. The temp file it writes used to have a
fixed name, so concurrent constructions shared it: the first os.replace moved it away and the second
raised FileNotFoundError, having already left a partially written array behind. A later run then
mmapped the short file and died with "mmap length is greater than file size". run_r6.sh starts
training arms concurrently under ARMS_PARALLEL, so a fresh box could hit this on any parallel launch.
"""
import gc
import os
import threading

import numpy as np
import pytest

from vaani.data.rirs import RirBank

N, L = 8, 64


@pytest.fixture
def bank_npz(tmp_path):
    p = tmp_path / "bank_test.npz"
    rng = np.random.default_rng(0)
    np.savez(p, speech=rng.standard_normal((N, 2, L)).astype(np.float32),
             noise=rng.standard_normal((N, 2, L)).astype(np.float32),
             rt60=rng.random(N).astype(np.float32))
    return p


def test_concurrent_construction_leaves_a_complete_cache(bank_npz, monkeypatch):
    real_save = np.save
    start = threading.Barrier(2)
    first = threading.Event()

    def slow_save(file, arr, *a, **kw):
        # widen the window between writing the temp file and renaming it, so a shared
        # temp name is guaranteed to collide rather than merely likely to
        out = real_save(file, arr, *a, **kw)
        if not first.is_set():
            first.set()
            start.wait(timeout=5)
        return out

    monkeypatch.setattr(np, "save", slow_save)

    errors, banks = [], []

    def build():
        try:
            banks.append(RirBank(bank_npz))
        except Exception as e:                      # noqa: BLE001 - the failure under test
            errors.append(e)
        finally:
            try:
                start.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass

    ts = [threading.Thread(target=build) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)

    assert not errors, f"concurrent construction raised {errors!r}"
    assert len(banks) == 2

    # the cache on disk must be complete and readable by a fresh reader
    fresh = RirBank(bank_npz)
    assert len(fresh) == N
    assert fresh.speech.shape == (N, 2, L)
    assert fresh.noise.shape == (N, 2, L)
    assert not list(bank_npz.parent.glob("*.tmp.npy")), "a temp file was left behind"


def _windows_replace(real_replace):
    """os.replace as Windows behaves when the destination is mmapped by another reader: refused."""
    def replace(src, dst):
        if os.path.exists(dst):
            raise PermissionError(13, "Access is denied", str(dst))
        return real_replace(src, dst)
    return replace


def test_losing_writer_keeps_the_published_cache_on_any_os(bank_npz, monkeypatch):
    # the Windows branch runs on Linux too: a first reader publishes and mmaps, then a late writer finds every
    # destination locked and must keep the published files rather than raise
    first = RirBank(bank_npz)
    speech = np.array(first.speech)
    del first; gc.collect()                         # release the memmaps so Windows lets the unlink below through
    parts = sorted(bank_npz.parent.glob("bank_test.*.npy"))
    parts[0].unlink()                               # force the late writer back through the unpack path
    monkeypatch.setattr(os, "replace", _windows_replace(os.replace))
    late = RirBank(bank_npz)
    assert len(late) == N
    assert np.array_equal(np.asarray(late.speech), speech)
    assert not list(bank_npz.parent.glob("*.tmp.npy")), "a temp file was left behind"


def test_permission_error_without_a_published_file_still_raises(bank_npz, monkeypatch):
    def refuse(src, dst):
        raise PermissionError(13, "Access is denied", str(dst))
    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        RirBank(bank_npz)
    assert not list(bank_npz.parent.glob("*.tmp.npy")), "a temp file was left behind"
