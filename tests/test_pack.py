"""The packed corpus must be a pure read optimisation: same bytes, same rng draws, or the
training stream silently desynchronises from every run that came before it."""
import json

import numpy as np
import pytest
import soundfile as sf

from vaani.data.dataset import _load
from vaani.data.pack import BLOB_NAME, INDEX_NAME, PackedCorpus, open_pack

SR = 16000


def _write_corpus(tmp_path, n_files=6, seed=0):
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_files):
        frames = int(rng.integers(SR // 2, 5 * SR))      # some shorter than a crop, some longer
        ch = 2 if i % 3 == 2 else 1                      # DEMAND-style stereo rows exist in the pool
        x = rng.integers(-32768, 32767, size=(frames, ch), dtype=np.int16)
        p = tmp_path / f"c{i}.flac"
        sf.write(p, x if ch > 1 else x[:, 0], SR, subtype="PCM_16")
        paths.append(p)
    return paths


def _pack(tmp_path, paths):
    root = tmp_path / "pack"; root.mkdir()
    index, off = {}, 0
    with open(root / BLOB_NAME, "wb") as blob:
        for p in paths:
            x, _ = sf.read(p, dtype="int16", always_2d=True)
            blob.write(np.ascontiguousarray(x.reshape(-1), dtype="<i2").tobytes())
            index[str(p).replace("\\", "/")] = [off, int(x.shape[0]), int(x.shape[1])]
            off += x.size
    (root / INDEX_NAME).write_text(json.dumps(dict(sr=SR, files=index)), encoding="utf-8")
    return root


def _load_reference(path, n, rng):
    """`_load` exactly as it was before the pack existed."""
    info = sf.info(path)
    if n is None or info.frames <= n:
        x, _ = sf.read(path, dtype="float32"); return x
    start = int(rng.integers(0, info.frames - n + 1))
    x, _ = sf.read(path, dtype="float32", start=start, frames=n); return x


@pytest.mark.parametrize("n", [None, 2 * SR, 4 * SR, 10 ** 9])
def test_packed_read_is_bit_identical_and_rng_preserving(tmp_path, n):
    paths = _write_corpus(tmp_path)
    pk = PackedCorpus(_pack(tmp_path, paths))
    for p in paths:
        r_ref, r_pack = np.random.default_rng([7, 1]), np.random.default_rng([7, 1])
        ref, got = _load_reference(str(p), n, r_ref), _load(str(p), n, r_pack, pk)
        assert ref.shape == got.shape, f"{p}: shape {ref.shape} != {got.shape}"
        assert np.array_equal(ref, got), f"{p}: packed bytes differ"
        # the branch consumes the rng identically, so every later draw in the item still lines up
        assert int(r_ref.integers(0, 2 ** 31)) == int(r_pack.integers(0, 2 ** 31)), f"{p}: rng diverged"


def test_unpacked_path_falls_back_to_soundfile(tmp_path):
    paths = _write_corpus(tmp_path, n_files=2)
    pk = PackedCorpus(_pack(tmp_path, paths[:1]))     # only the first file is packed
    rng_a, rng_b = np.random.default_rng([3]), np.random.default_rng([3])
    assert np.array_equal(_load(str(paths[1]), 2 * SR, rng_a, pk),
                          _load_reference(str(paths[1]), 2 * SR, rng_b))


def test_missing_pack_degrades_instead_of_raising(tmp_path):
    """Disk pressure on a rented box must slow training down, not kill it."""
    gone = PackedCorpus(tmp_path / "does_not_exist")
    assert gone.frames("anything.flac") is None
    assert gone.read("anything.flac") is None
    assert open_pack(tmp_path / "does_not_exist") is None
    paths = _write_corpus(tmp_path, n_files=1)
    rng_a, rng_b = np.random.default_rng([5]), np.random.default_rng([5])
    assert np.array_equal(_load(str(paths[0]), 2 * SR, rng_a, gone),
                          _load_reference(str(paths[0]), 2 * SR, rng_b))
