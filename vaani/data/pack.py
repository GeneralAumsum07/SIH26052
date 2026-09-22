"""Packed corpus: one int16 memmap plus an index, read instead of decoding FLAC per sample.

The training loader reads 3-5 audio files per item, 20k items per epoch. Each read was an
`sf.info` open (for the frame count) plus an `sf.read` open plus a FLAC decode. Packing the
corpus into a single int16 blob turns that into a page-cache slice.

Bit-identical by construction: the corpus is written as PCM_16 by `to_flac16k`, and libsndfile's
float32 conversion of a 16-bit source is exactly `int16 / 32768.0` (verified over 200 files
across six corpora, 2026-09-22). `scripts/pack_corpus.py --verify` re-checks it on demand.

The index carries the EXACT frame count read from each file at pack time. That matters more than
it looks: `_load` branches on `frames <= n`, and one frame of error either way would flip the
branch, change the number of rng draws, and silently desynchronise the training stream from
every previous run. A duration estimated from the manifest's `duration_s` float could do that;
a count read from the file cannot.
"""
import json
from pathlib import Path

import numpy as np

SR = 16000
SCALE = np.float32(1.0 / 32768.0)
INDEX_NAME, BLOB_NAME = "index.json", "audio.i16"


def _key(path) -> str:
    """Manifests carry posix-ish repo-relative paths on both platforms; normalise for lookup."""
    return str(path).replace("\\", "/").lstrip("./")


class PackedCorpus:
    """Read-only view over a packed corpus. Open lazily per worker: a memmap handle does not
    survive pickling across worker spawn, the same reason RirBank is opened lazily."""

    def __init__(self, root):
        self.root = Path(root)
        self._index = None
        self._blob = None

    @property
    def index(self) -> dict:
        """{} when the pack is unreadable, so every lookup misses and the caller falls back to
        soundfile. A pack removed mid-run (disk pressure on a rented box) must slow training
        down, not kill it - the audio it serves is only ever a faster copy of the real files."""
        if self._index is None:
            try:
                with open(self.root / INDEX_NAME, encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                self._index = {}
                return self._index
            if meta.get("sr") != SR:
                raise ValueError(f"pack sample rate {meta.get('sr')} != {SR}")
            self._index = meta["files"]
        return self._index

    @property
    def blob(self) -> np.memmap | None:
        if self._blob is None:
            try:
                self._blob = np.memmap(self.root / BLOB_NAME, dtype=np.int16, mode="r")
            except OSError:
                return None
        return self._blob

    def frames(self, path) -> int | None:
        e = self.index.get(_key(path))
        return None if e is None else e[1]

    def read(self, path, start: int = 0, frames: int | None = None) -> np.ndarray | None:
        """(frames, channels) float32, or 1-D when mono - matching sf.read's shape exactly.
        Returns None when the path is not packed, so the caller can fall back to soundfile."""
        e = self.index.get(_key(path))
        if e is None:
            return None
        off, n_frames, ch = e
        take = n_frames - start if frames is None else min(frames, n_frames - start)
        if take <= 0:
            return np.zeros((0,) if ch == 1 else (0, ch), np.float32)
        blob = self.blob
        if blob is None:
            return None
        a = off + start * ch
        x = np.asarray(blob[a:a + take * ch], dtype=np.int16).astype(np.float32) * SCALE
        return x if ch == 1 else x.reshape(take, ch)


def open_pack(root) -> PackedCorpus | None:
    """None when no pack is present, so an unpacked checkout still trains."""
    if not root:
        return None
    root = Path(root)
    return PackedCorpus(root) if (root / INDEX_NAME).exists() and (root / BLOB_NAME).exists() else None
