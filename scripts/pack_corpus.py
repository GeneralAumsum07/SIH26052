"""Pack every audio file named by the manifests into one int16 blob + index (see vaani/data/pack.py).

    uv run python scripts/pack_corpus.py --manifests data/manifests/*.parquet --out data/pack
    uv run python scripts/pack_corpus.py --out data/pack --verify        # re-check bit-identity

Packing is a pure read optimisation: the bytes it serves are the bytes soundfile would decode.
--verify proves that on a random sample rather than asserting it, because the whole point of the
pack is that nobody has to wonder whether the training data changed.
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani.data import manifests                                   # noqa: E402
from vaani.data.pack import BLOB_NAME, INDEX_NAME, SCALE, SR, _key  # noqa: E402


def _paths(patterns):
    out = []
    for pat in patterns:
        out.extend(sorted(glob.glob(pat)))
    if not out:
        raise SystemExit(f"no manifest matches {patterns}")
    return out


def build(manifest_paths, out: Path, chunk_frames=1 << 20):
    import pandas as pd
    df = pd.concat([manifests.read(p) for p in manifest_paths])
    paths = list(dict.fromkeys(df.path.astype(str)))   # de-dup, keep manifest order
    out.mkdir(parents=True, exist_ok=True)
    index, off, missing, skipped = {}, 0, 0, 0
    with open(out / BLOB_NAME, "wb") as blob:
        for i, p in enumerate(paths):
            if not Path(p).exists():
                missing += 1
                continue
            info = sf.info(p)
            if info.samplerate != SR:
                skipped += 1      # the pack is a 16 kHz store; anything else stays on the soundfile path
                continue
            x, _ = sf.read(p, dtype="int16", always_2d=True)
            ch = x.shape[1]
            blob.write(np.ascontiguousarray(x.reshape(-1), dtype="<i2").tobytes())
            index[_key(p)] = [off, int(x.shape[0]), int(ch)]
            off += x.shape[0] * ch
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(paths)} packed, {off * 2 / 1e9:.1f} GB", flush=True)
    meta = dict(sr=SR, dtype="int16", scale=float(SCALE), samples=off,
                manifests=[str(m) for m in manifest_paths], files=index)
    (out / INDEX_NAME).write_text(json.dumps(meta), encoding="utf-8")
    print(f"packed {len(index)} files, {off * 2 / 1e9:.2f} GB; {missing} missing, {skipped} non-16k skipped")
    return index


def verify(out: Path, n=200, seed=0):
    """Read n random packed files both ways and require exact equality."""
    from vaani.data.pack import PackedCorpus
    pk = PackedCorpus(out)
    keys = list(pk.index)
    random.Random(seed).shuffle(keys)
    bad = 0
    for k in keys[:n]:
        ref, _ = sf.read(k, dtype="float32")
        got = pk.read(k)
        if got is None or ref.shape != got.shape or not np.array_equal(ref, got):
            bad += 1
            if bad <= 3:
                print("  MISMATCH", k)
    print(f"verify: {min(n, len(keys)) - bad}/{min(n, len(keys))} bit-identical, {bad} mismatches")
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", nargs="*", default=["data/manifests/*.parquet"])
    ap.add_argument("--out", default="data/pack")
    ap.add_argument("--verify", action="store_true", help="only re-check an existing pack")
    ap.add_argument("--verify-n", type=int, default=200)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not a.verify:
        build(_paths(a.manifests), out)
    return 0 if verify(out, a.verify_n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
