"""Manifest = one parquet table per corpus listing every usable file.

Split assignment lives in the manifest so mixing code never decides membership.
"""
import hashlib
from pathlib import Path

import pandas as pd

COLUMNS = ["source_id", "corpus", "kind", "group_id", "speaker_id", "path",
           "duration_s", "licence", "split", "sha1", "noise_class"]
# kind: "speech" | "noise" | "impulse"
# group_id: speaker_id for speech; source-video/recording id for noise
# noise_class: stationary | changing | impulsive | "" for speech


def stable_hash(s: str) -> int:
    """Process-independent hash. Python's built-in hash() is salted per interpreter."""
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def write(rows: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS)
    # byte-identical files under different ids (56 in the DNS shard) hash to different splits: keep one
    dup = df.sha1.duplicated()
    if dup.any(): print(f"[manifest] dropping {int(dup.sum())} byte-identical duplicates"); df = df[~dup]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def content_hash(paths: list[str | Path]) -> str:
    """Hash of manifests used by a run, written into run.json."""
    h = hashlib.sha1()
    for p in sorted(map(str, paths)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]
