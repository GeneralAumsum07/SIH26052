"""Write data/manifests/librispeech_100h.parquet: the full train-clean-100 scan the r3+ recipes train on.
fetch_data.py only writes the 20 h librispeech.parquet, so every fresh box needed this by hand (2026-09-20, -21)."""
from pathlib import Path

from vaani.data import manifests, sources

if __name__ == "__main__":
    rows = sources.scan_librispeech(Path("data/download/librispeech"), Path("data/raw"), None)
    manifests.write(rows, Path("data/manifests/librispeech_100h.parquet"))
    print("librispeech_100h", len(rows))
