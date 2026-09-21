"""Refuse (exit 1) if any manifest still points into data/download: remote_setup.sh deletes that directory once the
scans are done, and a manifest row that survived there would train on a missing file."""
import glob
import sys

import pandas as pd

bad = [(f, p) for f in glob.glob("data/manifests/*.parquet") for p in pd.read_parquet(f, columns=["path"])["path"]
       if p.replace("\\", "/").lstrip("./").startswith("data/download")]
for f, p in bad[:10]:
    print(f"{f}: {p}", file=sys.stderr)
print(f"{len(bad)} manifest rows under data/download")
sys.exit(1 if bad else 0)
