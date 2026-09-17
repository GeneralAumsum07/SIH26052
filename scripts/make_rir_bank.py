import argparse
from pathlib import Path
from vaani.data.rirs import build_bank

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/rirs/bank.npz")
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
build_bank(Path(a.out), a.n, a.seed)
print("wrote", a.out)
