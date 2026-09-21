import argparse
from pathlib import Path
from vaani.data.rirs import MAX_LEN, SR, build_bank

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/rirs/bank.npz")
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
# plan 2.9: --armoured-frac 0.2 --max-len-s 1.0 for the r3 training bank; the eval bank stays at the defaults
ap.add_argument("--armoured-frac", type=float, default=0.0)
ap.add_argument("--max-len-s", type=float, default=MAX_LEN / SR)
ap.add_argument("--workers", type=int, default=None, help="simulation processes; default all cores. Output is identical for any value")
a = ap.parse_args()
build_bank(Path(a.out), a.n, a.seed, armoured_frac=a.armoured_frac, max_len=int(a.max_len_s * SR), workers=a.workers)
print("wrote", a.out)
