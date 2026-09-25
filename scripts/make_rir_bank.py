import argparse
from pathlib import Path

from vaani.data.rirs import MAX_LEN, SR, build_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rirs/bank.npz")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    # plan 2.9: --armoured-frac 0.2 --max-len-s 1.0 for the r3 training bank; the eval bank stays at the defaults
    ap.add_argument("--armoured-frac", type=float, default=0.0)
    ap.add_argument("--max-len-s", type=float, default=MAX_LEN / SR)
    ap.add_argument("--workers", type=int, default=None,
                    help="simulation processes; default all cores. Output is identical for any value")
    # M6 (plan 11.1) opt-ins; both unset = the legacy bank, byte-identical
    ap.add_argument("--receiver-radius", type=float, default=None,
                    help="armoured ray-tracer receiver radius in m (legacy 0.3); rays scale to keep the hit count")
    ap.add_argument("--seed-namespace", choices=["eval"], default=None,
                    help="'eval': a draw stream disjoint from every training bank of the same seed")
    a = ap.parse_args()
    build_bank(Path(a.out), a.n, a.seed, armoured_frac=a.armoured_frac,
               max_len=int(a.max_len_s * SR), workers=a.workers,
               receiver_radius=a.receiver_radius, seed_namespace=a.seed_namespace)
    print("wrote", a.out)


# build_bank simulates in a `spawn` pool, and spawn re-imports this file in every child. Without this
# guard each child re-ran the module-level build_bank() call and started a pool of its own, which goes
# exponential: on a 128-core box that meant load 117, a 111 MB log of multiprocessing bootstrap
# tracebacks, and no bank ever written. Keep the entry point behind the guard.
if __name__ == "__main__":
    main()
