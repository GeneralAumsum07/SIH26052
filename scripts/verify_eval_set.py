"""Gate a rendered/copied eval split on completeness, not on a copied hash file.

A half-finished transfer can carry a valid EVALSET_HASH (it is copied first), so the hash is recomputed
from the JSON metas present (a missing meta changes the digest) and every meta must have its mix/clean
WAVs; buckets that carry twin clips must have one per item. Exit 1 on any mismatch.
Usage: verify_eval_set.py data/eval_r2/test eda217ab2a38
"""
import hashlib, sys
from pathlib import Path


def verify(root: Path, expected: str) -> list[str]:
    problems = []
    metas = sorted(root.rglob("*.json"))
    if not metas:
        return [f"{root}: no metas"]
    h = hashlib.sha1()
    for p in metas:
        h.update(p.read_bytes())
        for suffix in (".mix.wav", ".clean.wav"):
            if not p.with_suffix(suffix).exists(): problems.append(f"missing {p.with_suffix(suffix)}")
    for d in {p.parent for p in metas}:
        twins, n = len(list(d.glob("*.twin.mix.wav"))), len(list(d.glob("*.json")))
        if 0 < twins < n: problems.append(f"{d}: {twins}/{n} twin clips")
    if (got := h.hexdigest()[:12]) != expected: problems.append(f"recomputed hash {got} != {expected} ({len(metas)} metas)")
    return problems


if __name__ == "__main__":
    root, expected = Path(sys.argv[1]), sys.argv[2]
    bad = verify(root, expected)
    print("\n".join(bad) if bad else f"{root}: {expected} verified ({len(list(root.rglob('*.json')))} items)")
    sys.exit(1 if bad else 0)
