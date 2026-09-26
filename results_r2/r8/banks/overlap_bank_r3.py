"""Room overlap between RIR banks, for the bank_r3 ablation arm: (1) the builder's draws re-derived from the seed code
(no simulation), checked against every stored rt60; (2) the laptop files: speech RIRs of each derived shared pair.

usage (repo root): .venv/Scripts/python.exe results_r2/r8/banks/overlap_bank_r3.py results_r2/r8/banks/overlap_bank_r3.json
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from vaani.data.rirs import bank_rng, draw_room_params

# (seed, namespace, n, armoured_frac, shuffle) from configs/data/r8_banks.json built_with and
# results_r2/r8/testset/PROTOCOL.md. bank.npz predates the armoured mode (c053f7b), whose arm shuffle consumes draws
# even at frac 0: only the unshuffled legacy stream reproduces its stored rt60.
SPEC = {"bank": (0, None, 5000, 0.0, False), "bank_r3": (0, None, 5000, 0.2, True),
        "bank_r8": (8, "train", 5000, 0.2, True), "bank_eval_r8": (2610, "eval", 1000, 0.0, True)}
R = Path("data/rirs")


def params(seed, ns, n, frac, shuffle, n_noise=3):
    rng = bank_rng(seed, ns)
    arm = np.zeros(n, bool); arm[:int(round(n * frac))] = True
    if shuffle:
        rng.shuffle(arm)
    out = []
    for a in arm:
        p = draw_room_params(rng, n_noise, bool(a))
        key = np.concatenate([p["dims"], [p["rt60"]], p["head"], [p["yaw"]], *p["noise_pos"]]).astype(np.float64)
        out.append((hashlib.sha1(key.tobytes()).hexdigest(), bool(a), np.float32(p["rt60"])))
    return out


P = {k: params(*v) for k, v in SPEC.items()}
res = {"spec": {k: dict(zip(("seed", "namespace", "n", "armoured_frac", "shuffle"), v)) for k, v in SPEC.items()},
       "derivation_check": {}, "derived": {}, "files": {}}
for k in SPEC:   # the derivation is the bank only if it reproduces every stored (drawn) rt60
    f = R / f"{k}.rt60.npy"
    if f.exists():
        res["derivation_check"][k] = dict(rt60_equal=int((np.array([r for *_, r in P[k]], np.float32) == np.load(f)).sum()),
                                          n=len(P[k]))
pairs_r3_bank = []
for a, b in [("bank_r3", "bank"), ("bank_r3", "bank_eval_r8"), ("bank_r8", "bank"), ("bank_r8", "bank_eval_r8"),
             ("bank_r3", "bank_r8")]:
    hb = {h: i for i, (h, *_) in enumerate(P[b])}
    pairs = [(i, hb[h]) for i, (h, *_) in enumerate(P[a]) if h in hb]
    res["derived"][f"{a}~{b}"] = dict(shared_rooms=len(pairs), of_a=len(P[a]), of_b=len(P[b]),
                                      share_of_b=round(len(pairs) / len(P[b]), 4),
                                      shared_armoured_in_a=sum(P[a][i][1] for i, _ in pairs), first_pairs=pairs[:5])
    if (a, b) == ("bank_r3", "bank"):
        pairs_r3_bank = pairs

# the derived bank_r3 ~ bank pairs on the laptop files: equal rt60, speech RIR equal on bank.npz's 0.6 s length
rt = {k: np.load(R / f"{k}.rt60.npy") for k in ("bank", "bank_r3")}
sa = np.load(R / "bank_r3.speech.npy", mmap_mode="r"); sb = np.load(R / "bank.speech.npy", mmap_mode="r")
L = sb.shape[-1]
diffs = [float(np.max(np.abs(sa[i, :, :L] - sb[j]))) for i, j in pairs_r3_bank]
res["files"]["bank_r3~bank"] = dict(
    pairs=len(pairs_r3_bank), rt60_equal=int(sum(rt["bank_r3"][i] == rt["bank"][j] for i, j in pairs_r3_bank)),
    speech_rir_bitequal_on_prefix=int(sum(d == 0.0 for d in diffs)), speech_rir_maxdiff=max(diffs) if diffs else None,
    tail_beyond_prefix_nonzero=int(sum(bool(np.any(sa[i, :, L:])) for i, _ in pairs_r3_bank)),
    shapes=dict(bank_r3=list(sa.shape), bank=list(sb.shape)))
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "spec"}, indent=1))
