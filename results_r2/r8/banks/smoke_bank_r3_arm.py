"""Laptop smoke of the opt-in bank_r3 arm (scripts/gen_r8_configs.py --bank-arm): build its train dataset the way
vaani.train builds it, on the manifests present here and the laptop's bank_r3, draw N items and record which room
kind each room-path draw asked for and got. Also the pickled size of the train and val dataset objects (what a spawned
loader worker receives). No training step.

usage (repo root): .venv/Scripts/python.exe results_r2/r8/banks/smoke_bank_r3_arm.py results_r2/r8/banks/smoke_bank_r3_arm.json [N]
"""
import json, pickle, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import gen_r8_configs as G
from vaani import train
from vaani.data.dataset import DynamicMixDataset
from vaani.data.mixer import MixConfig

N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
fe = G.load(Path("."), G.FULL["fe"]); rv = G.load(Path("."), G.FULL["refvalid"])
cfg = next(c for s, c, _ in G.arms(fe, rv, bank_arm=True) if s == "ab7_bank_r3")
d = cfg["data"]
have = [m for m in d["manifests"] if Path(m).exists()]
absent = [m for m in d["manifests"] if m not in have]
dsk = dict(with_dsp=train.needs_dsp(cfg), controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"),
           pack_root=d.get("pack", "data/pack"), ref_corrupt=d.get("ref_corrupt"), fe_inputs=True,
           exclude_groups_file=d["exclude_groups_file"])
t0 = time.perf_counter()
ds = DynamicMixDataset(have, "train", d["bank"], MixConfig(**d["mix"]), d["crop_s"], d["epoch_len"], cfg["seed"], **dsk)
vds = DynamicMixDataset(have, "val", d["bank"], MixConfig(**d["mix"]), d["crop_s"], 200, cfg["seed"] + 1, **dsk)
build_s = time.perf_counter() - t0
pk = dict(train=len(pickle.dumps(ds)), val=len(pickle.dumps(vds)))   # before the lazy bank/pack handles open
calls = []
bank = ds.bank
orig = bank.sample


def spy(rng, armoured=None):
    r = orig(rng, armoured=armoured)
    calls.append(dict(asked_armoured=armoured, rt60=r["rt60"]))
    return r


bank.sample = spy
arm_rt = set(np.asarray(bank.rt60)[np.load(d["bank"])["armoured"]].tolist())
items = []
for i in range(N):
    n0 = len(calls)
    it = ds[i]
    m = it["meta"]
    for c in calls[n0:]:
        c["got_armoured"] = c["rt60"] in arm_rt
    items.append(dict(i=i, scene=(m.get("scene") or {}).get("name") if isinstance(m.get("scene"), dict) else m.get("scene"),
                      path=m.get("path"), rooms=calls[n0:], finite=bool(np.isfinite(it["mix"].numpy()).all()),
                      shape=list(it["mix"].shape), ref_avail=it.get("ref_avail") is not None))
out = dict(config="ab7_bank_r3 (gen_r8_configs.py --bank-arm)", bank=d["bank"], bank_pick_active=bank._pick is not None,
           manifests_used=have, manifests_absent_here=absent, items=items, build_s_smoke=round(build_s, 1),
           pickled_bytes=pk, rows=dict(train_speech=len(ds.speech), train_noise=len(ds.noise), val_speech=len(vds.speech),
                                       val_noise=len(vds.noise)),
           note="timing is SMOKE on a shared laptop, non-reportable; box-only manifests absent here")
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "items"}, indent=1, default=str))
for it in items:
    print(it["i"], it["scene"], it["path"], [(c["asked_armoured"], c["got_armoured"]) for c in it["rooms"]], it["finite"])
