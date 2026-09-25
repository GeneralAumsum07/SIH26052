"""Per-scene level-chain table from a data_gates.py --calib run (medians of the per-item CSV, shares of items).

usage: python results_r2/r8/calib/scene_table.py results_r2/r8/calib/trace_r8test   (writes scene_table.csv there)
"""
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1])
t = pd.read_csv(d / "level_trace_items.csv")
rows = []
for name, g in t.groupby("scene", sort=False):
    med = lambda c: round(float(pd.to_numeric(g[c], errors="coerce").median()), 1) if c in g else None   # noqa: E731
    share = lambda c: round(float(g[c].astype(float).mean()), 3)   # noqa: E731
    dom = {k: float((g[f"rails_dom_{k}"].fillna(0) * g["rails_sample_frac"]).sum())
           for k in ("speech", "bed", "near", "point", "wind", "impulse") if f"rails_dom_{k}" in g}
    tot = sum(dom.values())
    alone = lambda k: round(float((pd.to_numeric(g[f"{k}_peak"], errors="coerce") >= 123.01).mean()), 3) if f"{k}_peak" in g else 0.0   # noqa: E731
    rows.append(dict(
        scene=name, items=len(g), bed_A=med("bed_A"), bed_Z=med("bed_Z"), bed_Z_minus_A=round(med("bed_Z") - med("bed_A"), 1),
        bed_crest=med("bed_crest"), bed_peak=med("bed_peak"), speech_set=med("speech_spl_set"), speech_peak=med("speech_peak"),
        total_A=med("total_A"), total_Z=med("total_Z"), total_crest=med("total_crest"), total_peak=med("total_peak"),
        share_past_knee=share("past_knee"), share_past_aop=share("past_aop"), share_past_rails=share("past_rails"),
        knee_sample_frac=round(float(g["knee_sample_frac"].mean()), 4), rails_sample_frac=round(float(g["rails_sample_frac"].mean()), 5),
        overloaded_meta=share("overloaded_meta"), clipped_meta=share("clipped_meta"),
        alone_rails_speech=alone("speech"), alone_rails_bed=alone("bed"), alone_rails_near=alone("near"),
        alone_rails_wind=alone("wind"), alone_rails_impulse=alone("impulse"),
        railed_by=" ".join(f"{k}:{v / tot:.2f}" for k, v in sorted(dom.items(), key=lambda kv: -kv[1]) if v / tot >= 0.01) if tot else "",
        rails_if_Z=share("past_rails_if_Z"), rails_if_hpf_after=share("past_rails_if_hpf_after"),
        rails_if_near_split=share("past_rails_if_near_split"), rails_if_fs130=share("past_rails_if_fs130"),
        sat_err_db_knee105=med("sat_err_db_knee105"), sat_err_db_tanh120=med("sat_err_db_tanh120")))
out = pd.DataFrame(rows)
out.to_csv(d / "scene_table.csv", index=False)
print(out.to_string(index=False))
