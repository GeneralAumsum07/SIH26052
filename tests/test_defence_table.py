"""scripts/defence_table.py: per category x SNR cells, PASS/FAIL marks, pass3, paired deltas, gaps."""
import sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import defence_table as T


def _csv(path, system, snr_out, stoi, pesq, legacy=False):
    rows = []
    for cat in ("gunshot", "vehicle"):
        for snr in (-5.0, 5.0):
            for i in range(6):
                r = {"system": system, "id": f"{i:04d}", "bucket": f"{cat}_{snr:g}", "snr_in": snr,
                     "snr_out": snr_out + i * 0.1, "si_sdr": 0.0, "stoi": stoi, "pesq_wb": pesq, "dnsmos_ovrl": 3.0}
                if not legacy:
                    r.update(category=cat, noise_source=f"mad:v/{i % 3}", impulse_source=f"gunshots:g/{i % 2}" if cat == "gunshot" else "")
                rows.append(r)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_table_marks_pass3_and_delta(tmp_path):
    a = _csv(tmp_path / "sysA.csv", "sysA", 20.0, 0.95, 3.5)
    b = _csv(tmp_path / "raw.csv", "raw", 0.0, 0.5, 1.2)
    df = T.load([a, b])
    assert set(df.category) == {"gunshot", "vehicle"}
    # gunshot items cluster by recording, others by bed clip
    g = df[(df.category == "gunshot")]; v = df[df.category == "vehicle"]
    assert set(g.cluster) == {"gunshots:g/0", "gunshots:g/1"} and set(v.cluster) == {"mad:v/0", "mad:v/1", "mad:v/2"}
    md = T.table(df, ["sysA", "raw"], "raw", "abc123", n_boot=50)
    assert "abc123" in md and "## sysA minus raw" in md and "## Gaps" in md and "drone" in md
    row = next(l for l in md.splitlines() if l.startswith("| gunshot | -5 |") and "PASS" in l)
    assert row.count("PASS") == 3 and row.endswith("| 1.00 [1.00, 1.00] |")
    raw_row = [l for l in md.split("## raw")[1].splitlines() if l.startswith("| vehicle | 5 |")][0]
    assert raw_row.count("FAIL") == 3 and raw_row.endswith("| 0.00 [0.00, 0.00] |")
    delta = md.split("## sysA minus raw")[1]
    assert "| vehicle | 5 | 20.00" in delta   # paired SNR_out delta is exactly 20 per item


def test_legacy_csv_without_tags(tmp_path):
    df = T.load([_csv(tmp_path / "old.csv", "old", 10.0, 0.8, 2.0, legacy=True)])
    assert set(df.category) == {"gunshot", "vehicle"} and (df.cluster == "").all()
    md = T.table(df, ["old"], "raw", None, n_boot=20)
    assert "| vehicle | -5 |" in md and "~" not in md.split("## Gaps")[0].split("| vehicle | -5 |")[1].splitlines()[0]


def test_mark():
    assert T.mark("stoi", 0.9, 0.95) == "PASS" and T.mark("stoi", 0.8, 0.9) == "~" and T.mark("stoi", 0.5, 0.6) == "FAIL"
    assert T.mark("dnsmos_ovrl", 3, 3) == "" and T.mark("stoi", np.nan, np.nan) == ""
