"""scripts/r8_test_table.py on synthetic CSVs with the PROTOCOL columns (no r8 test score is ever read here)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import r8_test_table as T  # noqa: E402

COLS = ["system", "id", "bucket", "noise_class", "snr_in", "clipped", "ref_dropout", "impulse_peak_db", "fault",
        "speech_source", "snr_out", "si_sdr", "stoi", "pesq_wb", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
        "recovery_s", "asr_text", "category", "noise_source", "impulse_source"]
QUAL = {"r8": (21.0, 0.95, 3.2), "r7": (16.0, 0.88, 2.6), "raw": (0.0, 0.60, 1.2)}   # snr_out offset, stoi, pesq


def _items():
    """(category, bucket, snr_in, noise_source, impulse_source, fault, clipped, ref_dropout) per synthetic item."""
    out = []
    for snr in (-5.0, 5.0, 10.0):
        for i in range(6):
            out.append(("v1/stationary", f"stationary_{snr:g}", snr, f"dnsn:n{i % 3}", "", "", i == 0 and snr == 5.0, False))
    for i in range(4):
        out.append(("v1/clean", "clean_inf", np.inf, "", "", "", False, False))
        out.append(("defence/gunshot", "gunshot_0", 0.0, f"mad:b{i}", f"gun:g{i % 2}", "", False, False))
        out.append(("loud/ears_loud", "loud_ears_0", 0.0, f"mad:l{i}", "", "", False, False))
        out.append(("fault/fault_clip_mild", "fault_clip_mild_0", 0.0, "", "", "fault_clip_mild", True, False))
    for i, snr in enumerate([-12.0, -3.0, 4.0, 9.0, 16.0, 22.0]):
        out.append(("v2/patrol", "v2_patrol", snr, f"demand:S{i % 2}+x", "", "", i == 0, False))
        out.append(("v2/apc", "v2_apc", snr - 10, f"mad:v{i % 3}", "", "", True, False))
    return out


def _csv(path, role, spec=None, n_drop=0, dup=False):
    base, stoi, pesq = QUAL[role]
    rows = []
    for k, (cat, bucket, snr, ns, imp, fault, clipped, drop) in enumerate(_items()):
        s_in = snr if np.isfinite(snr) else 30.0
        rows.append({"system": spec or role, "id": f"{k:04d}", "bucket": bucket, "noise_class": "x", "snr_in": snr,
                     "clipped": clipped, "ref_dropout": drop, "impulse_peak_db": "", "fault": fault,
                     "speech_source": "ls:1", "snr_out": (s_in + base) if role != "raw" else s_in, "si_sdr": 0.0,
                     "stoi": stoi, "pesq_wb": pesq, "dnsmos_sig": 3, "dnsmos_bak": 3, "dnsmos_ovrl": 3.0 + stoi,
                     "recovery_s": 0.2 if cat == "defence/gunshot" else "", "asr_text": "", "category": cat,
                     "noise_source": ns, "impulse_source": imp})
    df = pd.DataFrame(rows, columns=COLS).iloc[n_drop:]
    if dup:
        df = pd.concat([df, df.iloc[:1]])
    df.to_csv(path, index=False)
    return path


def _paths(tmp_path, **kw):
    return {r: _csv(tmp_path / f"{r}.csv", r, **kw.get(r, {})) for r in T.ROLES}


N = len(_items())


def test_load_derives_subset_cluster_and_pass3(tmp_path):
    df = T.load(_paths(tmp_path), expect_items=N)
    assert set(df.subset) == {"v1", "defence", "loud", "fault", "v2"}
    g = df[(df.name == "gunshot") & (df.role == "r8")]
    assert set(g.cluster) == {"gun:g0", "gun:g1"}                     # gunshot: impulse recording
    assert set(df[df.name == "stationary"].cluster) == {"dnsn:n0", "dnsn:n1", "dnsn:n2"}
    f = df[(df.subset == "fault") & (df.role == "r8")]
    assert f.cluster.str.startswith("item:").all() and f.cluster.nunique() == len(f)   # no bed: own cluster
    assert df[df.subset == "v2"].snr.isna().all() and np.isinf(df[df.name == "clean"].snr).all()
    r8 = df[df.role == "r8"]; raw = df[df.role == "raw"]
    assert r8[r8.subset != "v2"].pass3.all() and not raw.pass3.any()
    assert df.clipped.dtype == bool


@pytest.mark.parametrize("kw,msg", [({"r8": {"n_drop": 1}}, "rows, expected"), ({"r7": {"dup": True}}, "duplicate"),
                                    ({"raw": {"spec": None}}, None)])
def test_load_refuses_short_or_duplicated(tmp_path, kw, msg):
    paths = _paths(tmp_path, **kw)
    if msg is None:
        assert len(T.load(paths, expect_items=N)) == 3 * N
        return
    with pytest.raises(ValueError, match=msg):
        T.load(paths, expect_items=N)


def test_load_refuses_mismatched_sets_partial_files_and_missing_category(tmp_path):
    paths = _paths(tmp_path)
    short = _csv(tmp_path / "r7_short.csv", "r7", n_drop=2)
    with pytest.raises(ValueError, match="different"):
        T.load({**paths, "r7": short}, expect_items=0)
    assert len(T.load({**paths, "r7": short}, expect_items=0, allow_partial=True)) == 3 * N - 2
    part = _csv(tmp_path / "r8.partial1.csv", "r8")
    with pytest.raises(ValueError, match="partial"):
        T.load({**paths, "r8": part}, expect_items=N)
    d = pd.read_csv(paths["raw"]).drop(columns="category"); d.to_csv(tmp_path / "nocat.csv", index=False)
    with pytest.raises(ValueError, match="category"):
        T.load({**paths, "raw": tmp_path / "nocat.csv"}, expect_items=N)
    d = pd.read_csv(paths["raw"]); d.loc[0, "system"] = "other"; d.to_csv(tmp_path / "two.csv", index=False)
    with pytest.raises(ValueError, match="one system"):
        T.load({**paths, "raw": tmp_path / "two.csv"}, expect_items=N)


def _section(md, head):
    return md.split(head, 1)[1].split("\n## ", 1)[0]


def test_table_layout_marks_and_paired_deltas(tmp_path):
    md, cells = T.build(T.load(_paths(tmp_path), expect_items=N), n_boot=50, hash_note="HASHNOTE")
    assert "HASHNOTE" in md
    heads = [l for l in md.splitlines() if l.startswith("## ")]
    # pre-registered order: headline, per-cell subsets, v2 scenes, fault kept apart after them, recovery, limits
    order = ["## Headline", "## v1 subset", "## defence", "## Lombard/loud", "## v2 mixer scenes: per scene",
             "## reliability faults", "## Recovery", "## Known limits"]
    idx = [next(i for i, h in enumerate(heads) if h.startswith(o)) for o in order]
    assert idx == sorted(idx)
    head = _section(md, "## Headline")
    nom = [l for l in head.splitlines() if l.startswith("| v1-nominal envelope | r8 |")][0]
    assert "| 11 |" in nom                                            # 5 and 10 dB, 6 each, minus one clipped item
    assert nom.endswith("| 1.00 [1.00, 1.00] | PASS |")
    raw_nom = [l for l in head.splitlines() if l.startswith("| v1-nominal envelope | raw |")][0]
    assert raw_nom.count("FAIL") == 4 and raw_nom.endswith("| 0.00 [0.00, 0.00] | FAIL |")
    assert "r8 minus r7, paired per item" in head and "r8 minus raw, paired per item" in head
    d = head.split("r8 minus raw")[1]
    assert [l for l in d.splitlines() if l.startswith("| v1-nominal envelope |")][0].startswith(
        "| v1-nominal envelope | 11 | 21.00 [21.00, 21.00] | 0.350 [0.350, 0.350] | 2.00 [2.00, 2.00]")
    cell = _section(md, "## v1 subset")
    for tab in cell.split("\n\n"):                                  # every markdown table is rectangular
        rows = [l for l in tab.splitlines() if l.startswith("|")]
        assert len({l.count("|") for l in rows}) <= 1
    assert "| stationary | -5 | r7 | 6 |" in cell and "| clean | clean | r8 | 4 |" in cell
    loud = _section(md, "## Lombard/loud")
    assert "| ears_loud | 0 | r8 | 4 |" in loud and "optimistic" in loud
    r7r = [l for l in _section(md, "## defence").splitlines() if l.startswith("| gunshot | 0 | r7 |")][0]
    assert r7r.count("PASS") == 4 and r7r.endswith("| PASS |")         # 16 dB / 0.88 / 2.6 all clear
    assert "| 0.200 |" in _section(md, "## Recovery")


def test_v2_scene_rows_carry_snr_quantiles_and_bands(tmp_path):
    md, cells = T.build(T.load(_paths(tmp_path), expect_items=N), n_boot=50)
    v2 = _section(md, "## v2 mixer scenes: per scene")
    q = [l for l in v2.splitlines() if l.startswith("| patrol | 6 |")][0]
    want = np.quantile([-12.0, -3.0, 4.0, 9.0, 16.0, 22.0], [0.1, 0.5, 0.9])
    assert " / ".join(f"{v:.1f}" for v in want) in q and q.endswith("| 0.17 |")
    assert v2.index("| patrol |") < v2.index("| apc |")
    assert "| [-inf, -5) dB | r8 |" in v2 and "| [15, inf) dB | raw |" in v2
    band = [l for l in v2.splitlines() if l.startswith("| [15, inf) dB | r8 |")][0]
    assert "| 2 |" in band                                           # patrol 16 and 22 dB; apc tops out at 12
    c = cells[(cells.section == "v2_snr_in") & (cells.category == "v2/patrol") & (cells.metric == "snr_in_q50")]
    assert abs(c["mean"].iloc[0] - want[1]) < 1e-9


def test_fault_section_is_separate_and_cells_csv_holds_every_row(tmp_path):
    md, cells = T.build(T.load(_paths(tmp_path), expect_items=N), n_boot=50)
    f = _section(md, "## reliability faults")
    assert "fault_none" in f and "| fault_clip_mild | 0 | r8 | 4 |" in f
    x = cells[(cells.section == "cell") & (cells.category == "fault/fault_clip_mild") & (cells.system == "r8")
              & (cells.metric == "stoi")]
    assert len(x) == 1 and abs(x["mean"].iloc[0] - 0.95) < 1e-9
    assert {"r8-r7", "r8-raw"} <= set(cells.system)


def _bless(monkeypatch, paths):
    """Register the synthetic CSVs' v2 digest as root B's, so main() takes them as scored on the test root."""
    df = T.load(paths, expect_items=N)
    d = T.v2_digest(df[df.role == "r8"])
    monkeypatch.setattr(T, "V2_DIGEST", {T.EVALSET_HASH: d, "ed024af085a2": "0" * 12})
    return df, d


def test_hash_check_and_main_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "set"; root.mkdir()
    assert "not checked" in T.check_hash(root)
    (root / "EVALSET_HASH").write_text("deadbeef0000\n")
    with pytest.raises(ValueError, match="registers"):
        T.check_hash(root)
    (root / "EVALSET_HASH").write_text(T.EVALSET_HASH + "\n")
    p = _paths(tmp_path); _bless(monkeypatch, p)
    md, cells = T.main(["--r8", str(p["r8"]), "--r7", str(p["r7"]), "--raw", str(p["raw"]), "--eval-root", str(root),
                        "--expect-items", str(N), "--boot", "20", "--out", str(tmp_path / "out/table.md")])
    assert (tmp_path / "out/table.md").read_text(encoding="utf-8") == md and "matches PROTOCOL" in md
    assert len(pd.read_csv(tmp_path / "out/table_cells.csv")) == len(cells)


def test_defaults_are_the_protocol_paths():
    proto = (Path(__file__).resolve().parents[1] / "results_r2/r8/testset/PROTOCOL.md").read_text(encoding="utf-8")
    for p in T.DEFAULTS.values():
        assert p in proto
    assert T.EVALSET_HASH in proto and f"{T.N_ITEMS:,}" in proto


def _cols(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def test_pesq_nan_counts_and_footnote_under_every_pesq_table(tmp_path):
    p = _paths(tmp_path)
    d = pd.read_csv(p["r8"], dtype={"id": str})
    d.loc[d.category == "defence/gunshot", "pesq_wb"] = np.nan                                 # 4 PESQ-child failures
    d.loc[d.category == "v1/clean", ["snr_out", "stoi", "pesq_wb", "dnsmos_ovrl"]] = np.nan    # 4 failed clips, apart
    d.to_csv(p["r8"], index=False)
    r = pd.read_csv(p["raw"], dtype={"id": str}); r.loc[r.id == "0000", "pesq_wb"] = np.nan; r.to_csv(p["raw"], index=False)
    df = T.load(p, expect_items=N)
    assert int(df[df.role == "r8"].pesq_nan.sum()) == 4 and int(df[df.role == "raw"].pesq_nan.sum()) == 1
    md, cells = T.build(df, n_boot=20)
    assert "4 failed clips, 4 pesq_nan" in md and "0 failed clips, 1 pesq_nan" in md and T.PESQ_NOTE in md
    assert "2 of 2,280" in T.PESQ_NOTE and "results_r2/r8/native_crash/asan/sweep_relabel_test.tsv" in T.PESQ_NOTE
    lines = md.splitlines()
    heads = [i for i, l in enumerate(lines) if l.startswith("|") and "PESQ" in l and "---" not in l
             and ("| system |" in l or "dPESQ" in l)]
    assert len(heads) > 10
    for i in heads:                                              # every PESQ table: marked head, pesq_nan, footnote
        c = _cols(lines[i]); k = c.index("PESQ†") if "PESQ†" in c else c.index("dPESQ†")
        assert c[k + 1] == "pesq_nan"
        j = next(n for n in range(i, len(lines)) if not lines[n].startswith("|"))
        assert lines[j] == "" and lines[j + 1] == T.PESQ_FOOT
    assert md.count(T.PESQ_FOOT) == len(heads)
    sec = _section(md, "## defence").splitlines()
    g8 = _cols([l for l in sec if l.startswith("| gunshot | 0 | r8 |")][0])
    g7 = _cols([l for l in sec if l.startswith("| gunshot | 0 | r7 |")][0])
    assert g8[6] == "n/a" and g8[7] == "4" and g7[7] == "0"      # PESQ n/a (all NaN), pesq_nan 4 vs 0
    c = cells[(cells.section == "cell") & (cells.category == "defence/gunshot") & (cells.metric == "pesq_nan")]
    assert c.set_index("system")["mean"].to_dict() == {"r8": 4, "r7": 0, "raw": 0, "r8-r7": 4, "r8-raw": 4}
    h = cells[(cells.section == "headline") & (cells.category == "all") & (cells.subset == "v1")
              & (cells.metric == "pesq_nan")].set_index("system")["mean"].to_dict()
    assert h == {"r8": 0, "r7": 0, "raw": 1, "r8-r7": 0, "r8-raw": 1}   # paired: either side's NaN counts
    for tab in md.split("\n\n"):                                 # still rectangular with the new column
        rows = [l for l in tab.splitlines() if l.startswith("|")]
        assert len({l.count("|") for l in rows}) <= 1


def test_superseded_root_is_refused_unless_flagged(tmp_path, monkeypatch):
    root = tmp_path / "set"; root.mkdir(); (root / "EVALSET_HASH").write_text("ed024af085a2\n")
    with pytest.raises(ValueError, match="superseded"):
        T.check_hash(root)
    assert "SUPERSEDED" in T.check_hash(root, allow_superseded=True)
    p = _paths(tmp_path); df, d = _bless(monkeypatch, p)
    assert T.check_render(df).startswith("Every CSV")
    monkeypatch.setattr(T, "V2_DIGEST", {T.EVALSET_HASH: "0" * 12, "ed024af085a2": d})   # the CSVs came from root A
    with pytest.raises(ValueError, match="superseded root ed024af085a2"):
        T.check_render(df)
    assert T.check_render(df, allow_superseded=True).count("SUPERSEDED") == 3
    monkeypatch.setattr(T, "V2_DIGEST", {T.EVALSET_HASH: "0" * 12, "ed024af085a2": "1" * 12})
    with pytest.raises(ValueError, match="no known render"):
        T.check_render(df)
    assert "not verified" in T.check_render(df, allow_partial=True)
    monkeypatch.setattr(T, "V2_DIGEST", {T.EVALSET_HASH: "0" * 12, "ed024af085a2": d})
    args = ["--r8", str(p["r8"]), "--r7", str(p["r7"]), "--raw", str(p["raw"]), "--eval-root", str(root),
            "--expect-items", str(N), "--boot", "10", "--out", str(tmp_path / "o/table.md")]
    with pytest.raises(ValueError, match="superseded"):
        T.main(args)
    (root / "EVALSET_HASH").write_text(T.EVALSET_HASH + "\n")            # right root, CSVs from the old render
    with pytest.raises(ValueError, match="superseded root"):
        T.main(args)
    (root / "EVALSET_HASH").write_text("ed024af085a2\n")
    md, _ = T.main(args + ["--allow-superseded"])
    assert md.count("SUPERSEDED") == 4                                  # the root and each of the three CSVs


def test_v2_digests_match_the_rendered_indexes():
    """V2_DIGEST against both roots' index.csv (read only); skipped where no r8 test root is on disk (the box)."""
    repo = Path(__file__).resolve().parents[1]
    seen = 0
    for h, r in {T.EVALSET_HASH: T.EVAL_ROOT, **T.SUPERSEDED}.items():
        idx = repo / r / "index.csv"
        if not idx.exists():
            continue
        assert (repo / r / "EVALSET_HASH").read_text().strip() == h
        x = pd.read_csv(idx, dtype={"id": str, "bucket": str}, usecols=["bucket", "id", "category", "snr_db"])
        x = x.assign(subset=x.category.str.split("/", n=1).str[0], snr_in=pd.to_numeric(x.snr_db))
        assert T.v2_digest(x) == T.V2_DIGEST[h]; seen += 1
    if not seen:
        pytest.skip("no r8 test root on this machine")


def test_default_root_is_b_and_matches_the_testset_hash_file():
    assert T.EVAL_ROOT == "data/eval_r8_test_b/test" and T.EVALSET_HASH == "5bfda53eacbf"
    assert "ed024af085a2" in T.SUPERSEDED
    h = Path(__file__).resolve().parents[1] / T.TESTSET / "EVALSET_HASH"
    assert h.read_text().strip() == T.EVALSET_HASH
