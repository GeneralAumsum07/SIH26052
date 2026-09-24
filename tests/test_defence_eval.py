"""vaani/eval.py carries index.csv tags (category, noise_source, impulse_source) into the result CSV."""
import csv, json, sys

import numpy as np, soundfile as sf

from vaani import eval as E

BASE = ["system", "id", "bucket", "noise_class", "snr_in", "clipped", "ref_dropout", "impulse_peak_db", "fault",
        "speech_source", "snr_out", "si_sdr", "stoi", "pesq_wb", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
        "recovery_s", "asr_text"]


def _set(root, index=True, meta_tags=None):
    rng = np.random.default_rng(0); split = root / "test"; rows = []
    for b in ("gunshot_0", "vehicle_5"):
        d = split / b; d.mkdir(parents=True)
        c = (rng.standard_normal(16000) * 0.1).astype(np.float32)
        m = np.stack([c + 0.05 * rng.standard_normal(16000), 0.05 * rng.standard_normal(16000)]).astype(np.float32)
        sf.write(d / "0000.mix.wav", m.T, 16000, subtype="FLOAT"); sf.write(d / "0000.clean.wav", c, 16000, subtype="FLOAT")
        meta = {"snr_db": 0.0, "noise_class": "stationary", "impulse_onsets_s": [], **(meta_tags or {})}
        json.dump(meta, open(d / "0000.json", "w"))
        rows.append({"bucket": b, "id": "0000", "category": b.split("_")[0], "noise_source": f"mad:{b}/1",
                     "impulse_source": "gunshots:g/1" if b.startswith("gun") else ""})
    if index:
        with open(split / "index.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return root


def _run(root, out, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval", "--system", "raw", "--eval-root", str(root), "--out", str(out),
                                      "--workers", "0", "--asr-threads", "1"])
    E.main()
    with open(out, encoding="utf-8", newline="") as fh:
        r = csv.reader(fh); head = next(r)
    return head, list(csv.DictReader(open(out, encoding="utf-8", newline="")))


def test_index_tags_are_appended_columns(tmp_path, monkeypatch):
    head, rows = _run(_set(tmp_path / "e"), tmp_path / "r.csv", monkeypatch)
    assert head == BASE + E.EXTRA_COLS
    by = {r["bucket"]: r for r in rows}
    assert by["gunshot_0"]["category"] == "gunshot" and by["gunshot_0"]["impulse_source"] == "gunshots:g/1"
    assert by["vehicle_5"]["noise_source"] == "mad:vehicle_5/1" and by["vehicle_5"]["impulse_source"] == ""
    assert all(np.isfinite(float(r["stoi"])) for r in rows)


def test_sets_without_an_index_keep_the_old_header(tmp_path, monkeypatch):
    head, _ = _run(_set(tmp_path / "e", index=False), tmp_path / "r.csv", monkeypatch)
    assert head == BASE


def test_meta_tags_win_over_the_index(tmp_path):
    row = E.fill_extra({"bucket": "gunshot_0", "id": "0000", "category": "from_meta", "noise_source": None},
                       {("gunshot_0", "0000"): {"category": "idx", "noise_source": "mad:x/1", "impulse_source": "s"}})
    assert row["category"] == "from_meta" and row["noise_source"] == "mad:x/1" and row["impulse_source"] == "s"


def test_resume_appends_only_missing_rows_and_cuts_a_torn_line(tmp_path, monkeypatch):
    root = _set(tmp_path / "e"); full, part = tmp_path / "full.csv", tmp_path / "part.csv"
    _run(root, full, monkeypatch)
    lines = full.read_bytes().splitlines(keepends=True)
    part.write_bytes(b"".join(lines[:2]) + lines[2][:10])   # header, one row, a row killed mid-write
    monkeypatch.setattr(sys, "argv", ["eval", "--system", "raw", "--eval-root", str(root), "--out", str(part),
                                      "--workers", "0", "--asr-threads", "1", "--resume"])
    E.main()
    assert part.read_bytes() == full.read_bytes()


def test_dead_worker_exits_nonzero_and_resume_finishes(tmp_path, monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    import pytest
    root = _set(tmp_path / "e"); full, out = tmp_path / "full.csv", tmp_path / "r.csv"
    _run(root, full, monkeypatch)
    real = E._stream

    def dies_after_one(ex, idx, window):
        it = real(ex, idx, window); yield next(it); raise BrokenProcessPool("worker died")
    monkeypatch.setattr(E, "_stream", dies_after_one)
    monkeypatch.setattr(sys, "argv", ["eval", "--system", "raw", "--eval-root", str(root), "--out", str(out),
                                      "--workers", "0", "--asr-threads", "1"])
    with pytest.raises(SystemExit) as ex:
        E.main()
    assert ex.value.code == 3 and len(out.read_bytes().splitlines()) == 2
    monkeypatch.setattr(E, "_stream", real)
    monkeypatch.setattr(sys, "argv", sys.argv + ["--resume"])
    E.main()
    assert out.read_bytes() == full.read_bytes()


def test_ordered_raises_when_a_real_worker_dies():
    import os
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    import pytest
    with ProcessPoolExecutor(1) as ex:
        with pytest.raises(BrokenProcessPool):
            list(E._ordered(ex, os._exit, [1, 1, 1], 2))


def test_ordered_keeps_order_with_a_bounded_window():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as ex:
        assert list(E._ordered(ex, lambda i: i * i, range(10), 3)) == [i * i for i in range(10)]
