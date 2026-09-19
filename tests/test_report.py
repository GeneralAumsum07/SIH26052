import pandas as pd
from vaani import report


def test_wer_counts_substitutions_insertions_deletions():
    assert report.wer("the cat sat", "the cat sat") == 0.0
    assert report.wer("the dog sat", "the cat sat") == 1 / 3
    assert report.wer("the cat", "the cat sat") == 1 / 3
    assert report.wer("the big cat sat", "the cat sat") == 1 / 3
    assert report.wer("The cat, sat!", "the cat sat") == 0.0
    assert pd.isna(report.wer("anything", ""))


def test_add_wer_joins_on_bucket_and_id(tmp_path):
    ref = tmp_path / "ref.csv"
    pd.DataFrame([dict(id="0000", bucket="b", asr_text="hello world")]).to_csv(ref, index=False)
    df = pd.DataFrame([dict(id="0000", bucket="b", asr_text="hello there"), dict(id="0001", bucket="b", asr_text="x")])
    out = report.add_wer(df, ref)
    assert out.wer.tolist()[0] == 0.5 and pd.isna(out.wer.tolist()[1])


def test_add_wer_leaves_missing_hypothesis_nan(tmp_path):
    ref = tmp_path / "ref.csv"
    pd.DataFrame([dict(id="0000", bucket="b", asr_text="hello world")]).to_csv(ref, index=False)
    df = pd.DataFrame([dict(id="0000", bucket="b", asr_text=float("nan"))])
    assert pd.isna(report.add_wer(df, ref).wer.iloc[0])


def test_report_counts_unrecovered_bursts(tmp_path):
    """inf recovery = burst clip never re-converged; it must be counted, not dropped with the no-burst NaNs."""
    import sys
    from vaani import report
    p = tmp_path / "x.csv"
    p.write_text("system,id,bucket,noise_class,snr_in,clipped,ref_dropout,impulse_peak_db,snr_out,si_sdr,stoi,pesq_wb,recovery_s,asr_text\n"
                 "s,0,impulsive_0,impulsive,0,False,False,6,10,10,0.9,2,0.1,\n"
                 "s,1,impulsive_0,impulsive,0,False,False,6,10,10,0.9,2,inf,\n"
                 "s,2,stationary_0,stationary,0,False,False,,10,10,0.9,2,,\n")
    out = tmp_path / "m.md"; old = sys.argv; sys.argv = ["report.py", str(p), "--out", str(out)]
    try: report.main()
    finally: sys.argv = old
    assert "failures=1/2" in out.read_text(encoding="utf-8")
