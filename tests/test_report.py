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
