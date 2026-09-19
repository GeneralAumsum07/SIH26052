import pandas as pd
from vaani.data import manifests


def _row(i, sha, split):
    return dict(source_id=f"s{i}", corpus="c", kind="noise", group_id=f"g{i}", speaker_id="", path=f"p{i}",
                duration_s=1.0, licence="", split=split, sha1=sha, noise_class="stationary")


def test_write_drops_byte_identical_audio(tmp_path):
    """DNS ships the same recording under several ids; one copy per split-hash would leak train into test."""
    manifests.write([_row(0, "aaa", "train"), _row(1, "aaa", "test"), _row(2, "bbb", "test")], tmp_path / "m.parquet")
    df = manifests.read(tmp_path / "m.parquet")
    assert len(df) == 2 and df.sha1.is_unique
