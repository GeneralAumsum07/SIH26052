"""Re-apply the noise_class rules in vaani.data.sources to existing manifests without re-scanning the audio.

Used after the 2026-09-20 crest audit moved MAD shooting/shelling/footsteps and four ESC-50 classes out of
"impulsive". Rows the class maps no longer call impulsive get the same label a fresh scan would give them.
Frozen eval renders on disk are untouched; re-rendering with the relabelled manifests changes the recorded_* buckets.
"""
import argparse

import pandas as pd
import soundfile as sf

from vaani.data.sources import ESC50_IMPULSIVE, MAD_CLASS_MAP, stationarity_class


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i, r in df[df.kind == "noise"].iterrows():
        corpus, cat = r.corpus, r.source_id.split(":", 1)[1].split("/", 1)[0]
        if corpus == "mad":
            df.at[i, "noise_class"] = MAD_CLASS_MAP[cat]
        elif corpus == "esc50" and cat not in ESC50_IMPULSIVE and r.noise_class == "impulsive":
            x, sr = sf.read(r.path, dtype="float32"); df.at[i, "noise_class"] = stationarity_class(x, sr)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("manifests", nargs="+"); a = ap.parse_args()
    for p in a.manifests:
        before = pd.read_parquet(p); after = relabel(before)
        changed = (before.noise_class != after.noise_class).sum()
        after.to_parquet(p, index=False)
        print(f"{p}: {changed} rows relabelled -> {after[after.kind == 'noise'].groupby('noise_class').size().to_dict()}")
