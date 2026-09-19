"""Transcribe each eval item's clean reference with the same Whisper model eval.py uses.
The eval sets carry no ground-truth text, so WER in the matrix is measured against this
transcript (ASR consistency), never against a human reference."""
import argparse, csv
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from vaani.data.dataset import RenderedDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="data/eval"); ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/asr/clean.csv"); ap.add_argument("--asr-device", default="cpu")
    a = ap.parse_args()
    from vaani.asr import load_whisper, transcribe
    asr = load_whisper(a.asr_device)
    ds = RenderedDataset(Path(a.eval_root) / a.split)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["id", "bucket", "asr_text"]); w.writeheader()
        for p in tqdm(ds.items, desc="clean asr"):
            clean, _ = sf.read(str(p).replace(".mix.wav", ".clean.wav"), dtype="float32")
            w.writerow(dict(id=p.name[: -len(".mix.wav")], bucket=p.parent.name, asr_text=transcribe(asr, clean)))


if __name__ == "__main__":
    main()
