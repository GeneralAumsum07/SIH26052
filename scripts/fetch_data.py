"""Resumable fetch + manifest build. Run repeatedly; it skips what exists."""
import argparse
import tarfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from vaani.data import manifests, sources


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + have
        with open(dst, "ab") as f, tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def extract(tar: Path, to: Path) -> None:
    if (to / ".done").exists():
        return
    to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(to, filter="data")
    (to / ".done").touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/round1.yaml")
    ap.add_argument("--dns-shards", nargs="*", default=[], help="DNS-5 shard URLs for round 2")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    raw, mdir = Path(cfg["raw_root"]), Path(cfg["manifest_dir"])

    ls = cfg["sources"]["librispeech"]
    tar = Path("data/download/train-clean-100.tar.gz")
    download(ls["url"], tar)
    extract(tar, Path(ls["extract_to"]))
    manifests.write(sources.scan_librispeech(Path(ls["extract_to"]), raw, ls["max_hours"]), mdir / "librispeech.parquet")

    cv = cfg["sources"]["cv_hi"]
    if Path(cv["extract_to"], "validated.tsv").exists():
        manifests.write(sources.scan_commonvoice_hi(Path(cv["extract_to"]), raw, cv["min_snr_db"]), mdir / "cv_hi.parquet")
    else:
        print("[cv_hi] not found - skipped (manual download required)")

    mad = cfg["sources"]["mad"]
    if Path(mad["extract_to"]).exists():
        manifests.write(sources.scan_mad(Path(mad["extract_to"]), raw), mdir / "mad.parquet")
    else:
        print("[mad] not found - skipped (manual download required)")

    for url in a.dns_shards:
        tar = Path("data/download/dns") / Path(url).name
        download(url, tar)
        to = Path("data/download/dns") / tar.stem.replace(".tar", "")
        extract(tar, to)
        fn = sources.scan_dns_noise if "noise" in url else sources.scan_dns_speech
        manifests.write(fn(to, raw), mdir / f"dns_{tar.stem}.parquet")


if __name__ == "__main__":
    main()
