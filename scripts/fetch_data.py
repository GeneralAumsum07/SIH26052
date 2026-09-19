"""Resumable fetch + manifest build. Run repeatedly; it skips what exists."""
import argparse
import tarfile
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

from vaani.data import manifests, sources


def download(url: str, dst: Path) -> None:
    # Hosts that ignore Range (GitHub archives) would otherwise re-download a finished file on every run.
    ok = dst.with_name(dst.name + ".ok")
    if ok.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:
            ok.touch()
            return
        r.raise_for_status()
        # Server may ignore Range and send 200 + full body; appending that would corrupt the file.
        if have and r.status_code != 206:
            have = 0
        mode = "ab" if have else "wb"
        total = int(r.headers.get("content-length", 0)) + have
        with open(dst, mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    ok.touch()


def extract(tar: Path, to: Path) -> None:
    if (to / ".done").exists():
        return
    to.mkdir(parents=True, exist_ok=True)
    if tar.suffix == ".zip":
        with zipfile.ZipFile(tar) as z:
            z.extractall(to)
    else:
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

    esc = cfg["sources"].get("esc50")
    if esc:
        zp = Path("data/download/esc50/master.zip")
        download(esc["url"], zp)
        extract(zp, Path("data/download/esc50"))
        manifests.write(sources.scan_esc50(Path(esc["extract_to"]), raw), mdir / "esc50.parquet")

    cv = cfg["sources"]["cv_hi"]
    if Path(cv["archive"]).exists():
        extract(Path(cv["archive"]), Path("data/download/cv_hi"))
    if Path(cv["extract_to"], "validated.tsv").exists():
        manifests.write(sources.scan_commonvoice_hi(Path(cv["extract_to"]), raw, cv["min_snr_db"]), mdir / "cv_hi.parquet")
    else:
        print("[cv_hi] not found - skipped (manual download required)")

    mad = cfg["sources"]["mad"]
    if Path(mad["archive"]).exists():
        extract(Path(mad["archive"]), Path("data/download/mad"))
    if Path(mad["extract_to"], "training.csv").exists():
        manifests.write(sources.scan_mad(Path(mad["extract_to"]), raw), mdir / "mad.parquet")
    else:
        print("[mad] not found - skipped (manual download required)")

    gun = cfg["sources"].get("gunshots")
    if gun:
        zp = Path(gun["archive"])
        download(gun["url"], zp)
        extract(zp, zp.parent)
        manifests.write(sources.scan_gunshots(Path(gun["extract_to"]), raw), mdir / "gunshots.parquet")

    dr = cfg["sources"].get("drone")
    if dr:
        zp = Path(dr["archive"])
        download(dr["url"], zp)
        extract(zp, zp.parent)
        manifests.write(sources.scan_drone(Path(dr["extract_to"]), raw), mdir / "drone.parquet")

    for url in a.dns_shards:
        tar = Path("data/download/dns") / Path(url).name
        download(url, tar)
        to = Path("data/download/dns") / tar.stem.replace(".tar", "")
        extract(tar, to)
        fn = sources.scan_dns_noise if "noise" in url else sources.scan_dns_speech
        manifests.write(fn(to, raw), mdir / f"dns_{tar.stem}.parquet")


if __name__ == "__main__":
    main()
