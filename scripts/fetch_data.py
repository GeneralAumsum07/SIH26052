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
    ap.add_argument("--only", default=None, help="comma-separated source names to (re)scan; default all")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    raw, mdir = Path(cfg["raw_root"]), Path(cfg["manifest_dir"])
    # a full rescan re-reads every clip of every source (~1 h) and once died on a locked cv_hi file while the
    # queue only wanted the gunshot zip; --only limits the run to the named sources
    only = set(a.only.split(",")) if a.only else None
    cfg["sources"] = {k: v for k, v in cfg["sources"].items() if only is None or k in only}

    ls = cfg["sources"].get("librispeech")
    if ls:
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

    cv = cfg["sources"].get("cv_hi")
    if cv and Path(cv["archive"]).exists():
        extract(Path(cv["archive"]), Path("data/download/cv_hi"))
    if cv and Path(cv["extract_to"], "validated.tsv").exists():
        manifests.write(sources.scan_commonvoice_hi(Path(cv["extract_to"]), raw, cv["min_snr_db"]), mdir / "cv_hi.parquet")
    elif cv:
        print("[cv_hi] not found - skipped (manual download required)")

    mad = cfg["sources"].get("mad")
    if mad and Path(mad["archive"]).exists():
        extract(Path(mad["archive"]), Path("data/download/mad"))
    if mad and Path(mad["extract_to"], "training.csv").exists():
        manifests.write(sources.scan_mad(Path(mad["extract_to"]), raw), mdir / "mad.parquet")
    elif mad:
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

    ears = cfg["sources"].get("ears")
    if ears:
        # per-speaker zips from GitHub Releases; scan whatever has finished downloading (the queue adds more)
        root = Path(ears["extract_to"]); root.mkdir(parents=True, exist_ok=True)
        for zp in sorted(Path(ears["download_dir"]).glob("p*.zip")):
            if zp.with_name(zp.name + ".ok").exists():
                extract(zp, root / zp.stem)
        if any(root.rglob("*.wav")):
            manifests.write(sources.scan_ears(root, raw), mdir / "ears.parquet")

    for url in a.dns_shards:
        tar = Path("data/download/dns") / Path(url).name
        download(url, tar)
        to = Path("data/download/dns") / tar.stem.replace(".tar", "")
        extract(tar, to)
        fn = sources.scan_dns_noise if "noise" in url else sources.scan_dns_speech
        manifests.write(fn(to, raw), mdir / f"dns_{tar.stem}.parquet")


if __name__ == "__main__":
    main()
