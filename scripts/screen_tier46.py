"""Tier 4.6 validation screen (plan §5): pick at most one post-filter setting on data/eval_r2/val, never on test.

The frozen anchor's first-stage spectra are computed once per val clip and cached on disk, keyed by the anchor's
SHA256, the val split's file digest, the checkpoint's DSP config and the STFT definition; a cache written for any
other checkpoint or split is refused rather than silently reused. Each grid setting is then just a cheap post-filter
+ iSTFT + metrics pass over the cached spectra. Per-setting CSVs use the vaani.eval layout so the selection reuses
vaani.tier46_gate.compare (same nominal slice, same bucket/severe/burst/recovery definitions as the test gate).
"""
import argparse, csv, hashlib, itertools, json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd, torch, yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ is not a package
from vaani import metrics, tier46_gate  # noqa: E402
from vaani.data.dataset import RenderedDataset  # noqa: E402
from vaani.dsp import stft  # noqa: E402
from vaani.dsp.postfilter import DEFAULTS, ResidualPostFilter  # noqa: E402
from vaani.eval import _ckpt_spectrum_fn, enhance_fn  # noqa: E402

GRID = {"postfilter": [dict(gain_floor=g, noise_bias=b) for g, b in itertools.product((0.70, 0.85), (1.0, 1.5))],
        "refiner": None}   # one trained cascade checkpoint, passed with --checkpoint
SCREEN_GATES = ("utility", "intelligibility", "per_bucket", "severe_burst", "recovery")   # paired_uncertainty is reported, not required, on val
COLS = ["system", "id", "bucket", "noise_class", "snr_in", "clipped", "ref_dropout", "impulse_peak_db", "fault", "speech_source",
        "snr_out", "si_sdr", "stoi", "pesq_wb", "recovery_s"]


def setting_name(s):
    if s is None: return "anchor"
    return "refiner" if "cascade" in s else "gf%.2f_nb%.1f" % (s["gain_floor"], s["noise_bias"])


def cache_key(proto, split):
    """Everything the cached spectra depend on. Changing any of it must miss the cache."""
    a = proto["anchor"]; cfg = a["config"] if isinstance(a["config"], dict) else {}
    parts = {"anchor_sha256": a["sha256"], "split": split,
             "split_digest": hashlib.sha256(json.dumps(proto["splits"][split]["files"], sort_keys=True).encode()).hexdigest(),
             "dsp": {"controller_on": cfg.get("controller_on"), "dsp": cfg.get("dsp"), "model": cfg.get("model"), "model_cfg": cfg.get("model_cfg")},
             "stft": {"n_fft": stft.N_FFT, "hop": stft.HOP, "win": stft.WIN}}
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16], parts


def open_cache(cache_dir: Path, proto, split):
    """Create or validate the spectra cache; a directory written for another anchor/split raises."""
    key, parts = cache_key(proto, split)
    meta = cache_dir / "cache_meta.json"
    if meta.exists():
        have = json.loads(meta.read_text())
        if have.get("key") != key:
            raise RuntimeError(f"spectra cache {cache_dir} belongs to another anchor/split (key {have.get('key')} != {key}); refusing to reuse it")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"key": key, **parts}, indent=1, default=str))
    return key


# ---- worker side -------------------------------------------------------------------------------------------------
_ds = _spec = _cache = _settings = _casc = None


def _init(root, ckpt, cache_dir, settings):
    global _ds, _spec, _cache, _settings, _casc
    torch.set_num_threads(1)
    _ds = RenderedDataset(root); _spec, _ = _ckpt_spectrum_fn(ckpt, device="cpu"); _cache = Path(cache_dir); _settings = settings
    cas = [s["cascade"] for s in settings if "cascade" in s]
    _casc = enhance_fn(f"cascade:{cas[0]}", device="cpu") if cas else None   # the refiner runs its own frozen stage + DSP, no cached spectra


def _spectrum(mix, path: Path):
    if path.exists():
        return np.load(path)["y"]
    with torch.no_grad():
        y = torch.view_as_complex(_spec(mix)[0].contiguous()).numpy().astype(np.complex64)
    tmp = path.with_suffix(".tmp.npz"); np.savez(tmp, y=y); tmp.replace(path)   # atomic so a killed run never leaves a torn file
    return y


def _wave(y, setting, length):
    z = y if setting is None else ResidualPostFilter(**setting).process(y)   # fresh state per clip and per setting
    return stft.istft(torch.view_as_real(torch.from_numpy(np.ascontiguousarray(z)))[None], length=length)[0].numpy()


def _work(i):
    it = _ds[i]; meta = it["meta"]; mix, clean = it["mix"].numpy(), it["clean"].numpy()
    stem = f"{meta['bucket']}__{meta['id']}"
    y = _spectrum(mix, _cache / f"{stem}.npz")
    yt = _spectrum(it["twin"].numpy(), _cache / f"{stem}.twin.npz") if "twin" in it and meta.get("impulse_onsets_s") else None
    rows = []
    for s in [None] + list(_settings):
        est = _casc(mix) if s is not None and "cascade" in s else _wave(y, s, mix.shape[1])
        row = dict(system=setting_name(s), id=meta.get("id"), bucket=meta.get("bucket"), noise_class=meta.get("noise_class"),
                   snr_in=meta.get("snr_db"), clipped=meta.get("clipped"), ref_dropout=meta.get("ref_dropout"),
                   impulse_peak_db=meta.get("impulse_peak_db"), fault=meta.get("fault"), speech_source=meta.get("speech_source"),
                   snr_out=metrics.snr_db(clean, est), si_sdr=metrics.si_sdr_db(clean, est),
                   stoi=metrics.stoi(clean, est), pesq_wb=metrics.pesq_wb(clean, est), recovery_s=float("nan"))
        if yt is not None:
            twin = _casc(it["twin"].numpy()) if s is not None and "cascade" in s else _wave(yt, s, it["twin"].shape[1])
            row["recovery_s"] = metrics.recovery_time_s(est, twin, meta["impulse_onsets_s"][0])
        rows.append(row)
    return rows


# ---- selection ---------------------------------------------------------------------------------------------------
def select(results, kind):
    """results: {setting_name: (setting dict, compare() dict)}. Highest nominal dSNR among settings passing the screen
    gates; ties by dPESQ, then higher gain floor, then name. Returns (name, setting) or (None, None)."""
    ok = [(n, s, r) for n, (s, r) in results.items() if all(r["gates"].get(g, False) for g in SCREEN_GATES)]
    if not ok: return None, None
    ok.sort(key=lambda t: (-t[2]["nominal"]["d_snr_out"]["mean"], -t[2]["nominal"]["d_pesq_wb"]["mean"], -t[1].get("gain_floor", 0), t[0]))
    return ok[0][0], ok[0][1]


def screen(kind, protocol_path, eval_root, split, out_dir, workers, grid=None, check_split=True, checkpoint=None):
    proto = json.loads(Path(protocol_path).read_text()); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "refiner":
        if not checkpoint: raise SystemExit("refiner screen needs --checkpoint <cascade best.pt>")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ck["config"].get("first_stage", {}).get("sha256") != proto["anchor"]["sha256"]:
            raise SystemExit(f"{checkpoint} was trained on another first stage than the protocol anchor; refusing to screen")
        grid = [{"cascade": str(checkpoint)}]
    if split == "test": raise SystemExit("selection runs on val only; test is the frozen benchmark")
    root = Path(eval_root) / split
    if check_split:   # the val files must be the ones frozen in anchor.json; tuning on drifted data is not a screen
        from tier46_protocol import _split_manifest
        problems, files, keys = _split_manifest(root)
        rec = proto["splits"][split]
        if problems or files != rec["files"] or keys != [list(k) for k in rec["keys"]]:
            raise SystemExit(f"{split} split differs from the frozen manifest; refusing to screen")
    settings = grid or GRID[kind]
    key = open_cache(out_dir / "spectra", proto, split)
    ckpt = proto["anchor"]["frozen_copy"]
    ds = RenderedDataset(root); names = [setting_name(None)] + [setting_name(s) for s in settings]
    rows = {n: [] for n in names}
    with Pool(workers, initializer=_init, initargs=(root, ckpt, out_dir / "spectra", settings)) as pool:
        for rs in tqdm(pool.imap_unordered(_work, range(len(ds)), chunksize=4), total=len(ds), desc=f"screen {kind}"):
            for r in rs: rows[r["system"]].append(r)
    for n in names:
        with open(out_dir / f"{n}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(sorted(rows[n], key=lambda r: (r["bucket"], r["id"])))
    val_proto = {"keys": proto["splits"][split]["keys"]}   # the gate pairs on this split's keys, not the test keys
    results = {setting_name(s): (s, tier46_gate.compare(out_dir / "anchor.csv", out_dir / f"{setting_name(s)}.csv", val_proto, kind)) for s in settings}
    name, chosen = select(results, kind)
    summary = {"kind": kind, "split": split, "cache_key": key, "anchor_sha256": proto["anchor"]["sha256"], "selected": name,
               "settings": {n: {"setting": s, "gates": r["gates"], "nominal": r["nominal"], "aggregates": r.get("aggregates"), "recovery": r.get("recovery")}
                            for n, (s, r) in results.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if chosen is None:
        print("no setting passed the screen gates on val; the post-filter experiment stops here (nothing is evaluated on test)")
        return None
    prov = {"screen": (out_dir / "summary.json").as_posix(), "anchor_sha256": proto["anchor"]["sha256"], "cache_key": key, "selected": name}
    if kind == "refiner":
        sel = {"cascade_checkpoint": chosen["cascade"], "eval_system": f"cascade:{chosen['cascade']}", "provenance": prov}
    else:
        sel = {"base_checkpoint": ckpt, "postfilter": {**{k: v for k, v in DEFAULTS.items() if k != "epsilon"}, **chosen}, "provenance": prov}
    sel_path = out_dir.parent / f"{kind}.selected.yaml"
    if sel_path.exists(): raise SystemExit(f"{sel_path} already exists; a pre-registered selection is never overwritten")
    sel_path.write_text(yaml.safe_dump(sel, sort_keys=False))
    print(f"selected {name}: {chosen}  ->  {sel_path}")
    return sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=list(GRID))
    ap.add_argument("--protocol", default="results_r2/tier46/anchor.json"); ap.add_argument("--eval-root", default="data/eval_r2")
    ap.add_argument("--split", default="val"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint", help="refiner: the trained cascade best.pt")
    a = ap.parse_args()
    screen(a.kind, a.protocol, a.eval_root, a.split, a.out, a.workers, checkpoint=a.checkpoint)


if __name__ == "__main__":
    main()
