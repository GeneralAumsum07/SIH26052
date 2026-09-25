#!/usr/bin/env python3
"""Score real, non-synthetic recordings: MAD "communication" clips and the two-channel web WAV the Pi ran.

    uv run --with numba python scripts/score_real.py --workers 2          # score (resumes), then summarise
    uv run --with numba python scripts/score_real.py --summarise-only     # rebuild scores.csv / summary.csv / table.md
    uv run --with numba python scripts/score_real.py --name r8 --onnx <cascade.onnx> --config <model_config.json>

Every other number in the repo is scored on `mixer.mix()` output. These clips are recordings, so there is no
clean reference: the metrics are DNSMOS P.835 (SIG/BAK/OVRL) and the level-based attenuation proxy of
`vaani.physical`, both reference-free proxies, not SNR/STOI/PESQ.

(a) MAD label 0 ("communication": radio speech recorded in military noise), which `sources.scan_mad` drops as
    not-noise. One clip per YouTube video (seeded), at most `--max-clips`, cropped to `--crop-s`. The clips are
    single-channel, so the stream system runs under two labelled reference conditions. `ref_zero` is the headline
    row: reference zeroed, at validity 0 where the model takes a validity input (the trained reference-absent mode;
    for r7, which has none, it is a dead-reference fault case). `ref_dup` (reference = the primary, a duplicated
    mono feed) is a labelled stress row, never the headline.
(b) C:/Users/Rachit/Downloads/abcd.wav, stereo 44.1 kHz, L = primary, R = reference, as run on the Pi, resampled
    to 16 kHz (polyphase 160/441). Scored as `stereo_LR` plus the same two single-channel conditions.

Baselines on the same clips: `raw` (the primary, passthrough) and `gtcrn_pretrained` (mono, DNS3 weights). Both
ignore the reference, so their condition is `mono`. The stream system (`--name`, default r7 = deploy/r7) runs through
`vaani.live.StreamEngine` (the board path). Per-clip rows are appended to `results_r2/real/work/rows.jsonl` (r7) or
`work/rows_<name>.jsonl` as they finish, so a killed run resumes; the summary pools every cache.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vaani import physical  # noqa: E402

MAD_ROOT = REPO / "data/download/mad/MAD_dataset"
WEB_WAV = Path("C:/Users/Rachit/Downloads/abcd.wav")
OUT = REPO / "results_r2/real"
SR = 16000
METRICS = ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "atten20_frac", "mean_atten_db"]
ROW_KEYS = ["source", "clip", "video", "seconds", "system", "condition", *METRICS,
            "gate_mean", "burst_frac", "limiter_frac"]
# which row a condition is in the tables (Rachit, 2026-09-25): ref_zero heads single-channel audio, ref_dup is stress
ROW = {"mono": "baseline", "ref_zero": "single-channel headline", "stereo_LR": "as recorded", "ref_dup": "stress"}
COND_ORDER = ["ref_zero", "stereo_LR", "ref_dup"]
ENGINE = ("r7", str(physical.ONNX), str(physical.CONFIG))     # (name, onnx, model_config) of the stream system


def _video_id(url: str, fallback: str) -> str:
    v = parse_qs(urlparse(url).query).get("v")
    return v[0] if v else fallback


def scan_mad_communication(root: Path = MAD_ROOT) -> list[dict]:
    """Every MAD label-0 row, with the YouTube video id it was cut from (folders are per split, not per video)."""
    rows = []
    for name in ("training.csv", "test.csv"):
        with open(root / name, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["label"] != "0":
                    continue
                p = Path(r["path"])
                rows.append({"clip": f"mad:{p.parent.parent.name}/{p.parent.name}_{p.stem}", "path": root / p,
                             "video": _video_id(r["youtube url"], f"{p.parent.parent.name}/{p.parent.name}")})
    return rows


def select_subset(rows: list[dict], max_clips: int, seed: int) -> list[dict]:
    """One clip per video (independent rows for the bootstrap), then at most max_clips videos; deterministic."""
    rng = np.random.default_rng(seed)
    by_vid = {}
    for r in sorted(rows, key=lambda r: r["clip"]):
        by_vid.setdefault(r["video"], []).append(r)
    picked = [v[int(rng.integers(len(v)))] for _, v in sorted(by_vid.items())]
    if len(picked) > max_clips:
        picked = [picked[i] for i in sorted(rng.choice(len(picked), max_clips, replace=False))]
    return picked


def load_mono(path: Path, crop_s: float) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = physical.to_16k(x.T, sr)[0]
    return x[:int(crop_s * SR)]


_GTCRN = None


def _init_worker():
    import torch
    torch.set_num_threads(1)      # two workers on a shared box: one core each


def _gtcrn(prim):
    global _GTCRN
    if _GTCRN is None:
        from vaani.models import baselines
        _GTCRN = baselines.get("gtcrn_pretrained")
    return _GTCRN.enhance(prim[None].astype(np.float32)).astype(np.float32)


def score_item(item: dict, engine=ENGINE) -> list[dict]:
    """All system x condition rows for one clip. item: clip, video, source, and prim (+ ref for stereo)."""
    name, onnx, config = engine
    prim = np.asarray(item["prim"], np.float32)
    runs = [("raw", "mono", prim, None), ("gtcrn_pretrained", "mono", _gtcrn(prim), None)]
    conds = [("ref_zero", np.zeros_like(prim)), ("ref_dup", prim.copy())]
    if item.get("ref") is not None:
        conds.insert(0, ("stereo_LR", np.asarray(item["ref"], np.float32)))
    for cond, ref in conds:
        # ref_zero runs at validity 0 on a validity-input model; run_engine leaves r7's call unchanged
        y, diag = physical.run_engine(np.stack([prim, ref]), onnx, config, ref_valid=cond != "ref_zero")
        runs.append((name, cond, y, diag))
    nan = float("nan")
    out = []
    for system, cond, y, diag in runs:
        d = physical.dnsmos(y)
        out.append({"source": item["source"], "clip": item["clip"], "video": item["video"],
                    "seconds": len(prim) / SR, "system": system, "condition": cond,
                    "dnsmos_sig": d["sig"], "dnsmos_bak": d["bak"], "dnsmos_ovrl": d["ovrl"],
                    **{k: v for k, v in physical.attenuation_proxy(prim, y).items() if k != "active_frames"},
                    **(diag or {"gate_mean": nan, "burst_frac": nan, "limiter_frac": nan})})
    return out


def _mad_task(args):
    r, crop_s, engine = args
    return score_item({"source": "mad_communication", "clip": r["clip"], "video": r["video"],
                       "prim": load_mono(r["path"], crop_s)}, engine)


def build_items(a) -> list:
    tasks = []
    if WEB_WAV.exists() and not a.no_web:
        tasks.append(("web", None))
    for r in select_subset(scan_mad_communication(), a.max_clips, a.seed):
        tasks.append(("mad", r))
    return tasks


def _run(task, crop_s, engine=ENGINE):
    kind, r = task
    if kind == "web":
        import soundfile as sf
        x, sr = sf.read(WEB_WAV, dtype="float32", always_2d=True)
        x16 = physical.to_16k(x.T, sr)                      # 44.1k -> 16k, 160/441 (the earlier repro's resampler)
        return score_item({"source": "web_abcd", "clip": "web:abcd", "video": "abcd", "prim": x16[0], "ref": x16[1]},
                          engine)
    return _mad_task((r, crop_s, engine))


def _run_star(args):
    return _run(*args)


def score(a):
    from vaani.dsp import nlms
    if not nlms._HAVE_NUMBA and not a.allow_pure_python:
        sys.exit("numba not importable: run under 'uv run --with numba' (pure-Python NLMS is ~26 ms/hop)")
    work = OUT / "work"; work.mkdir(parents=True, exist_ok=True)
    engine = (a.name, a.onnx, a.config)
    cache = _cache(work, a.name)
    done = set()
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)[0]["clip"])
    tasks = build_items(a)
    (work / "subset.json").write_text(json.dumps(
        [{"clip": "web:abcd", "path": str(WEB_WAV)} if k == "web" else {"clip": r["clip"], "path": str(r["path"]),
                                                                          "video": r["video"]} for k, r in tasks],
        indent=1), encoding="utf-8")
    todo = [t for t in tasks if ("web:abcd" if t[0] == "web" else t[1]["clip"]) not in done]
    print(f"{len(tasks)} clips, {len(done)} cached, {len(todo)} to score, {a.workers} worker(s)", flush=True)
    with open(cache, "a", encoding="utf-8") as fh:
        def put(rows):
            fh.write(json.dumps(rows) + "\n"); fh.flush()
            print(f"  {rows[0]['clip']}: " + ", ".join(f"{r['system']}/{r['condition']} {r['dnsmos_ovrl']:.2f}"
                                                        for r in rows), flush=True)
        if a.workers <= 1:
            _init_worker()
            for t in todo:
                put(_run(t, a.crop_s, engine))
        else:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker) as pool:
                for rows in pool.imap_unordered(_run_star, [(t, a.crop_s, engine) for t in todo]):
                    put(rows)


def _cache(work: Path, name: str) -> Path:
    return work / ("rows.jsonl" if name == "r7" else f"rows_{name}.jsonl")


def load_rows(work: Path) -> list[dict]:
    """Every cache's rows; raw/gtcrn rows repeat across caches and are kept once (the first, r7's cache first)."""
    caches = sorted(work.glob("rows*.jsonl"), key=lambda p: (p.name != "rows.jsonl", p.name))
    seen, rows = set(), []
    for c in caches:
        for line in c.read_text(encoding="utf-8").splitlines():
            for r in (json.loads(line) if line.strip() else []):
                k = (r["source"], r["clip"], r["system"], r["condition"])
                if k not in seen:
                    seen.add(k); rows.append(r)
    rows.sort(key=lambda r: (r["source"], r["clip"], r["system"], r["condition"]))
    return rows


def row_order(rows) -> list[tuple[str, str]]:
    """Baselines, then per stream system (r7 first) the headline ref_zero, the as-recorded stereo, the ref_dup stress."""
    order = [("raw", "mono"), ("gtcrn_pretrained", "mono")]
    streams = sorted({r["system"] for r in rows if r["condition"] != "mono"}, key=lambda s: (s != "r7", s))
    return order + [(s, c) for s in streams for c in COND_ORDER]


def summarise():
    from vaani.report import ci
    rows = load_rows(OUT / "work")
    with open(OUT / "scores.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_KEYS); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in ROW_KEYS})
    raw = {(r["source"], r["clip"]): r for r in rows if r["system"] == "raw"}
    order = row_order(rows)
    summ = []
    for src in sorted({r["source"] for r in rows}):
        for system, cond in order:
            g = [r for r in rows if r["source"] == src and r["system"] == system and r["condition"] == cond]
            if not g:
                continue
            s = {"source": src, "system": system, "condition": cond, "row": ROW[cond], "n_clips": len(g)}
            for m in METRICS:
                s[m], s[m + "_lo"], s[m + "_hi"] = ci([r[m] for r in g])
            s["d_ovrl_vs_raw"], s["d_ovrl_vs_raw_lo"], s["d_ovrl_vs_raw_hi"] = ci(
                [r["dnsmos_ovrl"] - raw[(src, r["clip"])]["dnsmos_ovrl"] for r in g])
            summ.append(s)
    keys = list(summ[0])
    with open(OUT / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for s in summ:
            w.writerow({k: (f"{s[k]:.4f}" if isinstance(s[k], float) else s[k]) for k in keys})
    write_table(summ)
    return summ


def _fmt(s, m, p=2):
    v, lo, hi = s[m], s[m + "_lo"], s[m + "_hi"]
    if not np.isfinite(v):
        return "n/a"
    if s["n_clips"] < 2:
        return f"{v:.{p}f}"
    return f"{v:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def write_table(summ):
    L = ["# Real recordings: reference-free proxies (generated by scripts/score_real.py)", "",
         "DNSMOS P.835 and the attenuation proxy are reference-free proxies, **not** SNR/STOI/PESQ: these clips have "
         "no clean reference. Mean [95% bootstrap CI over clips, 1000 resamples, vaani.report.ci]; one clip gives a "
         "point value. `atten>20dB` = fraction of active 20 ms input frames (energy within 30 dB of the clip's p99; "
         "not a VAD) that the output cut by more than 20 dB. `dOVRL` = paired per-clip OVRL minus raw.", "",
         "Rows: `single-channel headline` = `ref_zero`, the reference zeroed (validity 0 where the model takes a "
         "validity input; r7 has none, so for r7 it is a dead-reference fault case). `stress` = `ref_dup`, reference "
         "= primary (a duplicated mono feed): a labelled stress row, never the headline. `as recorded` = "
         "`stereo_LR`, the recording's own R channel as reference (two-channel sources only; the only row at the "
         "two-mic design point). `baseline` = `mono`, the system ignores the reference.", ""]
    for src in dict.fromkeys(s["source"] for s in summ):
        L += [f"## {src}", "", "| row | system | condition | n | SIG | BAK | OVRL | dOVRL vs raw | atten>20dB |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in (x for x in summ if x["source"] == src):
            lab = f"**{s['row']}**" if s["row"].endswith("headline") else s["row"]
            L.append(f"| {lab} | {s['system']} | {s['condition']} | {s['n_clips']} | {_fmt(s, 'dnsmos_sig')} | "
                     f"{_fmt(s, 'dnsmos_bak')} | {_fmt(s, 'dnsmos_ovrl')} | {_fmt(s, 'd_ovrl_vs_raw')} | "
                     f"{_fmt(s, 'atten20_frac')} |")
        L.append("")
    (OUT / "table.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--crop-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-web", action="store_true", help="skip the abcd.wav rows")
    ap.add_argument("--allow-pure-python", action="store_true", help="run without numba (slow; smoke tests only)")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="output folder (default results_r2/real)")
    ap.add_argument("--name", default=ENGINE[0], help="stream system label (default r7; its rows cache is rows.jsonl)")
    ap.add_argument("--onnx", default=ENGINE[1], help="StreamEngine ONNX (default deploy/r7/cascade.onnx)")
    ap.add_argument("--config", default=ENGINE[2], help="its model_config.json (default deploy/r7/model_config.json)")
    a = ap.parse_args(argv)
    OUT = Path(a.out)
    if not a.summarise_only:
        score(a)
    summarise()


if __name__ == "__main__":
    main()
