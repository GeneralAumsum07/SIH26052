"""Reference-condition scoring on VAL only (spec 6.2/8, plan G3): the same val clips under deterministic reference
conditions, one CSV row per (clip, condition), plus a per-condition Markdown table.

Conditions (the primary channel is never touched):
  present        as rendered
  absent         reference zeroed; capture-path availability 0 (reaches a ref_validity model and a DSP ref_policy)
  burst_dropout  two known dropouts (25-37.5 % and 62.5-72.5 % of the clip), availability 0 inside them
  gain_m12       reference -12 dB
  lowpass        obstruction over the reference mic: one-pole 800 Hz low-pass, -12 dB
  delay          reference 24 samples (1.5 ms) late
  clipped        reference clipped at 10 % of its peak
  talker_leak    the talker added to the reference at -2 dB re primary (the web-WAV failure)
  ild_<d>        far-field reference: the primary's own noise plus the talker at <d> dB re primary, d in -14..0
Only absent and burst_dropout carry availability 0: every other fault is a present-but-wrong reference.

Systems (model-agnostic runner, so r8 candidates plug in unchanged):
  ckpt:<best.pt>             any vaani.train checkpoint (backbone or cascade; build_model decides the class). The
                             model gets ref_avail whenever its forward accepts it; the DSP gets it through ref_policy.
  onnx:<graph.onnx>@<ckpt>   an exported streaming graph, DSP from the checkpoint's config; a ref_avail input gets availability
  py:<module>:<factory>      factory() -> object with enhance(mix (2,n) float32, ref_avail (n,) bool | None) -> (n,)
  <baseline name>            vaani.models.baselines

Metrics: SNR_out, STOI, PESQ-WB, speech loss (share of speech-active 20 ms frames whose projected speech gain is below
-15 dB; same definition as the diag_webaudio frame_stats), longest lost run, recovery after each reconnect
(metrics.recovery_time_s against the present-condition output) and a click proxy at the dropout edges.
A clip fails a condition when its speech loss > 0.15 (the G-gate p95 clip bound) or SNR_out < SNR_in - 1 dB.

usage: python scripts/eval_refvalid.py --system ckpt:results_r2/runs/r7_e256_wr64_refiner/best.pt \
           --out results_r2/r8/r7_refconditions_val --per-bucket 4 --workers 2
"""
import argparse, csv, importlib, inspect, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR, F = 16000, 320   # 20 ms analysis frames for speech loss
ILDS = (-14, -10, -8, -6, -4, -2, 0)
CONDITIONS = ("present", "absent", "burst_dropout", "gain_m12", "lowpass", "delay", "clipped", "talker_leak",
              *(f"ild_{d}" for d in ILDS))
LOSS_FAIL, SNR_FAIL_DB = 0.15, 1.0
SPANS = ((0.25, 0.375), (0.625, 0.725))   # burst_dropout, as clip fractions


def _shift(x, d):
    y = np.zeros_like(x); y[d:] = x[:len(x) - d]
    return y


def apply_condition(cond, mix, clean):
    """(2,n) mix, (n,) clean -> (mix', ref_avail (n,) bool or None, transition sample indices)."""
    p, r = mix[0].copy(), mix[1].copy(); n = len(p); avail = None; edges = []
    if cond == "present":
        pass
    elif cond == "absent":
        r[:] = 0; avail = np.zeros(n, bool)
    elif cond == "burst_dropout":
        avail = np.ones(n, bool)
        for a, b in SPANS:
            a, b = int(a * n), int(b * n); r[a:b] = 0; avail[a:b] = False; edges += [a, b]
    elif cond == "gain_m12":
        r *= 10 ** (-12 / 20)
    elif cond == "lowpass":
        k = float(np.exp(-2 * np.pi * 800 / SR)); r = lfilter([1 - k], [1, -k], r).astype(np.float32) * 10 ** (-12 / 20)
    elif cond == "delay":
        r = _shift(r, 24)
    elif cond == "clipped":
        lvl = 0.1 * float(np.abs(r).max()); r = np.clip(r, -lvl, lvl)
    elif cond == "talker_leak":
        r = r + clean * 10 ** (-2 / 20)
    elif cond.startswith("ild_"):
        r = (p - clean) + clean * 10 ** (int(cond[4:]) / 20)   # far-field noise identical at both mics
    else:
        raise ValueError(cond)
    return np.stack([p, np.clip(r, -1, 1)]).astype(np.float32), avail, edges


def frame_stats(clean, y, prim):
    """diag_webaudio speech-loss definition (projection gain per 20 ms speech-active frame)."""
    k = len(clean) // F
    C, Y, P = (v[: k * F].reshape(k, F).astype(np.float64) for v in (clean, y, prim))
    ec = (C ** 2).mean(1) + 1e-12; en = ((P - C) ** 2).mean(1) + 1e-12; ey = (Y ** 2).mean(1) + 1e-12
    act = 10 * np.log10(ec) > 10 * np.log10(ec.max()) - 30
    a = (Y * C).sum(1) / ((C * C).sum(1) + 1e-12)
    g_db = 20 * np.log10(np.maximum(a, 1e-4))
    lost = act & (g_db < -15)
    dom = act & (10 * np.log10(ec / en) > 6)
    lost_dom = dom & (10 * np.log10(ey / ec) < -15)
    run = best = 0
    for v in lost:
        run = run + 1 if v else 0; best = max(best, run)
    return {"speech_loss": float(lost.sum() / max(1, act.sum())), "speech_loss_dom": float(lost_dom.sum() / max(1, dom.sum())),
            "gain_med_db": float(np.median(g_db[act])) if act.any() else float("nan"), "longest_lost_s": best * F / SR}


def click_db(y, edges, win=160):
    """Largest sample step within +-10 ms of a dropout edge over the clip's 99th-percentile step (dB); NaN without edges."""
    if not edges:
        return float("nan")
    d = np.abs(np.diff(y)); ref = np.percentile(d, 99) + 1e-9
    m = max(float(d[max(0, e - win):e + win].max()) for e in edges)
    return float(20 * np.log10(m / ref + 1e-12))


# --- systems ---

def make_system(spec):
    """-> enhance(mix, ref_avail) -> (n,) float32 on CPU."""
    import torch
    from vaani.dsp import pipeline, stft
    if spec.startswith("py:"):
        mod, fn = spec[3:].rsplit(":", 1); obj = getattr(importlib.import_module(mod), fn)()
        return obj.enhance
    if spec.startswith("onnx:"):
        return _onnx_system(spec)
    if not spec.startswith("ckpt:"):
        from vaani.models import baselines
        b = baselines.get(spec)
        return lambda mix, avail: b.enhance(mix)
    from vaani.models import cascade
    from vaani.train import build_model
    ck = torch.load(spec[5:], map_location="cpu", weights_only=True); cfg = ck["config"]
    m = cascade.FrozenCascade.from_config(cfg) if cfg["model"] == cascade.MODEL_NAME else build_model(cfg["model"], model_cfg=cfg.get("model_cfg"))
    m.load_state_dict(ck["model"]); m.eval()
    takes_avail = "ref_avail" in inspect.signature(m.forward).parameters
    has_policy = (cfg.get("dsp") or {}).get("ref_policy") is not None

    @torch.no_grad()
    def enhance(mix, avail):
        x = torch.from_numpy(mix)[None]
        if cfg["model"] == "gtcrn":
            return stft.istft(m(stft.stft(x[:, 0])), length=mix.shape[1])[0].numpy()
        r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"), ref_avail=avail if has_policy else None)
        x = torch.from_numpy(r["mix"])[None]
        spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1)
        feats = torch.from_numpy(r["features"])[None]
        if takes_avail and avail is not None:
            fa = r["ref_avail"] if "ref_avail" in r else pipeline.frame_avail(avail, r["features"].shape[0])
            out = m(spec6, feats, torch.from_numpy(fa)[None])
        else:
            out = m(spec6, feats)
        return stft.istft(out, length=mix.shape[1])[0].numpy()
    return enhance


def _onnx_system(spec):
    """`onnx:<graph>@<ckpt>`: the streamed graph; a ref_validity graph (export_refvalid) gets the per-frame availability."""
    import torch
    from vaani import export
    from vaani.dsp import pipeline, stft
    graph, ckpt = spec[5:].rsplit("@", 1)
    cfg = torch.load(ckpt, map_location="cpu", weights_only=True)["config"]
    sess = export.load_session(graph)
    names, zero = export.zero_caches(sess)
    takes_avail = any(i.name == export.REF_AVAIL for i in sess.get_inputs())
    has_policy = (cfg.get("dsp") or {}).get("ref_policy") is not None

    def enhance(mix, avail):
        r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"), ref_avail=avail if has_policy else None)
        x = torch.from_numpy(r["mix"])[None]
        spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1).numpy()
        feats = np.ascontiguousarray(r["features"][None], dtype=np.float32)
        fa = None
        if takes_avail and avail is not None:
            fa = r["ref_avail"] if "ref_avail" in r else pipeline.frame_avail(avail, feats.shape[1])
        out, _ = export.stream_onnx(sess, spec6, feats, names, [c.copy() for c in zero], ref_avail=fa)
        return stft.istft(torch.from_numpy(out), length=mix.shape[1])[0].numpy()
    return enhance


# --- workers ---

_ds = _fn = _sysname = None


def _init(system, root):
    global _ds, _fn, _sysname
    import torch
    torch.set_num_threads(1)
    from vaani.data.dataset import RenderedDataset
    _ds, _fn, _sysname = RenderedDataset(root), make_system(system), system


def _work(i):
    from vaani import metrics
    it = _ds[i]; meta = it["meta"]; mix, clean = it["mix"].numpy(), it["clean"].numpy()
    rows, y_present = [], None
    for cond in CONDITIONS:
        base = dict(system=_sysname, id=meta["id"], bucket=meta["bucket"], noise_class=meta.get("noise_class"),
                    snr_in=meta.get("snr_db"), condition=cond)
        try:
            m2, avail, edges = apply_condition(cond, mix, clean)
            y = np.asarray(_fn(m2, avail), np.float32)[: mix.shape[1]]
            if cond == "present":
                y_present = y
            clean_item = bool(meta.get("clean_bucket"))
            row = dict(base, snr_in_meas=metrics.snr_db(clean, m2[0]) if not clean_item else float("nan"),
                       snr_out=metrics.snr_db(clean, y), stoi=metrics.stoi(clean, y), pesq_wb=metrics.pesq_wb(clean, y),
                       **frame_stats(clean, y, m2[0]), click_db=click_db(y, edges))
            rec = [metrics.recovery_time_s(y, y_present, b / SR) for b in edges[1::2]] if edges and y_present is not None else []
            row["recovery_s"] = float(max(rec)) if rec else float("nan")
            row["fail"] = int(row["speech_loss"] > LOSS_FAIL or (not clean_item and row["snr_out"] < row["snr_in_meas"] - SNR_FAIL_DB))
        except Exception as e:   # one bad clip never aborts the run
            print(f"{meta['id']} {cond} failed: {e!r}", flush=True)
            row = dict(base, fail=-1)
        rows.append(row)
    return rows


COLS = ["system", "id", "bucket", "noise_class", "snr_in", "condition", "snr_in_meas", "snr_out", "stoi", "pesq_wb",
        "speech_loss", "speech_loss_dom", "gain_med_db", "longest_lost_s", "recovery_s", "click_db", "fail"]


def select(root, per_bucket):
    from vaani.data.dataset import RenderedDataset
    items = RenderedDataset(root).items; seen, idx = {}, []
    for i, p in enumerate(items):
        b = p.parent.name
        if seen.get(b, 0) < per_bucket:
            seen[b] = seen.get(b, 0) + 1; idx.append(i)
    return idx


def summarise(csv_path, md_path, cmd):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype={"id": str}); df["id"] = df.bucket + "/" + df.id; ok = df[df.fail >= 0]
    g = ok.groupby("condition", sort=False)
    t = pd.DataFrame({"n": g.size(), "snr_out": g.snr_out.mean(), "stoi": g.stoi.mean(), "pesq_wb": g.pesq_wb.mean(),
                      "speech_loss": g.speech_loss.mean(), "speech_loss_p95": g.speech_loss.quantile(0.95),
                      "longest_lost_s": g.longest_lost_s.mean(), "recovery_s_max": g.recovery_s.max(),
                      "click_db_max": g.click_db.max(), "fails": g.fail.sum()}).reindex([c for c in CONDITIONS if c in g.groups])
    pres = ok[ok.condition == "present"].set_index("id")
    d = []
    for c in t.index:
        x = ok[ok.condition == c].set_index("id"); j = x.index.intersection(pres.index)
        d.append(float((x.loc[j, "snr_out"] - pres.loc[j, "snr_out"]).mean()))
    t.insert(2, "d_snr_vs_present", d)
    errors = int((df.fail < 0).sum())
    lines = [f"# Reference conditions on VAL: {df.system.iloc[0]}", "",
             f"Source: `{Path(csv_path).name}` ({len(df)} rows, {ok.id.nunique()} val clips, {errors} errored rows).",
             f"Command: `{cmd}`", "",
             "Fail = speech loss > 0.15 or SNR_out < SNR_in - 1 dB (per clip). recovery_s: burst_dropout only, worst of the",
             "two reconnects, against the same clip's present-condition output. click_db: largest sample step at a dropout",
             "edge over the clip's p99 step. VAL only; no test-set number here.", "",
             t.to_markdown(floatfmt=".3f"), ""]
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True); ap.add_argument("--out", required=True, help="path stem; writes .csv and .md")
    ap.add_argument("--eval-root", default="data/eval_r2"); ap.add_argument("--split", default="val")
    ap.add_argument("--per-bucket", type=int, default=4); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N selected clips (smoke)")
    a = ap.parse_args(argv)
    if a.split != "val":
        raise SystemExit("eval_refvalid scores VAL only: selection decisions are never made on test")
    root = Path(a.eval_root) / a.split; idx = select(root, a.per_bucket)[: a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); csv_path = out.with_suffix(".csv")
    done = set()
    if csv_path.exists():   # resumable: clips already fully written are skipped
        import pandas as pd
        prev = pd.read_csv(csv_path, dtype={"id": str}); cnt = prev.groupby(["bucket", "id"]).size()
        done = {f"{b}/{i}" for (b, i), c in cnt.items() if c == len(CONDITIONS)}
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    from vaani.data.dataset import RenderedDataset
    items = RenderedDataset(root).items
    todo = [i for i in idx if f"{items[i].parent.name}/{items[i].name[:-len('.mix.wav')]}" not in done]
    new = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        if a.workers:
            with Pool(a.workers, initializer=_init, initargs=(a.system, root)) as pool:
                for k, rows in enumerate(pool.imap(_work, todo)):
                    w.writerows(rows); fh.flush(); print(f"{k + 1}/{len(todo)}", flush=True)
        else:
            _init(a.system, root)
            for k, i in enumerate(todo):
                w.writerows(_work(i)); fh.flush(); print(f"{k + 1}/{len(todo)}", flush=True)
    cmd = "python scripts/eval_refvalid.py " + " ".join(argv if argv is not None else sys.argv[1:])
    print(summarise(csv_path, out.with_suffix(".md"), cmd).to_string())


if __name__ == "__main__":
    main()
