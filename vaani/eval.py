"""Per-clip metrics over a rendered split. One CSV row per clip."""
import argparse, csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import numpy as np, torch
from tqdm import tqdm

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline, stft
from vaani.models import baselines
from vaani.train import build_model


def enhance_fn(spec: str, device=None):
    if not spec.startswith("ckpt:"):
        return baselines.get(spec).enhance
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(spec[5:], map_location="cpu", weights_only=True); cfg = ck["config"]
    m = build_model(cfg["model"]).to(device); m.load_state_dict(ck["model"]); m.eval()

    @torch.no_grad()
    def f(mix):
        x = torch.from_numpy(mix)[None].to(device)
        if cfg["model"] == "gtcrn":
            out = m(stft.stft(x[:, 0]))
        else:
            # DSP pipeline must run exactly as training did, hence the checkpoint's own controller_on
            r = pipeline.run(mix, controller_on=cfg["controller_on"])
            spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]),
                                stft.stft(torch.from_numpy(r["n_hat"])[None].to(device))], -1)
            out = m(spec6, torch.from_numpy(r["features"])[None].to(device))
        return stft.istft(out, length=mix.shape[1])[0].cpu().numpy()
    return f


_ds = _fn = _sys = None


def _init(system, root, split):
    # per-process state: the tiny model on CPU (no 8x CUDA contexts) and one torch thread so 8 workers don't oversubscribe
    global _ds, _fn, _sys
    torch.set_num_threads(1)
    _ds, _fn, _sys = RenderedDataset(root / split), enhance_fn(system, device="cpu"), system


def _nan_row(meta):
    return dict(system=_sys, id=meta.get("id"), bucket=meta.get("bucket"), noise_class=meta.get("noise_class"),
                snr_in=meta.get("snr_db"), clipped=meta.get("clipped"), ref_dropout=meta.get("ref_dropout"),
                impulse_peak_db=meta.get("impulse_peak_db"), snr_out=float("nan"), si_sdr=float("nan"),
                stoi=float("nan"), pesq_wb=float("nan"), recovery_s=float("nan"), asr_text="")


def _work(i):
    """DSP + enhance + objective metrics for one clip; the enhanced signal rides back for ASR in the parent."""
    it = _ds[i]; meta = it["meta"]
    # one bad clip must never abort the whole run: log and fall through to a NaN row
    try:
        mix, clean = it["mix"].numpy(), it["clean"].numpy()
        est = _fn(mix)
        row = _nan_row(meta)
        row.update(snr_out=metrics.snr_db(clean, est), si_sdr=metrics.si_sdr_db(clean, est),
                   stoi=metrics.stoi(clean, est), pesq_wb=metrics.pesq_wb(clean, est))
        if "twin" in it and meta["impulse_onsets_s"]:
            row["recovery_s"] = metrics.recovery_time_s(est, _fn(it["twin"].numpy()), meta["impulse_onsets_s"][0])
        return row, est
    except Exception as e:
        print(f"clip {meta.get('id')} failed: {e!r}")
        return _nan_row(meta), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--eval-root", default="data/eval"); ap.add_argument("--out", required=True)
    ap.add_argument("--asr", action="store_true", help="also compute WER with faster-whisper (supporting evidence only)")
    ap.add_argument("--asr-device", default="cpu", help="cpu|cuda; ASR is not in the deployed path, so cuda only speeds up eval")
    ap.add_argument("--workers", type=int, default=8, help="CPU processes for DSP+metrics (0 = in-process, for debugging); Whisper stays in this process")
    ap.add_argument("--asr-threads", type=int, default=4, help="concurrent Whisper decodes (GPU batches them)")
    a = ap.parse_args()
    root = Path(a.eval_root); n = len(RenderedDataset(root / a.split))
    asr = None
    if a.asr:
        try:
            from vaani.asr import load_whisper; asr = load_whisper(a.asr_device, threads=a.asr_threads)
        except ImportError:
            print("faster-whisper not installed; --asr rows will be NaN")  # never a hard dependency
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "id", "bucket", "noise_class", "snr_in", "clipped", "ref_dropout", "impulse_peak_db",
            "snr_out", "si_sdr", "stoi", "pesq_wb", "recovery_s", "asr_text"]
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        # imap keeps CSV order deterministic; results stream back so ASR overlaps the workers' DSP/PESQ
        if a.workers == 0: _init(a.system, root, a.split)
        pool_cm = Pool(a.workers, initializer=_init, initargs=(a.system, root, a.split)) if a.workers else nullcontext()
        with pool_cm as pool, ThreadPoolExecutor(a.asr_threads) as tp:
            stream = pool.imap(_work, range(n)) if pool else map(_work, range(n))
            def transcribe(item):
                row, est = item
                if asr is not None and est is not None:
                    try:
                        segs, _ = asr.transcribe(est, language=None, beam_size=1)
                        row["asr_text"] = " ".join(s.text for s in segs).strip()  # generator: decode happens here, in the thread
                    except Exception as e:
                        print(f"asr {row['id']} failed: {e!r}")
                return row
            for row in tqdm(tp.map(transcribe, stream), total=n, desc=a.system):
                w.writerow(row)

if __name__ == "__main__":
    main()
