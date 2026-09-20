"""Per-clip metrics over a rendered split. One CSV row per clip."""
import argparse, csv
from collections import deque
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
    """`ckpt:<path>` runs a checkpoint; `post:<yaml>` runs its `base_checkpoint` then `vaani.dsp.postfilter` on the output
    spectrum before the one iSTFT (Tier 4.6); anything else is a baseline name."""
    if spec.startswith("post:"):
        import yaml
        from vaani.dsp.postfilter import ResidualPostFilter
        pcfg = yaml.safe_load(open(spec[5:], encoding="utf-8"))
        spec_fn, _ = _ckpt_spectrum_fn(pcfg["base_checkpoint"], device)
        pf_kwargs = pcfg.get("postfilter") or {}

        def f(mix):
            out = spec_fn(mix)                                    # (1,F,T,2) torch on device
            y = torch.view_as_complex(out[0].contiguous()).cpu().numpy().astype(np.complex64)   # (F,T)
            z = ResidualPostFilter(**pf_kwargs).process(y)        # fresh state per clip (twins included)
            zt = torch.view_as_real(torch.from_numpy(z))[None].to(out.device)
            return stft.istft(zt, length=mix.shape[1])[0].cpu().numpy()
        return f
    if not spec.startswith("ckpt:"):
        return baselines.get(spec).enhance
    spec_fn, _ = _ckpt_spectrum_fn(spec[5:], device)

    def f(mix):
        return stft.istft(spec_fn(mix), length=mix.shape[1])[0].cpu().numpy()
    return f


def _ckpt_spectrum_fn(path, device=None):
    """The checkpoint's full output spectrum (mask + deep-filter taps), still on `device`; iSTFT is the caller's."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=True); cfg = ck["config"]
    m = build_model(cfg["model"], model_cfg=cfg.get("model_cfg")).to(device); m.load_state_dict(ck["model"]); m.eval()

    @torch.no_grad()
    def spec_of(mix):
        x = torch.from_numpy(mix)[None].to(device)
        if cfg["model"] == "gtcrn":
            return m(stft.stft(x[:, 0]))
        # DSP pipeline must run exactly as training did, hence the checkpoint's own controller_on
        r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"))
        x = torch.from_numpy(r["mix"])[None].to(device)   # limited when the checkpoint trained with the limiter
        spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]),
                            stft.stft(torch.from_numpy(r["n_hat"])[None].to(device))], -1)
        return m(spec6, torch.from_numpy(r["features"])[None].to(device))
    return spec_of, cfg


_ds = _fn = _sys = None


def _init(system, root, split, dnsmos=False):
    # per-process state: the tiny model on CPU (no 8x CUDA contexts) and one torch thread so 8 workers don't oversubscribe
    global _ds, _fn, _sys, _mos
    torch.set_num_threads(1)
    _ds, _fn, _sys = RenderedDataset(root / split), enhance_fn(system, device="cpu"), system
    _mos = None
    if dnsmos:
        from vaani.dnsmos import DNSMOS; _mos = DNSMOS()


def _nan_row(meta):
    return dict(system=_sys, id=meta.get("id"), bucket=meta.get("bucket"), noise_class=meta.get("noise_class"),
                snr_in=meta.get("snr_db"), clipped=meta.get("clipped"), ref_dropout=meta.get("ref_dropout"),
                impulse_peak_db=meta.get("impulse_peak_db"), fault=meta.get("fault"), speech_source=meta.get("speech_source"), snr_out=float("nan"), si_sdr=float("nan"),
                stoi=float("nan"), pesq_wb=float("nan"), dnsmos_sig=float("nan"), dnsmos_bak=float("nan"), dnsmos_ovrl=float("nan"),
                recovery_s=float("nan"), asr_text="")


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
        if _mos is not None:   # non-intrusive P.835 on the enhanced output only
            m = _mos(est); row.update(dnsmos_sig=m["sig"], dnsmos_bak=m["bak"], dnsmos_ovrl=m["ovrl"])
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
    ap.add_argument("--dnsmos", action="store_true", help="also score DNSMOS P.835 SIG/BAK/OVRL (deploy/dnsmos/sig_bak_ovr.onnx)")
    a = ap.parse_args()
    root = Path(a.eval_root); n = len(RenderedDataset(root / a.split))
    asr = None
    if a.asr:
        try:
            from vaani.asr import load_whisper, transcribe as whisper_text; asr = load_whisper(a.asr_device, threads=a.asr_threads)
        except ImportError:
            print("faster-whisper not installed; --asr rows will be NaN")  # never a hard dependency
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "id", "bucket", "noise_class", "snr_in", "clipped", "ref_dropout", "impulse_peak_db", "fault", "speech_source",
            "snr_out", "si_sdr", "stoi", "pesq_wb", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "recovery_s", "asr_text"]
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        # imap keeps CSV order deterministic; results stream back so ASR overlaps the workers' DSP/PESQ
        if a.workers == 0: _init(a.system, root, a.split, a.dnsmos)
        pool_cm = Pool(a.workers, initializer=_init, initargs=(a.system, root, a.split, a.dnsmos)) if a.workers else nullcontext()
        with pool_cm as pool, ThreadPoolExecutor(a.asr_threads) as tp:
            stream = pool.imap(_work, range(n)) if pool else map(_work, range(n))
            def transcribe(item):
                row, est = item
                if asr is not None and est is not None:
                    try:
                        row["asr_text"] = whisper_text(asr, est)
                    except Exception as e:
                        print(f"asr {row['id']} failed: {e!r}")
                return row
            # executor.map would drain the whole stream before yielding; a bounded deque keeps ~2x threads in flight
            pending, bar = deque(), tqdm(total=n, desc=a.system)
            for item in stream:
                pending.append(tp.submit(transcribe, item))
                if len(pending) >= 2 * a.asr_threads:
                    w.writerow(pending.popleft().result()); bar.update()
            while pending:
                w.writerow(pending.popleft().result()); bar.update()
            bar.close()

if __name__ == "__main__":
    main()
