"""Per-clip metrics over a rendered split. One CSV row per clip."""
import argparse, csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np, torch
from tqdm import tqdm

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline, stft
from vaani.models import baselines, cascade
from vaani.train import build_model


def enhance_fn(spec: str, device=None):
    """`ckpt:<path>` runs a checkpoint; `post:<yaml>` runs its `base_checkpoint` then `vaani.dsp.postfilter` on the output
    spectrum before the one iSTFT (Tier 4.6); anything else is a baseline name."""
    if spec.startswith("conditional:"):
        import yaml
        cfg = yaml.safe_load(open(spec[len("conditional:"):], encoding="utf-8"))
        spec_fn, _ = _ckpt_spectrum_fn(cfg["base_checkpoint"], device, conditional_cfg=cfg.get("conditional", {}))
        def conditional_audio(mix):
            return stft.istft(spec_fn(mix), length=mix.shape[1])[0].cpu().numpy()
        conditional_audio.conditional_runtime = spec_fn.conditional_runtime
        return conditional_audio
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
    if spec.startswith("onnx:"):      # `onnx:<graph.onnx>@<checkpoint.pt>`: score the exported graph itself
        graph, ckpt = spec[5:].rsplit("@", 1)
        spec_fn = _onnx_spectrum_fn(graph, ckpt)
    elif spec.startswith("cascade:"):   # Tier 4.6 frozen first stage + refiner; one self-contained checkpoint
        spec_fn, cfg = _ckpt_spectrum_fn(spec[8:], device)
        if cfg["model"] != cascade.MODEL_NAME: raise ValueError(f"{spec[8:]} is a {cfg['model']!r} checkpoint, not a cascade")
    elif spec.startswith("ckpt:"):
        spec_fn, _ = _ckpt_spectrum_fn(spec[5:], device)
    else:
        return baselines.get(spec).enhance

    def f(mix):
        return stft.istft(spec_fn(mix), length=mix.shape[1])[0].cpu().numpy()
    return f


def _onnx_spectrum_fn(onnx_path, ckpt_path):
    """The exported graph's output spectrum, streamed frame by frame as the embedded loop runs it.

    Scoring the graph rather than the checkpoint is what makes an optimized export (quantized,
    pruned, or a future TensorRT-targeted graph) measurable in SNR/STOI/PESQ instead of only in
    bytes and milliseconds. The checkpoint is still required and is not redundant: it carries the
    DSP configuration the weights were trained behind (`controller_on`, `dsp`), which the graph
    does not encode. Caches come from the graph's own declared shapes and are zeroed per clip --
    carrying them between clips would leak one item's state into the next.

    CPU only, and `device` is deliberately not a parameter: the deployment claim is a single-core
    CPU claim, and eval workers already run one torch thread each.
    """
    from vaani import export  # keeps onnxruntime off the import path of the ordinary checkpoint eval

    cfg = torch.load(ckpt_path, map_location="cpu", weights_only=True)["config"]
    sess = export.load_session(onnx_path)
    cache_names, zero = export.zero_caches(sess)

    def spec_of(mix):
        r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"))
        x = torch.from_numpy(r["mix"])[None]   # limited when the checkpoint trained with the limiter
        spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]),
                           stft.stft(torch.from_numpy(r["n_hat"])[None])], -1).numpy()
        feats = np.ascontiguousarray(r["features"][None], dtype=np.float32)
        out, _ = export.stream_onnx(sess, spec6, feats, cache_names, [c.copy() for c in zero])
        return torch.from_numpy(out)
    return spec_of


def _ckpt_spectrum_fn(path, device=None, conditional_cfg=None):
    """The checkpoint's full output spectrum (mask + deep-filter taps), still on `device`; iSTFT is the caller's."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=True); cfg = ck["config"]
    m = cascade.FrozenCascade.from_config(cfg) if cfg["model"] == cascade.MODEL_NAME else build_model(cfg["model"], model_cfg=cfg.get("model_cfg"))
    m = m.to(device); m.load_state_dict(ck["model"]); m.eval()
    runtime = None
    if conditional_cfg is not None:
        if cfg["model"] != cascade.MODEL_NAME:
            raise ValueError("conditional refinement requires a cascade checkpoint")
        from vaani.models.conditional_refiner import ConditionalRefinerRuntime
        runtime = ConditionalRefinerRuntime(m, **conditional_cfg)

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
        feats = torch.from_numpy(r["features"])[None].to(device)
        return runtime(spec6, feats, torch.from_numpy(r["reliability"]).to(device)) if runtime is not None else m(spec6, feats)
    if runtime is not None:
        spec_of.conditional_runtime = runtime
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
        # One BLAS/OpenMP thread per worker. _init caps torch and vaani/dnsmos.py caps onnxruntime,
        # but numpy's BLAS has its own pool and reads the environment at import, so it has to be set
        # here, before the workers exist - the same reason rirs.build_bank sets it around its pool.
        # Without it N workers each start a thread per core and spend their time contending: the
        # symptom is a run that starts fast and settles back to its single-worker rate.
        _caps = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        _saved = {k: os.environ.get(k) for k in _caps}
        os.environ.update(_caps)
        try:
            pool_cm = Pool(a.workers, initializer=_init, initargs=(a.system, root, a.split, a.dnsmos)) if a.workers else nullcontext()
        finally:
            for _k, _v in _saved.items():
                if _v is None: os.environ.pop(_k, None)
                else: os.environ[_k] = _v
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
