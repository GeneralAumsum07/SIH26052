"""Per-clip metrics over a rendered split. One CSV row per clip."""
import argparse, csv
from pathlib import Path

import numpy as np, torch
from tqdm import tqdm

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline, stft
from vaani.models import baselines
from vaani.train import build_model


def enhance_fn(spec: str):
    if not spec.startswith("ckpt:"):
        return baselines.get(spec).enhance
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--eval-root", default="data/eval"); ap.add_argument("--out", required=True)
    ap.add_argument("--asr", action="store_true", help="also compute WER with faster-whisper (supporting evidence only)")
    a = ap.parse_args()
    ds = RenderedDataset(Path(a.eval_root) / a.split); fn = enhance_fn(a.system)
    asr = None
    if a.asr:
        try:
            from faster_whisper import WhisperModel; asr = WhisperModel("small", device="cpu", compute_type="int8")
        except ImportError:
            print("faster-whisper not installed; --asr rows will be NaN")  # never a hard dependency
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "id", "bucket", "noise_class", "snr_in", "clipped", "ref_dropout", "impulse_peak_db",
            "snr_out", "si_sdr", "stoi", "pesq_wb", "recovery_s", "asr_text"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for it in tqdm(ds, desc=a.system):
            mix, clean, meta = it["mix"].numpy(), it["clean"].numpy(), it["meta"]
            est = fn(mix)
            row = dict(system=a.system, id=meta["id"], bucket=meta["bucket"], noise_class=meta["noise_class"],
                       snr_in=meta["snr_db"], clipped=meta["clipped"], ref_dropout=meta["ref_dropout"],
                       impulse_peak_db=meta["impulse_peak_db"], snr_out=metrics.snr_db(clean, est),
                       si_sdr=metrics.si_sdr_db(clean, est), stoi=metrics.stoi(clean, est), pesq_wb=metrics.pesq_wb(clean, est),
                       recovery_s=float("nan"), asr_text="")
            if "twin" in it and meta["impulse_onsets_s"]:
                row["recovery_s"] = metrics.recovery_time_s(est, fn(it["twin"].numpy()), meta["impulse_onsets_s"][0])
            if asr is not None:
                segs, _ = asr.transcribe(est, language=None, beam_size=1)
                row["asr_text"] = " ".join(s.text for s in segs).strip()
            w.writerow(row)


if __name__ == "__main__":
    main()
