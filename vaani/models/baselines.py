"""Non-trained comparison rows. Common API: enhance(mix (2,T)) -> (T,)."""
import os, shutil, subprocess, tempfile
from pathlib import Path

import numpy as np, torch
from scipy.signal import resample_poly

from vaani.dsp import pipeline, stft
from vaani.models.gtcrn import GTCRN

CKPT = Path(__file__).parent / "checkpoints" / "model_trained_on_dns3.tar"


class Raw:
    def enhance(self, mix): return mix[0].copy()


class NlmsOnly:
    """Classical: primary minus the guarded NLMS estimate."""
    def enhance(self, mix):
        return (mix[0] - pipeline.run(mix)["n_hat"]).astype(np.float32)


class GtcrnPretrained:
    def __init__(self):
        self.m = GTCRN().eval()
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)  # weights_only: no optimizer/pickle trust needed
        self.m.load_state_dict(ck["model"])

    def enhance(self, mix):
        with torch.no_grad():
            x = torch.from_numpy(mix[0])[None]
            return stft.istft(self.m(stft.stft(x)), length=x.shape[-1])[0].numpy()


class RNNoise:
    """Requires the `rnnoise_demo` binary on PATH (48 kHz raw PCM16 in/out).
    Labelled in reports: different sample rate and training data.
    No working Python rnnoise binding builds cleanly on Windows here, so this
    stays a documented stub - name registered, enhance() raises."""
    def enhance(self, mix):
        exe = shutil.which("rnnoise_demo")
        if exe is None:
            raise NotImplementedError("rnnoise binding unavailable on this platform")
        x48 = resample_poly(mix[0], 3, 1)
        with tempfile.TemporaryDirectory() as d:
            i, o = Path(d, "i.raw"), Path(d, "o.raw")
            (np.clip(x48, -1, 1) * 32767).astype(np.int16).tofile(i)
            subprocess.run([exe, str(i), str(o)], check=True, capture_output=True)
            y48 = np.fromfile(o, np.int16).astype(np.float32) / 32767
        return resample_poly(y48, 1, 3)[: mix.shape[1]].astype(np.float32)


class DeepFilterNet3:
    """Published single-channel comparator (Schröter et al. 2023: 48 kHz, ~2.3M params, trained on DNS4).
    Labelled in reports: mono, 48 kHz model, ~45x our parameter budget. Lives in `.venv-dfn` (py3.11, numpy<2,
    torch 2.0.1) so it cannot pollute the main env; scripts/dfn_worker.py loads it once per eval process and
    enhances clips over a tab-separated stdin line protocol through temp wavs (~0.3 s/clip vs ~20 s/process)."""
    WORKER = Path(__file__).parents[2] / "scripts" / "dfn_worker.py"
    PY = Path(__file__).parents[2] / ".venv-dfn" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def __init__(self):
        if not self.PY.exists():
            raise NotImplementedError(f"{self.PY} missing: python3.11 venv with deepfilternet, torch==2.0.1 cpu, soundfile")
        self.p = subprocess.Popen([str(self.PY), str(self.WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        assert self.p.stdout.readline().strip() == "ready", "dfn_worker failed to start"

    def enhance(self, mix):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            i, o = Path(d, "i.wav"), Path(d, "o.wav")
            sf.write(i, mix[0], 16000, subtype="FLOAT")
            self.p.stdin.write(f"{i}\t{o}\n"); self.p.stdin.flush()
            r = self.p.stdout.readline().strip()
            if r != "ok":
                raise RuntimeError(f"dfn_worker: {r}")
            y, _ = sf.read(o, dtype="float32")
        return y[: mix.shape[1]].astype(np.float32)


def get(name: str):
    return {"raw": Raw, "nlms_only": NlmsOnly, "gtcrn_pretrained": GtcrnPretrained, "rnnoise": RNNoise,
            "deepfilternet3": DeepFilterNet3}[name]()
