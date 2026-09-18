"""Non-trained comparison rows. Common API: enhance(mix (2,T)) -> (T,)."""
import shutil, subprocess, tempfile
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


def get(name: str):
    return {"raw": Raw, "nlms_only": NlmsOnly, "gtcrn_pretrained": GtcrnPretrained, "rnnoise": RNNoise}[name]()
