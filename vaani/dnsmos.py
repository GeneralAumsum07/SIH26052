"""DNSMOS P.835 (Reddy et al., ICASSP 2022): non-intrusive SIG / BAK / OVRL on the enhanced output.

Mirrors microsoft/DNS-Challenge DNSMOS/dnsmos_local.py for the non-personalised sig_bak_ovr.onnx: 9.01 s windows
hopped by 1 s, clip repeated until it fills one window, per-window raw scores mapped through the published
polynomial fits, then averaged. The model takes the raw 16 kHz waveform (no mel front-end); the P.808 model is
not used. Reported beside PESQ/STOI as a second opinion that needs no clean reference - the eval sets have one,
so this is a check on the intrusive metrics, not a replacement.
"""
from pathlib import Path

import numpy as np

SR = 16000
WINDOW = 144160                       # 9.01 s, the fixed input length of sig_bak_ovr.onnx
HOP = SR                              # 1 s
MODEL = Path(__file__).resolve().parents[1] / "deploy/dnsmos/sig_bak_ovr.onnx"
# polynomial fits from dnsmos_local.py (is_personalized_MOS=False); raw -> MOS
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


class DNSMOS:
    def __init__(self, model: Path = MODEL):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1   # eval workers already fill the cores
        self.sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])

    def __call__(self, x: np.ndarray) -> dict:
        """x: mono float waveform at 16 kHz. Returns {sig, bak, ovrl} MOS (1..5)."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        while len(x) < WINDOW:                # reference repeats short clips instead of padding with silence
            x = np.concatenate([x, x])
        n_hops = int(np.floor(len(x) / SR) - WINDOW / SR) + 1
        raw = []
        for k in range(max(n_hops, 1)):
            seg = x[k * HOP:k * HOP + WINDOW]
            if len(seg) < WINDOW:
                continue
            raw.append(self.sess.run(None, {"input_1": seg[None, :]})[0][0])
        sig, bak, ovr = np.mean(raw, axis=0)
        return dict(sig=float(_P_SIG(sig)), bak=float(_P_BAK(bak)), ovrl=float(_P_OVR(ovr)))
