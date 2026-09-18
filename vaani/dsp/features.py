"""Per-frame impulse & reliability features. Order is a contract (spec 5.2).
Everything is cheap by design - these run on the embedded CPU every 16 ms.
"""
import numpy as np

N_FEATURES = 18
FEATURE_NAMES = (
    "log_energy_delta", "spectral_flux", "peak_to_rms", "clip_frac_primary",
    "clip_frac_reference", "speech_presence",
    *[f"coherence_b{i}" for i in range(8)],
    "level_diff_db", "ref_dropout", "nlms_health", "prev_gate",
)
assert len(FEATURE_NAMES) == N_FEATURES == 18
CLIP = 0.99
# 8 ERB-ish band edges in STFT bins (16 kHz, 257 bins): coarse enough to be robust
BAND_EDGES = [1, 4, 8, 14, 22, 34, 52, 80, 257]


class FrameFeatures:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha  # smoothing for coherence / speech-presence
        self.reset()

    def reset(self):
        self.prev_log_e = -12.0
        self.prev_mag = None
        self.Spp = np.zeros(257); self.Srr = np.zeros(257); self.Spr = np.zeros(257, complex)
        self.sp_smooth = 0.0

    def compute(self, p, r, P, R, nlms_health, prev_gate):
        f = np.zeros(N_FEATURES, np.float32)
        e = float((p ** 2).mean() + 1e-10); log_e = 10 * np.log10(e)
        f[0] = log_e - self.prev_log_e; self.prev_log_e = log_e
        mag = np.abs(P)
        f[1] = 0.0 if self.prev_mag is None else float(np.sum(np.maximum(mag - self.prev_mag, 0)) / (np.sum(self.prev_mag) + 1e-8))
        self.prev_mag = mag
        f[2] = float(np.abs(p).max() / (np.sqrt(e) + 1e-8))
        f[3] = float((np.abs(p) > CLIP).mean()); f[4] = float((np.abs(r) > CLIP).mean())
        er = float((r ** 2).mean() + 1e-10)
        # near-mouth speech: primary >> reference. far-field noise: ~equal.
        # Threshold sits above typical near-field level gaps (~14 dB from mic
        # placement alone) so ordinary two-mic geometry does not permanently
        # freeze adaptation; only a pronounced, sustained gap counts as "speech".
        ratio_db = 10 * np.log10(e / er)
        sp = float(np.clip((ratio_db - 12.0) / 18.0, 0, 1))     # 12 dB -> 0, 30 dB -> 1
        self.sp_smooth = self.alpha * self.sp_smooth + (1 - self.alpha) * sp
        f[5] = self.sp_smooth
        a = self.alpha
        self.Spp = a * self.Spp + (1 - a) * np.abs(P) ** 2
        self.Srr = a * self.Srr + (1 - a) * np.abs(R) ** 2
        self.Spr = a * self.Spr + (1 - a) * P * np.conj(R)
        coh = np.abs(self.Spr) ** 2 / (self.Spp * self.Srr + 1e-12)
        for i in range(8):
            f[6 + i] = float(coh[BAND_EDGES[i]:BAND_EDGES[i + 1]].mean())
        f[14] = float(np.clip(ratio_db, -40, 40))
        f[15] = 1.0 if ratio_db > 30.0 else 0.0
        f[16] = float(nlms_health); f[17] = float(prev_gate)
        return f
