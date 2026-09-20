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
HOP, SUB = 256, 64                 # 16 ms hop split into 4 ms sub-blocks for the onset detector
SUB_HIST = 25                      # 100 ms of sub-block history behind the onset test
FLOOR_UP, FLOOR_DOWN, FLOOR_MAX = 0.002, 0.3, 3.0   # ratio-floor tracker: ~8 s up, ~50 ms down, capped at the mic-mismatch bound
SP_RELEASE = 0.85                  # speech-presence release per frame (~100 ms)
E_FLOOR_UP, E_FLOOR_DOWN = 0.02, 0.3   # primary energy floor: ~0.8 s up, ~50 ms down (a minimum-statistics stand-in)


class FrameFeatures:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha  # smoothing for coherence / speech-presence
        self.reset()

    def reset(self):
        self.sub_hist = None                        # last 100 ms of 4 ms sub-block energies; seeded by frame 0
        self.sub_hist_r = None                      # same for the reference mic (differential jump, plan 2.7a)
        self.diff_jump = 0.0                        # primary onset minus reference onset (dB); not a feature slot
        self.prev_mag = None
        self.Spp = np.zeros(257); self.Srr = np.zeros(257); self.Spr = np.zeros(257, complex)
        self.sp_smooth = 0.0
        self.ratio_floor = 0.0                      # tracked inter-mic level ratio of the noise (dB)
        self.e_floor = None                         # slow minimum tracker of primary frame energy (2.8 local-SNR proxy)
        self.prim_margin = 0.0                      # current primary energy over that floor (dB); not a feature slot

    def compute(self, p, r, P, R, nlms_health, prev_gate):
        f = np.zeros(N_FEATURES, np.float32)
        e = float((p ** 2).mean() + 1e-10)
        # sub-frame onset: the newest hop split into 4 ms blocks against the median of the preceding 100 ms.
        # A frame-to-frame delta dilutes a 5 ms gunshot onset 6x inside a 32 ms window; this does not.
        sub = (p[-HOP:].reshape(-1, SUB) ** 2).mean(axis=1) + 1e-10
        if self.sub_hist is None:
            self.sub_hist = np.full(SUB_HIST, sub.mean())   # no history yet: the first frame is its own floor
        f[0] = float(10 * np.log10(sub.max() / np.median(self.sub_hist)))
        self.sub_hist = np.concatenate([self.sub_hist[len(sub):], sub])
        # the same onset test on the reference: a far-field burst jumps alike at both mics (diff ~ 0 dB), a
        # consonant is near-mouth and jumps more at the primary. Kept off the 18-slot feature contract on
        # purpose: the controller reads it directly, the model's FiLM/ONNX signature does not move.
        sub_r = (r[-HOP:].reshape(-1, SUB) ** 2).mean(axis=1) + 1e-10
        if self.sub_hist_r is None:
            self.sub_hist_r = np.full(SUB_HIST, sub_r.mean())
        self.diff_jump = f[0] - float(10 * np.log10(sub_r.max() / np.median(self.sub_hist_r)))
        self.sub_hist_r = np.concatenate([self.sub_hist_r[len(sub_r):], sub_r])
        mag = np.abs(P)
        f[1] = 0.0 if self.prev_mag is None else float(np.sum(np.maximum(mag - self.prev_mag, 0)) / (np.sum(self.prev_mag) + 1e-8))
        self.prev_mag = mag
        # primary-only local SNR: independent of the reference gain, unlike the ratio margin (refgain_-12dB
        # pins the ratio floor at its cap and every frame would look like clean speech)
        if self.e_floor is None: self.e_floor = e
        self.e_floor += (E_FLOOR_UP if e > self.e_floor else E_FLOOR_DOWN) * (e - self.e_floor)
        self.prim_margin = float(10 * np.log10(e / self.e_floor))
        f[2] = float(np.abs(p).max() / (np.sqrt(e) + 1e-8))
        f[3] = float((np.abs(p) > CLIP).mean()); f[4] = float((np.abs(r) > CLIP).mean())
        er = float((r ** 2).mean() + 1e-10)
        # near-mouth speech: primary >> reference. far-field noise: ~equal.
        ratio_db = 10 * np.log10(e / er)
        # floor = slow tracker of the ratio's minimum (far-field noise, ~0 dB +/- mic mismatch); speech is
        # what sits well above it, whatever the reference gain of this headset happens to be. Capped: a
        # ratio above FLOOR_MAX is never noise, so constant speech cannot drag the floor up to itself.
        rate = FLOOR_UP if ratio_db > self.ratio_floor else FLOOR_DOWN
        self.ratio_floor = min(self.ratio_floor + rate * (ratio_db - self.ratio_floor), FLOOR_MAX)
        # mixer physics: speech is >= 8 dB above the noise ratio (ref_speech_gain <= -8 dB); noise scatters +/-3
        sp = float(np.clip((ratio_db - self.ratio_floor - 2.0) / 4.0, 0, 1))   # +2 dB -> 0, +6 dB -> 1
        # fast attack, slow release: an NLMS with an 80 ms time constant learns the speech path in the
        # lag of a symmetric smoother, so the first speech frame must already read as speech
        self.sp_smooth = max(sp, SP_RELEASE * self.sp_smooth)
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
