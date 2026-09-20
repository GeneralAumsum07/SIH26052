"""Sub-block transient limiter ahead of the STFT (plan 2.7b).

A 30-45 dB blast inside one 32 ms window owns the whole frame: the mask, the features and the NLMS all see
a frame that is nothing but burst. Clamping the burst to a fixed headroom over the running programme level
keeps the rest of the frame usable, and does so 2 ms at a time so the damage stays inside the sub-block.

Level alone cannot tell a blast from the first syllable after silence in 2 ms, and a level-only limiter
clamps every such onset by tens of dB. The headset geometry can: a blast is far-field (about equal at both
mics), a syllable is near-mouth (primary >= 8 dB over the noise ratio). So the limiter engages only on
sub-blocks that are BOTH over the ceiling AND far-field. One shared gain on both mics keeps the inter-mic
ratio the features rely on intact. Zero added latency in block processing: each sub-block's gain is
computed from that sub-block and applied before the hop reaches the STFT.
"""
import numpy as np

SUB = 32                       # 2 ms sub-blocks at 16 kHz
HEADROOM_DB = 26.0             # diag_controller sweep 2026-09-20 (eval_r2): 18 dB FPR 0.052, 22 dB 0.035, 26 dB 0.025 at TPR 0.92; blasts sit 25-45 dB above speech RMS
RELEASE_MS = 50.0
ENV_MS = 500.0                 # running programme-level tracker (both mics pooled)
FAR_FIELD_DB = 4.0             # engage only when the sub-block prim/ref ratio is within this of the noise floor
FLOOR_UP, FLOOR_DOWN, FLOOR_MAX = 0.002, 0.3, 3.0   # same ratio-floor tracker as features.py, per sub-block
MIN_ENV = 1e-4                 # -80 dBFS: nothing below this is ever limited


class Limiter:
    def __init__(self, headroom_db: float = HEADROOM_DB, release_ms: float = RELEASE_MS, env_ms: float = ENV_MS,
                 far_field_db: float = FAR_FIELD_DB, sub: int = SUB, sr: int = 16000):
        self.thr = 10 ** (headroom_db / 20)
        self.rel = 1 - np.exp(-sub / (release_ms * 1e-3 * sr))
        self.env_a = 1 - np.exp(-sub / (env_ms * 1e-3 * sr))
        self.far, self.sub = far_field_db, sub
        self.reset()

    def reset(self):
        self.env = None           # seeded by the first sub-block, like the onset history in features.py
        self.floor = 0.0          # tracked inter-mic level ratio of the noise (dB)
        self.gain = 1.0
        self.engaged = 0          # sub-blocks with gain < 1 (diagnostics; caller may zero it)

    def process_block(self, p: np.ndarray, r: np.ndarray):
        """One hop of both mics -> limited copies. Tail blocks shorter than SUB are handled."""
        p, r = p.astype(np.float32, copy=True), r.astype(np.float32, copy=True)
        for i in range(0, len(p), self.sub):
            bp, br = p[i:i + self.sub], r[i:i + self.sub]
            ep, er = float((bp ** 2).mean() + 1e-10), float((br ** 2).mean() + 1e-10)
            ratio_db = 10 * np.log10(ep / er)
            rate = FLOOR_UP if ratio_db > self.floor else FLOOR_DOWN
            self.floor = min(self.floor + rate * (ratio_db - self.floor), FLOOR_MAX)
            rms = float(np.sqrt(0.5 * (ep + er)))
            if self.env is None:
                self.env = rms
            peak = float(max(np.abs(bp).max(), np.abs(br).max())) + 1e-9
            ceiling = self.thr * max(self.env, MIN_ENV)
            if peak > ceiling and ratio_db <= self.floor + self.far:
                self.gain = min(self.gain, ceiling / peak)   # instant attack; env is not taught by the burst
                self.engaged += 1
            else:
                self.gain += self.rel * (1.0 - self.gain)    # exponential release, ~50 ms
                if self.gain > 0.999: self.gain = 1.0         # snap so released audio is bit-identical to the input
                # near-mouth speech may exceed the ceiling: it is programme, so the level tracker learns it
                self.env += self.env_a * (rms - self.env)
            bp *= self.gain; br *= self.gain
        return p, r
