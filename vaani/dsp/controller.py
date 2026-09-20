"""Burst / reliability gating. Rule-based with hysteresis.

Burst = large energy jump AND both mics hit at similar level. A consonant
is a jump too, but it is near-mouth, so the level difference stays large;
that second condition is what keeps speech transients from freezing the
filter or being treated as noise.
"""
import numpy as np

from vaani.dsp.features import FEATURE_NAMES

_I = {n: i for i, n in enumerate(FEATURE_NAMES)}


class Controller:
    def __init__(self, jump_db=12.0, level_diff_max_db=3.0, hold_frames=4, ramp_frames=12,
                 speech_freeze=0.5, diff_jump_max_db=None, block_margin_db=0.0):
        self.jump_db, self.ld_max, self.hold, self.ramp, self.sp_freeze = jump_db, level_diff_max_db, hold_frames, ramp_frames, speech_freeze
        # plan 2.7a: when set, the burst test is "onset >= jump_db AND differential jump <= diff_jump_max_db" and
        # the frame-level level_diff test is dropped (it reads ~0 dB for consonants too once the noise is loud).
        self.dj_max = diff_jump_max_db
        # plan 2.8: the blocking matrix adapts only while the primary sits this far above its own tracked noise
        # floor (local SNR); trained on noisy speech it learns the noise path too and strips the reference
        self.block_margin = block_margin_db
        self.reset()

    def reset(self):
        self.hold_left = 0
        self.gate = 1.0
        self.ramp_pos = self.ramp   # fully ramped
        self.speech_adapt = 0.0     # 2.8: the blocking matrix adapts when this is 1 (talker active, nothing else wrong)

    def step(self, f: np.ndarray, diff_jump: float = 0.0, limiter_hit: bool = False, prim_margin: float = 0.0):
        jump = f[_I["log_energy_delta"]]
        ld = f[_I["level_diff_db"]]
        far = (diff_jump <= self.dj_max) if self.dj_max is not None else (ld <= self.ld_max)
        # the limiter squashes the very onset the jump test looks for (TPR 0.90 -> 0.46 measured), but it only
        # engages on far-field over-ceiling sub-blocks, so its engagement is itself the burst detection
        burst_now = ((jump >= self.jump_db) and far) or limiter_hit
        if burst_now:
            self.hold_left = self.hold
        burst = self.hold_left > 0
        if self.hold_left > 0:
            self.hold_left -= 1

        overload = f[_I["clip_frac_primary"]] > 0.01 or f[_I["clip_frac_reference"]] > 0.01
        dropout = f[_I["ref_dropout"]] > 0.5
        speech = f[_I["speech_presence"]] > self.sp_freeze
        freeze = burst or overload or dropout or speech
        self.speech_adapt = float(speech and prim_margin >= self.block_margin and not (burst or overload or dropout))
        if freeze:
            self.ramp_pos = 0; self.gate = 0.0
        else:
            self.ramp_pos = min(self.ramp, self.ramp_pos + 1)
            self.gate = self.ramp_pos / self.ramp   # linear ramp ~200 ms at 16 ms hop

        coh_ok = float(np.clip(f[_I["coherence_b2"]:_I["coherence_b2"] + 4].mean() * 2, 0, 1))
        reliability = (1 - f[_I["clip_frac_primary"]]) * (1 - f[_I["ref_dropout"]]) * (0.5 + 0.5 * coh_ok) * (0.5 + 0.5 * (1 - f[_I["nlms_health"]]))
        return float(self.gate), bool(burst), float(np.clip(reliability, 0, 1))
