"""Conservative residual-noise post-filter on the network's output spectrum (Tier 4.6 plan §4).

Decision-directed Wiener gain with a causal rolling-minimum estimate of the residual PSD. Runs per frame on the
complete first-stage output (mask + deep-filter taps), before the single iSTFT, so it adds no lookahead and ports to
C as a few float32 arrays of state. Deliberately simpler than MMSE-LSA (no expint), and clamped by a high gain floor:
the minimum tracker is not a trustworthy noise estimate during sustained speech, and the frozen-test gate, not this
code, decides whether the trade against STOI was worth it.
"""
import numpy as np

DEFAULTS = dict(alpha_power=0.8, alpha_noise=0.9, alpha_dd=0.95, min_frames=64, warmup_frames=16,
                noise_bias=1.5, gain_floor=0.8, epsilon=1e-10)


class ResidualPostFilter:
    def __init__(self, n_bins=257, **config):
        unknown = set(config) - set(DEFAULTS)
        if unknown: raise TypeError(f"unknown post-filter keys {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **config}; self.n_bins = n_bins
        self.reset()

    def reset(self):
        c = self.cfg
        self.t = 0
        self.smooth = np.zeros(self.n_bins, np.float32)
        self.ring = np.zeros((c["min_frames"], self.n_bins), np.float32); self.n_valid = 0   # causal window, current + past
        self.noise = np.zeros(self.n_bins, np.float32)
        self.prev_out_power = np.zeros(self.n_bins, np.float32)
        self.prev_gain = np.ones(self.n_bins, np.float32)

    def process_frame(self, y: np.ndarray) -> np.ndarray:
        """complex64 (F,) first-stage spectrum -> complex64 (F,). Real and imaginary parts get the same real gain."""
        c = self.cfg; y = np.asarray(y, np.complex64)
        power = (y.real ** 2 + y.imag ** 2).astype(np.float32)
        self.smooth = power if self.t == 0 else c["alpha_power"] * self.smooth + (1 - c["alpha_power"]) * power
        self.ring[self.t % c["min_frames"]] = self.smooth; self.n_valid = min(self.n_valid + 1, c["min_frames"])
        cand = c["noise_bias"] * self.ring[: self.n_valid].min(0)
        self.noise = cand if self.t == 0 else c["alpha_noise"] * self.noise + (1 - c["alpha_noise"]) * cand
        self.noise = np.maximum(self.noise, c["epsilon"]).astype(np.float32)
        gamma = np.minimum(power / self.noise, 1000.0)
        xi = c["alpha_dd"] * self.prev_out_power / self.noise + (1 - c["alpha_dd"]) * np.maximum(gamma - 1, 0)
        gain = np.clip(xi / (1 + xi), c["gain_floor"], 1.0)
        pad = np.concatenate([gain[:1], gain, gain[-1:]])                       # replicated edge bins
        gain = np.maximum(c["gain_floor"], 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:])
        gain = np.where(gain > self.prev_gain, gain, 0.8 * self.prev_gain + 0.2 * gain)   # speech onsets restore at once
        if self.t < c["warmup_frames"]: gain = np.ones_like(gain)
        gain = gain.astype(np.float32)
        z = (gain * y).astype(np.complex64)
        self.prev_out_power = (z.real ** 2 + z.imag ** 2).astype(np.float32)
        self.prev_gain = gain
        self.t += 1
        return z

    def process(self, spec: np.ndarray) -> np.ndarray:
        """Whole clip (F, T) complex -> (F, T); a convenience wrapper over process_frame (same state sequence)."""
        self.reset()
        return np.stack([self.process_frame(spec[:, t]) for t in range(spec.shape[1])], 1)
