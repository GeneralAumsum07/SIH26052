"""Host-side conditional refinement, not an ONNX-traced Python branch.

Only c1/c2 are conditional. The cheap c0 projection always enters the history,
so a skipped frame does not freeze time or feed stale context on reactivation.
Thresholds are validation hypotheses, not a clean-speech detector or safety floor.
"""
import math

import torch
from torch.nn import functional as F

from vaani.models.residual_refiner import init_refine_cache, N_BINS


class ConditionalRefinerRuntime:
    def __init__(self, cascade, mode="conditional", snr_threshold_db=12., speech_threshold=.05,
                 reliability_threshold=None, crossfade=1.):
        if mode not in {"conditional", "always", "bypass"}:
            raise ValueError("mode must be conditional, always or bypass")
        if not math.isfinite(crossfade) or not 0 <= crossfade <= 1:
            raise ValueError("crossfade (residual blend) must be in [0,1]")
        if not math.isfinite(snr_threshold_db) or not 0 <= speech_threshold <= 1:
            raise ValueError("invalid SNR/speech thresholds")
        if reliability_threshold is not None and not 0 <= reliability_threshold <= 1:
            raise ValueError("reliability_threshold must be in [0,1]")
        self.cascade = cascade.eval()
        self.mode, self.snr_threshold_db, self.speech_threshold = mode, snr_threshold_db, speech_threshold
        self.reliability_threshold, self.crossfade = reliability_threshold, crossfade
        self.fired = self.total = 0

    @property
    def stats(self):
        m = self.cascade.refiner
        c0 = N_BINS * 8 * m.hidden
        heavy = N_BINS * (m.hidden ** 2 * (m.past + 1) * 3 + 2 * m.hidden)
        rate = self.fired / self.total if self.total else 0.
        return dict(fired=self.fired, total=self.total, fire_rate=rate,
                    avg_matrix_macs_per_frame=c0 + rate * heavy,
                    always_on_matrix_macs_per_frame=c0 + heavy)

    def reset_stats(self):
        self.fired = self.total = 0

    def _decision(self, primary, enhanced, reliability, speech_presence):
        if self.mode != "conditional":
            return self.mode == "always"
        signal = enhanced.float().square().mean()
        residual = (primary - enhanced).float().square().mean()
        snr = float(10 * torch.log10((signal + 1e-10) / (residual + 1e-10)))
        # The spectral ratio is a proxy when the DSP scalar is unavailable.
        presence = float((signal / (primary.float().square().mean() + 1e-10)).clamp(0, 1)) if speech_presence is None else float(speech_presence)
        unreliable = self.reliability_threshold is not None and reliability is not None and float(reliability) < self.reliability_threshold
        return unreliable or (presence >= self.speech_threshold and snr < self.snr_threshold_db)

    @torch.no_grad()
    def step(self, primary, reference, enhanced, cache, reliability=None, speech_presence=None):
        if self.reliability_threshold is not None and reliability is None:
            raise ValueError("configured reliability threshold requires DSP reliability")
        if primary.shape[0] != 1 or primary.shape[2] != 1:
            raise ValueError("conditional runtime operates on one stream and one frame")
        m = self.cascade.refiner
        f, scale, y = m._features(primary, reference, enhanced)
        h0 = F.relu(m.c0(f))
        history = torch.cat([cache, h0], 2)
        fired = self.crossfade > 0 and self._decision(primary, enhanced, reliability, speech_presence)
        self.total += 1; self.fired += int(fired)
        if fired:
            h1 = F.relu(m.c1(F.pad(history, (1, 1, 0, 0))))
            refined = m._out(h1, scale, y)
            # A constant residual blend bounds the correction; it is not claimed
            # to eliminate temporal discontinuity at threshold crossings.
            out = refined if self.crossfade == 1 else enhanced + self.crossfade * (refined - enhanced)
        else:
            out = enhanced
        return out, history[:, :, 1:], fired

    @torch.no_grad()
    def forward(self, spec6, feats, reliability=None):
        if spec6.shape[0] != 1 or spec6.shape[2] < 1:
            raise ValueError("conditional runtime requires one nonempty clip")
        if self.reliability_threshold is not None and reliability is None:
            raise ValueError("configured reliability threshold requires the DSP reliability sequence")
        enhanced = self.cascade.first(spec6, feats)
        return self.refine(spec6, feats, enhanced, reliability)

    @torch.no_grad()
    def refine(self, spec6, feats, enhanced, reliability=None):
        """Reuse a fixed first-stage spectrum when screening several policies."""
        if spec6.shape[0] != 1 or spec6.shape[2] < 1:
            raise ValueError("conditional runtime requires one nonempty clip")
        if self.reliability_threshold is not None and reliability is None:
            raise ValueError("configured reliability threshold requires DSP reliability")
        m = self.cascade.refiner
        cache = init_refine_cache(spec6.device, m.hidden, m.past)
        outputs = []
        for t in range(spec6.shape[2]):
            r = None if reliability is None else reliability.reshape(-1)[t]
            out, cache, _ = self.step(spec6[:, :, t:t+1, :2], spec6[:, :, t:t+1, 2:4],
                                      enhanced[:, :, t:t+1], cache, r, feats[0, t, 5])
            outputs.append(out)
        return torch.cat(outputs, 2)

    __call__ = forward
