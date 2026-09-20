"""GSC-style blocking matrix on the reference path (plan 2.8).

The reference mic hears the talker too (about -8..-14 dB below the primary, ~5 samples later). Above roughly
+5 dB SNR that leaked speech is the loudest thing in the reference, so the noise-path NLMS learns the speech
path instead and n_hat tracks the speech (ERLE < 0 at +15 dB in diag_controller). The fix is the classic
generalised-sidelobe-canceller move: estimate the mouth -> reference path from the primary and subtract it,
ref_b = ref - h_s * prim, and hand ref_b to the noise-path NLMS. h_s is a short NLMS adapted ONLY while the
controller sees speech (the complement of the noise filter's gate) and it is never reset, so it is a
long-term estimate of the headset geometry rather than of any one frame. Noise loses little: h_s has ~-10 dB
gain, so the far-field noise in ref_b is the reference's noise minus a tenth of the primary's.

Reuses the NLMS reference loop with the roles swapped (primary=ref, reference=prim), so the C port is the
same kernel called twice.
"""
import numpy as np

from vaani.dsp.nlms import NLMS

TAPS = 16          # covers the 0-15 sample mouth -> reference lag of the 12 cm geometry with margin
MU = 0.01          # slow: the path is fixed hardware, the estimate should be too


class BlockingMatrix:
    def __init__(self, taps: int = TAPS, mu: float = MU):
        self.f = NLMS(taps=taps, mu=mu)

    def reset(self):
        self.f.reset()

    def process_block(self, prim: np.ndarray, ref: np.ndarray, speech_gate: float) -> np.ndarray:
        """ref_b for this hop. speech_gate 1 = adapt (talker active, no burst/overload/dropout), 0 = frozen."""
        y, _ = self.f.process_block(ref, prim, speech_gate)
        return (ref - y).astype(np.float32)
