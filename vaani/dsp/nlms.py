"""Guarded NLMS. reference -> primary noise path estimate.

Output is the noise ESTIMATE n_hat, not 'cleaned' audio: the neural model
decides how to use it, so an unconverged or speech-leaking filter cannot
delete speech before inference. The controller supplies `gate`, which
scales the step size (0 = frozen) during speech, bursts, overload and
reference dropout.

Written sample-by-sample in plain NumPy on purpose: the DSP lead ports
this loop to C and checks it against the golden vectors. The numba path
below is a straight transcription of that loop for training-time speed
only -- it must never diverge from the pure-Python reference.
"""
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


def _process_pure(w, buf, primary, reference, mu, eps):
    """Reference loop: this exact structure is what gets ported to C."""
    n = len(primary)
    n_hat = np.empty(n, np.float32)
    for i in range(n):
        buf[1:] = buf[:-1]; buf[0] = reference[i]
        y = float(w @ buf)
        n_hat[i] = y
        if mu > 0.0:
            e = primary[i] - y
            w += (mu * e / (float(buf @ buf) + eps)) * buf
    return n_hat


if _HAVE_NUMBA:
    @njit(cache=True)
    def _process_numba(w, buf, primary, reference, mu, eps):
        n = len(primary)
        n_hat = np.empty(n, np.float32)
        for i in range(n):
            buf[1:] = buf[:-1]; buf[0] = reference[i]
            y = np.float32(0.0)
            for k in range(len(w)):
                y += w[k] * buf[k]
            n_hat[i] = y
            if mu > 0.0:
                e = primary[i] - y
                energy = np.float32(0.0)
                for k in range(len(buf)):
                    energy += buf[k] * buf[k]
                step = mu * e / (energy + eps)
                for k in range(len(w)):
                    w[k] += step * buf[k]
        return n_hat


class NLMS:
    def __init__(self, taps: int = 64, mu: float = 0.05, eps: float = 1e-6, force_pure: bool = False):
        self.taps, self.mu, self.eps = taps, mu, eps
        self.force_pure = force_pure  # test hook: bypass numba even when available
        self.reset()

    def reset(self):
        self.w = np.zeros(self.taps, np.float32)
        self.buf = np.zeros(self.taps, np.float32)   # most recent reference samples, newest first

    def process_block(self, primary: np.ndarray, reference: np.ndarray, gate: float):
        n = len(primary)
        mu = np.float32(self.mu * float(np.clip(gate, 0.0, 1.0)))
        eps = np.float32(self.eps)
        use_numba = _HAVE_NUMBA and not self.force_pure
        fn = _process_numba if use_numba else _process_pure
        n_hat = fn(self.w, self.buf, primary, reference, mu, eps)
        resid = primary - n_hat
        ratio = float((resid ** 2).mean() / ((primary ** 2).mean() + 1e-12))
        return n_hat, min(ratio, 1.0)
