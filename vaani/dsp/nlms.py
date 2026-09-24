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
    """Reference loop, ported to C as-is. Explicit float32 sums in k order (no Python
    float(), no BLAS dot) so the numba kernel and the C port are the same algorithm."""
    n = len(primary)
    taps = len(w)
    n_hat = np.empty(n, np.float32)
    for i in range(n):
        buf[1:] = buf[:-1]; buf[0] = reference[i]
        y = np.float32(0.0)
        for k in range(taps):
            y += w[k] * buf[k]
        n_hat[i] = y
        if mu > 0.0:
            e = primary[i] - y
            energy = np.float32(0.0)
            for k in range(taps):
                energy += buf[k] * buf[k]
            step = mu * e / (energy + eps)
            for k in range(taps):
                w[k] += step * buf[k]
    return n_hat


def _process_robust_pure(w, buf, pbuf, st, primary, reference, mu, eps, eps_rel, w_max, sub, drop_ratio, lt_alpha):
    """Plan 2.8 dropout-safe variant; used only when NLMS(robust=...) is set, so the default loop above is untouched.
    st = [long-term reference power, frozen sub-blocks count]. pbuf delays the primary by len(pbuf)-1 samples in the
    error path (D=0: pbuf has one slot, e = primary[i]). A sub-block whose reference power falls drop_ratio below the
    long-term power is treated as a dropout: no adaptation. eps grows with the long-term reference power, and ||w|| is
    clamped, so a near-silent reference can neither blow the step up nor leave huge taps behind for the reconnect."""
    n = len(primary)
    taps = len(w)
    d = len(pbuf)
    n_hat = np.empty(n, np.float32)
    for s0 in range(0, n, sub):
        s1 = min(n, s0 + sub)
        e_sub = np.float32(0.0)
        for i in range(s0, s1):
            e_sub += reference[i] * reference[i]
        e_sub = e_sub / np.float32(s1 - s0)
        adapt = mu > 0.0 and not (st[0] > 0.0 and e_sub < drop_ratio * st[0])
        if not adapt:
            st[1] += 1.0
        for i in range(s0, s1):
            buf[1:] = buf[:-1]; buf[0] = reference[i]
            pbuf[1:] = pbuf[:-1]; pbuf[0] = primary[i]
            r2 = reference[i] * reference[i]
            st[0] = r2 if st[0] == 0.0 else st[0] + lt_alpha * (r2 - st[0])
            y = np.float32(0.0)
            for k in range(taps):
                y += w[k] * buf[k]
            n_hat[i] = y
            if adapt:
                e = pbuf[d - 1] - y
                energy = np.float32(0.0)
                for k in range(taps):
                    energy += buf[k] * buf[k]
                step = mu * e / (energy + eps + eps_rel * np.float32(taps) * st[0])
                nrm = np.float32(0.0)
                for k in range(taps):
                    w[k] += step * buf[k]
                    nrm += w[k] * w[k]
                if nrm > w_max * w_max:
                    g = w_max / np.sqrt(nrm)
                    for k in range(taps):
                        w[k] *= g
    return n_hat


if _HAVE_NUMBA:
    _process_robust_numba = njit(cache=True)(_process_robust_pure)

    @njit(cache=True)
    def _process_numba(w, buf, primary, reference, mu, eps):
        """Literal transcription of _process_pure -- same accumulation order."""
        n = len(primary)
        taps = len(w)
        n_hat = np.empty(n, np.float32)
        for i in range(n):
            buf[1:] = buf[:-1]; buf[0] = reference[i]
            y = np.float32(0.0)
            for k in range(taps):
                y += w[k] * buf[k]
            n_hat[i] = y
            if mu > 0.0:
                e = primary[i] - y
                energy = np.float32(0.0)
                for k in range(taps):
                    energy += buf[k] * buf[k]
                step = mu * e / (energy + eps)
                for k in range(taps):
                    w[k] += step * buf[k]
        return n_hat


ROBUST_DEFAULTS = dict(eps_rel=0.01, w_max=10.0, sub=64, drop_db=-30.0, lt_alpha=1e-4, delay=0)


class NLMS:
    def __init__(self, taps: int = 64, mu: float = 0.05, eps: float = 1e-6, force_pure: bool = False, robust=None):
        """robust: None = the r1..r7 loop, bit-exact; True or a dict over ROBUST_DEFAULTS = the plan 2.8 dropout-safe
        loop (eps tied to long-term reference power, ||w|| clamp, per-sub-block dropout freeze, primary delay `delay`)."""
        self.taps, self.mu, self.eps = taps, mu, eps
        self.force_pure = force_pure  # test hook: bypass numba even when available
        self.robust = None if not robust else {**ROBUST_DEFAULTS, **(robust if isinstance(robust, dict) else {})}
        if self.robust and (int(self.robust["delay"]) < 0 or int(self.robust["sub"]) < 1 or self.robust["w_max"] <= 0):
            raise ValueError("robust NLMS needs delay >= 0, sub >= 1 and w_max > 0")
        self.reset()

    def reset(self):
        self.w = np.zeros(self.taps, np.float32)
        self.buf = np.zeros(self.taps, np.float32)   # most recent reference samples, newest first
        if self.robust:
            self.pbuf = np.zeros(int(self.robust["delay"]) + 1, np.float32)   # primary delay line, newest first
            self.st = np.zeros(2, np.float32)          # long-term reference power, frozen sub-block count

    def process_block(self, primary: np.ndarray, reference: np.ndarray, gate: float):
        n = len(primary)
        mu = np.float32(self.mu * float(np.clip(gate, 0.0, 1.0)))
        eps = np.float32(self.eps)
        use_numba = _HAVE_NUMBA and not self.force_pure
        if self.robust:
            r = self.robust; fn = _process_robust_numba if use_numba else _process_robust_pure
            n_hat = fn(self.w, self.buf, self.pbuf, self.st, primary, reference, mu, eps, np.float32(r["eps_rel"]),
                       np.float32(r["w_max"]), int(r["sub"]), np.float32(10 ** (r["drop_db"] / 10)), np.float32(r["lt_alpha"]))
        else:
            fn = _process_numba if use_numba else _process_pure
            n_hat = fn(self.w, self.buf, primary, reference, mu, eps)
        resid = primary - n_hat
        ratio = float((resid ** 2).mean() / ((primary ** 2).mean() + 1e-12))
        return n_hat, min(ratio, 1.0)
