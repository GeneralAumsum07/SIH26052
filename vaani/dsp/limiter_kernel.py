"""Compiled and vectorized kernels of vaani.dsp.limiter.Limiter (low-delay plan, Section 3.3 / Task 2).

The Python loop costs about half the loader's time per item (plan Section 2.7). The limiter's per-sub-block
statistics depend only on the input and its gain recurrence is scalar, so both kernels compute the same function:

  numba   the per-sub-block loop compiled, with float32 accumulation that reproduces NumPy's pairwise sums (8
          partial accumulators for blocks of 8..128 samples) and the gain applied as a float32 multiply. Used by
          every r8 arm; preflight already fails without numba, so the loader never falls back silently.
  numpy   statistics for every sub-block at once, then the scalar recurrence: the reference and the fallback
          outside r8.

Both match `Limiter` within 1e-6 (tests/test_low_delay_frontend.py); most clips are bit-identical, the rest differ
by one float32 ulp from summation order. `LimiterKernel` has Limiter's interface (process_block, reset, engaged,
the state attributes), so it drops in wherever Limiter runs; r7's default path keeps the loop.

Chunk invariance: any chunking whose chunk length is a multiple of the sub-block gives the same output (the state
is carried per sub-block), so the r8 limiter is delay-free and chunk-invariant at 96-, 128- and 256-sample hops.
"""
from __future__ import annotations

import numpy as np

from vaani.dsp.limiter import FLOOR_DOWN, FLOOR_MAX, MIN_ENV, Limiter

try:
    import numba
except ImportError:   # optional outside r8: kernel="numba" then raises
    numba = None

KERNELS = ("loop", "numpy", "numba")


def _stats_numpy(p: np.ndarray, r: np.ndarray, sub: int):
    """Per-sub-block (ep, er, peak) as the loop computes them: float32 means (+1e-10 in float32), float32 peaks."""
    n = len(p)
    full = n // sub
    eps = np.float32(1e-10)
    ep = np.empty(full + (n % sub > 0), np.float64)
    er, pk = np.empty_like(ep), np.empty_like(ep)
    if full:
        bp, br = p[:full * sub].reshape(full, sub), r[:full * sub].reshape(full, sub)
        ep[:full] = ((bp ** 2).mean(1) + eps).astype(np.float64)
        er[:full] = ((br ** 2).mean(1) + eps).astype(np.float64)
        pk[:full] = np.maximum(np.abs(bp).max(1), np.abs(br).max(1)).astype(np.float64)
    if n % sub:
        bp, br = p[full * sub:], r[full * sub:]
        ep[full] = float((bp ** 2).mean() + eps)
        er[full] = float((br ** 2).mean() + eps)
        pk[full] = float(max(np.abs(bp).max(), np.abs(br).max()))
    return ep, er, pk


def _recurrence(ep, er, pk, st, prm):
    """Scalar gain recurrence over sub-blocks; st = [has_env, env, floor, gain, engaged, run] (updated in place).
    Returns the per-sub-block gains (float64)."""
    thr, rel, env_a, far, fix, env_a_hold, hold_max, floor_up = prm
    has_env, env, floor, gain, engaged, run = st
    g = np.empty(len(ep))
    for i in range(len(ep)):
        ratio_db = 10 * np.log10(ep[i] / er[i])
        rate = floor_up if ratio_db > floor else FLOOR_DOWN
        floor = min(floor + rate * (ratio_db - floor), FLOOR_MAX)
        rms = float(np.sqrt(0.5 * (ep[i] + er[i])))
        if not has_env:
            env, has_env = rms, True
        peak = pk[i] + 1e-9
        ceiling = thr * max(env, MIN_ENV)
        hit = peak > ceiling and ratio_db <= floor + far
        if fix:
            run = run + 1 if hit else 0
            hit = hit and run <= hold_max
        if hit:
            gain = min(gain, ceiling / peak)
            engaged += 1
            if fix:
                env += env_a_hold * (rms - env)
        else:
            gain += rel * (1.0 - gain)
            if gain > 0.999:
                gain = 1.0
            env += env_a * (rms - env)
        g[i] = gain
    st[:] = [has_env, env, floor, gain, engaged, run]
    return g


if numba is not None:
    @numba.njit(cache=True, fastmath=False)
    def _pairwise_sq_mean(x, a, b):
        """float32 mean of x[a:b]**2 with NumPy's pairwise order for n < 8 and 8 <= n <= 128."""
        n = b - a
        if n < 8:
            res = np.float32(0.0)
            for i in range(a, b):
                v = x[i] * x[i]
                res = res + v
        else:
            r0 = x[a] * x[a]; r1 = x[a + 1] * x[a + 1]; r2 = x[a + 2] * x[a + 2]; r3 = x[a + 3] * x[a + 3]
            r4 = x[a + 4] * x[a + 4]; r5 = x[a + 5] * x[a + 5]; r6 = x[a + 6] * x[a + 6]; r7 = x[a + 7] * x[a + 7]
            m = n - n % 8
            for i in range(8, m, 8):
                j = a + i
                r0 += x[j] * x[j]; r1 += x[j + 1] * x[j + 1]; r2 += x[j + 2] * x[j + 2]; r3 += x[j + 3] * x[j + 3]
                r4 += x[j + 4] * x[j + 4]; r5 += x[j + 5] * x[j + 5]; r6 += x[j + 6] * x[j + 6]; r7 += x[j + 7] * x[j + 7]
            res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            for i in range(a + m, b):
                res += x[i] * x[i]
        return res / np.float32(n)

    @numba.njit(cache=True, fastmath=False)
    def _limit_numba(p, r, sub, st, thr, rel, env_a, far, fix, env_a_hold, hold_max, floor_up, floor_down, floor_max,
                     min_env):
        has_env, env, floor, gain = st[0] > 0.5, st[1], st[2], st[3]
        engaged, run = int(st[4]), int(st[5])
        n = p.shape[0]
        op = p.copy(); orr = r.copy()
        eps = np.float32(1e-10)
        a = 0
        while a < n:
            b = min(a + sub, n)
            ep = np.float64(_pairwise_sq_mean(p, a, b) + eps)
            er = np.float64(_pairwise_sq_mean(r, a, b) + eps)
            pmax = np.float32(0.0)
            for i in range(a, b):
                v = abs(p[i])
                if v > pmax:
                    pmax = v
                v = abs(r[i])
                if v > pmax:
                    pmax = v
            ratio_db = 10 * np.log10(ep / er)
            rate = floor_up if ratio_db > floor else floor_down
            floor = min(floor + rate * (ratio_db - floor), floor_max)
            rms = np.sqrt(0.5 * (ep + er))
            if not has_env:
                env = rms
                has_env = True
            peak = np.float64(pmax) + 1e-9
            ceiling = thr * max(env, min_env)
            hit = peak > ceiling and ratio_db <= floor + far
            if fix:
                run = run + 1 if hit else 0
                hit = hit and run <= hold_max
            if hit:
                gain = min(gain, ceiling / peak)
                engaged += 1
                if fix:
                    env += env_a_hold * (rms - env)
            else:
                gain += rel * (1.0 - gain)
                if gain > 0.999:
                    gain = 1.0
                env += env_a * (rms - env)
            g32 = np.float32(gain)
            for i in range(a, b):
                op[i] = op[i] * g32
                orr[i] = orr[i] * g32
            a = b
        st[0] = 1.0 if has_env else 0.0
        st[1] = env; st[2] = floor; st[3] = gain; st[4] = engaged; st[5] = run
        return op, orr


class LimiterKernel(Limiter):
    """Limiter with a selectable kernel ("numba", "numpy" or "loop"); same constructor arguments and interface."""

    def __init__(self, *args, kernel: str = "numba", **kw):
        if kernel not in KERNELS:
            raise ValueError(f"limiter kernel must be one of {KERNELS}, got {kernel!r}")
        if kernel == "numba" and numba is None:
            raise ImportError("limiter kernel 'numba' needs numba (the r8 preflight requires it)")
        self.kernel = kernel
        super().__init__(*args, **kw)

    def _params(self):
        return (float(self.thr), float(self.rel), float(self.env_a), float(self.far), bool(self.fix_latch),
                float(self.env_a_hold), int(self.hold_max), float(self.floor_up))

    def _state(self):
        return [self.env is not None, 0.0 if self.env is None else float(self.env), float(self.floor),
                float(self.gain), int(self.engaged), int(self.run)]

    def _set_state(self, st):
        has_env, env, floor, gain, engaged, run = st
        self.env = float(env) if has_env else None
        self.floor, self.gain, self.engaged, self.run = float(floor), float(gain), int(engaged), int(run)

    def process_block(self, p: np.ndarray, r: np.ndarray):
        if self.kernel == "loop":
            return super().process_block(p, r)
        p = np.ascontiguousarray(p, np.float32); r = np.ascontiguousarray(r, np.float32)
        if len(p) == 0:
            return p.copy(), r.copy()
        if self.kernel == "numpy":
            st = self._state()
            ep, er, pk = _stats_numpy(p, r, self.sub)
            g = _recurrence(ep, er, pk, st, self._params())
            self._set_state(st)
            gs = np.repeat(g.astype(np.float32), self.sub)[:len(p)]
            return p * gs, r * gs
        st = np.array(self._state(), np.float64)
        thr, rel, env_a, far, fix, env_a_hold, hold_max, floor_up = self._params()
        op, orr = _limit_numba(p, r, self.sub, st, thr, rel, env_a, far, fix, env_a_hold, hold_max, floor_up,
                               float(FLOOR_DOWN), float(FLOOR_MAX), float(MIN_ENV))
        self._set_state([st[0] > 0.5, st[1], st[2], st[3], int(st[4]), int(st[5])])
        return op, orr


def make_limiter(cfg, kernel: str | None = None) -> Limiter:
    """dsp.limiter (True or Limiter kwargs) -> a limiter; kernel None/"loop" keeps the legacy Python loop."""
    kw = cfg if isinstance(cfg, dict) else {}
    if kernel in (None, "loop"):
        return Limiter(**kw)
    return LimiterKernel(kernel=kernel, **kw)
