"""Metric definitions. SNR here is error-based against the clean reference
(distortion counts as noise); SI-SDR is scale-invariant. They are different
numbers and are reported separately - never call SI-SDR 'output SNR'.
PESQ: wideband P.862.2 at 16 kHz. ITU has withdrawn P.862 in favour of P.863;
we report it because the brief requests it.

pesq 0.0.4's C code reads out of bounds on some noisy inputs (utterance_split leaves a split utterance's start
negative, then indexes VAD[<0]; results_r2/r8/native_crash/README.md). It segfaults the process only sometimes,
so pesq_wb runs the unchanged C call in a persistent child process: a child death is a NaN plus a warning, never
a dead training run or a hung pool. VAANI_PESQ_ISOLATE=0 calls it in-process instead.
"""
import atexit, hashlib, os, pickle, queue, struct, subprocess, sys, threading

import numpy as np
from pesq import pesq as _pesq
from pystoi import stoi as _stoi

SR = 16000


def snr_db(clean, est):
    return float(10 * np.log10((clean ** 2).sum() / (((est - clean) ** 2).sum() + 1e-12) + 1e-12))


def si_sdr_db(clean, est):
    a = (est * clean).sum() / ((clean ** 2).sum() + 1e-12); s = a * clean
    return float(10 * np.log10((s ** 2).sum() / (((est - s) ** 2).sum() + 1e-12) + 1e-12))


def stoi(clean, est):
    return float(_stoi(clean, est, SR, extended=False))


def pesq_wb_inproc(clean, est):
    try: return float(_pesq(SR, clean, est, "wb"))
    except Exception: return float("nan")  # NoUtterancesError etc.: never crash a run


# Child loop: the same call and the same exception->NaN rule as pesq_wb_inproc; imports only numpy + pesq.
# pickle is safe here: both ends are this module, over the child's private stdin/stdout pipes.
_WORKER_SRC = r'''
import faulthandler, os, pickle, struct, sys
out = os.fdopen(os.dup(1), "wb"); os.dup2(2, 1)   # the C code printf()s on some errors; keep it off the reply pipe
faulthandler.enable()                                 # a crash leaves its stack in the parent's log
from pesq import pesq
inp = sys.stdin.buffer
def rd(n):
    b = inp.read(n)
    if len(b) < n: sys.exit(0)                        # parent closed the pipe
    return b
while True:
    clean, est = pickle.loads(rd(struct.unpack("<Q", rd(8))[0]))
    try: v = float(pesq(16000, clean, est, "wb"))
    except Exception: v = float("nan")
    b = pickle.dumps(v); out.write(struct.pack("<Q", len(b)) + b); out.flush()
'''


class _PesqWorker:
    """One persistent child per process, started on first use and restarted after a death."""

    def __init__(self, src=_WORKER_SRC, timeout=None):
        self.src, self.timeout, self.proc, self.pid, self.q = src, timeout, None, None, None
        self.lock = threading.Lock()
        self.failures = {"crash": 0, "timeout": 0}

    def reset_after_fork(self):
        # a forked child must not share the parent's pipes (or a lock held at fork time)
        self.proc, self.pid, self.q, self.lock = None, None, None, threading.Lock()

    def _start(self):
        env = {**os.environ, **{k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        self.proc = subprocess.Popen([sys.executable, "-c", self.src], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     env=env)
        self.pid, self.q = os.getpid(), queue.Queue()
        threading.Thread(target=self._read, args=(self.proc.stdout, self.q), daemon=True).start()

    @staticmethod
    def _read(f, q):
        try:
            while True:
                h = f.read(8)
                if len(h) < 8: break
                n = struct.unpack("<Q", h)[0]; b = f.read(n)
                if len(b) < n: break
                q.put(pickle.loads(b))
        except (OSError, ValueError, EOFError):
            pass
        q.put(None)   # EOF: the child is gone

    def stop(self):
        p, self.proc = self.proc, None
        if p is None or self.pid != os.getpid():
            return
        try: p.stdin.close()
        except OSError: pass
        try: p.wait(5)
        except subprocess.TimeoutExpired: p.kill()

    def __call__(self, clean, est):
        msg = pickle.dumps((clean, est), protocol=pickle.HIGHEST_PROTOCOL)
        timeout = self.timeout or 60 + 2 * max(len(clean), len(est)) / SR   # pesq runs ~0.1 s per 6 s clip; a hang is not a result
        with self.lock:
            if self.proc is None or self.pid != os.getpid() or self.proc.poll() is not None:
                self._start()
            kind = "crash"
            try:
                self.proc.stdin.write(struct.pack("<Q", len(msg)) + msg); self.proc.stdin.flush()
                v = self.q.get(timeout=timeout)
            except OSError:
                v = None
            except queue.Empty:
                v, kind = None, "timeout"
            if v is not None:
                return v
            p, self.proc = self.proc, None
            p.kill(); rc = p.wait()
            self.failures[kind] += 1
        h = hashlib.sha1(np.ascontiguousarray(clean).tobytes() + np.ascontiguousarray(est).tobytes()).hexdigest()[:12]
        where = ""
        if os.environ.get("VAANI_PESQ_CRASH_DIR"):
            d = os.environ["VAANI_PESQ_CRASH_DIR"]; os.makedirs(d, exist_ok=True)
            where = os.path.join(d, f"pesq_{kind}_{h}.npz"); np.savez(where, clean=clean, est=est)
        print(f"WARNING: pesq child {kind} (rc={rc}) on input {h} (n={len(clean)}/{len(est)}); PESQ-WB = NaN. "
              f"Known pesq 0.0.4 out-of-bounds read, results_r2/r8/native_crash/README.md {where}".rstrip(),
              file=sys.stderr, flush=True)
        return float("nan")


_worker = _PesqWorker()
atexit.register(_worker.stop)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_worker.reset_after_fork)


def pesq_failures():
    """Child deaths in this process so far: {"crash", "timeout", "total"}; each one scored NaN."""
    f = dict(_worker.failures); f["total"] = f["crash"] + f["timeout"]
    return f


def pesq_wb(clean, est):
    """PESQ-WB, bit-identical to pesq_wb_inproc on every input that does not kill the C code; NaN on those that do."""
    if os.environ.get("VAANI_PESQ_ISOLATE", "1") == "0" or not (isinstance(clean, np.ndarray) and isinstance(est, np.ndarray)):
        return pesq_wb_inproc(clean, est)   # non-arrays keep the old in-process path
    return _worker(clean, est)


def _envelope_db(x, frame=320):
    f = x[: len(x) // frame * frame].reshape(-1, frame)
    return 10 * np.log10((f ** 2).mean(axis=1) + 1e-10)


def recovery_time_s(est_burst, est_twin, burst_onset_s, thresh_db=3.0, hold_s=0.2, frame=320):
    """Time after the burst until the speech envelope of the burst run stays
    within thresh_db of the no-burst twin for hold_s. inf if never (NaN is reserved for "no burst")."""
    d = np.abs(_envelope_db(est_burst, frame) - _envelope_db(est_twin, frame))
    k0 = int(burst_onset_s * SR / frame); hold = int(hold_s * SR / frame)
    ok = d < thresh_db
    for k in range(k0, len(ok) - hold):
        if ok[k:k + hold].all():
            return (k - k0) * frame / SR
    return float("inf")
